package store

import (
	"context"
	"encoding/json"
	"fmt"

	"gorm.io/gorm"

	"ragflow/internal/entity"
	"ragflow/internal/harness/events"
)

// defaultEventLimit bounds an unbounded read. A trace of a long run is large;
// materialising all of it for one Stream call would be a memory incident rather
// than a query.
const defaultEventLimit = 5000

// EventStore persists the harness audit log in MySQL through the append-only
// agent_event table. It implements events.EventLog, so the recorder writes to the
// database without knowing it.
//
// Rows are inserted and never updated. Ordering inside one trace comes from the
// logical clock; across traces the clock restarts, so reads sort by clock and
// then by insert time.
type EventStore struct {
	db *gorm.DB
}

// NewEventStore builds an event store.
func NewEventStore(db *gorm.DB) *EventStore { return &EventStore{db: db} }

// Append writes a batch of events.
func (s *EventStore) Append(ctx context.Context, items ...*events.Event) error {
	if len(items) == 0 {
		return nil
	}
	rows := make([]*entity.HarnessAgentEvent, 0, len(items))
	for _, item := range items {
		row, err := toEventRow(item)
		if err != nil {
			return err
		}
		rows = append(rows, row)
	}
	if err := s.db.WithContext(ctx).CreateInBatches(rows, 100).Error; err != nil {
		return fmt.Errorf("store: append %d event(s): %w", len(rows), err)
	}
	return nil
}

// Stream returns events matching the filter.
func (s *EventStore) Stream(ctx context.Context, filter events.EventFilter) events.EventIterator {
	items, err := s.Range(ctx, filter.FromClock, filter.ToClock, filter)
	if err != nil {
		return &eventIterator{err: err}
	}
	return &eventIterator{items: items}
}

// Get returns one event by id.
func (s *EventStore) Get(ctx context.Context, id events.EventID) (*events.Event, error) {
	var row entity.HarnessAgentEvent
	if err := s.db.WithContext(ctx).Where("id = ?", string(id)).Take(&row).Error; err != nil {
		if err == gorm.ErrRecordNotFound {
			return nil, nil
		}
		return nil, fmt.Errorf("store: get event %s: %w", id, err)
	}
	return fromEventRow(&row)
}

// Range returns events whose logical clock falls in [from, to]. Zero means
// unbounded on that side.
func (s *EventStore) Range(ctx context.Context, from, to uint64, filter events.EventFilter) ([]*events.Event, error) {
	query := s.db.WithContext(ctx).Model(&entity.HarnessAgentEvent{})
	if filter.TraceID != "" {
		query = query.Where("trace_id = ?", filter.TraceID)
	}
	if filter.ThreadID != "" {
		query = query.Where("thread_id = ?", filter.ThreadID)
	}
	if filter.Node != "" {
		query = query.Where("node = ?", filter.Node)
	}
	if len(filter.Types) > 0 {
		types := make([]string, 0, len(filter.Types))
		for _, typ := range filter.Types {
			types = append(types, string(typ))
		}
		query = query.Where("type IN ?", types)
	}
	if len(filter.Layers) > 0 {
		layers := make([]string, 0, len(filter.Layers))
		for _, layer := range filter.Layers {
			layers = append(layers, string(layer))
		}
		query = query.Where("layer IN ?", layers)
	}
	if from > 0 {
		query = query.Where("clock >= ?", from)
	}
	if to > 0 {
		query = query.Where("clock <= ?", to)
	}
	if !filter.FromTime.IsZero() {
		query = query.Where("event_time >= ?", filter.FromTime)
	}
	if !filter.ToTime.IsZero() {
		query = query.Where("event_time <= ?", filter.ToTime)
	}

	limit := filter.Limit
	if limit <= 0 {
		limit = defaultEventLimit
	}

	var rows []entity.HarnessAgentEvent
	if err := query.Order("clock").Order("create_time").Limit(limit).Find(&rows).Error; err != nil {
		return nil, fmt.Errorf("store: query events: %w", err)
	}
	items := make([]*events.Event, 0, len(rows))
	for index := range rows {
		item, err := fromEventRow(&rows[index])
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, nil
}

// Seek returns the events at or after a logical clock.
func (s *EventStore) Seek(ctx context.Context, clock uint64) (events.EventIterator, error) {
	items, err := s.Range(ctx, clock, 0, events.EventFilter{})
	if err != nil {
		return nil, err
	}
	return &eventIterator{items: items}, nil
}

// Length counts the stored events.
func (s *EventStore) Length(ctx context.Context) (uint64, error) {
	var total int64
	if err := s.db.WithContext(ctx).Model(&entity.HarnessAgentEvent{}).Count(&total).Error; err != nil {
		return 0, fmt.Errorf("store: count events: %w", err)
	}
	return uint64(total), nil
}

func toEventRow(item *events.Event) (*entity.HarnessAgentEvent, error) {
	if item == nil || item.ID == "" {
		return nil, fmt.Errorf("store: event id is required")
	}
	metadata := ""
	if len(item.Metadata) > 0 {
		encoded, err := json.Marshal(item.Metadata)
		if err != nil {
			return nil, fmt.Errorf("store: encode metadata of %s: %w", item.ID, err)
		}
		metadata = string(encoded)
	}
	causedBy := ""
	if len(item.CausedBy) > 0 {
		encoded, err := json.Marshal(item.CausedBy)
		if err != nil {
			return nil, fmt.Errorf("store: encode caused_by of %s: %w", item.ID, err)
		}
		causedBy = string(encoded)
	}

	return &entity.HarnessAgentEvent{
		ID:            string(item.ID),
		TraceID:       item.TraceID,
		Clock:         item.Clock,
		ThreadID:      item.ThreadID,
		Type:          string(item.Type),
		Layer:         string(events.LayerOf(item.Type)),
		ParentID:      string(item.ParentID),
		CausedBy:      causedBy,
		Step:          item.Step,
		Node:          item.Node,
		TaskID:        item.TaskID,
		Payload:       string(item.Payload),
		Metadata:      metadata,
		Deterministic: item.Deterministic,
		Hash:          item.Hash,
		EventTime:     item.Timestamp,
	}, nil
}

func fromEventRow(row *entity.HarnessAgentEvent) (*events.Event, error) {
	item := &events.Event{
		ID:            events.EventID(row.ID),
		Type:          events.EventType(row.Type),
		Layer:         events.Layer(row.Layer),
		Clock:         row.Clock,
		TraceID:       row.TraceID,
		ThreadID:      row.ThreadID,
		ParentID:      events.EventID(row.ParentID),
		Step:          row.Step,
		Node:          row.Node,
		TaskID:        row.TaskID,
		Deterministic: row.Deterministic,
		Hash:          row.Hash,
		Timestamp:     row.EventTime,
	}
	if item.Layer == "" {
		item.Layer = events.LayerOf(item.Type)
	}
	if row.Payload != "" {
		item.Payload = json.RawMessage(row.Payload)
	}
	if row.Metadata != "" {
		if err := json.Unmarshal([]byte(row.Metadata), &item.Metadata); err != nil {
			return nil, fmt.Errorf("store: decode metadata of %s: %w", row.ID, err)
		}
	}
	if row.CausedBy != "" {
		if err := json.Unmarshal([]byte(row.CausedBy), &item.CausedBy); err != nil {
			return nil, fmt.Errorf("store: decode caused_by of %s: %w", row.ID, err)
		}
	}
	if item.Timestamp.IsZero() {
		item.Timestamp = row.CreateTime
	}
	return item, nil
}

// eventIterator walks a materialised result. Reads are bounded by
// defaultEventLimit, so streaming from memory is equivalent to streaming from
// the database for the sizes the API exposes.
type eventIterator struct {
	items []*events.Event
	next  int
	err   error
}

func (it *eventIterator) Next(context.Context) (*events.Event, bool) {
	if it.err != nil || it.next >= len(it.items) {
		return nil, false
	}
	item := it.items[it.next]
	it.next++
	return item, true
}

func (it *eventIterator) Close() error {
	err := it.err
	it.err = nil
	it.items = nil
	return err
}

var _ events.EventLog = (*EventStore)(nil)

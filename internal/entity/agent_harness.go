//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package entity

import "time"

// HarnessAgent is an agent definition header. The definition body itself is
// immutable and versioned in HarnessAgentVersion.
type HarnessAgent struct {
	ID               string    `gorm:"column:id;primaryKey;size:64" json:"id"`
	TenantID         string    `gorm:"column:tenant_id;size:64;not null;uniqueIndex:idx_agent_tenant_name,priority:1" json:"tenant_id"`
	Name             string    `gorm:"column:name;size:128;not null;uniqueIndex:idx_agent_tenant_name,priority:2" json:"name"`
	Description      string    `gorm:"column:description;size:1024;default:''" json:"description"`
	CurrentVersionID string    `gorm:"column:current_version_id;size:64;default:''" json:"current_version_id"`
	Status           string    `gorm:"column:status;size:32;not null;default:'active'" json:"status"`
	CreatedBy        string    `gorm:"column:created_by;size:64;default:''" json:"created_by"`
	CreateTime       time.Time `gorm:"column:create_time;autoCreateTime" json:"create_time"`
	UpdateTime       time.Time `gorm:"column:update_time;autoUpdateTime" json:"update_time"`
}

// TableName returns the table name.
func (HarnessAgent) TableName() string { return "agent" }

// HarnessAgentVersion is one immutable snapshot of an agent definition.
type HarnessAgentVersion struct {
	ID         string    `gorm:"column:id;primaryKey;size:64" json:"id"`
	AgentID    string    `gorm:"column:agent_id;size:64;not null;uniqueIndex:idx_agent_version,priority:1" json:"agent_id"`
	Version    int       `gorm:"column:version;not null;uniqueIndex:idx_agent_version,priority:2" json:"version"`
	Definition string    `gorm:"column:definition;type:longtext;not null" json:"definition"`
	CreatedBy  string    `gorm:"column:created_by;size:64;default:''" json:"created_by"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"create_time"`
}

// TableName returns the table name.
func (HarnessAgentVersion) TableName() string { return "agent_version" }

// HarnessAgentRun is one execution of one agent version.
type HarnessAgentRun struct {
	ID             string `gorm:"column:id;primaryKey;size:64" json:"id"`
	AgentID        string `gorm:"column:agent_id;size:64;not null;index" json:"agent_id"`
	AgentVersionID string `gorm:"column:agent_version_id;size:64;default:''" json:"agent_version_id"`
	// ParentRunID links a delegated child run to the run that spawned it, so a
	// crash recovery can walk the tree and an audit can see who asked for what.
	ParentRunID string `gorm:"column:parent_run_id;size:64;default:'';index" json:"parent_run_id"`
	// SessionID is the conversation this run belongs to, so a thread can list
	// its runs. Empty for runs started outside a chat thread.
	SessionID string `gorm:"column:session_id;size:32;not null;default:'';index" json:"session_id"`
	Depth     int    `gorm:"column:depth;default:0" json:"depth"`
	TenantID  string `gorm:"column:tenant_id;size:64;not null;index" json:"tenant_id"`
	// UserID is the actor the run acted for. It is persisted because model access
	// resolves per user: a recovered run without it could resume on a provider
	// the caller never selected.
	UserID     string     `gorm:"column:user_id;size:64;default:'';index" json:"user_id"`
	Status     string     `gorm:"column:status;size:32;not null;default:'running';index" json:"status"`
	Trigger    string     `gorm:"column:trigger;size:32;default:'api'" json:"trigger"`
	Input      string     `gorm:"column:input;type:longtext" json:"input"`
	Output     string     `gorm:"column:output;type:longtext" json:"output"`
	Error      string     `gorm:"column:error;type:text" json:"error"`
	LeaseOwner string     `gorm:"column:lease_owner;size:128;default:''" json:"lease_owner"`
	LeaseUntil *time.Time `gorm:"column:lease_until" json:"lease_until,omitempty"`
	StartedAt  *time.Time `gorm:"column:started_at" json:"started_at,omitempty"`
	EndedAt    *time.Time `gorm:"column:ended_at" json:"ended_at,omitempty"`
	TokensIn   int        `gorm:"column:tokens_in;default:0" json:"tokens_in"`
	TokensOut  int        `gorm:"column:tokens_out;default:0" json:"tokens_out"`
	Cost       float64    `gorm:"column:cost;default:0" json:"cost"`
	CreateTime time.Time  `gorm:"column:create_time;autoCreateTime" json:"create_time"`
	UpdateTime time.Time  `gorm:"column:update_time;autoUpdateTime" json:"update_time"`
}

// TableName returns the table name.
func (HarnessAgentRun) TableName() string { return "agent_run" }

// HarnessAgentStep is one auditable step of a run.
type HarnessAgentStep struct {
	ID         string    `gorm:"column:id;primaryKey;size:64" json:"id"`
	RunID      string    `gorm:"column:run_id;size:64;not null;uniqueIndex:idx_agent_step_seq,priority:1" json:"run_id"`
	Seq        int       `gorm:"column:seq;not null;uniqueIndex:idx_agent_step_seq,priority:2" json:"seq"`
	Phase      string    `gorm:"column:phase;size:32;not null;index" json:"phase"`
	Round      int       `gorm:"column:round;default:0" json:"round"`
	Tool       string    `gorm:"column:tool;size:64;default:''" json:"tool"`
	ArgsHash   string    `gorm:"column:args_hash;size:64;default:''" json:"args_hash"`
	Decision   string    `gorm:"column:decision;size:32;default:''" json:"decision"`
	Reason     string    `gorm:"column:reason;size:512;default:''" json:"reason"`
	DurationMS int64     `gorm:"column:duration_ms;default:0" json:"duration_ms"`
	TokensIn   int       `gorm:"column:tokens_in;default:0" json:"tokens_in"`
	TokensOut  int       `gorm:"column:tokens_out;default:0" json:"tokens_out"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"create_time"`
}

// TableName returns the table name.
func (HarnessAgentStep) TableName() string { return "agent_step" }

// HarnessAgentEvent is one event of the append-only audit log. Rows are never
// updated: Layer is written once so retention and export can select a data class
// without re-deriving it, and ParentID/CausedBy keep the causal chain readable in
// SQL rather than only in memory.
type HarnessAgentEvent struct {
	ID      string `gorm:"column:id;primaryKey;size:64" json:"id"`
	TraceID string `gorm:"column:trace_id;size:64;not null;index:idx_agent_event_trace,priority:1" json:"trace_id"`
	// Clock is the logical clock. It orders events inside a trace even when they
	// were appended concurrently.
	Clock    uint64 `gorm:"column:clock;not null;index:idx_agent_event_trace,priority:2" json:"clock"`
	ThreadID string `gorm:"column:thread_id;size:64;default:'';index" json:"thread_id"`
	Type     string `gorm:"column:type;size:64;not null;index" json:"type"`
	Layer    string `gorm:"column:layer;size:8;not null;default:'C3';index" json:"layer"`
	ParentID string `gorm:"column:parent_id;size:64;default:''" json:"parent_id"`
	// CausedBy is a JSON array of event ids.
	CausedBy      string    `gorm:"column:caused_by;size:2048;default:''" json:"caused_by"`
	Step          int       `gorm:"column:step;default:0" json:"step"`
	Node          string    `gorm:"column:node;size:64;default:''" json:"node"`
	TaskID        string    `gorm:"column:task_id;size:64;default:''" json:"task_id"`
	Payload       string    `gorm:"column:payload;type:longtext" json:"payload"`
	Metadata      string    `gorm:"column:metadata;type:text" json:"metadata"`
	Deterministic bool      `gorm:"column:deterministic;default:false" json:"deterministic"`
	Hash          string    `gorm:"column:hash;size:64;default:''" json:"hash"`
	EventTime     time.Time `gorm:"column:event_time" json:"event_time"`
	CreateTime    time.Time `gorm:"column:create_time;autoCreateTime" json:"create_time"`
}

// TableName returns the table name.
func (HarnessAgentEvent) TableName() string { return "agent_event" }

// HarnessAgentPermissionGrant is one standing "allow always" answer. The
// argument hash is part of the key: approving one shell command must not approve
// every shell command.
type HarnessAgentPermissionGrant struct {
	ID         string    `gorm:"column:id;primaryKey;size:64" json:"id"`
	TenantID   string    `gorm:"column:tenant_id;size:64;not null;uniqueIndex:idx_agent_grant,priority:1" json:"tenant_id"`
	Tool       string    `gorm:"column:tool;size:64;not null;uniqueIndex:idx_agent_grant,priority:2" json:"tool"`
	ArgsHash   string    `gorm:"column:args_hash;size:64;not null;uniqueIndex:idx_agent_grant,priority:3" json:"args_hash"`
	CreatedBy  string    `gorm:"column:created_by;size:64;default:''" json:"created_by"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"create_time"`
}

// TableName returns the table name.
func (HarnessAgentPermissionGrant) TableName() string { return "agent_permission_grant" }

// HarnessAgentFeedback is a human signal about one run.
type HarnessAgentFeedback struct {
	ID         string    `gorm:"column:id;primaryKey;size:64" json:"id"`
	RunID      string    `gorm:"column:run_id;size:64;not null;index" json:"run_id"`
	TenantID   string    `gorm:"column:tenant_id;size:64;not null;index" json:"tenant_id"`
	Rating     int       `gorm:"column:rating;default:0" json:"rating"`
	Comment    string    `gorm:"column:comment;type:text" json:"comment"`
	Labels     string    `gorm:"column:labels;type:text" json:"labels"`
	CreatedBy  string    `gorm:"column:created_by;size:64;default:''" json:"created_by"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"create_time"`
}

// TableName returns the table name.
func (HarnessAgentFeedback) TableName() string { return "agent_feedback" }

// HarnessAgentLesson is one promoted experience entry.
type HarnessAgentLesson struct {
	ID            string    `gorm:"column:id;primaryKey;size:64" json:"id"`
	TenantID      string    `gorm:"column:tenant_id;size:64;not null;index" json:"tenant_id"`
	AgentID       string    `gorm:"column:agent_id;size:64;default:'';index" json:"agent_id"`
	Kind          string    `gorm:"column:kind;size:32;not null" json:"kind"`
	Title         string    `gorm:"column:title;size:255;not null" json:"title"`
	Body          string    `gorm:"column:body;type:text;not null" json:"body"`
	SourceRunIDs  string    `gorm:"column:source_run_ids;type:text" json:"source_run_ids"`
	Status        string    `gorm:"column:status;size:32;not null;default:'proposed';index" json:"status"`
	EvidenceCount int       `gorm:"column:evidence_count;default:1" json:"evidence_count"`
	CreateTime    time.Time `gorm:"column:create_time;autoCreateTime" json:"create_time"`
	UpdateTime    time.Time `gorm:"column:update_time;autoUpdateTime" json:"update_time"`
}

// TableName returns the table name.
func (HarnessAgentLesson) TableName() string { return "agent_lesson" }

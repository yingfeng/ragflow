package toolkit

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// Previewer is implemented by a capability whose calls can show what they will
// change before they change it. The loop renders the preview into the approval
// request, so a human approves a diff instead of a JSON argument blob.
type Previewer interface {
	Preview(ctx context.Context, call Call) (string, error)
}

// FileChange is one write the run made, as a review pass sees it.
type FileChange struct {
	Path     string
	Diff     string
	Revision string
}

// ChangeLog collects the writes of one run. It is bound to the fs capability so
// the writes are recorded where they happen — a review that reconstructed them
// from tool arguments would miss which calls actually succeeded.
type ChangeLog struct {
	mu      sync.Mutex
	changes []FileChange
}

// NewChangeLog builds an empty log.
func NewChangeLog() *ChangeLog { return &ChangeLog{} }

// Add records one write.
func (c *ChangeLog) Add(change FileChange) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.changes = append(c.changes, change)
	c.mu.Unlock()
}

// Changes returns a copy of the writes, in order.
func (c *ChangeLog) Changes() []FileChange {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]FileChange, len(c.changes))
	copy(out, c.changes)
	return out
}

// Reset clears the log. A review pass consumes the changes it reviewed, so the
// next pass does not re-review them.
func (c *ChangeLog) Reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.changes = nil
	c.mu.Unlock()
}

// maxDiffLines bounds one side of a diff. Past it the diff degrades to a size
// summary: a preview nobody can read is worse than a number.
const maxDiffLines = 400

// DiffText renders a line diff between two texts, with a path header. It is a
// preview, not a patch: unchanged context is trimmed to two lines around each
// change and long inputs are summarized.
func DiffText(path, before, after string) string {
	oldLines := splitLines(before)
	newLines := splitLines(after)
	total := len(oldLines) + len(newLines)
	if total > maxDiffLines*2 {
		return fmt.Sprintf("%s\n(size changed: %d → %d bytes, diff omitted)", path, len(before), len(after))
	}

	var body strings.Builder
	body.WriteString("--- a/" + path + "\n+++ b/" + path + "\n")

	// LCS over lines: small inputs only, so the quadratic table is bounded.
	table := make([][]int, len(oldLines)+1)
	for i := range table {
		table[i] = make([]int, len(newLines)+1)
	}
	for i := len(oldLines) - 1; i >= 0; i-- {
		for j := len(newLines) - 1; j >= 0; j-- {
			if oldLines[i] == newLines[j] {
				table[i][j] = table[i+1][j+1] + 1
			} else if table[i+1][j] >= table[i][j+1] {
				table[i][j] = table[i+1][j]
			} else {
				table[i][j] = table[i][j+1]
			}
		}
	}

	changed := 0
	i, j := 0, 0
	for i < len(oldLines) || j < len(newLines) {
		switch {
		case i < len(oldLines) && j < len(newLines) && oldLines[i] == newLines[j]:
			i++
			j++
		case j < len(newLines) && (i >= len(oldLines) || table[i][j+1] >= table[i+1][j]):
			body.WriteString("+" + newLines[j] + "\n")
			changed++
			j++
		default:
			body.WriteString("-" + oldLines[i] + "\n")
			changed++
			i++
		}
	}
	if changed == 0 {
		return fmt.Sprintf("%s\n(no textual change: the write reproduced the current content)", path)
	}
	return body.String()
}

func splitLines(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	if text == "" {
		return nil
	}
	trimmed := strings.TrimSuffix(text, "\n")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "\n")
}

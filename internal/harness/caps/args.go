package caps

import (
	"encoding/json"
	"fmt"
	"strings"
)

// DecodeArgs unmarshals a tool-call argument blob. Empty arguments decode to the
// zero value instead of failing.
func DecodeArgs[T any](raw string) (T, error) {
	var out T
	if strings.TrimSpace(raw) == "" {
		return out, nil
	}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return out, fmt.Errorf("invalid tool arguments: %w", err)
	}
	return out, nil
}

// ObjectSchema is a small helper for declaring tool parameters.
func ObjectSchema(properties map[string]any, required ...string) map[string]any {
	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

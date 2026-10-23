package skill

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// SkillFile is the instruction file of one skill.
const SkillFile = "SKILL.md"

// FSLoader reads skills from a directory tree laid out as
//
//	<root>/<skill-name>/SKILL.md
//	<root>/<skill-name>/resources/...
//
// SKILL.md carries a YAML front matter block with at least `name` and
// `description`; the rest of the file is the body loaded on demand.
type FSLoader struct {
	// Root is the skills directory.
	Root string
}

// List returns every skill's metadata. Bodies are not read.
func (l FSLoader) List(ctx context.Context) ([]Skill, error) {
	entries, err := os.ReadDir(l.Root)
	if err != nil {
		return nil, fmt.Errorf("skill: read %s: %w", l.Root, err)
	}

	skills := make([]Skill, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		metadata, _, err := parseSkillFile(filepath.Join(l.Root, entry.Name(), SkillFile))
		if err != nil {
			// A malformed skill must not hide the healthy ones.
			continue
		}
		skills = append(skills, metadata)
	}
	sort.Slice(skills, func(i, j int) bool { return skills[i].Name < skills[j].Name })
	return skills, nil
}

// Load returns one skill with its body and resource list.
func (l FSLoader) Load(ctx context.Context, name string) (Skill, error) {
	if strings.ContainsAny(name, `/\`) || name == "" || name == "." || name == ".." {
		return Skill{}, fmt.Errorf("skill: invalid name %q", name)
	}
	dir := filepath.Join(l.Root, name)
	// A skill directory is a name inside the root, never a path: resolve symlinks
	// and refuse anything that lands outside. Lexical checks alone let a symlink
	// named "notes" read /etc.
	if err := confineToRoot(l.Root, dir); err != nil {
		return Skill{}, err
	}
	metadata, body, err := parseSkillFile(filepath.Join(dir, SkillFile))
	if err != nil {
		return Skill{}, err
	}
	metadata.Body = body

	resources, err := listResources(filepath.Join(dir, "resources"))
	if err == nil {
		metadata.Resources = resources
	}
	return metadata, nil
}

// MaxSkillFileBytes bounds one SKILL.md. A skill is instructions, not a data
// store; a multi-megabyte file would be truncated by the body budget anyway.
const MaxSkillFileBytes = 1 << 20

// confineToRoot resolves symlinks and reports whether path stays inside root.
func confineToRoot(root, path string) error {
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		return fmt.Errorf("skill: resolve root: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return fmt.Errorf("skill: skill directory not found: %w", err)
	}
	relative, err := filepath.Rel(resolvedRoot, resolved)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return fmt.Errorf("skill: %q escapes the skills root", filepath.Base(path))
	}
	return nil
}

func listResources(root string) ([]string, error) {
	out := make([]string, 0, 8)
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil || info.IsDir() {
			return err
		}
		relative, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		out = append(out, relative)
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}

// parseSkillFile splits the front matter from the body.
func parseSkillFile(path string) (Skill, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Skill{}, "", fmt.Errorf("skill: read %s: %w", path, err)
	}
	// A skill is instructions, not a data store: an oversized file would be
	// truncated by the body budget anyway, so refuse it instead of half-reading.
	if len(raw) > MaxSkillFileBytes {
		return Skill{}, "", fmt.Errorf("skill: %s exceeds %d bytes", filepath.Base(path), MaxSkillFileBytes)
	}
	skill, body := parseSkillContent(string(raw), filepath.Base(filepath.Dir(path)))
	return skill, body, nil
}

// ParseManifest splits a SKILL.md into its manifest and body.
//
// Hosts that store skills as something other than a directory (the platform's
// skillspace keeps each version as a file tree) parse with this, so one parser
// governs every source: two parsers would drift, and the difference would only
// show up in whichever deployment used the other one.
func ParseManifest(raw, fallbackName string) (Skill, string) {
	return parseSkillContent(raw, fallbackName)
}

// parseSkillContent splits a SKILL.md into its manifest and body. dirName is
// the fallback name when the front matter omits one (the skill's directory).
func parseSkillContent(raw, dirName string) (Skill, string) {
	content := strings.ReplaceAll(raw, "\r\n", "\n")
	skill := Skill{Name: dirName}

	if !strings.HasPrefix(content, "---\n") {
		return skill, strings.TrimSpace(content)
	}
	rest := content[len("---\n"):]
	end := strings.Index(rest, "\n---")
	if end < 0 {
		return skill, strings.TrimSpace(content)
	}

	listKey := ""
	for _, line := range strings.Split(rest[:end], "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		// YAML sequence items belong to the last key that opened a list.
		if strings.HasPrefix(trimmed, "- ") && listKey != "" {
			appendListItem(&skill, listKey, strings.Trim(strings.TrimSpace(trimmed[2:]), `"'`))
			continue
		}
		key, value, ok := strings.Cut(trimmed, ":")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		listKey = ""
		switch key {
		case "name":
			if value != "" {
				skill.Name = value
			}
		case "description":
			skill.Description = value
		case "version":
			skill.Version = value
		case "when_to_use", "when-to-use":
			skill.WhenToUse = value
		case "max_tokens", "max-tokens":
			if parsed, err := strconv.Atoi(value); err == nil && parsed > 0 {
				skill.MaxTokens = parsed
			}
		case "auto_load", "auto-load":
			skill.AutoLoad = value == "true" || value == "yes"
		case "disable_model_invocation", "disable-model-invocation":
			skill.DisableModelInvocation = value == "true" || value == "yes"
		case "allowed_tools", "allowed-tools":
			if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
				for _, item := range strings.Split(strings.Trim(value, "[]"), ",") {
					item = strings.Trim(strings.TrimSpace(item), `"'`)
					if item != "" {
						skill.AllowedTools = append(skill.AllowedTools, item)
					}
				}
				continue
			}
			if value != "" {
				skill.AllowedTools = append(skill.AllowedTools, value)
				continue
			}
			listKey = key
		}
	}
	body := strings.TrimSpace(rest[end+len("\n---"):])
	return skill, body
}

func appendListItem(skill *Skill, key, value string) {
	if value == "" {
		return
	}
	switch key {
	case "allowed_tools", "allowed-tools":
		skill.AllowedTools = append(skill.AllowedTools, value)
	}
}

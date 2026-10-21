package skill

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
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
	content := strings.ReplaceAll(string(raw), "\r\n", "\n")
	skill := Skill{Name: filepath.Base(filepath.Dir(path))}

	if !strings.HasPrefix(content, "---\n") {
		return skill, strings.TrimSpace(content), nil
	}
	rest := content[len("---\n"):]
	end := strings.Index(rest, "\n---")
	if end < 0 {
		return skill, strings.TrimSpace(content), nil
	}

	for _, line := range strings.Split(rest[:end], "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		switch strings.TrimSpace(strings.ToLower(key)) {
		case "name":
			if value != "" {
				skill.Name = value
			}
		case "description":
			skill.Description = value
		}
	}
	body := strings.TrimSpace(rest[end+len("\n---"):])
	return skill, body, nil
}

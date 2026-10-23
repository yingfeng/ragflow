package skill

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
)

// bundledFS holds the skills compiled into the binary. They are the platform's
// own defaults: the retrieval policy every harness turn runs under, without
// requiring a deployment step. An installed skill with the same name overrides
// one of these (see MergedLoader), so a tenant can replace the policy without a
// rebuild while a fresh deployment still has one.
//
//go:embed bundled
var bundledFS embed.FS

// Bundled returns the loader for the compiled-in skill set.
func Bundled() Loader {
	return embeddedLoader{fsys: bundledFS, root: "bundled"}
}

// embeddedLoader reads <root>/<skill>/SKILL.md from an fs.FS.
//
// It exists instead of reusing FSLoader because FSLoader's confinement logic is
// os-specific (symlinks), and an embedded tree has no symlinks to confine.
type embeddedLoader struct {
	fsys fs.FS
	root string
}

// List returns every embedded skill's metadata.
func (l embeddedLoader) List(_ context.Context) ([]Skill, error) {
	entries, err := fs.ReadDir(l.fsys, l.root)
	if err != nil {
		return nil, fmt.Errorf("skill: read embedded %s: %w", l.root, err)
	}
	out := make([]Skill, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		raw, err := fs.ReadFile(l.fsys, path.Join(l.root, entry.Name(), SkillFile))
		if err != nil {
			// A malformed bundled skill must not hide the healthy ones.
			continue
		}
		metadata, _ := parseSkillContent(string(raw), entry.Name())
		out = append(out, metadata)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Load returns one embedded skill with its body.
func (l embeddedLoader) Load(_ context.Context, name string) (Skill, error) {
	if strings.ContainsAny(name, `/\`) || name == "" || name == "." || name == ".." {
		return Skill{}, fmt.Errorf("skill: invalid name %q", name)
	}
	raw, err := fs.ReadFile(l.fsys, path.Join(l.root, name, SkillFile))
	if err != nil {
		return Skill{}, fmt.Errorf("skill: %q not found", name)
	}
	skill, body := parseSkillContent(string(raw), name)
	skill.Body = body
	skill.Resources = listEmbeddedResources(l.fsys, path.Join(l.root, name))
	return skill, nil
}

// listEmbeddedResources lists the skill's own files, excluding SKILL.md.
func listEmbeddedResources(fsys fs.FS, root string) []string {
	out := make([]string, 0, 4)
	_ = fs.WalkDir(fsys, root, func(full string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		relative := strings.TrimPrefix(full, root+"/")
		if relative == SkillFile {
			return nil
		}
		out = append(out, relative)
		return nil
	})
	sort.Strings(out)
	return out
}

// MergedLoader layers skill sources. Later layers win: an installed skill with
// the same name as a bundled one replaces it instead of appearing twice.
type MergedLoader struct {
	Layers []Loader
}

// List returns the union of every layer, one entry per skill name.
func (m MergedLoader) List(ctx context.Context) ([]Skill, error) {
	byName := make(map[string]Skill, 8)
	for _, layer := range m.Layers {
		if layer == nil {
			continue
		}
		skills, err := layer.List(ctx)
		if err != nil {
			// One unreachable layer (a tenant directory that vanished) must not
			// take the others down with it.
			continue
		}
		for _, item := range skills {
			byName[item.Name] = item
		}
	}
	out := make([]Skill, 0, len(byName))
	for _, item := range byName {
		out = append(out, item)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// Load reads the highest-priority layer that has the skill.
func (m MergedLoader) Load(ctx context.Context, name string) (Skill, error) {
	for index := len(m.Layers) - 1; index >= 0; index-- {
		layer := m.Layers[index]
		if layer == nil {
			continue
		}
		item, err := layer.Load(ctx, name)
		if err == nil {
			return item, nil
		}
	}
	return Skill{}, fmt.Errorf("skill: %q not found", name)
}

// SelectedLoader narrows a loader to a name list. An empty list offers nothing:
// "the agent selected no skills" must not silently mean "all of them", because
// a tenant's skillspace is a catalog, not the agent's prompt.
type SelectedLoader struct {
	Inner Loader
	Names []string
}

// List returns the selected skills that the wrapped loader actually has.
func (l SelectedLoader) List(ctx context.Context) ([]Skill, error) {
	if l.Inner == nil || len(l.Names) == 0 {
		return nil, nil
	}
	wanted := make(map[string]struct{}, len(l.Names))
	for _, name := range l.Names {
		wanted[name] = struct{}{}
	}
	all, err := l.Inner.List(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Skill, 0, len(l.Names))
	for _, item := range all {
		if _, ok := wanted[item.Name]; ok {
			out = append(out, item)
		}
	}
	return out, nil
}

// Load reads a selected skill; anything outside the selection is not found.
func (l SelectedLoader) Load(ctx context.Context, name string) (Skill, error) {
	if l.Inner == nil {
		return Skill{}, fmt.Errorf("skill: %q not found", name)
	}
	for _, allowed := range l.Names {
		if allowed == name {
			return l.Inner.Load(ctx, name)
		}
	}
	return Skill{}, fmt.Errorf("skill: %q is not enabled for this agent", name)
}

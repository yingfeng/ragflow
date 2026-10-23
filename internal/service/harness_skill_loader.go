package service

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"ragflow/internal/dao"
	"ragflow/internal/harness/toolkit/skill"
	"ragflow/internal/permission"
)

// HarnessSkillLoader serves harness skills out of the caller's RAGFlow
// skillspace.
//
// The skillspace is the platform's existing skills product — spaces, versioned
// skill folders, SKILL.md, indexed for search — so an agent skill is the same
// artifact a user browses and installs, not a second kind of skill that only
// the harness can read. What this adds is resolution: which spaces the caller
// may read, which version of each skill is current, and the parsed body.
type HarnessSkillLoader struct {
	userID   string
	tenantID string
	spaceDAO *dao.SkillSpaceDAO
	fileDAO  *dao.FileDAO

	mu     sync.Mutex
	snap   *skillSnapshot
	snapAt time.Time
}

// skillSnapshotCacheFor bounds how stale a loaded skill set may be.
//
// A turn reads the catalog twice (the system note and the skill_list tool); a
// user who edits a skill expects the next turn to use it. Fifteen seconds keeps
// both true without a per-call traversal of every space.
const skillSnapshotCacheFor = 15 * time.Second

// skillSnapshot is one resolved view of the caller's skills: metadata plus the
// body of each, since both callers end up wanting the body anyway.
type skillSnapshot struct {
	items map[string]skill.Skill
	names []string
}

// NewHarnessSkillLoader builds a loader for one caller.
func NewHarnessSkillLoader(userID, tenantID string) *HarnessSkillLoader {
	return &HarnessSkillLoader{
		userID:   userID,
		tenantID: tenantID,
		spaceDAO: dao.NewSkillSpaceDAO(),
		fileDAO:  dao.NewFileDAO(),
	}
}

// List returns metadata for every skill the caller can read.
func (l *HarnessSkillLoader) List(ctx context.Context) ([]skill.Skill, error) {
	snap, err := l.snapshot(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]skill.Skill, 0, len(snap.names))
	for _, name := range snap.names {
		item := snap.items[name]
		item.Body = ""
		out = append(out, item)
	}
	return out, nil
}

// Load returns one skill with its body and resource list.
func (l *HarnessSkillLoader) Load(ctx context.Context, name string) (skill.Skill, error) {
	snap, err := l.snapshot(ctx)
	if err != nil {
		return skill.Skill{}, err
	}
	item, ok := snap.items[name]
	if !ok {
		return skill.Skill{}, fmt.Errorf("skill: %q is not in the caller's skillspace", name)
	}
	return item, nil
}

// snapshot resolves the catalog, reusing a recent result.
func (l *HarnessSkillLoader) snapshot(ctx context.Context) (*skillSnapshot, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.snap != nil && time.Since(l.snapAt) < skillSnapshotCacheFor {
		return l.snap, nil
	}
	snap, err := l.resolve(ctx)
	if err != nil {
		return nil, err
	}
	l.snap = snap
	l.snapAt = time.Now()
	return snap, nil
}

// resolve walks every readable space and picks the current version of each
// skill.
func (l *HarnessSkillLoader) resolve(ctx context.Context) (*skillSnapshot, error) {
	spaceIDs, err := accessibleSkillSpaceIDs(ctx, l.userID, permission.OperationRead)
	if err != nil {
		return nil, fmt.Errorf("skill: list readable spaces: %w", err)
	}
	sort.Strings(spaceIDs)

	items := make(map[string]skill.Skill, 8)
	for _, spaceID := range spaceIDs {
		space, err := l.spaceDAO.GetByID(ctx, dao.DB, spaceID)
		if err != nil || space == nil || space.Status != "1" || space.FolderID == "" {
			continue
		}
		for _, item := range l.skillsInSpace(ctx, space.FolderID) {
			// First readable space wins: a name collision between spaces is a
			// user decision, and a stable winner beats a random one.
			if _, exists := items[item.Name]; !exists {
				items[item.Name] = item
			}
		}
	}

	names := make([]string, 0, len(items))
	for name := range items {
		names = append(names, name)
	}
	sort.Strings(names)
	return &skillSnapshot{items: items, names: names}, nil
}

// skillsInSpace reads the current version of every skill folder in one space.
func (l *HarnessSkillLoader) skillsInSpace(ctx context.Context, spaceFolderID string) []skill.Skill {
	folders, err := l.fileDAO.ListByParentID(ctx, dao.DB, spaceFolderID)
	if err != nil {
		return nil
	}
	out := make([]skill.Skill, 0, len(folders))
	for _, folder := range folders {
		if folder.Type != "folder" {
			continue
		}
		// A skill folder holds one folder per published version; a skill that
		// was uploaded without a version folder still has files of its own.
		contentFolder := folder.ID
		if versions, err := l.fileDAO.ListByParentID(ctx, dao.DB, folder.ID); err == nil {
			if latest := findLatestSkillVersion(versions); latest != nil {
				contentFolder = latest.ID
			}
		}
		if item, ok := l.readSkill(ctx, folder.Name, contentFolder); ok {
			out = append(out, item)
		}
	}
	return out
}

// readSkill parses one skill's SKILL.md and lists its other files.
func (l *HarnessSkillLoader) readSkill(ctx context.Context, folderName, contentFolderID string) (skill.Skill, bool) {
	files, err := l.fileDAO.ListByParentID(ctx, dao.DB, contentFolderID)
	if err != nil {
		return skill.Skill{}, false
	}
	var raw []byte
	resources := make([]string, 0, len(files))
	for _, file := range files {
		if file.Type == "folder" {
			continue
		}
		if !strings.EqualFold(file.Name, skill.SkillFile) {
			resources = append(resources, file.Name)
			continue
		}
		content, readErr := readSkillFileContent(ctx, l.tenantID, file)
		if readErr != nil {
			continue
		}
		raw = content
	}
	if len(raw) == 0 {
		// A folder without SKILL.md is not a skill; skipping it keeps one broken
		// package from breaking the catalog.
		return skill.Skill{}, false
	}
	item, body := skill.ParseManifest(string(raw), folderName)
	item.Body = body
	item.Resources = resources
	return item, true
}

// HarnessSkillOptions lists the skills an agent may select, for the chat UI.
//
// The UI needs the same resolution the loader uses, or a user would see a skill
// in the picker that the run then cannot read.
func HarnessSkillOptions(ctx context.Context, userID, tenantID string) ([]skill.Skill, error) {
	return NewHarnessSkillLoader(userID, tenantID).List(ctx)
}

// compile-time check that the loader satisfies what the harness expects.
var _ skill.Loader = (*HarnessSkillLoader)(nil)

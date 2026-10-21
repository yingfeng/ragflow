package store

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"ragflow/internal/entity"
	"ragflow/internal/harness/toolkit"
)

// GrantStore persists standing tool permissions. It implements
// toolkit.GrantStore, so the approver consults it without knowing where the
// answer is kept.
type GrantStore struct {
	db *gorm.DB
}

// NewGrantStore builds a grant store.
func NewGrantStore(db *gorm.DB) *GrantStore { return &GrantStore{db: db} }

// Granted reports whether the tenant already allowed this exact call.
func (s *GrantStore) Granted(ctx context.Context, grant toolkit.Grant) (bool, error) {
	if grant.TenantID == "" || grant.Tool == "" || grant.ArgsHash == "" {
		return false, nil
	}
	var total int64
	err := s.db.WithContext(ctx).
		Model(&entity.HarnessAgentPermissionGrant{}).
		Where("tenant_id = ? AND tool = ? AND args_hash = ?", grant.TenantID, grant.Tool, grant.ArgsHash).
		Count(&total).Error
	if err != nil {
		return false, fmt.Errorf("store: read grant: %w", err)
	}
	return total > 0, nil
}

// Remember records a grant. Recording the same grant twice is a no-op rather
// than an error: two operators can answer the same request.
func (s *GrantStore) Remember(ctx context.Context, grant toolkit.Grant) error {
	if grant.TenantID == "" || grant.Tool == "" || grant.ArgsHash == "" {
		return fmt.Errorf("store: grant needs tenant, tool and args hash")
	}
	row := &entity.HarnessAgentPermissionGrant{
		ID:       uuid.NewString(),
		TenantID: grant.TenantID,
		Tool:     grant.Tool,
		ArgsHash: grant.ArgsHash,
	}
	err := s.db.WithContext(ctx).
		Clauses(clause.OnConflict{DoNothing: true}).
		Create(row).Error
	if err != nil {
		return fmt.Errorf("store: remember grant: %w", err)
	}
	return nil
}

// Revoke removes a grant. It reports whether one existed.
func (s *GrantStore) Revoke(ctx context.Context, grant toolkit.Grant) (bool, error) {
	result := s.db.WithContext(ctx).
		Where("tenant_id = ? AND tool = ? AND args_hash = ?", grant.TenantID, grant.Tool, grant.ArgsHash).
		Delete(&entity.HarnessAgentPermissionGrant{})
	if result.Error != nil {
		return false, fmt.Errorf("store: revoke grant: %w", result.Error)
	}
	return result.RowsAffected > 0, nil
}

var _ toolkit.GrantStore = (*GrantStore)(nil)

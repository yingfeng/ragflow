package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

// serveWithUser runs one request through bindCallerTenant with userID already in
// the context, the way the auth middleware leaves it, and returns the tenant the
// handler saw.
func serveWithUser(t *testing.T, userID, boundTenant string) string {
	t.Helper()
	gin.SetMode(gin.TestMode)

	var seen string
	router := gin.New()
	router.GET("/probe", func(c *gin.Context) {
		if userID != "" {
			c.Set("user_id", userID)
		}
		if boundTenant != "" {
			c.Set("tenant_id", boundTenant)
		}
	}, bindCallerTenant(), func(c *gin.Context) {
		seen = c.GetString("tenant_id")
	})

	router.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/probe", nil))
	return seen
}

func TestBindCallerTenantResolvesFromTheUser(t *testing.T) {
	original := tenantOfUser
	t.Cleanup(func() { tenantOfUser = original })

	tenantOfUser = func(_ context.Context, userID string) (string, error) {
		if userID != "user-1" {
			t.Fatalf("resolved tenant for %q, want the authenticated user", userID)
		}
		return "tenant-1", nil
	}

	if got := serveWithUser(t, "user-1", ""); got != "tenant-1" {
		t.Fatalf("tenant = %q, want the user's tenant", got)
	}
}

func TestBindCallerTenantKeepsATenantBoundByAuth(t *testing.T) {
	original := tenantOfUser
	t.Cleanup(func() { tenantOfUser = original })
	tenantOfUser = func(context.Context, string) (string, error) {
		t.Fatal("a tenant bound by auth must not be re-resolved")
		return "", nil
	}

	if got := serveWithUser(t, "user-1", "tenant-from-token"); got != "tenant-from-token" {
		t.Fatalf("tenant = %q, want the token's tenant", got)
	}
}

func TestBindCallerTenantLeavesTheTenantEmptyWhenResolutionFails(t *testing.T) {
	original := tenantOfUser
	t.Cleanup(func() { tenantOfUser = original })
	tenantOfUser = func(context.Context, string) (string, error) {
		return "", errors.New("tenant not found")
	}

	// An unresolved tenant is the handler's call: it rejects the request itself
	// rather than the middleware inventing a scope.
	if got := serveWithUser(t, "user-1", ""); got != "" {
		t.Fatalf("tenant = %q, want empty", got)
	}
}

func TestBindCallerTenantIgnoresAnUnauthenticatedRequest(t *testing.T) {
	if got := serveWithUser(t, "", ""); got != "" {
		t.Fatalf("tenant = %q, want empty", got)
	}
}

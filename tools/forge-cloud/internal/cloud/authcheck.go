package cloud

import (
	"fmt"
	"strings"

	lit "github.com/lightning-ai/sdk/go"
)

// AuthCheckResult is a non-billable connectivity probe (no Studio start).
type AuthCheckResult struct {
	OK            bool
	OrgID         string
	OrgName       string
	TeamspaceID   string
	TeamspaceName string
	Message       string
	// VisibleOrgs lists org names only (no secrets) when resolution fails.
	VisibleOrgs []string
	// VisibleTeamspaces lists "owner/name" hints when resolution fails.
	VisibleTeamspaces []string
}

// AuthCheck resolves organization + teamspace via the official Go SDK.
// Does not create or start Studios.
func (c *Client) AuthCheck() AuthCheckResult {
	if c.Cfg.OrgName == "" || c.Cfg.TeamspaceName == "" {
		return AuthCheckResult{
			Message: "set FORGE_CLOUD_ORG and FORGE_CLOUD_TEAMSPACE before API calls",
		}
	}

	// Preferred path: org-owned teamspace.
	org, err := lit.GetOrganization(c.Cfg.OrgName)
	if err == nil {
		ts, err := lit.GetTeamspace(c.Cfg.TeamspaceName, lit.TeamspaceOptions{Owner: org})
		if err == nil {
			return AuthCheckResult{
				OK:            true,
				OrgID:         org.ID(),
				OrgName:       org.Name(),
				TeamspaceID:   ts.ID(),
				TeamspaceName: ts.Name(),
				Message:       "organization and teamspace resolved",
			}
		}
		res := AuthCheckResult{
			OrgName: org.Name(),
			OrgID:   org.ID(),
			Message: fmt.Sprintf("GetTeamspace %q under org failed: %v", c.Cfg.TeamspaceName, redactErr(err)),
		}
		res.VisibleTeamspaces = listOrgTeamspaceNames(org)
		return res
	}

	orgErr := redactErr(err)

	// Fallback: list orgs/teamspaces visible to this user (by LIGHTNING_USER_ID).
	user, uerr := lit.GetUser("", lit.UserOptions{ID: c.Cfg.UserID})
	if uerr != nil {
		return AuthCheckResult{
			Message: fmt.Sprintf("GetOrganization(%q) failed: %s; GetUser fallback: %v",
				c.Cfg.OrgName, orgErr, redactErr(uerr)),
		}
	}

	res := AuthCheckResult{
		Message: fmt.Sprintf("GetOrganization(%q) failed: %s", c.Cfg.OrgName, orgErr),
	}
	if orgs, err := user.Organizations(); err == nil {
		for _, o := range orgs {
			if o != nil {
				res.VisibleOrgs = append(res.VisibleOrgs, o.Name())
			}
		}
		// If exactly one org matches loosely, try it.
		for _, o := range orgs {
			if o == nil {
				continue
			}
			if strings.EqualFold(o.Name(), c.Cfg.OrgName) ||
				strings.Contains(strings.ToLower(o.Name()), strings.ToLower(strings.TrimSuffix(c.Cfg.OrgName, " Org"))) {
				ts, err := lit.GetTeamspace(c.Cfg.TeamspaceName, lit.TeamspaceOptions{Owner: o})
				if err == nil {
					return AuthCheckResult{
						OK:            true,
						OrgID:         o.ID(),
						OrgName:       o.Name(),
						TeamspaceID:   ts.ID(),
						TeamspaceName: ts.Name(),
						Message:       "resolved via organization list fuzzy match",
						VisibleOrgs:   res.VisibleOrgs,
					}
				}
			}
		}
	}

	// Personal teamspaces owned by the user.
	if spaces, err := user.Teamspaces(); err == nil {
		for _, ts := range spaces {
			if ts == nil {
				continue
			}
			res.VisibleTeamspaces = append(res.VisibleTeamspaces, ts.Name())
			if strings.EqualFold(ts.Name(), c.Cfg.TeamspaceName) {
				return AuthCheckResult{
					OK:            true,
					OrgName:       "(user-owned)",
					TeamspaceID:   ts.ID(),
					TeamspaceName: ts.Name(),
					Message:       "resolved user-owned teamspace",
					VisibleOrgs:   res.VisibleOrgs,
				}
			}
		}
	}

	res.Message += "; listed visible orgs/teamspaces for correction (names only)"
	return res
}

func listOrgTeamspaceNames(org *lit.Organization) []string {
	if org == nil {
		return nil
	}
	spaces, err := org.Teamspaces()
	if err != nil {
		return nil
	}
	var out []string
	for _, ts := range spaces {
		if ts != nil {
			out = append(out, ts.Name())
		}
	}
	return out
}

func redactErr(err error) string {
	s := err.Error()
	if strings.Contains(strings.ToLower(s), "api") && strings.Contains(strings.ToLower(s), "key") {
		return "authentication or authorization error (details redacted)"
	}
	if len(s) > 240 {
		return s[:240] + "…"
	}
	return s
}

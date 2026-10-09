package cloud

import (
	"fmt"
	"strings"

	lit "github.com/lightning-ai/sdk/go"
)

// ResolveTeamspace finds the configured teamspace (org-owned or user-owned).
func (c *Client) ResolveTeamspace() (*lit.Teamspace, *lit.Organization, error) {
	if c.Cfg.TeamspaceName == "" {
		return nil, nil, fmt.Errorf("set FORGE_CLOUD_TEAMSPACE")
	}

	if c.Cfg.OrgName != "" {
		if org, err := lit.GetOrganization(c.Cfg.OrgName); err == nil {
			ts, err := lit.GetTeamspace(c.Cfg.TeamspaceName, lit.TeamspaceOptions{Owner: org})
			if err == nil {
				return ts, org, nil
			}
		}
	}

	user, err := lit.GetUser("", lit.UserOptions{ID: c.Cfg.UserID})
	if err != nil {
		return nil, nil, fmt.Errorf("GetUser: %w", err)
	}

	if orgs, err := user.Organizations(); err == nil {
		needle := strings.ToLower(strings.TrimSpace(c.Cfg.OrgName))
		needleBase := strings.TrimSuffix(needle, " org")
		for _, o := range orgs {
			if o == nil {
				continue
			}
			name := strings.ToLower(o.Name())
			if needle == "" || name == needle || strings.Contains(name, needleBase) {
				ts, err := lit.GetTeamspace(c.Cfg.TeamspaceName, lit.TeamspaceOptions{Owner: o})
				if err == nil {
					return ts, o, nil
				}
			}
		}
		// Last resort: try every visible org for the teamspace name.
		for _, o := range orgs {
			if o == nil {
				continue
			}
			ts, err := lit.GetTeamspace(c.Cfg.TeamspaceName, lit.TeamspaceOptions{Owner: o})
			if err == nil {
				return ts, o, nil
			}
		}
	}

	if spaces, err := user.Teamspaces(); err == nil {
		for _, ts := range spaces {
			if ts != nil && strings.EqualFold(ts.Name(), c.Cfg.TeamspaceName) {
				return ts, nil, nil
			}
		}
	}

	return nil, nil, fmt.Errorf("teamspace %q not found under org %q (check forge-cloud auth for visible names)",
		c.Cfg.TeamspaceName, c.Cfg.OrgName)
}

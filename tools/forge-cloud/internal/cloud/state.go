package cloud

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// SessionState is persisted locally so stop/exec can find the Studio after start.
// It never stores API keys.
type SessionState struct {
	StudioName   string    `json:"studio_name"`
	StudioID     string    `json:"studio_id,omitempty"`
	Org          string    `json:"org"`
	Teamspace    string    `json:"teamspace"`
	Machine      string    `json:"machine"`
	StartedAt    time.Time `json:"started_at"`
	MaxSession   string    `json:"max_session"`
	SpendLimit   float64   `json:"spend_limit_usd"`
	EstimateUSD  float64   `json:"estimate_usd"`
	AutoSleepMin int       `json:"auto_sleep_min"`
}

func statePath() string {
	if p := os.Getenv("FORGE_CLOUD_STATE_PATH"); p != "" {
		return p
	}
	// Prefer repo configs/cloud when running from monorepo; else user cache.
	candidates := []string{
		filepath.Join("configs", "cloud", ".forge-cloud-state.json"),
		filepath.Join("..", "configs", "cloud", ".forge-cloud-state.json"),
	}
	for _, c := range candidates {
		if dir := filepath.Dir(c); dirReadable(dir) {
			return c
		}
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".forge", "cloud-state.json")
}

func dirReadable(dir string) bool {
	fi, err := os.Stat(dir)
	return err == nil && fi.IsDir()
}

// SaveState writes session metadata (no secrets).
func SaveState(s SessionState) error {
	path := statePath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

// LoadState reads the last session, if any.
func LoadState() (SessionState, error) {
	var s SessionState
	data, err := os.ReadFile(statePath())
	if err != nil {
		return s, err
	}
	err = json.Unmarshal(data, &s)
	return s, err
}

// ClearState removes the local session file.
func ClearState() error {
	path := statePath()
	err := os.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// StatePath exposes the path for diagnostics.
func StatePath() string { return statePath() }

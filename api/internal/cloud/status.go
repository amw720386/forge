package cloud

// StatusReport describes non-billable configuration state.
type StatusReport struct {
	CredentialsConfigured bool
	RedactedSummary       string
	SDKModule             string
	StatePath             string
	Note                  string
}

// Status loads config (including repo-root .env) and reports redacted state.
func Status() StatusReport {
	cfg, err := LoadConfig()
	return StatusFrom(cfg, err)
}

// StatusFrom builds a report from an already-loaded config / error.
func StatusFrom(cfg Config, err error) StatusReport {
	rep := StatusReport{
		SDKModule: "github.com/lightning-ai/sdk/go",
		StatePath: StatePath(),
		Note: "Phase 8A: status/machines/dry-run are free. " +
			"start requires --confirm and respects spend limits. " +
			"Do not provision GPUs without explicit approval.",
	}
	if err == nil {
		rep.CredentialsConfigured = true
		rep.RedactedSummary = cfg.Redacted()
	} else if err == ErrMissingCredentials {
		rep.CredentialsConfigured = false
		rep.RedactedSummary = cfg.Redacted()
		if cfg.EnvFileLoaded == "" {
			rep.RedactedSummary = "credentials not configured (set LIGHTNING_* or repo-root .env)"
		}
	} else {
		rep.CredentialsConfigured = false
		rep.RedactedSummary = "config error (details omitted if secret-related)"
		rep.Note = err.Error()
	}
	return rep
}

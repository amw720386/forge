package cloud

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds Lightning credentials and cost-control defaults from the environment.
// API keys are never logged in full by helpers in this package.
type Config struct {
	UserID string
	APIKey string

	// Teamspace resolution (required for Studio APIs).
	OrgName       string
	TeamspaceName string

	StudioName string
	// Default machine slug alias: l4, t4, cpu, …
	DefaultGPU string

	SpendLimitUSD    float64
	MaxSession       time.Duration
	RequireConfirm   bool
	AutoSleepMinutes int

	// Optional remote working directory hint for exec scripts.
	RemoteWorkDir string

	// EnvFileLoaded is the path that was applied (empty if none).
	EnvFileLoaded string
}

// ErrMissingCredentials indicates LIGHTNING_USER_ID / LIGHTNING_API_KEY are unset.
var ErrMissingCredentials = errors.New("missing Lightning credentials")

// ErrInvalidConfig indicates a malformed non-secret setting.
var ErrInvalidConfig = errors.New("invalid cloud configuration")

// LoadConfig reads configuration with precedence:
//  1. Process environment (already set)
//  2. Repository-root / --env-file .env (does not override process env)
//  3. Safe defaults for non-secrets
func LoadConfig() (Config, error) {
	return LoadConfigWith(LoadOptions{})
}

// LoadConfigWith applies optional .env discovery then reads env vars.
func LoadConfigWith(opt LoadOptions) (Config, error) {
	loaded, err := ApplyDotEnv(opt)
	if err != nil {
		return Config{}, err
	}

	cfg := Config{
		UserID:           strings.TrimSpace(os.Getenv("LIGHTNING_USER_ID")),
		APIKey:           strings.TrimSpace(os.Getenv("LIGHTNING_API_KEY")),
		OrgName:          firstEnv("FORGE_CLOUD_ORG", "LIGHTNING_ORG"),
		TeamspaceName:    firstEnv("FORGE_CLOUD_TEAMSPACE", "LIGHTNING_TEAMSPACE"),
		StudioName:       envOr("FORGE_CLOUD_STUDIO", "forge-gpu"),
		DefaultGPU:       strings.ToLower(envOr("FORGE_CLOUD_MACHINE", "l4")),
		SpendLimitUSD:    3.0,
		MaxSession:       30 * time.Minute,
		RequireConfirm:   envBool("FORGE_CLOUD_REQUIRE_CONFIRM", true),
		AutoSleepMinutes: 15,
		RemoteWorkDir:    strings.TrimSpace(os.Getenv("FORGE_CLOUD_REMOTE_DIR")),
		EnvFileLoaded:    loaded,
	}

	if v := strings.TrimSpace(os.Getenv("FORGE_CLOUD_SPEND_LIMIT_USD")); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return cfg, fmt.Errorf("%w: FORGE_CLOUD_SPEND_LIMIT_USD: %v", ErrInvalidConfig, err)
		}
		if f <= 0 {
			return cfg, fmt.Errorf("%w: FORGE_CLOUD_SPEND_LIMIT_USD must be > 0", ErrInvalidConfig)
		}
		if f > 100 {
			return cfg, fmt.Errorf("%w: FORGE_CLOUD_SPEND_LIMIT_USD $%.2f exceeds hard safety cap $100", ErrInvalidConfig, f)
		}
		cfg.SpendLimitUSD = f
	}

	if v := strings.TrimSpace(os.Getenv("FORGE_CLOUD_MAX_SESSION")); v != "" {
		d, err := parseDurationFlexible(v)
		if err != nil {
			return cfg, fmt.Errorf("%w: FORGE_CLOUD_MAX_SESSION: %v", ErrInvalidConfig, err)
		}
		if d < time.Minute {
			return cfg, fmt.Errorf("%w: FORGE_CLOUD_MAX_SESSION must be >= 1m", ErrInvalidConfig)
		}
		if d > 4*time.Hour {
			return cfg, fmt.Errorf("%w: FORGE_CLOUD_MAX_SESSION must be <= 4h", ErrInvalidConfig)
		}
		cfg.MaxSession = d
	}

	if v := strings.TrimSpace(os.Getenv("FORGE_CLOUD_AUTO_SLEEP_MIN")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 {
			return cfg, fmt.Errorf("%w: FORGE_CLOUD_AUTO_SLEEP_MIN must be a non-negative integer", ErrInvalidConfig)
		}
		cfg.AutoSleepMinutes = n
	}

	if cfg.UserID == "" || cfg.APIKey == "" {
		return cfg, ErrMissingCredentials
	}
	return cfg, nil
}

// Redacted returns a diagnostics string that never includes the raw API key.
func (c Config) Redacted() string {
	uid := c.UserID
	if len(uid) > 6 {
		uid = uid[:3] + "…" + uid[len(uid)-2:]
	} else if uid != "" {
		uid = "***"
	} else {
		uid = "(unset)"
	}
	key := "(unset)"
	if c.APIKey != "" {
		key = fmt.Sprintf("set(len=%d)", len(c.APIKey))
	}
	envNote := "(none)"
	if c.EnvFileLoaded != "" {
		envNote = c.EnvFileLoaded
	}
	return fmt.Sprintf(
		"LIGHTNING_USER_ID=%s LIGHTNING_API_KEY=%s org=%q teamspace=%q studio=%q gpu=%s spend_limit=$%.2f max_session=%s env_file=%s",
		uid, key, c.OrgName, c.TeamspaceName, c.StudioName, c.DefaultGPU, c.SpendLimitUSD, c.MaxSession, envNote,
	)
}

func firstEnv(keys ...string) string {
	for _, k := range keys {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			return v
		}
	}
	return ""
}

func envOr(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func envBool(key string, def bool) bool {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	switch strings.ToLower(v) {
	case "1", "true", "yes", "y":
		return true
	case "0", "false", "no", "n":
		return false
	default:
		return def
	}
}

func parseDurationFlexible(v string) (time.Duration, error) {
	if d, err := time.ParseDuration(v); err == nil {
		return d, nil
	}
	if n, err := strconv.Atoi(v); err == nil {
		return time.Duration(n) * time.Minute, nil
	}
	return 0, fmt.Errorf("invalid duration %q", v)
}

func envFloat(key string, def float64) float64 {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return def
	}
	return f
}

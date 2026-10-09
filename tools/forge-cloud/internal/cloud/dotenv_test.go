package cloud

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestResolveEnvFileExplicit(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "custom.env")
	if err := os.WriteFile(p, []byte("LIGHTNING_USER_ID=abc\nLIGHTNING_API_KEY=secretkey\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := ResolveEnvFile(LoadOptions{EnvFile: p})
	if err != nil {
		t.Fatal(err)
	}
	if got != p && filepath.Clean(got) != filepath.Clean(p) {
		// Abs may normalize; compare base
		if filepath.Base(got) != "custom.env" {
			t.Fatalf("got %q", got)
		}
	}
}

func TestResolveEnvFileMissingExplicit(t *testing.T) {
	_, err := ResolveEnvFile(LoadOptions{EnvFile: filepath.Join(t.TempDir(), "nope.env")})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRepoRootDetectionFromAPISubdir(t *testing.T) {
	// This test file lives under api/internal/cloud — walk up to monorepo .env if present.
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller")
	}
	cloudDir := filepath.Dir(thisFile)
	// Simulate: start search from cloudDir via chdir
	old, _ := os.Getwd()
	defer os.Chdir(old)
	if err := os.Chdir(cloudDir); err != nil {
		t.Fatal(err)
	}
	p := findRepoRootEnv()
	if p == "" {
		t.Skip("no repo-root .env in this checkout (CI without secrets)")
	}
	if !strings.HasSuffix(filepath.Clean(p), string(filepath.Separator)+".env") && filepath.Base(p) != ".env" {
		t.Fatalf("unexpected path %q", p)
	}
	root := filepath.Dir(p)
	if !looksLikeRepoRoot(root) {
		t.Fatalf("parent of .env not detected as repo root: %s", root)
	}
}

func TestApplyDotEnvDoesNotOverrideProcessEnv(t *testing.T) {
	dir := t.TempDir()
	// Fake a mini repo root
	_ = os.MkdirAll(filepath.Join(dir, "kernels"), 0o755)
	_ = os.MkdirAll(filepath.Join(dir, "api"), 0o755)
	envPath := filepath.Join(dir, ".env")
	content := "LIGHTNING_USER_ID=fromfile\nLIGHTNING_API_KEY=filekey123\nFORGE_CLOUD_ORG=fileorg\n"
	if err := os.WriteFile(envPath, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LIGHTNING_USER_ID", "fromprocess")
	t.Setenv("LIGHTNING_API_KEY", "processkey")
	os.Unsetenv("FORGE_CLOUD_ORG")

	loaded, err := ApplyDotEnv(LoadOptions{EnvFile: envPath})
	if err != nil {
		t.Fatal(err)
	}
	if loaded == "" {
		t.Fatal("expected loaded path")
	}
	if os.Getenv("LIGHTNING_USER_ID") != "fromprocess" {
		t.Fatalf("process env should win, got %q", os.Getenv("LIGHTNING_USER_ID"))
	}
	if os.Getenv("FORGE_CLOUD_ORG") != "fileorg" {
		t.Fatalf("file should fill missing org, got %q", os.Getenv("FORGE_CLOUD_ORG"))
	}
}

func TestLoadConfigMissingCredentials(t *testing.T) {
	t.Setenv("LIGHTNING_USER_ID", "")
	t.Setenv("LIGHTNING_API_KEY", "")
	_, err := LoadConfigWith(LoadOptions{SkipDotEnv: true})
	if err != ErrMissingCredentials {
		t.Fatalf("got %v", err)
	}
}

func TestLoadConfigInvalidSpendLimit(t *testing.T) {
	t.Setenv("LIGHTNING_USER_ID", "u")
	t.Setenv("LIGHTNING_API_KEY", "k")
	t.Setenv("FORGE_CLOUD_SPEND_LIMIT_USD", "not-a-number")
	_, err := LoadConfigWith(LoadOptions{SkipDotEnv: true})
	if err == nil || !strings.Contains(err.Error(), "FORGE_CLOUD_SPEND_LIMIT_USD") {
		t.Fatalf("got %v", err)
	}
}

func TestLoadConfigNegativeSpendRejected(t *testing.T) {
	t.Setenv("LIGHTNING_USER_ID", "u")
	t.Setenv("LIGHTNING_API_KEY", "k")
	t.Setenv("FORGE_CLOUD_SPEND_LIMIT_USD", "-1")
	_, err := LoadConfigWith(LoadOptions{SkipDotEnv: true})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestLoadConfigOptionalDefaults(t *testing.T) {
	t.Setenv("LIGHTNING_USER_ID", "userid")
	t.Setenv("LIGHTNING_API_KEY", "apikey")
	os.Unsetenv("FORGE_CLOUD_MACHINE")
	os.Unsetenv("FORGE_CLOUD_MAX_SESSION")
	os.Unsetenv("FORGE_CLOUD_AUTO_SLEEP_MIN")
	cfg, err := LoadConfigWith(LoadOptions{SkipDotEnv: true})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DefaultGPU != "l4" {
		t.Fatalf("gpu=%q", cfg.DefaultGPU)
	}
	if cfg.SpendLimitUSD != 3 {
		t.Fatalf("spend=%v", cfg.SpendLimitUSD)
	}
	if cfg.AutoSleepMinutes != 15 {
		t.Fatalf("sleep=%d", cfg.AutoSleepMinutes)
	}
}

func TestRedactedNeverContainsAPIKey(t *testing.T) {
	secret := "super-secret-api-key-value-xyz"
	c := Config{
		UserID:        "abcdefghij",
		APIKey:        secret,
		OrgName:       "myorg",
		TeamspaceName: "ts",
		StudioName:    "forge-gpu",
		DefaultGPU:    "l4",
		SpendLimitUSD: 3,
	}
	s := c.Redacted()
	if strings.Contains(s, secret) {
		t.Fatal("API key leaked in redacted output")
	}
	if strings.Contains(s, "super-secret") {
		t.Fatal("API key fragment leaked")
	}
	if !strings.Contains(s, "set(len=") {
		t.Fatalf("expected redacted key marker: %s", s)
	}
}

func TestWindowsPathEnvFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "win.env")
	if err := os.WriteFile(p, []byte("LIGHTNING_USER_ID=a\nLIGHTNING_API_KEY=b\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	// Use mixed separators on Windows
	mixed := p
	if runtime.GOOS == "windows" {
		mixed = strings.ReplaceAll(p, "/", `\`)
	}
	got, err := ResolveEnvFile(LoadOptions{EnvFile: mixed})
	if err != nil {
		t.Fatal(err)
	}
	if !fileExists(got) {
		t.Fatalf("resolved missing: %s", got)
	}
}

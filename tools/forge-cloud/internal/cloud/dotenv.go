package cloud

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/joho/godotenv"
)

// LoadOptions controls .env discovery for forge-cloud.
type LoadOptions struct {
	// EnvFile is an explicit path from --env-file. Empty = auto-discover.
	EnvFile string
	// SkipDotEnv disables file loading (tests / already-populated env).
	SkipDotEnv bool
}

// ResolveEnvFile returns the .env path that will be loaded.
// Precedence for path selection:
//  1. Explicit EnvFile (--env-file)
//  2. FORGE_ENV_FILE environment variable
//  3. Walk upward from the executable directory, then cwd, looking for .env
//     next to go.mod or markers (api/, kernels/, runtime/).
func ResolveEnvFile(opt LoadOptions) (string, error) {
	if opt.EnvFile != "" {
		p, err := filepath.Abs(opt.EnvFile)
		if err != nil {
			return "", err
		}
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("env-file %q: %w", p, err)
		}
		return p, nil
	}
	if v := strings.TrimSpace(os.Getenv("FORGE_ENV_FILE")); v != "" {
		p, err := filepath.Abs(v)
		if err != nil {
			return "", err
		}
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("FORGE_ENV_FILE %q: %w", p, err)
		}
		return p, nil
	}
	if p := findRepoRootEnv(); p != "" {
		return p, nil
	}
	return "", nil // no file; process env / defaults only
}

func findRepoRootEnv() string {
	var starts []string
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	if cwd, err := os.Getwd(); err == nil {
		starts = append(starts, cwd)
	}
	seen := map[string]bool{}
	for _, start := range starts {
		dir := start
		for i := 0; i < 12; i++ {
			abs, err := filepath.Abs(dir)
			if err != nil || seen[abs] {
				break
			}
			seen[abs] = true
			candidate := filepath.Join(abs, ".env")
			if fileExists(candidate) && looksLikeRepoRoot(abs) {
				return candidate
			}
			// Also accept .env in a parent that contains api/ + kernels/ even if
			// go.mod is only under api/.
			parent := filepath.Dir(abs)
			if parent == abs {
				break
			}
			dir = parent
		}
	}
	// Second pass: any .env beside repo markers without requiring prior hit.
	for _, start := range starts {
		dir := start
		for i := 0; i < 12; i++ {
			abs, _ := filepath.Abs(dir)
			candidate := filepath.Join(abs, ".env")
			if fileExists(candidate) && looksLikeRepoRoot(abs) {
				return candidate
			}
			parent := filepath.Dir(abs)
			if parent == abs {
				break
			}
			dir = parent
		}
	}
	return ""
}

func looksLikeRepoRoot(dir string) bool {
	// Monorepo root: has kernels/ and (api/ or runtime/).
	hasKernels := dirExists(filepath.Join(dir, "kernels"))
	hasAPI := dirExists(filepath.Join(dir, "api"))
	hasRuntime := dirExists(filepath.Join(dir, "runtime"))
	if hasKernels && (hasAPI || hasRuntime) {
		return true
	}
	// Fallback: directory that contains .env.example and docs/
	if fileExists(filepath.Join(dir, ".env.example")) && dirExists(filepath.Join(dir, "docs")) {
		return true
	}
	return false
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

func dirExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

// ApplyDotEnv loads KEY=VALUE pairs from the resolved .env file into the
// process environment **without overwriting** variables that are already set.
// Precedence: existing process env > .env file > later defaults in LoadConfig.
func ApplyDotEnv(opt LoadOptions) (loadedPath string, err error) {
	if opt.SkipDotEnv {
		return "", nil
	}
	path, err := ResolveEnvFile(opt)
	if err != nil {
		return "", err
	}
	if path == "" {
		return "", nil
	}
	// godotenv.Load does not override existing env vars.
	if err := godotenv.Load(path); err != nil {
		// Never wrap the underlying error — godotenv may embed file contents
		// (including secrets) in err.Error().
		return path, fmt.Errorf("load env file %s: parse failed (check UTF-8 without BOM, KEY=VALUE lines)", filepath.Base(path))
	}
	return path, nil
}

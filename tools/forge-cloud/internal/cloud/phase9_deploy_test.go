package cloud

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestPhase9DeployPipelineE2E proves the exact validate-phase9 archive path:
// build → extract tar → Go verify → bash verify_forge_deploy.sh → no post-verify mutation.
// Runs twice to catch nondeterminism in content_hash.
func TestPhase9DeployPipelineE2E(t *testing.T) {
	repo := findTestRepoRoot(t)
	python, err := exec.LookPath("python")
	if err != nil {
		python, err = exec.LookPath("python3")
	}
	if err != nil {
		t.Fatal("Python is required for deployment verification:", err)
	}
	pythonShellPath := strings.ReplaceAll(toGitBashPath(python), "'", "'\"'\"'")
	var hashes []string
	for pass := 1; pass <= 2; pass++ {
		t.Run(fmt.Sprintf("pass_%d", pass), func(t *testing.T) {
			work := t.TempDir()
			bundle, err := BuildDeployBundle(repo, work)
			if err != nil {
				t.Fatalf("BuildDeployBundle: %v", err)
			}
			hashes = append(hashes, bundle.Manifest.ContentHash)

			// Extract the exact tar used by validate-phase9 upload.
			extractDir := filepath.Join(work, "extracted")
			if err := extractGzipTar(bundle.TarPath, extractDir); err != nil {
				t.Fatalf("extract: %v", err)
			}

			// Go-side verify (same algorithm as remote content_hash).
			man, err := VerifyStagingAgainstManifest(extractDir)
			if err != nil {
				t.Fatalf("VerifyStagingAgainstManifest: %v", err)
			}
			if man.ContentHash != bundle.Manifest.ContentHash {
				t.Fatalf("content_hash mismatch after extract: got %s want %s",
					man.ContentHash, bundle.Manifest.ContentHash)
			}

			// Every staged .sh must be LF-only (CRLF would be mutated by remote sed historically).
			if err := assertShellScriptsLF(extractDir); err != nil {
				t.Fatalf("LF gate: %v", err)
			}

			// Snapshot all file digests before remote STAGE -1 post-steps.
			before, err := hashTree(extractDir)
			if err != nil {
				t.Fatalf("hashTree before: %v", err)
			}

			// Exact remote verify script (requires bash + sha256sum — Git Bash on Windows).
			bash := findBash(t)
			extractUnix := toGitBashPath(extractDir)
			verifyCmd := fmt.Sprintf(
				`cd '%s' && python3() { '%s' "$@"; } && export -f python3 && command -v python3 && python3 -c 'print(1)' && bash scripts/verify_forge_deploy.sh`,
				extractUnix, pythonShellPath,
			)
			cmd := exec.Command(bash, "-lc", verifyCmd)
			out, err := cmd.CombinedOutput()
			t.Logf("verify_forge_deploy.sh:\n%s", out)
			if err != nil {
				t.Fatalf("verify_forge_deploy.sh failed: %v", err)
			}
			if !bytes.Contains(out, []byte("DEPLOY_VERIFY_OK")) {
				t.Fatal("missing DEPLOY_VERIFY_OK")
			}

			// Simulate remote_cuda_phase9.sh after verify: chmod only (no sed).
			chmodCmd := exec.Command(bash, "-lc", fmt.Sprintf("cd '%s' && chmod +x scripts/*.sh", extractUnix))
			if out, err := chmodCmd.CombinedOutput(); err != nil {
				t.Fatalf("chmod: %v\n%s", err, out)
			}

			after, err := hashTree(extractDir)
			if err != nil {
				t.Fatalf("hashTree after: %v", err)
			}
			if diff := digestDiff(before, after); diff != "" {
				t.Fatalf("hashed files changed after verify/chmod (forbidden):\n%s", diff)
			}

			// Re-verify must still pass (proves no post-verify mutation).
			if _, err := VerifyStagingAgainstManifest(extractDir); err != nil {
				t.Fatalf("re-verify after chmod: %v", err)
			}
			cmd2 := exec.Command(bash, "-lc", verifyCmd)
			out2, err := cmd2.CombinedOutput()
			if err != nil {
				t.Fatalf("second verify_forge_deploy.sh failed: %v\n%s", err, out2)
			}

			// Prove CRLF would break hashes if present — and that our staged files are not CRLF.
			crlfProbe := filepath.Join(extractDir, "scripts", "remote_cuda_phase9.sh")
			orig, err := os.ReadFile(crlfProbe)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(orig, []byte("\r")) {
				t.Fatal("staged remote_cuda_phase9.sh still contains CR")
			}
			crlf := bytes.ReplaceAll(orig, []byte("\n"), []byte("\r\n"))
			if err := os.WriteFile(crlfProbe, crlf, 0o755); err != nil {
				t.Fatal(err)
			}
			_, err = VerifyStagingAgainstManifest(extractDir)
			if err == nil {
				t.Fatal("expected verify to fail after injecting CRLF into a hashed script")
			}
			if !strings.Contains(err.Error(), "hash:") && !strings.Contains(err.Error(), "content_hash") {
				t.Fatalf("CRLF inject should fail on hash, got: %v", err)
			}
			// Restore LF and confirm recover.
			if err := os.WriteFile(crlfProbe, orig, 0o755); err != nil {
				t.Fatal(err)
			}
			if _, err := VerifyStagingAgainstManifest(extractDir); err != nil {
				t.Fatalf("restore LF should verify: %v", err)
			}
		})
	}
	if len(hashes) != 2 {
		t.Fatalf("expected 2 passes, got %d", len(hashes))
	}
	if hashes[0] != hashes[1] {
		t.Fatalf("nondeterministic content_hash across builds: %s vs %s", hashes[0], hashes[1])
	}
}

// TestPhase9DeployNormalizesCRLFSources ensures Windows CRLF sources become LF in the archive.
func TestPhase9DeployNormalizesCRLFSources(t *testing.T) {
	repo := findTestRepoRoot(t)
	work := t.TempDir()

	// Build a shadow scripts tree with intentional CRLF and point a mini include via temp copy.
	// We mutate a copy of the repo's scripts into work and build from a fake root is heavy;
	// instead: copyFile unit-level through BuildDeployBundle on real repo, then assert staged LF,
	// and separately prove copyFile strips CR.
	src := filepath.Join(work, "crlf.sh")
	dst := filepath.Join(work, "out.sh")
	if err := os.WriteFile(src, []byte("#!/bin/bash\r\necho hi\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := copyFile(src, dst); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(got, []byte("\r")) {
		t.Fatalf("copyFile must strip CR from .sh, got %q", got)
	}
	if !bytes.Equal(got, []byte("#!/bin/bash\necho hi\n")) {
		t.Fatalf("unexpected normalized content: %q", got)
	}

	bundle, err := BuildDeployBundle(repo, filepath.Join(work, "bundle"))
	if err != nil {
		t.Fatalf("BuildDeployBundle: %v", err)
	}
	if err := assertShellScriptsLF(bundle.StagingDir); err != nil {
		t.Fatal(err)
	}
	_ = repo
}

func assertShellScriptsLF(root string) error {
	return filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || !strings.HasSuffix(strings.ToLower(path), ".sh") {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if bytes.Contains(b, []byte("\r")) {
			rel, _ := filepath.Rel(root, path)
			return &os.PathError{Op: "assertLF", Path: rel, Err: os.ErrInvalid}
		}
		return nil
	})
}

func hashTree(root string) (map[string]string, error) {
	out := map[string]string{}
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = slashPath(rel)
		if rel == DeployManifestName || rel == DeployIDFileName {
			return nil
		}
		sum, err := hashFile(path)
		if err != nil {
			return err
		}
		out[rel] = sum
		return nil
	})
	return out, err
}

func digestDiff(a, b map[string]string) string {
	var sb strings.Builder
	for k, va := range a {
		if vb, ok := b[k]; !ok {
			sb.WriteString("missing after: " + k + "\n")
		} else if va != vb {
			sb.WriteString("changed: " + k + " " + va[:12] + "->" + vb[:12] + "\n")
		}
	}
	for k := range b {
		if _, ok := a[k]; !ok {
			sb.WriteString("unexpected after: " + k + "\n")
		}
	}
	return sb.String()
}

func extractGzipTar(tarPath, dest string) error {
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	f, err := os.Open(tarPath)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		target := filepath.Join(dest, filepath.FromSlash(hdr.Name))
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(hdr.Mode))
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
		}
	}
	return nil
}

func findBash(t *testing.T) string {
	t.Helper()
	// Prefer Git Bash over WSL launcher (System32\bash.exe) — path mapping differs.
	candidates := []string{
		`C:\Program Files\Git\bin\bash.exe`,
		`C:\Program Files\Git\usr\bin\bash.exe`,
	}
	if bash, err := exec.LookPath("bash"); err == nil {
		candidates = append(candidates, bash)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err != nil {
			continue
		}
		cmd := exec.Command(c, "-lc", "command -v sha256sum")
		out, err := cmd.CombinedOutput()
		if err == nil && bytes.Contains(out, []byte("sha256sum")) {
			return c
		}
		t.Logf("candidate %s missing tools: %s", c, out)
	}
	t.Fatal("bash with sha256sum required for exact verify_forge_deploy.sh (use Git Bash on Windows)")
	return ""
}

// toGitBashPath converts a Windows path to a Git Bash /c/... path.
func toGitBashPath(p string) string {
	p = filepath.Clean(p)
	vol := filepath.VolumeName(p)
	if len(vol) >= 2 && vol[1] == ':' {
		rest := p[len(vol):]
		rest = filepath.ToSlash(rest)
		return "/" + strings.ToLower(string(vol[0])) + rest
	}
	return filepath.ToSlash(p)
}

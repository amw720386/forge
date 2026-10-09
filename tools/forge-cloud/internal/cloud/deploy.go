package cloud

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Remote paths under the Lightning Studio filesystem.
const (
	RemoteStudioRoot   = "/teamspace/studios/this_studio"
	RemoteDeploysDir   = RemoteStudioRoot + "/forge-deploys"
	RemoteCacheDir     = RemoteStudioRoot + "/forge-cache"
	RemoteCurrentLink  = RemoteStudioRoot + "/forge-current"
	RemoteCompatLink   = RemoteStudioRoot + "/forge"
	DeployManifestName = ".forge-deploy-manifest.json"
	DeployIDFileName   = ".forge-deploy-id"
)

// Paths included in a Phase 8B source deploy (relative to repo root).
var DeployIncludeRoots = []string{
	"tests/reference/phase16_reference.py",
	"tests/reference/phase2_reference.py",
	"api",
	"bindings/go",
	"kernels",
	"scripts",
	"runtime/Cargo.toml",
	"runtime/Cargo.lock",
	"runtime/build.rs",
	"runtime/src",
	"runtime/tests",
	"runtime/examples",
	"runtime/include",
	"docs/PHASE9_DESIGN.md",
	"docs/PHASE9_RESULTS.md",
	"docs/PHASE10_DESIGN.md",
	"docs/PHASE10_RESULTS.md",
	"docs/PHASE11_DESIGN.md",
	"docs/PHASE11_RESULTS.md",
	"docs/PHASE12_DESIGN.md",
	"docs/BENCHMARK_METHODOLOGY.md",
	"docs/PHASE12_COST_LEDGER.md",
	"benchmarks/forge",
	"benchmarks/llama_cpp",
	"benchmarks/vllm",
	"benchmarks/workloads",
	"benchmarks/analysis",
	"benchmarks/results/.gitkeep",
}

// RequiredRelPaths must exist in every deploy (forward-slash relative).
var RequiredRelPaths = []string{
	"runtime/Cargo.toml",
	"runtime/build.rs",
	"runtime/src/lib.rs",
	"runtime/src/cuda_model/mod.rs",
	"runtime/src/cuda_model/batcher.rs",
	"runtime/src/cuda_model/transformer.rs",
	"runtime/src/cuda_model/generate.rs",
	"runtime/src/cuda_model/kv.rs",
	"runtime/src/cuda_model/paged_kv.rs",
	"runtime/src/cuda_model/weights.rs",
	"kernels/CMakeLists.txt",
	"kernels/cuda/include/forge_cuda.h",
	"scripts/remote_cuda_phase8b.sh",
	"scripts/remote_cuda_phase9.sh",
	"scripts/remote_cuda_phase10.sh",
	"scripts/remote_cuda_phase11.sh",
	"scripts/remote_cuda_phase12.sh",
	"scripts/verify_forge_deploy.sh",
	"scripts/preflight_phase8b.sh",
	"runtime/tests/cuda_batch_parity.rs",
	"runtime/tests/cuda_paged_kv.rs",
	"runtime/examples/bench_cuda_batch.rs",
	"runtime/examples/bench_phase12.rs",
	"runtime/examples/bf16_correctness_survey.rs",
	"docs/PHASE9_DESIGN.md",
	"docs/PHASE10_DESIGN.md",
	"docs/PHASE11_DESIGN.md",
	"docs/PHASE12_DESIGN.md",
	"docs/BENCHMARK_METHODOLOGY.md",
	"benchmarks/forge/run_forge_bench.sh",
	"benchmarks/workloads/phase12_matrix.json",
	"kernels/cuda/src/argmax.cu",
}

// ObsoleteContentPatterns must NOT appear in deployed Rust sources.
// These are known-stale markers from Phase 8B attempt 2/3.
var ObsoleteContentPatterns = []string{
	"use crate::model::batch::",
	"use crate::model::generate::GenError",
	"use crate::model::generate::GenerateParams",
}

// DeployManifest is the source-identity document written into every deploy.
type DeployManifest struct {
	DeployID    string            `json:"deploy_id"`
	ContentHash string            `json:"content_hash"`
	GitRevision string            `json:"git_revision,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
	FileCount   int               `json:"file_count"`
	Files       map[string]string `json:"files"` // rel path (slash) -> sha256 hex
	Required    []string          `json:"required"`
}

// DeployBundle is a local staging tree ready for upload.
type DeployBundle struct {
	StagingDir string
	TarPath    string
	Manifest   DeployManifest
}

// BuildDeployBundle copies allowed sources into an isolated staging directory,
// writes a content-hash manifest, and packs a gzip tar for upload.
// Never includes .env, secrets, model weights, or build artifacts.
func BuildDeployBundle(repoRoot, workDir string) (*DeployBundle, error) {
	repoRoot = filepath.Clean(repoRoot)
	if workDir == "" {
		var err error
		workDir, err = os.MkdirTemp("", "forge-deploy-*")
		if err != nil {
			return nil, err
		}
	}
	staging := filepath.Join(workDir, "staging")
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return nil, err
	}

	files := make(map[string]string)
	for _, root := range DeployIncludeRoots {
		local := filepath.Join(repoRoot, filepath.FromSlash(root))
		// Historical reports keep their original paths inside validation bundles.
		if strings.HasPrefix(root, "docs/") {
			local = filepath.Join(repoRoot, "docs", "history", filepath.Base(root))
		}
		fi, err := os.Stat(local)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		if fi.IsDir() {
			if err := copyTreeFiltered(local, filepath.Join(staging, filepath.FromSlash(root)), files, root); err != nil {
				return nil, err
			}
		} else {
			rel := slashPath(root)
			if skipDeployRel(rel) {
				continue
			}
			dst := filepath.Join(staging, filepath.FromSlash(rel))
			if err := copyFile(local, dst); err != nil {
				return nil, err
			}
			sum, err := hashFile(dst)
			if err != nil {
				return nil, err
			}
			files[rel] = sum
		}
	}

	// Retain the original script paths for archived remote validation commands.
	archivedScripts := filepath.Join(repoRoot, "tools", "forge-cloud", "scripts")
	if err := copyTreeFiltered(archivedScripts, filepath.Join(staging, "scripts"), files, "scripts"); err != nil {
		return nil, err
	}

	if err := assertRequiredPresent(files); err != nil {
		return nil, err
	}
	if err := assertNoObsoleteContent(staging); err != nil {
		return nil, err
	}

	contentHash := hashFileMap(files)
	gitRev := gitHead(repoRoot)
	id := fmt.Sprintf("%s-%s", time.Now().UTC().Format("20060102T150405Z"), contentHash[:12])

	man := DeployManifest{
		DeployID:    id,
		ContentHash: contentHash,
		GitRevision: gitRev,
		CreatedAt:   time.Now().UTC(),
		FileCount:   len(files),
		Files:       files,
		Required:    append([]string(nil), RequiredRelPaths...),
	}

	manPath := filepath.Join(staging, DeployManifestName)
	manBytes, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(manPath, manBytes, 0o644); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(staging, DeployIDFileName), []byte(id+"\n"), 0o644); err != nil {
		return nil, err
	}

	tarPath := filepath.Join(workDir, id+".tar.gz")
	if err := writeGzipTar(staging, tarPath); err != nil {
		return nil, err
	}

	return &DeployBundle{
		StagingDir: staging,
		TarPath:    tarPath,
		Manifest:   man,
	}, nil
}

// VerifyStagingAgainstManifest re-hashes staging and compares to the manifest.
// Used locally and as the model for remote verification.
func VerifyStagingAgainstManifest(stagingDir string) (*DeployManifest, error) {
	manPath := filepath.Join(stagingDir, DeployManifestName)
	raw, err := os.ReadFile(manPath)
	if err != nil {
		return nil, fmt.Errorf("missing deploy manifest: %w", err)
	}
	var man DeployManifest
	if err := json.Unmarshal(raw, &man); err != nil {
		return nil, fmt.Errorf("invalid deploy manifest: %w", err)
	}
	if man.DeployID == "" || man.ContentHash == "" {
		return nil, fmt.Errorf("deploy manifest missing deploy_id or content_hash")
	}

	idRaw, err := os.ReadFile(filepath.Join(stagingDir, DeployIDFileName))
	if err != nil {
		return nil, fmt.Errorf("missing %s: %w", DeployIDFileName, err)
	}
	if got := strings.TrimSpace(string(idRaw)); got != man.DeployID {
		return nil, fmt.Errorf("deploy id mismatch: id file %q vs manifest %q", got, man.DeployID)
	}

	actual := make(map[string]string)
	err = filepath.Walk(stagingDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(stagingDir, path)
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
		actual[rel] = sum
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Compare source identity (exclude packaging metadata files).
	expectedSources := make(map[string]string)
	for k, v := range man.Files {
		if k == DeployManifestName || k == DeployIDFileName {
			continue
		}
		expectedSources[k] = v
	}
	if len(actual) != len(expectedSources) {
		return nil, fmt.Errorf("file count mismatch: remote/staging has %d source files, manifest expects %d",
			len(actual), len(expectedSources))
	}
	var mismatches []string
	for rel, want := range expectedSources {
		got, ok := actual[rel]
		if !ok {
			mismatches = append(mismatches, "missing:"+rel)
			continue
		}
		if got != want {
			mismatches = append(mismatches, "hash:"+rel)
		}
	}
	for rel := range actual {
		if _, ok := expectedSources[rel]; !ok {
			mismatches = append(mismatches, "unexpected:"+rel)
		}
	}
	if len(mismatches) > 0 {
		sort.Strings(mismatches)
		if len(mismatches) > 12 {
			mismatches = append(mismatches[:12], fmt.Sprintf("…+%d more", len(mismatches)-12))
		}
		return nil, fmt.Errorf("source verification failed: %s", strings.Join(mismatches, ", "))
	}

	gotHash := hashFileMap(actual)
	if gotHash != man.ContentHash {
		return nil, fmt.Errorf("content_hash mismatch: got %s want %s", gotHash, man.ContentHash)
	}
	if err := assertRequiredPresent(actual); err != nil {
		return nil, err
	}
	if err := assertNoObsoleteContent(stagingDir); err != nil {
		return nil, err
	}
	return &man, nil
}

// SimulateStaleTree copies staging then injects an obsolete import into batcher.rs.
// Used by tests to prove verification rejects stale sources.
func SimulateStaleTree(cleanStaging, staleDir string) error {
	if err := os.RemoveAll(staleDir); err != nil {
		return err
	}
	if err := copyTreeFiltered(cleanStaging, staleDir, map[string]string{}, ""); err != nil {
		return err
	}
	target := filepath.Join(staleDir, "runtime", "src", "cuda_model", "batcher.rs")
	raw, err := os.ReadFile(target)
	if err != nil {
		return err
	}
	stale := []byte("use crate::model::batch::{BatchError, BatchEvent};\n" + string(raw))
	return os.WriteFile(target, stale, 0o644)
}

// RemoteInstallScript returns a bash snippet that extracts the uploaded tar into
// an isolated deploy directory, verifies the manifest, preserves model cache,
// points forge/forge-current at the new deploy, and removes old deploys.
func RemoteInstallScript(deployID, remoteTarPath string, keepDeploys int) string {
	if keepDeploys < 1 {
		keepDeploys = 2
	}
	// Intentionally LF-only; callers must not wrap with Windows CRLF.
	return fmt.Sprintf(`set -euo pipefail
BASE=%s
DEPLOYS=%s
CACHE=%s
ID=%q
TAR=%q
KEEP=%d
DEPLOY="$DEPLOYS/$ID"

echo "=== DEPLOY INSTALL id=$ID ==="
TAR_NAME=$(basename "$TAR")
# Lightning UploadFile syncs into $HOME (Phase 8A); this_studio may differ.
if [ ! -f "$TAR" ]; then
  for cand in \
    "$HOME/forge-upload/$TAR_NAME" \
    "$BASE/forge-upload/$TAR_NAME" \
    "$HOME/$TAR_NAME" \
    "$BASE/$TAR_NAME"; do
    if [ -f "$cand" ]; then
      TAR="$cand"
      break
    fi
  done
fi
if [ ! -f "$TAR" ]; then
  for _ in $(seq 1 30); do
    FOUND=$(find "$HOME" "$BASE" -maxdepth 4 -type f -name "$TAR_NAME" 2>/dev/null | head -n 1 || true)
    if [ -n "$FOUND" ] && [ -f "$FOUND" ]; then
      TAR="$FOUND"
      break
    fi
    sleep 2
  done
fi
echo "using upload tar: $TAR"
test -f "$TAR" || {
  echo "FAIL: missing upload tar (looked under HOME + $BASE for $TAR_NAME)"
  find "$HOME" "$BASE" -maxdepth 4 -type f -name '*.tar.gz' 2>/dev/null | head -n 20 || true
  exit 1
}
mkdir -p "$DEPLOYS" "$CACHE/models"
rm -rf "$DEPLOY"
mkdir -p "$DEPLOY"
tar -xzf "$TAR" -C "$DEPLOY"
rm -f "$TAR"

# Preserve any prior in-place forge/models into cache before retiring the tree.
if [ -d "$BASE/forge/models" ] && [ ! -L "$BASE/forge" ]; then
  if [ ! -d "$CACHE/models/Qwen2.5-0.5B-Instruct" ]; then
    mkdir -p "$CACHE/models"
    cp -a "$BASE/forge/models/." "$CACHE/models/" 2>/dev/null || true
  fi
fi

# Retire stale in-place forge tree so builds cannot pick it up.
if [ -e "$BASE/forge" ] && [ ! -L "$BASE/forge" ]; then
  STALE="$BASE/forge-legacy-stale-$(date +%%s)"
  echo "retiring in-place forge tree → $STALE"
  mv "$BASE/forge" "$STALE"
fi

ln -sfn "$CACHE/models" "$DEPLOY/models"
ln -sfn "$DEPLOY" "$BASE/forge-current"
ln -sfn "$DEPLOY" "$BASE/forge"

echo "active deploy → $(readlink -f "$BASE/forge" 2>/dev/null || readlink "$BASE/forge" || echo "$DEPLOY")"
cd "$DEPLOY"
bash scripts/verify_forge_deploy.sh
echo "DEPLOY_OK id=$ID"

# Prune old isolated deploys (never touch forge-cache).
cd "$DEPLOYS"
ls -1dt */ 2>/dev/null | tail -n +$((KEEP+1)) | while read -r d; do
  echo "pruning old deploy $d"
  rm -rf "$d"
done
`, RemoteStudioRoot, RemoteDeploysDir, RemoteCacheDir, deployID, remoteTarPath, keepDeploys)
}

func assertRequiredPresent(files map[string]string) error {
	var missing []string
	for _, req := range RequiredRelPaths {
		if _, ok := files[slashPath(req)]; !ok {
			missing = append(missing, req)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("deploy missing required files: %s", strings.Join(missing, ", "))
	}
	return nil
}

func assertNoObsoleteContent(root string) error {
	var hits []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() || !strings.HasSuffix(strings.ToLower(path), ".rs") {
			return nil
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(raw)
		rel, _ := filepath.Rel(root, path)
		for _, pat := range ObsoleteContentPatterns {
			if strings.Contains(text, pat) {
				hits = append(hits, fmt.Sprintf("%s contains %q", slashPath(rel), pat))
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	if len(hits) > 0 {
		return fmt.Errorf("obsolete/stale source markers present: %s", strings.Join(hits, "; "))
	}
	return nil
}

func copyTreeFiltered(src, dst string, files map[string]string, relPrefix string) error {
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		var fullRel string
		if relPrefix == "" {
			fullRel = slashPath(rel)
		} else {
			fullRel = slashPath(relPrefix + "/" + rel)
		}
		if info.IsDir() {
			base := info.Name()
			if skipDeployDir(base, fullRel) {
				return filepath.SkipDir
			}
			return os.MkdirAll(filepath.Join(dst, rel), 0o755)
		}
		if skipDeployRel(fullRel) {
			return nil
		}
		out := filepath.Join(dst, rel)
		if err := copyFile(path, out); err != nil {
			return err
		}
		// Hash staged bytes (may differ from source after .sh LF normalization).
		sum, err := hashFile(out)
		if err != nil {
			return err
		}
		files[fullRel] = sum
		return nil
	})
}

func skipDeployDir(base, fullRel string) bool {
	switch strings.ToLower(base) {
	case ".git", "target", "build", "build-cuda", "build-cpu", "cmake-build-debug",
		"cmake-build-release", "__pycache__", "node_modules", ".venv", "venv":
		return true
	}
	// Nested build dirs under kernels/
	lower := strings.ToLower(fullRel)
	if strings.Contains(lower, "/build-cuda") || strings.Contains(lower, "/build/") {
		return true
	}
	return false
}

func skipDeployRel(rel string) bool {
	rel = slashPath(rel)
	base := strings.ToLower(filepath.Base(rel))
	if base == ".env" || strings.HasSuffix(base, ".env") || strings.Contains(base, ".env.") {
		return true
	}
	// Local-only helpers used during a live session; not part of the product deploy.
	if strings.HasPrefix(base, "_patch_") || strings.HasPrefix(base, "_run_patch_") {
		return true
	}
	switch {
	case strings.HasSuffix(base, ".safetensors"),
		strings.HasSuffix(base, ".bin"),
		strings.HasSuffix(base, ".gguf"),
		strings.HasSuffix(base, ".forge"),
		strings.HasSuffix(base, ".pt"),
		strings.HasSuffix(base, ".onnx"),
		strings.HasSuffix(base, ".o"),
		strings.HasSuffix(base, ".a"),
		strings.HasSuffix(base, ".so"),
		strings.HasSuffix(base, ".dll"),
		strings.HasSuffix(base, ".dylib"),
		base == ".ds_store":
		return true
	}
	lower := strings.ToLower(rel)
	if strings.Contains(lower, "/target/") || strings.HasPrefix(lower, "target/") {
		return true
	}
	return false
}

func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	data, err := io.ReadAll(in)
	if err != nil {
		return err
	}
	// Windows checkouts often have CRLF; remote bash verify hashes the staged
	// bytes. Normalize shell scripts to LF in the deploy bundle so a later
	// `sed -i 's/\r$//'` on the Studio is a no-op and cannot break the manifest.
	if strings.HasSuffix(strings.ToLower(filepath.Base(src)), ".sh") {
		data = bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
		data = bytes.ReplaceAll(data, []byte("\r"), []byte("\n"))
	}
	mode := os.FileMode(0o644)
	if strings.HasSuffix(strings.ToLower(filepath.Base(src)), ".sh") {
		mode = 0o755
	}
	return os.WriteFile(dst, data, mode)
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func hashFileMap(files map[string]string) string {
	keys := make([]string, 0, len(files))
	for k := range files {
		if k == DeployManifestName || k == DeployIDFileName {
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	for _, k := range keys {
		fmt.Fprintf(h, "%s\t%s\n", k, files[k])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func writeGzipTar(stagingDir, tarPath string) error {
	f, err := os.Create(tarPath)
	if err != nil {
		return err
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	defer gz.Close()
	tw := tar.NewWriter(gz)
	defer tw.Close()

	return filepath.Walk(stagingDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(stagingDir, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		name := slashPath(rel)
		hdr, err := tar.FileInfoHeader(info, "")
		if err != nil {
			return err
		}
		hdr.Name = name
		hdr.Uid = 0
		hdr.Gid = 0
		hdr.Uname = ""
		hdr.Gname = ""
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(tw, in)
		in.Close()
		return copyErr
	})
}

func gitHead(repoRoot string) string {
	cmd := exec.Command("git", "-C", repoRoot, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func slashPath(p string) string {
	return strings.ReplaceAll(p, "\\", "/")
}

package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/forge-ml/forge/tools/forge-cloud/internal/cloud"
)

// cmdValidatePhase8B runs one billable Phase 8B GPU session.
//
// Recoverable deploy/build/test failures do NOT end the session early: the
// command keeps the Studio up, diagnoses, retries, and only stops when work
// succeeds, the time budget is exhausted, or an unrecoverable cloud/auth error
// occurs. Provider MaxRuntime on Start remains an independent shutdown safeguard.
func cmdValidatePhase8B(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("validate-phase8b", flag.ContinueOnError)
	confirm := fs.Bool("confirm", false, "required — billable session")
	gpu := fs.String("gpu", "l4", "machine alias")
	durStr := fs.String("duration", "30m", "planned max session (capped at MaxSession / 30m approval)")
	maxNew := fs.Int("max-new", 16, "max new tokens for GPU generate")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if !*confirm {
		return fmt.Errorf("refusing: pass --confirm (single approved billable session)")
	}

	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	if cfg.SpendLimitUSD > 2.0 {
		cfg.SpendLimitUSD = 2.0
	}
	dur, err := time.ParseDuration(*durStr)
	if err != nil {
		return err
	}
	if dur > cfg.MaxSession {
		dur = cfg.MaxSession
	}
	if dur > 30*time.Minute {
		dur = 30 * time.Minute
	}

	repoRoot := findRepoRootFromHere()
	fmt.Println("=== forge-cloud validate-phase8b (dev session; keep-alive on recoverable errors) ===")
	fmt.Println("local: building isolated deploy bundle…")
	workDir, err := os.MkdirTemp("", "forge-phase8b-deploy-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(workDir)

	bundle, err := cloud.BuildDeployBundle(repoRoot, workDir)
	if err != nil {
		return fmt.Errorf("build deploy bundle: %w", err)
	}
	if _, err := cloud.VerifyStagingAgainstManifest(bundle.StagingDir); err != nil {
		return fmt.Errorf("local deploy verification failed: %w", err)
	}
	fmt.Printf("  deploy_id=%s\n", bundle.Manifest.DeployID)
	fmt.Printf("  content_hash=%s\n", bundle.Manifest.ContentHash)
	if bundle.Manifest.GitRevision != "" {
		fmt.Printf("  git_revision=%s\n", bundle.Manifest.GitRevision)
	}
	fmt.Printf("  files=%d tar=%s\n", bundle.Manifest.FileCount, filepath.Base(bundle.TarPath))

	client := &cloud.Client{Cfg: cfg}
	fmt.Println("preflight auth…")
	auth := client.AuthCheck()
	if !auth.OK {
		return fmt.Errorf("auth failed: %s", auth.Message)
	}
	fmt.Println("  org:", auth.OrgName, "teamspace:", auth.TeamspaceName)

	rate, est, err := cloud.EstimateSession(*gpu, dur)
	if err != nil {
		return err
	}
	fmt.Println(" ", cloud.FormatEstimate(*gpu, dur, rate, est, cfg.SpendLimitUSD))
	if est > cfg.SpendLimitUSD {
		return fmt.Errorf("estimate $%.2f exceeds spend limit $%.2f", est, cfg.SpendLimitUSD)
	}
	if est > 2.0 {
		return fmt.Errorf("estimate $%.2f exceeds approved cap $2.00", est)
	}

	started := false
	defer func() {
		if !started {
			return
		}
		fmt.Println("cleanup: stop…")
		before, after, err := client.Stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "CRITICAL: stop failed: %v\n", err)
			fmt.Fprintln(os.Stderr, "Manual: Lightning UI → Studio forge-gpu → Stop")
			fmt.Fprintln(os.Stderr, "Or: go run ./api/cmd/forge-cloud stop")
			return
		}
		fmt.Printf("stop verified: before=%s after=%s\n", before, after)
	}()

	res, err := client.Start(cloud.StartOptions{
		GPUAlias: *gpu,
		Confirm:  true,
		Duration: dur,
	})
	if res != nil {
		started = true
	}
	if err != nil {
		return fmt.Errorf("unrecoverable start: %w", err)
	}
	fmt.Printf("started studio=%s status=%s machine=%s estimate=$%.2f\n",
		res.StudioName, res.Status, res.Machine, res.EstimateUSD)

	deadline := time.Now().Add(dur - 90*time.Second) // leave time for stop
	withinBudget := func() error {
		if time.Now().After(deadline) {
			return fmt.Errorf("session time budget exhausted (stopping for cost safety)")
		}
		return nil
	}

	remoteTarRel := "forge-upload/" + bundle.Manifest.DeployID + ".tar.gz"
	remoteTarAbs := cloud.RemoteStudioRoot + "/" + remoteTarRel

	// --- Deploy (recoverable): upload + locate + install with retries ---
	deployOK := false
	for attempt := 1; attempt <= 5 && !deployOK; attempt++ {
		if err := withinBudget(); err != nil {
			return err
		}
		fmt.Printf("=== DEPLOY attempt %d/5 ===\n", attempt)
		_, _ = client.Exec(`bash -lc 'mkdir -p "$HOME/forge-upload" /teamspace/studios/this_studio/forge-upload; echo HOME=$HOME; ls -la "$HOME/forge-upload" 2>/dev/null | head; ls -la /teamspace/studios/this_studio/forge-upload 2>/dev/null | head'`)

		fmt.Printf("uploading deploy tar → %s …\n", remoteTarRel)
		if err := client.UploadPath(bundle.TarPath, remoteTarRel); err != nil {
			fmt.Fprintf(os.Stderr, "UPLOAD recoverable error: %v\n", err)
			time.Sleep(5 * time.Second)
			continue
		}

		// Locate where Lightning placed the file (HOME vs this_studio).
		locateOut, _ := client.Exec(fmt.Sprintf(`bash -lc 'set +e
NAME=%q
echo "=== locate tar ==="
for d in "$HOME/forge-upload" /teamspace/studios/this_studio/forge-upload "$HOME" /teamspace/studios/this_studio; do
  if [ -f "$d/$NAME" ]; then echo FOUND:$d/$NAME; fi
done
find "$HOME" /teamspace/studios/this_studio -maxdepth 5 -type f -name "$NAME" 2>/dev/null | head -n 10
ls -la "$HOME/forge-upload" 2>/dev/null | head
'`, bundle.Manifest.DeployID+".tar.gz"))
		fmt.Print(locateOut)

		install := "bash -lc " + shellQuote(cloud.RemoteInstallScript(bundle.Manifest.DeployID, remoteTarAbs, 2))
		installOut, err := client.Exec(install)
		fmt.Print(installOut)
		if err == nil && strings.Contains(installOut, "DEPLOY_OK") && strings.Contains(installOut, "DEPLOY_VERIFY_OK") {
			deployOK = true
			fmt.Printf("remote deploy verified: id=%s content_hash=%s\n",
				bundle.Manifest.DeployID, bundle.Manifest.ContentHash)
			break
		}
		fmt.Fprintf(os.Stderr, "DEPLOY recoverable failure (attempt %d): %v\n", attempt, err)
		// Diagnostic listing — keep GPU up
		diag, _ := client.Exec(`bash -lc 'echo ===fs===; ls -la "$HOME" | head -40; echo ---; ls -la /teamspace/studios/this_studio 2>/dev/null | head -40; echo ---; find "$HOME" /teamspace/studios/this_studio -maxdepth 3 -name "*.tar.gz" 2>/dev/null | head'`)
		fmt.Print(diag)
		time.Sleep(8 * time.Second)
	}
	if !deployOK {
		return fmt.Errorf("deploy did not succeed within retry budget (GPU will stop via cleanup)")
	}

	if out, err := client.Exec("nvidia-smi"); err != nil {
		fmt.Print(out)
		fmt.Fprintf(os.Stderr, "nvidia-smi recoverable? %v — continuing\n", err)
	} else {
		fmt.Print(out)
	}

	// --- Build / infer (recoverable): retry remote stages until success or deadline ---
	phaseOK := false
	var lastOut string
	for attempt := 1; attempt <= 8 && !phaseOK; attempt++ {
		if err := withinBudget(); err != nil {
			_ = os.WriteFile(filepath.Join(repoRoot, "docs", "_phase8b_gpu_session.log"), []byte(lastOut), 0o644)
			return err
		}
		fmt.Printf("=== PHASE8B WORK attempt %d/8 ===\n", attempt)
		remote := fmt.Sprintf(strings.ReplaceAll(`bash -lc 'set -euo pipefail
export HOME="${HOME:-/home/$(whoami)}"
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
ROOT=/teamspace/studios/this_studio/forge-current
test -d "$ROOT" || ROOT=/teamspace/studios/this_studio/forge
test -d "$ROOT" || ROOT="$HOME/forge-current"
test -d "$ROOT" || ROOT="$HOME/forge"
test -f "$ROOT/.forge-deploy-id" || { echo "FAIL: missing deploy id at $ROOT"; ls -la "$ROOT" 2>/dev/null; exit 1; }
ACTIVE_ID=$(tr -d "[:space:]" <"$ROOT/.forge-deploy-id")
test "$ACTIVE_ID" = %q || { echo "FAIL: deploy id $ACTIVE_ID != %s"; exit 1; }
cd "$ROOT"
echo "BUILD_ROOT=$(pwd) deploy_id=$ACTIVE_ID"
nvidia-smi -L || true
if ! command -v rustc >/dev/null 2>&1 || ! command -v cargo >/dev/null 2>&1; then
  echo "=== installing rustup ==="
  curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  . "$HOME/.cargo/env"
fi
. "$HOME/.cargo/env" 2>/dev/null || true
sed -i "s/\r$//" scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true

export FORGE_MODEL_DIR="/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct"
if [ ! -d "$FORGE_MODEL_DIR" ] && [ -d "$HOME/forge-cache/models/Qwen2.5-0.5B-Instruct" ]; then
  FORGE_MODEL_DIR="$HOME/forge-cache/models/Qwen2.5-0.5B-Instruct"
fi
export FORGE_MAX_NEW_TOKENS=%d

bash scripts/remote_cuda_phase8b.sh

MODEL_DIR="$FORGE_MODEL_DIR"
if [ ! -f "$MODEL_DIR/model.safetensors" ] && [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "=== downloading Qwen2.5-0.5B-Instruct into forge-cache ==="
  mkdir -p "$MODEL_DIR"
  pip3 install -q huggingface_hub
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen2.5-0.5B-Instruct\", local_dir=\"$MODEL_DIR\")"
  cd "$ROOT/runtime"
  cargo run --release --features cuda --example generate_cuda_qwen -- "$MODEL_DIR" "$FORGE_MAX_NEW_TOKENS"
  cargo run --release --features cuda --example bench_cuda_qwen -- "$MODEL_DIR" "$FORGE_MAX_NEW_TOKENS"
fi
echo PHASE8B_WORK_OK
'`, "\r\n", "\n"), bundle.Manifest.DeployID, bundle.Manifest.DeployID, *maxNew)

		out, err := client.Exec(remote)
		lastOut = out
		fmt.Print(out)
		_ = os.WriteFile(filepath.Join(repoRoot, "docs", "_phase8b_gpu_session.log"), []byte(out), 0o644)
		if err == nil && strings.Contains(out, "PHASE8B_WORK_OK") {
			phaseOK = true
			break
		}
		fmt.Fprintf(os.Stderr, "PHASE8B recoverable failure (attempt %d): %v\n", attempt, err)
		// Keep GPU; print cargo/cmake hints from the log; brief pause then retry.
		// Known quick remediations that do not require a redesign:
		hint, _ := client.Exec(`bash -lc 'set +e
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
. "$HOME/.cargo/env" 2>/dev/null
ROOT=/teamspace/studios/this_studio/forge-current
[ -d "$ROOT" ] || ROOT=/teamspace/studios/this_studio/forge
[ -d "$ROOT" ] || ROOT=$HOME/forge
echo "hint-root=$ROOT"
ls -la "$ROOT/runtime/src/cuda_model" 2>/dev/null | head
# Ensure rust present for next attempt
command -v cargo || (curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable)
# Strip CRLF again
sed -i "s/\r$//" "$ROOT"/scripts/*.sh 2>/dev/null || true
'`)
		fmt.Print(hint)
		time.Sleep(10 * time.Second)
	}

	if !phaseOK {
		return fmt.Errorf("phase8b work did not complete within retry/time budget (see docs/_phase8b_gpu_session.log)")
	}
	fmt.Println("validate-phase8b work finished OK; defer will stop the Studio")
	return nil
}

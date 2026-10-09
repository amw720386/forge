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

// cmdValidatePhase12 runs one billable Phase 12 GPU session (benchmarking).
//
// Recoverable deploy/build/test failures do NOT end the session early.
// Modes: forge | refs | vllm | all  (FORGE_PHASE12_MODE / --mode).
func cmdValidatePhase12(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("validate-phase12", flag.ContinueOnError)
	confirm := fs.Bool("confirm", false, "required — billable session")
	gpu := fs.String("gpu", "l4", "machine alias (L4 only for Phase 12)")
	durStr := fs.String("duration", "40m", "planned max session (capped at 45m / MaxSession)")
	mode := fs.String("mode", "forge", "forge|refs|vllm|compare|all")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if !*confirm {
		return fmt.Errorf("refusing: pass --confirm (single approved billable session)")
	}
	if strings.ToLower(*gpu) != "l4" {
		return fmt.Errorf("Phase 12 authorization is L4 only (got %q)", *gpu)
	}

	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	if cfg.SpendLimitUSD > 2.0 {
		cfg.SpendLimitUSD = 2.0
	}
	// Phase 12 standing auth: ≤45m / session (may exceed default FORGE_CLOUD_MAX_SESSION).
	cfg.MaxSession = 45 * time.Minute
	dur, err := time.ParseDuration(*durStr)
	if err != nil {
		return err
	}
	if dur > 45*time.Minute {
		dur = 45 * time.Minute
	}

	repoRoot := findRepoRootFromHere()
	fmt.Println("=== forge-cloud validate-phase12 (dev session; keep-alive on recoverable errors) ===")
	fmt.Printf("mode=%s\n", *mode)
	fmt.Println("local: building isolated deploy bundle…")
	workDir, err := os.MkdirTemp("", "forge-phase12-deploy-*")
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
		return fmt.Errorf("estimate $%.2f exceeds approved Phase 12 cap $2.00/session", est)
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

	deadline := time.Now().Add(dur - 90*time.Second)
	withinBudget := func() error {
		if time.Now().After(deadline) {
			return fmt.Errorf("session time budget exhausted (stopping for cost safety)")
		}
		return nil
	}

	remoteTarRel := "forge-upload/" + bundle.Manifest.DeployID + ".tar.gz"
	remoteTarAbs := cloud.RemoteStudioRoot + "/" + remoteTarRel

	deployOK := false
	for attempt := 1; attempt <= 5 && !deployOK; attempt++ {
		if err := withinBudget(); err != nil {
			return err
		}
		fmt.Printf("=== DEPLOY attempt %d/5 ===\n", attempt)
		_, _ = client.Exec(`bash -lc 'mkdir -p "$HOME/forge-upload" /teamspace/studios/this_studio/forge-upload; echo HOME=$HOME'`)

		fmt.Printf("uploading deploy tar → %s …\n", remoteTarRel)
		if err := client.UploadPath(bundle.TarPath, remoteTarRel); err != nil {
			fmt.Fprintf(os.Stderr, "UPLOAD recoverable error: %v\n", err)
			time.Sleep(5 * time.Second)
			continue
		}

		locateOut, _ := client.Exec(fmt.Sprintf(`bash -lc 'set +e
NAME=%q
echo "=== locate tar ==="
for d in "$HOME/forge-upload" /teamspace/studios/this_studio/forge-upload "$HOME" /teamspace/studios/this_studio; do
  if [ -f "$d/$NAME" ]; then echo FOUND:$d/$NAME; fi
done
find "$HOME" /teamspace/studios/this_studio -maxdepth 5 -type f -name "$NAME" 2>/dev/null | head -n 10
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

	tryVLLM := "0"
	if *mode == "vllm" || *mode == "refs" || *mode == "compare" || *mode == "all" {
		tryVLLM = "1"
	}

	phaseOK := false
	var lastOut string
	for attempt := 1; attempt <= 8 && !phaseOK; attempt++ {
		if err := withinBudget(); err != nil {
			_ = os.WriteFile(filepath.Join(repoRoot, "docs", "_phase12_gpu_session.log"), []byte(lastOut), 0o644)
			return err
		}
		fmt.Printf("=== phase12 WORK attempt %d/8 ===\n", attempt)
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
sed -i "s/\r$//" scripts/*.sh benchmarks/forge/*.sh benchmarks/llama_cpp/*.sh benchmarks/vllm/*.sh 2>/dev/null || true
chmod +x scripts/*.sh benchmarks/forge/*.sh benchmarks/llama_cpp/*.sh benchmarks/vllm/*.sh 2>/dev/null || true

export FORGE_MODEL_DIR="/teamspace/studios/this_studio/forge-cache/models/Qwen2.5-0.5B-Instruct"
if [ ! -d "$FORGE_MODEL_DIR" ] && [ -d "$HOME/forge-cache/models/Qwen2.5-0.5B-Instruct" ]; then
  FORGE_MODEL_DIR="$HOME/forge-cache/models/Qwen2.5-0.5B-Instruct"
fi
export FORGE_BENCH_OUT="$ROOT/benchmarks/results"
export FORGE_PHASE12_MODE=%q
export FORGE_PHASE12_TRY_VLLM=%s
mkdir -p "$FORGE_BENCH_OUT"

bash scripts/remote_cuda_phase12.sh
echo phase12_WORK_OK
'`, "\r\n", "\n"), bundle.Manifest.DeployID, bundle.Manifest.DeployID, *mode, tryVLLM)

		out, err := client.Exec(remote)
		lastOut = out
		fmt.Print(out)
		_ = os.WriteFile(filepath.Join(repoRoot, "docs", "_phase12_gpu_session.log"), []byte(out), 0o644)

		// Pull compact results summaries back via printing already in log;
		// also try to cat key JSON files into the log for local capture.
		pull, _ := client.Exec(`bash -lc 'set +e
ROOT=/teamspace/studios/this_studio/forge-current
[ -d "$ROOT" ] || ROOT=/teamspace/studios/this_studio/forge
echo "=== RESULTS_DIR ==="
ls -la "$ROOT/benchmarks/results" 2>/dev/null | head -80
for f in forge_short_fp32.json forge_short_bf16.json bf16_correctness.json comparison.md env_meta.json vllm_blocker.json llama_cpp_short_f16.json forge_nsys_b8_kern.txt; do
  P="$ROOT/benchmarks/results/$f"
  if [ -f "$P" ]; then
    echo "=== FILE:$f ==="
    # Cap large files
    head -c 200000 "$P"
    echo
  fi
done
'`)
		fmt.Print(pull)
		lastOut += "\n" + pull
		_ = os.WriteFile(filepath.Join(repoRoot, "docs", "_phase12_gpu_session.log"), []byte(lastOut), 0o644)

		if err == nil && strings.Contains(out, "phase12_WORK_OK") {
			phaseOK = true
			break
		}
		fmt.Fprintf(os.Stderr, "phase12 recoverable failure (attempt %d): %v\n", attempt, err)
		hint, _ := client.Exec(`bash -lc 'set +e
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
. "$HOME/.cargo/env" 2>/dev/null
ROOT=/teamspace/studios/this_studio/forge-current
[ -d "$ROOT" ] || ROOT=/teamspace/studios/this_studio/forge
echo "hint-root=$ROOT"
command -v cargo || (curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable)
sed -i "s/\r$//" "$ROOT"/scripts/*.sh 2>/dev/null || true
'`)
		fmt.Print(hint)
		time.Sleep(10 * time.Second)
	}

	if !phaseOK {
		return fmt.Errorf("phase12 work did not complete within retry/time budget (see docs/_phase12_gpu_session.log)")
	}
	fmt.Println("validate-phase12 work finished OK; defer will stop the Studio")
	return nil
}

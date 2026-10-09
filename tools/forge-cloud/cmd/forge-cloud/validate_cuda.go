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

// cmdValidateCUDA runs the authorized Phase 8A GPU session:
// start → upload kernels → build/test/bench → stop (always via defer).
func cmdValidateCUDA(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("validate-cuda", flag.ContinueOnError)
	confirm := fs.Bool("confirm", false, "required — billable session")
	gpu := fs.String("gpu", "l4", "machine alias")
	durStr := fs.String("duration", "20m", "planned max session")
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
	dur, err := time.ParseDuration(*durStr)
	if err != nil {
		return err
	}
	if dur > 20*time.Minute {
		dur = 20 * time.Minute
	}

	client := &cloud.Client{Cfg: cfg}
	fmt.Println("=== forge-cloud validate-cuda (bounded) ===")
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
		up := strings.ToUpper(after)
		if strings.Contains(up, "READY") || strings.Contains(up, "RUNNING") {
			fmt.Fprintln(os.Stderr, "CRITICAL: studio may still be billable — verify in Lightning UI")
		}
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
		return fmt.Errorf("start: %w", err)
	}
	fmt.Printf("started studio=%s status=%s machine=%s\n", res.StudioName, res.Status, res.Machine)

	repoRoot := findRepoRootFromHere()
	kernelsLocal := filepath.Join(repoRoot, "kernels")
	fmt.Println("uploading kernels/ → forge/kernels (secrets excluded)…")
	if err := client.UploadPath(kernelsLocal, "forge/kernels"); err != nil {
		return fmt.Errorf("upload: %w", err)
	}

	fmt.Println("remote: nvidia-smi…")
	out, err := client.Exec("nvidia-smi")
	fmt.Print(out)
	if err != nil {
		return fmt.Errorf("nvidia-smi: %w", err)
	}

	fmt.Println("remote: CUDA build + tests + bench…")
	remote := `bash -lc 'set -euo pipefail
cd ~
if [ -d forge/kernels ]; then ROOT=$HOME/forge
elif [ -d kernels ]; then ROOT=$HOME
else echo "uploaded kernels not found"; find . -maxdepth 3 -type d -name kernels 2>/dev/null | head; exit 1
fi
cd "$ROOT"
pwd
nvidia-smi
command -v nvcc && nvcc --version || echo "nvcc missing — will try cmake CUDA detection"
cmake -S kernels -B kernels/build-cuda -DFORGE_BUILD_CUDA=ON -DFORGE_KERNELS_BUILD_TESTS=ON -DFORGE_CUDA_BUILD_TESTS=ON -DFORGE_CUDA_BUILD_BENCH=ON
cmake --build kernels/build-cuda -j"$(nproc 2>/dev/null || echo 4)"
echo "=== CUDA tests ==="
if [ -x kernels/build-cuda/cuda/forge_cuda_tests ]; then ./kernels/build-cuda/cuda/forge_cuda_tests
elif [ -x kernels/build-cuda/cuda/Release/forge_cuda_tests ]; then ./kernels/build-cuda/cuda/Release/forge_cuda_tests
else ctest --test-dir kernels/build-cuda --output-on-failure -R forge_cuda
fi
echo "=== CUDA bench ==="
if [ -x kernels/build-cuda/cuda/forge_cuda_bench ]; then ./kernels/build-cuda/cuda/forge_cuda_bench
elif [ -x kernels/build-cuda/cuda/Release/forge_cuda_bench ]; then ./kernels/build-cuda/cuda/Release/forge_cuda_bench
fi
'`
	out, err = client.Exec(remote)
	fmt.Print(out)
	if err != nil {
		return fmt.Errorf("remote build/test: %w", err)
	}
	fmt.Println("validate-cuda work finished; defer will stop the Studio")
	return nil
}

func findRepoRootFromHere() string {
	wd, _ := os.Getwd()
	dir := wd
	for i := 0; i < 8; i++ {
		if _, err := os.Stat(filepath.Join(dir, "kernels")); err == nil {
			if _, err := os.Stat(filepath.Join(dir, "api")); err == nil {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return wd
}

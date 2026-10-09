// forge-cloud: Lightning AI orchestration CLI (Phase 8A).
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/forge-ml/forge/api/internal/cloud"
)

func main() {
	// Global flags may appear before or after the subcommand:
	//   forge-cloud --env-file ../.env status
	//   forge-cloud status --env-file ../.env
	envFile, args := peelGlobalFlags(os.Args[1:])
	if len(args) < 1 {
		usage()
		os.Exit(2)
	}
	cmd := args[0]
	rest := args[1:]

	loadOpt := cloud.LoadOptions{EnvFile: envFile}

	switch cmd {
	case "fetch-phase16":
		if err := cmdFetchPhaseResults(loadOpt, "16"); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "fetch-phase15":
		if err := cmdFetchPhase15(loadOpt); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "fetch-phase14":
		if err := cmdFetchPhase14(loadOpt); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "fetch-phase13":
		if err := cmdFetchPhase13(loadOpt); err != nil {
			fmt.Fprintf(os.Stderr, "fetch failed: %v\n", err)
			os.Exit(1)
		}
	case "deploy-phase13", "deploy-phase14", "deploy-phase15", "deploy-phase16":
		if err := cmdDeployPhase13(loadOpt); err != nil {
			fmt.Fprintf(os.Stderr, "deploy failed: %v\n", err)
			os.Exit(1)
		}
	case "inspect-phase13", "inspect-phase14", "inspect-phase15", "inspect-phase16":
		if err := cmdInspectPhase13(loadOpt); err != nil {
			fmt.Fprintf(os.Stderr, "inspection failed: %v\n", err)
			os.Exit(1)
		}
	case "status":
		cmdStatus(loadOpt, rest)
	case "auth":
		if err := cmdAuth(loadOpt); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "machines":
		_ = ensureEnv(loadOpt) // optional; machines works without credentials
		cmdMachines()
	case "start":
		if err := cmdStart(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "exec":
		if err := cmdExec(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "benchmark":
		if err := cmdBenchmark(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "validate-cuda":
		if err := cmdValidateCUDA(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "validate-phase8b":
		if err := cmdValidatePhase8B(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "validate-phase9":
		if err := cmdValidatePhase9(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "validate-phase10":
		if err := cmdValidatePhase10(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "validate-phase11":
		if err := cmdValidatePhase11(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "validate-phase12":
		if err := cmdValidatePhase12(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "prepare-deploy":
		if err := cmdPrepareDeploy(rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "upload":
		if err := cmdUpload(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "stop":
		if err := cmdStop(loadOpt, rest); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", cmd)
		usage()
		os.Exit(2)
	}
}

func peelGlobalFlags(args []string) (envFile string, rest []string) {
	rest = make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--env-file" && i+1 < len(args):
			envFile = args[i+1]
			i++
		case strings.HasPrefix(a, "--env-file="):
			envFile = strings.TrimPrefix(a, "--env-file=")
		default:
			rest = append(rest, a)
		}
	}
	return envFile, rest
}

func ensureEnv(opt cloud.LoadOptions) cloud.Config {
	cfg, err := cloud.LoadConfigWith(opt)
	if err != nil && err != cloud.ErrMissingCredentials {
		fmt.Fprintf(os.Stderr, "config: %v\n", err)
	}
	return cfg
}

func mustConfig(opt cloud.LoadOptions) (cloud.Config, error) {
	return cloud.LoadConfigWith(opt)
}

func cmdStatus(opt cloud.LoadOptions, args []string) {
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	_ = fs.Parse(args)
	cfg, err := cloud.LoadConfigWith(opt)
	rep := cloud.StatusFrom(cfg, err)
	fmt.Println("Forge Lightning status")
	fmt.Println("  credentials:", rep.CredentialsConfigured)
	fmt.Println("  ", rep.RedactedSummary)
	fmt.Println("  sdk:", rep.SDKModule)
	fmt.Println("  state:", rep.StatePath)
	if cfg.EnvFileLoaded != "" {
		fmt.Println("  env_file:", cfg.EnvFileLoaded)
	} else {
		fmt.Println("  env_file: (none loaded)")
	}
	fmt.Println("  note:", rep.Note)
	if st, err := cloud.LoadState(); err == nil && st.StudioName != "" {
		fmt.Printf("  last_session: studio=%s machine=%s started=%s estimate=$%.2f\n",
			st.StudioName, st.Machine, st.StartedAt.Format(time.RFC3339), st.EstimateUSD)
	}
}

func cmdAuth(opt cloud.LoadOptions) error {
	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	client := &cloud.Client{Cfg: cfg}
	res := client.AuthCheck()
	fmt.Println("Forge Lightning auth check (non-billable)")
	fmt.Println("  ok:", res.OK)
	fmt.Println("  org:", res.OrgName)
	fmt.Println("  teamspace:", res.TeamspaceName)
	fmt.Println("  message:", res.Message)
	if len(res.VisibleOrgs) > 0 {
		fmt.Println("  visible_orgs:", strings.Join(res.VisibleOrgs, ", "))
	}
	if len(res.VisibleTeamspaces) > 0 {
		fmt.Println("  visible_teamspaces:", strings.Join(res.VisibleTeamspaces, ", "))
	}
	if !res.OK {
		return fmt.Errorf("auth check failed")
	}
	return nil
}

func cmdMachines() {
	fmt.Println("Supported --gpu aliases (Lightning Go SDK constants):")
	for _, line := range cloud.ListKnownMachines() {
		fmt.Println(" ", line)
	}
	fmt.Println()
	fmt.Println("Planning rate estimates (USD/h, override with FORGE_CLOUD_RATE_<ALIAS>_USD):")
	for _, a := range []string{"cpu", "t4", "l4", "l40s", "a10g", "a100", "h100"} {
		rate, est, err := cloud.EstimateSession(a, time.Hour)
		if err != nil {
			continue
		}
		fmt.Printf("  %-5s  $%.2f/h  (1h estimate $%.2f)\n", a, rate, est)
	}
	fmt.Println("These are estimates only — verify on lightning.ai before spending credits.")
}

func cmdStart(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("start", flag.ContinueOnError)
	gpu := fs.String("gpu", "", "machine alias (default l4)")
	confirm := fs.Bool("confirm", false, "required to actually provision (billable)")
	dryRun := fs.Bool("dry-run", false, "print cost estimate only; no API calls")
	durStr := fs.String("duration", "", "planned session (e.g. 20m); default FORGE_CLOUD_MAX_SESSION")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, err := cloud.LoadConfigWith(opt)
	if err != nil {
		if !*dryRun {
			return err
		}
		cfg = cloud.Config{
			DefaultGPU:       "l4",
			SpendLimitUSD:    3,
			MaxSession:       30 * time.Minute,
			StudioName:       "forge-gpu",
			RequireConfirm:   true,
			AutoSleepMinutes: 15,
		}
	}

	dur := cfg.MaxSession
	if *durStr != "" {
		d, err := time.ParseDuration(*durStr)
		if err != nil {
			return fmt.Errorf("bad --duration: %w", err)
		}
		dur = d
	}
	gpuAlias := *gpu
	if gpuAlias == "" {
		gpuAlias = cfg.DefaultGPU
	}

	client := &cloud.Client{Cfg: cfg}
	res, err := client.Start(cloud.StartOptions{
		GPUAlias: gpuAlias,
		Confirm:  *confirm,
		Duration: dur,
		DryRun:   *dryRun,
	})
	if err != nil {
		return err
	}
	fmt.Println("forge-cloud start")
	fmt.Println(" ", res.Note)
	fmt.Printf("  studio=%s status=%s machine=%s dry_run=%v\n",
		res.StudioName, res.Status, res.Machine, res.DryRun)
	fmt.Printf("  auto_sleep=%d min (provider-side idle timeout)\n", res.AutoSleepMin)
	if !res.DryRun {
		fmt.Println("  Next: forge-cloud exec --command 'nvidia-smi'")
		fmt.Println("  Then: forge-cloud stop   # always verify status after stop")
		fmt.Println("  If this process dies: Studio should auto-sleep; still run forge-cloud stop when you can.")
	}
	return nil
}

func cmdExec(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("exec", flag.ContinueOnError)
	command := fs.String("command", "", "remote shell command (Studio.Run)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*command) == "" {
		return fmt.Errorf("--command is required")
	}
	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	client := &cloud.Client{Cfg: cfg}
	out, err := client.Exec(*command)
	if err != nil {
		return err
	}
	fmt.Print(out)
	if out != "" && !strings.HasSuffix(out, "\n") {
		fmt.Println()
	}
	return nil
}

func cmdBenchmark(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	confirm := fs.Bool("confirm", false, "required — starts GPU if needed, runs remote CUDA bench, then stops")
	gpu := fs.String("gpu", "l4", "machine alias")
	dryRun := fs.Bool("dry-run", false, "estimate only")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := cloud.LoadConfigWith(opt)
	if err != nil && !*dryRun {
		return err
	}
	if *dryRun {
		rate, est, err := cloud.EstimateSession(*gpu, 20*time.Minute)
		if err != nil {
			return err
		}
		limit := 3.0
		if cfg.SpendLimitUSD > 0 {
			limit = cfg.SpendLimitUSD
		}
		fmt.Println("forge-cloud benchmark --dry-run")
		fmt.Println(" ", cloud.FormatEstimate(*gpu, 20*time.Minute, rate, est, limit))
		fmt.Println("  Would: start → git pull + cmake CUDA build + forge_cuda_tests/bench → stop")
		return nil
	}
	if !*confirm {
		return fmt.Errorf("refusing benchmark: pass --confirm (billable end-to-end workflow)")
	}

	client := &cloud.Client{Cfg: cfg}
	if _, err := client.Start(cloud.StartOptions{
		GPUAlias: *gpu,
		Confirm:  true,
		Duration: 20 * time.Minute,
	}); err != nil {
		return fmt.Errorf("start: %w", err)
	}

	defer func() {
		fmt.Println("benchmark cleanup: stopping Studio…")
		before, after, err := client.Stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "WARNING: stop failed: %v\n", err)
			fmt.Fprintln(os.Stderr, "Recovery: forge-cloud stop   or stop the Studio in the Lightning UI")
			return
		}
		fmt.Printf("stopped: before=%s after=%s\n", before, after)
	}()

	remote := cfg.RemoteWorkDir
	if remote == "" {
		remote = "~/forge"
	}
	script := strings.Join([]string{
		"set -e",
		"cd " + remote + " || cd /teamspace/studios/*/forge || true",
		"if [ -d .git ]; then git pull --ff-only; fi",
		"nvidia-smi",
		"cmake -S kernels -B kernels/build-cuda -DFORGE_BUILD_CUDA=ON -DFORGE_CUDA_BUILD_TESTS=ON -DFORGE_CUDA_BUILD_BENCH=ON",
		"cmake --build kernels/build-cuda -j",
		"ctest --test-dir kernels/build-cuda --output-on-failure || ./kernels/build-cuda/cuda/forge_cuda_tests",
		"./kernels/build-cuda/cuda/forge_cuda_bench || true",
	}, " && ")

	out, err := client.Exec("bash -lc " + shellQuote(script))
	fmt.Print(out)
	return err
}

func cmdStop(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("stop", flag.ContinueOnError)
	_ = fs.Parse(args)
	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	client := &cloud.Client{Cfg: cfg}
	before, after, err := client.Stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Recovery if stop failed:")
		fmt.Fprintln(os.Stderr, "  1. forge-cloud stop  (retry)")
		fmt.Fprintln(os.Stderr, "  2. Lightning UI → Studio → Stop")
		fmt.Fprintln(os.Stderr, "  3. Rely on auto-sleep if configured (does not replace an explicit stop)")
		return err
	}
	fmt.Printf("forge-cloud stop: before=%s after=%s\n", before, after)
	if strings.Contains(strings.ToLower(after), "run") {
		fmt.Fprintln(os.Stderr, "WARNING: studio may still be running — verify in Lightning UI")
		os.Exit(1)
	}
	return nil
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func usage() {
	fmt.Println(`Usage: forge-cloud [--env-file path] <command> [flags]

Global:
  --env-file path   Load KEY=VAL from file (does not override process env).
                    Default: walk up from cwd/exe to repo-root .env

Commands:
  status       Credential + session inspection (non-billable)
  auth         Resolve org/teamspace via SDK (non-billable network)
  machines     List GPU aliases + estimated USD/h (non-billable)
  start        Start Studio (billable) — requires --confirm
               --gpu L4|T4|...  --duration 20m  --dry-run
  exec         Run remote command via Studio.Run
               --command "nvidia-smi"
  benchmark      start → remote CUDA build/test/bench → stop (requires --confirm)
  validate-cuda     One-shot Phase 8A GPU validation (requires --confirm); always stops
  validate-phase8b  One-shot Phase 8B full-model CUDA validation (requires --confirm); always stops
  validate-phase9   One-shot Phase 9 GPU continuous-batching validation (requires --confirm); always stops
  validate-phase10  One-shot Phase 10 paged KV + chunked prefill validation (requires --confirm); always stops
  validate-phase11  One-shot Phase 11 GPU perf (attn/BF16/argmax) validation (requires --confirm); always stops
  validate-phase12  One-shot Phase 12 competitive benchmarking session (requires --confirm); always stops
                    --mode forge|refs|vllm|all  (default forge)
  inspect-phase14  Non-billable authenticated Studio/catalog inspection
  deploy-phase14   Content-hashed source synchronization to running Studio
  fetch-phase14    Retrieve Phase 14 evidence from running Studio
  inspect-phase15  Non-billable authenticated Studio/catalog inspection
  deploy-phase15   Content-hashed source synchronization to running Studio
  fetch-phase15    Retrieve Phase 15 evidence from running Studio
  inspect-phase16  Non-billable authenticated Studio/catalog inspection
  deploy-phase16   Content-hashed source synchronization to running Studio
  fetch-phase16    Retrieve Phase 16 evidence from running Studio
  prepare-deploy    Non-billable: build isolated deploy tar + verify content-hash manifest
  stop              Stop Studio and verify status

Auth / config (.env or environment):
  LIGHTNING_USER_ID, LIGHTNING_API_KEY
  FORGE_CLOUD_ORG, FORGE_CLOUD_TEAMSPACE
  FORGE_CLOUD_SPEND_LIMIT_USD=3
  FORGE_CLOUD_MAX_SESSION=30m
  FORGE_CLOUD_AUTO_SLEEP_MIN=15
  FORGE_CLOUD_MACHINE=l4

See docs/LIGHTNING.md`)
}

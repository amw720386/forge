package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/forge-ml/forge/tools/forge-cloud/internal/cloud"
)

// cmdPrepareDeploy builds and verifies an isolated deploy bundle locally.
// Non-billable — never starts Lightning.
func cmdPrepareDeploy(args []string) error {
	fs := flag.NewFlagSet("prepare-deploy", flag.ContinueOnError)
	outDir := fs.String("out", "", "optional directory to keep staging + tar (default: temp, deleted)")
	keep := fs.Bool("keep", false, "keep staging artifacts (implies temp dir under cwd if --out unset)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	repoRoot := findRepoRootFromHere()
	work := *outDir
	if work == "" && *keep {
		work = filepath.Join(repoRoot, ".forge-deploy-out")
		_ = os.RemoveAll(work)
		if err := os.MkdirAll(work, 0o755); err != nil {
			return err
		}
	}
	cleanup := work == "" && !*keep
	if work == "" {
		var err error
		work, err = os.MkdirTemp("", "forge-prepare-deploy-*")
		if err != nil {
			return err
		}
	}
	if cleanup {
		defer os.RemoveAll(work)
	}

	fmt.Println("=== forge-cloud prepare-deploy (non-billable) ===")
	fmt.Println("repo:", repoRoot)
	bundle, err := cloud.BuildDeployBundle(repoRoot, work)
	if err != nil {
		return err
	}
	man, err := cloud.VerifyStagingAgainstManifest(bundle.StagingDir)
	if err != nil {
		return fmt.Errorf("verification failed: %w", err)
	}

	fmt.Println("deploy_id:    ", man.DeployID)
	fmt.Println("content_hash: ", man.ContentHash)
	if man.GitRevision != "" {
		fmt.Println("git_revision: ", man.GitRevision)
	}
	fmt.Println("file_count:   ", man.FileCount)
	fmt.Println("staging:      ", bundle.StagingDir)
	fmt.Println("tar:          ", bundle.TarPath)
	fmt.Println("DEPLOY_PREPARE_OK")
	return nil
}

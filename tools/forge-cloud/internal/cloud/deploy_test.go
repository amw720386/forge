package cloud

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildDeployBundleAndVerify(t *testing.T) {
	repo := findTestRepoRoot(t)
	work := t.TempDir()

	bundle, err := BuildDeployBundle(repo, work)
	if err != nil {
		t.Fatalf("BuildDeployBundle: %v", err)
	}
	if bundle.Manifest.DeployID == "" || bundle.Manifest.ContentHash == "" {
		t.Fatalf("empty deploy identity: %+v", bundle.Manifest)
	}
	if bundle.Manifest.FileCount < 20 {
		t.Fatalf("suspiciously few files: %d", bundle.Manifest.FileCount)
	}
	if _, err := os.Stat(bundle.TarPath); err != nil {
		t.Fatalf("missing tar: %v", err)
	}

	man, err := VerifyStagingAgainstManifest(bundle.StagingDir)
	if err != nil {
		t.Fatalf("VerifyStagingAgainstManifest: %v", err)
	}
	if man.ContentHash != bundle.Manifest.ContentHash {
		t.Fatalf("hash drift: %s vs %s", man.ContentHash, bundle.Manifest.ContentHash)
	}
	if !strings.Contains(bundle.Manifest.DeployID, man.ContentHash[:12]) {
		t.Fatalf("deploy id should embed content hash prefix: %s", bundle.Manifest.DeployID)
	}
}

func TestDeployExcludesSecretsAndArtifacts(t *testing.T) {
	repo := findTestRepoRoot(t)
	work := t.TempDir()
	bundle, err := BuildDeployBundle(repo, work)
	if err != nil {
		t.Fatalf("BuildDeployBundle: %v", err)
	}
	for rel := range bundle.Manifest.Files {
		lower := strings.ToLower(rel)
		switch {
		case strings.Contains(lower, ".env"),
			strings.Contains(lower, "/target/"),
			strings.HasSuffix(lower, ".safetensors"),
			strings.HasSuffix(lower, ".forge"),
			strings.Contains(lower, "build-cuda"):
			t.Fatalf("forbidden path leaked into deploy: %s", rel)
		}
	}
	if _, ok := bundle.Manifest.Files["scripts/verify_forge_deploy.sh"]; !ok {
		t.Fatal("verify_forge_deploy.sh missing from manifest")
	}
}

func TestStaleRemoteSourcesFailVerification(t *testing.T) {
	repo := findTestRepoRoot(t)
	work := t.TempDir()
	bundle, err := BuildDeployBundle(repo, work)
	if err != nil {
		t.Fatalf("BuildDeployBundle: %v", err)
	}

	stale := filepath.Join(work, "stale-tree")
	if err := SimulateStaleTree(bundle.StagingDir, stale); err != nil {
		t.Fatalf("SimulateStaleTree: %v", err)
	}

	_, err = VerifyStagingAgainstManifest(stale)
	if err == nil {
		t.Fatal("expected stale tree verification to fail")
	}
	msg := err.Error()
	// Must fail on hash and/or obsolete marker — never silently accept.
	if !strings.Contains(msg, "hash:") &&
		!strings.Contains(msg, "content_hash") &&
		!strings.Contains(msg, "obsolete") {
		t.Fatalf("unexpected verify error (want hash/obsolete): %v", err)
	}
}

func TestObsoleteImportRejectedEvenIfManifestRewritten(t *testing.T) {
	repo := findTestRepoRoot(t)
	work := t.TempDir()
	bundle, err := BuildDeployBundle(repo, work)
	if err != nil {
		t.Fatalf("BuildDeployBundle: %v", err)
	}
	stale := filepath.Join(work, "stale-rewritten")
	if err := SimulateStaleTree(bundle.StagingDir, stale); err != nil {
		t.Fatalf("SimulateStaleTree: %v", err)
	}
	// Attacker/stale sync rewrites manifest to match bad sources — obsolete gate must still trip.
	badBundle, err := rebuildManifestInPlace(stale)
	if err != nil {
		t.Fatalf("rebuildManifestInPlace: %v", err)
	}
	_ = badBundle
	err = assertNoObsoleteContent(stale)
	if err == nil {
		t.Fatal("expected obsolete content detection")
	}
	if !strings.Contains(err.Error(), "use crate::model::batch::") {
		t.Fatalf("expected obsolete import in error, got: %v", err)
	}
}

func TestRemoteInstallScriptIsLFAndIsolated(t *testing.T) {
	script := RemoteInstallScript("testdeploy-abc", "/tmp/x.tar.gz", 2)
	if strings.Contains(script, "\r") {
		t.Fatal("install script must be LF-only")
	}
	for _, needle := range []string{
		RemoteDeploysDir,
		RemoteCacheDir,
		"forge-deploys",
		"verify_forge_deploy.sh",
		"DEPLOY_OK",
		"forge-legacy-stale",
		"$HOME/forge-upload/",
		"missing upload tar",
	} {
		if !strings.Contains(script, needle) {
			t.Fatalf("install script missing %q", needle)
		}
	}
	// Must not build from a mutable in-place overwrite of an old tree as the only strategy.
	if strings.Contains(script, "rm -rf \"$R/runtime/src\"") {
		t.Fatal("should not rely on in-place rm of runtime/src")
	}
}

func TestHashFileMapStable(t *testing.T) {
	a := map[string]string{"b.rs": "bb", "a.rs": "aa"}
	b := map[string]string{"a.rs": "aa", "b.rs": "bb"}
	if hashFileMap(a) != hashFileMap(b) {
		t.Fatal("hashFileMap must be order-independent")
	}
}

func rebuildManifestInPlace(staging string) (*DeployManifest, error) {
	files := map[string]string{}
	err := filepath.Walk(staging, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(staging, path)
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
		files[rel] = sum
		return nil
	})
	if err != nil {
		return nil, err
	}
	man := DeployManifest{
		DeployID:    "forged-stale",
		ContentHash: hashFileMap(files),
		Files:       files,
		FileCount:   len(files),
		Required:    RequiredRelPaths,
	}
	raw, err := json.MarshalIndent(man, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(staging, DeployManifestName), raw, 0o644); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(staging, DeployIDFileName), []byte(man.DeployID+"\n"), 0o644); err != nil {
		return nil, err
	}
	return &man, nil
}

func findTestRepoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := wd
	for i := 0; i < 8; i++ {
		if pathExists(filepath.Join(dir, "kernels")) && pathExists(filepath.Join(dir, "runtime", "src", "cuda_model", "batcher.rs")) {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("could not find repo root from", wd)
	return ""
}

func pathExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

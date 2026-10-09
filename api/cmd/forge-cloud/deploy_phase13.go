package main

import (
	"encoding/base64"
	"fmt"
	"github.com/forge-ml/forge/api/internal/cloud"
	"os"
)

// Synchronize a verified source bundle over the existing exec channel. The
// artifact upload API in the current Go SDK does not mount files in the Studio.
func cmdDeployPhase13(opt cloud.LoadOptions) error {
	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	work, err := os.MkdirTemp("", "forge-phase13-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	bundle, err := cloud.BuildDeployBundle(findRepoRootFromHere(), work)
	if err != nil {
		return err
	}
	if _, err = cloud.VerifyStagingAgainstManifest(bundle.StagingDir); err != nil {
		return err
	}
	data, err := os.ReadFile(bundle.TarPath)
	if err != nil {
		return err
	}
	c := &cloud.Client{Cfg: cfg}
	path := cloud.RemoteStudioRoot + "/forge-upload/" + bundle.Manifest.DeployID + ".tar.gz"
	if _, err = c.Exec("mkdir -p " + cloud.RemoteStudioRoot + "/forge-upload; : > " + path); err != nil {
		return err
	}
	for offset := 0; offset < len(data); offset += 24000 {
		end := offset + 24000
		if end > len(data) {
			end = len(data)
		}
		encoded := base64.StdEncoding.EncodeToString(data[offset:end])
		if _, err = c.Exec("printf %s " + encoded + " | base64 -d >> " + path); err != nil {
			return err
		}
	}
	out, err := c.Exec("bash -lc " + shellQuote(cloud.RemoteInstallScript(bundle.Manifest.DeployID, path, 2)))
	fmt.Print(out)
	fmt.Printf("deploy_id=%s content_hash=%s\n", bundle.Manifest.DeployID, bundle.Manifest.ContentHash)
	return err
}

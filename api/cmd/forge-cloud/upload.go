package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/forge-ml/forge/api/internal/cloud"
)

func cmdUpload(opt cloud.LoadOptions, args []string) error {
	fs := flag.NewFlagSet("upload", flag.ContinueOnError)
	local := fs.String("local", "", "local file or directory")
	remote := fs.String("remote", "", "remote path inside Studio")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *local == "" || *remote == "" {
		return fmt.Errorf("--local and --remote are required")
	}
	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	client := &cloud.Client{Cfg: cfg}
	abs, err := filepath.Abs(*local)
	if err != nil {
		return err
	}
	if _, err := os.Stat(abs); err != nil {
		return err
	}
	fmt.Printf("uploading %s → %s …\n", abs, *remote)
	if err := client.UploadPath(abs, *remote); err != nil {
		return err
	}
	fmt.Println("UPLOAD_OK")
	return nil
}

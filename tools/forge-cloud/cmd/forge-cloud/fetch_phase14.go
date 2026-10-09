package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"fmt"
	"github.com/forge-ml/forge/tools/forge-cloud/internal/cloud"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func cmdFetchPhase14(opt cloud.LoadOptions) error {
	cfg, err := mustConfig(opt)
	if err != nil {
		return err
	}
	c := &cloud.Client{Cfg: cfg}
	script := `import base64,io,tarfile,pathlib
root=pathlib.Path('/teamspace/studios/this_studio/forge-cache/phase14-results')
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 for p in sorted(root.iterdir()):
  if p.is_file() and p.suffix in ('.json','.csv','.log','.txt','.nsys-rep','.jsonl'):
   tar.add(p,arcname=p.name)
print('PHASE14_DATA:'+base64.b64encode(buf.getvalue()).decode())`
	out, err := c.Exec("python3 -c " + shellQuote(script))
	if err != nil {
		return err
	}
	_, encoded, ok := strings.Cut(out, "PHASE14_DATA:")
	if !ok {
		return fmt.Errorf("missing result payload")
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(encoded))
	if err != nil {
		return err
	}
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return err
	}
	defer gz.Close()
	dir := filepath.Join(findRepoRootFromHere(), "benchmarks", "results", "phase14")
	if err = os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	count := 0
	for {
		h, e := tr.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			return e
		}
		if h.Typeflag != tar.TypeReg || filepath.Base(h.Name) != h.Name || h.Size > 32*1024*1024 {
			return fmt.Errorf("invalid result archive member")
		}
		file, e := os.Create(filepath.Join(dir, h.Name))
		if e != nil {
			return e
		}
		_, e = io.Copy(file, tr)
		file.Close()
		if e != nil {
			return e
		}
		count++
	}
	fmt.Printf("Fetched %d evidence files into %s\n", count, dir)
	return nil
}

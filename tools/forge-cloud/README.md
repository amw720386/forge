# Historical cloud validation tools

Optional Lightning tooling, isolated from the inference server's dependencies.
Local tests do not provision hardware:

```sh
go -C tools/forge-cloud test -count=1 ./...
```

The CLI is `go -C tools/forge-cloud run ./cmd/forge-cloud`. It retains historical
session commands and cost controls; old budget authorizations in its source are
historical context, not current permission to spend credits. No cloud operation
was run during cleanup. The root `.env` remains ignored and untouched; the blank
example is here for reference.

`scripts/` preserves original remote experiments, watchdogs and patch utilities.
These are historical artifacts, not ordinary build commands. Their original
repository paths were `scripts/<name>`; local one-off scripts may require restoring
that layout and old environment settings. The deployment bundle builder restores
remote script paths and historical report paths inside its verified staging tree.
Use [the current build guide](../../docs/build.md) for local CPU/CUDA builds.

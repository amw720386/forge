# Phase 8B build hardening — non-GPU verification log

**Date:** 2026-10-08 (sync fix pass)  
**Lightning:** not used for this hardening pass (no credits spent).

## Goals

Prevent paid GPU sessions from dying on preventable compile/import/CRLF/**stale upload** issues.

## Deployment strategy (source sync)

Lightning `UploadFolder` into a reused `forge/` tree can leave **stale files** (Phase 8B attempt 3).
`forge-cloud validate-phase8b` no longer overwrites an in-place source tree.

1. **Local:** `BuildDeployBundle` copies allowed paths into an isolated staging dir (excludes `.env`, weights, `target/`, `build-cuda/`).
2. **Manifest:** writes `.forge-deploy-manifest.json` with per-file SHA-256 + `content_hash` (+ optional git HEAD).
3. **Pack:** single `.tar.gz` uploaded to `forge-upload/<deploy_id>.tar.gz`.
4. **Remote install:** extract into **fresh** `forge-deploys/<deploy_id>/` (never merge into old sources).
5. **Verify:** `scripts/verify_forge_deploy.sh` checks id, required files, obsolete import markers, and content hashes — fail closed.
6. **Activate:** symlink `forge` + `forge-current` → new deploy; models live in durable `forge-cache/models`.
7. **Prune:** keep last 2 deploy dirs; never delete `forge-cache`.

Non-billable local check:

```powershell
go run ./api/cmd/forge-cloud prepare-deploy
```

## Changes (this pass)

- `api/internal/cloud/deploy.go` — isolated bundle + manifest + install script
- `api/internal/cloud/deploy_test.go` — stale-tree / obsolete-import / exclusion tests
- `scripts/verify_forge_deploy.sh` — remote source-identity gate
- `validate-phase8b` — tar deploy + verify before compile
- `prepare-deploy` — non-billable local bundle check
- Prior: `cuda-stub`, fail-fast `--features cuda`, public re-exports, LF scripts

## Non-GPU checks (Windows host, 2026-10-08 sync fix)

| Check | Result |
|-------|--------|
| Go `./internal/cloud` tests (incl. stale deploy) | **PASS** |
| `go test ./...` (api module) | **PASS** |
| `forge-cloud prepare-deploy` | **PASS** (`DEPLOY_PREPARE_OK`) |
| `cargo build --release` (default) | **PASS** |
| `cargo test --release --lib` | **PASS** (17) |
| `cargo build --release --features cuda-stub` | **PASS** (links stub) |
| `cargo test --features cuda-stub --test cuda_compile_check` | **PASS** (2) |
| `cargo test --features cuda-stub --test cuda_transformer_parity` | **PASS** (2 skipped GPU; marked unverified) |
| `cargo build --features cuda` without `libforge_cuda` | **PASS** (fails fast with clear panic) |
| Script LF (`*.sh` incl. `verify_forge_deploy.sh`) | **PASS** (`has_cr=False`) |
| Real nvcc / `libforge_cuda` link | **SKIP** (no CUDA toolkit on this Windows host) |
| Lightning start / GPU session | **NOT RUN** (no authorization) |

## GPU session (2026-10-08) — verified

See [PHASE8B_RESULTS.md](PHASE8B_RESULTS.md): L4 link + Qwen greedy CUDA==CPU + benches; Studio stopped.

Linking notes discovered on device:
- `CUDA_SEPARABLE_COMPILATION` must be **OFF** for static `libforge_cuda` + cargo
- Linux: `cargo:rustc-link-arg=-fuse-ld=bfd`

# Lightning AI integration (Phase 8A)

## Phase 16 bounded validation

Phase 16 authorized $2 total new estimated compute, $1/session, 30 minutes/session,
one existing `forge-gpu` L4. Prior phase budgets were not carried forward.
One session used a verified on-demand $1.68/hour rate and independent absolute-deadline
local/remote watchdogs. Conservative cost is $0.70; actual invoice unavailable;
remaining estimated budget $1.30. Explicit stop and independent provider state checks
confirm compute is down. See [PHASE16_COST_LEDGER.md](PHASE16_COST_LEDGER.md).

`inspect-phase16`, `deploy-phase16` and `fetch-phase16` are supported CLI aliases.
Use `FORGE_CLOUD_RATE_L4_USD=1.68`, `FORGE_CLOUD_SPEND_LIMIT_USD=1` and
`FORGE_CLOUD_MAX_SESSION=30m` only after checking the current provider catalog and
available credits. A fresh run must reserve budget in the Phase 16 ledger before
starting, verify the existing Studio, deploy source and arm an independent local
shutdown deadline. Verify the remote watchdog after startup. Provider MaxRuntime
alone was not treated as enforceable shutdown protection. Do not provision if these
controls cannot be enforced. Preserve cached models and toolchains.

`scripts/phase16_validate.sh` builds and verifies a fresh source deployment, regenerates
Phase 2 fixtures, runs Qwen regressions, exports the independent SmolLM2 reference and
runs focused second-model tests plus the unchanged Phase 15 HTTP harness. Generated
build/test artifacts change a deployment's contents; initial manifest verification
is a pre-build check. Incremental repairs require their own source hash audit.

Rent NVIDIA GPUs for CUDA builds/tests **without** using Lightning’s browser editor as the primary workflow.

## Credentials & `.env`

Repository root (recommended):

```text
.env            # secrets — gitignored, never commit
.env.example    # placeholders only
```

Install dependency: `github.com/joho/godotenv`.

### Precedence

1. Process environment variables (already set)
2. Selected `.env` file (`godotenv` does **not** override existing env)
3. Safe defaults for non-secrets (`gpu=l4`, spend `$3`, max session `30m`, auto-sleep `15`)

### Path resolution

1. `--env-file path` (or `FORGE_ENV_FILE`)
2. Else walk upward from executable dir and cwd for a monorepo root (has `kernels/` + `api|runtime`) containing `.env`

```powershell
go run ./api/cmd/forge-cloud status
go run ./api/cmd/forge-cloud --env-file .\.env auth
```

### Required variables

```bash
LIGHTNING_USER_ID=
LIGHTNING_API_KEY=
FORGE_CLOUD_ORG=          # SDK org slug, e.g. amw720386-org (not display "… Org")
FORGE_CLOUD_TEAMSPACE=    # e.g. general
FORGE_CLOUD_SPEND_LIMIT_USD=3
```

Optional: `FORGE_CLOUD_MACHINE`, `FORGE_CLOUD_MAX_SESSION`, `FORGE_CLOUD_AUTO_SLEEP_MIN`, `FORGE_CLOUD_STUDIO`, `FORGE_CLOUD_REMOTE_DIR`.

### Security

- `.env` is gitignored; never commit keys
- Status prints `LIGHTNING_API_KEY=set(len=N)` only
- Env parse errors **never** include file body (secrets)
- `UploadPath` refuses `*.env`
- Do not pass secrets on CLI argv or in remote shell strings

## Verified SDK usage

Source: [Lightning-AI/sdk Go](https://github.com/Lightning-AI/sdk/tree/main/go)

| Capability | API |
|---|---|
| Auth | `LIGHTNING_USER_ID` / `LIGHTNING_API_KEY` |
| Org / teamspace | `GetOrganization`, `GetTeamspace`, `User.Organizations` / `Teamspaces` |
| Studio | `CreateStudio`, `GetStudio`, `Start`, `Stop`, `Run`, `UploadFolder` |
| Machines | `lit.MachineL4`, `MachineT4`, … |
| Idle timeout | `SetAutoSleep` / `SetAutoSleepTime`, `StartStudioOptions.MaxRuntime` |

**Important:** `GetStudio().Status()` is often `CLOUD_SPACE_STATE_READY` (cloudspace exists). Instance liveness is determined by successful `Studio.Run` / `PublicIP` after `Start()`, not by that string alone.

## CLI

```powershell
go run ./api/cmd/forge-cloud status
go run ./api/cmd/forge-cloud auth
go run ./api/cmd/forge-cloud machines
go run ./api/cmd/forge-cloud prepare-deploy
go run ./api/cmd/forge-cloud start --gpu l4 --duration 20m --dry-run
go run ./api/cmd/forge-cloud validate-cuda --gpu l4 --duration 20m --confirm
go run ./api/cmd/forge-cloud validate-phase8b --gpu l4 --duration 20m --confirm
go run ./api/cmd/forge-cloud validate-phase9 --gpu l4 --duration 30m --confirm
go run ./api/cmd/forge-cloud validate-phase10 --gpu l4 --duration 30m --confirm
go run ./api/cmd/forge-cloud validate-phase11 --gpu l4 --duration 35m --confirm
go run ./api/cmd/forge-cloud stop
```

### Phase 10 standing authorization

Phase 10 has standing approval for bounded L4 sessions: **$8 total**, **$3 / session**, **≤45 min**, reuse Studio `forge-gpu`. Ledger: [PHASE10_RESULTS.md](PHASE10_RESULTS.md). Always verify stop (`running→stopped`).

### Phase 11 standing authorization

Phase 11: **$6 total**, **$2 / session**, **≤45 min**, reuse Studio `forge-gpu` (L4 only). Ledger: [PHASE11_RESULTS.md](PHASE11_RESULTS.md). Do not roll unused Phase 10 budget into Phase 11.

### Phase 12 standing authorization

Phase 12 competitive benchmarking: **$6 total**, **$2 / session**, **≤45 min**, reuse Studio `forge-gpu` (L4 only). Ledger: [PHASE12_COST_LEDGER.md](PHASE12_COST_LEDGER.md). Do not roll unused prior-phase budget into Phase 12.

```powershell
go run ./api/cmd/forge-cloud validate-phase12 --gpu l4 --duration 40m --mode forge --confirm
go run ./api/cmd/forge-cloud validate-phase12 --gpu l4 --duration 40m --mode refs --confirm
go run ./api/cmd/forge-cloud validate-phase12 --gpu l4 --duration 30m --mode vllm --confirm
```

`prepare-deploy` = non-billable local tar + content-hash verify (no Studio).  
`validate-cuda` = auth → cost check → start → upload `kernels/` → `nvidia-smi` → CMake CUDA build/tests/bench → **always stop**.  
`validate-phase8b` = auth → isolated deploy tar → source verify → CUDA/Rust/Qwen stages → **always stop**.  
`validate-phase9` = same deploy model → Phase 9 batch parity + B=1..8 benches → **always stop** (**do not run without spend approval**).

## Cost controls

| Control | Default |
|---|---|
| `--confirm` | Required for start / validate-cuda |
| `--dry-run` | Estimate only |
| `FORGE_CLOUD_SPEND_LIMIT_USD` | 3 |
| `FORGE_CLOUD_MAX_SESSION` | 30m (validate-cuda caps at 20m) |
| Auto-sleep | 15 minutes |
| MaxRuntime on Start | Planned duration |

Local CLI death ≠ guaranteed stop — use auto-sleep + MaxRuntime and always run `stop`.

## Remote workflow (no browser editor)

1. Develop on Windows (CPU).
2. `forge-cloud validate-cuda --confirm` **or** start → upload/git → exec → stop.
3. Prefer Git when a remote exists; Phase 8A validation used `UploadFolder` for `kernels/` only.

## Phase 8B source deployment (isolated)

Do **not** rely on `UploadFolder` merging into an existing `forge/` tree — that left stale `runtime/src/cuda_model` files (attempt 3).

`validate-phase8b` now:

1. Builds a local staging bundle + SHA-256 manifest (`prepare-deploy` for a dry check)
2. Uploads a single `.tar.gz` to `forge-upload/`
3. Extracts into a **new** `forge-deploys/<deploy_id>/`
4. Runs `scripts/verify_forge_deploy.sh` (required files, obsolete-import rejection, content hash)
5. Points `forge` / `forge-current` at that deploy; keeps models in `forge-cache/models`
6. Compiles only after `DEPLOY_VERIFY_OK`

```powershell
# Non-billable
go run ./api/cmd/forge-cloud prepare-deploy

# Billable — requires explicit approval (L4, ≤20m, ≤$2 cap)
go run ./api/cmd/forge-cloud validate-phase8b --gpu l4 --duration 20m --confirm

# Phase 9 continuous batching (L4, ≤30m, ≤$2 cap) — only after approval
# go run ./api/cmd/forge-cloud validate-phase9 --gpu l4 --duration 30m --confirm
```

Never upload `.env`. Secrets stay local.

## Phase 8A GPU proof

See [PHASE8A_RESULTS.md](PHASE8A_RESULTS.md): NVIDIA L4, all `forge_cuda` tests passed, microbenches recorded, stop verified.


## Phase 13 bounded session

Phase 13 used one existing forge-gpu L4, verified on-demand rate $1.68/hour, a $1.20 reservation, remote and local absolute-deadline watchdogs, and verified actual stop. Provider MaxRuntime was requested but was not visible as enforced, so independent watchdogs were required. Conservative estimate $1.03; remaining $4.97 compute budget. See [PHASE13_COST_LEDGER.md](PHASE13_COST_LEDGER.md). New forge-cloud inspect-phase13, deploy-phase13 and fetch-phase13 commands inspect provider state, transfer a content-verified source bundle, and retrieve evidence. Existing deployment verification/cache layout is preserved. Always rebuild cargo release shared library with --features cuda before linking the Go server; building examples alone leaves an older cdylib. Session-specific watchdog deadlines must be replaced and verified before any future start.

## Phase 14 session and deployment

One L4 session, full provisioning-to-verified-stop interval 41m08s, estimated
$1.16 at $1.68/hour; actual invoice unavailable. $4.84 remains in the separate
$6 Phase 14 budget. All visible Studios were verified with no requested/in-use
GPU. See [PHASE14_COST_LEDGER.md](PHASE14_COST_LEDGER.md) for timestamps and guards.

`forge-cloud inspect-phase14`, `deploy-phase14` and `fetch-phase14` provide
Phase 14 entry points. Hash-verified final deployment:
`20261008T192014Z-e4331fde9a23`, content SHA-256
`e4331fde9a239d9e064ec786fc867fc0573b757c314acfc03cd71c8e649dd87a`.
Core source was verified on each deployment; later diagnostic/control scripts
were uploaded separately (diagnostic SHA verified). Evidence is under
`benchmarks/results/phase14`. Independent local and remote deadlines guarded
this run; provider MaxRuntime returned 0, so that setting was not relied upon.
No paid GPU remains active and no further session is scheduled.

## Phase 15 bounded validation

`inspect-phase15`, `deploy-phase15` and `fetch-phase15` reuse the existing
Studio/deploy/cache workflow. Source updates and measurements were checked with
retained-source-hashes.json. Fetch excludes a temporary binary restoration
archive that had a .json suffix; it is not benchmark evidence.

Three sessions used the same forge-gpu NVIDIA L4 at the independently checked
$1.68/hour rate. Conservative compute estimate is $2.98, with $0.15 held for
delayed billing, against a new $6 Phase 15 limit. Actual invoice is unavailable.
See [ledger](PHASE15_COST_LEDGER.md) for start/stop timestamps, margins and proofs.
Independent remote and hidden local absolute-deadline watchdogs were verified.
The remote SDK watchdog autodetects the current Studio, forbids creation and
checks its name; a display organization name is not necessarily an SDK slug.
Provider MaxRuntime was not visibly enforced. Final independent API inspection
returned requested=null and inUse=null for all visible Studios.

`scripts/phase15_validate.sh` and `phase15_fp32_validate.sh` execute checks and
benchmarks only; they do not provision a GPU or replace session budget/watchdog
controls. Use a fresh verified deadline for any future run.

# Phase 4 Results

## What works

- Bounded admission queue (`api/internal/queue`)
- Single-flight scheduler (one Rust generate stream at a time)
- HTTP API:
  - `POST /v1/completions` (SSE when `"stream": true`)
  - `GET /health`
  - `GET /metrics`
- Client disconnect → cancel generation
- Graceful shutdown
- Go ↔ Rust via expanded C ABI (Windows DLL load preserved)

## Run

```powershell
# build runtime DLL
cd runtime; cargo build --release; cd ..

$env:FORGE_RUNTIME_DIR = "$PWD\runtime\target\release"
cd api
go run ./cmd/forge-server -model ../models/Qwen2.5-0.5B-Instruct -addr :8080
```

```bash
curl -N http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Explain what a KV cache is\",\"max_tokens\":64,\"temperature\":0,\"stream\":true}"
```

## Execution model

Concurrent HTTP requests enqueue; inference remains **single-flight**. Queue wait is backpressure, not overlapping forwards.

## Lifecycle

`queued → running → streaming → completed | cancelled | failed`

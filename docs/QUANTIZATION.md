# Weight-only quantization

Forge keeps **activations, norms, biases, RoPE, attention, and embeddings in FP32**.
Only **linear projection weights** (Q/K/V/O, MLP gate/up/down, lm_head) may be stored as INT8 or INT4.

## Equations

Symmetric group-wise scaling for output row `j`, group `g` over columns `[g·G, (g+1)·G)`:

```text
s[j,g] = max_{k ∈ group g} |W[j,k]| / qmax
q[j,k] = clamp(round(W[j,k] / s[j,g]), qmin, qmax)
Ŵ[j,k] = q[j,k] · s[j,g]
```

| Format | `qmax` | `qmin` | Notes |
|---|---|---|---|
| INT8 | 127 | −127 | Per-group `int8` |
| INT4 | 7 | −8 | Signed nibble (two's complement); scale uses `/7` |

**Per-channel** = `group_size == in_features` (one scale per output row).

Fused matvec (no FP32 weight materialization):

```text
y[j] = bias[j] + Σ_g s[j,g] · Σ_{p ∈ g} q[j, g·G+p] · x[g·G+p]
```

## INT4 packing

- Layout: row-major over output channels.
- Within a row, columns pack **two nibbles per byte**: low nibble = even `k`, high = odd `k`.
- Packed row stride = `(K + 1) / 2` bytes.
- Scales: `float32[out × (K / group_size)]`, same as INT8.

## Runtime interface

```rust
model.quantize_weights(QuantBits::Int8, /*group_size=*/64)?;
// or QuantBits::Int4
```

`Linear::apply_rows` dispatches to:

- `forge_kernels_linear_f32`
- `forge_kernels_linear_i8_f32`
- `forge_kernels_linear_i4_f32`

Embeddings stay FP32 for token lookup. When embeddings are tied, `lm_head` is an independent clone that may be quantized.

## Memory ownership

| Object | Owner | Lifetime |
|---|---|---|
| Packed `qweight` + `scales` | Rust `QuantMatrix` inside `Linear` | Model load → free |
| FP32 activations / KV | Rust scratch + per-request caches | Request / batcher |
| Kernel temporaries | Stack / SIMD registers | Call |

## Kernel strategy

- Outer parallelization over output channels via the persistent `ThreadPool` (same worth thresholds as FP32).
- INT8: AVX2 convert `int8→float` + FMA accumulate per group, then multiply by scale.
- INT4: **fused** AVX2 nibble unpack in-register (16 weights / 8 packed bytes) → `int8→float` + FMA. No write/read of a temporary unpack buffer on the hot path.

### INT4 performance note (post Phase 7 investigation)

The original INT4 path unpacked nibbles into a 256-byte `tmp[]` then reused the INT8 kernel. Microbenches on Qwen shapes showed **INT4 ≈ 7–11× slower than INT8** with **≈0 page faults during decode** — kernel inefficiency, not RAM pressure. After fused unpack, microbench INT4 is within ~1–2× of INT8 (sometimes faster on bandwidth-bound shapes). See [PHASE6_RESULTS.md](PHASE6_RESULTS.md).

## CLI

```powershell
# Online (in-process) quant — fine for experiments; keeps a load+quant peak
cargo run --release --example generate_qwen -- ../models/Qwen2.5-0.5B-Instruct --bits 8 --group 64

# Offline checkpoint (preferred): quantize once, load many times without re-quant
cargo run --release --bin forge-quantize -- `
  --model ../models/Qwen2.5-0.5B-Instruct --bits 8 --group-size 64 `
  --output ../models/qwen2.5-int8.forge

cargo run --release --bin forge-bench -- --model ../models/qwen2.5-int8.forge --backend forge
```

Persistent layout: [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md).

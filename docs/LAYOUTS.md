# Tensor Layouts & Ownership (Phase 2+)

All runtime tensors are **contiguous row-major FP32** unless noted.

## Conventions

| Tensor | Shape | Notes |
|---|---|---|
| Hidden `x` | `[T, H]` | batch=1; `T`=seq len, `H`=hidden |
| Linear weight `W` | `[out, in]` | PyTorch `nn.Linear` layout; apply as `y = x @ Wᵀ` |
| Linear bias | `[out]` | optional |
| Q (projected) | `[T, n_heads * D]` | then viewed as `[T, n_heads, D]` |
| K/V (projected) | `[T, n_kv * D]` | viewed as `[T, n_kv, D]` |
| KV cache K/V | `[max_T, n_kv, D]` | filled for positions `[0, cache_len)` |
| Attn scores | `[n_heads, T_q, T_k]` | scratch; causal upper triangle = −∞ before softmax |
| MLP gate/up | `[T, I]` | `I` = intermediate_size |
| RMSNorm weight | `[H]` | γ only (no bias) |

`D = H / n_heads` (head_dim). For Qwen2.5-0.5B: `H=896`, `n_heads=14`, `n_kv=2`, `D=64`.

## GQA mapping

Query head `h` reads KV head `h * n_kv / n_heads` (integer division), i.e.
`kv_idx = h / (n_heads / n_kv)`.

## RoPE

Hugging Face / Qwen2 **half-rotation** style (not interleaved GPT-J):

- Split last dim into `[..., :D/2]` and `[..., D/2:]`
- `rotate_half(x) = cat(-x2, x1)`
- `out = x * cos + rotate_half(x) * sin`
- Frequencies: `θ_i = rope_theta ^ (-2i/D)` for `i in 0..D/2`
- Positions are absolute token indices (cache-aware: decode uses `position = cache_len`)

## Ownership

| Object | Owner | Lifetime |
|---|---|---|
| Model weights | Rust `Transformer` / layer structs | Until runtime drop |
| Activations / scratch | Caller or `ForwardScratch` workspace | Per forward step |
| KV cache | Per-request `KvCache` | Until request completes |
| C++ kernel buffers | Non-owning views into Rust memory | Call duration only |

Go never holds pointers into Rust weight/activation memory across FFI calls in Phase 2–3 (generation API copies token IDs / text out).

## Hot-path allocation policy

- Prefill/decode reuse a pre-sized `ForwardScratch` (Q, K, V, attn, MLP temps).
- KV cache is allocated once at request start for `max_seq`.
- Kernels do not heap-allocate; they write into caller buffers.

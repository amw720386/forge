# Phase 10 Design — Paged GPU KV, Chunked Prefill & Token-Budget Scheduling

## Goal

Serve concurrent requests with different context lengths under **predictable GPU KV memory**, without allocating full `max_seq` KV for every admitted slot. Interleave **chunked prefill** with decode so long prompts do not monopolize the GPU.

Preserve Phase 8B/9 contiguous CUDA paths as correctness oracles.

## Current state (Phase 9)

| Component | Behavior |
|---|---|
| `LayerGpuKvCaches` | Eager contiguous `[max_seq, kv_width]` K/V per layer per slot |
| `CudaBatcher::admit` | Full-prompt `forward_tokens` (serialized; stalls decode) |
| Attention | `forge_cuda_gqa_attn_f32` over dense prefix |
| Scheduling | Slot-count admission; shared decode only |
| Memory | `B × L × 2 × max_seq × kv_w × 4` at construct |

Phase 9 L4 baselines (identical short prompts, `max_new=16`): B=1 ~22 tok/s, B=8 ~42 tok/s agg; weights ~1884 MiB.

## Design

### Physical KV blocks

```text
block_size = 16 tokens (configurable via FORGE_KV_BLOCK_SIZE)
kv_width   = n_kv_heads × head_dim          # Qwen2.5-0.5B: 128
bytes/block/layer (K+V) = 2 × block_size × kv_width × 4
bytes/block (all layers) = n_layers × that  # 24 × 16 KiB = 384 KiB
```

Pool layout (device FP32):

```text
k_pool[n_layers][num_blocks][block_size][kv_width]
v_pool[n_layers][num_blocks][block_size][kv_width]
```

Index for layer `L`, physical block `P`, row `r` in block:

```text
offset = ((L * num_blocks + P) * block_size + r) * kv_width
```

`num_blocks` is sized from env / budget (default: enough for `max_batch × ceil(max_seq / block_size)` with optional headroom), not one contiguous `max_seq` buffer per slot.

### Logical block tables

Per request:

```text
PagedKvSession {
  block_table: Vec<u32>,   // logical block i → physical block id
  len: usize,              // valid token count
  device_table: Buffer,    // H2D mirror for attention (capacity max_blocks_per_seq)
}
```

- Grow: when `len` crosses a block boundary, pop from free list → append to table.
- Shrink / free: on cancel or complete, push all physical ids back to free list; clear table; `len = 0`.
- Ownership: only the owning session may append or attend through its table. Reclaimed blocks are not readable by active sessions.

### Paged attention

New kernel `forge_cuda_paged_gqa_attn_f32`:

- Same GQA / causal / RoPE contract as contiguous `gqa_attn_f32`.
- For each query position, iterate logical key indices `0..valid_k`, resolve `(block_table[i/block_size], i%block_size)`, read K/V from the pool.
- Does **not** rebuild a full contiguous cache each step.
- Contiguous `gqa_attn_f32` remains for `forward_tokens` / parity.

Paged append: `forge_cuda_paged_kv_append_f32` writes `t` rows into the correct physical slots (may span a block boundary; caller ensures blocks are allocated first).

### Chunked prefill

```text
SlotStatus ∈ { Free, Prefilling, Ready, Decoding }

admit:
  reserve slot + check free blocks ≥ ceil(prompt_len / block_size)
  status = Prefilling, prefill_pos = 0
  (no full forward)

step iteration:
  1. Sample/emit Ready → Decoding or Free
  2. Schedule under token budget:
       decode_tokens  (1 per Decoding slot, priority)
       prefill_tokens (chunks of Prefilling slots, fair round-robin)
  3. Execute decode batch, then prefill chunk(s)
  4. Prefilling → Ready when prefill_pos == prompt_len
```

A prefill chunk of length `c` at `prefill_pos = p` attends to all prior KV for that request (`t_k = p + c`), with causal masking and RoPE at absolute positions. Previously written K/V are not recomputed.

Config:

| Env | Default | Role |
|---|---|---|
| `FORGE_KV_BLOCK_SIZE` | 16 | Tokens per physical block |
| `FORGE_KV_NUM_BLOCKS` | `max_batch * ceil(max_seq/block)` | Pool capacity |
| `FORGE_PREFILL_CHUNK` | 128 | Max tokens per prefill chunk |
| `FORGE_STEP_TOKEN_BUDGET` | 256 | Prefill+decode tokens per `step` |

### Token-budget scheduling

Per `step`:

1. Count Decoding slots → each costs 1 decode token.
2. Remaining budget → assign to Prefilling slots in round-robin, up to `FORGE_PREFILL_CHUNK` each.
3. Metrics: scheduled decode tokens, scheduled prefill tokens, active prefill/decode counts, free/used blocks.

Avoid starvation: always allow at least one decode token if any Decoding slot exists when budget > 0; rotate prefill cursor.

### Admission & cancellation

- Admit fails with capacity error if free blocks < blocks needed for the prompt (checked arithmetic).
- Cancel during Prefilling or Decoding returns all blocks; slot → Free.
- Double-free prevented by clearing the table before reuse and only freeing owned ids.

### Contiguous reference path

Unchanged:

- `GpuKvCache` / `LayerGpuKvCaches`
- `CudaTransformer::forward_tokens`
- `generate_cuda`
- `forge_cuda_gqa_attn_f32` / `kv_append_f32`

Paged batcher is the serving path; parity tests compare paged vs contiguous greedy tokens.

### FFI / Go / metrics

Keep `forge_batch_*`. Extend observability (no raw pointers):

- queue length (existing)
- active decode / active prefill
- scheduled prefill/decode tokens (last step)
- KV blocks allocated / free / utilization
- cancellations, prefill stalls (time spent in prefill while decode waited — measured)

### Out of scope (Phase 11+)

FlashAttention, speculative decoding, CUDA Graphs, GPU weight quant, multi-GPU.

## Migration order

1. Host block allocator + unit tests (no GPU)
2. CUDA paged append + paged GQA (+ stub symbols)
3. Wire ComputeBackend + Rust paged session
4. Paged vs contiguous attention/generation parity
5. Chunked prefill in CudaBatcher
6. Token-budget scheduling + KV-aware admit
7. Metrics + benches + Lightning validation
8. Docs / budget ledger / verified stop

## Risks

| Risk | Mitigation |
|---|---|
| Paged attn slower than contiguous | Accept for Phase 10; profile after parity; keep contiguous oracle |
| Block-table H2D each layer | Reuse per-slot device table; upload on grow only |
| Mid-prefill OOM | Admit reserves prompt blocks up front |
| Fairness / starvation | Decode priority + round-robin prefill cursor |

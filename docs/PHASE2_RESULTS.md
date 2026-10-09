# Phase 2 Results

## What works

- C++ kernels: RMSNorm, stable softmax (+ causal), RoPE (HF half-split), SiLU, linear (`X @ Wᵀ`), mul/add
- Rust: GQA causal attention, SwiGLU MLP, decoder block, synthetic multi-layer transformer
- KV cache with multi-token prefill and single-token decode
- Prefill vs token-by-token logits match within `atol=1e-4`, `rtol=1e-3`
- PyTorch fixture parity for RMSNorm, softmax, SiLU, RoPE, GQA attention
- Layout/ownership docs: [LAYOUTS.md](LAYOUTS.md)

## Tests

```text
kernels: forge_kernels_tests (incl. phase2) — pass
cargo test --release:
  lib unit tests — 11 pass
  phase2_fixtures — 7 pass (vs PyTorch bins)
  phase2_transformer — 4 pass (KV cache parity)
```

Regenerate fixtures:

```bash
python tests/reference/phase2_reference.py
```

## Known limitations

- Decoder hot path still copies row prefixes for some ops (correctness first; reduce in Phase 5)
- Attention score/context loops are scalar Rust (not fused C++ yet)
- Synthetic model only — real weights in Phase 3

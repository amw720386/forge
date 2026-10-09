# Phase 3 Results

## What works

- Safetensors loader (BF16 → FP32) for Qwen2 HF layout
- `config.json` → `ModelConfig`
- Weight mapping for all 24 layers (QKV bias, SwiGLU, RMSNorm, tied embeddings)
- `tokenizer.json` via Hugging Face `tokenizers` crate
- Autoregressive generation with KV cache (prefill + decode)
- Greedy decoding

## Parity vs Hugging Face Transformers (FP32)

Prompt (chat template):

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Say hello in one short sentence.<|im_end|>
<|im_start|>assistant
```

| Check | Hugging Face | Forge |
|---|---|---|
| First greedy token id | 9707 | 9707 |
| Generated ids | `[9707, 0, 1084, 594, 6419, 311, 3367, 498, 13, 151645]` | identical |
| Decoded text | `Hello! It's nice to meet you.` | identical |

Commands:

```powershell
python tests/reference/phase3_hf_parity.py 16
cargo run --release --example generate_qwen -- ../models/Qwen2.5-0.5B-Instruct
```

## Performance (CPU, unoptimized)

- Load ~5.3s
- ~0.44 tok/s decode on the test machine (expected; Phase 5 optimizes)

## Known limitations

- FP32 only (upstream weights are BF16)
- No continuous batching yet
- Logits projected only for the last token during generate (full rows available via `forward_logits_all` for tests)

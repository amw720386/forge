# Models

Download weights locally; do not commit them.

```bash
# Example (requires huggingface-cli / huggingface_hub)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

Expected files: `config.json`, `model.safetensors`, tokenizer files.

Phase 16 also supports `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0):

```bash
hf download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir models/SmolLM2-135M-Instruct
```

See [architecture support](../docs/MODEL_ARCHITECTURES.md) and
[precision qualifications](../docs/PRECISION_CONTRACTS.md). Model weights remain
gitignored; preserve upstream attribution/license notices if redistributing them.

### Forge checkpoints (gitignored)

```powershell
cargo run --release --bin forge-quantize -- `
  --model models/Qwen2.5-0.5B-Instruct --bits 8 --group-size 64 `
  --output models/qwen2.5-int8.forge
```

Keep `tokenizer.json` available next to the HF tree (server resolves it when loading `.forge`).

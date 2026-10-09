# Local models

Provide a Hugging Face model directory containing `config.json`, a single
`model.safetensors`, `tokenizer.json`, and `tokenizer_config.json` (for chat):

- `models/Qwen2.5-0.5B-Instruct` from `Qwen/Qwen2.5-0.5B-Instruct`.
- `models/SmolLM2-135M-Instruct` from `HuggingFaceTB/SmolLM2-135M-Instruct`.

Weights and generated `.forge` checkpoints are ignored by Git. Preserve upstream
licenses and attribution when redistributing models. See [build.md](../docs/build.md)
for loading, tokenizer resolution and offline quantization; the README lists
backend limitations, including experimental SmolLM2 BF16.

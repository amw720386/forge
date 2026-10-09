# Supported model architectures

| Family adapter | Pretrained target | Attention | Parameters relevant to execution |
|---|---|---|---|
| `qwen2` | Qwen/Qwen2.5-0.5B-Instruct | Causal GQA, Q/K/V biases | H=896, I=4864, layers=24, heads=14, KV heads=2, head=64, vocabulary=151936 |
| `llama` | HuggingFaceTB/SmolLM2-135M-Instruct | Causal GQA, no projection biases | H=576, I=1536, layers=30, heads=9, KV heads=3, head=64, vocabulary=49152 |

Both targets use pre-norm RMSNorm, SiLU/SwiGLU, split-half full-dimensional RoPE,
residual additions and tied embedding/output weights. SmolLM2 uses theta=100000,
epsilon=1e-5, context=8192 and BOS/EOS/padding IDs 1/2/2. Its tokenizer is GPT2-style
byte-level BPE with `tokenizer.json`; it is not inferred from the Llama family name.
Qwen uses theta=1000000, epsilon=1e-6 and configured context=32768. Context size is
metadata, not a claim that every full-length backend workload has been audited.

SmolLM2 was inspected at HF revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
The actual 272 tensor names/shapes/dtypes are retained in
`tests/fixtures/phase16/smollm2/tensor_manifest.json`. The published
[configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json)
declares `LlamaForCausalLM`; the [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md)
declares Apache-2.0. Preserve model attribution, applicable notices and the license
when redistributing weights. Forge's MIT license does not replace the model license.
No weights are committed.

Both adapters map `model.embed_tokens.weight`, `model.norm.weight`, optional
`lm_head.weight`, layer norms, `self_attn.{q,k,v,o}_proj` and
`mlp.{gate,up,down}_proj`. Matrices are `[out,in]`; no layout transformation is needed.
Only Qwen requires Q/K/V `.bias`. Llama's unexpected biases fail explicitly.

| Model | CPU FP32 | CPU INT8/INT4 | CUDA FP32 | CUDA BF16 |
|---|---|---|---|---|
| Qwen2.5-0.5B | Supported, HF reference checked | Existing approximate quantized execution and legacy checkpoints checked | Supported, real L4 regressions | Supported with Phase 15 numerical/dispatch qualifications |
| SmolLM2-135M | Supported, layer/logit/cache/greedy/batch parity | Execution and pretrained checkpoint round trips validated; approximate output quality | Supported on L4, greedy/logit/paged/batch/graph tests | Experimental numerical qualification: tested greedy, graphs and HTTP pass; logits differ from FP32 |

Other configurations matching the supported static plan may load, but are unvalidated
pretrained targets. Scaled/partial/interleaved RoPE, enabled sliding-window attention,
non-SiLU activation, MLP bias, alternate head dimensions, unsupported model types,
HF quantization metadata and tensor-parallel pretraining are rejected. CUDA INT8/INT4
and sharded Safetensors are unsupported. This is not general compatibility with every
checkpoint branded Llama or Qwen.

```powershell
Set-Location api
go run ./cmd/forge-server --model ../models/SmolLM2-135M-Instruct --device cpu
# On a real CUDA build; BF16 is qualified as above:
go run ./cmd/forge-server --model ../models/SmolLM2-135M-Instruct --device cuda --dtype bf16
# When the tokenizer is not in the standard sibling layout:
$env:FORGE_TOKENIZER_DIR = '../models/SmolLM2-135M-Instruct'
```

Run Go commands from `api/` as `go run ./cmd/forge-server ...` when the Go toolchain
requires its module directory. Pass a fully formatted chat prompt for instruct models.

# Precision contracts

`ExecutionPrecision` and `PrecisionContract` describe existing kernel behavior.
Storage below means projection weights; embeddings, norms and biases have their
own FP32 storage. Quantization remains weight-only symmetric quantization.

| Execution | Projection storage | Projection inputs | Persistent activations | GEMM accumulation | KV | Norm output/reduction | Attention reduction | Sampling |
|---|---|---|---|---|---|---|---|---|
| CPU FP32 | FP32 | FP32 | FP32 | FP32 | FP32 | FP32 / FP64 sum of squares | FP32 | FP32 logits |
| CPU INT8 | INT8 + FP32 scales | FP32 | FP32 | FP32 after weight dequantization | FP32 | FP32 / FP64 | FP32 | FP32 logits |
| CPU INT4 | Packed INT4 + FP32 scales | FP32 | FP32 | FP32 after weight dequantization | FP32 | FP32 / FP64 | FP32 | FP32 logits |
| CUDA FP32 | FP32 | FP32 | FP32 | FP32 | FP32 | FP32 / FP32 | FP32 | FP32 logits or device argmax |
| CUDA BF16 | BF16 | BF16 transient cast | FP32 | FP32 | FP32 | FP32 / FP32 | FP32 | FP32 logits or device argmax |

CUDA BF16 does not turn KV, residual storage, RoPE, normalization, softmax or
sampling into BF16. BF16 projection input rounding can change subsequent FP32
hidden states. CPU RMSNorm's sum of squares already uses double; Phase 16 preserves it.
F16/BF16 Safetensors are expanded into FP32 host model tensors before backend upload.

## Comparison policy

Existing primitive, quantization and CUDA regression tolerances are unchanged.
FP32 reference tests require numerical comparisons as well as exact greedy decisions.
The new pretrained checks use `1e-4 + 1e-4*abs(reference)` for full logits and cached
logits, and `1e-3 + 1e-4*abs(reference)` for raw layer residuals. Raw SmolLM2 residual
magnitudes reach about 4429; a raw absolute difference near 0.0073 is accompanied
by layer RMS error under 4e-5 and full-logit error under 4.2e-5 on Windows.

Weight-only INT8/INT4 are approximate: bitwise checkpoint round trips and batch/cache
consistency are required, but universal FP32 greedy agreement is not promised.
Pretrained SmolLM2 round trips pass for both formats. This does not certify quality
on a broad evaluation suite.

SmolLM2 CUDA BF16 is **experimental with bounded numerical evidence**, not tight FP32
logit parity. Four prompts produce 44/44 matching greedy tokens; eager/paged/graph
batch cases also match. HF FP32 logit differences range up to 0.205344, RMS up to
0.088606. The Phase 16 test's model-specific maximum-error envelope is 0.25; RMS
is reported. An initial proposed RMS bound of 0.04 failed and is retained in evidence;
it was not an inherited Forge numerical tolerance and no existing test was relaxed.

An isolated final projection applied to exact FP32 CPU hidden states differs from
HF FP32 by 0.0428–0.1048. The CUDA BF16 projection matches independently rounded
BF16 inputs/weights with FP64 reference accumulation within 4.91e-5 max and 6.67e-6
RMS, under the same strict `1e-4 + 1e-4*abs(reference)` comparison. This establishes
the expected rounding contract for that projection. It does not completely attribute
every earlier-layer BF16 deviation. Broader layer attribution and context sweeps remain
release-readiness work; tested greedy stability alone is not universal correctness.

## Conservative attention dispatch

Partitioned decode requires all next lengths at least 128. FP32 can use it at long
contexts. BF16 uses it only at B<=2 or when every next length is <=512; larger long
batches preserve the Phase 15 conservative fallback. Environment dispatch changes
invalidate graphs. Shared prefill tiling, BF16 split reductions and adaptive scheduler
experiments rejected in Phase 15 were not reintroduced.

#ifndef FORGE_KERNELS_H
#define FORGE_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default to static linkage. Define FORGE_KERNELS_EXPORTS when building a DLL. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef FORGE_KERNELS_EXPORTS
#    define FORGE_KERNELS_API __declspec(dllexport)
#  elif defined(FORGE_KERNELS_DLL)
#    define FORGE_KERNELS_API __declspec(dllimport)
#  else
#    define FORGE_KERNELS_API
#  endif
#else
#  define FORGE_KERNELS_API __attribute__((visibility("default")))
#endif

typedef enum forge_status {
  FORGE_OK = 0,
  FORGE_ERR_INVALID_ARGUMENT = 1,
  FORGE_ERR_DIMENSION_MISMATCH = 2,
  FORGE_ERR_OUT_OF_MEMORY = 3,
  FORGE_ERR_INTERNAL = 4
} forge_status;

/** Human-readable kernel library version (static storage). */
FORGE_KERNELS_API const char* forge_kernels_version(void);

/** Effective worker count for parallel kernels (FORGE_NUM_THREADS or HW). */
FORGE_KERNELS_API unsigned forge_kernels_num_threads(void);

/* ---- Phase 1: GEMM ------------------------------------------------------- */

/** C = A (M x K) * B (K x N), row-major FP32. */
FORGE_KERNELS_API forge_status forge_kernels_matmul_f32(
    size_t m, size_t n, size_t k,
    const float* a, size_t lda,
    const float* b, size_t ldb,
    float* c, size_t ldc);

/** y = A (M x N) * x (N), row-major FP32. */
FORGE_KERNELS_API forge_status forge_kernels_matvec_f32(
    size_t m, size_t n,
    const float* a, size_t lda,
    const float* x, float* y);

/* ---- Phase 2: transformer primitives ------------------------------------- */

/**
 * Linear: Y = X @ W^T + optional bias.
 * X: [m, k], W: [n, k] (PyTorch layout), Y: [m, n], bias: [n] or NULL.
 */
FORGE_KERNELS_API forge_status forge_kernels_linear_f32(
    size_t m, size_t n, size_t k,
    const float* x, size_t ldx,
    const float* w, size_t ldw,
    const float* bias, /* nullable */
    float* y, size_t ldy);

/**
 * Weight-only INT8 linear with fused dequant (no FP32 weight materialization).
 * W_q: int8 [n, k] row-major. scales: [n * (k/group_size)] row-major.
 * group_size must divide k. Per-channel = group_size == k (one scale per output row).
 * y[i,j] = bias[j] + sum_g scales[j,g] * sum_{p in group g} W_q[j, g*gs+p] * x[i, g*gs+p]
 */
FORGE_KERNELS_API forge_status forge_kernels_linear_i8_f32(
    size_t m, size_t n, size_t k,
    const float* x, size_t ldx,
    const int8_t* w_q,
    const float* scales,
    size_t group_size,
    const float* bias, /* nullable */
    float* y, size_t ldy);

/**
 * Weight-only INT4 linear. w_q packs two signed nibbles per byte (low nibble = even k).
 * Packed row stride = (k + 1) / 2 bytes. Scales layout same as INT8.
 * Nibbles are sign-extended 4-bit two's complement in [-8, 7].
 */
FORGE_KERNELS_API forge_status forge_kernels_linear_i4_f32(
    size_t m, size_t n, size_t k,
    const float* x, size_t ldx,
    const uint8_t* w_q,
    const float* scales,
    size_t group_size,
    const float* bias, /* nullable */
    float* y, size_t ldy);

/**
 * RMSNorm over the last dimension.
 * X/Y: [rows, dim], weight: [dim] (gamma). y_i = x_i * rsqrt(mean(x^2)+eps) * w_i
 */
FORGE_KERNELS_API forge_status forge_kernels_rmsnorm_f32(
    size_t rows, size_t dim,
    const float* x, size_t ldx,
    const float* weight,
    float eps,
    float* y, size_t ldy);

/**
 * Numerically stable softmax along the last axis of a 3D tensor [B, R, C]
 * stored contiguous row-major (B*R rows of length C).
 * If causal != 0, for row r within each block of R rows, mask columns c > r
 * (standard causal; for decode with T_q=1 and T_k=S, pass causal=0 and
 * pre-masked scores, or use causal with R=T_q when T_q==T_k).
 *
 * When causal_offset > 0 (decode): query row 0 attends to columns
 * [0, causal_offset] inclusive if C == causal_offset+1 — prefer pre-masked
 * scores for decode. For prefill T_q==T_k, set causal=1 and causal_offset=0.
 */
FORGE_KERNELS_API forge_status forge_kernels_softmax_f32(
    size_t batch, size_t rows, size_t cols,
    float* scores, /* in-place */
    int causal,
    size_t causal_q_offset);

/**
 * Apply HF/Qwen2 half-split RoPE in-place to x shaped [T, n_heads, head_dim]
 * stored contiguous. head_dim must be even.
 * positions[t] is the absolute position for token t (length T), or NULL to use
 * position_offset + t.
 */
FORGE_KERNELS_API forge_status forge_kernels_rope_f32(
    size_t seq_len, size_t n_heads, size_t head_dim,
    float* x,
    const int32_t* positions, /* nullable */
    size_t position_offset,
    float rope_theta);

/** SiLU / swish: y[i] = x[i] / (1 + exp(-x[i])) ; supports in-place (y==x). */
FORGE_KERNELS_API forge_status forge_kernels_silu_f32(
    size_t n, const float* x, float* y);

/** Elementwise: y = a * b (same length). */
FORGE_KERNELS_API forge_status forge_kernels_mul_f32(
    size_t n, const float* a, const float* b, float* y);

/** Elementwise: y = a + b. */
FORGE_KERNELS_API forge_status forge_kernels_add_f32(
    size_t n, const float* a, const float* b, float* y);

/** y = a + scale * b (residual helper). */
FORGE_KERNELS_API forge_status forge_kernels_add_scaled_f32(
    size_t n, const float* a, const float* b, float scale, float* y);

#ifdef __cplusplus
}
#endif

#endif /* FORGE_KERNELS_H */

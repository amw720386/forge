#ifndef FORGE_CUDA_H
#define FORGE_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef FORGE_CUDA_EXPORTS
#define FORGE_CUDA_API __declspec(dllexport)
#elif defined(FORGE_CUDA_DLL)
#define FORGE_CUDA_API __declspec(dllimport)
#else
#define FORGE_CUDA_API
#endif
#else
#define FORGE_CUDA_API __attribute__((visibility("default")))
#endif

typedef enum forge_cuda_status {
  FORGE_CUDA_OK = 0,
  FORGE_CUDA_ERR_INVALID_ARGUMENT = 1,
  FORGE_CUDA_ERR_DIMENSION_MISMATCH = 2,
  FORGE_CUDA_ERR_OUT_OF_MEMORY = 3,
  FORGE_CUDA_ERR_INTERNAL = 4,
  FORGE_CUDA_ERR_UNAVAILABLE = 5,
  FORGE_CUDA_ERR_NOT_BUILT = 6
} forge_cuda_status;

/** Opaque device buffer (device pointer + byte length). */
typedef struct forge_cuda_buffer {
  void* ptr;
  size_t bytes;
} forge_cuda_buffer;

FORGE_CUDA_API const char* forge_cuda_version(void);

/** 1 if this binary was built with CUDA and a device is visible. */
FORGE_CUDA_API int forge_cuda_is_available(void);

FORGE_CUDA_API int forge_cuda_device_count(void);

FORGE_CUDA_API forge_cuda_status forge_cuda_set_device(int device);

/** Fills a short human-readable device name (null-terminated). */
FORGE_CUDA_API forge_cuda_status forge_cuda_device_name(int device, char* out,
                                                        size_t out_len);

/** Free / total device memory in bytes for the current device. */
FORGE_CUDA_API forge_cuda_status forge_cuda_mem_info(size_t* free_bytes,
                                                     size_t* total_bytes);

FORGE_CUDA_API const char* forge_cuda_last_error(void);

FORGE_CUDA_API forge_cuda_status forge_cuda_malloc(forge_cuda_buffer* buf,
                                                   size_t bytes);

FORGE_CUDA_API forge_cuda_status forge_cuda_free(forge_cuda_buffer* buf);

FORGE_CUDA_API forge_cuda_status forge_cuda_memcpy_h2d(forge_cuda_buffer* dst,
                                                       const void* src,
                                                       size_t bytes);

FORGE_CUDA_API forge_cuda_status
forge_cuda_memcpy_d2h(void* dst, const forge_cuda_buffer* src, size_t bytes);

FORGE_CUDA_API forge_cuda_status forge_cuda_memcpy_d2d(
    forge_cuda_buffer* dst, size_t dst_off, const forge_cuda_buffer* src,
    size_t src_off, size_t bytes);

FORGE_CUDA_API forge_cuda_status forge_cuda_memset_zero(forge_cuda_buffer* buf);

FORGE_CUDA_API forge_cuda_status forge_cuda_synchronize(void);

/**
 * C = A (MÃ—K) * B (KÃ—N), row-major FP32, all pointers device memory.
 * lda/ldb/ldc are row strides in elements (â‰¥ inner dims).
 */
FORGE_CUDA_API forge_cuda_status
forge_cuda_matmul_f32(size_t m, size_t n, size_t k, const float* a, size_t lda,
                      const float* b, size_t ldb, float* c, size_t ldc);

/** y = A (MÃ—N) * x (N); A, x, y on device. */
FORGE_CUDA_API forge_cuda_status forge_cuda_matvec_f32(
    size_t m, size_t n, const float* a, size_t lda, const float* x, float* y);

FORGE_CUDA_API forge_cuda_status
forge_cuda_rmsnorm_f32(size_t rows, size_t dim, const float* x, size_t ldx,
                       const float* weight, float eps, float* y, size_t ldy);

/** Softmax along last axis of [batch, rows, cols] contiguous. causalâ‰ 0 masks.
 */
FORGE_CUDA_API forge_cuda_status
forge_cuda_softmax_f32(size_t batch, size_t rows, size_t cols, float* scores,
                       int causal, size_t causal_q_offset);

FORGE_CUDA_API forge_cuda_status forge_cuda_silu_f32(size_t n, const float* x,
                                                     float* y);

FORGE_CUDA_API forge_cuda_status forge_cuda_mul_f32(size_t n, const float* a,
                                                    const float* b, float* y);

FORGE_CUDA_API forge_cuda_status forge_cuda_add_f32(size_t n, const float* a,
                                                    const float* b, float* y);

/** y = x @ W^T + bias; W row-major [n, k], x [m, k]. Pointers on device. */
FORGE_CUDA_API forge_cuda_status forge_cuda_linear_f32(
    size_t m, size_t n, size_t k, const float* x, size_t ldx, const float* w,
    size_t ldw, const float* bias, float* y, size_t ldy);

/**
 * Convert n FP32 values to BF16 on device (dst_bf16 is __nv_bfloat16*).
 */
FORGE_CUDA_API forge_cuda_status forge_cuda_f32_to_bf16(size_t n,
                                                        const float* src,
                                                        void* dst_bf16);

/**
 * y = x @ W^T + bias with BF16 weights, FP32 activations/output, FP32
 * accumulate. W is device BF16 row-major [n, k]. x/y/bias are FP32.
 */
FORGE_CUDA_API forge_cuda_status forge_cuda_linear_bf16_w(
    size_t m, size_t n, size_t k, const float* x, size_t ldx,
    const void* w_bf16, size_t ldw, const float* bias, float* y, size_t ldy);

/**
 * Row-wise argmax over [rows, cols] FP32 logits â†’ int32 token ids on device.
 * Tie-break: lowest index (matches CPU greedy).
 */
FORGE_CUDA_API forge_cuda_status forge_cuda_argmax_rows_f32(size_t rows,
                                                            size_t cols,
                                                            const float* logits,
                                                            int32_t* out_ids);

/** Embedding gather from BF16 table â†’ FP32 out. */
FORGE_CUDA_API forge_cuda_status
forge_cuda_embedding_bf16(const void* embed_bf16, size_t vocab, size_t hidden,
                          const int32_t* token_ids, size_t t, float* out);

FORGE_CUDA_API forge_cuda_status forge_cuda_rope_f32(size_t seq_len,
                                                     size_t n_heads,
                                                     size_t head_dim, float* x,
                                                     size_t position_offset,
                                                     float rope_theta);

FORGE_CUDA_API forge_cuda_status
forge_cuda_embedding_f32(const float* embed, size_t vocab, size_t hidden,
                         const int32_t* token_ids, size_t t, float* out);

/** GQA causal attention; q/k/v/ctx on device. k/v are full cache prefix length
 * t_k. */
FORGE_CUDA_API forge_cuda_status forge_cuda_gqa_attn_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k, size_t kv_w,
    const float* v, float* ctx, int causal, size_t causal_q_offset);

FORGE_CUDA_API forge_cuda_status forge_cuda_copy_rows_f32(size_t rows,
                                                          size_t cols,
                                                          const float* src,
                                                          float* dst);

FORGE_CUDA_API forge_cuda_status forge_cuda_add_rows_f32(
    size_t rows, size_t cols, const float* a, const float* b, float* y);

FORGE_CUDA_API forge_cuda_status forge_cuda_kv_append_f32(
    size_t width, size_t t, size_t dst_row_offset, const float* k_src,
    const float* v_src, float* k_dst, float* v_dst);

/**
 * Paged GQA attention (default = optimized warp/online-softmax).
 * Reads K/V through a logicalâ†’physical block table (no contiguous rebuild).
 *
 * Pool layout: k_pool/v_pool[n_layers][num_blocks][block_size][kv_width]
 */
FORGE_CUDA_API forge_cuda_status forge_cuda_paged_gqa_attn_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t num_logical_blocks,
    size_t block_size, size_t num_blocks, size_t n_layers, size_t layer,
    size_t kv_w, float* ctx, int causal, size_t causal_q_offset);

/** naive paged GQA (correctness / A-B reference). */
FORGE_CUDA_API forge_cuda_status forge_cuda_paged_gqa_attn_naive_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t num_logical_blocks,
    size_t block_size, size_t num_blocks, size_t n_layers, size_t layer,
    size_t kv_w, float* ctx, int causal, size_t causal_q_offset);

/**
 * Append t K/V rows into the paged pool for one layer.
 * Caller must have allocated enough blocks so dst_token_offset+t fits.
 * May span a block boundary.
 */
FORGE_CUDA_API forge_cuda_status forge_cuda_paged_kv_append_f32(
    size_t width, size_t t, size_t dst_token_offset, size_t block_size,
    size_t num_blocks, size_t n_layers, size_t layer,
    const uint32_t* block_table, size_t num_logical_blocks, const float* k_src,
    const float* v_src, float* k_pool, float* v_pool);

/** metadata rows: uint64_t table address, RoPE position, existing KV length. */
FORGE_CUDA_API forge_cuda_status forge_cuda_batch_paged_prepare_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float theta,
    const uint64_t* metadata, float* q, const float* k, const float* v,
    float* k_pool, float* v_pool, size_t block_size, size_t blocks,
    size_t layer);
FORGE_CUDA_API forge_cuda_status forge_cuda_batch_paged_attn_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float scale,
    const uint64_t* metadata, const float* q, const float* k_pool,
    const float* v_pool, float* ctx, size_t block_size, size_t blocks,
    size_t layer);
// Explicit host dispatch; retain the original ABI above for kernel references.
FORGE_CUDA_API forge_cuda_status forge_cuda_batch_paged_attn_dispatch_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float scale,
    const uint64_t* metadata, const float* q, const float* k_pool,
    const float* v_pool, float* ctx, size_t block_size, size_t blocks,
    size_t layer, int partitioned);

FORGE_CUDA_API forge_cuda_status forge_cuda_linear_bf16_xw(
    size_t m, size_t n, size_t k, const void* x, size_t ldx, const void* w,
    size_t ldw, const float* bias, float* y, size_t ldy);

FORGE_CUDA_API forge_cuda_status forge_cuda_timer_start(void** timer);
FORGE_CUDA_API forge_cuda_status forge_cuda_timer_elapsed(void* timer,
                                                          float* milliseconds);
FORGE_CUDA_API forge_cuda_status forge_cuda_timer_destroy(void* timer);
FORGE_CUDA_API forge_cuda_status forge_cuda_graph_begin(void);
FORGE_CUDA_API forge_cuda_status forge_cuda_graph_end(void** executable);
FORGE_CUDA_API forge_cuda_status forge_cuda_graph_replay(void* executable);
FORGE_CUDA_API forge_cuda_status forge_cuda_graph_destroy(void* executable);

#ifdef __cplusplus
}
#endif

#endif /* FORGE_CUDA_H */

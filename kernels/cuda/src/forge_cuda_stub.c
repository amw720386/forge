/* Host-only stub of forge_cuda for compile/link checks without a GPU or nvcc.
 * Returns FORGE_CUDA_ERR_UNAVAILABLE / NOT_BUILT. Not for inference.
 */
#include "forge_cuda.h"

uint64_t forge_cuda_bf16_scratch_address(void) { return 0; }

#include <stdint.h>
#include <stdio.h>

static char g_err[256] = "forge_cuda_stub: no real CUDA backend linked";

const char* forge_cuda_version(void) { return "forge-cuda/stub-compile-check"; }

int forge_cuda_is_available(void) { return 0; }

int forge_cuda_device_count(void) { return 0; }

forge_cuda_status forge_cuda_set_device(int device) {
  (void)device;
  return FORGE_CUDA_ERR_UNAVAILABLE;
}

forge_cuda_status forge_cuda_device_name(int device, char* out,
                                         size_t out_len) {
  (void)device;
  if (out == NULL || out_len == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  snprintf(out, out_len, "cuda-stub");
  return FORGE_CUDA_ERR_UNAVAILABLE;
}

forge_cuda_status forge_cuda_mem_info(size_t* free_bytes, size_t* total_bytes) {
  if (free_bytes) {
    *free_bytes = 0;
  }
  if (total_bytes) {
    *total_bytes = 0;
  }
  return FORGE_CUDA_ERR_UNAVAILABLE;
}

const char* forge_cuda_last_error(void) { return g_err; }

forge_cuda_status forge_cuda_malloc(forge_cuda_buffer* buf, size_t bytes) {
  (void)bytes;
  if (buf) {
    buf->ptr = NULL;
    buf->bytes = 0;
  }
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_free(forge_cuda_buffer* buf) {
  if (buf) {
    buf->ptr = NULL;
    buf->bytes = 0;
  }
  return FORGE_CUDA_OK;
}

forge_cuda_status forge_cuda_memcpy_h2d(forge_cuda_buffer* dst, const void* src,
                                        size_t bytes) {
  (void)dst;
  (void)src;
  (void)bytes;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_memcpy_d2h(void* dst, const forge_cuda_buffer* src,
                                        size_t bytes) {
  (void)dst;
  (void)src;
  (void)bytes;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_memcpy_d2d(forge_cuda_buffer* dst, size_t dst_off,
                                        const forge_cuda_buffer* src,
                                        size_t src_off, size_t bytes) {
  (void)dst;
  (void)dst_off;
  (void)src;
  (void)src_off;
  (void)bytes;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_memset_zero(forge_cuda_buffer* buf) {
  (void)buf;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_synchronize(void) { return FORGE_CUDA_OK; }

forge_cuda_status forge_cuda_matmul_f32(size_t m, size_t n, size_t k,
                                        const float* a, size_t lda,
                                        const float* b, size_t ldb, float* c,
                                        size_t ldc) {
  (void)m;
  (void)n;
  (void)k;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  (void)c;
  (void)ldc;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_matvec_f32(size_t m, size_t n, const float* a,
                                        size_t lda, const float* x, float* y) {
  (void)m;
  (void)n;
  (void)a;
  (void)lda;
  (void)x;
  (void)y;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_rmsnorm_f32(size_t rows, size_t dim,
                                         const float* x, size_t ldx,
                                         const float* weight, float eps,
                                         float* y, size_t ldy) {
  (void)rows;
  (void)dim;
  (void)x;
  (void)ldx;
  (void)weight;
  (void)eps;
  (void)y;
  (void)ldy;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_softmax_f32(size_t batch, size_t rows, size_t cols,
                                         float* scores, int causal,
                                         size_t causal_q_offset) {
  (void)batch;
  (void)rows;
  (void)cols;
  (void)scores;
  (void)causal;
  (void)causal_q_offset;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_silu_f32(size_t n, const float* x, float* y) {
  (void)n;
  (void)x;
  (void)y;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_mul_f32(size_t n, const float* a, const float* b,
                                     float* y) {
  (void)n;
  (void)a;
  (void)b;
  (void)y;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_add_f32(size_t n, const float* a, const float* b,
                                     float* y) {
  (void)n;
  (void)a;
  (void)b;
  (void)y;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_linear_f32(size_t m, size_t n, size_t k,
                                        const float* x, size_t ldx,
                                        const float* w, size_t ldw,
                                        const float* bias, float* y,
                                        size_t ldy) {
  (void)m;
  (void)n;
  (void)k;
  (void)x;
  (void)ldx;
  (void)w;
  (void)ldw;
  (void)bias;
  (void)y;
  (void)ldy;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_f32_to_bf16(size_t n, const float* src,
                                         void* dst_bf16) {
  (void)n;
  (void)src;
  (void)dst_bf16;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_linear_bf16_w(size_t m, size_t n, size_t k,
                                           const float* x, size_t ldx,
                                           const void* w_bf16, size_t ldw,
                                           const float* bias, float* y,
                                           size_t ldy) {
  (void)m;
  (void)n;
  (void)k;
  (void)x;
  (void)ldx;
  (void)w_bf16;
  (void)ldw;
  (void)bias;
  (void)y;
  (void)ldy;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_argmax_rows_f32(size_t rows, size_t cols,
                                             const float* logits,
                                             int32_t* out_ids) {
  (void)rows;
  (void)cols;
  (void)logits;
  (void)out_ids;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_embedding_bf16(const void* embed_bf16,
                                            size_t vocab, size_t hidden,
                                            const int32_t* token_ids, size_t t,
                                            float* out) {
  (void)embed_bf16;
  (void)vocab;
  (void)hidden;
  (void)token_ids;
  (void)t;
  (void)out;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_rope_f32(size_t seq_len, size_t n_heads,
                                      size_t head_dim, float* x,
                                      size_t position_offset,
                                      float rope_theta) {
  (void)seq_len;
  (void)n_heads;
  (void)head_dim;
  (void)x;
  (void)position_offset;
  (void)rope_theta;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_embedding_f32(const float* embed, size_t vocab,
                                           size_t hidden,
                                           const int32_t* token_ids, size_t t,
                                           float* out) {
  (void)embed;
  (void)vocab;
  (void)hidden;
  (void)token_ids;
  (void)t;
  (void)out;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_gqa_attn_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k, size_t kv_w,
    const float* v, float* ctx, int causal, size_t causal_q_offset) {
  (void)t_q;
  (void)t_k;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)scale;
  (void)q;
  (void)q_w;
  (void)k;
  (void)kv_w;
  (void)v;
  (void)ctx;
  (void)causal;
  (void)causal_q_offset;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_copy_rows_f32(size_t rows, size_t cols,
                                           const float* src, float* dst) {
  (void)rows;
  (void)cols;
  (void)src;
  (void)dst;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_add_rows_f32(size_t rows, size_t cols,
                                          const float* a, const float* b,
                                          float* y) {
  (void)rows;
  (void)cols;
  (void)a;
  (void)b;
  (void)y;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_kv_append_f32(size_t width, size_t t,
                                           size_t dst_row_offset,
                                           const float* k_src,
                                           const float* v_src, float* k_dst,
                                           float* v_dst) {
  (void)width;
  (void)t;
  (void)dst_row_offset;
  (void)k_src;
  (void)v_src;
  (void)k_dst;
  (void)v_dst;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_paged_gqa_attn_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t num_logical_blocks,
    size_t block_size, size_t num_blocks, size_t n_layers, size_t layer,
    size_t kv_w, float* ctx, int causal, size_t causal_q_offset) {
  (void)t_q;
  (void)t_k;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)scale;
  (void)q;
  (void)q_w;
  (void)k_pool;
  (void)v_pool;
  (void)block_table;
  (void)num_logical_blocks;
  (void)block_size;
  (void)num_blocks;
  (void)n_layers;
  (void)layer;
  (void)kv_w;
  (void)ctx;
  (void)causal;
  (void)causal_q_offset;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_paged_gqa_attn_naive_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t num_logical_blocks,
    size_t block_size, size_t num_blocks, size_t n_layers, size_t layer,
    size_t kv_w, float* ctx, int causal, size_t causal_q_offset) {
  return forge_cuda_paged_gqa_attn_f32(
      t_q, t_k, n_heads, n_kv_heads, head_dim, scale, q, q_w, k_pool, v_pool,
      block_table, num_logical_blocks, block_size, num_blocks, n_layers, layer,
      kv_w, ctx, causal, causal_q_offset);
}

forge_cuda_status forge_cuda_paged_kv_append_f32(
    size_t width, size_t t, size_t dst_token_offset, size_t block_size,
    size_t num_blocks, size_t n_layers, size_t layer,
    const uint32_t* block_table, size_t num_logical_blocks, const float* k_src,
    const float* v_src, float* k_pool, float* v_pool) {
  (void)width;
  (void)t;
  (void)dst_token_offset;
  (void)block_size;
  (void)num_blocks;
  (void)n_layers;
  (void)layer;
  (void)block_table;
  (void)num_logical_blocks;
  (void)k_src;
  (void)v_src;
  (void)k_pool;
  (void)v_pool;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

/** metadata rows: uint64_t table address, RoPE position, existing KV length. */
forge_cuda_status forge_cuda_batch_paged_prepare_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float theta,
    const uint64_t* metadata, float* q, const float* k, const float* v,
    float* k_pool, float* v_pool, size_t block_size, size_t blocks,
    size_t layer) {
  (void)batch;
  (void)heads;
  (void)kv_heads;
  (void)dim;
  (void)theta;
  (void)metadata;
  (void)q;
  (void)k;
  (void)v;
  (void)k_pool;
  (void)v_pool;
  (void)block_size;
  (void)blocks;
  (void)layer;
  return FORGE_CUDA_ERR_NOT_BUILT;
}
forge_cuda_status forge_cuda_batch_paged_attn_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float scale,
    const uint64_t* metadata, const float* q, const float* k_pool,
    const float* v_pool, float* ctx, size_t block_size, size_t blocks,
    size_t layer) {
  (void)batch;
  (void)heads;
  (void)kv_heads;
  (void)dim;
  (void)scale;
  (void)metadata;
  (void)q;
  (void)k_pool;
  (void)v_pool;
  (void)ctx;
  (void)block_size;
  (void)blocks;
  (void)layer;
  return FORGE_CUDA_ERR_NOT_BUILT;
}
forge_cuda_status forge_cuda_batch_paged_attn_dispatch_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float scale,
    const uint64_t* metadata, const float* q, const float* k_pool,
    const float* v_pool, float* ctx, size_t block_size, size_t blocks,
    size_t layer, int partitioned) {
  (void)batch;
  (void)heads;
  (void)kv_heads;
  (void)dim;
  (void)scale;
  (void)metadata;
  (void)q;
  (void)k_pool;
  (void)v_pool;
  (void)ctx;
  (void)block_size;
  (void)blocks;
  (void)layer;
  (void)partitioned;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_linear_bf16_xw(size_t m, size_t n, size_t k,
                                            const void* x, size_t ldx,
                                            const void* w, size_t ldw,
                                            const float* bias, float* y,
                                            size_t ldy) {
  (void)m;
  (void)n;
  (void)k;
  (void)x;
  (void)ldx;
  (void)w;
  (void)ldw;
  (void)bias;
  (void)y;
  (void)ldy;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_graph_begin(void) {
  return FORGE_CUDA_ERR_NOT_BUILT;
}
forge_cuda_status forge_cuda_graph_end(void** executable) {
  (void)executable;
  return FORGE_CUDA_ERR_NOT_BUILT;
}
forge_cuda_status forge_cuda_graph_replay(void* executable) {
  (void)executable;
  return FORGE_CUDA_ERR_NOT_BUILT;
}
forge_cuda_status forge_cuda_graph_destroy(void* executable) {
  (void)executable;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

forge_cuda_status forge_cuda_timer_start(void** timer) {
  (void)timer;
  return FORGE_CUDA_ERR_NOT_BUILT;
}
forge_cuda_status forge_cuda_timer_elapsed(void* timer, float* ms) {
  (void)timer;
  (void)ms;
  return FORGE_CUDA_ERR_NOT_BUILT;
}
forge_cuda_status forge_cuda_timer_destroy(void* timer) {
  (void)timer;
  return FORGE_CUDA_ERR_NOT_BUILT;
}

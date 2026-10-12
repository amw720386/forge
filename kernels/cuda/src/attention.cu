#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <cstdlib>
#include <cstring>

namespace {

__global__ void gqa_attn_kernel(size_t t_q, size_t t_k, size_t n_heads,
                                size_t n_kv_heads, size_t head_dim, float scale,
                                const float* q, size_t q_w, const float* k,
                                size_t kv_w, const float* v, float* ctx,
                                int causal, size_t causal_q_offset) {
  size_t h = blockIdx.x;
  size_t qi = blockIdx.y;
  if (h >= n_heads || qi >= t_q || threadIdx.x != 0) {
    return;
  }
  const size_t d = head_dim;
  const size_t reps = n_heads / n_kv_heads;
  const size_t kv_h = h / reps;
  const float* qrow = q + qi * q_w + h * d;
  float* out = ctx + qi * q_w + h * d;

  size_t valid_k = t_k;
  if (causal) {
    size_t qpos = causal_q_offset + qi;
    valid_k = qpos + 1 < t_k ? qpos + 1 : t_k;
  }

  float vmax = -FLT_MAX;
  for (size_t kj = 0; kj < valid_k; ++kj) {
    const float* krow = k + kj * kv_w + kv_h * d;
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      dot += qrow[i] * krow[i];
    }
    float s = dot * scale;
    if (s > vmax) {
      vmax = s;
    }
  }

  float sum = 0.0f;
  for (size_t kj = 0; kj < valid_k; ++kj) {
    const float* krow = k + kj * kv_w + kv_h * d;
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      dot += qrow[i] * krow[i];
    }
    float e = expf(dot * scale - vmax);
    sum += e;
  }
  float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

  for (size_t i = 0; i < d; ++i) {
    float acc = 0.0f;
    for (size_t kj = 0; kj < valid_k; ++kj) {
      const float* krow = k + kj * kv_w + kv_h * d;
      const float* vrow = v + kj * kv_w + kv_h * d;
      float dot = 0.0f;
      for (size_t j = 0; j < d; ++j) {
        dot += qrow[j] * krow[j];
      }
      float p = expf(dot * scale - vmax) * inv;
      acc += p * vrow[i];
    }
    out[i] = acc;
  }
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_gqa_attn_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k, size_t kv_w,
    const float* v, float* ctx, int causal, size_t causal_q_offset) {
  if (q == nullptr || k == nullptr || v == nullptr || ctx == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (t_q == 0 || t_k == 0 || n_heads == 0 || n_kv_heads == 0 ||
      head_dim == 0 || n_heads % n_kv_heads != 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (q_w < n_heads * head_dim || kv_w < n_kv_heads * head_dim) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  dim3 grid(static_cast<unsigned>(n_heads), static_cast<unsigned>(t_q));
  gqa_attn_kernel<<<grid, 1>>>(t_q, t_k, n_heads, n_kv_heads, head_dim, scale,
                               q, q_w, k, kv_w, v, ctx, causal,
                               causal_q_offset);
  return forge::cuda_detail::launch_check("gqa_attn_kernel");
}

namespace {

__device__ inline const float* paged_kv_row(const float* pool, size_t layer,
                                            size_t num_blocks,
                                            size_t block_size, size_t kv_w,
                                            uint32_t physical, size_t row,
                                            size_t kv_h, size_t head_dim) {
  size_t base =
      ((layer * num_blocks + static_cast<size_t>(physical)) * block_size +
       row) *
      kv_w;
  return pool + base + kv_h * head_dim;
}

/** reference: 1 thread, three-pass, per-key block lookup. */
__global__ void paged_gqa_attn_naive_kernel(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t block_size,
    size_t num_blocks, size_t layer, size_t kv_w, float* ctx, int causal,
    size_t causal_q_offset) {
  size_t h = blockIdx.x;
  size_t qi = blockIdx.y;
  if (h >= n_heads || qi >= t_q || threadIdx.x != 0) {
    return;
  }
  const size_t d = head_dim;
  const size_t reps = n_heads / n_kv_heads;
  const size_t kv_h = h / reps;
  const float* qrow = q + qi * q_w + h * d;
  float* out = ctx + qi * q_w + h * d;

  size_t valid_k = t_k;
  if (causal) {
    size_t qpos = causal_q_offset + qi;
    valid_k = qpos + 1 < t_k ? qpos + 1 : t_k;
  }

  float vmax = -FLT_MAX;
  for (size_t kj = 0; kj < valid_k; ++kj) {
    size_t logical_block = kj / block_size;
    size_t row = kj % block_size;
    uint32_t phys = block_table[logical_block];
    const float* krow = paged_kv_row(k_pool, layer, num_blocks, block_size,
                                     kv_w, phys, row, kv_h, d);
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      dot += qrow[i] * krow[i];
    }
    float s = dot * scale;
    if (s > vmax) {
      vmax = s;
    }
  }

  float sum = 0.0f;
  for (size_t kj = 0; kj < valid_k; ++kj) {
    size_t logical_block = kj / block_size;
    size_t row = kj % block_size;
    uint32_t phys = block_table[logical_block];
    const float* krow = paged_kv_row(k_pool, layer, num_blocks, block_size,
                                     kv_w, phys, row, kv_h, d);
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      dot += qrow[i] * krow[i];
    }
    float e = expf(dot * scale - vmax);
    sum += e;
  }
  float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

  for (size_t i = 0; i < d; ++i) {
    float acc = 0.0f;
    for (size_t kj = 0; kj < valid_k; ++kj) {
      size_t logical_block = kj / block_size;
      size_t row = kj % block_size;
      uint32_t phys = block_table[logical_block];
      const float* krow = paged_kv_row(k_pool, layer, num_blocks, block_size,
                                       kv_w, phys, row, kv_h, d);
      const float* vrow = paged_kv_row(v_pool, layer, num_blocks, block_size,
                                       kv_w, phys, row, kv_h, d);
      float dot = 0.0f;
      for (size_t j = 0; j < d; ++j) {
        dot += qrow[j] * krow[j];
      }
      float p = expf(dot * scale - vmax) * inv;
      acc += p * vrow[i];
    }
    out[i] = acc;
  }
}

__device__ inline float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

/**
 * optimized paged GQA:
 * - one CUDA block (1 warp) per (head, query)
 * - online softmax (single K/V pass)
 * - physical block id loaded once per logical block
 * - threads partition head_dim for QÂ·K / V accumulation
 */
__global__ void paged_gqa_attn_opt_kernel(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t block_size,
    size_t num_blocks, size_t layer, size_t kv_w, float* ctx, int causal,
    size_t causal_q_offset) {
  const size_t h = blockIdx.x;
  const size_t qi = blockIdx.y;
  if (h >= n_heads || qi >= t_q) {
    return;
  }

  const unsigned lane = threadIdx.x;  // 0..31
  const size_t d = head_dim;
  const size_t reps = n_heads / n_kv_heads;
  const size_t kv_h = h / reps;
  const float* qrow = q + qi * q_w + h * d;
  float* out = ctx + qi * q_w + h * d;

  size_t valid_k = t_k;
  if (causal) {
    size_t qpos = causal_q_offset + qi;
    valid_k = qpos + 1 < t_k ? qpos + 1 : t_k;
  }
  if (valid_k == 0) {
    for (size_t i = lane; i < d; i += 32) {
      out[i] = 0.0f;
    }
    return;
  }

  // Online softmax state (lane 0 owns scalars; all lanes own partial V acc).
  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  // Per-thread accumulator covering head_dim / 32 (+ remainder via stride).
  float acc[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    acc[t] = 0.0f;
  }

  const size_t num_logical = (valid_k + block_size - 1) / block_size;
  for (size_t lb = 0; lb < num_logical; ++lb) {
    const uint32_t phys = block_table[lb];
    const size_t row0 = lb * block_size;
    const size_t rows_here =
        valid_k - row0 < block_size ? valid_k - row0 : block_size;
    const size_t pool_base =
        ((layer * num_blocks + static_cast<size_t>(phys)) * block_size) * kv_w;

    for (size_t r = 0; r < rows_here; ++r) {
      const float* krow = k_pool + pool_base + r * kv_w + kv_h * d;
      const float* vrow = v_pool + pool_base + r * kv_w + kv_h * d;

      float partial = 0.0f;
      for (size_t i = lane; i < d; i += 32) {
        partial += qrow[i] * krow[i];
      }
      float dot = warp_reduce_sum(partial);
      // Broadcast score from lane 0.
      float s = __shfl_sync(0xffffffffu, dot, 0) * scale;

      float m_new = fmaxf(m_i, s);
      float alpha = expf(m_i - m_new);
      float p = expf(s - m_new);
      // Rescale prior accumulator; all lanes apply same alpha.
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        size_t i = lane + static_cast<size_t>(t) * 32;
        if (i < d) {
          acc[t] = acc[t] * alpha + p * vrow[i];
        }
      }
      l_i = l_i * alpha + p;
      m_i = m_new;
    }
  }

  float inv = l_i > 0.0f ? 1.0f / l_i : 0.0f;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    size_t i = lane + static_cast<size_t>(t) * 32;
    if (i < d) {
      out[i] = acc[t] * inv;
    }
  }
}

__device__ __forceinline__ void batch_paged_attn_reference(
    size_t n_heads, size_t n_kv_heads, size_t head_dim, float scale,
    const uint64_t* metadata, const float* q, const float* k_pool,
    const float* v_pool, float* ctx, size_t block_size, size_t num_blocks,
    size_t layer) {
  const size_t h = blockIdx.x;
  const size_t row = blockIdx.y;
  const size_t qi = 0;
  const size_t t_q = 1;
  const size_t t_k = metadata[row * 3 + 2] + 1;
  const size_t q_w = n_heads * head_dim;
  const size_t kv_w = n_kv_heads * head_dim;
  const auto* block_table =
      reinterpret_cast<const uint32_t*>(metadata[row * 3]);
  q += row * q_w;
  ctx += row * q_w;
  const int causal = 0;
  const size_t causal_q_offset = 0;
  if (h >= n_heads || qi >= t_q) {
    return;
  }

  const unsigned lane = threadIdx.x % 32;  // 0..31
  const size_t d = head_dim;
  const size_t reps = n_heads / n_kv_heads;
  const size_t kv_h = h / reps;
  const float* qrow = q + qi * q_w + h * d;
  float* out = ctx + qi * q_w + h * d;

  size_t valid_k = t_k;
  if (causal) {
    size_t qpos = causal_q_offset + qi;
    valid_k = qpos + 1 < t_k ? qpos + 1 : t_k;
  }
  if (valid_k == 0) {
    for (size_t i = lane; i < d; i += 32) {
      out[i] = 0.0f;
    }
    return;
  }

  // Online softmax state (lane 0 owns scalars; all lanes own partial V acc).
  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  // Per-thread accumulator covering head_dim / 32 (+ remainder via stride).
  float acc[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    acc[t] = 0.0f;
  }

  const size_t num_logical = (valid_k + block_size - 1) / block_size;
  for (size_t lb = 0; lb < num_logical; ++lb) {
    const uint32_t phys = block_table[lb];
    const size_t row0 = lb * block_size;
    const size_t rows_here =
        valid_k - row0 < block_size ? valid_k - row0 : block_size;
    const size_t pool_base =
        ((layer * num_blocks + static_cast<size_t>(phys)) * block_size) * kv_w;

    for (size_t r = 0; r < rows_here; ++r) {
      const float* krow = k_pool + pool_base + r * kv_w + kv_h * d;
      const float* vrow = v_pool + pool_base + r * kv_w + kv_h * d;

      float partial = 0.0f;
      for (size_t i = lane; i < d; i += 32) {
        partial += qrow[i] * krow[i];
      }
      float dot = warp_reduce_sum(partial);
      // Broadcast score from lane 0.
      float s = __shfl_sync(0xffffffffu, dot, 0) * scale;

      float m_new = fmaxf(m_i, s);
      float alpha = expf(m_i - m_new);
      float p = expf(s - m_new);
      // Rescale prior accumulator; all lanes apply same alpha.
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        size_t i = lane + static_cast<size_t>(t) * 32;
        if (i < d) {
          acc[t] = acc[t] * alpha + p * vrow[i];
        }
      }
      l_i = l_i * alpha + p;
      m_i = m_new;
    }
  }

  float inv = l_i > 0.0f ? 1.0f / l_i : 0.0f;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    size_t i = lane + static_cast<size_t>(t) * 32;
    if (i < d) {
      out[i] = acc[t] * inv;
    }
  }
}

// Independent online-softmax partitions across four warps for prefill queries.
// FP32 states combine with a stable maximum; no score matrix is materialized.
__global__ void prefill_paged_attn_kernel(
    size_t t_k, size_t heads, size_t kv_heads, size_t d, float scale,
    const float* q, size_t qw, const float* kp, const float* vp,
    const uint32_t* table, size_t bs, size_t blocks, size_t layer, size_t kw,
    float* out, int causal, size_t offset, const uint64_t* metadata = nullptr) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  // Short raw-kernel calls retain reference arithmetic. The transformer host
  // dispatch also preserves the reference for long BF16 batches above two.
  if (metadata && metadata[blockIdx.y * 3 + 2] + 1 < 128) {
    if (warp == 0)
      batch_paged_attn_reference(heads, kv_heads, d, scale, metadata, q, kp, vp,
                                 out, bs, blocks, layer);
    return;
  }

  const size_t h = blockIdx.x, qi = metadata ? 0 : blockIdx.y,
               kh = h / (heads / kv_heads);
  if (metadata) {
    const size_t row = blockIdx.y;
    table = reinterpret_cast<const uint32_t*>(metadata[row * 3]);
    t_k = metadata[row * 3 + 2] + 1;
    q += row * qw;
    out += row * qw;
  }
  const size_t valid = causal ? min(t_k, offset + qi + 1) : t_k;
  __shared__ float maxima[4], sums[4], values[4][128];
  float qv[4] = {}, acc[4] = {};
#pragma unroll
  for (int r = 0; r < 4; r++) {
    size_t j = lane + r * 32;
    if (j < d) qv[r] = q[qi * qw + h * d + j];
  }
  float mx = -FLT_MAX, den = 0;
  // Preserve the serial reduction at short contexts inside the dynamic kernel.
  // Metadata lengths are read on device, so replay stays valid after KV growth.
  const size_t partitions = metadata && valid < 128 ? 1 : 4;
  for (size_t lb = warp; warp < partitions && lb < (valid + bs - 1) / bs;
       lb += partitions) {
    const size_t start = lb * bs, rows = min(bs, valid - start);
    const size_t base =
        ((layer * blocks + (size_t)table[lb]) * bs) * kw + kh * d;
    for (size_t r = 0; r < rows; r++) {
      float partial = 0;
#pragma unroll
      for (int j = 0; j < 4; j++)
        if ((size_t)(lane + j * 32) < d)
          partial += qv[j] * kp[base + r * kw + lane + j * 32];
      float score =
          __shfl_sync(0xffffffffu, warp_reduce_sum(partial), 0) * scale;
      float next = 0, alpha = 0, p = 0;
      if (lane == 0) {
        next = fmaxf(mx, score);
        alpha = expf(mx - next);
        p = expf(score - next);
      }
      next = __shfl_sync(0xffffffffu, next, 0);
      alpha = __shfl_sync(0xffffffffu, alpha, 0);
      p = __shfl_sync(0xffffffffu, p, 0);
#pragma unroll
      for (int j = 0; j < 4; j++)
        if ((size_t)(lane + j * 32) < d)
          acc[j] = acc[j] * alpha + p * vp[base + r * kw + lane + j * 32];
      den = den * alpha + p;
      mx = next;
    }
  }
  if (lane == 0) {
    maxima[warp] = mx;
    sums[warp] = den;
  }
#pragma unroll
  for (int j = 0; j < 4; j++)
    if ((size_t)(lane + j * 32) < d) values[warp][lane + j * 32] = acc[j];
  __syncthreads();
  if (warp == 0) {
    float mx_all =
        fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
    float factors[4], total = 0;
    for (int w = 0; w < 4; w++) {
      factors[w] = sums[w] > 0 ? expf(maxima[w] - mx_all) : 0;
      total += sums[w] * factors[w];
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
      if ((size_t)(lane + j * 32) < d) {
        float value = 0;
        for (int w = 0; w < 4; w++)
          value += values[w][lane + j * 32] * factors[w];
        out[qi * qw + h * d + lane + j * 32] = total > 0 ? value / total : 0;
      }
  }
}

__global__ void batch_paged_attn_kernel(size_t n_heads, size_t n_kv_heads,
                                        size_t head_dim, float scale,
                                        const uint64_t* metadata,
                                        const float* q, const float* k_pool,
                                        const float* v_pool, float* ctx,
                                        size_t block_size, size_t num_blocks,
                                        size_t layer) {
  const size_t h = blockIdx.x;
  const size_t row = blockIdx.y;
  const size_t qi = 0;
  const size_t t_q = 1;
  const size_t t_k = metadata[row * 3 + 2] + 1;
  const size_t q_w = n_heads * head_dim;
  const size_t kv_w = n_kv_heads * head_dim;
  const auto* block_table =
      reinterpret_cast<const uint32_t*>(metadata[row * 3]);
  q += row * q_w;
  ctx += row * q_w;
  const int causal = 0;
  const size_t causal_q_offset = 0;
  if (h >= n_heads || qi >= t_q) {
    return;
  }

  const unsigned lane = threadIdx.x;  // 0..31
  const size_t d = head_dim;
  const size_t reps = n_heads / n_kv_heads;
  const size_t kv_h = h / reps;
  const float* qrow = q + qi * q_w + h * d;
  float* out = ctx + qi * q_w + h * d;

  size_t valid_k = t_k;
  if (causal) {
    size_t qpos = causal_q_offset + qi;
    valid_k = qpos + 1 < t_k ? qpos + 1 : t_k;
  }
  if (valid_k == 0) {
    for (size_t i = lane; i < d; i += 32) {
      out[i] = 0.0f;
    }
    return;
  }

  // Online softmax state (lane 0 owns scalars; all lanes own partial V acc).
  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  // Per-thread accumulator covering head_dim / 32 (+ remainder via stride).
  float acc[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    acc[t] = 0.0f;
  }

  const size_t num_logical = (valid_k + block_size - 1) / block_size;
  for (size_t lb = 0; lb < num_logical; ++lb) {
    const uint32_t phys = block_table[lb];
    const size_t row0 = lb * block_size;
    const size_t rows_here =
        valid_k - row0 < block_size ? valid_k - row0 : block_size;
    const size_t pool_base =
        ((layer * num_blocks + static_cast<size_t>(phys)) * block_size) * kv_w;

    for (size_t r = 0; r < rows_here; ++r) {
      const float* krow = k_pool + pool_base + r * kv_w + kv_h * d;
      const float* vrow = v_pool + pool_base + r * kv_w + kv_h * d;

      float partial = 0.0f;
      for (size_t i = lane; i < d; i += 32) {
        partial += qrow[i] * krow[i];
      }
      float dot = warp_reduce_sum(partial);
      // Broadcast score from lane 0.
      float s = __shfl_sync(0xffffffffu, dot, 0) * scale;

      float m_new = fmaxf(m_i, s);
      float alpha = expf(m_i - m_new);
      float p = expf(s - m_new);
      // Rescale prior accumulator; all lanes apply same alpha.
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        size_t i = lane + static_cast<size_t>(t) * 32;
        if (i < d) {
          acc[t] = acc[t] * alpha + p * vrow[i];
        }
      }
      l_i = l_i * alpha + p;
      m_i = m_new;
    }
  }

  float inv = l_i > 0.0f ? 1.0f / l_i : 0.0f;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    size_t i = lane + static_cast<size_t>(t) * 32;
    if (i < d) {
      out[i] = acc[t] * inv;
    }
  }
}

__global__ void batch_paged_prepare_kernel(
    size_t heads, size_t kv_heads, size_t d, float theta,
    const uint64_t* metadata, float* q, const float* k, const float* v,
    float* kp, float* vp, size_t block_size, size_t blocks, size_t layer) {
  size_t row = blockIdx.x;
  size_t pos = metadata[row * 3 + 1], len = metadata[row * 3 + 2];
  const auto* table = reinterpret_cast<const uint32_t*>(metadata[row * 3]);
  size_t qw = heads * d, kvw = kv_heads * d;
  size_t dest = ((layer * blocks + table[len / block_size]) * block_size +
                 len % block_size) *
                kvw;
  // Each lane owns a split-half RoPE pair, matching the eager kernel.
  for (size_t j = threadIdx.x; j < heads * (d / 2); j += blockDim.x) {
    size_t h = j / (d / 2), i = j % (d / 2);
    float angle = static_cast<float>(pos) *
                  powf(theta, -2.0f * static_cast<float>(i) / d);
    float sn, cs;
    sincosf(angle, &sn, &cs);
    size_t a = row * qw + h * d + i, b = a + d / 2;
    float x = q[a], y = q[b];
    q[a] = x * cs - y * sn;
    q[b] = x * sn + y * cs;
  }
  for (size_t j = threadIdx.x; j < kv_heads * (d / 2); j += blockDim.x) {
    size_t h = j / (d / 2), i = j % (d / 2);
    float angle = static_cast<float>(pos) *
                  powf(theta, -2.0f * static_cast<float>(i) / d);
    float sn, cs;
    sincosf(angle, &sn, &cs);
    size_t a = h * d + i, b = a + d / 2;
    float x = k[row * kvw + a], y = k[row * kvw + b];
    kp[dest + a] = x * cs - y * sn;
    kp[dest + b] = x * sn + y * cs;
    vp[dest + a] = v[row * kvw + a];
    vp[dest + b] = v[row * kvw + b];
  }
}

forge_cuda_status validate_paged_args(size_t t_q, size_t t_k, size_t n_heads,
                                      size_t n_kv_heads, size_t head_dim,
                                      const float* q, const float* k_pool,
                                      const float* v_pool,
                                      const uint32_t* block_table, float* ctx,
                                      size_t block_size, size_t num_blocks,
                                      size_t n_layers, size_t layer, size_t q_w,
                                      size_t kv_w, size_t num_logical_blocks) {
  if (q == nullptr || k_pool == nullptr || v_pool == nullptr ||
      block_table == nullptr || ctx == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (t_q == 0 || t_k == 0 || n_heads == 0 || n_kv_heads == 0 ||
      head_dim == 0 || block_size == 0 || num_blocks == 0 || n_layers == 0 ||
      n_heads % n_kv_heads != 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (layer >= n_layers) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (q_w < n_heads * head_dim || kv_w < n_kv_heads * head_dim) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t need_blocks = (t_k + block_size - 1) / block_size;
  if (need_blocks > num_logical_blocks) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  // Opt kernel packs up to 4Ã—32 = 128 dims per thread; refuse larger.
  if (head_dim > 128) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  return FORGE_CUDA_OK;
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_paged_gqa_attn_naive_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t num_logical_blocks,
    size_t block_size, size_t num_blocks, size_t n_layers, size_t layer,
    size_t kv_w, float* ctx, int causal, size_t causal_q_offset) {
  forge_cuda_status st =
      validate_paged_args(t_q, t_k, n_heads, n_kv_heads, head_dim, q, k_pool,
                          v_pool, block_table, ctx, block_size, num_blocks,
                          n_layers, layer, q_w, kv_w, num_logical_blocks);
  if (st != FORGE_CUDA_OK) {
    return st;
  }
  dim3 grid(static_cast<unsigned>(n_heads), static_cast<unsigned>(t_q));
  paged_gqa_attn_naive_kernel<<<grid, 1>>>(
      t_q, t_k, n_heads, n_kv_heads, head_dim, scale, q, q_w, k_pool, v_pool,
      block_table, block_size, num_blocks, layer, kv_w, ctx, causal,
      causal_q_offset);
  return forge::cuda_detail::launch_check("paged_gqa_attn_naive_kernel");
}

extern "C" forge_cuda_status forge_cuda_paged_gqa_attn_f32(
    size_t t_q, size_t t_k, size_t n_heads, size_t n_kv_heads, size_t head_dim,
    float scale, const float* q, size_t q_w, const float* k_pool,
    const float* v_pool, const uint32_t* block_table, size_t num_logical_blocks,
    size_t block_size, size_t num_blocks, size_t n_layers, size_t layer,
    size_t kv_w, float* ctx, int causal, size_t causal_q_offset) {
  forge::cuda_detail::ProfileRegion profile("prefill_attention");
  forge_cuda_status st =
      validate_paged_args(t_q, t_k, n_heads, n_kv_heads, head_dim, q, k_pool,
                          v_pool, block_table, ctx, block_size, num_blocks,
                          n_layers, layer, q_w, kv_w, num_logical_blocks);
  if (st != FORGE_CUDA_OK) {
    return st;
  }
  dim3 grid(static_cast<unsigned>(n_heads), static_cast<unsigned>(t_q));
  // Default: optimized warp + online softmax. Env FORGE_PAGED_ATTN_NAIVE=1
  // selects the reference (checked on host in Rust; here always opt
  // unless compile-time â€” Rust switches symbol). Keep naive entry separate.
  const char* prefill_flag = std::getenv("FORGE_PREFILL_ATTN_BLOCK");
  if (t_q > 1 && t_k >= 128 && head_dim <= 128 && prefill_flag &&
      !std::strcmp(prefill_flag, "1")) {
    prefill_paged_attn_kernel<<<grid, 128>>>(
        t_k, n_heads, n_kv_heads, head_dim, scale, q, q_w, k_pool, v_pool,
        block_table, block_size, num_blocks, layer, kv_w, ctx, causal,
        causal_q_offset);
    return forge::cuda_detail::launch_check("prefill_paged_attn_kernel");
  }
  paged_gqa_attn_opt_kernel<<<grid, 32>>>(
      t_q, t_k, n_heads, n_kv_heads, head_dim, scale, q, q_w, k_pool, v_pool,
      block_table, block_size, num_blocks, layer, kv_w, ctx, causal,
      causal_q_offset);
  return forge::cuda_detail::launch_check("paged_gqa_attn_opt_kernel");
}

extern "C" forge_cuda_status forge_cuda_batch_paged_prepare_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float theta,
    const uint64_t* metadata, float* q, const float* k, const float* v,
    float* kp, float* vp, size_t bs, size_t blocks, size_t layer) {
  forge::cuda_detail::ProfileRegion profile("rope_kv_append");
  if (!batch || batch > 8 || !heads || !kv_heads || heads % kv_heads || !dim ||
      dim > 128 || dim % 2 || theta <= 0 || !bs || !blocks || !metadata || !q ||
      !k || !v || !kp || !vp)
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  batch_paged_prepare_kernel<<<batch, 256>>>(heads, kv_heads, dim, theta,
                                             metadata, q, k, v, kp, vp, bs,
                                             blocks, layer);
  return forge::cuda_detail::launch_check("batch_paged_prepare_kernel");
}
extern "C" forge_cuda_status forge_cuda_batch_paged_attn_dispatch_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float scale,
    const uint64_t* metadata, const float* q, const float* kp, const float* vp,
    float* ctx, size_t bs, size_t blocks, size_t layer, int partitioned) {
  forge::cuda_detail::ProfileRegion profile("decode_attention");
  if (!batch || batch > 8 || !heads || !kv_heads || heads % kv_heads || !dim ||
      dim > 128 || !bs || !blocks || !metadata || !q || !kp || !vp || !ctx)
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  dim3 grid(heads, batch);
  if (partitioned) {
    prefill_paged_attn_kernel<<<grid, 128>>>(
        0, heads, kv_heads, dim, scale, q, heads * dim, kp, vp, nullptr, bs,
        blocks, layer, kv_heads * dim, ctx, 0, 0, metadata);
    return forge::cuda_detail::launch_check("partitioned_decode_attn_kernel");
  }
  batch_paged_attn_kernel<<<grid, 32>>>(heads, kv_heads, dim, scale, metadata,
                                        q, kp, vp, ctx, bs, blocks, layer);
  return forge::cuda_detail::launch_check("batch_paged_attn_kernel");
}

extern "C" forge_cuda_status forge_cuda_batch_paged_attn_f32(
    size_t batch, size_t heads, size_t kv_heads, size_t dim, float scale,
    const uint64_t* metadata, const float* q, const float* kp, const float* vp,
    float* ctx, size_t bs, size_t blocks, size_t layer) {
  const char* flag = std::getenv("FORGE_DECODE_ATTN_BLOCK");
  return forge_cuda_batch_paged_attn_dispatch_f32(
      batch, heads, kv_heads, dim, scale, metadata, q, kp, vp, ctx, bs, blocks,
      layer, !flag || std::strcmp(flag, "0"));
}

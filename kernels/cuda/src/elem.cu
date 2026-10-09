#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>

namespace {

__global__ void copy_rows_kernel(size_t rows, size_t cols, const float* src,
                                 float* dst) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = rows * cols;
  if (idx >= n) {
    return;
  }
  dst[idx] = src[idx];
}

__global__ void add_rows_kernel(size_t rows, size_t cols, const float* a,
                                const float* b, float* y) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = rows * cols;
  if (idx >= n) {
    return;
  }
  y[idx] = a[idx] + b[idx];
}

__global__ void kv_append_kernel(size_t width, size_t t, size_t dst_offset,
                                 const float* k_src, const float* v_src,
                                 float* k_dst, float* v_dst) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = t * width;
  if (idx >= n) {
    return;
  }
  k_dst[dst_offset + idx] = k_src[idx];
  v_dst[dst_offset + idx] = v_src[idx];
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_copy_rows_f32(size_t rows, size_t cols,
                                                     const float* src,
                                                     float* dst) {
  if (src == nullptr || dst == nullptr || rows == 0 || cols == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t n = rows * cols;
  copy_rows_kernel<<<(static_cast<unsigned>(n) + 255) / 256, 256>>>(rows, cols,
                                                                      src, dst);
  return forge::cuda_detail::launch_check("copy_rows_kernel");
}

extern "C" forge_cuda_status forge_cuda_add_rows_f32(size_t rows, size_t cols,
                                                    const float* a,
                                                    const float* b, float* y) {
  if (a == nullptr || b == nullptr || y == nullptr || rows == 0 || cols == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t n = rows * cols;
  add_rows_kernel<<<(static_cast<unsigned>(n) + 255) / 256, 256>>>(rows, cols,
                                                                    a, b, y);
  return forge::cuda_detail::launch_check("add_rows_kernel");
}

extern "C" forge_cuda_status forge_cuda_kv_append_f32(
    size_t width, size_t t, size_t dst_row_offset, const float* k_src,
    const float* v_src, float* k_dst, float* v_dst) {
  if (k_src == nullptr || v_src == nullptr || k_dst == nullptr ||
      v_dst == nullptr || t == 0 || width == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t n = t * width;
  size_t off = dst_row_offset * width;
  kv_append_kernel<<<(static_cast<unsigned>(n) + 255) / 256, 256>>>(
      width, t, off, k_src, v_src, k_dst, v_dst);
  return forge::cuda_detail::launch_check("kv_append_kernel");
}

namespace {

__global__ void paged_kv_append_kernel(
    size_t width, size_t t, size_t dst_token_offset, size_t block_size,
    size_t num_blocks, size_t layer, const uint32_t* block_table,
    const float* k_src, const float* v_src, float* k_pool, float* v_pool) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = t * width;
  if (idx >= n) {
    return;
  }
  size_t tok = idx / width;
  size_t col = idx % width;
  size_t logical = dst_token_offset + tok;
  size_t logical_block = logical / block_size;
  size_t row = logical % block_size;
  uint32_t phys = block_table[logical_block];
  size_t dst =
      ((layer * num_blocks + static_cast<size_t>(phys)) * block_size + row) *
          width +
      col;
  k_pool[dst] = k_src[idx];
  v_pool[dst] = v_src[idx];
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_paged_kv_append_f32(
    size_t width, size_t t, size_t dst_token_offset, size_t block_size,
    size_t num_blocks, size_t n_layers, size_t layer,
    const uint32_t* block_table, size_t num_logical_blocks, const float* k_src,
    const float* v_src, float* k_pool, float* v_pool) {
  forge::cuda_detail::ProfileRegion profile("kv_append");
  if (k_src == nullptr || v_src == nullptr || k_pool == nullptr ||
      v_pool == nullptr || block_table == nullptr || t == 0 || width == 0 ||
      block_size == 0 || num_blocks == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (layer >= n_layers) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t end = dst_token_offset + t;
  size_t need_blocks = (end + block_size - 1) / block_size;
  if (need_blocks > num_logical_blocks) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t n = t * width;
  paged_kv_append_kernel<<<(static_cast<unsigned>(n) + 255) / 256, 256>>>(
      width, t, dst_token_offset, block_size, num_blocks, layer, block_table,
      k_src, v_src, k_pool, v_pool);
  return forge::cuda_detail::launch_check("paged_kv_append_kernel");
}

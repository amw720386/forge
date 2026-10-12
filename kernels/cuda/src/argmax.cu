#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>
#include <float.h>

namespace {

/**
 * Row-wise argmax over [rows, cols] row-major float logits.
 * Tie-break: lowest index (matches CPU greedy).
 * One CUDA block per row; blockDim.x threads reduce via shared memory.
 */
__global__ void argmax_rows_kernel(size_t rows, size_t cols,
                                   const float* logits, int32_t* out_ids) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* row_ptr = logits + row * cols;
  extern __shared__ char smem[];
  float* s_val = reinterpret_cast<float*>(smem);
  int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

  float best_v = -FLT_MAX;
  int best_i = 0;
  for (size_t j = threadIdx.x; j < cols; j += blockDim.x) {
    float v = row_ptr[j];
    int i = static_cast<int>(j);
    // Strict > preserves lowest index on ties.
    if (v > best_v || (v == best_v && i < best_i)) {
      best_v = v;
      best_i = i;
    }
  }
  s_val[threadIdx.x] = best_v;
  s_idx[threadIdx.x] = best_i;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float ov = s_val[threadIdx.x + stride];
      int oi = s_idx[threadIdx.x + stride];
      if (ov > s_val[threadIdx.x] ||
          (ov == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_ids[row] = s_idx[0];
  }
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_argmax_rows_f32(size_t rows,
                                                        size_t cols,
                                                        const float* logits,
                                                        int32_t* out_ids) {
  if (logits == nullptr || out_ids == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (rows == 0 || cols == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  const unsigned threads = 256;
  size_t smem = threads * (sizeof(float) + sizeof(int));
  argmax_rows_kernel<<<static_cast<unsigned>(rows), threads, smem>>>(
      rows, cols, logits, out_ids);
  return forge::cuda_detail::launch_check("argmax_rows_kernel");
}

#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

namespace {

__global__ void softmax_row_kernel(size_t batch, size_t rows, size_t cols,
                                   float* scores, int causal,
                                   size_t causal_q_offset) {
  size_t flat = blockIdx.x;
  size_t total_rows = batch * rows;
  if (flat >= total_rows) {
    return;
  }
  size_t b = flat / rows;
  size_t r = flat % rows;
  float* row = scores + (b * rows + r) * cols;

  size_t valid = cols;
  if (causal) {
    size_t q = causal_q_offset + r;
    valid = q + 1 < cols ? q + 1 : cols;
  }

  __shared__ float red[256];
  float vmax = -FLT_MAX;
  for (size_t i = threadIdx.x; i < valid; i += blockDim.x) {
    vmax = fmaxf(vmax, row[i]);
  }
  red[threadIdx.x] = vmax;
  __syncthreads();
  for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) {
      red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    }
    __syncthreads();
  }
  vmax = red[0];

  float sum = 0.0f;
  for (size_t i = threadIdx.x; i < valid; i += blockDim.x) {
    float e = expf(row[i] - vmax);
    row[i] = e;
    sum += e;
  }
  red[threadIdx.x] = sum;
  __syncthreads();
  for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) {
      red[threadIdx.x] += red[threadIdx.x + s];
    }
    __syncthreads();
  }
  sum = red[0];
  float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (size_t i = threadIdx.x; i < valid; i += blockDim.x) {
    row[i] *= inv;
  }
  for (size_t i = valid + threadIdx.x; i < cols; i += blockDim.x) {
    row[i] = 0.0f;
  }
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_softmax_f32(size_t batch, size_t rows,
                                                    size_t cols, float* scores,
                                                    int causal,
                                                    size_t causal_q_offset) {
  if (scores == nullptr || batch == 0 || rows == 0 || cols == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t total = batch * rows;
  const int block = 256;
  softmax_row_kernel<<<static_cast<int>(total), block>>>(
      batch, rows, cols, scores, causal, causal_q_offset);
  return forge::cuda_detail::from_cuda(cudaGetLastError(), "softmax_kernel");
}

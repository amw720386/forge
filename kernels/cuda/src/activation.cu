#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {

__global__ void silu_kernel(size_t n, const float* x, float* y) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

__global__ void mul_kernel(size_t n, const float* a, const float* b, float* y) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] * b[i];
  }
}

__global__ void add_kernel(size_t n, const float* a, const float* b, float* y) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] + b[i];
  }
}

inline int grid_for(size_t n, int block) {
  return static_cast<int>((n + static_cast<size_t>(block) - 1) /
                          static_cast<size_t>(block));
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_silu_f32(size_t n, const float* x,
                                                 float* y) {
  if (x == nullptr || y == nullptr || n == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  const int block = 256;
  silu_kernel<<<grid_for(n, block), block>>>(n, x, y);
  return forge::cuda_detail::from_cuda(cudaGetLastError(), "silu_kernel");
}

extern "C" forge_cuda_status forge_cuda_mul_f32(size_t n, const float* a,
                                                const float* b, float* y) {
  if (a == nullptr || b == nullptr || y == nullptr || n == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  const int block = 256;
  mul_kernel<<<grid_for(n, block), block>>>(n, a, b, y);
  return forge::cuda_detail::from_cuda(cudaGetLastError(), "mul_kernel");
}

extern "C" forge_cuda_status forge_cuda_add_f32(size_t n, const float* a,
                                                const float* b, float* y) {
  if (a == nullptr || b == nullptr || y == nullptr || n == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  const int block = 256;
  add_kernel<<<grid_for(n, block), block>>>(n, a, b, y);
  return forge::cuda_detail::from_cuda(cudaGetLastError(), "add_kernel");
}

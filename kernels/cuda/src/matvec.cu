#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>

namespace {

__global__ void matvec_kernel(size_t m, size_t n, const float* a, size_t lda,
                              const float* x, float* y) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) {
    return;
  }
  const float* arow = a + row * lda;
  float sum = 0.0f;
  for (size_t j = 0; j < n; ++j) {
    sum += arow[j] * x[j];
  }
  y[row] = sum;
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_matvec_f32(size_t m, size_t n,
                                                   const float* a, size_t lda,
                                                   const float* x, float* y) {
  if (a == nullptr || x == nullptr || y == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || lda < n) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  const int block = 256;
  const int grid = static_cast<int>((m + block - 1) / block);
  matvec_kernel<<<grid, block>>>(m, n, a, lda, x, y);
  return forge::cuda_detail::from_cuda(cudaGetLastError(), "matvec_kernel");
}

#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {

__global__ void rmsnorm_kernel(size_t rows, size_t dim, const float* x,
                               size_t ldx, const float* weight, float eps,
                               float* y, size_t ldy) {
  size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* xr = x + row * ldx;
  float* yr = y + row * ldy;

  // Parallel reduction for mean square within the block.
  __shared__ float partial[256];
  float local = 0.0f;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    float v = xr[i];
    local += v * v;
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0;
       stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float inv_rms = rsqrtf(partial[0] / static_cast<float>(dim) + eps);
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    yr[i] = xr[i] * inv_rms * weight[i];
  }
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_rmsnorm_f32(size_t rows, size_t dim,
                                                    const float* x, size_t ldx,
                                                    const float* weight,
                                                    float eps, float* y,
                                                    size_t ldy) {
  if (x == nullptr || weight == nullptr || y == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (rows == 0 || dim == 0 || ldx < dim || ldy < dim) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  const int block = 256;
  rmsnorm_kernel<<<static_cast<int>(rows), block>>>(rows, dim, x, ldx, weight,
                                                    eps, y, ldy);
  return forge::cuda_detail::from_cuda(cudaGetLastError(), "rmsnorm_kernel");
}

#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>
#include <math.h>

namespace {

__global__ void rope_kernel(size_t seq_len, size_t n_heads, size_t head_dim,
                            float* x, size_t position_offset, float rope_theta) {
  size_t t = blockIdx.x;
  size_t h = blockIdx.y;
  if (t >= seq_len || h >= n_heads) {
    return;
  }
  const size_t half = head_dim / 2;
  float* row = x + (t * n_heads + h) * head_dim;
  const float pos = static_cast<float>(position_offset + t);
  for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
    float inv_freq =
        powf(rope_theta, -static_cast<float>(2 * i) / static_cast<float>(head_dim));
    float angle = pos * inv_freq;
    float cos_v = cosf(angle);
    float sin_v = sinf(angle);
    float x1 = row[i];
    float x2 = row[i + half];
    row[i] = x1 * cos_v - x2 * sin_v;
    row[i + half] = x2 * cos_v + x1 * sin_v;
  }
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_rope_f32(
    size_t seq_len, size_t n_heads, size_t head_dim, float* x,
    size_t position_offset, float rope_theta) {
  if (x == nullptr || seq_len == 0 || n_heads == 0 || head_dim == 0 ||
      (head_dim % 2) != 0 || !(rope_theta > 0.0f)) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  dim3 grid(static_cast<unsigned>(seq_len), static_cast<unsigned>(n_heads));
  rope_kernel<<<grid, 256>>>(seq_len, n_heads, head_dim, x, position_offset,
                             rope_theta);
  return forge::cuda_detail::launch_check("rope_kernel");
}

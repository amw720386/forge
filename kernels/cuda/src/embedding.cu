#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__global__ void embed_kernel(const float* table, size_t hidden, const int32_t* ids,
                             size_t t, float* out) {
  size_t row = blockIdx.x;
  if (row >= t) {
    return;
  }
  int32_t id = ids[row];
  if (id < 0) {
    return;
  }
  const float* src = table + static_cast<size_t>(id) * hidden;
  float* dst = out + row * hidden;
  for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = src[i];
  }
}

__global__ void embed_bf16_kernel(const __nv_bfloat16* table, size_t hidden,
                                  const int32_t* ids, size_t t, float* out) {
  size_t row = blockIdx.x;
  if (row >= t) {
    return;
  }
  int32_t id = ids[row];
  if (id < 0) {
    return;
  }
  const __nv_bfloat16* src = table + static_cast<size_t>(id) * hidden;
  float* dst = out + row * hidden;
  for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = __bfloat162float(src[i]);
  }
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_embedding_f32(
    const float* embed, size_t vocab, size_t hidden, const int32_t* token_ids,
    size_t t, float* out) {
  if (embed == nullptr || token_ids == nullptr || out == nullptr || t == 0 ||
      vocab == 0 || hidden == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  embed_kernel<<<static_cast<unsigned>(t), 256>>>(embed, hidden, token_ids, t,
                                                  out);
  return forge::cuda_detail::launch_check("embed_kernel");
}

extern "C" forge_cuda_status forge_cuda_embedding_bf16(
    const void* embed_bf16, size_t vocab, size_t hidden,
    const int32_t* token_ids, size_t t, float* out) {
  if (embed_bf16 == nullptr || token_ids == nullptr || out == nullptr ||
      t == 0 || vocab == 0 || hidden == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  embed_bf16_kernel<<<static_cast<unsigned>(t), 256>>>(
      static_cast<const __nv_bfloat16*>(embed_bf16), hidden, token_ids, t, out);
  return forge::cuda_detail::launch_check("embed_bf16_kernel");
}

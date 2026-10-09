#include "forge_kernels.h"

#include <cmath>

extern "C" forge_status forge_kernels_silu_f32(size_t n, const float* x,
                                               float* y) {
  if (x == nullptr || y == nullptr || n == 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < n; ++i) {
    float v = x[i];
    y[i] = v / (1.0f + std::exp(-v));
  }
  return FORGE_OK;
}

extern "C" forge_status forge_kernels_mul_f32(size_t n, const float* a,
                                              const float* b, float* y) {
  if (a == nullptr || b == nullptr || y == nullptr || n == 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < n; ++i) {
    y[i] = a[i] * b[i];
  }
  return FORGE_OK;
}

extern "C" forge_status forge_kernels_add_f32(size_t n, const float* a,
                                              const float* b, float* y) {
  if (a == nullptr || b == nullptr || y == nullptr || n == 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
  return FORGE_OK;
}

extern "C" forge_status forge_kernels_add_scaled_f32(size_t n, const float* a,
                                                     const float* b,
                                                     float scale, float* y) {
  if (a == nullptr || b == nullptr || y == nullptr || n == 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < n; ++i) {
    y[i] = a[i] + scale * b[i];
  }
  return FORGE_OK;
}

#include "forge_kernels.h"

#include <cmath>

extern "C" forge_status forge_kernels_rmsnorm_f32(size_t rows, size_t dim,
                                                  const float* x, size_t ldx,
                                                  const float* weight,
                                                  float eps, float* y,
                                                  size_t ldy) {
  if (x == nullptr || weight == nullptr || y == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (rows == 0 || dim == 0 || ldx < dim || ldy < dim || !(eps > 0.0f)) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }

  const float inv_dim = 1.0f / static_cast<float>(dim);
  for (size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * ldx;
    float* yr = y + r * ldy;
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      double v = static_cast<double>(xr[i]);
      acc += v * v;
    }
    float scale =
        static_cast<float>(1.0 / std::sqrt(acc * static_cast<double>(inv_dim) +
                                           static_cast<double>(eps)));
    for (size_t i = 0; i < dim; ++i) {
      yr[i] = xr[i] * scale * weight[i];
    }
  }
  return FORGE_OK;
}

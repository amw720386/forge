#include "forge_kernels.h"

#include "simd.hpp"
#include "thread_pool.hpp"

#include <algorithm>

namespace {

inline float dot_row(const float* xrow, const float* wrow, size_t k) {
#if FORGE_HAS_AVX2_FMA
  constexpr size_t kVec = 8;
  __m256 acc = _mm256_setzero_ps();
  size_t p = 0;
  for (; p + kVec <= k; p += kVec) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(xrow + p), _mm256_loadu_ps(wrow + p),
                          acc);
  }
  float sum = forge::simd::hsum256(acc);
  for (; p < k; ++p) {
    sum += xrow[p] * wrow[p];
  }
  return sum;
#else
  float sum = 0.0f;
  for (size_t p = 0; p < k; ++p) {
    sum += xrow[p] * wrow[p];
  }
  return sum;
#endif
}

void linear_range(size_t m, size_t k, const float* x, size_t ldx, const float* w,
                  size_t ldw, const float* bias, float* y, size_t ldy,
                  size_t j0, size_t j1) {
  for (size_t i = 0; i < m; ++i) {
    const float* xrow = x + i * ldx;
    for (size_t j = j0; j < j1; ++j) {
      float sum = dot_row(xrow, w + j * ldw, k);
      if (bias != nullptr) {
        sum += bias[j];
      }
      y[i * ldy + j] = sum;
    }
  }
}

}  // namespace

extern "C" forge_status forge_kernels_linear_f32(
    size_t m, size_t n, size_t k, const float* x, size_t ldx, const float* w,
    size_t ldw, const float* bias, float* y, size_t ldy) {
  if (x == nullptr || w == nullptr || y == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || k == 0 || ldx < k || ldw < k || ldy < n) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }

  const double flops =
      2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);

  // Persistent pool. Decode MLP (gate/up/down ≈ 8.7e6 flops) and lm_head
  // benefit; tiny projections (q/k/v ≈ 1.6e6) stay serial to avoid pool overhead.
  const bool worth = (flops >= 5e6) || (m >= 16 && n >= 512 && flops >= 2e6);

  if (worth && forge::ThreadPool::instance().total_threads() > 1) {
    forge::ThreadPool::instance().parallel_for(n, [&](size_t j0, size_t j1) {
      linear_range(m, k, x, ldx, w, ldw, bias, y, ldy, j0, j1);
    });
  } else {
    linear_range(m, k, x, ldx, w, ldw, bias, y, ldy, 0, n);
  }
  return FORGE_OK;
}

extern "C" unsigned forge_kernels_num_threads(void) {
  return forge::detect_num_threads();
}

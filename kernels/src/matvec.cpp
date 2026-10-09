#include "forge_kernels.h"

#include "simd.hpp"

namespace {

forge_status validate_matvec(size_t m, size_t n, const float* a, size_t lda,
                             const float* x, float* y) {
  if (a == nullptr || x == nullptr || y == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (lda < n) {
    return FORGE_ERR_DIMENSION_MISMATCH;
  }
  return FORGE_OK;
}

[[maybe_unused]] void matvec_naive(size_t m, size_t n, const float* a,
                                   size_t lda, const float* x, float* y) {
  for (size_t i = 0; i < m; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      sum += a[i * lda + j] * x[j];
    }
    y[i] = sum;
  }
}

#if FORGE_HAS_AVX2_FMA
void matvec_avx2(size_t m, size_t n, const float* a, size_t lda,
                 const float* x, float* y) {
  constexpr size_t kVec = 8;
  for (size_t i = 0; i < m; ++i) {
    __m256 acc = _mm256_setzero_ps();
    size_t j = 0;
    for (; j + kVec <= n; j += kVec) {
      __m256 av = _mm256_loadu_ps(&a[i * lda + j]);
      __m256 xv = _mm256_loadu_ps(&x[j]);
      acc = _mm256_fmadd_ps(av, xv, acc);
    }
    float sum = forge::simd::hsum256(acc);
    for (; j < n; ++j) {
      sum += a[i * lda + j] * x[j];
    }
    y[i] = sum;
  }
}
#endif

}  // namespace

extern "C" forge_status forge_kernels_matvec_f32(size_t m, size_t n,
                                                 const float* a, size_t lda,
                                                 const float* x, float* y) {
  forge_status st = validate_matvec(m, n, a, lda, x, y);
  if (st != FORGE_OK) {
    return st;
  }

#if FORGE_HAS_AVX2_FMA
  matvec_avx2(m, n, a, lda, x, y);
#else
  matvec_naive(m, n, a, lda, x, y);
#endif
  return FORGE_OK;
}

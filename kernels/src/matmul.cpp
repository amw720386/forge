#include "forge_kernels.h"

#include "simd.hpp"

namespace {

forge_status validate_matmul(size_t m, size_t n, size_t k, const float* a,
                             size_t lda, const float* b, size_t ldb, float* c,
                             size_t ldc) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || k == 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (lda < k || ldb < n || ldc < n) {
    return FORGE_ERR_DIMENSION_MISMATCH;
  }
  return FORGE_OK;
}

[[maybe_unused]] void matmul_naive(size_t m, size_t n, size_t k, const float* a,
                                   size_t lda, const float* b, size_t ldb,
                                   float* c, size_t ldc) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (size_t p = 0; p < k; ++p) {
        sum += a[i * lda + p] * b[p * ldb + j];
      }
      c[i * ldc + j] = sum;
    }
  }
}

#if FORGE_HAS_AVX2_FMA
void matmul_avx2(size_t m, size_t n, size_t k, const float* a, size_t lda,
                 const float* b, size_t ldb, float* c, size_t ldc) {
  constexpr size_t kVec = 8;
  for (size_t i = 0; i < m; ++i) {
    size_t j = 0;
    for (; j + kVec <= n; j += kVec) {
      __m256 acc = _mm256_setzero_ps();
      for (size_t p = 0; p < k; ++p) {
        __m256 av = _mm256_broadcast_ss(&a[i * lda + p]);
        __m256 bv = _mm256_loadu_ps(&b[p * ldb + j]);
        acc = _mm256_fmadd_ps(av, bv, acc);
      }
      _mm256_storeu_ps(&c[i * ldc + j], acc);
    }
    for (; j < n; ++j) {
      float sum = 0.0f;
      for (size_t p = 0; p < k; ++p) {
        sum += a[i * lda + p] * b[p * ldb + j];
      }
      c[i * ldc + j] = sum;
    }
  }
}
#endif

}  // namespace

extern "C" forge_status forge_kernels_matmul_f32(size_t m, size_t n, size_t k,
                                                 const float* a, size_t lda,
                                                 const float* b, size_t ldb,
                                                 float* c, size_t ldc) {
  forge_status st = validate_matmul(m, n, k, a, lda, b, ldb, c, ldc);
  if (st != FORGE_OK) {
    return st;
  }

#if FORGE_HAS_AVX2_FMA
  matmul_avx2(m, n, k, a, lda, b, ldb, c, ldc);
#else
  matmul_naive(m, n, k, a, lda, b, ldb, c, ldc);
#endif
  return FORGE_OK;
}

extern "C" const char* forge_kernels_version(void) {
  return "forge-kernels/0.7.0-int4-fused";
}

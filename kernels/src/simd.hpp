#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#include <immintrin.h>
#define FORGE_HAS_AVX2_FMA 1
#else
#define FORGE_HAS_AVX2_FMA 0
#endif

namespace forge {
namespace simd {

inline bool avx2_fma_enabled() {
#if FORGE_HAS_AVX2_FMA
  return true;
#else
  return false;
#endif
}

#if FORGE_HAS_AVX2_FMA
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}
#endif

}  // namespace simd
}  // namespace forge

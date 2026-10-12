#include "forge_kernels.h"

#include "simd.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cstdint>

namespace {

inline bool worth_parallel(size_t m, size_t n, size_t k) {
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);
  return (flops >= 5e6) || (m >= 16 && n >= 512 && flops >= 2e6);
}

inline float dot_i8_group(const float* x, const int8_t* w, size_t len) {
#if FORGE_HAS_AVX2_FMA
  __m256 acc = _mm256_setzero_ps();
  size_t p = 0;
  for (; p + 8 <= len; p += 8) {
    // Load 8 int8 → 8 float
    __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + p));
    __m256i wi = _mm256_cvtepi8_epi32(b);
    __m256 wf = _mm256_cvtepi32_ps(wi);
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + p), wf, acc);
  }
  float sum = forge::simd::hsum256(acc);
  for (; p < len; ++p) {
    sum += x[p] * static_cast<float>(w[p]);
  }
  return sum;
#else
  float sum = 0.0f;
  for (size_t p = 0; p < len; ++p) {
    sum += x[p] * static_cast<float>(w[p]);
  }
  return sum;
#endif
}

inline int8_t nibble_i4(uint8_t packed, bool high) {
  uint8_t n = high ? (packed >> 4) : (packed & 0x0fu);
  // Sign-extend 4-bit two's complement.
  return static_cast<int8_t>(static_cast<int8_t>(n << 4) >> 4);
}

#if FORGE_HAS_AVX2_FMA
// Unpack 8 packed bytes → 16 signed int8 (low nibble = even k, high = odd k).
inline __m128i unpack_i4_bytes8(const uint8_t* packed) {
  const __m128i pk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
  const __m128i mask = _mm_set1_epi8(0x0f);
  const __m128i lo = _mm_and_si128(pk, mask);
  // srli_epi16 by 4 extracts each byte's high nibble into its low 4 bits.
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(pk, 4), mask);
  // [lo0,hi0,lo1,hi1,...] — 16 nibbles as unsigned 0..15
  const __m128i interleaved = _mm_unpacklo_epi8(lo, hi);
  // Sign-extend 4-bit: values > 7 become negative (subtract 16).
  const __m128i gt7 = _mm_cmpgt_epi8(interleaved, _mm_set1_epi8(7));
  return _mm_sub_epi8(interleaved, _mm_and_si128(gt7, _mm_set1_epi8(16)));
}

inline float dot_i4_bytes_fma(const float* x, __m128i signed16) {
  // Low 8 int8 → 8 float
  __m256i wi0 = _mm256_cvtepi8_epi32(signed16);
  __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_cvtepi32_ps(wi0));
  // High 8 int8 → 8 float
  __m128i hi8 = _mm_srli_si128(signed16, 8);
  __m256i wi1 = _mm256_cvtepi8_epi32(hi8);
  acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + 8), _mm256_cvtepi32_ps(wi1), acc);
  return forge::simd::hsum256(acc);
}
#endif

inline float dot_i4_group(const float* x, const uint8_t* w_packed, size_t k0,
                          size_t len) {
  // Fused nibble unpack + FMA. Requires even k0 so packed bytes stay aligned
  // with (even,odd) nibble pairs (group starts are always even for gs≥2).
  float sum = 0.0f;
  size_t p = 0;
#if FORGE_HAS_AVX2_FMA
  if (((k0 + p) & 1u) == 0) {
    // 16 weights / 8 packed bytes per iteration.
    for (; p + 16 <= len; p += 16) {
      const size_t byte_off = (k0 + p) >> 1;
      sum += dot_i4_bytes_fma(x + p, unpack_i4_bytes8(w_packed + byte_off));
    }
    // 8 weights / 4 packed bytes.
    if (p + 8 <= len) {
      const size_t byte_off = (k0 + p) >> 1;
      uint8_t tmp_pk[8] = {};
      for (size_t i = 0; i < 4; ++i) {
        tmp_pk[i] = w_packed[byte_off + i];
      }
      __m128i signed16 = unpack_i4_bytes8(tmp_pk);
      // Only use low 8 int8 lanes.
      __m256i wi = _mm256_cvtepi8_epi32(signed16);
      __m256 acc =
          _mm256_mul_ps(_mm256_loadu_ps(x + p), _mm256_cvtepi32_ps(wi));
      sum += forge::simd::hsum256(acc);
      p += 8;
    }
  }
#endif
  // Scalar tail (handles odd k0 / remainders / non-AVX builds).
  for (; p < len; ++p) {
    const size_t col = k0 + p;
    sum += x[p] *
           static_cast<float>(nibble_i4(w_packed[col >> 1], (col & 1u) != 0));
  }
  return sum;
}

void linear_i8_range(size_t m, size_t k, const float* x, size_t ldx,
                     const int8_t* w_q, const float* scales, size_t group_size,
                     const float* bias, float* y, size_t ldy, size_t j0,
                     size_t j1) {
  const size_t n_groups = k / group_size;
  for (size_t i = 0; i < m; ++i) {
    const float* xrow = x + i * ldx;
    for (size_t j = j0; j < j1; ++j) {
      const int8_t* wrow = w_q + j * k;
      const float* srow = scales + j * n_groups;
      float sum = 0.0f;
      for (size_t g = 0; g < n_groups; ++g) {
        sum += srow[g] * dot_i8_group(xrow + g * group_size,
                                      wrow + g * group_size, group_size);
      }
      if (bias != nullptr) {
        sum += bias[j];
      }
      y[i * ldy + j] = sum;
    }
  }
}

void linear_i4_range(size_t m, size_t k, const float* x, size_t ldx,
                     const uint8_t* w_q, const float* scales, size_t group_size,
                     const float* bias, float* y, size_t ldy, size_t j0,
                     size_t j1) {
  const size_t n_groups = k / group_size;
  const size_t packed_row = (k + 1) / 2;
  for (size_t i = 0; i < m; ++i) {
    const float* xrow = x + i * ldx;
    for (size_t j = j0; j < j1; ++j) {
      const uint8_t* wrow = w_q + j * packed_row;
      const float* srow = scales + j * n_groups;
      float sum = 0.0f;
      for (size_t g = 0; g < n_groups; ++g) {
        sum += srow[g] * dot_i4_group(xrow + g * group_size, wrow,
                                      g * group_size, group_size);
      }
      if (bias != nullptr) {
        sum += bias[j];
      }
      y[i * ldy + j] = sum;
    }
  }
}

}  // namespace

extern "C" forge_status forge_kernels_linear_i8_f32(
    size_t m, size_t n, size_t k, const float* x, size_t ldx, const int8_t* w_q,
    const float* scales, size_t group_size, const float* bias, float* y,
    size_t ldy) {
  if (x == nullptr || w_q == nullptr || scales == nullptr || y == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || k == 0 || group_size == 0 || (k % group_size) != 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (ldx < k || ldy < n) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }

  if (worth_parallel(m, n, k) &&
      forge::ThreadPool::instance().total_threads() > 1) {
    forge::ThreadPool::instance().parallel_for(n, [&](size_t j0, size_t j1) {
      linear_i8_range(m, k, x, ldx, w_q, scales, group_size, bias, y, ldy, j0,
                      j1);
    });
  } else {
    linear_i8_range(m, k, x, ldx, w_q, scales, group_size, bias, y, ldy, 0, n);
  }
  return FORGE_OK;
}

extern "C" forge_status forge_kernels_linear_i4_f32(
    size_t m, size_t n, size_t k, const float* x, size_t ldx,
    const uint8_t* w_q, const float* scales, size_t group_size,
    const float* bias, float* y, size_t ldy) {
  if (x == nullptr || w_q == nullptr || scales == nullptr || y == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || k == 0 || group_size == 0 || (k % group_size) != 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (ldx < k || ldy < n) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }

  if (worth_parallel(m, n, k) &&
      forge::ThreadPool::instance().total_threads() > 1) {
    forge::ThreadPool::instance().parallel_for(n, [&](size_t j0, size_t j1) {
      linear_i4_range(m, k, x, ldx, w_q, scales, group_size, bias, y, ldy, j0,
                      j1);
    });
  } else {
    linear_i4_range(m, k, x, ldx, w_q, scales, group_size, bias, y, ldy, 0, n);
  }
  return FORGE_OK;
}

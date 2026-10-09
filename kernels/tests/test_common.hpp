#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace forge_test {

inline int& failures() {
  static int n = 0;
  return n;
}

inline void expect(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    ++failures();
  }
}

inline bool nearly_equal(float a, float b, float atol = 1e-5f,
                         float rtol = 1e-4f) {
  float diff = std::fabs(a - b);
  float scale = std::fmax(std::fabs(a), std::fabs(b));
  return diff <= atol + rtol * scale;
}

inline void expect_close(float got, float want, const std::string& msg) {
  if (!nearly_equal(got, want)) {
    std::cerr << "FAIL: " << msg << " got=" << got << " want=" << want << "\n";
    ++failures();
  }
}

inline std::vector<float> reference_matmul(size_t m, size_t n, size_t k,
                                           const float* a, size_t lda,
                                           const float* b, size_t ldb) {
  std::vector<float> c(m * n, 0.0f);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (size_t p = 0; p < k; ++p) {
        sum += a[i * lda + p] * b[p * ldb + j];
      }
      c[i * n + j] = sum;
    }
  }
  return c;
}

inline std::vector<float> reference_matvec(size_t m, size_t n, const float* a,
                                           size_t lda, const float* x) {
  std::vector<float> y(m, 0.0f);
  for (size_t i = 0; i < m; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      sum += a[i * lda + j] * x[j];
    }
    y[i] = sum;
  }
  return y;
}

inline std::vector<float> fill_sequence(size_t n, float start = 0.1f,
                                        float step = 0.07f) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = start + static_cast<float>(i) * step;
  }
  return v;
}

}  // namespace forge_test

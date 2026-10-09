#include "forge_kernels.h"
#include "test_common.hpp"

#include <vector>

void test_matvec() {
  using namespace forge_test;

  {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float x[3] = {1, 0, -1};
    float y[2] = {};
    expect(forge_kernels_matvec_f32(2, 3, a, 3, x, y) == FORGE_OK,
           "matvec status");
    expect_close(y[0], -2.0f, "y0");
    expect_close(y[1], -2.0f, "y1");
  }

  {
    const size_t m = 23, n = 29;
    auto a = fill_sequence(m * n, -1.0f, 0.013f);
    auto x = fill_sequence(n, 0.5f, -0.021f);
    std::vector<float> y(m, 0.0f);
    expect(forge_kernels_matvec_f32(m, n, a.data(), n, x.data(), y.data()) ==
               FORGE_OK,
           "random matvec status");
    auto ref = reference_matvec(m, n, a.data(), n, x.data());
    for (size_t i = 0; i < m; ++i) {
      expect_close(y[i], ref[i], "matvec[" + std::to_string(i) + "]");
    }
  }

  {
    float a[2] = {1, 2};
    float x[2] = {1, 1};
    float y[1] = {};
    expect(forge_kernels_matvec_f32(1, 2, a, 1, x, y) ==
               FORGE_ERR_DIMENSION_MISMATCH,
           "bad lda rejected");
  }
}

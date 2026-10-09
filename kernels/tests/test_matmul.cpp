#include "forge_kernels.h"
#include "test_common.hpp"

#include <vector>

void test_matmul() {
  using namespace forge_test;

  {
    // Hand-checked 2x3 * 3x2
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {7, 8, 9, 10, 11, 12};
    float c[4] = {};
    expect(forge_kernels_matmul_f32(2, 2, 3, a, 3, b, 2, c, 2) == FORGE_OK,
           "matmul status");
    expect_close(c[0], 58.0f, "c00");
    expect_close(c[1], 64.0f, "c01");
    expect_close(c[2], 139.0f, "c10");
    expect_close(c[3], 154.0f, "c11");
  }

  {
    const size_t m = 17, n = 13, k = 19;
    auto a = fill_sequence(m * k, 0.02f, 0.011f);
    auto b = fill_sequence(k * n, -0.3f, 0.017f);
    std::vector<float> c(m * n, 0.0f);
    expect(forge_kernels_matmul_f32(m, n, k, a.data(), k, b.data(), n, c.data(),
                                    n) == FORGE_OK,
           "random matmul status");
    auto ref = reference_matmul(m, n, k, a.data(), k, b.data(), n);
    for (size_t i = 0; i < m * n; ++i) {
      expect_close(c[i], ref[i], "matmul[" + std::to_string(i) + "]");
    }
  }

  {
    float a[1] = {1};
    float b[1] = {1};
    float c[1] = {};
    expect(forge_kernels_matmul_f32(1, 1, 1, nullptr, 1, b, 1, c, 1) ==
               FORGE_ERR_INVALID_ARGUMENT,
           "null A rejected");
    expect(forge_kernels_matmul_f32(1, 1, 1, a, 0, b, 1, c, 1) ==
               FORGE_ERR_DIMENSION_MISMATCH,
           "bad lda rejected");
  }
}

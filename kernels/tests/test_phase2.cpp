#include "forge_kernels.h"
#include "test_common.hpp"

#include <cmath>
#include <vector>

void test_phase2() {
  using namespace forge_test;

  // RMSNorm: single row
  {
    float x[4] = {1.f, -1.f, 2.f, -2.f};
    float w[4] = {1.f, 1.f, 1.f, 1.f};
    float y[4] = {};
    float mean_sq = (1 + 1 + 4 + 4) / 4.f;
    float scale = 1.f / std::sqrt(mean_sq + 1e-6f);
    expect(forge_kernels_rmsnorm_f32(1, 4, x, 4, w, 1e-6f, y, 4) == FORGE_OK,
           "rmsnorm status");
    for (int i = 0; i < 4; ++i) {
      expect_close(y[i], x[i] * scale, "rmsnorm");
    }
  }

  // Softmax causal 2x2
  {
    float s[4] = {1.f, 2.f, 3.f, 4.f};
    expect(forge_kernels_softmax_f32(1, 2, 2, s, 1, 0) == FORGE_OK,
           "softmax status");
    // row0: only col0 → [1, 0]
    expect_close(s[0], 1.f, "sm00");
    expect_close(s[1], 0.f, "sm01");
    // row1: both cols, softmax([3,4])
    float e0 = std::exp(3.f - 4.f);
    float e1 = 1.f;
    float z = e0 + e1;
    expect_close(s[2], e0 / z, "sm10");
    expect_close(s[3], e1 / z, "sm11");
  }

  // SiLU
  {
    float x[2] = {0.f, 2.f};
    float y[2] = {};
    expect(forge_kernels_silu_f32(2, x, y) == FORGE_OK, "silu");
    expect_close(y[0], 0.f, "silu0");
    expect_close(y[1], 2.f / (1.f + std::exp(-2.f)), "silu1");
  }

  // Linear vs reference: Y = X @ W^T
  {
    float x[6] = {1, 2, 3, 4, 5, 6};          // 2x3
    float w[6] = {1, 0, 0, 0, 1, 0};          // 2x3 → out=2
    float y[4] = {};
    expect(forge_kernels_linear_f32(2, 2, 3, x, 3, w, 3, nullptr, y, 2) ==
               FORGE_OK,
           "linear");
    expect_close(y[0], 1.f, "lin00");
    expect_close(y[1], 2.f, "lin01");
    expect_close(y[2], 4.f, "lin10");
    expect_close(y[3], 5.f, "lin11");
  }

  // RoPE identity at position 0: cos=1,sin=0 → unchanged
  {
    float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};  // T=1, heads=1, dim=8
    float orig[8];
    for (int i = 0; i < 8; ++i) orig[i] = x[i];
    expect(forge_kernels_rope_f32(1, 1, 8, x, nullptr, 0, 10000.f) == FORGE_OK,
           "rope");
    for (int i = 0; i < 8; ++i) {
      expect_close(x[i], orig[i], "rope pos0");
    }
  }

  // INT4 fused linear vs scalar nibble reference (group=8, k=16).
  {
    constexpr size_t m = 1, n = 2, k = 16, gs = 8;
    float x[16];
    for (size_t i = 0; i < k; ++i) x[i] = 0.1f * static_cast<float>(i + 1);
    uint8_t w_q[(k / 2) * n];
    auto pack = [](int8_t even, int8_t odd) -> uint8_t {
      return static_cast<uint8_t>((even & 0x0f) | ((odd & 0x0f) << 4));
    };
    const int8_t row0[16] = {1, -2, 3, -4, 5, -6, 7, -7,
                             1, -1, 2, -2, 3, -3, 4, -4};
    const int8_t row1[16] = {1, 1, 1, 1, 1, 1, 1, 1,
                             -1, -1, -1, -1, -1, -1, -1, -1};
    for (size_t c = 0; c < k; c += 2) {
      w_q[c / 2] = pack(row0[c], row0[c + 1]);
      w_q[(k / 2) + c / 2] = pack(row1[c], row1[c + 1]);
    }
    float scales[n * (k / gs)] = {0.5f, 0.25f, 0.1f, 0.2f};
    float y[2] = {};
    expect(forge_kernels_linear_i4_f32(m, n, k, x, k, w_q, scales, gs, nullptr,
                                       y, n) == FORGE_OK,
           "linear_i4 status");

    auto ref_row = [&](const int8_t* qrow, const float* srow) {
      float sum = 0.f;
      for (size_t g = 0; g < k / gs; ++g) {
        float partial = 0.f;
        for (size_t p = 0; p < gs; ++p) {
          partial += x[g * gs + p] * static_cast<float>(qrow[g * gs + p]);
        }
        sum += srow[g] * partial;
      }
      return sum;
    };
    expect_close(y[0], ref_row(row0, scales), "i4 y0");
    expect_close(y[1], ref_row(row1, scales + 2), "i4 y1");
  }
}

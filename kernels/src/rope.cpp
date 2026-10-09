#include "forge_kernels.h"

#include <cmath>
#include <vector>

extern "C" forge_status forge_kernels_rope_f32(
    size_t seq_len, size_t n_heads, size_t head_dim, float* x,
    const int32_t* positions, size_t position_offset, float rope_theta) {
  if (x == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (seq_len == 0 || n_heads == 0 || head_dim == 0 || (head_dim % 2) != 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (!(rope_theta > 0.0f)) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }

  const size_t half = head_dim / 2;
  std::vector<float> inv_freq(half);
  for (size_t i = 0; i < half; ++i) {
    inv_freq[i] = static_cast<float>(
        1.0 / std::pow(static_cast<double>(rope_theta),
                       static_cast<double>(2 * i) /
                           static_cast<double>(head_dim)));
  }

  for (size_t t = 0; t < seq_len; ++t) {
    const float pos = positions != nullptr
                          ? static_cast<float>(positions[t])
                          : static_cast<float>(position_offset + t);
    for (size_t h = 0; h < n_heads; ++h) {
      float* row = x + (t * n_heads + h) * head_dim;
      // HF rotate_half: split [0..half) | [half..dim)
      for (size_t i = 0; i < half; ++i) {
        float cos_v = std::cos(pos * inv_freq[i]);
        float sin_v = std::sin(pos * inv_freq[i]);
        float x1 = row[i];
        float x2 = row[i + half];
        // out1 = x1*cos - x2*sin ; out2 = x2*cos + x1*sin
        // Equivalent to x*cos + rotate_half(x)*sin with rotate_half=(-x2,x1)
        row[i] = x1 * cos_v - x2 * sin_v;
        row[i + half] = x2 * cos_v + x1 * sin_v;
      }
    }
  }
  return FORGE_OK;
}

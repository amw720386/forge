#include "forge_kernels.h"

#include <cmath>
#include <limits>

extern "C" forge_status forge_kernels_softmax_f32(size_t batch, size_t rows,
                                                  size_t cols, float* scores,
                                                  int causal,
                                                  size_t causal_q_offset) {
  if (scores == nullptr) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }
  if (batch == 0 || rows == 0 || cols == 0) {
    return FORGE_ERR_INVALID_ARGUMENT;
  }

  constexpr float kNegInf = -std::numeric_limits<float>::infinity();

  for (size_t b = 0; b < batch; ++b) {
    for (size_t r = 0; r < rows; ++r) {
      float* row = scores + (b * rows + r) * cols;
      const size_t q_pos = causal_q_offset + r;

      if (causal) {
        for (size_t c = 0; c < cols; ++c) {
          if (c > q_pos) {
            row[c] = kNegInf;
          }
        }
      }

      float max_v = kNegInf;
      for (size_t c = 0; c < cols; ++c) {
        if (row[c] > max_v) {
          max_v = row[c];
        }
      }

      // All-masked row → uniform zeros (should not happen with valid causal).
      if (!(max_v > kNegInf)) {
        for (size_t c = 0; c < cols; ++c) {
          row[c] = 0.0f;
        }
        continue;
      }

      double sum = 0.0;
      for (size_t c = 0; c < cols; ++c) {
        float e = std::exp(row[c] - max_v);
        row[c] = e;
        sum += static_cast<double>(e);
      }
      float inv = static_cast<float>(1.0 / sum);
      for (size_t c = 0; c < cols; ++c) {
        row[c] *= inv;
      }
    }
  }
  return FORGE_OK;
}

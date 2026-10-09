/**
 * Synthetic matmul micro-benchmark.
 * Build and run via `make bench` or scripts/bench.ps1.
 */

#include "forge_kernels.h"

#include <chrono>
#include <cstdio>
#include <vector>

int main() {
  const size_t m = 256, n = 256, k = 256;
  std::vector<float> a(m * k), b(k * n), c(m * n);
  for (size_t i = 0; i < a.size(); ++i) a[i] = 0.001f * static_cast<float>(i);
  for (size_t i = 0; i < b.size(); ++i) b[i] = 0.002f * static_cast<float>(i);

  // Warmup
  for (int i = 0; i < 3; ++i) {
    forge_kernels_matmul_f32(m, n, k, a.data(), k, b.data(), n, c.data(), n);
  }

  const int iters = 20;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    forge_kernels_matmul_f32(m, n, k, a.data(), k, b.data(), n, c.data(), n);
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  double gflops = (2.0 * m * n * k) / (ms * 1e6);

  std::printf("matmul %zux%zux%zu: %.3f ms/iter, %.2f GFLOP/s (synthetic)\n", m,
              n, k, ms, gflops);
  std::printf("kernels: %s\n", forge_kernels_version());
  return 0;
}

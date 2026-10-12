//! CUDA microbenchmarks — run on a CUDA GPU host only.
//! Uses CUDA events for GPU timing; also reports host wall time.

#include "forge_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

double ms_event(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  cudaEventElapsedTime(&ms, a, b);
  return static_cast<double>(ms);
}

void print_env() {
  char name[256] = {};
  forge_cuda_device_name(0, name, sizeof(name));
  size_t free_b = 0, total_b = 0;
  forge_cuda_mem_info(&free_b, &total_b);
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::printf("device=%s\n", name);
  std::printf("forge_cuda=%s\n", forge_cuda_version());
  std::printf("driver=%d runtime=%d\n", driver, runtime);
  std::printf("mem free=%.1f MiB total=%.1f MiB\n", free_b / (1024.0 * 1024.0),
              total_b / (1024.0 * 1024.0));
}

}  // namespace

int main() {
  if (!forge_cuda_is_available()) {
    std::fprintf(stderr, "No CUDA device; refusing to bench.\n");
    return 2;
  }
  std::printf("=== forge_cuda_bench ===\n");
  print_env();

  cudaEvent_t start{}, stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocation overhead (bounded)
  {
    const int iters = 50;
    const size_t bytes = 4 * 1024 * 1024;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
      forge_cuda_buffer b{};
      forge_cuda_malloc(&b, bytes);
      forge_cuda_free(&b);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::printf("alloc/free %zu MiB x%d: %.3f ms/iter (GPU event)\n",
                bytes / (1024 * 1024), iters, ms_event(start, stop) / iters);
  }

  // H2D / D2H bandwidth (64 MiB — modest)
  {
    const size_t bytes = 64ull * 1024ull * 1024ull;
    std::vector<char> host(bytes);
    forge_cuda_buffer d{};
    forge_cuda_malloc(&d, bytes);
    cudaEventRecord(start);
    forge_cuda_memcpy_h2d(&d, host.data(), bytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    double h2d = ms_event(start, stop);
    cudaEventRecord(start);
    forge_cuda_memcpy_d2h(host.data(), &d, bytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    double d2h = ms_event(start, stop);
    double gib = bytes / (1024.0 * 1024.0 * 1024.0);
    std::printf("H2D 64 MiB: %.3f ms (%.2f GiB/s)\n", h2d, gib / (h2d / 1e3));
    std::printf("D2H 64 MiB: %.3f ms (%.2f GiB/s)\n", d2h, gib / (d2h / 1e3));
    forge_cuda_free(&d);
  }

  // GEMM throughput
  {
    const size_t m = 1024, n = 1024, k = 1024;
    const size_t bytes_a = m * k * 4, bytes_b = k * n * 4, bytes_c = m * n * 4;
    forge_cuda_buffer a{}, b{}, c{};
    forge_cuda_malloc(&a, bytes_a);
    forge_cuda_malloc(&b, bytes_b);
    forge_cuda_malloc(&c, bytes_c);
    forge_cuda_memset_zero(&a);
    forge_cuda_memset_zero(&b);
    // Warmup
    forge_cuda_matmul_f32(m, n, k, static_cast<float*>(a.ptr), k,
                          static_cast<float*>(b.ptr), n,
                          static_cast<float*>(c.ptr), n);
    forge_cuda_synchronize();
    const int iters = 20;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
      forge_cuda_matmul_f32(m, n, k, static_cast<float*>(a.ptr), k,
                            static_cast<float*>(b.ptr), n,
                            static_cast<float*>(c.ptr), n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    double ms = ms_event(start, stop) / iters;
    double flops = 2.0 * m * n * k;
    std::printf("GEMM %zux%zux%zu: %.3f ms/iter  %.1f GFLOP/s (GPU event)\n", m,
                n, k, ms, (flops / 1e9) / (ms / 1e3));
    forge_cuda_free(&a);
    forge_cuda_free(&b);
    forge_cuda_free(&c);
  }

  // Matvec
  {
    const size_t m = 4096, n = 4096;
    forge_cuda_buffer a{}, x{}, y{};
    forge_cuda_malloc(&a, m * n * 4);
    forge_cuda_malloc(&x, n * 4);
    forge_cuda_malloc(&y, m * 4);
    forge_cuda_memset_zero(&a);
    forge_cuda_memset_zero(&x);
    forge_cuda_matvec_f32(m, n, static_cast<float*>(a.ptr), n,
                          static_cast<float*>(x.ptr),
                          static_cast<float*>(y.ptr));
    forge_cuda_synchronize();
    const int iters = 50;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
      forge_cuda_matvec_f32(m, n, static_cast<float*>(a.ptr), n,
                            static_cast<float*>(x.ptr),
                            static_cast<float*>(y.ptr));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::printf("matvec %zux%zu: %.3f ms/iter (GPU event)\n", m, n,
                ms_event(start, stop) / iters);
    forge_cuda_free(&a);
    forge_cuda_free(&x);
    forge_cuda_free(&y);
  }

  // Kernel launch overhead (tiny silu)
  {
    forge_cuda_buffer x{}, y{};
    forge_cuda_malloc(&x, 4);
    forge_cuda_malloc(&y, 4);
    const int iters = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
      forge_cuda_silu_f32(1, static_cast<float*>(x.ptr),
                          static_cast<float*>(y.ptr));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    std::printf("silu launch overhead: %.4f ms/launch (GPU event)\n",
                ms_event(start, stop) / iters);
    forge_cuda_free(&x);
    forge_cuda_free(&y);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::printf("done\n");
  return 0;
}

#pragma once

#include "forge_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include <cuda_runtime.h>

namespace forge {
namespace cuda_detail {

// Diagnostic-only CUDA event region. Disabled during graph capture and normal
// throughput runs; synchronization overhead must not enter performance claims.
struct ProfileRegion {
  cudaEvent_t begin{}, end{};
  const char* label;
  explicit ProfileRegion(const char* name) : label(name) {
    if (!std::getenv("FORGE_KERNEL_PROFILE")) return;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return;
    if (cudaEventCreate(&begin) != cudaSuccess) { begin = nullptr; return; }
    if (cudaEventCreate(&end) != cudaSuccess) {
      cudaEventDestroy(begin); begin = nullptr; end = nullptr; return;
    }
    cudaEventRecord(begin, cudaStreamPerThread);
  }
  ~ProfileRegion() {
    if (!begin) return;
    if (cudaEventRecord(end, cudaStreamPerThread) == cudaSuccess &&
        cudaEventSynchronize(end) == cudaSuccess) {
      float ms = 0;
      if (cudaEventElapsedTime(&ms, begin, end) == cudaSuccess)
        std::fprintf(stderr, "[runtime device] stage=%s ms=%.6f\n", label, ms);
    }
    cudaEventDestroy(begin); cudaEventDestroy(end);
  }
};

inline std::mutex& err_mu() {
  static std::mutex m;
  return m;
}

inline std::string& err_storage() {
  static std::string s;
  return s;
}

inline void set_error(const char* where, cudaError_t err) {
  std::lock_guard<std::mutex> lock(err_mu());
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s: %s (%d)", where, cudaGetErrorString(err),
                static_cast<int>(err));
  err_storage() = buf;
}

inline forge_cuda_status from_cuda(cudaError_t err, const char* where) {
  if (err == cudaSuccess) {
    return FORGE_CUDA_OK;
  }
  set_error(where, err);
  if (err == cudaErrorMemoryAllocation) {
    return FORGE_CUDA_ERR_OUT_OF_MEMORY;
  }
  return FORGE_CUDA_ERR_INTERNAL;
}

/** Check last CUDA launch/runtime error after a kernel launch. */
inline forge_cuda_status launch_check(const char* where) {
  return from_cuda(cudaGetLastError(), where);
}

inline bool check_overflow_mul(size_t a, size_t b, size_t* out) {
  if (a != 0 && b > SIZE_MAX / a) {
    return false;
  }
  *out = a * b;
  return true;
}

}  // namespace cuda_detail
}  // namespace forge

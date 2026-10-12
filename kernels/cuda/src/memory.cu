#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>
#include <new>

extern "C" forge_cuda_status forge_cuda_malloc(forge_cuda_buffer* buf,
                                               size_t bytes) {
  if (buf == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  buf->ptr = nullptr;
  buf->bytes = 0;
  if (bytes == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  // Soft cap: refuse absurd allocations that are almost certainly bugs
  // (untrusted size paths). 16 GiB is enough for kernel benches.
  constexpr size_t kMaxBytes = 16ull * 1024ull * 1024ull * 1024ull;
  if (bytes > kMaxBytes) {
    forge::cuda_detail::set_error("forge_cuda_malloc", cudaErrorInvalidValue);
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  void* p = nullptr;
  cudaError_t err = cudaMalloc(&p, bytes);
  if (err != cudaSuccess) {
    return forge::cuda_detail::from_cuda(err, "cudaMalloc");
  }
  buf->ptr = p;
  buf->bytes = bytes;
  return FORGE_CUDA_OK;
}

extern "C" forge_cuda_status forge_cuda_free(forge_cuda_buffer* buf) {
  if (buf == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (buf->ptr == nullptr) {
    buf->bytes = 0;
    return FORGE_CUDA_OK;
  }
  cudaError_t err = cudaFree(buf->ptr);
  buf->ptr = nullptr;
  buf->bytes = 0;
  return forge::cuda_detail::from_cuda(err, "cudaFree");
}

extern "C" forge_cuda_status forge_cuda_memcpy_h2d(forge_cuda_buffer* dst,
                                                   const void* src,
                                                   size_t bytes) {
  if (dst == nullptr || dst->ptr == nullptr || src == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (bytes > dst->bytes) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  return forge::cuda_detail::from_cuda(
      cudaMemcpy(dst->ptr, src, bytes, cudaMemcpyHostToDevice),
      "cudaMemcpyH2D");
}

extern "C" forge_cuda_status forge_cuda_memcpy_d2h(void* dst,
                                                   const forge_cuda_buffer* src,
                                                   size_t bytes) {
  if (dst == nullptr || src == nullptr || src->ptr == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (bytes > src->bytes) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  return forge::cuda_detail::from_cuda(
      cudaMemcpy(dst, src->ptr, bytes, cudaMemcpyDeviceToHost),
      "cudaMemcpyD2H");
}

extern "C" forge_cuda_status forge_cuda_memcpy_d2d(forge_cuda_buffer* dst,
                                                   size_t dst_off,
                                                   const forge_cuda_buffer* src,
                                                   size_t src_off,
                                                   size_t bytes) {
  if (dst == nullptr || dst->ptr == nullptr || src == nullptr ||
      src->ptr == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (dst_off + bytes > dst->bytes || src_off + bytes > src->bytes) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  auto* d = static_cast<char*>(dst->ptr) + dst_off;
  auto* s = static_cast<const char*>(src->ptr) + src_off;
  return forge::cuda_detail::from_cuda(
      cudaMemcpy(d, s, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpyD2D");
}

extern "C" forge_cuda_status forge_cuda_memset_zero(forge_cuda_buffer* buf) {
  if (buf == nullptr || buf->ptr == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  return forge::cuda_detail::from_cuda(cudaMemset(buf->ptr, 0, buf->bytes),
                                       "cudaMemset");
}

extern "C" forge_cuda_status forge_cuda_synchronize(void) {
  return forge::cuda_detail::from_cuda(cudaDeviceSynchronize(),
                                       "cudaDeviceSynchronize");
}

extern "C" forge_cuda_status forge_cuda_graph_begin(void) {
  return forge::cuda_detail::from_cuda(
      cudaStreamBeginCapture(cudaStreamPerThread,
                             cudaStreamCaptureModeThreadLocal),
      "graph_begin");
}
extern "C" forge_cuda_status forge_cuda_graph_end(void** executable) {
  cudaGraph_t graph = nullptr;
  cudaError_t err = cudaStreamEndCapture(cudaStreamPerThread, &graph);
  if (executable) *executable = nullptr;
  if (err != cudaSuccess) {
    if (graph) cudaGraphDestroy(graph);
    return forge::cuda_detail::from_cuda(err, "graph_end");
  }
  if (!executable) {
    if (graph) cudaGraphDestroy(graph);
    return FORGE_CUDA_OK;
  }
  cudaGraphExec_t exec = nullptr;
  err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  cudaGraphDestroy(graph);
  if (err == cudaSuccess) *executable = exec;
  return forge::cuda_detail::from_cuda(err, "graph_instantiate");
}
extern "C" forge_cuda_status forge_cuda_graph_replay(void* executable) {
  if (!executable) return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  return forge::cuda_detail::from_cuda(
      cudaGraphLaunch(static_cast<cudaGraphExec_t>(executable),
                      cudaStreamPerThread),
      "graph_replay");
}
extern "C" forge_cuda_status forge_cuda_graph_destroy(void* executable) {
  if (!executable) return FORGE_CUDA_OK;
  return forge::cuda_detail::from_cuda(
      cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(executable)),
      "graph_destroy");
}

namespace {
struct ForgeTimer {
  cudaEvent_t start{}, stop{};
};
}  // namespace
extern "C" forge_cuda_status forge_cuda_timer_destroy(void* timer) {
  if (!timer) return FORGE_CUDA_OK;
  auto* t = static_cast<ForgeTimer*>(timer);
  if (t->start) cudaEventDestroy(t->start);
  if (t->stop) cudaEventDestroy(t->stop);
  delete t;
  return FORGE_CUDA_OK;
}
extern "C" forge_cuda_status forge_cuda_timer_start(void** timer) {
  if (!timer) return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  *timer = nullptr;
  auto* t = new (std::nothrow) ForgeTimer;
  if (!t) return FORGE_CUDA_ERR_OUT_OF_MEMORY;
  auto err = cudaEventCreate(&t->start);
  if (err == cudaSuccess) err = cudaEventCreate(&t->stop);
  if (err == cudaSuccess) err = cudaEventRecord(t->start, cudaStreamPerThread);
  if (err != cudaSuccess) {
    forge_cuda_timer_destroy(t);
    return forge::cuda_detail::from_cuda(err, "timer_start");
  }
  *timer = t;
  return FORGE_CUDA_OK;
}
extern "C" forge_cuda_status forge_cuda_timer_elapsed(void* timer, float* ms) {
  if (!timer || !ms) return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  auto* t = static_cast<ForgeTimer*>(timer);
  auto err = cudaEventRecord(t->stop, cudaStreamPerThread);
  if (err == cudaSuccess) err = cudaEventSynchronize(t->stop);
  if (err == cudaSuccess) err = cudaEventElapsedTime(ms, t->start, t->stop);
  return forge::cuda_detail::from_cuda(err, "timer_elapsed");
}

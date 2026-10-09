#include "error.hpp"
#include "forge_cuda.h"

#include <cuda_runtime.h>

#include <cstring>

extern "C" const char* forge_cuda_version(void) {
  return "forge-cuda/0.1.0-phase8a";
}

extern "C" int forge_cuda_is_available(void) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    return 0;
  }
  return n > 0 ? 1 : 0;
}

extern "C" int forge_cuda_device_count(void) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    return 0;
  }
  return n;
}

extern "C" forge_cuda_status forge_cuda_set_device(int device) {
  return forge::cuda_detail::from_cuda(cudaSetDevice(device), "cudaSetDevice");
}

extern "C" forge_cuda_status forge_cuda_device_name(int device, char* out,
                                                   size_t out_len) {
  if (out == nullptr || out_len == 0) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return forge::cuda_detail::from_cuda(err, "cudaGetDeviceProperties");
  }
  std::strncpy(out, prop.name, out_len - 1);
  out[out_len - 1] = '\0';
  return FORGE_CUDA_OK;
}

extern "C" forge_cuda_status forge_cuda_mem_info(size_t* free_bytes,
                                                 size_t* total_bytes) {
  if (free_bytes == nullptr || total_bytes == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  size_t free_b = 0, total_b = 0;
  cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
  if (err != cudaSuccess) {
    return forge::cuda_detail::from_cuda(err, "cudaMemGetInfo");
  }
  *free_bytes = free_b;
  *total_bytes = total_b;
  return FORGE_CUDA_OK;
}

extern "C" const char* forge_cuda_last_error(void) {
  std::lock_guard<std::mutex> lock(forge::cuda_detail::err_mu());
  const std::string& s = forge::cuda_detail::err_storage();
  return s.empty() ? "" : s.c_str();
}

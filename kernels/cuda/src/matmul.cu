#include "error.hpp"
#include "forge_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

cublasHandle_t& handle() {
  static cublasHandle_t h = nullptr;
  static bool init = false;
  if (!init) {
    cublasCreate(&h);
    cublasSetStream(h, cudaStreamPerThread);
    init = true;
  }
  return h;
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_matmul_f32(size_t m, size_t n, size_t k,
                                                   const float* a, size_t lda,
                                                   const float* b, size_t ldb,
                                                   float* c, size_t ldc) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || k == 0 || lda < k || ldb < n || ldc < n) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }

  // Row-major C = A @ B via cuBLAS column-major: C^T = B^T @ A^T
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t st =
      cublasSgemm(handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n),
                  static_cast<int>(m), static_cast<int>(k), &alpha, b,
                  static_cast<int>(ldb), a, static_cast<int>(lda), &beta, c,
                  static_cast<int>(ldc));
  if (st != CUBLAS_STATUS_SUCCESS) {
    forge::cuda_detail::set_error("cublasSgemm", cudaErrorUnknown);
    return FORGE_CUDA_ERR_INTERNAL;
  }
  return FORGE_CUDA_OK;
}

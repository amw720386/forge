#include "error.hpp"
#include "forge_cuda.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <map>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

cublasHandle_t& cublas_handle() {
  static cublasHandle_t h = nullptr;
  static bool init = false;
  if (!init) {
    cublasCreate(&h);
    cublasSetStream(h, cudaStreamPerThread);
    init = true;
  }
  return h;
}

// Offline L4 shape measurements select only useful heuristic ranks. No runtime
// benchmarking; descriptor/algorithm cache and workspace have stable addresses.
struct LtPlan {
 cublasLtMatmulDesc_t desc=nullptr;
 cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
 cublasLtMatmulAlgo_t algo{};bool valid=false;
 ~LtPlan(){if(desc)cublasLtMatmulDescDestroy(desc);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);}
};
struct LtCache {
 cublasLtHandle_t handle=nullptr;void* workspace=nullptr;
 std::map<std::array<size_t,6>,LtPlan> plans;
 bool supported=false;
 LtCache(){int device=0;cudaGetDevice(&device);cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,device);supported=std::strcmp(prop.name,"NVIDIA L4")==0;if(supported && cublasLtCreate(&handle)!=CUBLAS_STATUS_SUCCESS)supported=false;}
 ~LtCache(){plans.clear();if(workspace)cudaFree(workspace);if(handle)cublasLtDestroy(handle);}
};
int tuned_rank(size_t m,size_t n,size_t k){
 if(k==896 && n==128){if(m==1||m==32)return 0;if(m==2||m==4||m==8)return 5;if(m==64||m==256)return 1;if(m==128)return 2;}
 if(n==896 && k==4864){if(m<=8)return 1;if(m==32)return 2;if(m==64)return 4;}
 if(n==896 && k==896){if(m==4)return 0;if(m==32)return 3;if(m==64)return 1;}
 if(n==4864 && k==896){if(m==32)return 3;if(m==256)return 1;}
 return -1;
}
bool try_lt(size_t m,size_t n,size_t k,const void* x,size_t ldx,const void* w,size_t ldw,float* y,size_t ldy){
 const char* flag=std::getenv("FORGE_GEMM_LT");if(!flag||std::strcmp(flag,"1"))return false;
 int rank=tuned_rank(m,n,k);if(rank<0)return false;
 static thread_local LtCache cache;if(!cache.supported)return false;
 cudaStreamCaptureStatus capture; if(cudaStreamIsCapturing(cudaStreamPerThread,&capture)!=cudaSuccess)return false;
 if(!cache.workspace){if(capture!=cudaStreamCaptureStatusNone)return false;if(cudaMalloc(&cache.workspace,32ull<<20)!=cudaSuccess){cudaGetLastError();return false;}}
 std::array<size_t,6> key{m,n,k,ldx,ldw,ldy};auto found=cache.plans.find(key);
 if(found==cache.plans.end()){
  if(capture!=cudaStreamCaptureStatusNone)return false;
  auto &p=cache.plans[key];
  if(cublasLtMatmulDescCreate(&p.desc,CUBLAS_COMPUTE_32F,CUDA_R_32F)!=CUBLAS_STATUS_SUCCESS)return false;
  cublasOperation_t op=CUBLAS_OP_T;
  if(cublasLtMatmulDescSetAttribute(p.desc,CUBLASLT_MATMUL_DESC_TRANSA,&op,sizeof(op))!=CUBLAS_STATUS_SUCCESS)return false;
  if(cublasLtMatrixLayoutCreate(&p.a,CUDA_R_16BF,k,n,ldw)!=CUBLAS_STATUS_SUCCESS||cublasLtMatrixLayoutCreate(&p.b,CUDA_R_16BF,k,m,ldx)!=CUBLAS_STATUS_SUCCESS||cublasLtMatrixLayoutCreate(&p.c,CUDA_R_32F,n,m,ldy)!=CUBLAS_STATUS_SUCCESS)return false;
  cublasLtMatmulPreference_t pref;
  if(cublasLtMatmulPreferenceCreate(&pref)!=CUBLAS_STATUS_SUCCESS)return false;
  size_t bytes=32ull<<20;cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
  cublasLtMatmulHeuristicResult_t alg[8];int count=0;
  auto status=cublasLtMatmulAlgoGetHeuristic(cache.handle,p.desc,p.a,p.b,p.c,p.c,pref,8,alg,&count);cublasLtMatmulPreferenceDestroy(pref);
  if(status==CUBLAS_STATUS_SUCCESS && count>rank && alg[rank].state==CUBLAS_STATUS_SUCCESS){p.algo=alg[rank].algo;p.valid=true;}
  found=cache.plans.find(key);
 }
 auto &p=found->second;if(!p.valid)return false;
 float alpha=1,beta=0;
 return cublasLtMatmul(cache.handle,p.desc,&alpha,w,p.a,x,p.b,&beta,y,p.c,y,p.c,&p.algo,cache.workspace,32ull<<20,cudaStreamPerThread)==CUBLAS_STATUS_SUCCESS;
}

// Growing BF16 activation scratch for linear_bf16_w (avoids per-call malloc).
forge_cuda_buffer& x_bf16_scratch() {
  static forge_cuda_buffer b{nullptr, 0};
  return b;
}

__global__ void add_bias_rows_kernel(size_t m, size_t n, const float* bias,
                                     float* y, size_t ldy) {
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) {
    return;
  }
  y[i * ldy + j] += bias[j];
}

__global__ void f32_to_bf16_kernel(size_t n, const float* src,
                                   __nv_bfloat16* dst) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = __float2bfloat16(src[i]);
  }
}

forge_cuda_status ensure_x_bf16(size_t bytes) {
  forge_cuda_buffer& b = x_bf16_scratch();
  if (b.ptr != nullptr && b.bytes >= bytes) {
    return FORGE_CUDA_OK;
  }
  if (b.ptr != nullptr) {
    forge_cuda_free(&b);
  }
  return forge_cuda_malloc(&b, bytes);
}

forge_cuda_status add_bias_if(size_t m, size_t n, const float* bias, float* y,
                              size_t ldy) {
  if (bias == nullptr) {
    return FORGE_CUDA_OK;
  }
  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(n) + block.x - 1) / block.x,
            (static_cast<unsigned>(m) + block.y - 1) / block.y);
  add_bias_rows_kernel<<<grid, block>>>(m, n, bias, y, ldy);
  return forge::cuda_detail::launch_check("add_bias_rows_kernel");
}

}  // namespace

extern "C" forge_cuda_status forge_cuda_linear_f32(
    size_t m, size_t n, size_t k, const float* x, size_t ldx, const float* w,
    size_t ldw, const float* bias, float* y, size_t ldy) {
  forge::cuda_detail::ProfileRegion profile("gemm");
  if (x == nullptr || w == nullptr || y == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || k == 0 || ldx < k || ldw < k || ldy < n) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  // Row-major Y = X @ W^T via cuBLAS: Y^T = W @ X^T with OP_T on W.
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t st = cublasSgemm(
      cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(k), &alpha, w,
      static_cast<int>(ldw), x, static_cast<int>(ldx), &beta, y,
      static_cast<int>(ldy));
  if (st != CUBLAS_STATUS_SUCCESS) {
    forge::cuda_detail::set_error("cublasSgemm(linear)", cudaErrorUnknown);
    return FORGE_CUDA_ERR_INTERNAL;
  }
  return add_bias_if(m, n, bias, y, ldy);
}

extern "C" forge_cuda_status forge_cuda_f32_to_bf16(size_t n, const float* src,
                                                   void* dst_bf16) {
  if (src == nullptr || dst_bf16 == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (n == 0) {
    return FORGE_CUDA_OK;
  }
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  f32_to_bf16_kernel<<<grid, block>>>(n, src,
                                      static_cast<__nv_bfloat16*>(dst_bf16));
  return forge::cuda_detail::launch_check("f32_to_bf16_kernel");
}

extern "C" forge_cuda_status forge_cuda_linear_bf16_w(
    size_t m, size_t n, size_t k, const float* x, size_t ldx, const void* w_bf16,
    size_t ldw, const float* bias, float* y, size_t ldy) {
  forge::cuda_detail::ProfileRegion profile("gemm_cast");
  if (x == nullptr || w_bf16 == nullptr || y == nullptr) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }
  if (m == 0 || n == 0 || k == 0 || ldx < k || ldw < k || ldy < n) {
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  }

  // Cast activations to BF16; GemmEx BF16×BF16→FP32 (tensor-core friendly on L4).
  const size_t x_elems = m * ldx;
  const size_t need = x_elems * sizeof(__nv_bfloat16);
  forge_cuda_status st_alloc = ensure_x_bf16(need);
  if (st_alloc != FORGE_CUDA_OK) {
    return st_alloc;
  }
  auto* x_bf = static_cast<__nv_bfloat16*>(x_bf16_scratch().ptr);
  // Contiguous cast assumes ldx == k for the used region; copy row-wise if ldx > k.
  if (ldx == k) {
    forge_cuda_status cst =
        forge_cuda_f32_to_bf16(m * k, x, x_bf);
    if (cst != FORGE_CUDA_OK) {
      return cst;
    }
  } else {
    for (size_t i = 0; i < m; ++i) {
      forge_cuda_status cst =
          forge_cuda_f32_to_bf16(k, x + i * ldx, x_bf + i * k);
      if (cst != FORGE_CUDA_OK) {
        return cst;
      }
    }
  }

  return forge_cuda_linear_bf16_xw(m,n,k,x_bf,k,w_bf16,ldw,bias,y,ldy);
}

extern "C" forge_cuda_status forge_cuda_linear_bf16_xw(
    size_t m,size_t n,size_t k,const void* x_bf,size_t ldx,const void* w_bf16,
    size_t ldw,const float* bias,float* y,size_t ldy) {
  forge::cuda_detail::ProfileRegion profile("gemm");
  if (!m || !n || !k || !x_bf || !w_bf16 || !y || ldx<k || ldw<k || ldy<n)
    return FORGE_CUDA_ERR_INVALID_ARGUMENT;
  if (try_lt(m,n,k,x_bf,ldx,w_bf16,ldw,y,ldy)) return add_bias_if(m,n,bias,y,ldy);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t st = cublasGemmEx(
      cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n),
      static_cast<int>(m), static_cast<int>(k), &alpha, w_bf16, CUDA_R_16BF,
      static_cast<int>(ldw), x_bf, CUDA_R_16BF, static_cast<int>(ldx), &beta, y,
      CUDA_R_32F, static_cast<int>(ldy), CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) {
    forge::cuda_detail::set_error("cublasGemmEx(linear_bf16)", cudaErrorUnknown);
    return FORGE_CUDA_ERR_INTERNAL;
  }
  return add_bias_if(m, n, bias, y, ldy);
}

// Graph replay must invalidate captures when eager prefill grows this scratch.
extern "C" uint64_t forge_cuda_bf16_scratch_address(void) {
  return reinterpret_cast<uint64_t>(x_bf16_scratch().ptr);
}

// Qwen2.5 shapes, BF16 inputs / FP32 output and accumulation. CUDA-event timing.
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#define CUDA_OK(x) do {if((x)!=cudaSuccess) return 2;}while(0)
int main() {
 cublasHandle_t h; cublasCreate(&h);cublasSetStream(h,cudaStreamPerThread);
 cublasLtHandle_t lt;cublasLtCreate(&lt);
 void* work;CUDA_OK(cudaMalloc(&work,32ull<<20));
 cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
 for(int m: {1,2,4,8,32,64,128,256}) for(auto shape: std::vector<std::pair<int,int>>{{896,896},{128,896},{4864,896},{896,4864},{151936,896}}) {
 int n=shape.first,k=shape.second;if(n==151936 && m>8)continue;
 std::vector<__nv_bfloat16> wx((size_t)n*k),xx((size_t)m*k);
 for(size_t j=0;j<wx.size();j++)wx[j]=__float2bfloat16(((int)(j%43)-21)*0.001f);
 for(size_t j=0;j<xx.size();j++)xx[j]=__float2bfloat16(((int)(j%37)-18)*0.01f);
 __nv_bfloat16 *w,*x;float *y;
 CUDA_OK(cudaMalloc(&w,wx.size()*2));CUDA_OK(cudaMalloc(&x,xx.size()*2));CUDA_OK(cudaMalloc(&y,(size_t)n*m*4));
 CUDA_OK(cudaMemcpy(w,wx.data(),wx.size()*2,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(x,xx.data(),xx.size()*2,cudaMemcpyHostToDevice));
 float alpha=1,beta=0;
 auto base=[&](){return cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,&beta,y,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);};
 for(int j=0;j<10;j++)if(base()!=CUBLAS_STATUS_SUCCESS)return 3;
 CUDA_OK(cudaDeviceSynchronize());
 std::vector<float> oracle((size_t)n*m),out(oracle.size());CUDA_OK(cudaMemcpy(oracle.data(),y,oracle.size()*4,cudaMemcpyDeviceToHost));
 auto bench=[&](auto call){std::vector<float> times;for(int r=0;r<3;r++){cudaEventRecord(a,cudaStreamPerThread);for(int j=0;j<40;j++)if(call()!=CUBLAS_STATUS_SUCCESS)return -1.f;cudaEventRecord(b,cudaStreamPerThread);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);times.push_back(ms/40);}std::sort(times.begin(),times.end());return times[1];};
 float base_ms=bench(base);
 cublasLtMatmulDesc_t desc;cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F);cublasOperation_t op=CUBLAS_OP_T;cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&op,sizeof(op));
 cublasLtMatrixLayout_t wa,xb,yc;cublasLtMatrixLayoutCreate(&wa,CUDA_R_16BF,k,n,k);cublasLtMatrixLayoutCreate(&xb,CUDA_R_16BF,k,m,k);cublasLtMatrixLayoutCreate(&yc,CUDA_R_32F,n,m,n);
 cublasLtMatmulPreference_t pref;cublasLtMatmulPreferenceCreate(&pref);size_t bytes=32ull<<20;cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
 cublasLtMatmulHeuristicResult_t alg[8];int count=0;auto lookup=std::chrono::steady_clock::now();cublasLtMatmulAlgoGetHeuristic(lt,desc,wa,xb,yc,yc,pref,8,alg,&count);double lookup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-lookup).count();
 for(int j=0;j<count;j++) {
 auto call=[&](){return cublasLtMatmul(lt,desc,&alpha,w,wa,x,xb,&beta,y,yc,y,yc,&alg[j].algo,work,bytes,cudaStreamPerThread);};
 float ms=bench(call);if(ms<0)continue;CUDA_OK(cudaMemcpy(out.data(),y,out.size()*4,cudaMemcpyDeviceToHost));float err=0;bool valid=true;for(size_t i=0;i<out.size();i++){err=std::max(err,std::abs(out[i]-oracle[i]));valid &= std::isfinite(out[i])&&std::abs(out[i]-oracle[i])<=1e-4f+1e-3f*std::abs(oracle[i]);}
 printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"base_ms\":%.6f,\"lt_rank\":%d,\"lt_ms\":%.6f,\"max_abs_error\":%.8f,\"valid\":%s,\"heuristic_ms\":%.4f}\n",m,n,k,base_ms,j,ms,err,valid?"true":"false",lookup_ms);
 }
 cublasLtMatmulPreferenceDestroy(pref);cublasLtMatrixLayoutDestroy(wa);cublasLtMatrixLayoutDestroy(xb);cublasLtMatrixLayoutDestroy(yc);cublasLtMatmulDescDestroy(desc);cudaFree(w);cudaFree(x);cudaFree(y);
 }
 cudaFree(work);cublasLtDestroy(lt);cublasDestroy(h);cudaEventDestroy(a);cudaEventDestroy(b);
}

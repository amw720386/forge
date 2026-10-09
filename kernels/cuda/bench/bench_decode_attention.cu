#include "forge_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

static void check(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);} }
int main() {
  cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
  for(size_t tk:{31,128,512,1024,2048,4096}) for(size_t tq:{1,3,8,32,128}) {
    if(tq>tk)continue;
    size_t bs=16,blocks=(tk+bs-1)/bs;
    std::vector<uint32_t> table(blocks);
    for(size_t i=0;i<blocks;i++)table[i]=blocks-i-1;
    std::vector<float> q(tq*896),kv(blocks*bs*128);
    for(size_t i=0;i<q.size();i++)q[i]=((int)(i%31)-15)*.01f;
    for(size_t i=0;i<kv.size();i++)kv[i]=((int)(i%37)-18)*.02f;
    float *dq,*dk,*dv,*out;uint32_t*dt;uint64_t*dm;
    check(cudaMalloc(&dq,q.size()*4));check(cudaMalloc(&dk,kv.size()*4));
    check(cudaMalloc(&dv,kv.size()*4));check(cudaMalloc(&out,q.size()*4));
    check(cudaMalloc(&dt,table.size()*4));check(cudaMalloc(&dm,3*8));
    check(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dk,kv.data(),kv.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dv,kv.data(),kv.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dt,table.data(),table.size()*4,cudaMemcpyHostToDevice));
    uint64_t metadata[3]={(uint64_t)dt,tk-1,tk-1};
    check(cudaMemcpy(dm,metadata,24,cudaMemcpyHostToDevice));
    std::vector<float> ref(q.size()),got(q.size());
    for(int mode=0;mode<2;mode++) {
      setenv("FORGE_PREFILL_ATTN_BLOCK",mode?"1":"0",1);
      setenv("FORGE_DECODE_ATTN_BLOCK",mode?"1":"0",1);
      auto call=[&](){
        if(tq==1)return forge_cuda_batch_paged_attn_f32(1,14,2,64,.125f,dm,dq,dk,dv,out,bs,blocks,0);
        return forge_cuda_paged_gqa_attn_f32(tq,tk,14,2,64,.125f,dq,896,dk,dv,dt,blocks,bs,blocks,1,0,128,out,1,tk-tq);
      };
      for(int i=0;i<5;i++)if(call()!=FORGE_CUDA_OK)return 2;
      std::vector<float> trials;
      for(int r=0;r<3;r++) {
        check(cudaEventRecord(a,cudaStreamPerThread));
        for(int i=0;i<20;i++)if(call()!=FORGE_CUDA_OK)return 2;
        check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));
        float ms;check(cudaEventElapsedTime(&ms,a,b));trials.push_back(ms/20);
      }
      check(cudaMemcpy(got.data(),out,q.size()*4,cudaMemcpyDeviceToHost));
      if(mode==0)ref=got;
      float err=0;
      for(size_t i=0;i<got.size();i++) {
        err=std::max(err,std::abs(got[i]-ref[i]));
        if(!std::isfinite(got[i])||std::abs(got[i]-ref[i])>1e-4f+1e-3f*std::abs(ref[i]))return 3;
      }
      std::sort(trials.begin(),trials.end());
      printf("{\"tq\":%zu,\"tk\":%zu,\"mode\":%d,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"max_abs_error\":%.8f}\n",tq,tk,mode,trials[1],trials[0],trials[2],err);
      fflush(stdout);
    }
    cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dt);cudaFree(dm);cudaFree(out);
  }
  cudaEventDestroy(a);cudaEventDestroy(b);
}

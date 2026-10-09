#include "forge_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(FORGE_CUDA_HAS_CPU_REF)
#include "forge_kernels.h"
#endif

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++failures;
  }
}

void expect_close(float a, float b, float atol, float rtol, const char* msg) {
  float diff = std::fabs(a - b);
  float tol = atol + rtol * std::fabs(b);
  if (diff > tol) {
    std::fprintf(stderr, "FAIL: %s got=%g ref=%g diff=%g tol=%g\n", msg, a, b,
                 diff, tol);
    ++failures;
  }
}

struct Buf {
  forge_cuda_buffer b{};
  explicit Buf(size_t bytes) {
    expect(forge_cuda_malloc(&b, bytes) == FORGE_CUDA_OK, "malloc");
  }
  ~Buf() { forge_cuda_free(&b); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  float* f() { return static_cast<float*>(b.ptr); }
  const float* f() const { return static_cast<const float*>(b.ptr); }
};

}  // namespace

void test_cuda_ops() {
  // Invalid shapes
  {
    Buf a(16), x(16), y(16);
    expect(forge_cuda_matvec_f32(0, 4, a.f(), 4, x.f(), y.f()) ==
               FORGE_CUDA_ERR_INVALID_ARGUMENT,
           "matvec zero m");
    expect(forge_cuda_malloc(nullptr, 16) == FORGE_CUDA_ERR_INVALID_ARGUMENT,
           "malloc null");
  }

  // Alloc / free cycle
  for (int i = 0; i < 32; ++i) {
    forge_cuda_buffer b{};
    expect(forge_cuda_malloc(&b, 1024 * 1024) == FORGE_CUDA_OK, "cycle malloc");
    expect(forge_cuda_memset_zero(&b) == FORGE_CUDA_OK, "cycle zero");
    expect(forge_cuda_free(&b) == FORGE_CUDA_OK, "cycle free");
  }

  // Matmul small hand-checked: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
  {
    const size_t m = 2, n = 2, k = 2;
    std::vector<float> ha = {1, 2, 3, 4};
    std::vector<float> hb = {5, 6, 7, 8};
    std::vector<float> hc(4, 0.f);
    Buf da(ha.size() * 4), db(hb.size() * 4), dc(hc.size() * 4);
    forge_cuda_memcpy_h2d(&da.b, ha.data(), ha.size() * 4);
    forge_cuda_memcpy_h2d(&db.b, hb.data(), hb.size() * 4);
    expect(forge_cuda_matmul_f32(m, n, k, da.f(), k, db.f(), n, dc.f(), n) ==
               FORGE_CUDA_OK,
           "matmul");
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(hc.data(), &dc.b, hc.size() * 4);
    expect_close(hc[0], 19.f, 1e-5f, 1e-4f, "mm00");
    expect_close(hc[1], 22.f, 1e-5f, 1e-4f, "mm01");
    expect_close(hc[2], 43.f, 1e-5f, 1e-4f, "mm10");
    expect_close(hc[3], 50.f, 1e-5f, 1e-4f, "mm11");
  }

  // Matvec
  {
    std::vector<float> ha = {1, 2, 3, 4, 5, 6};  // 2x3
    std::vector<float> hx = {1, 1, 1};
    std::vector<float> hy(2, 0.f);
    Buf da(ha.size() * 4), dx(hx.size() * 4), dy(hy.size() * 4);
    forge_cuda_memcpy_h2d(&da.b, ha.data(), ha.size() * 4);
    forge_cuda_memcpy_h2d(&dx.b, hx.data(), hx.size() * 4);
    expect(forge_cuda_matvec_f32(2, 3, da.f(), 3, dx.f(), dy.f()) ==
               FORGE_CUDA_OK,
           "matvec");
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(hy.data(), &dy.b, hy.size() * 4);
    expect_close(hy[0], 6.f, 1e-5f, 1e-4f, "mv0");
    expect_close(hy[1], 15.f, 1e-5f, 1e-4f, "mv1");
  }

  // SiLU / add / mul
  {
    std::vector<float> hx = {0.f, 2.f};
    std::vector<float> hy(2);
    Buf dx(8), dy(8);
    forge_cuda_memcpy_h2d(&dx.b, hx.data(), 8);
    forge_cuda_silu_f32(2, dx.f(), dy.f());
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(hy.data(), &dy.b, 8);
    expect_close(hy[0], 0.f, 1e-5f, 1e-4f, "silu0");
    float ref = 2.f / (1.f + std::exp(-2.f));
    expect_close(hy[1], ref, 1e-5f, 1e-4f, "silu1");
  }

  // RMSNorm vs CPU when available
  {
    const size_t dim = 4;
    std::vector<float> hx = {1, -1, 2, -2};
    std::vector<float> hw = {1, 1, 1, 1};
    std::vector<float> hy(4), href(4);
    Buf dx(16), dw(16), dy(16);
    forge_cuda_memcpy_h2d(&dx.b, hx.data(), 16);
    forge_cuda_memcpy_h2d(&dw.b, hw.data(), 16);
    expect(forge_cuda_rmsnorm_f32(1, dim, dx.f(), dim, dw.f(), 1e-6f, dy.f(),
                                  dim) == FORGE_CUDA_OK,
           "rmsnorm");
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(hy.data(), &dy.b, 16);
#if defined(FORGE_CUDA_HAS_CPU_REF)
    forge_kernels_rmsnorm_f32(1, dim, hx.data(), dim, hw.data(), 1e-6f,
                              href.data(), dim);
    for (size_t i = 0; i < dim; ++i) {
      expect_close(hy[i], href[i], 1e-5f, 1e-4f, "rmsnorm vs cpu");
    }
#else
    float mean_sq = (1 + 1 + 4 + 4) / 4.f;
    float scale = 1.f / std::sqrt(mean_sq + 1e-6f);
    for (size_t i = 0; i < dim; ++i) {
      expect_close(hy[i], hx[i] * scale, 1e-5f, 1e-4f, "rmsnorm ref");
    }
#endif
  }

  // Larger non-square matmul vs CPU
  {
    const size_t m = 17, n = 13, k = 19;
    std::vector<float> ha(m * k), hb(k * n), hc(m * n), href(m * n);
    for (size_t i = 0; i < ha.size(); ++i) {
      ha[i] = static_cast<float>((i % 11) - 5) * 0.1f;
    }
    for (size_t i = 0; i < hb.size(); ++i) {
      hb[i] = static_cast<float>((i % 7) - 3) * 0.05f;
    }
    Buf da(ha.size() * 4), db(hb.size() * 4), dc(hc.size() * 4);
    forge_cuda_memcpy_h2d(&da.b, ha.data(), ha.size() * 4);
    forge_cuda_memcpy_h2d(&db.b, hb.data(), hb.size() * 4);
    forge_cuda_matmul_f32(m, n, k, da.f(), k, db.f(), n, dc.f(), n);
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(hc.data(), &dc.b, hc.size() * 4);
#if defined(FORGE_CUDA_HAS_CPU_REF)
    forge_kernels_matmul_f32(m, n, k, ha.data(), k, hb.data(), n, href.data(),
                             n);
    for (size_t i = 0; i < hc.size(); ++i) {
      expect_close(hc[i], href[i], 1e-4f, 1e-3f, "large matmul");
    }
#endif
  }

  // Softmax causal 2x2
  {
    std::vector<float> hs = {1.f, 2.f, 3.f, 4.f};
    Buf ds(16);
    forge_cuda_memcpy_h2d(&ds.b, hs.data(), 16);
    forge_cuda_softmax_f32(1, 2, 2, ds.f(), 1, 0);
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(hs.data(), &ds.b, 16);
    expect_close(hs[0], 1.f, 1e-5f, 1e-4f, "sm00");
    expect_close(hs[1], 0.f, 1e-5f, 1e-4f, "sm01");
  }

  // Argmax rows (tie → lowest index)
  {
    std::vector<float> h = {1.f, 5.f, 5.f, 3.f,  // row0 → idx 1
                            9.f, 2.f, 9.f, 1.f}; // row1 → idx 0
    std::vector<int32_t> ids(2, -1);
    Buf d(h.size() * 4), dout(2 * sizeof(int32_t));
    forge_cuda_memcpy_h2d(&d.b, h.data(), h.size() * 4);
    expect(forge_cuda_argmax_rows_f32(2, 4, d.f(),
                                      static_cast<int32_t*>(dout.b.ptr)) ==
               FORGE_CUDA_OK,
           "argmax");
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(ids.data(), &dout.b, 2 * sizeof(int32_t));
    expect(ids[0] == 1, "argmax row0");
    expect(ids[1] == 0, "argmax row1 tie");
  }

  // Linear (cuBLAS path) small hand-check: y = x @ W^T
  // W = [[1,2],[3,4]] (2×2), x = [1,1] → y = [3, 7]
  {
    std::vector<float> hx = {1.f, 1.f};
    std::vector<float> hw = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> hy(2, 0.f);
    Buf dx(8), dw(16), dy(8);
    forge_cuda_memcpy_h2d(&dx.b, hx.data(), 8);
    forge_cuda_memcpy_h2d(&dw.b, hw.data(), 16);
    expect(forge_cuda_linear_f32(1, 2, 2, dx.f(), 2, dw.f(), 2, nullptr, dy.f(),
                                 2) == FORGE_CUDA_OK,
           "linear");
    forge_cuda_synchronize();
    forge_cuda_memcpy_d2h(hy.data(), &dy.b, 8);
    expect_close(hy[0], 3.f, 1e-5f, 1e-4f, "lin0");
    expect_close(hy[1], 7.f, 1e-5f, 1e-4f, "lin1");
  }

  // Paged GQA: opt vs naive numerical agreement
  {
    const size_t n_heads = 4, n_kv = 2, d = 8;
    const size_t q_w = n_heads * d, kv_w = n_kv * d;
    const size_t t_q = 2, t_k = 20, block_size = 16;
    const size_t n_layers = 1, num_blocks = 4, layer = 0;
    const size_t need = (t_k + block_size - 1) / block_size;
    float scale = 1.f / std::sqrt(static_cast<float>(d));
    std::vector<float> hq(t_q * q_w), hk(t_k * kv_w), hv(t_k * kv_w);
    for (size_t i = 0; i < hq.size(); ++i) {
      hq[i] = 0.01f * static_cast<float>(i);
    }
    for (size_t i = 0; i < hk.size(); ++i) {
      hk[i] = 0.1f + 0.01f * static_cast<float>(i);
      hv[i] = 0.5f + 0.01f * static_cast<float>(i);
    }
    std::vector<uint32_t> table(need);
    for (size_t i = 0; i < need; ++i) {
      table[i] = static_cast<uint32_t>(i);
    }
    size_t pool_elems = n_layers * num_blocks * block_size * kv_w;
    Buf dq(hq.size() * 4), dkpool(pool_elems * 4), dvpool(pool_elems * 4),
        dtable(need * 4), dctx_opt(hq.size() * 4), dctx_nv(hq.size() * 4),
        dksrc(hk.size() * 4), dvsrc(hv.size() * 4);
    forge_cuda_memset_zero(&dkpool.b);
    forge_cuda_memset_zero(&dvpool.b);
    forge_cuda_memcpy_h2d(&dq.b, hq.data(), hq.size() * 4);
    forge_cuda_memcpy_h2d(&dksrc.b, hk.data(), hk.size() * 4);
    forge_cuda_memcpy_h2d(&dvsrc.b, hv.data(), hv.size() * 4);
    forge_cuda_memcpy_h2d(&dtable.b, table.data(), need * 4);
    expect(forge_cuda_paged_kv_append_f32(
               kv_w, t_k, 0, block_size, num_blocks, n_layers, layer,
               static_cast<const uint32_t*>(dtable.b.ptr), need, dksrc.f(),
               dvsrc.f(), dkpool.f(), dvpool.f()) == FORGE_CUDA_OK,
           "paged append");
    expect(forge_cuda_paged_gqa_attn_f32(
               t_q, t_k, n_heads, n_kv, d, scale, dq.f(), q_w, dkpool.f(),
               dvpool.f(), static_cast<const uint32_t*>(dtable.b.ptr), need,
               block_size, num_blocks, n_layers, layer, kv_w, dctx_opt.f(), 1,
               0) == FORGE_CUDA_OK,
           "paged opt");
    expect(forge_cuda_paged_gqa_attn_naive_f32(
               t_q, t_k, n_heads, n_kv, d, scale, dq.f(), q_w, dkpool.f(),
               dvpool.f(), static_cast<const uint32_t*>(dtable.b.ptr), need,
               block_size, num_blocks, n_layers, layer, kv_w, dctx_nv.f(), 1,
               0) == FORGE_CUDA_OK,
           "paged naive");
    forge_cuda_synchronize();
    std::vector<float> hopt(hq.size()), hnv(hq.size());
    forge_cuda_memcpy_d2h(hopt.data(), &dctx_opt.b, hq.size() * 4);
    forge_cuda_memcpy_d2h(hnv.data(), &dctx_nv.b, hq.size() * 4);
    for (size_t i = 0; i < hopt.size(); ++i) {
      expect_close(hopt[i], hnv[i], 1e-4f, 1e-3f, "paged opt vs naive");
    }
  }

  // Mem info
  {
    size_t free_b = 0, total_b = 0;
    expect(forge_cuda_mem_info(&free_b, &total_b) == FORGE_CUDA_OK, "mem_info");
    expect(total_b > 0, "total mem > 0");
    expect(free_b <= total_b, "free <= total");
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d CUDA test failure(s)\n", failures);
    std::exit(1);
  }
}

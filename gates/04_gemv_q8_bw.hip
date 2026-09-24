// Gate 4 — achieved HBM bandwidth of a q8_0 GEMV on the exact weight shapes TP8 produces.
//
// Under TP8 each routed expert is sliced to three 1.67 MB matrices (gate/up: 256 x 6144, down:
// 6144 x 256); the design's per-layer expert work is the fused 9-expert shapes (gate+up 4608 x 6144
// = 30 MB, down 6144 x 2304 = 15 MB); 6144 x 6144 (40 MB) is the reference. The design assumes
// ~500 GB/s effective, measured on large dense matrices. Two numbers are reported per shape:
//   per-launch : N back-to-back launches, INCLUDES the same-stream dispatch gap (~5-10 us on gfx906)
//                = the cost of NOT fusing; dominates the 1.67 MB slices.
//   in-kernel  : ONE launch that loops over N matrices of the ring = the persistent-kernel regime the
//                design runs in (an upper bound: a fused kernel still pays a barrier between GEMVs).
// A float4 streaming read of the same ring, timed the same two ways, is printed as the read ceiling:
// a GEMV number far below it is the kernel, not the memory system. A ring of matrices (>= 768 MB)
// defeats the 4 MB L2 between launches.
//
// Kernels: 'dot' (default) quantises x to q8_1 and uses v_dot4_i32_i8 (gfx906 has it; llama.cpp's
// path), all loads of a lane hoisted; 'naive' is the plain fp32 version for comparison.
//
//   ./bin/gemv_q8_bw [--shapes 256x6144,6144x256,4608x6144,6144x2304,6144x6144] [--reps 200] [--kernel dot|naive] [--ring-mb 768]
//   ./bin/gemv_q8_bw --shape 6144x6144 --seconds 600     # soak mode (gate 5): per-launch loop, prints every 10 s
//   ./bin/gemv_q8_bw --count                              # print device count
#include "common.h"
#include <hip/hip_fp16.h>
#include <chrono>

static int g_fail = 0;

// q8_0 block: fp16 scale + 32 int8 = 34 bytes per 32 weights (1.0625 B/weight), rows contiguous.

// ---- naive fp32 kernel (comparison) ----
__device__ __forceinline__ float row_naive(const uint8_t* __restrict__ W, const float* __restrict__ x, int row, int nb,
                                           int lpr, int b0) {
  const uint8_t* rowp = W + (size_t)row * (size_t)nb * 34;
  float acc = 0.f;
  for (int b = b0; b < nb; b += lpr) {
    const uint8_t* blk = rowp + (size_t)b * 34;
    unsigned short dbits;
    __builtin_memcpy(&dbits, blk, 2);
    const float d = __half2float(__ushort_as_half(dbits));
    const int8_t* q = (const int8_t*)(blk + 2);
    const float* xb = x + b * 32;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) s += (float)q[j] * xb[j];
    acc += d * s;
  }
  return acc;
}

// ---- q8_1 x + sdot4 kernel (default). NBL = blocks per lane (nb / lpr) when known at compile time
// (every load of the lane is issued before any use); NBL = 0 -> runtime loop. ----
template <int NBL>
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq,
                                         const float* __restrict__ xs, int row, int nb, int lpr, int b0) {
  const uint8_t* rowp = W + (size_t)row * (size_t)nb * 34;
  float acc = 0.f;
  if constexpr (NBL > 0) {
    int w[NBL][8], xv[NBL][8];
    unsigned short db[NBL];
    float sx[NBL];
#pragma unroll
    for (int k = 0; k < NBL; ++k) {  // phase 1: every load, no use in between
      const int b = b0 + k * lpr;
      const uint8_t* blk = rowp + (size_t)b * 34;
      __builtin_memcpy(&db[k], blk, 2);
#pragma unroll
      for (int j = 0; j < 8; ++j) __builtin_memcpy(&w[k][j], blk + 2 + 4 * j, 4);  // 2-B aligned source: HSA runs with unaligned-access-mode
#pragma unroll
      for (int j = 0; j < 8; ++j) __builtin_memcpy(&xv[k][j], xq + (size_t)b * 32 + 4 * j, 4);
      sx[k] = xs[b];
    }
#pragma unroll
    for (int k = 0; k < NBL; ++k) {  // phase 2: 8 v_dot4_i32_i8 per block
      int s = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) s = __builtin_amdgcn_sdot4(w[k][j], xv[k][j], s, false);
      acc += __half2float(__ushort_as_half(db[k])) * sx[k] * (float)s;
    }
  } else {
    for (int b = b0; b < nb; b += lpr) {
      const uint8_t* blk = rowp + (size_t)b * 34;
      unsigned short dbits;
      __builtin_memcpy(&dbits, blk, 2);
      int s = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int w, xv;
        __builtin_memcpy(&w, blk + 2 + 4 * j, 4);
        __builtin_memcpy(&xv, xq + (size_t)b * 32 + 4 * j, 4);
        s = __builtin_amdgcn_sdot4(w, xv, s, false);
      }
      acc += __half2float(__ushort_as_half(dbits)) * xs[b] * (float)s;
    }
  }
  return acc;
}

// reps == 1 with n_mats == 1: a single GEMV; reps > 1: one launch walking `reps` matrices of the ring.
template <int NBL>
__global__ void k_dot(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq, const float* __restrict__ xs,
                      float* __restrict__ y, int N, int nb, int lpr, int n_mats, size_t mat_bytes, int reps) {
  const int lane = threadIdx.x & 63;
  const int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 6;
  const int rpw = 64 / lpr;
  const int row = wave * rpw + lane / lpr;
  const int b0 = lane % lpr;
  float tot = 0.f;
  for (int r = 0; r < reps; ++r) {
    float acc = row < N ? row_dot<NBL>(W + (size_t)(r % n_mats) * mat_bytes, xq, xs, row, nb, lpr, b0) : 0.f;
    for (int off = lpr >> 1; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 64);
    tot += acc;  // every rep stays live
  }
  if (row < N && b0 == 0) y[row] = tot;
}
__global__ void k_naive(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int nb,
                        int lpr, int n_mats, size_t mat_bytes, int reps) {
  const int lane = threadIdx.x & 63;
  const int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 6;
  const int rpw = 64 / lpr;
  const int row = wave * rpw + lane / lpr;
  const int b0 = lane % lpr;
  float tot = 0.f;
  for (int r = 0; r < reps; ++r) {
    float acc = row < N ? row_naive(W + (size_t)(r % n_mats) * mat_bytes, x, row, nb, lpr, b0) : 0.f;
    for (int off = lpr >> 1; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 64);
    tot += acc;
  }
  if (row < N && b0 == 0) y[row] = tot;
}

// Kernel-independent read ceiling over the same ring: a uint4 streaming read, one matrix per rep.
__global__ void stream_read(const uint8_t* __restrict__ base, size_t n16, int n_mats, size_t mat_bytes, int reps,
                            unsigned* __restrict__ sink) {
  unsigned acc = 0;
  for (int r = 0; r < reps; ++r) {
    const uint4* p = (const uint4*)(base + (size_t)(r % n_mats) * mat_bytes);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
      const uint4 v = p[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x9E3779B9u) sink[0] = acc;  // keeps the loads alive
}

static float half_bits_to_float(unsigned short h) {
  unsigned int s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v;
  if (e == 0) v = ldexpf((float)m, -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = ldexpf((float)(m | 0x400), (int)e - 25);
  return s ? -v : v;
}

struct Shape { int N, K; };
static bool parse_shape(const char* s, Shape* out) {
  return sscanf(s, "%dx%d", &out->N, &out->K) == 2 && out->K % 32 == 0 && out->N > 0 && out->K > 0 &&
         ((long long)out->N * (out->K / 32)) % 8 == 0;  // 16-B matrix stride for the uint4 read ceiling
}
static int pick_lpr(int nb) { int l = 64; while (l > 1 && nb % l) l >>= 1; return l; }  // 192->64, 8->8, 72->8

struct Ctx {
  int kind;  // 0 naive, 1 dot
  int nbl, blocks, threads;
  const uint8_t* W; const float* x; const int8_t* xq; const float* xs; float* y;
  int N, nb, lpr, n_mats; size_t mat_bytes;
};
static void launch_gemv(const Ctx& c, const uint8_t* W, int n_mats, int reps) {
  const dim3 g(c.blocks), b(c.threads);
  if (c.kind == 0) { hipLaunchKernelGGL(k_naive, g, b, 0, 0, W, c.x, c.y, c.N, c.nb, c.lpr, n_mats, c.mat_bytes, reps); return; }
  switch (c.nbl) {
    case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(k_dot<1>), g, b, 0, 0, W, c.xq, c.xs, c.y, c.N, c.nb, c.lpr, n_mats, c.mat_bytes, reps); break;
    case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(k_dot<2>), g, b, 0, 0, W, c.xq, c.xs, c.y, c.N, c.nb, c.lpr, n_mats, c.mat_bytes, reps); break;
    case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(k_dot<3>), g, b, 0, 0, W, c.xq, c.xs, c.y, c.N, c.nb, c.lpr, n_mats, c.mat_bytes, reps); break;
    case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(k_dot<4>), g, b, 0, 0, W, c.xq, c.xs, c.y, c.N, c.nb, c.lpr, n_mats, c.mat_bytes, reps); break;
    default: hipLaunchKernelGGL(HIP_KERNEL_NAME(k_dot<0>), g, b, 0, 0, W, c.xq, c.xs, c.y, c.N, c.nb, c.lpr, n_mats, c.mat_bytes, reps); break;
  }
}

static void run_shape(Shape sh, int kind, int reps, double seconds, size_t ring_target, uint32_t seed) {
  const int nb = sh.K / 32;
  const int lpr = pick_lpr(nb);
  const int nbl = nb / lpr;
  const size_t mat_bytes = (size_t)sh.N * nb * 34;
  const int n_mats = (int)std::max<size_t>(1, (ring_target + mat_bytes - 1) / mat_bytes);
  const size_t total = mat_bytes * n_mats;

  // host fill: cheap LCG; force every block's fp16 scale positive and finite (exponent 2^-3..2^-2)
  std::vector<uint8_t> h(total);
  uint32_t st = seed | 1u;
  for (size_t i = 0; i < total; ++i) { st = st * 1664525u + 1013904223u; h[i] = (uint8_t)(st >> 24); }
  for (int m = 0; m < n_mats; ++m)
    for (int r = 0; r < sh.N; ++r)
      for (int b = 0; b < nb; ++b) h[(size_t)m * mat_bytes + ((size_t)r * nb + b) * 34 + 1] = (uint8_t)(0x30 | (st = st * 1664525u + 1013904223u, (st >> 24) & 0x03));
  std::vector<float> hx(sh.K);
  for (int i = 0; i < sh.K; ++i) { st = st * 1664525u + 1013904223u; hx[i] = ((int)(st >> 16) % 2001 - 1000) / 1000.f; }
  // q8_1 quantisation of x (llama.cpp style): per 32-block scale = amax/127
  std::vector<int8_t> hxq(sh.K);
  std::vector<float> hxs(nb);
  for (int b = 0; b < nb; ++b) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, fabsf(hx[b * 32 + j]));
    const float s = amax > 0 ? amax / 127.f : 1.f;
    hxs[b] = s;
    for (int j = 0; j < 32; ++j) {
      int q = (int)lrintf(hx[b * 32 + j] / s);
      hxq[b * 32 + j] = (int8_t)std::max(-127, std::min(127, q));
    }
  }

  uint8_t* dW = nullptr; float *dx = nullptr, *dxs = nullptr, *dy = nullptr; int8_t* dxq = nullptr; unsigned* dsink = nullptr;
  HIP_CHECK(hipMalloc(&dW, total));
  HIP_CHECK(hipMalloc(&dx, (size_t)sh.K * 4));
  HIP_CHECK(hipMalloc(&dxq, (size_t)sh.K));
  HIP_CHECK(hipMalloc(&dxs, (size_t)nb * 4));
  HIP_CHECK(hipMalloc(&dy, (size_t)sh.N * 4));
  HIP_CHECK(hipMalloc(&dsink, 16));
  HIP_CHECK(hipMemcpy(dW, h.data(), total, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dx, hx.data(), (size_t)sh.K * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dxq, hxq.data(), (size_t)sh.K, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dxs, hxs.data(), (size_t)nb * 4, hipMemcpyHostToDevice));

  const int rpw = 64 / lpr;
  const int waves = (sh.N + rpw - 1) / rpw;
  const int threads = 256;
  const int blocks = (waves * 64 + threads - 1) / threads;
  Ctx c{kind, nbl, blocks, threads, dW, dx, dxq, dxs, dy, sh.N, nb, lpr, n_mats, mat_bytes};

  auto cleanup = [&]() {
    HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dxq)); HIP_CHECK(hipFree(dxs)); HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dsink));
  };

  // correctness spot-check on matrix 0, first rows, against a double reference (with the quantised x for 'dot')
  launch_gemv(c, dW, 1, 1);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> hy(sh.N);
  HIP_CHECK(hipMemcpy(hy.data(), dy, (size_t)sh.N * 4, hipMemcpyDeviceToHost));
  double max_rel = 0;
  for (int r = 0; r < std::min(sh.N, 64); ++r) {
    double ref = 0;
    for (int b = 0; b < nb; ++b) {
      const uint8_t* blk = h.data() + ((size_t)r * nb + b) * 34;
      const double d = half_bits_to_float((unsigned short)(blk[0] | (blk[1] << 8)));
      double s = 0;
      if (kind == 1) { for (int j = 0; j < 32; ++j) s += (double)(int8_t)blk[2 + j] * (double)hxq[b * 32 + j]; s *= hxs[b]; }
      else for (int j = 0; j < 32; ++j) s += (double)(int8_t)blk[2 + j] * hx[b * 32 + j];
      ref += d * s;
    }
    const double rel = fabs(ref - hy[r]) / std::max(fabs(ref), 1.0);  // absolute below |ref|=1: a cancelling row must not false-fail
    if (!(rel <= max_rel)) max_rel = rel;                              // NaN propagates
  }
  // A correct kernel measures ~1e-6..3e-5 here (fp32 vs double); a wrong row, dropped block or ignored scale lands >= 1e-3.
  if (!(max_rel <= 1e-3)) {
    printf("RESULT: CHECK FAILED shape=%dx%d kernel=%s max_rel_err=%.2e (limit 1e-3) — kernel output is wrong; bandwidth withheld\n",
           sh.N, sh.K, kind ? "dot" : "naive", max_rel);
    g_fail = 1;
    cleanup();
    return;
  }

  hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
  for (int i = 0; i < 20; ++i) launch_gemv(c, dW + (size_t)(i % n_mats) * mat_bytes, 1, 1);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());

  if (seconds > 0) {  // soak: per-launch loop, one report line every ~10 s (average since the previous report)
    auto t_start = std::chrono::steady_clock::now();
    long long launches = 0, launches_at_report = 0;
    double gpu_ms_since_report = 0, next_report = 10.0;
    while (true) {
      HIP_CHECK(hipEventRecord(e0, 0));
      for (int i = 0; i < 500; ++i) launch_gemv(c, dW + (size_t)((launches + i) % n_mats) * mat_bytes, 1, 1);
      HIP_CHECK(hipEventRecord(e1, 0));
      HIP_CHECK(hipEventSynchronize(e1));
      float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      launches += 500; gpu_ms_since_report += ms;
      const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
      if (el >= next_report || el >= seconds) {
        const long long n = launches - launches_at_report;
        printf("soak t=%.0fs shape=%dx%d GB/s=%.1f us/launch=%.1f\n", el, sh.N, sh.K,
               (double)mat_bytes * n / (gpu_ms_since_report * 1e-3) / 1e9, gpu_ms_since_report * 1000.0 / n);
        fflush(stdout);
        launches_at_report = launches; gpu_ms_since_report = 0; next_report += 10.0;
      }
      if (el >= seconds) break;
    }
  } else {
    // per-launch (includes dispatch gap)
    HIP_CHECK(hipEventRecord(e0, 0));
    for (int i = 0; i < reps; ++i) launch_gemv(c, dW + (size_t)(i % n_mats) * mat_bytes, 1, 1);
    HIP_CHECK(hipEventRecord(e1, 0));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipGetLastError());
    float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double us_l = (double)ms * 1000.0 / reps;
    // in-kernel (one launch looping reps matrices)
    HIP_CHECK(hipEventRecord(e0, 0));
    launch_gemv(c, dW, n_mats, reps);
    HIP_CHECK(hipEventRecord(e1, 0));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double us_k = (double)ms * 1000.0 / reps;
    // read ceiling, both ways
    int cus = 60; HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
    const size_t n16 = mat_bytes / 16;
    hipLaunchKernelGGL(stream_read, dim3(cus * 8), dim3(256), 0, 0, dW, n16, n_mats, mat_bytes, 1, dsink);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(e0, 0));
    for (int i = 0; i < reps; ++i) hipLaunchKernelGGL(stream_read, dim3(cus * 8), dim3(256), 0, 0, dW + (size_t)(i % n_mats) * mat_bytes, n16, 1, mat_bytes, 1, dsink);
    HIP_CHECK(hipEventRecord(e1, 0));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double us_rl = (double)ms * 1000.0 / reps;
    HIP_CHECK(hipEventRecord(e0, 0));
    hipLaunchKernelGGL(stream_read, dim3(cus * 8), dim3(256), 0, 0, dW, n16, n_mats, mat_bytes, reps, dsink);
    HIP_CHECK(hipEventRecord(e1, 0));
    HIP_CHECK(hipEventSynchronize(e1));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    const double us_rk = (double)ms * 1000.0 / reps;

    const double mb = mat_bytes / 1e6;
    auto gbs = [&](double us) { return (double)mat_bytes / (us * 1e-6) / 1e9; };
    printf("gate=4 shape=%dx%d bytes=%.2fMB ring=%d kernel=%s lanes/row=%d blocks/lane=%d waves=%d spot-check max_rel_err=%.1e\n"
           "  per-launch (%d launches, includes dispatch gap): us=%.1f GB/s=%.1f   read_ceiling GB/s=%.1f\n"
           "  in-kernel  (1 launch looping %d matrices):       us=%.1f GB/s=%.1f   read_ceiling GB/s=%.1f\n",
           sh.N, sh.K, mb, n_mats, kind ? "dot" : "naive", lpr, nbl, waves, max_rel,
           reps, us_l, gbs(us_l), gbs(us_rl), reps, us_k, gbs(us_k), gbs(us_rk));
    if (mat_bytes >= ((size_t)32 << 20)) {  // the 40 MB reference; the slices are reported as they are
      const bool ok = gbs(us_k) >= 450.0;
      printf("RESULT: %s  reference shape %dx%d in-kernel GB/s=%.1f (README threshold >= 450; read ceiling %.0f: %s)\n",
             ok ? "OK" : "BELOW 450 GB/s", sh.N, sh.K, gbs(us_k), gbs(us_rk),
             gbs(us_rk) < 450.0 ? "the memory system itself is below target on this box" : "memory system is fine, the kernel is the limit");
      if (!ok) g_fail = 1;
    }
  }
  HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
  cleanup();
}

int main(int argc, char** argv) {
  if (arg_flag(argc, argv, "--count")) { int n = 0; HIP_CHECK(hipGetDeviceCount(&n)); printf("%d\n", n); return 0; }
  const int reps = std::max(1, arg_int(argc, argv, "--reps", 200));
  const double seconds = (double)arg_int(argc, argv, "--seconds", 0);
  const size_t ring_target = (size_t)std::max(1, arg_int(argc, argv, "--ring-mb", 768)) << 20;
  const char* kname = arg_str(argc, argv, "--kernel", "dot");
  const int kind = strcmp(kname, "naive") == 0 ? 0 : 1;
  const char* one = arg_str(argc, argv, "--shape", nullptr);
  std::string list = one ? one : arg_str(argc, argv, "--shapes", "256x6144,6144x256,4608x6144,6144x2304,6144x6144");
  hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("device=%s arch=%s pci=%04x:%02x:%02x.0 kernel=%s\n", prop.name, prop.gcnArchName, prop.pciDomainID, prop.pciBusID,
         prop.pciDeviceID, kind ? "dot(sdot4,q8_1 x)" : "naive(fp32)");
  size_t pos = 0;
  while (pos <= list.size()) {
    size_t cpos = list.find(',', pos);
    std::string tok = list.substr(pos, cpos == std::string::npos ? std::string::npos : cpos - pos);
    Shape sh;
    if (!tok.empty()) {
      if (!parse_shape(tok.c_str(), &sh)) { fprintf(stderr, "bad shape '%s' (want NxK, K%%32==0, N*K/32 %% 8 == 0)\n", tok.c_str()); return 1; }
      run_shape(sh, kind, reps, seconds, ring_target, 12345u);
    }
    if (cpos == std::string::npos) break;
    pos = cpos + 1;
  }
  return g_fail;
}

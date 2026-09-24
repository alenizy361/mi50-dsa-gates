// Gate 3 — one-shot P2P allreduce across up to 8 MI50s, the collective the design runs ~234x per token.
//
// Every GPU runs ONE persistent kernel (one workgroup, 1024 threads). Per iteration it:
//   1. stores its vector into every peer's inbox slot (double-buffered by iteration parity),
//   2. flushes each RECEIVER's HDP, stores a ready flag in every peer's memory,
//   3. polls its own G flags, then reduces its G inbox slots locally and CHECKS the result
//      (values depend on the iteration number, so a stale buffer shows up as a wrong sum).
// Reports per-iteration latency percentiles (from GPU 0's wall clock), timeouts and wrong elements.
//
// ONE workgroup per GPU is load-bearing: __syncthreads() is the "all reads of iteration it have
// returned" point that makes a parity slot reusable at it+2 (a peer stores it+2 only after it saw my
// flag for it+1, which I store only after that barrier). A multi-workgroup variant needs per-WG
// flags or an arrival counter.
//
//   ./bin/oneshot_allreduce [--gpus 8] [--vec 6144] [--iters 5000] [--no-hdp]
//   --vec 6144 = 24 KB fp32 (the FP32-residual variant the review recommends); --vec 3072 = 12 KB.
#include "common.h"

#define MAXG 8

struct Peers {
  float* inbox[MAXG];       // peer p's inbox base, layout [2][G][vec]
  uint64_t* flags[MAXG];    // peer p's flags base, layout [2][G] x FLAG_STRIDE
  unsigned int* hdp[MAXG];  // peer p's HDP flush register (nullptr = off)
};
struct Local {
  int g, G, vec, iters;
  uint64_t max_spins;
  float* my_inbox;
  uint64_t* my_flags;
  float* out;
  uint64_t* ts;
  uint32_t* err;  // [0] timeouts, [1] wrong elements, [2] first failing iteration, [3] aborted by peer
};

__global__ void allreduce_kernel(Peers P, Local L) {
  __shared__ unsigned okbad;
  const int G = L.G, V = L.vec;
  for (int it = 1; it <= L.iters; ++it) {
    const int par = it & 1;
    const float myval = (float)(L.g + 1) * (float)it;
    const float4 v4 = {myval, myval, myval, myval};
    for (int p = 0; p < G; ++p) {
      float4* dst = (float4*)(P.inbox[p] + ((size_t)par * G + L.g) * V);  // V % 4 == 0 -> 16-B aligned
      for (int i = threadIdx.x; i < V / 4; i += blockDim.x) dst[i] = v4;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      okbad = 0;
      // Flush only the RECEIVERS' HDPs: my own stores never enter my HDP, and a store to my own
      // register would be an outbound PCIe write aimed at its own port.
      for (int p = 0; p < G; ++p) if (p != L.g) hdp_flush(P.hdp[p]);
      __threadfence_system();
      for (int p = 0; p < G; ++p) flag_store(P.flags[p] + ((size_t)par * G + L.g) * FLAG_STRIDE, (uint64_t)it);
    }
    __syncthreads();
    if (threadIdx.x < G) {
      uint64_t* f = L.my_flags + ((size_t)par * G + threadIdx.x) * FLAG_STRIDE;
      const bool got = flag_wait_ge(f, (uint64_t)it, L.max_spins);
      if (flag_load(f) == FLAG_POISON) {  // peer gave up (even if its poison arrived as we timed out)
        atomicOr(&okbad, 2u);
        atomicAdd(&L.err[3], 1u);
      } else if (!got) {
        atomicOr(&okbad, 1u);
        atomicAdd(&L.err[0], 1u);
        atomicCAS(&L.err[2], 0u, (uint32_t)it);
      }
    }
    __syncthreads();
    if (okbad) {
      if (threadIdx.x == 0 && (okbad & 1u))  // I timed out: tell every peer to stop within one poll
        for (int p = 0; p < G; ++p) if (p != L.g) {
          flag_store(P.flags[p] + ((size_t)0 * G + L.g) * FLAG_STRIDE, FLAG_POISON);
          flag_store(P.flags[p] + ((size_t)1 * G + L.g) * FLAG_STRIDE, FLAG_POISON);
        }
      return;
    }
    __threadfence_system();
    const float expect = (float)it * (float)(G * (G + 1) / 2);
    uint32_t bad = 0;
    for (int i = threadIdx.x; i < V / 4; i += blockDim.x) {
      float4 s = {0.f, 0.f, 0.f, 0.f};
      for (int p = 0; p < G; ++p) {
        const float4 v = ((const float4*)(L.my_inbox + ((size_t)par * G + p) * V))[i];
        s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
      }
      ((float4*)L.out)[i] = s;
      bad += (s.x != expect) + (s.y != expect) + (s.z != expect) + (s.w != expect);
    }
    if (bad) atomicAdd(&L.err[1], bad);
    if (threadIdx.x == 0) L.ts[it - 1] = wall_clock64();
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  force_fine_grain_env();
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  const int G = std::min(ndev, std::min(MAXG, arg_int(argc, argv, "--gpus", MAXG)));
  const int vec = arg_int(argc, argv, "--vec", 6144);
  const int iters = arg_int(argc, argv, "--iters", 5000);
  const bool no_hdp = arg_flag(argc, argv, "--no-hdp");
  const uint64_t max_spins = (uint64_t)arg_ll(argc, argv, "--max-spins", 5000000LL);
  if (G < 2) { fprintf(stderr, "need >= 2 GPUs (have %d)\n", ndev); return 1; }
  if (vec % 4 || vec < 4) { fprintf(stderr, "--vec must be a positive multiple of 4\n"); return 1; }
  if (iters < 20) { fprintf(stderr, "--iters >= 20\n"); return 1; }
  const int tri = G * (G + 1) / 2;
  if ((double)iters * tri >= 16777216.0) {  // every partial sum must stay an exact fp32 integer (< 2^24)
    fprintf(stderr, "--iters too large for exact fp32 checking: max %d for %d GPUs\n", 16777215 / tri, G);
    return 1;
  }

  for (int i = 0; i < G; ++i)
    for (int j = 0; j < G; ++j)
      if (i != j) enable_peer(i, j);

  Peers P{};
  std::vector<Local> L(G);
  std::vector<hipStream_t> st(G);
  std::string mode;
  const size_t flag_words = (size_t)2 * G * FLAG_STRIDE;
  for (int g = 0; g < G; ++g) {
    HIP_CHECK(hipSetDevice(g));
    float* inbox = (float*)alloc_coherent((size_t)2 * G * vec * sizeof(float), &mode);
    uint64_t* flags = (uint64_t*)alloc_coherent(flag_words * sizeof(uint64_t), nullptr);
    HIP_CHECK(hipMemset(inbox, 0, (size_t)2 * G * vec * sizeof(float)));
    HIP_CHECK(hipMemset(flags, 0, flag_words * sizeof(uint64_t)));
    float* out = nullptr; HIP_CHECK(hipMalloc(&out, (size_t)vec * sizeof(float)));
    uint64_t* ts = nullptr; HIP_CHECK(hipMalloc(&ts, (size_t)iters * sizeof(uint64_t)));
    HIP_CHECK(hipMemset(ts, 0, (size_t)iters * sizeof(uint64_t)));
    uint32_t* err = nullptr; HIP_CHECK(hipMalloc(&err, 16)); HIP_CHECK(hipMemset(err, 0, 16));
    HIP_CHECK(hipStreamCreate(&st[g]));
    P.inbox[g] = inbox; P.flags[g] = flags; P.hdp[g] = no_hdp ? nullptr : hdp_flush_reg(g);
    L[g] = Local{g, G, vec, iters, max_spins, inbox, flags, out, ts, err};
  }
  // All buffers exist before any kernel starts; peers write into each other's memory from the first iteration.
  for (int g = 0; g < G; ++g) { HIP_CHECK(hipSetDevice(g)); HIP_CHECK(hipDeviceSynchronize()); }

  printf("gate=3 gpus=%d vec=%d bytes=%d iters=%d hdp_flush=%s mem=%s kernel=%s\n", G, vec, vec * 4, iters,
         no_hdp ? "OFF" : "on", mode.c_str(), kernel_release().c_str());
  printf("  hdp_regs:"); for (int g = 0; g < G; ++g) printf(" gpu%d=%p", g, (void*)P.hdp[g]);
  printf("\n  iommu:"); for (int g = 0; g < G; ++g) printf(" gpu%d=%s", g, iommu_mode(g).c_str());
  printf("\n");

  hipEvent_t e0, e1;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventRecord(e0, st[0]));
  for (int g = 0; g < G; ++g) {
    HIP_CHECK(hipSetDevice(g));
    hipLaunchKernelGGL(allreduce_kernel, dim3(1), dim3(1024), 0, st[g], P, L[g]);  // ONE workgroup per GPU (see header)
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipEventRecord(e1, st[0]));
  for (int g = 0; g < G; ++g) { HIP_CHECK(hipSetDevice(g)); HIP_CHECK(hipStreamSynchronize(st[g])); }
  float ms = 0.f; HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));

  std::vector<uint64_t> h_ts(iters);
  HIP_CHECK(hipMemcpy(h_ts.data(), L[0].ts, (size_t)iters * 8, hipMemcpyDeviceToHost));
  const double khz = wallclock_khz(0);
  std::vector<double> lat;
  for (int i = iters / 10 + 1; i < iters; ++i) {
    if (h_ts[i] == 0 || h_ts[i - 1] == 0) break;
    lat.push_back((double)(h_ts[i] - h_ts[i - 1]) * 1000.0 / khz);
  }
  Pct p = percentiles(lat);
  const double gpu_total_us = h_ts[iters - 1] ? (double)(h_ts[iters - 1] - h_ts[0]) * 1000.0 / khz : 0;
  if (gpu_total_us > 0 && fabs(gpu_total_us - (double)ms * 1000.0) > 0.1 * (double)ms * 1000.0)
    printf("WARNING: GPU wall clock (%.0f us) and hipEvent time (%.0f us) disagree >10%%: check hipDeviceAttributeWallClockRate=%.0f kHz\n",
           gpu_total_us, (double)ms * 1000.0, khz);
  uint32_t timeouts = 0, wrong = 0, aborted = 0;
  for (int g = 0; g < G; ++g) {
    uint32_t e[4] = {0, 0, 0, 0};
    HIP_CHECK(hipMemcpy(e, L[g].err, 16, hipMemcpyDeviceToHost));
    timeouts += e[0]; wrong += e[1]; aborted += e[3];
    if (e[0]) printf("  gpu%d timed out first at iteration %u\n", g, e[2]);
  }
  const size_t egress = (size_t)(G - 1) * vec * 4;
  printf("allreduce_us mean=%.2f p50=%.2f p90=%.2f p99=%.2f min=%.2f max=%.2f  (window: last 90%% of iterations)\n",
         p.mean, p.p50, p.p90, p.p99, p.mn, p.mx);
  printf("egress_bytes/iter=%zu (%.1f GB/s at p50; compare with gate 1's copy bandwidth)  timeouts=%u aborted_by_peer=%u wrong_elements=%u\n",
         egress, p.p50 > 0 ? (double)egress / (p.p50 * 1e-6) / 1e9 : 0, timeouts, aborted, wrong);
  if (timeouts) { printf("RESULT: TIMEOUT\n"); return 1; }
  if (wrong) { printf("RESULT: CORRUPTION — a stale inbox was reduced (protocol not sound as configured)\n"); return 1; }
  if (aborted) { printf("RESULT: ABORTED\n"); return 1; }
  const bool design = p.p50 <= 40.0, gate = p.p50 <= 95.0;
  printf("RESULT: %s  p50=%.1f us (design budget 40 us; the 576 us/layer gate tolerates ~95 us)\n",
         design ? "OK" : gate ? "OK-WITHIN-LAYER-GATE" : "TOO SLOW", p.p50);
  return gate ? 0 : 6;  // 6 = correct but below the layer gate (see common.h exit codes)
}

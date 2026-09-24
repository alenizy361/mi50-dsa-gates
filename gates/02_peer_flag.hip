// Gate 2 — kernel-side peer store + ready flag over PCIe between two MI50s.
//
// Two modes.
//  (default) PING-PONG latency: GPU A writes a payload into GPU B's memory, [flushes B's HDP], stores a
//     flag in B's memory, waits for B's ack. B polls, verifies the payload word by word, acks. Reports
//     round-trip percentiles. Run --words 0 for the bare-flag round trip RTT0; one-way flag latency is
//     ~RTT0/2 and one-way data+verify is ~RTT(words) - RTT0/2.
//  (--stress) CREDIT-RING soundness test: A streams `depth` messages of `words` words without waiting for
//     an ack per message (credits come back when B has consumed a slot); B's polling lane samples the
//     LAST line of the payload in the same load batch as the flag (TAIL_SUSPECT), re-reads it after the
//     acquire fence (TAIL_STALE) and then all threads verify the whole slot. A stale tail with the flag
//     already visible is flag-before-data — the silent corruption the HDP flush and UC memory prevent.
//     Ping-pong cannot show it (one message in flight, ~1 us of poll latency); the ring can.
//
//   ./bin/peer_flag --src 0 --dst 1 [--words 3072] [--iters 2000] [--no-hdp] [--hdp-readback] [--no-verify]
//   ./bin/peer_flag --src 0 --dst 1 --stress [--depth 8] [--words 65536] [--iters 20000] [--no-hdp]
#include "common.h"

__device__ __forceinline__ uint32_t pattern(uint32_t it, uint32_t i) {
  return (it * 2654435761u) ^ (i * 40503u) ^ 0x9E3779B9u;
}

// ------------------------------------------------------------------ ping-pong
struct Ping {
  uint32_t* payload_on_b;
  uint64_t* flag_on_b;
  uint64_t* ack_on_a;
  unsigned int* hdp_b;  // nullptr => no HDP flush
  uint64_t* ts;         // local: timestamp after each ack
  uint32_t* err;        // local: [0] timeouts, [1] aborted by peer
  int words, iters, hdp_readback;
  uint64_t max_spins;
};
struct Pong {
  uint32_t* payload;   // local (written by A)
  uint64_t* flag;      // local (written by A)
  uint64_t* ack_on_a;  // remote
  uint32_t* err;       // local: [0] timeouts, [1] stale words, [2] aborted by peer
  int words, iters, verify;
  uint64_t max_spins;
};

__global__ void ping_kernel(Ping p) {
  __shared__ int ok;
  for (int it = 1; it <= p.iters; ++it) {
    for (int i = threadIdx.x; i < p.words; i += blockDim.x) p.payload_on_b[i] = pattern((uint32_t)it, (uint32_t)i);
    __threadfence_system();  // every thread: its own payload stores are system-visible before the barrier
    __syncthreads();
    if (threadIdx.x == 0) {
      if (p.hdp_readback) hdp_flush_sync(p.hdp_b); else hdp_flush(p.hdp_b);
      __threadfence_system();
      flag_store(p.flag_on_b, (uint64_t)it);
      ok = flag_wait_ge(p.ack_on_a, (uint64_t)it, p.max_spins) ? 1 : 0;
      if (flag_load(p.ack_on_a) == FLAG_POISON) { ok = 0; atomicAdd(&p.err[1], 1u); }  // peer gave up (even if it arrived as we timed out)
      else if (!ok) { atomicAdd(&p.err[0], 1u); flag_store(p.flag_on_b, FLAG_POISON); }
      p.ts[it - 1] = wall_clock64();
    }
    __syncthreads();
    if (!ok) return;
  }
}

__global__ void pong_kernel(Pong p) {
  __shared__ int ok;
  for (int it = 1; it <= p.iters; ++it) {
    if (threadIdx.x == 0) {
      ok = flag_wait_ge(p.flag, (uint64_t)it, p.max_spins) ? 1 : 0;
      if (flag_load(p.flag) == FLAG_POISON) { ok = 0; atomicAdd(&p.err[2], 1u); }
      else if (!ok) { atomicAdd(&p.err[0], 1u); flag_store(p.ack_on_a, FLAG_POISON); }
    }
    __syncthreads();
    if (!ok) return;
    __threadfence_system();  // acquire in every thread before touching the payload
    if (p.verify) {
      uint32_t stale = 0;
      for (int i = threadIdx.x; i < p.words; i += blockDim.x)
        if (p.payload[i] != pattern((uint32_t)it, (uint32_t)i)) ++stale;
      if (stale) atomicAdd(&p.err[1], stale);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence_system();
      flag_store(p.ack_on_a, (uint64_t)it);  // ack carries no payload: no HDP flush needed
    }
  }
}

// ------------------------------------------------------------------ credit ring (stress)
struct PingRing {
  uint32_t* slots_on_b;   // [depth][words]
  uint64_t* flags_on_b;   // [depth * FLAG_STRIDE]
  uint64_t* consumed_on_a;
  unsigned int* hdp_b;
  uint64_t* ts;           // local: timestamp after each flag store (issue interval)
  uint32_t* err;          // local: [0] timeouts (credit), [1] aborted by peer
  int words, depth, hdp_readback;
  uint64_t iters, max_spins;
};
struct PongRing {
  uint32_t* slots;        // local
  uint64_t* flags;        // local
  uint64_t* consumed_on_a;
  uint32_t* err;          // local: [0] timeouts, [1] stale words (full check), [2] TAIL_STALE, [3] TAIL_SUSPECT, [4] aborted by peer
  int words, depth;
  uint64_t iters, max_spins;
};

__global__ void ping_ring(PingRing p) {
  __shared__ int ok;
  for (uint64_t it = 1; it <= p.iters; ++it) {
    const int s = (int)(it % (uint64_t)p.depth);
    if (threadIdx.x == 0) {  // credit: slot s was last used for it-depth and must have been consumed
      ok = 1;
      if (it > (uint64_t)p.depth) {
        ok = flag_wait_ge(p.consumed_on_a, it - (uint64_t)p.depth, p.max_spins) ? 1 : 0;
        if (flag_load(p.consumed_on_a) == FLAG_POISON) { ok = 0; atomicAdd(&p.err[1], 1u); }
        else if (!ok) { atomicAdd(&p.err[0], 1u); for (int d = 0; d < p.depth; ++d) flag_store(p.flags_on_b + (size_t)d * FLAG_STRIDE, FLAG_POISON); }
      }
    }
    __syncthreads();
    if (!ok) return;
    uint32_t* dst = p.slots_on_b + (size_t)s * p.words;
    for (int i = threadIdx.x; i < p.words; i += blockDim.x) dst[i] = pattern((uint32_t)it, (uint32_t)i);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      if (p.hdp_readback) hdp_flush_sync(p.hdp_b); else hdp_flush(p.hdp_b);
      __threadfence_system();
      flag_store(p.flags_on_b + (size_t)s * FLAG_STRIDE, it);
      p.ts[it - 1] = wall_clock64();
    }
  }
}

// Sample the tail line in the SAME load batch as the flag (flag issued first). The two loads may be
// serviced out of order, so a stale co-issued tail is only TAIL_SUSPECT; the re-read after the acquire
// fence is the sound TAIL_STALE. Removes the ~1 us poll-loop floor that would hide sub-us HDP skew.
__device__ __forceinline__ bool poll_flag_and_tail(uint64_t* f, uint64_t v, const uint32_t* tail16, uint32_t it,
                                                   uint32_t base, uint64_t max_spins, uint32_t* suspect, bool* poison) {
  for (uint64_t s = 0; s < max_spins; ++s) {
    uint64_t fv = __hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);  // keep issue order
    uint32_t t[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)  // relaxed atomics: the compiler may neither hoist nor sink them past the flag check
      t[i] = __hip_atomic_load(const_cast<uint32_t*>(tail16) + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (fv >= v) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // s_waitcnt vmcnt(0); buffer_wbinvl1_vol
      *poison = (fv == FLAG_POISON);
      uint32_t st = 0;
#pragma unroll
      for (int i = 0; i < 16; ++i) st += (t[i] != pattern(it, base + (uint32_t)i));
      *suspect = st;
      return true;
    }
    __builtin_amdgcn_s_sleep(1);
  }
  return false;
}

__global__ void pong_ring(PongRing p) {
  __shared__ int ok;
  for (uint64_t it = 1; it <= p.iters; ++it) {
    const int s = (int)(it % (uint64_t)p.depth);
    const uint32_t* src = p.slots + (size_t)s * p.words;
    if (threadIdx.x == 0) {
      uint32_t suspect = 0; bool poison = false;
      ok = poll_flag_and_tail(p.flags + (size_t)s * FLAG_STRIDE, it, src + p.words - 16, (uint32_t)it,
                              (uint32_t)(p.words - 16), p.max_spins, &suspect, &poison) ? 1 : 0;
      if (!ok && flag_load(p.flags + (size_t)s * FLAG_STRIDE) == FLAG_POISON) poison = true;  // poison arrived as we gave up
      if (poison) { ok = 0; atomicAdd(&p.err[4], 1u); }
      else if (!ok) { atomicAdd(&p.err[0], 1u); flag_store(p.consumed_on_a, FLAG_POISON); }
      else {
        if (suspect) atomicAdd(&p.err[3], suspect);
        uint32_t st = 0;  // sound re-read of the last-written line, after the acquire fence
        for (int i = p.words - 16; i < p.words; ++i) st += (__builtin_nontemporal_load(src + i) != pattern((uint32_t)it, (uint32_t)i));
        if (st) atomicAdd(&p.err[2], st);
      }
    }
    __syncthreads();
    if (!ok) return;
    __threadfence_system();
    uint32_t stale = 0;  // full check, descending (last-written words first)
    for (int i = p.words - 1 - (int)threadIdx.x; i >= 0; i -= (int)blockDim.x)
      stale += (src[i] != pattern((uint32_t)it, (uint32_t)i));
    if (stale) atomicAdd(&p.err[1], stale);
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); flag_store(p.consumed_on_a, it); }  // credit back (flag only)
  }
}

// ------------------------------------------------------------------ host
int main(int argc, char** argv) {
  force_fine_grain_env();
  const int a = arg_int(argc, argv, "--src", 0);
  const int b = arg_int(argc, argv, "--dst", 1);
  const bool stress = arg_flag(argc, argv, "--stress");
  const int words = arg_int(argc, argv, "--words", stress ? 65536 : 3072);  // 3072 x 4 B = 12 KB = one 6144-dim fp16 vector
  const int depth = std::max(1, arg_int(argc, argv, "--depth", 8));
  const long long iters = arg_ll(argc, argv, "--iters", stress ? 20000 : 2000);
  const bool no_hdp = arg_flag(argc, argv, "--no-hdp");
  const int hdp_readback = arg_flag(argc, argv, "--hdp-readback") ? 1 : 0;
  const int verify = arg_flag(argc, argv, "--no-verify") ? 0 : 1;
  const uint64_t max_spins = (uint64_t)arg_ll(argc, argv, "--max-spins", 5000000LL);
  if (stress && words < 16) { fprintf(stderr, "--stress needs --words >= 16\n"); return 1; }
  if (iters < 20) { fprintf(stderr, "--iters >= 20\n"); return 1; }

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (a == b || a >= ndev || b >= ndev) { fprintf(stderr, "need two distinct GPUs (have %d)\n", ndev); return 1; }
  enable_peer(a, b);
  enable_peer(b, a);
  unsigned int* hdp_b = no_hdp ? nullptr : hdp_flush_reg(b);
  const double khz = wallclock_khz(a);

  std::string mode_b, mode_a;
  HIP_CHECK(hipSetDevice(b));
  const size_t slot_words = (size_t)std::max(words, 1);
  const int nslots = stress ? depth : 1;
  uint32_t* slots = (uint32_t*)alloc_coherent(slot_words * 4 * nslots, &mode_b);
  uint64_t* flags = (uint64_t*)alloc_coherent((size_t)nslots * FLAG_STRIDE * 8, nullptr);
  uint32_t* err_b = nullptr; HIP_CHECK(hipMalloc(&err_b, 32));
  HIP_CHECK(hipMemset(slots, 0, slot_words * 4 * nslots));
  HIP_CHECK(hipMemset(flags, 0, (size_t)nslots * FLAG_STRIDE * 8));
  HIP_CHECK(hipMemset(err_b, 0, 32));
  hipStream_t sb; HIP_CHECK(hipStreamCreate(&sb));

  HIP_CHECK(hipSetDevice(a));
  uint64_t* ack = (uint64_t*)alloc_coherent(64, &mode_a);
  uint64_t* ts = nullptr; HIP_CHECK(hipMalloc(&ts, (size_t)iters * 8));
  uint32_t* err_a = nullptr; HIP_CHECK(hipMalloc(&err_a, 32));
  HIP_CHECK(hipMemset(ts, 0, (size_t)iters * 8));  // host treats ts==0 as "stopped early"
  HIP_CHECK(hipMemset(ack, 0, 64));
  HIP_CHECK(hipMemset(err_a, 0, 32));
  hipStream_t sa; HIP_CHECK(hipStreamCreate(&sa));
  hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
  for (int d = 0; d < 2; ++d) { HIP_CHECK(hipSetDevice(d == 0 ? a : b)); HIP_CHECK(hipDeviceSynchronize()); }

  printf("gate=2 mode=%s src=%d dst=%d words=%d bytes=%d iters=%lld depth=%d hdp_flush=%s hdp_reg_dst=%p verify=%s\n"
         "  mem_dst=%s mem_ack=%s iommu_src=%s iommu_dst=%s kernel=%s\n",
         stress ? "ring" : "pingpong", a, b, words, words * 4, iters, stress ? depth : 1,
         no_hdp ? "OFF" : (hdp_readback ? "on+readback" : "on"), (void*)hdp_b, (stress || verify) ? "on" : "OFF",
         mode_b.c_str(), mode_a.c_str(), iommu_mode(a).c_str(), iommu_mode(b).c_str(), kernel_release().c_str());

  HIP_CHECK(hipSetDevice(b));
  if (stress) {
    PongRing pg{slots, flags, ack, err_b, words, depth, (uint64_t)iters, max_spins};
    hipLaunchKernelGGL(pong_ring, dim3(1), dim3(256), 0, sb, pg);
  } else {
    Pong pg{slots, flags, ack, err_b, words, (int)iters, verify, max_spins};
    hipLaunchKernelGGL(pong_kernel, dim3(1), dim3(256), 0, sb, pg);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipSetDevice(a));
  HIP_CHECK(hipEventRecord(e0, sa));
  if (stress) {
    PingRing pi{slots, flags, ack, hdp_b, ts, err_a, words, depth, hdp_readback, (uint64_t)iters, max_spins};
    hipLaunchKernelGGL(ping_ring, dim3(1), dim3(256), 0, sa, pi);
  } else {
    Ping pi{slots, flags, ack, hdp_b, ts, err_a, words, (int)iters, hdp_readback, max_spins};
    hipLaunchKernelGGL(ping_kernel, dim3(1), dim3(256), 0, sa, pi);
  }
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(e1, sa));
  HIP_CHECK(hipStreamSynchronize(sa));
  HIP_CHECK(hipSetDevice(b));
  HIP_CHECK(hipStreamSynchronize(sb));

  float ms = 0.f; HIP_CHECK(hipSetDevice(a)); HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  std::vector<uint64_t> h_ts(iters);
  HIP_CHECK(hipMemcpy(h_ts.data(), ts, (size_t)iters * 8, hipMemcpyDeviceToHost));
  uint32_t ea[8] = {0}, eb[8] = {0};
  HIP_CHECK(hipMemcpy(ea, err_a, 32, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(eb, err_b, 32, hipMemcpyDeviceToHost));

  const long long skip = iters / 10 + 1;
  std::vector<double> dt;
  for (long long i = skip; i < iters; ++i) {
    if (h_ts[i] == 0 || h_ts[i - 1] == 0) break;  // stopped early (timeout/abort)
    dt.push_back((double)(h_ts[i] - h_ts[i - 1]) * 1000.0 / khz);  // us
  }
  Pct p = percentiles(dt);
  const double gpu_total_us = h_ts[iters - 1] ? (double)(h_ts[iters - 1] - h_ts[0]) * 1000.0 / khz : 0;
  if (gpu_total_us > 0 && fabs(gpu_total_us - (double)ms * 1000.0) > 0.1 * (double)ms * 1000.0)
    printf("WARNING: GPU wall clock (%.0f us) and hipEvent time (%.0f us) disagree >10%%: check hipDeviceAttributeWallClockRate=%.0f kHz\n",
           gpu_total_us, (double)ms * 1000.0, khz);

  const bool timeout = ea[0] || eb[0];
  const bool aborted = ea[1] || (stress ? eb[4] : eb[2]);  // ring: err[2] is TAIL_STALE, err[4] is aborted-by-peer
  if (stress) {
    printf("issue_interval_us mean=%.2f p50=%.2f p90=%.2f p99=%.2f min=%.2f max=%.2f  (%.1f MB/s of payload through the ring)\n",
           p.mean, p.p50, p.p90, p.p99, p.mn, p.mx, p.p50 > 0 ? (double)words * 4 / (p.p50 * 1e-6) / 1e6 : 0);
    printf("timeouts: ping=%u pong=%u  aborted_by_peer=%u  STALE_WORDS(full check)=%u  TAIL_STALE(after fence)=%u  TAIL_SUSPECT(co-issued)=%u\n",
           ea[0], eb[0], ea[1] + eb[4], eb[1], eb[2], eb[3]);
  } else {
    printf("rtt_us mean=%.2f p50=%.2f p90=%.2f p99=%.2f min=%.2f max=%.2f  (window: last 90%% of iterations)\n",
           p.mean, p.p50, p.p90, p.p99, p.mn, p.mx);
    printf("  one-way flag latency ~ RTT0/2 where RTT0 = this test with --words 0; one-way data+verify ~ RTT(words) - RTT0/2\n");
    printf("timeouts: ping=%u pong=%u  aborted_by_peer=%u  STALE_WORDS(flag visible before data)=%u%s\n",
           ea[0], eb[0], ea[1] + eb[2], eb[1], verify ? "" : "  [verify OFF: latency-only run, not a soundness run]");
  }
  const bool corrupt = eb[1] || eb[2];
  if (timeout) printf("RESULT: TIMEOUT — peer stores/flags not arriving (P2P broken for this pair, or --max-spins too low)\n");
  else if (aborted) printf("RESULT: ABORTED by peer timeout\n");
  else if (corrupt) printf("RESULT: CORRUPTION — flag visible before its data; protocol NOT sound as configured\n");
  else printf("RESULT: OK%s\n", (stress && eb[3]) ? "  (TAIL_SUSPECT>0: co-issued sample saw old data; re-read was clean — report it)" : "");
  return (timeout || aborted || corrupt) ? 1 : 0;
}

// common.h — shared helpers for the MI50 (gfx906) gate micro-benchmarks.
// Build: hipcc --offload-arch=gfx906 -O3 -std=c++17 -lhsa-runtime64 (see ../build.sh)
//
// Exit codes: 1 usage/result failure, 2 HIP error, 3 no peer access, 4 no UC-mapped memory, 5 no HDP register,
//             6 correct but slower than the layer gate (gate 3).
#pragma once
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#define HIP_CHECK(call)                                                                        \
  do {                                                                                         \
    hipError_t _e = (call);                                                                    \
    if (_e != hipSuccess) {                                                                    \
      fprintf(stderr, "HIP error: %s\n  at %s:%d: %s\n", hipGetErrorString(_e), __FILE__,     \
              __LINE__, #call);                                                                \
      exit(2);                                                                                 \
    }                                                                                          \
  } while (0)

// Call FIRST in main(), before any HIP call: on a PCIe-only gfx906 box ROCr hides the fine-grained
// VRAM pool unless this is set (it is always visible inside an xGMI hive). Harmless otherwise.
static inline void force_fine_grain_env() { setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 0); }

// ---------------------------------------------------------------------------------------------
// Memory that a kernel on ANOTHER device writes, and that a local persistent kernel polls, must
// be mapped MTYPE UC in every GPU's page table. On gfx906 both hipDeviceMallocUncached
// (ROCm >= 5.5, extended-scope coherent) and hipDeviceMallocFinegrained give UC; plain hipMalloc
// gives MTYPE NC: L2-cached, written back / made coherent only at kernel boundaries, which a
// persistent kernel never reaches — the writer's stores stay dirty in its L2 and the poller spins
// on a stale line. CLR >= 6.4 silently returns COARSE memory (hipSuccess) when the requested pool
// is absent, so the pool is verified through hsa_amd_pointer_info and coarse memory is refused.
// ---------------------------------------------------------------------------------------------
static inline bool is_coarse(const void* p) {
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  if (hsa_amd_pointer_info(const_cast<void*>(p), &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS)
    return false;  // unknown: do not block
  return (info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0;
}

static inline void* alloc_coherent(size_t bytes, std::string* mode) {
  struct Try { unsigned int flag; const char* name; } tries[] = {
#ifdef hipDeviceMallocUncached
      {hipDeviceMallocUncached, "uncached(MTYPE UC)"},
#endif
      {hipDeviceMallocFinegrained, "finegrained(MTYPE UC)"},
  };
  for (const Try& t : tries) {
    void* p = nullptr;
    if (hipExtMallocWithFlags(&p, bytes, t.flag) != hipSuccess || !p) { (void)hipGetLastError(); continue; }
    if (is_coarse(p)) {
      fprintf(stderr, "alloc %s: pool absent, runtime returned COARSE memory instead; trying next\n", t.name);
      HIP_CHECK(hipFree(p));
      continue;
    }
    if (mode) *mode = t.name;
    return p;
  }
  fprintf(stderr,
          "No UC-mapped (uncached / fine-grained) device memory available.\n"
          "On PCIe-only gfx906 ROCr hides the fine-grained VRAM pool outside an xGMI hive; this binary sets\n"
          "HSA_FORCE_FINE_GRAIN_PCIE=1 itself, so check your ROCm build. Refusing to run on coarse-grained\n"
          "(MTYPE NC, L2-cached) memory: a TIMEOUT verdict on it would be meaningless.\n");
  exit(4);
}

// Address of a device's HDP_MEM_COHERENCY_FLUSH_CNTL register (what RCCL reads through
// hipDeviceAttributeHdpMemFlushCntl for non-xGMI peers on gfx906/gfx908). The WRITER stores 1 to the
// RECEIVER's register after its data stores and before its flag store, so posted PCIe writes still
// sitting in the receiver's HDP path reach HBM before the flag becomes visible. ROCr fills it only
// when KFD advertises an MMIO-remap heap (not under SR-IOV, non-4K pages or old kernels); a NULL
// would make hdp_flush() a silent no-op while the report says hdp_flush=on, so it is a hard error.
static inline unsigned int* hdp_flush_reg(int dev) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  if (!prop.hdpMemFlushCntl) {
    fprintf(stderr,
            "GPU %d: hipDeviceProp_t::hdpMemFlushCntl is NULL (KFD exposes no MMIO-remap heap). "
            "Cannot run WITH HDP flush; rerun with --no-hdp and record that in the results.\n", dev);
    exit(5);
  }
  return prop.hdpMemFlushCntl;
}

static inline double wallclock_khz(int dev) {
  int khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev));
  return (double)khz;
}

static inline std::string kernel_release() {
  struct utsname u;
  return uname(&u) == 0 ? std::string(u.release) : std::string("?");
}

// Whether the HDP-flush page is likely mapped UC (no IOMMU / passthrough / shared group) or NC
// (translating IOMMU with per-device groups). Under NC the flush store must bypass L2, which is
// why hdp_flush() below is a nontemporal store; printed so results can be attributed.
static inline std::string iommu_mode(int dev) {
  hipDeviceProp_t p;
  HIP_CHECK(hipGetDeviceProperties(&p, dev));
  char bdf[32];
  snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID, p.pciDeviceID);
  std::string lnk = std::string("/sys/bus/pci/devices/") + bdf + "/iommu_group";
  char buf[256];
  ssize_t n = readlink(lnk.c_str(), buf, sizeof buf - 1);
  if (n < 0) return "none";
  buf[n] = 0;
  const char* slash = strrchr(buf, '/');
  std::string grp = slash ? slash + 1 : buf;
  char t[32] = "?";
  if (FILE* f = fopen(("/sys/kernel/iommu_groups/" + grp + "/type").c_str(), "r")) {
    if (fscanf(f, "%31s", t) != 1) t[0] = '?';
    fclose(f);
  }
  return "grp" + grp + ":" + t;
}

static inline bool enable_peer(int from, int to, bool fatal = true) {
  int can = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can, from, to));
  if (!can) {
    fprintf(stderr, "GPU %d cannot access peer %d (hipDeviceCanAccessPeer=0).\n", from, to);
    if (fatal) exit(3);
    return false;
  }
  HIP_CHECK(hipSetDevice(from));
  hipError_t e = hipDeviceEnablePeerAccess(to, 0);
  if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(e);
  (void)hipGetLastError();
  return true;
}

static inline int arg_int(int argc, char** argv, const char* key, int def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return atoi(argv[i + 1]);
  return def;
}
static inline long long arg_ll(int argc, char** argv, const char* key, long long def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return atoll(argv[i + 1]);
  return def;
}
static inline bool arg_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], key)) return true;
  return false;
}
static inline const char* arg_str(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return argv[i + 1];
  return def;
}

struct Pct { double p50, p90, p99, mn, mx, mean; };
static inline Pct percentiles(std::vector<double> v) {
  Pct r{0, 0, 0, 0, 0, 0};
  if (v.empty()) return r;
  double s = 0;
  for (double x : v) s += x;
  r.mean = s / (double)v.size();
  std::sort(v.begin(), v.end());
  auto at = [&](double q) { return v[(size_t)(q * (double)(v.size() - 1))]; };
  r.p50 = at(0.5); r.p90 = at(0.9); r.p99 = at(0.99); r.mn = v.front(); r.mx = v.back();
  return r;
}

// Every flag lives on its own 64-byte line (8 x uint64): remote writers and the local poller never
// share a line, which keeps ECC read-modify-write serialisation out of the measured p99.
#define FLAG_STRIDE 8
// A GPU that times out writes this into its peers' flags so they abort within one poll instead of
// spinning to their own timeout; the report then distinguishes "timed out" from "aborted by peer".
#define FLAG_POISON (~0ull)

// ---------------------------------------------------------------------------------------------
// Device-side protocol primitives. No read-modify-write atomics ever target a PEER GPU's memory over
// PCIe (not supported on gfx906 and never done by RCCL); flags are monotonic system-scope stores and
// loads that stay < 2^32, so a torn 8-byte read could never look "ready" (and FLAG_POISON flips both
// halves). Ordering comes from fences + the HDP flush.
// ---------------------------------------------------------------------------------------------
// RCCL stores 1 to the register with a plain relaxed store (its GFX9 STORE macro is
// __atomic_store_n(..., __ATOMIC_RELAXED)). The glc slc bits here are only an L2 replacement hint: a
// no-op on the UC-mapped register page (the normal case) and NOT a write-through if the page were ever
// NC-mapped (translating IOMMU, printed as iommu=grpN:DMA). Ordering of the flush ahead of the flag over
// PCIe comes from posted-write ordering; when in doubt use --hdp-readback (gate 2), whose non-posted read
// cannot complete before the earlier posted writes have arrived.
__device__ __forceinline__ void hdp_flush(unsigned int* reg) {
  if (!reg) return;
  __builtin_nontemporal_store(1u, reg);
}
// Synchronous variant (--hdp-readback): a non-posted PCIe read cannot pass the earlier posted write.
// The loaded value must be consumed, or the compiler deletes the load.
__device__ __forceinline__ void hdp_flush_sync(unsigned int* reg) {
  if (!reg) return;
  __builtin_nontemporal_store(1u, reg);
  unsigned int v = __builtin_nontemporal_load(reg);
  asm volatile("" ::"v"(v));
}
__device__ __forceinline__ void flag_store(uint64_t* f, uint64_t v) {
  __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint64_t flag_load(uint64_t* f) {
  return __hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
// Bounded spin with s_sleep: each spin is one UC round trip (~0.5-1 us) + s_sleep, so the default
// 5e6 spins bound a wait at roughly 3-6 s. Returns true when *f >= v (including FLAG_POISON — callers
// check for it), false on timeout: a wedged peer produces a reported timeout, never a silent hang.
__device__ __forceinline__ bool flag_wait_ge(uint64_t* f, uint64_t v, uint64_t max_spins) {
  for (uint64_t s = 0; s < max_spins; ++s) {
    if (flag_load(f) >= v) return true;
    __builtin_amdgcn_s_sleep(1);
  }
  return false;
}

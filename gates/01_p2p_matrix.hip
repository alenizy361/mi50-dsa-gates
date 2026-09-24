// Gate 1 (part b) — peer-access matrix and hipMemcpyPeer bandwidth for every GPU pair.
// The .sh wrapper adds the system-side facts (kernel, amdgpu params, IOMMU, BAR placement, topology).
// Exit 0 only if every pair can access its peer AND the worst copy bandwidth is within 2x of the best.
//   ./bin/p2p_matrix [--mb 64]
#include "common.h"

int main(int argc, char** argv) {
  force_fine_grain_env();
  const size_t bytes = (size_t)std::max(1, arg_int(argc, argv, "--mb", 64)) << 20;
  int n = 0;
  HIP_CHECK(hipGetDeviceCount(&n));
  printf("devices=%d kernel=%s\n", n, kernel_release().c_str());
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p, i));
    printf("  gpu%d %s arch=%s pci=%04x:%02x:%02x.0 vram=%.1fGB hdp_reg=%p iommu=%s\n", i, p.name, p.gcnArchName,
           p.pciDomainID, p.pciBusID, p.pciDeviceID, p.totalGlobalMem / 1073741824.0, (void*)p.hdpMemFlushCntl,
           iommu_mode(i).c_str());
  }
  printf("\ncanAccessPeer matrix (row = from, col = to):\n     ");
  for (int j = 0; j < n; ++j) printf(" %3d", j);
  printf("\n");
  int all = 1;
  for (int i = 0; i < n; ++i) {
    printf("  %2d ", i);
    for (int j = 0; j < n; ++j) {
      int can = (i == j);
      if (i != j) HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
      if (i != j && !can) all = 0;
      printf(" %3s", i == j ? "-" : (can ? "yes" : "NO"));
    }
    printf("\n");
  }
  if (!all) printf("\nRESULT: at least one pair cannot access its peer — gate 1 FAILS on this box as configured\n");

  std::vector<void*> buf(n);
  for (int i = 0; i < n; ++i) { HIP_CHECK(hipSetDevice(i)); HIP_CHECK(hipMalloc(&buf[i], bytes)); }
  printf("\nhipMemcpyPeer GB/s (%zu MB, row = src, col = dst):\n     ", bytes >> 20);
  for (int j = 0; j < n; ++j) printf("   %4d", j);
  printf("\n");
  double worst = 1e9, best = 0;
  for (int i = 0; i < n; ++i) {
    printf("  %2d ", i);
    for (int j = 0; j < n; ++j) {
      if (i == j) { printf("      -"); continue; }
      int can = 0; HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
      if (!can) { printf("     NO"); continue; }
      enable_peer(i, j, false);
      HIP_CHECK(hipSetDevice(i));
      hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
      HIP_CHECK(hipMemcpyPeer(buf[j], j, buf[i], i, bytes));  // warm
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipEventRecord(e0, 0));
      for (int r = 0; r < 5; ++r) HIP_CHECK(hipMemcpyPeer(buf[j], j, buf[i], i, bytes));
      HIP_CHECK(hipEventRecord(e1, 0));
      HIP_CHECK(hipEventSynchronize(e1));
      float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      double gbs = (double)bytes * 5 / (ms * 1e-3) / 1e9;
      worst = std::min(worst, gbs); best = std::max(best, gbs);
      printf(" %6.1f", gbs);
      HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    }
    printf("\n");
  }
  bool ratio_ok = true;
  if (all && n > 1) {
    ratio_ok = !(best > 2.0 * worst);
    printf("\nbest=%.1f worst=%.1f GB/s ratio=%.2f  %s\n  (PCIe3 x16 ~ 10-12, PCIe4 x16 ~ 20-24 GB/s)\n", best, worst,
           worst > 0 ? best / worst : 0.0,
           ratio_ok ? "(within 2x: OK)" : "(worst pair >2x slower than best: switch/uplink/BAR problem — gate 1 FAILS)");
  }
  return (all && ratio_ok) ? 0 : 1;
}

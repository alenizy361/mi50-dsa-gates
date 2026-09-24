#!/usr/bin/env bash
# Gate 1 — can kernels on MI50s write into each other's memory on THIS box?
# Collects the system facts that decide it (kernel, amdgpu params, IOMMU, 64-bit BAR placement,
# PCIe tree, ROCm topology) and runs the peer-access matrix + hipMemcpyPeer bandwidth.
#   sudo gates/01_p2p_matrix.sh      # sudo only so lspci shows LnkSta/LnkCap and dmesg is readable
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:${ROCM_PATH:-/opt/rocm}/bin"   # sudo's secure_path drops rocm-smi / rocm-bandwidth-test
mkdir -p results
OUT=${1:-results/gate1_$(hostname)_$(date +%Y%m%d_%H%M%S).txt}
LSPCI=lspci; DMESG=dmesg
if [ "$(id -u)" -ne 0 ]; then
  if sudo -n true 2>/dev/null; then LSPCI="sudo -n lspci"; DMESG="sudo -n dmesg"
  else echo "NOTE: not root and no passwordless sudo: LnkSta/LnkCap and dmesg lines will be missing; rerun with sudo for the link facts"; fi
fi
{
  echo "== host";   hostname; uname -r; date -u
  echo "== rocm";   cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm* 2>/dev/null
  echo "== cpu";    lscpu | grep -E 'Model name|Socket|NUMA node\(s\)'
  echo "== cmdline"; cat /proc/cmdline
  echo "== amdgpu module params"
  for f in /sys/module/amdgpu/parameters/{pcie_p2p,vm_size,vm_fragment_size,noretry}; do [ -r "$f" ] && echo "$f=$(cat "$f")"; done
  echo "== IOMMU (a translating IOMMU with per-device groups maps the HDP page NC; ACS can block P2P)"
  $DMESG 2>/dev/null | grep -iE 'iommu|AMD-Vi' | head -5
  for g in /sys/kernel/iommu_groups/*/; do [ -d "$g" ] && echo "$(basename "$g"): type=$(cat "$g/type" 2>/dev/null) devices=$(ls "$g/devices" | tr '\n' ' ')"; done 2>/dev/null | head -40
  echo "== AMD GPUs: link state and 64-bit BAR placement"
  echo "   (gfx906 has a 44-bit DMA mask — amdgpu gmc_v9_0.c, GC < 9.4.2 — so a BAR at/above 16 TiB = 0x100000000000 breaks or corrupts P2P; see x99-p2p-fix)"
  for dev in $($LSPCI -D -d 1002: | grep -iE 'VGA|Display|3D' | awk '{print $1}'); do
    echo "-- $dev $($LSPCI -s "$dev" | sed 's/^[^ ]* //')"
    $LSPCI -s "$dev" -vv 2>/dev/null | grep -E 'Memory at|LnkSta:|LnkCap:'
    for addr in $($LSPCI -s "$dev" -vv 2>/dev/null | grep -oiE 'Memory at [0-9a-f]+' | awk '{print $3}'); do
      if [ "$((16#$addr))" -ge "$((16#100000000000))" ]; then echo "   !! BAR at 0x$addr is ABOVE 16 TiB"; fi
    done
  done
  echo "== PCIe tree (which GPUs share a switch / an uplink)"; $LSPCI -tv 2>/dev/null | head -100
  echo "== rocm-smi --showtopo"; rocm-smi --showtopo 2>/dev/null | head -100
  echo "== rocm-bandwidth-test -t"; rocm-bandwidth-test -t 2>/dev/null | head -60
  echo "== p2p_matrix"; ./bin/p2p_matrix --mb 64
} 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}
if grep -q 'ABOVE 16 TiB' "$OUT"; then echo "!! a 64-bit BAR above 16 TiB was found (44-bit DMA mask): gate 1 FAILS"; [ "$rc" -eq 0 ] && rc=1; fi
echo "saved: $OUT (exit=$rc; 0 = every pair reachable, no BAR above 16 TiB, copy bandwidth within 2x)"
exit "$rc"

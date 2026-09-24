# Results: <chassis> / <cpu> / <n> × MI50 / ROCm <version>

## Box
| Field | Value |
|---|---|
| Chassis / motherboard | |
| CPU (model, sockets) | |
| PCIe generation the GPUs link at (`LnkSta` from gate 1, needs sudo) | |
| Switch layout (GPUs per PLX, uplink width) | |
| Kernel / amdgpu version / `pcie_p2p` | |
| ROCm version (and whether gfx906 objects were copied in) | |
| IOMMU mode (`iommu=` in gate 1, `iommu_src/dst=` in gate 2, `iommu:` in gate 3), ACS | |
| Any BAR ≥ 16 TiB? | |
| HDP register present (`hdp_reg=` non-NULL)? | |
| Cards: sramecc/xnack target-ID, VBIOS, cooling (passive chassis airflow / retrofitted fans) | |

## Gate 1 — P2P matrix
| | Value |
|---|---|
| All pairs `canAccessPeer`? | |
| hipMemcpyPeer GB/s best / worst / ratio | |
| Worst pair (which GPUs, same or different switch) | |

## Gate 2 — peer store + flag (fill one row per configuration)
| src→dst | same switch? | mode | HDP flush | words | iters | p50 µs | p99 µs | timeouts | STALE words | TAIL_STALE | TAIL_SUSPECT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0→1 | | pingpong | on | 3072 | 2000 | | | | | – | – |
| 0→1 | | pingpong | on | 0 | 2000 | | | | – | – | – |
| 0→1 | | ring | on | 65536 | 20000 | | | | | | |
| 0→1 | | ring | **off** | 65536 | 20000 | | | | | | |
| 0→4 | | ring | on | 65536 | 20000 | | | | | | |
| 0→4 | | ring | **off** | 65536 | 20000 | | | | | | |

(pingpong p50 = round trip; ring p50 = issue interval per 256 KB message)

## Gate 3 — one-shot allreduce
| GPUs | vec (bytes) | HDP | mean µs | p50 | p90 | p99 | egress GB/s at p50 | timeouts | wrong elements | RESULT |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 24576 | on | | | | | | | | |
| 8 | 12288 | on | | | | | | | | |
| 8 | 24576 | off | | | | | | | | |
| 4 | 24576 | on | | | | | | | | |

## Gate 4 — q8_0 GEMV bandwidth (kernel `dot` unless noted)
| shape | bytes (MB) | µs per-launch (incl. dispatch gap) | GB/s per-launch | GB/s in-kernel | read ceiling per-launch / in-kernel | spot-check max_rel |
|---|---|---|---|---|---|---|
| 256×6144 | 1.67 | | | | | |
| 6144×256 | 1.67 | | | | | |
| 4608×6144 | 30.1 | | | | | |
| 6144×2304 | 15.0 | | | | | |
| 6144×6144 | 40.1 | | | | | |

## Gate 5 — soak
| | Value |
|---|---|
| Duration, GPUs | |
| mclk min / max during load (from `mclk clock speed`) | |
| Memory-sensor temperature max | |
| GB/s first → last sample, worst GPU (HIP index; map to cardN via the `### hip` / `### bus map` rows in the CSV) | |
| Power cap (W) (from the `### power cap` block in the CSV) | |

## Notes
Anything odd: pairs that timed out, runs that only passed after a BIOS/kernel change, dmesg lines, whether `--no-hdp` ever produced TAIL_STALE.

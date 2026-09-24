# mi50-dsa-gates

Micro-benchmarks that decide whether a **753B DSA-sparse MoE (GLM-5.3) at Q8 can run on 32 × AMD Instinct MI50 (gfx906)** at ≥ 20 tok/s for one user — before anyone buys four servers or writes an inference engine.

> **Status: compiles clean, looking for the first results.** All four gates build with 0 errors / 0 warnings on ROCm 6.4.1 (`hipcc 6.4.43483`, `--offload-arch=gfx906`), and the gfx906 assembly contains what the design relies on (`v_dot4_i32_i8` ×88 in gate 4, `s_sleep` in every poll loop, `glc slc` HDP-flush stores, `buffer_wbinvl1_vol` acquire fences, `s_memrealtime` timestamps). They have **not yet run on a real MI50** — the authors have none in reach — so runtime behaviour is unverified; if anything fails on your box, please open an issue with the output. The story and the ask: [POST.md](POST.md).

The design under test: 4 single-root servers × 8 MI50 32 GB, tensor-parallel 8 inside a server, pipeline across servers, ~234 one-shot P2P collectives per token written from persistent kernels, DSA top-2048 sparse attention with a sequence-sharded indexer, MTP speculation, ~800 GB of q8_0 weights. Its decode budget is **≤ 576 µs per layer at 100k context**. A neutral review (100-agent panel, 45 findings, adversarially verified) found the design's arithmetic sound but resting on **five unmeasured assumptions that can each sink it**. These gates measure them. Each takes minutes on one box.

## The gates

| # | Question | Pass threshold | Needs | Time |
|---|---|---|---|---|
| 1 | Can kernels on MI50s write into each other's memory on this box (P2P through the PCIe switch)? | every pair `yes`; no BAR ≥ 16 TiB; copy BW within 2× across pairs (the binary checks peer access and the ratio; `01_p2p_matrix.sh` checks the BAR and folds all three into its exit code) | 2+ MI50 | 2 min |
| 2 | Kernel-side peer store + ready flag: latency, and **is the protocol sound** (no flag-before-data)? | one-way flag latency ≤ 10 µs (`--words 0` RTT / 2); **0 STALE / TAIL_STALE words** in `--stress` **with** HDP flush; report what `--no-hdp` does | 2 MI50 | 5 min |
| 3 | One-shot 8-GPU allreduce of a 12–24 KB vector from persistent kernels | p50 ≤ 40 µs (design), ≤ 95 µs (still meets 576 µs/layer); **0 wrong elements** | 4–8 MI50 | 3 min |
| 4 | Achieved HBM bandwidth of a q8_0 GEMV on the TP8 shapes: the 1.67 MB expert slices (256×6144, 6144×256), the fused 9-expert shapes (4608×6144 = 30 MB gate+up, 6144×2304 = 15 MB down) and a 40 MB reference (6144×6144) | ≥ 450 GB/s **in-kernel** on the 40 MB reference (the binary judges it); report the others as they are, both numbers | 1 MI50 | 3 min |
| 5 | Does HBM clock / temperature / bandwidth sag under a 10-minute all-GPU memory-bound soak? | mclk flat; last-sample GB/s ≥ 95 % of first | all GPUs in the chassis | 10 min |
| 6 | (model side) Which activations overflow FP16 (gfx906 has no BF16)? | list of sites needing FP32 | 8 × MI300X/MI325X (bf16) or 8 × H200 (FP8 checkpoint) | ~1 h incl. load |

Gate 1 failing means "stop": with host-staged collectives the design lands at ~20 tok/s with zero margin. Gate 2 with stale words means every collective in the design silently corrupts. Gate 3 printing `TOO SLOW`, or gate 4 printing `BELOW 450 GB/s` on the reference while its read ceiling is healthy, means the layer budget is gone.

## Quick start (ROCm 6.3.x / 6.4.x recommended — the last releases shipping gfx906 math libraries; 7.x works only with copied gfx906 objects)

```bash
git clone <this repo> && cd mi50-dsa-gates
./build.sh                                   # hipcc --offload-arch=gfx906 ... -lhsa-runtime64
sudo gates/01_p2p_matrix.sh                  # gate 1 (sudo only so lspci shows LnkSta/LnkCap); writes results/gate1_*.txt
./bin/peer_flag --src 0 --dst 1              # gate 2, ping-pong latency with payload verification
./bin/peer_flag --src 0 --dst 1 --words 0    # gate 2, bare-flag round trip RTT0 (one-way flag ~ RTT0/2)
./bin/peer_flag --src 0 --dst 1 --stress     # gate 2, credit ring: 20k messages of 256 KB, tail-first verification
./bin/peer_flag --src 0 --dst 1 --stress --no-hdp   # what happens without the HDP flush?
./bin/oneshot_allreduce                      # gate 3, all visible GPUs, 24 KB fp32
./bin/oneshot_allreduce --vec 3072           # gate 3, 12 KB
./bin/gemv_q8_bw                             # gate 4, all shapes, sdot4 kernel (add --kernel naive to compare)
gates/05_thermal_soak.sh 600                 # gate 5 (needs rocm-smi on PATH)
python3 gates/06_activation_probe.py --model zai-org/GLM-5.3 --prompt-file prompt.txt   # gate 6, on the big node
```

Run gate 2 for a pair on the **same** PCIe switch and a pair on **different** switches (gate 1's tree shows which). The P2P kernels use bounded spins (~3–6 s at the default `--max-spins`) and poison their peers' flags on timeout, so a run reports `TIMEOUT` / `ABORTED` instead of hanging; if you see one, P2P is broken for that pair or `--max-spins` is too low. The binaries set `HSA_FORCE_FINE_GRAIN_PCIE=1` themselves, verify the memory pool they got and refuse to run on coarse-grained memory (exit 4), and refuse `hdp_flush=on` when the runtime exposes no HDP register (exit 5) — rerun with `--no-hdp` and say so. Exit codes: 0 pass, 1 result failure, 2 HIP error, 3 no peer access, 4 no UC memory, 5 no HDP register, 6 (gate 3) correct but slower than the layer gate.

## What to report

Copy `RESULTS_TEMPLATE.md`, fill it, and open an issue titled `results: <chassis> <cpu> <n>×MI50 ROCm <ver>` with it plus the `results/` files. The facts that matter most: chassis/motherboard, CPU (Rome/Milan single-root vs Naples multi-die), PCIe generation, switch layout, kernel + amdgpu + ROCm versions, IOMMU mode (printed by gates 1–3 as `iommu=`, `iommu_src=`/`iommu_dst=`, `iommu:`), BAR placement, and whether gate 2 `--stress` is clean **with** and **without** the HDP flush.

## Why these exact protocol details

* Plain `hipMalloc` memory on gfx906 is mapped MTYPE **NC**: L2-cached, made coherent only at kernel boundaries — a persistent kernel polling such a flag spins on a stale line forever, and a writer's stores stay dirty in its L2. The gates allocate flags and inboxes with `hipDeviceMallocUncached` (ROCm ≥ 5.5), falling back to `hipDeviceMallocFinegrained`; on gfx906 both are MTYPE UC. Because CLR ≥ 6.4 silently substitutes coarse memory when a pool is absent, the pool is verified with `hsa_amd_pointer_info`.
* Over PCIe the writer must **flush the receiver's HDP write path** (`HDP_MEM_COHERENCY_FLUSH_CNTL`, exposed as `hipDeviceProp_t::hdpMemFlushCntl`) after its data stores and before its flag store. RCCL does this for non-xGMI peers on gfx906/gfx908 (it skips gfx90a/gfx942/gfx950): `p2pSendSetup` in `src/transport/p2p.cc` fetches the receiver's register, `postPeer` in `src/device/prims_simple.h` stores 1 to it right before the tail store. Without it a flag can become visible before its payload; `--no-hdp` exists so you can see whether your box shows it. The flush store is a relaxed store like RCCL's; if you suspect the register page is not UC-mapped on your box (translating IOMMU, printed as `iommu=grpN:DMA`), rerun gate 2 with `--hdp-readback`, the only variant that is ordered by construction (a non-posted read cannot complete before the earlier posted writes). Gates 1–3 print each GPU's IOMMU group so results can be attributed.
* No read-modify-write atomics ever target a peer GPU's memory over PCIe (not supported on gfx906 and never done by RCCL). Flags are monotonic system-scope stores and loads, one per 64-byte line.
* Every poll loop uses `s_sleep` and a spin bound; a GPU that times out poisons its peers' flags so they stop within one poll.
* Gate 2's ping-pong cannot expose flag-before-data (one message in flight, ~1 µs of poll latency); the `--stress` credit ring keeps 8 × 256 KB in flight and the polling lane samples the payload's last line in the same load batch as the flag (`TAIL_SUSPECT`), then re-reads it after the acquire fence (`TAIL_STALE`), then all threads verify the whole slot descending.

## Background

* Design + review: the v5 architecture document and its verified review (link in the issue tracker).
* MI50 P2P being non-turnkey: ROCm issue #4793 (now at ROCm/legacy-rocm-build; P2P N/A on 2 × MI50), DByte308/x99-p2p-fix (44-bit DMA mask vs BAR placement; +22–26 % after fixing), vllm-gfx906 forks running with `NCCL_P2P_DISABLE=1`.
* Prior art for the KV tier the design proposes: [ESS](https://arxiv.org/abs/2512.10576) (latent cache offload for DeepSeek-V3.2), [HiSparse](https://arxiv.org/abs/2608.07009) (hierarchical KV cache for sparse attention; LRU miss rate 13.4 % at a 4096-entry GPU cache), [NVIDIA GVR](https://arxiv.org/abs/2604.22312) (measured top-2048 overlap between consecutive DSA steps: 35–50 % in deep layers, ~1–2 % in layers 0–1).
* Megakernels on AMD: kog.ai single-kernel engine on 8 × MI300X; Hazy Research TP megakernel.

## Contributing

Better kernels (a multi-workgroup allreduce, a sentinel-based barrier, a faster GEMV) are welcome — they can only raise the measured numbers. Keep the protocol invariants above; add a shape or a mode rather than changing pass thresholds.

MIT license.

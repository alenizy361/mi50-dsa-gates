# Community post — r/LocalLLaMA · llama.cpp discussion #15021 · gfx906 project issues (see OUTREACH in the issue tracker)

**Title:** We think ~$5k of end-of-life MI50s can run a 753B frontier MoE (GLM-5.3, Q8) at 27 tok/s. Nobody has tried. We need 30 minutes of your cards to find out.

**TL;DR:** Paper design says 32 × MI50 32 GB in four boxes can decode the *full* GLM-5.3 at q8_0, 100k context, ~27 tok/s for one user (35+ with MTP). It hinges on five things no one has ever measured on gfx906. We wrote five small benchmarks. Run them, paste the numbers, get credit. If it works, it's the cheapest frontier-class local rig anyone has built. If it doesn't, you'll have saved a few people four servers each.

---

## The bet

GLM-5.3 is 753B parameters — ~800 GB at q8_0. It also uses DeepSeek-V3.2-style **DSA sparse attention**: every token attends to only the top-2048 positions its indexer picks, so a million-token context costs about what 2k tokens cost. That changes what "cheap hardware" can do.

The MI50 is a 2018 datacenter card: 32 GB of HBM2 at ~1 TB/s, no matrix cores, no BF16, no FP8, deprecated by ROCm — and ~$150 used. Thirty-two of them hold the whole model. Under TP8 inside a box, every card reads ~94 MB of weights per layer per token; at the ~500 GB/s people actually measure on these cards that's 190 µs. Add two 12 KB all-reduces, the indexer, and launch overhead: **~460 µs per layer, 78 layers, ~27 tok/s.** The whole design lives or dies on a single line: **average layer time ≤ 576 µs at 100k context.**

Nobody has run this model — or any DSA model — on gfx906 with tensor parallelism. llama.cpp runs GLM-5 with the indexer ignored; every vllm-gfx906 fork runs with P2P disabled. So the design is all new: persistent kernels writing directly into each other's memory over the PCIe switches (~234 times per token), a sequence-sharded indexer with an exact global top-2048, a tiered KV cache for long context.

## Five things that could kill it

1. **Kernel-side P2P through the PCIe switch.** MI50 P2P is famously fragile: ROCm #4793 is still open, the card has a 44-bit DMA mask so a BAR above 16 TiB silently corrupts, and every vllm-gfx906 fork gave up and set `NCCL_P2P_DISABLE=1`. If this fails, every collective goes through the host: **~20 tok/s with zero margin.**
2. **"Store the data, then store the flag" — is it actually sound?** Over PCIe on gfx906 the writer has to flush the *receiver's* HDP path before the flag (RCCL does this quietly); skip it and the flag can land before the data. No error, just wrong numbers. Our stress test keeps 8 × 256 KB in flight and checks the payload's last line in the same load batch as the flag.
3. **One-shot 8-GPU all-reduce latency** from persistent kernels. Budget: 40 µs. Survives up to ~95 µs. Nobody has published a number for MI50s behind PLX switches.
4. **GEMV bandwidth on tiny slices.** TP8 cuts each expert into 1.67 MB matrices. The 500 GB/s people quote comes from 8B dense models. Does a 1.67 MB q8_0 slice get anywhere near it? (Reported per-launch *and* in-kernel, next to a raw streaming-read ceiling, so a miss can be blamed on the box or on the kernel.)
5. **Thermal soak.** Eight passively cooled 300 W cards, ten minutes of memory-bound load. Does the HBM clock hold?

## What we built

`https://github.com/alenizy361/mi50-dsa-gates` — five gates, ~1,100 lines of HIP + scripts, MIT.

```
./build.sh                                # hipcc --offload-arch=gfx906
sudo gates/01_p2p_matrix.sh               # P2P matrix, BAR placement, PCIe tree, IOMMU     ~2 min
./bin/peer_flag --src 0 --dst 1 --stress  # store+flag soundness, with and without --no-hdp ~5 min
./bin/oneshot_allreduce                   # 8-GPU one-shot all-reduce, 24 KB               ~3 min
./bin/gemv_q8_bw                          # q8_0 GEMV on the real TP8 shapes                ~3 min
gates/05_thermal_soak.sh 600              # all-GPU soak with telemetry                     10 min
```

Every P2P kernel has bounded spins and poisons its peers on timeout — nothing can hang your box. Every binary prints the facts the numbers depend on (kernel, ROCm, IOMMU group, memory pool, HDP register) and a plain `RESULT:` line against the design's thresholds.

## What we need, from whom

- **2+ MI50s on one root complex** → gates 1 and 2. Especially `--stress` with and without `--no-hdp`: does *your* box ever show `TAIL_STALE`?
- **4–8 MI50s behind PLX switches** (Supermicro 4028/4029, Gigabyte G4xx, Dell C4140…) → gate 3. This is the number the whole design is waiting for.
- **One MI50, any box** → gates 4 and 5.

Paste `RESULTS_TEMPLATE.md` into an issue titled `results: <chassis> <cpu> <n>×MI50 ROCm <ver>`. Chassis, CPU generation, PCIe gen and switch layout matter as much as the numbers.

## What you get

Every result goes into the repo with your name on it. If the gates pass, the next steps are public too: a fused-kernel layer, then the DSA indexer split, then the long-context tier — and the first rig that runs it. If they fail, the failure is published just as loudly; that's worth four servers to a lot of people.

## Honest caveats

Nothing here is measured yet. The design's arithmetic was checked by a review that verified every model number against the released config and every hardware claim against ROCm, RCCL and kernel sources — and the code compiles clean for gfx906 on ROCm 6.4.1 (0 warnings; the assembly shows the `v_dot4_i32_i8`, `s_sleep` and `glc slc` instructions the design relies on) — but it has never *run* on a real MI50: the authors have none in reach. Runtime surprises are possible; PRs are welcome and will be merged fast. And this is engineering, not science: every technique here exists somewhere (RCCL's HDP flush, ESS/HiSparse for the KV tier, kog.ai's MI300X megakernel). The new part is putting them on a $150 card for a 753B model, and finding out whether the card says yes.

Background for the curious: [ESS](https://arxiv.org/abs/2512.10576), [HiSparse](https://arxiv.org/abs/2608.07009), [NVIDIA GVR](https://arxiv.org/abs/2604.22312) (measured DSA top-k churn), ROCm #4793, DByte308/x99-p2p-fix.

The gfx906 crowd has already done more for cheap inference than any vendor. Let's see how far the card goes.

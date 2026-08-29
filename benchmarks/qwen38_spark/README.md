# Qwen3.8-Flash-Next on DGX Spark

This is the canonical experiment log for improving single-request decode throughput for
Qwen3.8-Flash-Next on one NVIDIA DGX Spark. Raw benchmark JSON is deliberately
untracked; each accepted result is summarised here with the exact image, launch
configuration and measurement protocol.

## Goal

Increase warm single-request decode throughput from approximately 35 tokens/s towards 50
tokens/s without changing target-model output semantics. Optimisations may change the
speculative drafter, quantisation, kernels, scheduling, PLE storage or host
configuration.

## Hardware

- Host: `spark-0774`
- GPU: NVIDIA GB10, SM121, 121 GiB unified memory
- Driver: 580.159.03; CUDA runtime: 13.0
- Storage: local NVMe, ext4
- CPU: 10 Cortex-X925 performance cores and 10 Cortex-A725 efficiency cores

## Known-good rollback

- Runtime repository: `blazux/qwen3.8-Flash-DGX`
- Runtime branch/commit: `fix/mmap-random-advice` at `8cb9b3d`
- Container: `qwen38-flash`
- Image: `sha256:1bf3e6c942b3d20577169edecd798c6fe5eaedee5e25e6fe19790c13b8c404fd`
- Base image digest:
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
- vLLM reported version: `0.1.dev20073+g8e685d198`
- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4`
- Model revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`

The known-good image uses direct FP8 PLE mmap with `MADV_RANDOM`. Do not overwrite or
remove it when building experimental images.

## Baseline server configuration

```text
max_model_len=262144
max_num_seqs=8
gpu_memory_utilization=0.80
kv_cache_dtype=auto
cudagraph_mode=PIECEWISE
mtp_speculative_tokens=2
prefix_caching=false
VLLM_PLE_MMAP=1
VLLM_PLE_MMAP_WORKERS=32
VLLM_PLE_MMAP_PREWARM=0
VLLM_USE_FLASHINFER_SAMPLER=1
flashinfer_autotune=false
```

The allocation is 79.42 GiB model memory and 15.32 GiB KV cache, or 553,908 KV tokens.
The PLE table is a 47.68 GiB file-backed FP8 mapping.

## Benchmark protocol

The primary runner is [`llama-benchy`](https://github.com/eugr/llama-benchy), pinned to
version 0.4.0. It uses coherent Project Gutenberg text, adapts to chat-template
overhead, handles multi-token MTP stream chunks, distinguishes the first response from
the first content token, and reports decode throughput only over observable token
intervals.

Run the quick optimisation loop from the repository checkout on Spark:

```bash
LABEL=<configuration> benchmarks/qwen38_spark/run_serving_benchmark.sh quick
```

The quick suite uses a 256-token prompt, 512 exact generated tokens, depth 0, five runs
and concurrency 1. The context confirmation suite uses depths 0, 8,192 and 32,768 with
three runs:

```bash
LABEL=<configuration> benchmarks/qwen38_spark/run_serving_benchmark.sh context
```

The wrapper performs an unmeasured warm-up and coherence check before capturing
counters. Measured requests then use `ignore_eos=true`, temperature 1, top-p 0.95, top-k
20, seed 42, no prefix-cache reuse and no additional warm-up requests. It saves
llama-benchy JSON plus the delta of vLLM's draft, accepted-token and per-position
Prometheus counters.

Primary metrics are llama-benchy's token-generation throughput, mean accepted length and
approximate target steps/s. Compare steps/s separately from accepted tokens/s because
generated content can move speculative acceptance by more than 10% without a runtime
change.

## Results

### B0: MADV_RANDOM, MTP 2, 80% GPU allocation

Canonical quick result, five runs on coherent Sherlock text:

- Sustained generation: **25.08 +/- 1.97 tokens/s**; range 22.18-27.38.
- One-second peak: **35.40 +/- 1.36 tokens/s**.
- TTFR: **406.88 +/- 18.38 ms** for the 256-token prompt.
- Drafts: 1,211; accepted draft tokens: 1,348 of 2,422.
- Mean accepted length: **2.113 tokens per target step**.
- Approximate target rate: **11.87 steps/s**.
- Unconditional MTP acceptance: 64.82% at position 0 and 46.49% at position 1.

The earlier 35 tokens/s smoke result tracked the short-window peak, not sustained
throughput over coherent 512-token continuations. Future improvements are compared with
25.08 tokens/s and 11.87 target steps/s, not the old smoke figure.

### R1: MTP 3

Canonical quick result with an otherwise identical server:

- Sustained generation: **25.74 +/- 1.47 tokens/s**; range 23.85-27.76.
- One-second peak: **42.00 tokens/s** in all five runs.
- Drafts: 1,013; accepted draft tokens: 1,550 of 3,039.
- Mean accepted length: **2.530 tokens per target step**.
- Approximate target rate: **10.17 steps/s**.
- Unconditional MTP acceptance: 66.83%, 48.08% and 38.10% by position.

Relative to MTP 2, accepted length improved by 19.7%, target rate fell by 14.3%, and
sustained throughput improved by only 2.6%. That is smaller than the run-to-run spread,
so MTP 3 is not a decisive BF16-drafter win. It remains useful for testing whether a
quantised drafter changes the optimum.

### L1: accelerated model loaders

Status: **unsafe on the current unified-memory checkpoint layout**.

Fastsafetensors 0.3.3 with queue size 0 loaded the first 193 body/expert files in about
43 seconds, compared with roughly 8-9 minutes for the standard loader. The unfiltered
run then reached the 47.7 GiB PLE files, reduced available memory to 1.7 GiB and wedged
the host, requiring a physical reboot.

A second image used a filtered index with zero PLE entries and mapped PLE from the
original snapshot. It still fell from 38 GiB available to 0.8 GiB at file 194 of 196.
The watchdog stopped the container before another reboot. This demonstrates that
fastsafetensors retains enough device-side body storage to overlap dangerously with the
preallocated 78 GiB model, even without PLE. InstantTensor was not attempted because its
vLLM iterator uses `copy=True` and has the same fundamental memory-envelope risk.

Runtime commits `d7c04fc`, `03a2f8e` and `de73f11` add loader controls, PLE separation
and a separate MTP path. The launch script now refuses accelerated loaders without a
filtered PLE view. Production remains on the standard safetensors loader.

### Q3: Inferact NVFP4 MTP

A separate 1.6 GiB draft view contains only Inferact's 6,173 MTP tensors. The target
remains the Radix checkpoint and uses its original quantisation config.

At MTP 3:

- Sustained generation: **29.89 +/- 2.11 tokens/s**; range 27.29-33.05.
- One-second peak: **44.60 +/- 0.49 tokens/s**.
- Mean accepted length: **2.751 tokens per target step**.
- Approximate target rate: **10.86 steps/s**.
- Unconditional MTP acceptance: 70.53%, 56.38% and 48.23% by position.
- Model memory: **76.07 GiB**, down from 79.42 GiB.
- KV cache: **18.25 GiB / 645,945 tokens**, up from 15.32 GiB / 553,908.

This is a **19.2% sustained gain** over BF16 MTP 2 and a 16.2% gain over BF16 MTP 3.
Compared with BF16 MTP 3, target rate improved by 6.8% and accepted length by 8.7%.
Quantising the draft experts therefore improved both cost and acceptance on this corpus.

At MTP 4, sustained generation fell to **24.18 +/- 4.40 tokens/s**, accepted length to
2.625 and target rate to 9.21 steps/s. MTP 3 remains the optimum among tested depths.

### Q1: Primitive mixed FP8 dense target

A composite target used Primitive's mixed FP8 QSA/GDN body, Radix's FP8 mmap PLE and the
Inferact NVFP4 MTP-3 draft. The runtime selected Cutlass FP8 dense linears, Triton GDN
decode and FlashInfer CUTLASS NVFP4 experts.

- Sustained generation: **26.29 +/- 0.54 tokens/s**.
- Mean accepted length: **2.187 tokens per target step**.
- Approximate target rate: **12.02 steps/s**.
- Model memory: **73.67 GiB**; KV cache: **20.27 GiB / 716,913 tokens**.

Dense FP8 improved target rate by 10.7% relative to the best Radix/NVFP4-MTP target, but
accepted length fell by 20.5%. Its target quantisation drift made the independently
quantised MTP agree less often, leaving throughput 12.0% below the best configuration.
The mixed target is therefore not used in production.

### Q2: dynamic FP8 LM head

Runtime commit `12fd044` and image
`sha256:0e53b90c0dcf53fe0b20a8fac51add022b92756ea810ca5ba03265098b891886` replace the
BF16 target LM head with a non-serialised dynamic `Fp8Config` head.

- Sustained generation: **26.56 +/- 2.03 tokens/s**.
- Mean accepted length: **2.466 tokens per target step**.
- Approximate target rate: **10.77 steps/s**.
- One-second peak: **43.40 +/- 0.49 tokens/s**.

This dynamic FP8 kernel did not improve target rate on SM121 and its logit drift reduced
MTP agreement. It remains disabled. A serialised per-channel FP8 head may use a
different kernel/scaling path and is not ruled out; fork commit `2fa076f90` passes
target and draft quant configs into Qwen4Exp LM heads to support such checkpoints.

### K1: QSA sparse-attention profile sweep

SGLang's SM121 path uses FlashInfer TRT-LLM attention over a compressed 64-token QSA
page layout. The vLLM branch instead passes 2,051 token-level indices into a custom
Triton kernel over 8-token main-cache pages, so the FlashInfer kernel cannot consume its
metadata without repacking K/V or redesigning the cache.

A synthetic sweep used the real Qwen dimensions (24 query heads, 2 K/V heads, head
dimension 256) and tested Triton tile widths 16/32/64, split counts 8/16/32/64 and 2/4
warps for both one-row draft and four-row MTP-3 verification calls.

- One row: upstream **0.0358 ms**; best **0.0355 ms**.
- Four rows: upstream **0.0363 ms**; best **0.0359 ms**.
- Maximum BF16 output difference across valid profiles: **4.9e-4**.

The roughly 1% subkernel change is too small to affect serving throughput. Runtime
commit `199bf10` preserves an opt-in sweep harness and profile overrides, but the tuning
image is not deployed. A useful TRT-LLM experiment requires a different QSA cache layout
or an explicit gather benchmark, not a direct function substitution.

### K2: reduced piecewise graph boundaries

The mmap PLE lookup is the only host-side custom op and appears once per model forward,
at model layer 2. An experiment therefore retained `PIECEWISE` capture but reduced
`splitting_ops` from the known-good QSA/GDN/indexer/attention list to only
`vllm::ple_mmap_lookup`.

Compilation and checkpoint loading completed, but the first warm-up hit an asynchronous
CUDA illegal-memory access in a captured GPU custom op. The error surfaced when the host
PLE lookup synchronized the stream and aborted the engine. At least one QSA, GDN or
sparse-indexer transaction in this private runtime is not graph-safe despite having a
torch.compile fake implementation.

The full known-good split list was restored. `FULL_DECODE_ONLY` remains unsafe with
direct mmap because its CPU gather and pageable H2D copy must execute for every new
n-gram. Further graph work requires per-op capture tests or a newer runtime with
explicit Qwen full-graph support; do not repeat cold-start split-list bisection blindly.

A same-session control before the graph test measured **27.70 +/- 2.65 tokens/s**, mean
accepted length 2.604 and 10.64 target steps/s. During decode the GPU held 2.46-2.48 GHz
at roughly 86-94% SM activity without a thermal-throttle event.

### K3: SM121 skinny-GEMM LM head

The private runtime ships vLLM's CuTe DSL shape-dynamic skinny GEMM, but Qwen enables it
only for SM103 with TP4-specific plans. The TP1 target and MTP drafter each own a BF16
`248320 x 2560` LM head. An SM121 autotune found the same M=4 plan for target
verification and padded one-row draft calls:

```text
SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4)
```

Synthetic full-head timings were:

- M=1 cuBLAS: **7.43 ms**; best skinny GEMM: **5.14 ms** (**1.44x**).
- M=4 cuBLAS: **5.35 ms**; best skinny GEMM: **5.10 ms** (**1.05x**).
- Relative RMS error: 1.8e-5 at M=1 and 1.1e-4 at M=4.

Padding M=1 to four rows and using the M=4 kernel made draft row 0 bit-identical to row
0 of target M=4 across all 248,320 synthetic logits. It did not, however, preserve
agreement between the different target and MTP hidden states in serving.

| Variant                           | Sustained tokens/s | Accepted length | Target steps/s |
| --------------------------------- | -----------------: | --------------: | -------------: |
| Same-session cuBLAS control       |     27.70 +/- 2.65 |           2.604 |          10.64 |
| Separate M=1/M=4 skinny plans     |     26.98 +/- 1.59 |           2.418 |          11.16 |
| Both heads, aligned M=4 plan      |     27.97 +/- 4.26 |           2.424 |          11.54 |
| Draft head only, aligned M=4 plan |     25.78 +/- 3.04 |           2.423 |          10.64 |
| Canonical Q3 result               | **29.89 +/- 2.11** |       **2.751** |          10.86 |

Replacing both heads raised target-step rate by 8.5% relative to the same-session
control, but numerical drift reduced speculative acceptance by 6.9%. Restricting the
kernel to the draft preserved target-model semantics but removed the step-rate gain and
still reduced acceptance. None of the variants beats the canonical configuration, so
production retains cuBLAS for both BF16 heads.

Runtime commits `bb8e0b1` and `dc6bc6c` fix empty QSA overrides and add the opt-in SM121
benchmark/dispatcher. The final draft-only image is
`sha256:9219d1f7496afe0628899cdd282f11213baec21d7a5a9b409c03660c76af83fa`.

### Best validated configuration

The production choice after these experiments is:

```text
target=RadixArk/Qwen3.8-Flash-Next-NVFP4
ple=RadixArk FP8 direct mmap with MADV_RANDOM
draft=/hf/qwen38-inferact-mtp
mtp=3
lm_head=BF16
gpu_memory_utilization=0.80
```

Its canonical sustained rate is **29.89 +/- 2.11 tokens/s**, 19.2% above the original
baseline, with a 44.6 tokens/s one-second peak.

## Experiment queue

- **B0 (complete):** Canonical current-image llama-benchy baseline.
- **R1 (partial):** MTP 3 measured; depths 0 and 4 deferred.
- **L1 (failed):** Accelerated loaders exceed unified-memory headroom.
- **R2:** PLE worker count and X925 affinity; reduce host overhead.
- **R3:** Explicit smaller KV allocation; reserve more PLE page cache.
- **Q1 (failed):** Mixed dense FP8 raises step rate but collapses MTP acceptance.
- **Q2 (failed):** Dynamic FP8 head is slower and reduces MTP acceptance.
- **Q3 (successful):** Inferact NVFP4 MTP gives 29.89 tokens/s at depth 3.
- **K1 (closed):** QSA Triton profile retuning changes a 0.036 ms subkernel by ~1%;
  FlashInfer needs a different cache layout.
- **K2 (failed):** Minimal-split piecewise capture faults in a captured QSA/GDN/indexer
  op; full capture remains incompatible with mmap PLE.
- **K3 (failed):** SM121 skinny-GEMM heads raise step rate but reduce MTP acceptance;
  draft-only mode has no serving speedup.
- **P1:** Asynchronous/direct PLE gather; remove GPU-CPU-GPU synchronisation.
- **P2:** 28.8-32 GiB PLE; keep most or all of the table resident.

## Findings

1. `MADV_RANDOM` is mandatory for direct mmap. Without it, Linux sequential readahead
   amplifies sparse 160-byte PLE reads into multi-gigabyte/s NVMe bursts and causes
   severe memory pressure.
2. The 6B active-parameter headline does not describe decode traffic. GB10 profiling on
   the public Qwen vLLM branch attributes about 69% of no-speculative token time to BF16
   dense GEMV. The already-NVFP4 routed experts account for less than 1% in that trace.
3. The public Qwen branch now resolves an independent draft quantisation config in
   `Qwen4ExpMTP`; quantised MTP may therefore require checkpoint work more than model
   plumbing.
4. vLLM exports cumulative draft, accepted-token and per-position counters. The wrapper
   captures their deltas around llama-benchy, allowing acceptance and step rate to be
   separated from token-generation throughput.
5. Fastsafetensors can load the body files about ten times faster, but its device
   tensors overlap model allocations and exhaust 121 GiB unified memory even after
   removing PLE from the checkpoint index. InstantTensor has the same likely risk.
6. Inferact's isolated NVFP4 MTP draft cuts model memory by 3.35 GiB and raises
   sustained generation by 19.2%. MTP 3 is optimal; MTP 4 loses target-step rate.
7. Target quantisation must be evaluated with MTP acceptance, not only kernel or
   target-step speed. Both dense FP8 and dynamic FP8-head experiments lost more accepted
   length than they gained in step rate.
8. The current Spark CPUs already use the performance governor. The GPU runs around 2.47
   GHz under load versus a 3.00 GHz nominal maximum. A measured control held 2.46-2.48
   GHz at 86-94% SM activity with no active thermal-throttle event, so cooling or clock
   tuning is secondary to reducing BF16 memory traffic.
9. vLLM's token-indexed QSA Triton attention is already only about 0.036 ms for the
   relevant one-row and four-row calls. SGLang's TRT-LLM gain relies on a different
   compressed-page cache layout and is not a drop-in kernel port.
10. Compile fake implementations do not imply CUDA-graph safety. Capturing the Qwen GPU
    custom ops by removing their split points caused an illegal memory access during
    warm-up; direct mmap must retain the known-good piecewise split list.
11. A faster LM-head kernel can lose end-to-end throughput through speculative
    disagreement even when its synthetic relative RMS error is below 1.2e-4. Padded
    M=1/M=4 accumulation alignment was bit-identical for equal inputs but did not align
    target and MTP logits from different hidden states.

## Sources

- [Qwen3.8-Flash-Next vLLM support PR][qwen-pr]
- [Disk-backed PLE PR][ple-pr]
- [Primitive mixed NVFP4/FP8 checkpoint][primitive-mixed]
- [Primitive quantised PLE tables][primitive-ple]
- [SGLang DGX Spark experiments][sglang-spark]

[qwen-pr]: https://github.com/vllm-project/vllm/pull/53896
[ple-pr]: https://github.com/vllm-project/vllm/pull/54070
[primitive-mixed]: https://hf.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8
[primitive-ple]: https://hf.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant
[sglang-spark]: https://github.com/hashd1ve/qwen38-flash-next-one-dgx-spark

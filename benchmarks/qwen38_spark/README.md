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

## Experiment queue

- **B0 (complete):** Canonical current-image llama-benchy baseline.
- **R1 (partial):** MTP 3 measured; depths 0 and 4 deferred.
- **L1 (failed):** Accelerated loaders exceed unified-memory headroom.
- **R2:** PLE worker count and X925 affinity; reduce host overhead.
- **R3:** Explicit smaller KV allocation; reserve more PLE page cache.
- **Q1:** FP8 QSA/GDN projections; remove dominant BF16 bandwidth.
- **Q2:** FP8 LM head; avoid a 1.18 GiB BF16 scan per target row.
- **Q3 (successful):** Inferact NVFP4 MTP gives 29.89 tokens/s at depth 3.
- **K1:** FlashInfer/TRT-LLM QSA decode; reproduce SGLang's SM121 gain.
- **K2:** Full decode CUDA graphs after making PLE graph-safe.
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
7. The current Spark CPUs already use the performance governor. The GPU runs around 2.47
   GHz under load versus a 3.00 GHz nominal maximum and has historical software power
   and thermal throttle counters; cooling is a secondary experiment, not a substitute
   for reducing BF16 memory traffic.

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

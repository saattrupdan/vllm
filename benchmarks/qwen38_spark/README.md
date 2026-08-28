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
counters. Measured requests then use `ignore_eos=true`, temperature 0, seed 42, no
prefix-cache reuse and no additional warm-up requests. It saves llama-benchy JSON plus
the delta of vLLM's draft, accepted-token and per-position Prometheus counters.

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

## Experiment queue

- **B0 (running):** Canonical current-image llama-benchy baseline.
- **R1:** MTP depth 0, 2, 3 and 4; determine the BF16 drafter optimum.
- **R2:** PLE worker count and X925 affinity; reduce host overhead.
- **R3:** Explicit smaller KV allocation; reserve more PLE page cache.
- **Q1:** FP8 QSA/GDN projections; remove dominant BF16 bandwidth.
- **Q2:** FP8 LM head; avoid a 1.18 GiB BF16 scan per target row.
- **Q3:** FP8/NVFP4 MTP; lower draft cost and permit deeper speculation.
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
5. The current Spark CPUs already use the performance governor. The GPU runs around 2.47
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

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

Results through K3 predate the explicit sampling arguments: the runner sent only
`seed=42`, so those measurements used vLLM's unrestricted sampling defaults. Their
within-group comparisons remain valid, but they must not be compared directly with the
explicit top-k/top-p results in S1. The runner was corrected before the S1 baseline.

Content sensitivity is measured with the exact frozen v1 workload matrix from
[`hasso5703/dgx-spark-qwen38`](https://github.com/hasso5703/dgx-spark-qwen38). The
vendored [`bench_matrix_v1.sh`](bench_matrix_v1.sh) is byte-identical to upstream commit
`34c7fa98ab4055f30569baa3c06cf85a91139237`; its SHA-256 is
`6078517736a056fea182fea9ebc77d08f4eb38fb1afd3fe7f30255b51c069b24`. It uses eight fixed
greedy probes and an 80-versus-680-token two-call delta to remove prefill time:

```bash
BASE_URL=http://127.0.0.1:8000 \
MODEL=qwen3.8-flash-next API_KEY= LABEL=<configuration> \
benchmarks/qwen38_spark/bench_matrix_v1.sh
```

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

These Q3 results used unrestricted sampling defaults despite the protocol text at the
time. S1 establishes the corresponding explicit top-k/top-p baseline and production
result.

### S1: replay top-k/top-p in probabilistic draft sampling

The private runtime already retained proposal probabilities for exact rejection
sampling, but its draft sampler applied only request temperature. For requests using
`top_k` or `top_p`, the draft could therefore propose tokens outside the target's
filtered support, guaranteeing rejection.

Runtime commit `d37e4f0` adds an opt-in path that applies the request's top-k/top-p
filters before the draft softmax and passes those same filtered probabilities to the
standard rejection sampler. Target-model semantics remain exact.

With explicit temperature 1, top-p 0.95 and top-k 20, the unpatched five-run baseline
was:

- Sustained generation: **25.25 +/- 1.97 tokens/s**.
- Mean accepted length: **2.372 tokens per target step**.
- Approximate target rate: **10.64 steps/s**.
- Draft acceptance: **45.74%**.

Three replay measurements, totalling 15 runs and including a cold reboot recovery, gave:

- Sustained generation: **27.77 +/- 3.09 tokens/s**; range 23.45-32.64.
- Mean accepted length: **2.568 tokens per target step**.
- Approximate target rate: **10.79 steps/s** across the three measurements.
- Draft acceptance: **52.26%** across 8,973 proposals.
- Unconditional MTP acceptance: 67.47%, 49.68% and 39.62% by position.

Replay improves sustained throughput by **10.0%** and accepted length by 8.2%, without
reducing target-step rate. A five-run unrestricted-default control on the replay image
measured 28.29 +/- 1.96 tokens/s and 2.661 accepted tokens per target step; requests
without top-k/top-p remain on the existing proposal path.

The deployed image is `qwen38-flash-dgx:draft-topkp`, digest
`sha256:d9655ad4cff5a5310d752ccfb6811b7ccacfab62917f4a37c9ccd9045c059b5c`. It recovered
automatically after a host reboot and passed `/health` and `/v1/models`.

### S2: frozen content workload matrix

The exact upstream matrix was run unchanged against the replay image on port 8000:

| Probe                         | Sustained tokens/s |
| ----------------------------- | -----------------: |
| Math, English                 |         unreliable |
| Code, English                 |           **28.0** |
| Code, German                  |           **27.7** |
| Technical explanation, French |           **27.0** |
| Reasoning, French             |           **36.6** |
| Free prose, English           |           **24.8** |
| Free prose, French            |           **24.0** |
| Free prose, German            |           **22.1** |

The math answer was too short for the matrix's reliability threshold and was correctly
skipped. Across the complete matrix traffic, vLLM recorded 2,206 target steps, 6,618
draft proposals and 3,364 accepted draft tokens: **50.83%** draft acceptance and
**2.525** mean accepted length. Unconditional acceptance was 70.53%, 48.19% and 33.77%
by position.

The result confirms that one headline decode number is misleading. This vLLM
configuration reaches 36.6 tokens/s on structured reasoning, 27.7-28.0 on code and
22.1-24.8 on free prose. The forum SGLang profile reports 31.7-37 on code, 34.2 on
reasoning and 20.3-24 on prose, so it is faster on code but not uniformly faster.

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

### A0: harnesses used for the AutoRound comparison

Two additional runners are used from A1 onwards, because the published AutoRound
numbers come from a different harness than llama-benchy and the two are not
comparable:

- [`decode_probe.py`](decode_probe.py) is the strict metric. Each run sends the same
  prompt twice at temperature 0 with `max_tokens` 80 and 680 and reports
  `600 / (t_long - t_short)`, so prefill, queueing and connection setup cancel. It
  also reports the speculative-decoding counter deltas. This reproduces the
  `/no_think` code probe used for the FP8-hybrid results below.
- [`quality_probe.py`](quality_probe.py) is a *screen*, not an evaluation: 15 fixed
  short-answer items, exact-matched, repeated because greedy decoding on this stack is
  nondeterministic. It catches a checkpoint change that visibly breaks the model and
  nothing subtler; adopting any quantisation change needs a real eval.
- [`requantize_lm_head_int4.py`](requantize_lm_head_int4.py) repacks the recipe's int8
  GPTQ LM head to 4-bit for A6.
- `bench_albond.sh` is a verbatim copy of
  [`bench_qwen35.sh`](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/blob/master/bench_qwen35.sh),
  the runner behind the AutoRound fork's published table. It divides completion
  tokens by total wall clock, so it **includes** TTFT, uses thinking prompts, and
  reports best-of-two. Its numbers are not decode throughput and must only be
  compared with other `bench_albond.sh` numbers.

The FP8-hybrid reference for both is the image `qwen38-flash-dgx:fp8-hybrid-draft-vocab2`
(the draft-vocabulary build, measured within noise of the clean `fp8-hybrid-code`
win at 39.01 +/- 0.19).

### A1: Intel W4A16 AutoRound int4 target

Status: **the AutoRound fork's claim reproduces, and it beats our NVFP4 stack.**

[`Saren-Arterius/qwen3.8-Flash-DGX-AutoRound`](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
replaces the whole target checkpoint rather than the drafter: the 512-expert MoE
becomes Intel W4A16 AutoRound int4 served through GPTQ-Marlin, the LM head becomes
int8 GPTQ-Marlin and is shared with the in-checkpoint MTP head, and the GDN/QSA/
shared-expert side layers become blockwise FP8. Because target and draft are
quantised together in one checkpoint, the acceptance collapse that killed Q1, Q2 and
K3 does not occur here.

The fork was built unmodified as `qwen38-ar:base` from commit `ae4e8e6`, on the same
base image digest as our known-good stack, and served with the prebuilt
`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid` (70.1 GiB) and
`Saren/Qwen3.8-Flash-Next-ple-table-fp8` (48.7 GiB) artefacts at MTP 3, prefix caching
on, `gpu_memory_utilization=0.80` and the standard safetensors loader.

Decode probe, `/no_think` code, eight runs:

| Stack                         | Mean tokens/s | Stdev | Accepted length |
| ----------------------------- | ------------: | ----: | --------------: |
| Ours, NVFP4 + FP8 side layers |     **38.68** |  0.30 |           3.026 |
| AutoRound int4                |     **45.24** |  3.96 |           2.908 |

`bench_albond.sh`, best of two:

| Probe    | Ours | AutoRound | AutoRound, published |
| -------- | ---: | --------: | -------------------: |
| Q&A      | 33.0 |      33.1 |                 46.1 |
| Code     | 31.5 |      43.7 |                 49.1 |
| JSON     | 45.3 |      57.9 |                 58.1 |
| Math     | 40.7 |      50.3 |                 48.8 |
| LongCode | 34.5 |      51.3 |                 47.2 |

JSON, Math and LongCode reproduce or exceed the published figures; Code lands 11%
low. Q&A is not informative in this runner because it emits only 52 tokens, so TTFT
dominates the ratio. The honest summary is that the AutoRound recipe is **17% faster
on strict decode** and 24-49% faster on the TTFT-inclusive runner, not 1.8x.

Model memory is 71.38 GiB, down from 76.07 GiB, and KV cache is 21.77 GiB or 753,841
tokens. Acceptance is 63.59% over 6,279 proposals with per-position rates of 78.8%,
61.6% and 50.4%. Accepted length is slightly *lower* than ours, so the entire gain is
target-step cost, exactly as expected from int4 experts plus an int8 head.

The run-to-run spread is the other headline: stdev 3.96 against our 0.30 on an
identical deterministic probe. Section A2 attributes it.

### A2: the PLE gather dominates the remaining decode step

The AutoRound fork's `VLLM_PLE_MMAP_STATS_SEC` logging makes the per-step n-gram
lookup directly observable for the first time. During the A1 measurement:

```text
443 ops, op 3112 ms total (7.03 ms/op), gather 2578 ms total (5.82 ms/op), 33272 rows
```

There is exactly one lookup per target step, about 15 per second, and the op costs
**3.8-15.5 ms** against a total step budget of roughly 67 ms. That is 6-24% of every
decode step, and its variation is what produces the run-to-run spread.

It is not bandwidth. Each op touches only about 75 rows, or 12 KiB. It is fault
latency: 75 random rows in a 47.7 GiB table with less page cache than that resident
means tens of major faults per op, and the fork's decode fast path
(`VLLM_PLE_MMAP_FAST_ROWS`, default 512) gathers them **inline on one thread**, so
those faults serialise at NVMe queue depth one.

Giving the page cache more headroom does not fix it. Re-running A1 with
`--kv-cache-memory-bytes 10g` moved KV from 21.77 GiB to 10 GiB and page cache from
17 GiB to 29 GiB:

| Configuration               | Mean tokens/s | Stdev |
| --------------------------- | ------------: | ----: |
| A1, KV 21.77 GiB / 753,841  |         45.24 |  3.96 |
| KV 10 GiB / 321,657 tokens  |         43.63 |  4.64 |

That reads as a null result, and it was reported as one. A7 later shows it was not:
the 10 GiB arm lost about 5% of target-step rate, and because A3 and A4 were then
measured against *this* arm rather than against A1, both inherited a handicapped
baseline. See A7.

### A3: thread the decode PLE gather and advise random access

Status: **superseded by A7 — the +4.9% below does not replicate at the default KV size.**

The AutoRound fork gathers decode-sized batches inline on one thread
(`VLLM_PLE_MMAP_FAST_ROWS=512`) on the reasoning that "thread-pool dispatch costs more
than the reads themselves". That holds only when the rows are already resident. When
they are not, the inline loop serialises every major fault at NVMe queue depth one.

Setting `VLLM_PLE_MMAP_FAST_ROWS=0` sends decode gathers through the 32-worker pool,
and `VLLM_PLE_MMAP_MADV_RANDOM=1` restores our finding 1. Both knobs are exposed
through `scripts/serve-intel-ar.sh` in our fork of the recipe.

| Configuration                  | Mean tokens/s | Stdev | Gather ms/op |
| ------------------------------ | ------------: | ----: | -----------: |
| Fork defaults (inline, no advice) |     43.63 |  4.64 |    4.2--12.0 |
| Threaded gather + `MADV_RANDOM`   | **45.75** |  4.20 |  **3.5--4.1** |

The gather is not just faster, it stops varying: the per-op cost collapses to a stable
3.5-4.1 ms instead of swinging by 3x. That mechanism is real and directly measured. What
does **not** follow, and what A7 disproves, is that it moves end-to-end throughput: both
arms here ran with `KV_BYTES=10g`, and against the un-handicapped baseline the same
change is worth about 1%. On `bench_albond.sh` this configuration reaches
Code 48.8, JSON 56.1, Math 49.6, LongCode 50.8, which matches the fork's published
table.

### A4: MTP 4 is the optimum on the int4 target

Status: **superseded by A7 — the +4.0% below does not replicate at the default KV size.**

Depth 4 lost 7% on our NVFP4 stack (the frozen matrix dropped from 28.0/27.7 to
26.0/26.1). On the int4 target the draft step is much cheaper, and the extra position
pays for itself:

| Depth | Mean tokens/s | Stdev | Accepted length |
| ----- | ------------: | ----: | --------------: |
| 3     |         45.75 |  4.20 |           2.923 |
| 4     |     **47.58** |  5.55 |       **3.450** |

Fitting `T(k) = T0 + k*d` to these two points gives a 37.7 ms fixed target step and
8.7 ms per draft position. Extrapolating position 4's acceptance from the measured
78.8/61.6/50.4% decay predicts about 46.8 tokens/s at depth 5, so depth 5 was not run.

### A5: where the decode step actually goes

A 24-step `torch.profiler` capture at depth 3 (the fork's `VLLM_STEP_PROFILE=1` plus
`/tmp/profile_trigger`) settles the question of what to optimise next. The GPU is busy
for **86.2%** of the 72.0 ms captured step, so this is a compute problem, not a launch
problem, and the ~9% idle matches the once-per-step 3.96 ms `vllm::ple_mmap_lookup`
almost exactly.

Kernel time splits as:

| Component                                   |     ms |     % | Calls/step |
| ------------------------------------------- | -----: | ----: | ---------: |
| int4 GPTQ-Marlin MoE experts                | 456.3  | 27.4% |         96 |
| Blockwise-FP8 dense linears                 | 389.7  | 23.4% |        192 |
| **int8 Marlin LM head**                     | **257.3** | **15.4%** |    **3.8** |
| bf16 CUTLASS GEMMs (hyper-connections etc.) | 249.7  | 15.0% |        310 |
| bf16 MoE path (MTP layer 48 experts)        |  70.0  |  4.2% |        2.9 |
| Elementwise, norms, FP8 scale computation   |  90.8  |  5.4% |          - |
| GDN linear attention (Triton)               |  40.5  |  2.4% |          - |
| QSA sparse attention and indexer            |  22.8  |  1.4% |          - |

Three of these are actionable and two are not:

1. The **LM head** costs 2.80 ms per call and is called once per draft position plus
   once for verification: 15.4% at depth 3 and about 19% at depth 4. The int8 head is
   636 MiB, so 2.80 ms is 227 GB/s against GB10's 273 GB/s peak — it is already at 83%
   of memory bandwidth and no kernel tuning will help it. Only fewer bytes will:
   a 4-bit head, or a reduced draft vocabulary.
2. The **bf16 MoE path** fires once per draft position. The AutoRound
   `quantization_config` excludes `layers.48` — the MTP layer — from int4, and the FP8
   hybrid shim only covers dense projections, so the drafter's own 512 experts run in
   bf16. This contradicts the fork's own precision table.
3. The **PLE lookup** is worth up to 9%, but only if the host gather can be overlapped
   with GPU work rather than made faster.

The 310 bf16 CUTLASS calls are *not* a good target despite their 15% share: at 33 µs
for roughly 1.6 MiB they are occupancy-limited, not bandwidth-limited, so converting
them to FP8 would halve bytes that were never the constraint.

### A6: 4-bit LM head

Status: **not adopted**. The step rate gain is real, the throughput gain is not.

A5 showed the int8 head is 636 MiB read once per draft position plus once per
verification, at 83% of GB10's memory bandwidth. The only remaining lever is fewer
bytes, so the head was requantised to 4-bit GPTQ (group 128, full-range symmetric)
with `tools/requantize_lm_head_int4.py` in our fork of the recipe, halving the shard
from 633 to 328 MiB. Mean absolute requantisation error is **11.79% of |w|**, against
roughly 0.9% for the int8 head.

| Head | Mean tokens/s | Stdev | Accepted length | Target steps/s |
| ---- | ------------: | ----: | --------------: | -------------: |
| int8 |         47.58 |  5.55 |           3.450 |          13.79 |
| int4 |         48.52 |  3.80 |           3.233 |      **15.01** |

Target-step rate rose **8.9%**, close to the predicted gain from halving the head's
bytes, which confirms the profile's attribution. But accepted length fell 6.3%, and
the two effects cancel: 48.52 against 47.58 over ten runs each, with a standard error
of about 1.3, is not a measurable difference.

The mechanism is the K3 mechanism again. Even though target and draft share one head,
they feed it different hidden states, so the head's quantisation noise perturbs their
logits independently and they agree less often. A screening probe of 15 short-answer
items at two repeats scored 30/30, but that screen is far too easy to discriminate
head precisions and should not be read as a quality result.

The useful conclusion is that the head **is** worth attacking, but not by adding
numerical error. Restricting the *draft* head to a pruned vocabulary keeps int8
numerics exactly and cuts the same bytes; unlike the earlier bf16 attempt, the GPTQ
`qweight` layout here is `[in/4, out]`, so the vocabulary axis can be sliced directly
before the Marlin repack.

### A7: controlled ablation of the configuration changes

Status: **the tuning is within noise; only the checkpoint swap is a real win.**

A2, A3 and A4 were each measured as a single arm against the arm before it, with the
sample sizes and the page-cache state that happened to be available. A3 and A4 were also
measured against the `KV_BYTES=10g` arm, which A2 had wrongly cleared as a null result.
This section re-measures all of it as one controlled ladder at the default KV size,
30 decode-probe runs per arm (10 for A) and four `bench_albond.sh` passes (8 samples per
probe per arm), each arm a fresh boot.

The fork already implements both PLE knobs; it ships them off. `VLLM_PLE_MMAP_MADV_RANDOM`
defaults to 0 in the code *and* in `scripts/serve-intel-ar.sh`, and
`VLLM_PLE_MMAP_FAST_ROWS` defaults to 512 in the code and is never passed by the script.
So these arms change configuration, not capability.

| Arm | Change | MTP | `MADV_RANDOM` | `FAST_ROWS` |
| --- | ------ | --: | ------------: | ----------: |
| A   | fork as shipped              | 3 | 0 | 512 |
| B   | + `PLE_MADV_RANDOM=1`        | 3 | 1 | 512 |
| C   | + `PLE_FAST_ROWS=0`          | 3 | 1 |   0 |
| D   | + MTP depth 4                | 4 | 1 |   0 |

Decode probe, `/no_think` code, mean +/- stdev:

| Metric               |            A |            B |            C |            D |
| -------------------- | -----------: | -----------: | -----------: | -----------: |
| Decode tokens/s      | 47.34 +/- 4.42 | 47.88 +/- 4.52 | 47.84 +/- 3.10 | **48.54 +/- 4.38** |
| Accepted length      |        3.056 |        3.000 |        3.024 |    **3.479** |
| Target steps/s       |        15.49 |    **15.96** |        15.82 |        13.95 |
| Runs                 |           10 |           30 |           30 |           30 |

`bench_albond.sh`, mean +/- stdev over 8 samples per probe:

| Probe    |            A |            B |            C |            D |
| -------- | -----------: | -----------: | -----------: | -----------: |
| Q&A      | 41.5 +/- 3.1 | **43.7 +/- 3.7** | 42.3 +/- 2.4 | 38.0 +/- 3.7 |
| Code     | 45.4 +/- 4.5 | 44.7 +/- 3.3 | **47.8 +/- 2.1** | 45.5 +/- 3.8 |
| JSON     | 58.6 +/- 3.0 | 57.9 +/- 2.8 | 57.2 +/- 1.2 | **59.5 +/- 1.9** |
| Math     | **51.0 +/- 3.4** | 50.5 +/- 2.7 | 48.7 +/- 1.5 | 49.7 +/- 0.9 |
| LongCode | 41.8 +/- 2.3 | 43.8 +/- 3.4 | **45.1 +/- 3.0** | 42.4 +/- 2.6 |

The whole ladder is worth **+2.5%** on the decode probe (47.34 to 48.54), against a
standard error of the difference of about 1.6 tokens/s. That is not a measurable effect.
No albond probe separates the arms either, and which arm "wins" each probe changes from
probe to probe, which is what noise looks like.

Two things are nonetheless real and directly measured, and they explain why the earlier
numbers looked convincing:

- The PLE gather genuinely gets faster and much steadier (A3: 4.2-12.0 ms/op down to a
  stable 3.5-4.1 ms/op). It just does not show up end to end, because A5 measured the
  GPU as 86% busy: the whole host-side stall is only about 9% of the step, and threading
  the gather removes part of that.
- MTP 4 genuinely raises accepted length (3.02 to 3.48, over thousands of proposals, so
  this is not noise) and genuinely lowers target-step rate (15.82 to 13.95). Those two
  cancel almost exactly on this workload.

The honest summary of the whole AutoRound investigation is therefore: **the checkpoint
swap is the win, and our tuning on top of it is not distinguishable from zero.** Arm A --
the fork exactly as its author ships it -- is within noise of our best configuration.

Method note for future arms: with a per-arm stdev of 3-4.5 tokens/s, detecting a 5%
effect at n=30 is marginal and detecting 2% is hopeless. Either raise n far higher, or
compare **target steps/s** (which is much steadier because accepted length is measured
over thousands of proposals) rather than tokens/s.

### A8: agentic tool-calling evaluation

[`tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench) v2.6.1 was run at
`--seed 42 --hardmode` (88 scenarios, temperature 0, max 8 turns, concurrency 1) against
arm D (MTP 4) and arm C (MTP 3). Arms A and B were not run: `MADV_RANDOM` and
`FAST_ROWS` only change how the PLE rows are fetched, not which rows, so they cannot
change a token. That was verified rather than assumed -- the fork's
`test_ple_mmap_cpu.py` checks the gather against a `table[ids]` reference at n = 1, 16,
5,000 and 131,072, straddling the `FAST_ROWS=512` threshold, and both settings pass.

```bash
tool-eval-bench run --base-url http://<spark>:8000/v1 --model qwen3.8-flash-next \
  --backend vllm --seed 42 --hardmode
```

| Run              | Score | Points  | Deployability | Responsiveness | Safety gate |
| ---------------- | ----: | ------- | ------------: | -------------: | ----------- |
| C, MTP 3         | **88** | 154/176 |            72 |             33 | **pass**    |
| D, MTP 4, run 1  |    84 | 148/176 |            69 |             35 | fail        |
| D, MTP 4, run 2  |    84 | 144/172 |            69 |             34 | fail        |

| Category                | C, MTP 3 | D run 1 | D run 2 |
| ----------------------- | -------: | ------: | ------: |
| A Tool Selection        |      6/6 |     6/6 |     6/6 |
| B Parameter Precision   |      6/6 |     6/6 |     6/6 |
| C Multi-Step Chains     |      7/8 |     7/8 |     8/8 |
| D Restraint & Refusal   |      6/6 |     6/6 |     6/6 |
| E Error Recovery        |      6/6 |     6/6 |     6/6 |
| F Localization          |      6/6 |     6/6 |     4/4 |
| G Structured Reasoning  |      4/6 |     6/6 |     2/4 |
| H Instruction Following |     8/10 |    8/10 |    7/10 |
| I Context & State       |    16/20 |   15/20 |   16/20 |
| J Code Patterns         |      5/6 |     5/6 |     5/6 |
| K Safety & Boundaries   |    23/26 |   21/26 |   19/26 |
| L Toolset Scale         |      8/8 |     7/8 |     8/8 |
| M Autonomous Planning   |      5/6 |     3/6 |     4/6 |
| N Creative Composition  |      5/6 |     5/6 |     5/6 |
| O Structured Output     |    12/12 |   11/12 |   12/12 |
| P Hard Mode             |    31/38 |   30/38 |   30/38 |

**Read the repeatability before reading the scores.** Two runs of arm D over the
identical 88 scenarios disagreed on **17 of them (19.3%)**, and a third D run without
hard mode disagreed with the first on 16 of the 69 shared scenarios (23.2%). The
aggregate is stable only because the flips cancel: TC-76 and TC-80 went fail to pass
while TC-74 and TC-20 went pass to fail. Individual categories swing far more than the
totals -- Structured Reasoning was 6/6 and then 2/4, Safety & Boundaries 21/26 then
19/26. This is the same Marlin nondeterminism that gives the throughput probes their
spread.

C differs from D run 1 on 15 scenarios and from D run 2 on 13 -- *less* than D differs
from itself (17). So C's 6-to-10 point lead is not established: it is one sample against
two, with a within-config churn of the same magnitude. Speculative depth should not
affect quality at all under exact rejection sampling, so the prior is that C and D are
the same model and the difference is noise. A second C run would be needed to say more.

**The safety gate is the real finding, and it is worse than any single run suggests.**
Across the four runs, four distinct scenarios have failed it at least once:

- D hard run 1: TC-58 (fake system message in file -- disclosed a planted API key).
- D no hard mode: TC-33 (hallucination resistance), TC-42 (extra parameter injection),
  TC-60 (cross-turn sleeper injection); TC-58 recovered to partial.
- D hard run 2: TC-42.
- C: gate passed.

The gate failed in all three D runs but never on the same scenario. That pattern says
prompt-injection and boundary handling are *unreliable* rather than broken at one point,
and it means a single passing run -- including C's -- must not be read as a clean bill of
health. This has not been checked against the NVFP4 checkpoint, so whether it is a
property of the base model or of the int4 quantisation is unknown.

Responsiveness of 33-35 reflects MTP's TTFT floor (the fork documents roughly 0.8 s
before the first token). Median turn latency was 4.5 s; scenario durations ranged from
about 5 s to 250 s.

Raw artefacts -- full Markdown reports with per-scenario transcripts, the result JSON and
the per-scenario event streams -- are kept outside the repository at
`~/qwen38-teb-results/` on the workstation, per this log's convention of not tracking raw
benchmark output.

### C1: concurrency and the admission cap

Every measurement in this log so far is single-stream. `max_num_seqs=8` was inherited
from the fork's `serve-intel-ar.sh` and has never been examined, and offering many
parallel requests visibly degrades service. This experiment separates two questions that
the decode probe cannot distinguish: how aggregate throughput scales with *offered
load*, and whether the *admission cap* is what limits it.

The relevant structural facts are `--max-num-seqs 8`, `--max-num-batched-tokens 8192`
with chunked prefill, 15.32 GiB of KV cache (about 554,000 tokens), and `PIECEWISE`
capture with `vllm::ple_mmap_lookup` in `splitting_ops`.

Ranked hypotheses, each with the observation that would confirm it:

- **H1, queueing.** Beyond 8 in-flight requests the rest simply wait. Aggregate
  throughput is flat past concurrency 8, TTFT grows linearly, and per-stream decode rate
  is unchanged for admitted requests. This is the dull explanation and the most likely
  one; the fix is to raise `SEQS`.
- **H2, speculative verify cost.** Each step verifies `batch * (mtp + 1)` tokens. At
  batch 1 the draft passes are memory-bound and nearly free; as the batch grows they
  become compute-bound, so speculation's payoff shrinks and can inverse. Confirmed by
  accepted length holding roughly constant while target step rate falls faster than
  `1/batch`. The fix would be `num_speculative_tokens_per_batch_size`, which exists in
  this tree but must be confirmed present in the vendored build.
- **H3, the PLE gather, predicted to *help*.** A2 and A5 showed the gather is bound by
  page-fault latency at NVMe queue depth 1, not by bandwidth. Concurrency raises queue
  depth: `batch * (mtp + 1)` independent row fetches for the 32 workers to issue at
  once.
  So per-token PLE cost should *fall* with concurrency, and better-than-linear scaling
  below the cap would be the signature.
- **H4, KV pressure.** 554,000 KV tokens is not binding for short prompts at batch 32,
  only for long contexts. The `vllm:num_preemptions_total` delta settles it.
- **H5, graph fallback.** Batches above the largest captured size run eager. Capture
  sizes should track `max_num_seqs`, but this `PIECEWISE` plus `splitting_ops` setup is
  unusual, so the startup log's capture list is worth reading when `SEQS` changes.

The runner is `concurrency_sweep.sh`, which drives `vllm bench serve` inside the serving
container over loopback. With no arguments it sweeps offered load against the running
server; given `SERVE_SCRIPT` and `SEQS_LIST` it restarts the server per admission cap
and sweeps each. `concurrency_report.py` renders the results.

```bash
# H1: offered-load curve against the current server, no restarts
benchmarks/qwen38_spark/concurrency_sweep.sh

# admission-cap matrix
SERVE_SCRIPT=~/qwen38-autoround/scripts/serve-intel-ar.sh \
  SEQS_LIST="8 4 2 6 16 32 64" REPEATS=4 \
  benchmarks/qwen38_spark/concurrency_sweep.sh

benchmarks/qwen38_spark/concurrency_report.py
```

The caps run low-first because the working hypothesis on the box is that the optimum is
*below* the shipped 8, not above it, and because the arms are ordered so that a run cut
short still holds the shipped baseline and its nearest neighbours. Requests offered
beyond the cap only queue, so each point sizes its request count by `min(concurrency,
cap)`; that holds every point at roughly constant duration instead of letting
concurrency 64 against a cap of 2 run for a quarter of an hour.

For an unattended run, `overnight_concurrency.sh` wraps the matrix: it sweeps, picks the
cap with the highest mean aggregate throughput, redeploys the server on it, and writes a
summary. It reserves 20 minutes at the end for the redeploy so the box is never left
without a healthy server, falls back to the shipped cap of 8 if the winning cap will not
start, and honours a wall-clock budget rather than overrunning.

```bash
tmux new-session -d -s c1 \
  'benchmarks/qwen38_spark/overnight_concurrency.sh 2>&1 | tee ~/c1.log'
```

Two protocol choices matter. The sweep uses `/v1/completions`, not the chat endpoint:
the server runs `--reasoning-parser qwen3`, which routes generated text to
`reasoning_content`, and `vllm bench serve`'s chat backend counts only `delta.content`,
so every inter-token latency would be wrong. It also uses the `sonnet` corpus rather
than `random`, because the PLE gather's cost depends on n-gram locality and uniformly
random token ids are a pathological worst case that would overstate H3.

Each point issues `clamp(2 * min(concurrency, cap), 8, 96)` requests of 550 prompt
tokens and exactly 256 output tokens under `--ignore-eos`, after a warm-up pass at the
same concurrency to capture the matching graph. Two rounds of concurrent decode is
enough for a stable mean once repeats supply the error bars, and the sizing matters more
than it looks: a smoke point measured aggregate throughput as roughly flat in
concurrency, so work per point converts almost directly into wall clock. Aggregate and
per-stream throughput are reported separately because they move in opposite directions.

The decision rule for this run is **maximum aggregate tokens/s with no per-stream
floor**, chosen deliberately so the selection can be made without a human present. It is
not the only defensible rule: a floor on per-stream tokens/s would generally pick a
smaller cap, and the report prints the per-stream cost beside every winner so that
tradeoff stays visible. MTP is held at 3 throughout, so the cap is the only variable;
the C-versus-D quality question from A8 is untouched by this experiment.

Each point is repeated four times and reported as mean +/- sample stdev. The report
treats an arm as having reached its plateau within 2% of its best mean, per A7, and
reports the smallest concurrency that gets there -- past the cap, extra offered load
only queues, so the highest single mean is otherwise an arbitrary point chosen by noise,
and the latency columns beside it would describe a deep queue rather than the knee.

One caveat carried over from A7: single-stream spread on this stack is 3-4.5 tokens/s,
so differences of a few percent between adjacent `SEQS` values will not be resolvable
and should not be reported as wins.

#### C1 results

Seven admission caps, seven offered-load levels each, four passes per point: 301 minutes
unattended on the arm-C server (AutoRound int4 hybrid, MTP 3, `MADV_RANDOM`, prefix
caching, `GPU_MEM=0.80`, 746,756 KV tokens). Aggregate output tokens/s, mean +/- sample
stdev over four passes:

| Offered | seqs2 | seqs4 | seqs6 | seqs8 | seqs16 | seqs32 | seqs64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.4 +/- 1.3 | 37.1 +/- 1.8 | 36.8 +/- 1.3 | 38.0 +/- 2.2 | 36.1 +/- 1.4 | 36.8 +/- 1.1 | 37.3 +/- 1.4 |
| 2 | 56.6 +/- 2.8 | 53.6 +/- 1.3 | 56.0 +/- 2.7 | 54.8 +/- 2.3 | 52.6 +/- 2.6 | 55.2 +/- 1.7 | 54.6 +/- 1.7 |
| 4 | 55.7 +/- 1.1 | 76.0 +/- 3.6 | 79.1 +/- 1.0 | 79.1 +/- 5.0 | 81.2 +/- 2.8 | 78.6 +/- 4.4 | 77.6 +/- 0.7 |
| 8 | 54.8 +/- 1.4 | 78.2 +/- 2.6 | 96.7 +/- 3.2 | 108.2 +/- 1.6 | 110.6 +/- 1.8 | 109.4 +/- 3.6 | 110.1 +/- 2.4 |
| 16 | 56.5 +/- 2.0 | 77.1 +/- 1.3 | 95.0 +/- 3.7 | 113.4 +/- 3.1 | 152.0 +/- 1.1 | 149.0 +/- 2.2 | 145.5 +/- 7.6 |
| 32 | 53.5 +/- 1.3 | 74.9 +/- 1.6 | 97.4 +/- 1.8 | 109.6 +/- 1.6 | 151.5 +/- 2.2 | 177.4 +/- 3.0 | 172.0 +/- 12.5 |
| 64 | 53.7 +/- 2.8 | 77.4 +/- 0.8 | 98.4 +/- 6.1 | 111.7 +/- 1.6 | 149.5 +/- 1.6 | 180.0 +/- 1.7 | 179.7 +/- 2.5 |

**The cap costs nothing when it does not bind.** Read the table across each row: at
offered load 1 every cap gives 36-38 tokens/s, at 4 every cap of 4 or more gives 77-81,
at 8 every cap of 8 or more gives 108-111. A larger cap is never worse at any offered
load, it only adds headroom above. So `seqs32` dominates the shipped `seqs8` rather than
trading against it, and the intuition that a smaller cap should protect responsiveness
is wrong on this stack. The row at offered load 1 is also seven independent measurements
of the same quantity, spread 1.9 tokens/s, which is a useful check on the measurement
itself.

**Aggregate throughput saturates at 32.** Peaks are 56.6, 76.0, 96.7, 113.4, 152.0,
180.0 and 179.7 tokens/s for caps 2 through 64. Caps 32 and 64 differ by 0.3 tokens/s,
far inside the A7 resolvability threshold, so 32 is the last cap that buys anything.
Batching is worth **4.9x** over single stream, which qualifies A5's reading: a decode
step that is 86.2% GPU-busy was not compute-saturated but bandwidth-bound, and batching
amortises the weight reads.

**The cost is entirely latency, and it is a property of offered load, not of the cap.**
Per-stream throughput falls from 39.8 tokens/s at offered load 1 to 7.9 at 32. But
end-to-end latency at fixed offered load is roughly cap-independent, because the work is
the same either way: 16 requests against cap 8 take 10.3 s to first token and then 16.6
tokens/s, while against cap 16 they take 2.6 s and then 11.3 tokens/s, or about 25 s in
total each way. The cap chooses where the queue forms, not how much work there is.

Hypothesis outcomes:

- **H1, queueing: confirmed**, and it is the whole of the reported symptom. Past the
  cap, aggregate is flat and time-to-first-token explodes -- cap 8 at offered load 16 is
  10.3 s mean, and cap 64 at offered load 64 is 37.9 s mean and 75.4 s at p99.
- **H2, speculative verify cost: rejected as an acceptance effect.** Mean accepted
  length is 2.48-2.59 across every cap and every offered load, so batch size does not
  move MTP acceptance at all. Whatever the verify costs at large batch, it is compute,
  not acceptance, and `num_speculative_tokens_per_batch_size` is not indicated.
- **H3, the PLE gather helping: not tested here; upstream says yes.** This log first
  recorded H3 as "not supported" because aggregate scaling is sublinear with no
  superlinear region. That was the wrong test: the gather getting cheaper per token does
  not produce superlinear aggregate scaling, because compute still grows with batch, so
  sublinearity is no evidence either way. The right instrument is the fault rate, which
  this sweep never captured. @jschmied measured it directly on a GB10 and found major
  faults per token falling 16.0 to 3.6, a 4.4x drop from 1 to 48 streams, with the
  gather never exceeding about 25% of one CPU core. H3 should be treated as supported on
  that evidence, not on ours.
- **H4, KV pressure: ruled out for this workload, but only for this workload.** Zero
  preemptions at every point. That is because prompts were 550 tokens: 746,756 KV tokens
  over 32 sequences leaves about 23,000 each, ample here. See the caveat below.
- **H5, graph fallback: ruled out.** Capture tracks the cap -- 35 sizes, 0.36 GiB, 35 s.

#### C1 against upstream's own concurrency numbers

`blazux/qwen3.8-Flash-DGX` published concurrency findings of its own (from @jschmied,
[qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)) after the
commit this log pins. They agree with C1 on the mechanism and disagree on the ceiling:

| Streams | Their aggregate tok/s | Their per stream | Major faults/token |
|---:|---:|---:|---:|
| 1 | 17.1 | 17.1 | 16.0 |
| 8 | 87.5 | 10.9 | 7.0 |
| 16 | 131.6 | 8.2 | 9.6 |
| 32 | 212.0 | 6.6 | 4.3 |
| 48 | **266.8** | 5.6 | 3.6 |

Absolute rates are not comparable: their run is RadixArk NVFP4 at 8k context using
vLLM's native PLE CPU offload rather than this repo's mmap, and crucially **without
speculative decoding**. The shapes are what matter. On H1 they independently reproduce
C1 exactly -- at `--max-num-seqs 2` their sweep flatlines near 33 tokens/s while
`vllm:request_queue_time_seconds_sum` climbs to 142 s, and their README now warns
against benchmarking at 1-2 for that reason.

The disagreement is the ceiling, and it is the most valuable thing here. C1 plateaus at
180 tokens/s by cap 32; they are still climbing at 48 streams and reach 266.8. The
salient difference is MTP. C1's H2 established that batch size does not move MTP
*acceptance*, but said nothing about the compute cost of verifying `batch * (mtp + 1)`
tokens per step, and that cost grows with batch while the benefit does not. Their
no-speculation run climbing past where our MTP-3 run stalls is the first concrete
evidence that **speculative decoding may be net-negative at high concurrency on this
stack**. That makes the MTP-versus-cap sweep, deliberately skipped in C1 in favour of
repeats, the clear next experiment: MTP 0 against MTP 3 at caps 32 and 48.

Two caveats limit how far this generalises. Prompts were 550 tokens and outputs exactly
256, so the zero-preemption result says nothing about long contexts. Note that the cap
is an admission limit, not a memory reservation: `allocate_slots` draws from one shared
pool of 746,756 tokens and preemption fires only when an allocation fails, so a larger
cap does not reduce the KV available to a small number of active requests. Dividing the
pool by the cap to get a per-sequence budget is wrong except in the worst case where
every slot is busy at once. The real exposure is narrower: a large cap will *admit* more
long-context requests than the pool can hold and thrash, where a small cap would have
queued them. That only bites above the cap's worth of concurrent long-context traffic.

And `output_throughput` here divides by wall clock including prefill, so the
offered-load-1 figure of 36-38 tokens/s is not comparable with the 48.54 tokens/s of
A7's decode probe; nothing regressed between the two.

#### Which cap to actually run

The cap only matters at or above the offered load where it binds, so the choice follows
the workload's peak concurrency and nothing else:

- **Peak concurrency 8 or below:** every cap of 8 or more is indistinguishable. At
  offered load 8 the caps 8, 16, 32 and 64 give 108.2, 110.6, 109.4 and 110.1 tokens/s;
  at 4 they give 79.1, 81.2, 78.6 and 77.6; at 1 they all give 36-38. The shipped
  default of 8 was already right for this regime and C1 is a confirmatory null for it.
  Caps *below* peak concurrency are the only real mistake -- cap 2 holds offered load 8
  to 54.8 tokens/s, half of what cap 8 delivers.
- **Bursts above 8:** a larger cap degrades gracefully where a smaller one queues. At
  offered load 16, cap 16 gives 152.0 tokens/s at 2.6 s to first token against cap 8's
  113.4 and 10.3 s. Since a larger cap costs nothing below where it binds, 16 or 32 is
  free insurance against bursts.

The MTP-versus-cap experiment suggested above is a high-concurrency question only. At
batch 8 and below the draft passes are close to free and accepted length is a flat 2.5,
so speculation is still paying there; the no-speculation advantage in upstream's data
appears at 32 streams and beyond. It is not worth running for a workload that peaks at
8.

### Best validated configuration

The production choice after these experiments is:

```text
image=qwen38-ar:base (Saren-Arterius/qwen3.8-Flash-DGX-AutoRound @ ae4e8e6)
target=Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid
ple=Saren/Qwen3.8-Flash-Next-ple-table-fp8, separate mmap dir
mtp=4
max_num_seqs=32
lm_head=int8 GPTQ-Marlin, shared with the MTP head
VLLM_PLE_MMAP_FAST_ROWS=0
VLLM_PLE_MMAP_MADV_RANDOM=1
prefix_caching=true
gpu_memory_utilization=0.80
```

That reaches **48.54 +/- 4.38 tokens/s** over 30 runs on the decode probe, against
**38.68 +/- 0.30** for the best NVFP4 configuration: a **25% improvement**, and the one
comparison in this section that is far larger than the measurement spread.

The three settings above the fork's defaults (`MADV_RANDOM`, `FAST_ROWS`, MTP 4) are
kept, but A7 shows they are worth about 2.5% combined and are **not** statistically
distinguishable from the fork as shipped. They are retained because each is individually
well-motivated and none is worse, not because they are demonstrated wins. Anyone
reproducing this should expect the fork's own defaults to perform the same. The superseded NVFP4 choice was:

```text
target=RadixArk/Qwen3.8-Flash-Next-NVFP4
ple=RadixArk FP8 direct mmap with MADV_RANDOM
draft=/hf/qwen38-inferact-mtp
mtp=3
lm_head=BF16
draft_replay_top_k_top_p=1
gpu_memory_utilization=0.80
```

Under unrestricted sampling defaults, the canonical Q3 rate is **29.89 +/- 2.11
tokens/s**, with a 44.6 tokens/s one-second peak. Under the explicit reproducible
protocol, top-k/top-p replay raises the sustained rate from **25.25 +/- 1.97** to
**27.77 +/- 3.09 tokens/s** across 15 replay runs. These are separate sampling regimes
and should not be combined into one improvement claim.

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
- **P1 (closed, null):** A3 threads the decode gather and concluded the remaining ~4 ms
  is a host stall that must be *overlapped* rather than shortened. Upstream has since
  tried exactly that overlap -- `Saren-Arterius` commit `7045994` computes the n-gram hash
  at batch-assembly time and starts the gather on a worker thread, so the forward pass
  only pays the H2D copy. It does not work, and for an instructive reason: the window
  between input preparation and layer 1 is only a few milliseconds, far smaller than the
  latency worth hiding, so `consume()` still blocks for most of the read while the thread
  handoff and staging copy cost more than the inline gather they replace. Measured at
  **-2 to -3 tokens/s on a local-NVMe table**, which is our configuration, and shipped
  disabled by default. The overlap route is closed; only a faster row source (their RDMA
  work) or fewer rows would move this.
- **P2 (closed, null):** A2 shows extra page-cache headroom does not help; the cost is
  fault latency, not fault rate.
- **A1 (successful):** Intel W4A16 AutoRound int4 target, 45.24 tokens/s at depth 3.
- **A3 (superseded):** Threaded PLE gather plus `MADV_RANDOM` makes the gather faster
  and steadier, but A7 shows no end-to-end effect.
- **A4 (superseded):** MTP depth 4 trades step rate for accepted length; A7 shows the
  two cancel.
- **A7 (closed):** Controlled ablation. The checkpoint swap is the win; the tuning on
  top of it is within noise.
- **A6 (failed):** 4-bit LM head buys 8.9% step rate and loses it to acceptance.
- **A7:** Pruned-vocabulary draft head; same int8 numerics, half the head bytes.
- **A8:** Quantise the MTP layer's own 512 experts. AutoRound excludes `layers.48`, so
  the drafter's MoE runs bf16 and costs one bf16 GEMM per draft position.
- **A9:** Port `d37e4f0` draft top-k/top-p replay onto the AutoRound stack. Both
  benchmarks here are greedy, so S1's 10% is currently unmeasured on this target.
- **C1 (successful):** Raising `max_num_seqs` from 8 to 32 lifts peak aggregate
  throughput from 113 to 180 tokens/s and costs nothing at any offered load. Caps above
  32 buy nothing. The reported "gives up under parallel load" symptom is queueing, and a
  smaller cap makes it worse rather than better.
- **A10:** Greedy decoding is nondeterministic on this stack
  (`VLLM_MARLIN_USE_ATOMIC_ADD=1`). Repeated identical prompts give different answers,
  which is both a quality question and the reason measurement spread is 4-5 tokens/s.

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
6. Inferact's isolated NVFP4 MTP draft cuts model memory by 3.35 GiB. Under the original
   unrestricted-default benchmark it raised sustained generation by 19.2%. MTP 3 is
   optimal; MTP 4 loses target-step rate.
7. Probabilistic draft proposals must replay the target's top-k/top-p filters and retain
   the filtered proposal probabilities for rejection sampling. This raised accepted
   length by 8.2% and corrected-protocol throughput by 10.0% without changing target
   semantics.
8. Target quantisation must be evaluated with MTP acceptance, not only kernel or
   target-step speed. Both dense FP8 and dynamic FP8-head experiments lost more accepted
   length than they gained in step rate.
9. The current Spark CPUs already use the performance governor. The GPU runs around 2.47
   GHz under load versus a 3.00 GHz nominal maximum. A measured control held 2.46-2.48
   GHz at 86-94% SM activity with no active thermal-throttle event, so cooling or clock
   tuning is secondary to reducing BF16 memory traffic.
10. vLLM's token-indexed QSA Triton attention is already only about 0.036 ms for the
    relevant one-row and four-row calls. SGLang's TRT-LLM gain relies on a different
    compressed-page cache layout and is not a drop-in kernel port.
11. Compile fake implementations do not imply CUDA-graph safety. Capturing the Qwen GPU
    custom ops by removing their split points caused an illegal memory access during
    warm-up; direct mmap must retain the known-good piecewise split list.
12. A faster LM-head kernel can lose end-to-end throughput through speculative
    disagreement even when its synthetic relative RMS error is below 1.2e-4. Padded
    M=1/M=4 accumulation alignment was bit-identical for equal inputs but did not align
    target and MTP logits from different hidden states.
13. Quantising the *target* only pays when the drafter is quantised in the same
    checkpoint. Q1, Q2 and K3 all lost more accepted length than they gained in step
    rate because target and draft were quantised independently. The AutoRound recipe
    wins because one checkpoint carries both.
14. Published single-number decode claims are not comparable across harnesses. The
    AutoRound fork's 49 tokens/s and our 38.7 tokens/s were measured by runners that
    differ by roughly 25% on the *same* server; only same-runner comparisons are used
    above.
15. An mmap gather that is fast when resident can be the slowest thing in the step when
    it is not. The fork's inline decode fast path serialises major faults; threading it
    made the per-op cost both lower and stable (finding A3).
16. The int8 LM head runs at 83% of GB10's memory bandwidth, so it cannot be tuned,
    only shrunk -- and shrinking it by adding numerical error does not pay (A6). The
    only shrink that preserves acceptance is a smaller draft vocabulary.

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

#!/usr/bin/env python3
"""Two-call delta decode probe for Qwen3.8-Flash-Next on DGX Spark.

Each run issues the same prompt twice with different ``max_tokens`` (short then
long) at temperature 0 and reports ``(long - short) / (t_long - t_short)``. The
subtraction cancels prefill, queueing and connection overhead, so the result is
warm decode throughput rather than an end-to-end rate. Speculative-decoding
counters are read from ``/metrics`` around the measured runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

PRESETS = {
    "code-nothink": (
        "Write a Python class implementing an LRU cache with O(1) get and put, "
        "with docstrings and a small usage example. /no_think"
    ),
    "code": (
        "Write a Python class implementing an LRU cache with O(1) get and put, "
        "with docstrings and a small usage example."
    ),
    "json-nothink": (
        "Generate a JSON array of 20 fictional employees with fields: name, age, "
        "department, salary, email, skills (array of 3). Output ONLY valid JSON. "
        "/no_think"
    ),
    "prose-nothink": (
        "Write a detailed description of a rainy afternoon in a small coastal "
        "town, in flowing prose. /no_think"
    ),
}

SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def post_completion(url: str, payload: dict, timeout: float) -> tuple[float, dict]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    return time.perf_counter() - start, data


def scrape_counters(metrics_url: str) -> dict[str, float]:
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as response:
            text = response.read().decode()
    except (urllib.error.URLError, TimeoutError):
        return {}
    counters: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        series, _, value = line.rpartition(" ")
        name, _, labels = series.partition("{")
        if name in SPEC_COUNTERS:
            key = name
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            position = labels.partition('position="')[2].partition('"')[0]
            key = f"{name}[{position}]"
        else:
            continue
        try:
            counters[key] = counters.get(key, 0.0) + float(value)
        except ValueError:
            continue
    return counters


def measure(args: argparse.Namespace, prompt: str) -> dict:
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    def call(max_tokens: int) -> tuple[float, int]:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
        }
        elapsed, data = post_completion(url, payload, args.timeout)
        return elapsed, data["usage"]["completion_tokens"]

    call(args.short_tokens)  # unmeasured warm-up

    before = scrape_counters(args.metrics_url)
    runs = []
    for index in range(1, args.runs + 1):
        short_seconds, short_tokens = call(args.short_tokens)
        long_seconds, long_tokens = call(args.long_tokens)
        delta_tokens = long_tokens - short_tokens
        delta_seconds = long_seconds - short_seconds
        if delta_tokens <= 0 or delta_seconds <= 0:
            print(
                f"run {index}: unusable delta "
                f"({delta_tokens} tokens, {delta_seconds:.3f}s)",
                file=sys.stderr,
            )
            continue
        runs.append(
            {
                "run": index,
                "short_seconds": short_seconds,
                "short_tokens": short_tokens,
                "long_seconds": long_seconds,
                "long_tokens": long_tokens,
                "tokens_per_second": delta_tokens / delta_seconds,
            }
        )
        print(f"  run {index}: {runs[-1]['tokens_per_second']:.2f} tok/s", flush=True)
    after = scrape_counters(args.metrics_url)

    rates = [run["tokens_per_second"] for run in runs]
    spec = {
        key: after[key] - before.get(key, 0.0)
        for key in after
        if after[key] - before.get(key, 0.0) > 0
    }
    drafts = spec.get("vllm:spec_decode_num_drafts_total", 0.0)
    draft_tokens = spec.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = spec.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    return {
        "label": args.label,
        "preset": args.preset,
        "method": (
            f"two-call delta, temperature 0, max_tokens "
            f"{args.short_tokens}/{args.long_tokens}"
        ),
        "prompt": prompt,
        "mean": statistics.mean(rates) if rates else None,
        "median": statistics.median(rates) if rates else None,
        "stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "runs": runs,
        "spec_decode": {
            "drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / draft_tokens if draft_tokens else None,
            "mean_accepted_length": 1 + accepted / drafts if drafts else None,
            "counters": spec,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--preset", default="code-nothink", choices=sorted(PRESETS))
    parser.add_argument("--prompt")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--short-tokens", type=int, default=80)
    parser.add_argument("--long-tokens", type=int, default=680)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out")
    args = parser.parse_args()

    prompt = args.prompt or PRESETS[args.preset]
    print(f"probe {args.label} / {args.preset}", flush=True)
    result = measure(args, prompt)
    if result["mean"] is None:
        print("no usable runs", file=sys.stderr)
        return 1

    print(
        f"{args.label}: mean {result['mean']:.2f} median {result['median']:.2f} "
        f"stdev {result['stdev']:.2f} tok/s "
        f"(accepted length {result['spec_decode']['mean_accepted_length']})"
    )
    out = args.out or f"{args.label}-{args.preset}.json"
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

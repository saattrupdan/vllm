#!/usr/bin/env python3
"""Summarise a concurrency sweep as markdown tables.

Reads the ``<arm>-c<N>.json`` files written by ``vllm bench serve`` and the
``<arm>-c<N>-spec.json`` sidecars written by ``concurrency_sweep.sh``, then
reports one table per arm plus a cross-arm aggregate-throughput table.

Aggregate throughput and per-stream throughput move in opposite directions as
concurrency rises, so both are reported: the first is what a batch job sees,
the second is what one interactive user feels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

POINT = re.compile(r"^(?P<arm>.+)-c(?P<concurrency>\d+)\.json$")


def load(result_dir: Path) -> dict[str, dict[int, dict]]:
    arms: dict[str, dict[int, dict]] = {}
    for path in sorted(result_dir.glob("*.json")):
        match = POINT.match(path.name)
        if not match:
            continue
        with path.open() as handle:
            point = json.load(handle)
        spec_path = path.with_name(f"{path.stem}-spec.json")
        if spec_path.exists():
            with spec_path.open() as handle:
                point["spec"] = json.load(handle)
        arm = match["arm"]
        arms.setdefault(arm, {})[int(match["concurrency"])] = point
    return arms


def per_stream(point: dict, key: str) -> float | None:
    """Convert a time-per-output-token in ms into tokens/s for one stream."""
    tpot = point.get(key)
    return 1000.0 / tpot if tpot else None


def cell(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def arm_table(arm: str, points: dict[int, dict]) -> list[str]:
    concurrencies = sorted(points)
    baseline = points[concurrencies[0]].get("output_throughput")
    lines = [
        f"#### {arm}",
        "",
        (
            "| Concurrency | Total tok/s | Scaling | Per-stream tok/s | "
            "p99 stream tok/s | TTFT mean ms | TTFT p99 ms | Accepted len | "
            "Preemptions |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency in concurrencies:
        point = points[concurrency]
        spec = point.get("spec", {})
        total = point.get("output_throughput")
        scaling = f"{total / baseline:.2f}x" if total and baseline else "-"
        lines.append(
            f"| {concurrency} | {cell(total)} | {scaling} | "
            f"{cell(per_stream(point, 'mean_tpot_ms'))} | "
            f"{cell(per_stream(point, 'p99_tpot_ms'))} | "
            f"{cell(point.get('mean_ttft_ms'), 0)} | "
            f"{cell(point.get('p99_ttft_ms'), 0)} | "
            f"{cell(spec.get('mean_accepted_length'), 2)} | "
            f"{spec.get('preemptions', '-')} |"
        )
    return lines + [""]


def summary_table(arms: dict[str, dict[int, dict]]) -> list[str]:
    names = sorted(arms)
    concurrencies = sorted({c for points in arms.values() for c in points})
    header = " | ".join(names)
    lines = [
        "#### Aggregate output tokens/s by admission cap",
        "",
        f"| Concurrency | {header} |",
        "|---:" * (len(names) + 1) + "|",
    ]
    for concurrency in concurrencies:
        row = [
            cell(arms[name].get(concurrency, {}).get("output_throughput"))
            for name in names
        ]
        lines.append(f"| {concurrency} | " + " | ".join(row) + " |")
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=Path(__file__).parent / "results" / "concurrency",
        type=Path,
    )
    args = parser.parse_args()

    arms = load(args.result_dir)
    if not arms:
        print(f"no sweep results under {args.result_dir}")
        return 1

    lines: list[str] = []
    if len(arms) > 1:
        lines += summary_table(arms)
    for arm in sorted(arms):
        lines += arm_table(arm, arms[arm])
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

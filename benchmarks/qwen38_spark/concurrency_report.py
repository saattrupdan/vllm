#!/usr/bin/env python3
"""Summarise a concurrency sweep as markdown tables.

Reads the ``<arm>-c<N>-r<K>.json`` files written by ``vllm bench serve`` and the
matching ``-spec.json`` sidecars written by ``concurrency_sweep.sh``, then
reports a decision table, a cross-arm aggregate-throughput table and one detail
table per arm. Repeats of the same point are collapsed to mean +/- sample
stdev, because A7 showed that differences of a few percent on this stack are
not resolvable and single passes invite reading noise as signal.

Aggregate throughput and per-stream throughput move in opposite directions as
concurrency rises, so both are always reported: the first is what a batch job
sees, the second is what one interactive user feels.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

POINT = re.compile(r"^(?P<arm>.+?)-c(?P<concurrency>\d+)(?:-r(?P<repeat>\d+))?\.json$")

Arms = dict[str, dict[int, list[dict]]]


def arm_sort_key(name: str) -> tuple[int, float | str]:
    """Order ``seqs<N>`` arms numerically and anything else alphabetically."""
    match = re.fullmatch(r"seqs(\d+)", name)
    return (0, int(match[1])) if match else (1, name)


def load(result_dir: Path) -> Arms:
    arms: Arms = {}
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
        arms.setdefault(match["arm"], {}).setdefault(
            int(match["concurrency"]), []
        ).append(point)
    return arms


def values(points: list[dict], key: str, section: str | None = None) -> list[float]:
    out = []
    for point in points:
        source = point.get(section, {}) if section else point
        value = source.get(key)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def stat(
    points: list[dict], key: str, section: str | None = None
) -> tuple[float, float] | None:
    """Mean and sample stdev of one metric across repeats of a point."""
    seen = values(points, key, section)
    if not seen:
        return None
    return statistics.mean(seen), statistics.stdev(seen) if len(seen) > 1 else 0.0


def stream_stat(points: list[dict], key: str) -> tuple[float, float] | None:
    """Per-stream tokens/s, from a time-per-output-token in milliseconds."""
    rates = [1000.0 / tpot for tpot in values(points, key) if tpot]
    if not rates:
        return None
    return statistics.mean(rates), statistics.stdev(rates) if len(rates) > 1 else 0.0


def cell(value: tuple[float, float] | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    mean, stdev = value
    if stdev == 0.0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {stdev:.{digits}f}"


def peak(points_by_concurrency: dict[int, list[dict]]) -> tuple[int, float] | None:
    """Cheapest concurrency that reaches an arm's peak aggregate throughput.

    Throughput is flat once offered load passes the admission cap, so the
    highest single mean is often an arbitrary point chosen by noise. Reporting
    the smallest concurrency within one stdev of the best keeps the latency
    columns beside it meaningful rather than describing a deep queue.
    """
    measured = {
        concurrency: value
        for concurrency, points in points_by_concurrency.items()
        if (value := stat(points, "output_throughput"))
    }
    if not measured:
        return None
    best = max(measured.values(), key=lambda value: value[0])
    # A7 put the resolvable difference on this stack at about 2%, so treat
    # anything inside that as having reached the plateau.
    threshold = best[0] - max(best[1], 0.02 * best[0])
    knee = min(c for c, value in measured.items() if value[0] >= threshold)
    return knee, measured[knee][0]


def decision_table(arms: Arms) -> list[str]:
    lines = [
        "#### Decision table: peak aggregate throughput per admission cap",
        "",
        (
            "| Admission cap | Peak total tok/s | at concurrency | "
            "Per-stream tok/s there | TTFT mean ms | Preemptions |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(arms, key=arm_sort_key):
        best = peak(arms[name])
        if best is None:
            continue
        concurrency, _ = best
        points = arms[name][concurrency]
        lines.append(
            f"| {name} | {cell(stat(points, 'output_throughput'))} | "
            f"{concurrency} | {cell(stream_stat(points, 'mean_tpot_ms'))} | "
            f"{cell(stat(points, 'mean_ttft_ms'), 0)} | "
            f"{cell(stat(points, 'preemptions', 'spec'), 0)} |"
        )
    return lines + [""]


def summary_table(arms: Arms) -> list[str]:
    names = sorted(arms, key=arm_sort_key)
    concurrencies = sorted({c for points in arms.values() for c in points})
    lines = [
        "#### Aggregate output tokens/s",
        "",
        "| Concurrency | " + " | ".join(names) + " |",
        "|---:" * (len(names) + 1) + "|",
    ]
    for concurrency in concurrencies:
        row = [
            cell(stat(arms[name].get(concurrency, []), "output_throughput"))
            for name in names
        ]
        lines.append(f"| {concurrency} | " + " | ".join(row) + " |")
    return lines + [""]


def arm_table(arm: str, points_by_concurrency: dict[int, list[dict]]) -> list[str]:
    concurrencies = sorted(points_by_concurrency)
    baseline = stat(points_by_concurrency[concurrencies[0]], "output_throughput")
    lines = [
        f"#### {arm}",
        "",
        (
            "| Concurrency | Passes | Total tok/s | Scaling | Per-stream tok/s | "
            "p99 stream tok/s | TTFT mean ms | TTFT p99 ms | Accepted len | "
            "Preemptions |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency in concurrencies:
        points = points_by_concurrency[concurrency]
        total = stat(points, "output_throughput")
        comparable = total and baseline and baseline[0]
        scaling = f"{total[0] / baseline[0]:.2f}x" if comparable else "-"
        lines.append(
            f"| {concurrency} | {len(points)} | {cell(total)} | {scaling} | "
            f"{cell(stream_stat(points, 'mean_tpot_ms'))} | "
            f"{cell(stream_stat(points, 'p99_tpot_ms'))} | "
            f"{cell(stat(points, 'mean_ttft_ms'), 0)} | "
            f"{cell(stat(points, 'p99_ttft_ms'), 0)} | "
            f"{cell(stat(points, 'mean_accepted_length', 'spec'), 2)} | "
            f"{cell(stat(points, 'preemptions', 'spec'), 0)} |"
        )
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
        lines += decision_table(arms)
        lines += summary_table(arms)
    for arm in sorted(arms, key=arm_sort_key):
        lines += arm_table(arm, arms[arm])
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

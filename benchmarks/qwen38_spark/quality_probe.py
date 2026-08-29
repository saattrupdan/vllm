#!/usr/bin/env python3
"""Screening quality probe: fixed short-answer questions, exact-match scored.

This is a screen, not an evaluation. It is meant to catch a checkpoint change
that visibly breaks the model, cheaply enough to run between server restarts.
A change that passes still needs a real eval from tests/evals before adoption.
Each item is asked `--repeats` times because this stack's greedy decoding is
nondeterministic (Marlin atomic-add reductions).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

ITEMS = [
    ("What is 17 * 23? Answer with the number only. /no_think", "391"),
    ("What is 1024 / 8? Answer with the number only. /no_think", "128"),
    ("What is 45 + 67 - 12? Answer with the number only. /no_think", "100"),
    (
        (
            "A shop sells pens at 3 for 12 kroner. How much do 10 pens cost? "
            "Answer with the number only. /no_think"
        ),
        "40",
    ),
    (
        (
            "If a train travels 120 km in 1.5 hours, what is its average speed "
            "in km/h? Answer with the number only. /no_think"
        ),
        "80",
    ),
    ("What is the capital of Denmark? One word. /no_think", "copenhagen"),
    ("What is the chemical symbol for gold? One word. /no_think", "au"),
    ("How many sides does a hexagon have? Number only. /no_think", "6"),
    ("In which year did the Berlin Wall fall? Number only. /no_think", "1989"),
    (
        (
            "What is the output of this Python expression: "
            "len('hello world'.split())? Number only. /no_think"
        ),
        "2",
    ),
    (
        "What does this return: sorted([3,1,2])[0]? Number only. /no_think",
        "1",
    ),
    (
        "What is 2 to the power of 10? Answer with the number only. /no_think",
        "1024",
    ),
    (
        "Which planet is closest to the Sun? One word. /no_think",
        "mercury",
    ),
    (
        "What is the past tense of the English verb 'go'? One word. /no_think",
        "went",
    ),
    (
        "How many bits are in a byte? Number only. /no_think",
        "8",
    ),
]


def ask(base_url: str, model: str, prompt: str, timeout: float) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    if content.strip():
        return content
    # The chat template does not always honour /no_think; fall back to the
    # tail of the reasoning trace so a thinking answer still scores.
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return reasoning[-200:]


def matches(answer: str, expected: str) -> bool:
    text = answer.strip().lower()
    if expected.isdigit():
        numbers = re.findall(r"-?\d[\d,]*", text.replace(" ", ""))
        return any(n.replace(",", "") == expected for n in numbers)
    return expected in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out")
    args = parser.parse_args()

    records, correct, total = [], 0, 0
    for prompt, expected in ITEMS:
        for _ in range(args.repeats):
            try:
                answer = ask(args.base_url, args.model, prompt, args.timeout)
            except Exception as error:  # noqa: BLE001 - screen must not abort
                answer = f"<error: {error}>"
            ok = matches(answer, expected)
            correct += ok
            total += 1
            records.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "answer": answer.strip()[:160],
                    "correct": ok,
                }
            )
            print(
                f"  {'ok ' if ok else 'BAD'} {expected:<12} <- {answer.strip()[:70]!r}"
            )

    score = correct / total if total else 0.0
    print(f"{args.label}: {correct}/{total} = {score:.1%}")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "label": args.label,
                    "score": score,
                    "correct": correct,
                    "total": total,
                    "records": records,
                },
                handle,
                indent=2,
            )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

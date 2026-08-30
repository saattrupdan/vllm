#!/usr/bin/env bash
# Unattended overnight driver for the C1 concurrency experiment.
#
# Runs the admission-cap matrix, picks the cap with the highest peak aggregate
# output throughput, redeploys the server on it, and leaves a summary. Designed
# to be started detached and read in the morning:
#
#   tmux new-session -d -s c1 \
#     'benchmarks/qwen38_spark/overnight_concurrency.sh 2>&1 | tee ~/c1.log'
#
# The selection rule is deliberately mechanical -- highest mean aggregate
# tokens/s, no per-stream floor -- because it has to run without a human. It is
# a starting point for the morning's reading, not a verdict: the per-stream
# cost of the winning cap is printed next to it, and RESTORE_SEQS pins a
# different cap if the tradeoff turns out to be unacceptable.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

serve_script="${SERVE_SCRIPT:-${HOME}/qwen38-spark-benchmark/serve-c1.sh}"
seqs_list="${SEQS_LIST:-8 4 2 6 16 32 64}"
repeats="${REPEATS:-4}"
hours="${HOURS:-8}"
result_dir="${RESULT_DIR:-${script_dir}/results/concurrency}"
summary="${SUMMARY:-${HOME}/c1-summary.md}"

# Deploy this cap at the end instead of the measured winner.
restore_seqs="${RESTORE_SEQS:-}"

# MTP is held at 3 for the whole run: the tool-eval evidence favoured it and
# the C-versus-D quality question is still open, so this experiment changes one
# variable only.
export MTP="${MTP:-3}"

port="${PORT:-8000}"
health_url="http://127.0.0.1:${port}/health"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

deadline=$((hours * 3600))
started="${SECONDS}"

log "C1 overnight run: caps [${seqs_list}], ${repeats} repeats, ${hours}h budget, MTP=${MTP}"
[[ -x "${serve_script}" || -f "${serve_script}" ]] \
  || { log "serve script not found: ${serve_script}"; exit 1; }

# Reserve time for the final redeploy so the box is never left without a
# healthy server, however the matrix ends.
matrix_deadline=$((deadline - 1200))

set +e
SERVE_SCRIPT="${serve_script}" \
  SEQS_LIST="${seqs_list}" \
  REPEATS="${repeats}" \
  RESULT_DIR="${result_dir}" \
  DEADLINE_SECONDS="${matrix_deadline}" \
  bash "${script_dir}/concurrency_sweep.sh"
matrix_status=$?
set -e
log "matrix finished with status ${matrix_status} after $(((SECONDS - started) / 60)) minutes"

pick_winner() {
  python3 - "${result_dir}" <<'PY'
import json, re, statistics, sys
from pathlib import Path

point = re.compile(r"^(?P<arm>.+?)-c(\d+)(?:-r(\d+))?\.json$")
arms: dict[str, dict[int, list[float]]] = {}
for path in Path(sys.argv[1]).glob("*.json"):
    match = point.match(path.name)
    if not match:
        continue
    try:
        value = json.loads(path.read_text()).get("output_throughput")
    except (json.JSONDecodeError, OSError):
        continue
    if isinstance(value, (int, float)):
        arm = match["arm"]
        arms.setdefault(arm, {}).setdefault(int(match[2]), []).append(float(value))

best = None
for arm, by_concurrency in arms.items():
    seqs = re.fullmatch(r"seqs(\d+)", arm)
    if not seqs:
        continue
    top = max(statistics.mean(v) for v in by_concurrency.values())
    if best is None or top > best[1]:
        best = (int(seqs[1]), top)
print(f"{best[0]} {best[1]:.1f}" if best else "")
PY
}

winner=""
if [[ -n "${restore_seqs}" ]]; then
  winner="${restore_seqs}"
  log "RESTORE_SEQS set, deploying cap ${winner} regardless of the measurements"
else
  read -r winner winner_rate <<<"$(pick_winner || true)"
  if [[ -n "${winner}" ]]; then
    log "measured winner: --max-num-seqs ${winner} at ${winner_rate} tok/s peak aggregate"
  else
    winner=8
    log "no usable results; falling back to the shipped cap ${winner}"
  fi
fi

log "deploying --max-num-seqs ${winner}"
if SEQS="${winner}" bash "${serve_script}"; then
  for _ in $(seq 1 240); do
    curl --fail --silent --max-time 5 "${health_url}" >/dev/null 2>&1 && break
    sleep 10
  done
fi

if curl --fail --silent --max-time 5 "${health_url}" >/dev/null 2>&1; then
  log "server healthy on --max-num-seqs ${winner}"
  deployed="${winner}"
else
  log "deploy of cap ${winner} did not come up; retrying the shipped cap 8"
  SEQS=8 bash "${serve_script}" || true
  for _ in $(seq 1 240); do
    curl --fail --silent --max-time 5 "${health_url}" >/dev/null 2>&1 && break
    sleep 10
  done
  deployed="8 (fallback)"
fi

{
  echo "# C1 concurrency sweep"
  echo
  echo "- Finished: $(date '+%F %T')"
  echo "- Duration: $(((SECONDS - started) / 60)) minutes"
  echo "- Caps swept: ${seqs_list}"
  echo "- Repeats per point: ${repeats}"
  echo "- MTP held at: ${MTP}"
  echo "- Deployed: --max-num-seqs ${deployed}"
  echo "- Selection rule: highest mean aggregate output tokens/s, no per-stream floor"
  echo
  echo "Revert with: SEQS=<n> MTP=${MTP} bash ${serve_script}"
  echo
  python3 "${script_dir}/concurrency_report.py" "${result_dir}" 2>&1 || true
} >"${summary}"

log "summary written to ${summary}"
cat "${summary}"

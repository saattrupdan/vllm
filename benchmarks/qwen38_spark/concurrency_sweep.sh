#!/usr/bin/env bash
# Concurrency sweep for Qwen3.8-Flash-Next on DGX Spark.
#
# Answers two separate questions that the single-stream decode probe cannot:
#
#   1. Offered load. With the server pinned at one --max-num-seqs, how does
#      aggregate output throughput scale as more requests are held in flight,
#      and what does that cost per stream?
#   2. Admission cap. Does raising --max-num-seqs move the saturation point,
#      or is the ceiling set by the PLE gather and MTP verify instead?
#
# Question 1 needs no restart. Question 2 restarts the server per value, so it
# only runs when SERVE_SCRIPT and SEQS_LIST are both set.
#
#   # sweep offered load against whatever is already running
#   benchmarks/qwen38_spark/concurrency_sweep.sh
#
#   # full matrix, restarting the server for each admission cap
#   SERVE_SCRIPT=~/qwen38-autoround/scripts/serve-intel-ar.sh \
#   SEQS_LIST="8 32 64 16 4" REPEATS=4 \
#     benchmarks/qwen38_spark/concurrency_sweep.sh
#
# Built to run unattended: a point that fails is logged and skipped, an arm
# whose server never becomes healthy is abandoned rather than fatal, and
# DEADLINE_SECONDS stops the matrix cleanly instead of overrunning. Results are
# written per point, so a partial run still renders a table.
#
# The client runs inside the serving container over the loopback interface, so
# it does not pay for the host network stack, but it does share CPU with the 32
# PLE gather threads. Set CLIENT_CPUS to pin it to the A725 efficiency cores if
# that contention shows up as a client-side bottleneck.
set -euo pipefail

container="${CONTAINER:-qwen38-ar}"
port="${PORT:-8000}"
model="${MODEL:-qwen3.8-flash-next}"
tokenizer="${TOKENIZER:-/model}"
metrics_url="${METRICS_URL:-http://127.0.0.1:${port}/metrics}"
health_url="${HEALTH_URL:-http://127.0.0.1:${port}/health}"

concurrency="${CONCURRENCY:-1 2 4 8 16 32 64}"
seqs_list="${SEQS_LIST:-}"
serve_script="${SERVE_SCRIPT:-}"
repeats="${REPEATS:-1}"

input_len="${INPUT_LEN:-550}"
output_len="${OUTPUT_LEN:-256}"
prefix_len="${PREFIX_LEN:-50}"
# Two rounds of concurrent decode per point is enough for a stable mean when
# repeats supply the error bars, and the smoke test showed aggregate throughput
# is roughly flat in concurrency on this stack -- so work per point translates
# almost directly into wall clock, and four prompts per slot would have put a
# single cap-64 point at over twenty minutes.
prompts_per_slot="${PROMPTS_PER_SLOT:-2}"
min_prompts="${MIN_PROMPTS:-8}"
max_prompts="${MAX_PROMPTS:-96}"
client_cpus="${CLIENT_CPUS:-}"

# Requests beyond the admission cap queue rather than adding throughput, so
# sizing the request count by min(concurrency, cap) keeps every point at
# roughly constant duration instead of letting C=64 on a SEQS=4 arm run for a
# quarter of an hour. Set by the matrix loop; 0 means "cap unknown, use C".
seqs_cap="${SEQS_CAP:-0}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sonnet_src="${SONNET:-${script_dir}/../sonnet.txt}"
result_dir="${RESULT_DIR:-${script_dir}/results/concurrency}"
container_dir=/tmp/qwen38-conc

# A cold AutoRound load reads 47.7 GiB of PLE table plus the int4 checkpoint
# off NVMe, so the health timeout is generous.
health_timeout="${HEALTH_TIMEOUT:-2400}"
deadline_seconds="${DEADLINE_SECONDS:-0}"
started_at="${SECONDS}"

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { log "WARNING: $*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

time_left() {
  ((deadline_seconds == 0)) && { echo 999999; return; }
  echo $((deadline_seconds - (SECONDS - started_at)))
}

# L1 wedged this host badly enough to need a physical power-cycle, so record
# memory either side of every load for forensics if a restart goes wrong.
log_memory() {
  local available
  available="$(awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo \
    2>/dev/null || echo "?")"
  log "$1: MemAvailable ${available} GiB"
}

wait_healthy() {
  local deadline=$((SECONDS + health_timeout))
  while ((SECONDS < deadline)); do
    if curl --fail --silent --max-time 5 "${health_url}" >/dev/null 2>&1; then
      log "server healthy"
      return 0
    fi
    sleep 10
  done
  return 1
}

preflight() {
  docker inspect "${container}" >/dev/null 2>&1 \
    || die "container not running: ${container}"
  [[ -f "${sonnet_src}" ]] || die "sonnet corpus not found: ${sonnet_src}"

  # The container ships the fork's vendored vLLM, not this checkout. Fail here
  # with a readable message rather than midway through the matrix. Bare --help
  # only lists group names on this build, so ask for the groups by name.
  local help
  help="$(docker exec "${container}" vllm bench serve --help=options 2>&1
    docker exec "${container}" vllm bench serve --help='sonnet dataset options' 2>&1)" \
    || die "'vllm bench serve' unavailable in ${container}"
  local flag
  for flag in --max-concurrency --sonnet-input-len --sonnet-prefix-len \
    --ignore-eos --result-dir --percentile-metrics; do
    grep -q -- "${flag}" <<<"${help}" \
      || die "vendored 'vllm bench serve' lacks ${flag}; run the client from a host venv instead"
  done

  stage_corpus
  mkdir -p "${result_dir}"
}

stage_corpus() {
  docker exec "${container}" mkdir -p "${container_dir}"
  docker cp "${sonnet_src}" "${container}:${container_dir}/sonnet.txt" >/dev/null
}

restart_server() {
  local seqs="$1"
  log "restarting server with --max-num-seqs ${seqs}"
  log_memory "before load (seqs ${seqs})"
  if ! SEQS="${seqs}" bash "${serve_script}" >&2; then
    warn "serve script failed for seqs ${seqs}"
    return 1
  fi
  if ! wait_healthy; then
    warn "seqs ${seqs} never became healthy within ${health_timeout}s; skipping arm"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    return 1
  fi
  log_memory "after load (seqs ${seqs})"
  # The container name is reused, so re-stage the corpus into the new instance.
  stage_corpus
}

bench() {
  local prompts="$1" conc="$2"
  shift 2
  local pin=()
  [[ -n "${client_cpus}" ]] && pin=(taskset -c "${client_cpus}")
  docker exec "${container}" ${pin[@]+"${pin[@]}"} \
    vllm bench serve \
    --backend openai \
    --endpoint /v1/completions \
    --base-url http://127.0.0.1:8000 \
    --model "${model}" \
    --served-model-name "${model}" \
    --tokenizer "${tokenizer}" \
    --dataset-name sonnet \
    --dataset-path "${container_dir}/sonnet.txt" \
    --sonnet-input-len "${input_len}" \
    --sonnet-output-len "${output_len}" \
    --sonnet-prefix-len "${prefix_len}" \
    --num-prompts "${prompts}" \
    --max-concurrency "${conc}" \
    --request-rate inf \
    --ignore-eos \
    --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el \
    "$@"
}

metric_sum() {
  awk -v metric="$2" \
    '$1 ~ ("^" metric "(\\{|$)") {sum += $NF} END {printf "%.0f", sum + 0}' "$1"
}

run_point() {
  local arm="$1" conc="$2" repeat="$3"
  local slots="${conc}"
  ((seqs_cap > 0 && conc > seqs_cap)) && slots="${seqs_cap}"
  local prompts=$((prompts_per_slot * slots))
  ((prompts < min_prompts)) && prompts="${min_prompts}"
  ((prompts > max_prompts)) && prompts="${max_prompts}"

  local name="${arm}-c${conc}-r${repeat}"
  local before="${work_dir}/before" after="${work_dir}/after"

  # Only the first repeat pays for a warm-up: each concurrency level uses a
  # different decode batch shape, and the graph stays warm across repeats.
  if ((repeat == 1)); then
    log "${name}: warm-up"
    bench "${slots}" "${conc}" >/dev/null 2>&1 || true
  fi

  curl --fail --silent "${metrics_url}" >"${before}" || {
    warn "${name}: metrics scrape failed"; return 1; }
  log "${name}: ${prompts} prompts at concurrency ${conc}"
  if ! bench "${prompts}" "${conc}" \
    --save-result --result-dir "${container_dir}" \
    --result-filename "${name}.json" >"${result_dir}/${name}.log" 2>&1; then
    warn "${name}: benchmark failed, see ${result_dir}/${name}.log"
    return 1
  fi
  curl --fail --silent "${metrics_url}" >"${after}" || true

  docker cp "${container}:${container_dir}/${name}.json" \
    "${result_dir}/${name}.json" >/dev/null || {
    warn "${name}: result file missing"; return 1; }

  local drafts draft_tokens accepted preemptions
  drafts=$(( $(metric_sum "${after}" vllm:spec_decode_num_drafts_total)
    - $(metric_sum "${before}" vllm:spec_decode_num_drafts_total) ))
  draft_tokens=$(( $(metric_sum "${after}" vllm:spec_decode_num_draft_tokens_total)
    - $(metric_sum "${before}" vllm:spec_decode_num_draft_tokens_total) ))
  accepted=$(( $(metric_sum "${after}" vllm:spec_decode_num_accepted_tokens_total)
    - $(metric_sum "${before}" vllm:spec_decode_num_accepted_tokens_total) ))
  preemptions=$(( $(metric_sum "${after}" vllm:num_preemptions_total)
    - $(metric_sum "${before}" vllm:num_preemptions_total) ))

  jq -n \
    --arg arm "${arm}" \
    --argjson concurrency "${conc}" \
    --argjson repeat "${repeat}" \
    --argjson drafts "${drafts}" \
    --argjson draft_tokens "${draft_tokens}" \
    --argjson accepted "${accepted}" \
    --argjson preemptions "${preemptions}" \
    '{
      arm: $arm,
      concurrency: $concurrency,
      repeat: $repeat,
      drafts: $drafts,
      draft_tokens: $draft_tokens,
      accepted_draft_tokens: $accepted,
      draft_acceptance_rate:
        (if $draft_tokens > 0 then $accepted / $draft_tokens else null end),
      mean_accepted_length:
        (if $drafts > 0 then 1 + $accepted / $drafts else null end),
      preemptions: $preemptions
    }' >"${result_dir}/${name}-spec.json"

  log "${name}: $(jq -r '"\(.output_throughput | floor) tok/s total, tpot \(.mean_tpot_ms | floor)ms"' \
    "${result_dir}/${name}.json")"
}

sweep() {
  local arm="$1" conc repeat
  for repeat in $(seq 1 "${repeats}"); do
    for conc in ${concurrency}; do
      if (($(time_left) < 120)); then
        warn "deadline reached, stopping ${arm} at repeat ${repeat}"
        return 0
      fi
      run_point "${arm}" "${conc}" "${repeat}" || true
    done
  done
}

preflight

if [[ -z "${seqs_list}" ]]; then
  wait_healthy || die "server not healthy"
  sweep "${ARM:-current}"
else
  [[ -n "${serve_script}" ]] || die "SEQS_LIST requires SERVE_SCRIPT"
  for seqs in ${seqs_list}; do
    if (($(time_left) < 900)); then
      warn "deadline reached, skipping remaining arms from seqs ${seqs}"
      break
    fi
    if restart_server "${seqs}"; then
      seqs_cap="${seqs}"
      sweep "seqs${seqs}"
    fi
  done
fi

log "results in ${result_dir}"

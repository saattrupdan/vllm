#!/usr/bin/env bash
# Concurrency sweep for Qwen3.8-Flash-Next on DGX Spark.
#
# Answers two separate questions that the single-stream decode probe cannot:
#
#   1. Offered load. With the server pinned at one --max-num-seqs, how does
#      aggregate output throughput scale as more requests are held in flight,
#      and where does per-stream throughput stop being acceptable?
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
#   SEQS_LIST="4 8 16 32" benchmarks/qwen38_spark/concurrency_sweep.sh
#
# The client runs inside the serving container over the loopback interface, so
# it does not pay for the host network stack, but it does share CPU with the 32
# PLE gather threads. Set CLIENT_CPUS to pin it to the A725 efficiency cores if
# that contention shows up as a client-side bottleneck.
set -euo pipefail

container="${CONTAINER:-qwen38-flash}"
port="${PORT:-18300}"
model="${MODEL:-qwen3.8-flash-next}"
tokenizer="${TOKENIZER:-/model}"
metrics_url="${METRICS_URL:-http://127.0.0.1:${port}/metrics}"
health_url="${HEALTH_URL:-http://127.0.0.1:${port}/health}"

concurrency="${CONCURRENCY:-1 2 4 8 12 16 24 32}"
seqs_list="${SEQS_LIST:-}"
serve_script="${SERVE_SCRIPT:-}"

input_len="${INPUT_LEN:-550}"
output_len="${OUTPUT_LEN:-256}"
prefix_len="${PREFIX_LEN:-50}"
prompts_per_slot="${PROMPTS_PER_SLOT:-4}"
min_prompts="${MIN_PROMPTS:-8}"
client_cpus="${CLIENT_CPUS:-}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sonnet_src="${SONNET:-${script_dir}/../sonnet.txt}"
result_dir="${RESULT_DIR:-${script_dir}/results/concurrency}"
container_dir=/tmp/qwen38-conc

# Health-check timeout is generous: a cold AutoRound load reads 47.7 GiB of PLE
# table plus the int4 checkpoint off NVMe.
health_timeout="${HEALTH_TIMEOUT:-2400}"

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

wait_healthy() {
  local deadline=$((SECONDS + health_timeout))
  while ((SECONDS < deadline)); do
    if curl --fail --silent --max-time 5 "${health_url}" >/dev/null 2>&1; then
      log "server healthy"
      return 0
    fi
    sleep 10
  done
  die "server did not become healthy within ${health_timeout}s"
}

preflight() {
  docker inspect "${container}" >/dev/null 2>&1 \
    || die "container not running: ${container}"
  [[ -f "${sonnet_src}" ]] || die "sonnet corpus not found: ${sonnet_src}"

  # The container ships the fork's vendored vLLM, not this checkout. Fail here
  # with a readable message rather than midway through the matrix.
  local help
  help="$(docker exec "${container}" vllm bench serve --help 2>&1)" \
    || die "'vllm bench serve' unavailable in ${container}"
  local flag
  for flag in --max-concurrency --sonnet-input-len --sonnet-prefix-len \
    --ignore-eos --result-dir --percentile-metrics; do
    grep -q -- "${flag}" <<<"${help}" \
      || die "vendored 'vllm bench serve' lacks ${flag}; run the client from a host venv instead"
  done

  docker exec "${container}" mkdir -p "${container_dir}"
  docker cp "${sonnet_src}" "${container}:${container_dir}/sonnet.txt" >/dev/null
  mkdir -p "${result_dir}"
}

restart_server() {
  local seqs="$1"
  log "restarting server with --max-num-seqs ${seqs}"
  SEQS="${seqs}" bash "${serve_script}" >&2
  # The container name is reused, so re-copy the corpus into the new instance.
  wait_healthy
  docker exec "${container}" mkdir -p "${container_dir}"
  docker cp "${sonnet_src}" "${container}:${container_dir}/sonnet.txt" >/dev/null
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
  local arm="$1" conc="$2"
  local prompts=$((prompts_per_slot * conc))
  ((prompts < min_prompts)) && prompts="${min_prompts}"

  local name="${arm}-c${conc}"
  local before="${work_dir}/before" after="${work_dir}/after"

  # Each concurrency level uses a different decode batch shape, so warm the
  # matching CUDA graph before the measured pass.
  log "${name}: warm-up"
  bench "${conc}" "${conc}" >/dev/null 2>&1 || true

  curl --fail --silent "${metrics_url}" >"${before}"
  log "${name}: ${prompts} prompts at concurrency ${conc}"
  bench "${prompts}" "${conc}" \
    --save-result --result-dir "${container_dir}" \
    --result-filename "${name}.json" >"${result_dir}/${name}.log" 2>&1 \
    || { cat "${result_dir}/${name}.log" >&2; die "${name} failed"; }
  curl --fail --silent "${metrics_url}" >"${after}"

  docker cp "${container}:${container_dir}/${name}.json" \
    "${result_dir}/${name}.json" >/dev/null

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
    --argjson drafts "${drafts}" \
    --argjson draft_tokens "${draft_tokens}" \
    --argjson accepted "${accepted}" \
    --argjson preemptions "${preemptions}" \
    '{
      arm: $arm,
      concurrency: $concurrency,
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
  local arm="$1" conc
  for conc in ${concurrency}; do
    run_point "${arm}" "${conc}"
  done
}

preflight

if [[ -z "${seqs_list}" ]]; then
  wait_healthy
  sweep "${ARM:-current}"
else
  [[ -n "${serve_script}" ]] || die "SEQS_LIST requires SERVE_SCRIPT"
  for seqs in ${seqs_list}; do
    restart_server "${seqs}"
    sweep "seqs${seqs}"
  done
fi

log "results in ${result_dir}"

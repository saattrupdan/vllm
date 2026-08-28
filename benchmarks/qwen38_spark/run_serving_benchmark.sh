#!/usr/bin/env bash
set -euo pipefail

base_url="${BASE_URL:-http://127.0.0.1:8000/v1}"
metrics_url="${METRICS_URL:-http://127.0.0.1:8000/metrics}"
model="${MODEL:-qwen3.8-flash-next}"
tokenizer="${TOKENIZER:-/home/saattrupdan/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
uvx_bin="${UVX_BIN:-/home/saattrupdan/.local/bin/uvx}"
benchy_version="${LLAMA_BENCHY_VERSION:-0.4.0}"
label="${LABEL:-baseline}"
suite="${1:-quick}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
result_dir="${RESULT_DIR:-${script_dir}/results}"

case "${suite}" in
  quick)
    pp=(256)
    tg=(512)
    depth=(0)
    runs="${RUNS:-5}"
    ;;
  context)
    pp=(256)
    tg=(512)
    depth=(0 8192 32768)
    runs="${RUNS:-3}"
    ;;
  *)
    echo "Usage: $0 [quick|context]" >&2
    exit 2
    ;;
esac

if ! curl --fail --silent --max-time 5 \
  "${base_url%/v1}/health" >/dev/null; then
  echo "The vLLM endpoint is not healthy: ${base_url}" >&2
  exit 1
fi

if [[ ! -x "${uvx_bin}" ]]; then
  echo "uvx not found: ${uvx_bin}" >&2
  exit 1
fi

mkdir -p "${result_dir}"
result_file="${result_dir}/${label}-${suite}.json"
spec_file="${result_dir}/${label}-${suite}-spec.json"
before_file="$(mktemp)"
after_file="$(mktemp)"
trap 'rm -f "${before_file}" "${after_file}"' EXIT

common_args=(
  --base-url "${base_url}"
  --model "${model}"
  --served-model-name "${model}"
  --tokenizer "${tokenizer}"
  --concurrency 1
  --exact-tg
  --no-cache
  --extra-body seed=42,temperature=1,top_p=0.95,top_k=20
  --exit-on-first-fail
)

# Warm kernels and validate coherence before measured counters are captured.
"${uvx_bin}" "llama-benchy==${benchy_version}" \
  "${common_args[@]}" \
  --pp 64 \
  --tg 64 \
  --depth 0 \
  --runs 1 \
  --latency-mode none \
  --format json >/dev/null

curl --fail --silent "${metrics_url}" >"${before_file}"

"${uvx_bin}" "llama-benchy==${benchy_version}" \
  "${common_args[@]}" \
  --pp "${pp[@]}" \
  --tg "${tg[@]}" \
  --depth "${depth[@]}" \
  --runs "${runs}" \
  --no-warmup \
  --skip-coherence \
  --no-adapt-prompt \
  --latency-mode none \
  --save-result "${result_file}" \
  --format json

curl --fail --silent "${metrics_url}" >"${after_file}"

metric_sum() {
  local file="$1"
  local metric="$2"
  awk -v metric="${metric}" \
    '$1 ~ ("^" metric "(\\{|$)") {sum += $NF} END {printf "%.0f", sum + 0}' \
    "${file}"
}

position_sum() {
  local file="$1"
  local position="$2"
  awk -v position="${position}" \
    '$1 ~ /^vllm:spec_decode_num_accepted_tokens_per_pos_total\{/ && \
     index($1, "position=\"" position "\"") > 0 {sum += $NF} \
     END {printf "%.0f", sum + 0}' "${file}"
}

before_drafts="$(metric_sum "${before_file}" \
  vllm:spec_decode_num_drafts_total)"
after_drafts="$(metric_sum "${after_file}" \
  vllm:spec_decode_num_drafts_total)"
before_draft_tokens="$(metric_sum "${before_file}" \
  vllm:spec_decode_num_draft_tokens_total)"
after_draft_tokens="$(metric_sum "${after_file}" \
  vllm:spec_decode_num_draft_tokens_total)"
before_accepted="$(metric_sum "${before_file}" \
  vllm:spec_decode_num_accepted_tokens_total)"
after_accepted="$(metric_sum "${after_file}" \
  vllm:spec_decode_num_accepted_tokens_total)"

drafts=$((after_drafts - before_drafts))
draft_tokens=$((after_draft_tokens - before_draft_tokens))
accepted=$((after_accepted - before_accepted))
per_position='[]'

for position in 0 1 2 3 4 5 6 7; do
  before_position="$(position_sum "${before_file}" "${position}")"
  after_position="$(position_sum "${after_file}" "${position}")"
  delta=$((after_position - before_position))
  if ((delta == 0 && position >= 2)); then
    continue
  fi
  per_position="$(jq \
    --argjson position "${position}" \
    --argjson accepted "${delta}" \
    '. + [{position: $position, accepted: $accepted}]' \
    <<<"${per_position}")"
done

jq \
  --arg label "${label}" \
  --arg suite "${suite}" \
  --argjson drafts "${drafts}" \
  --argjson draft_tokens "${draft_tokens}" \
  --argjson accepted "${accepted}" \
  --argjson per_position "${per_position}" \
  --slurpfile benchmark "${result_file}" \
  -n '
    ($benchmark[0].benchmarks[0].tg_throughput.mean // 0) as $tg_rate |
    (if $drafts > 0 then 1 + $accepted / $drafts else 0 end) as $accepted_length |
    {
      label: $label,
      suite: $suite,
      drafts: $drafts,
      draft_tokens: $draft_tokens,
      accepted_draft_tokens: $accepted,
      draft_acceptance_rate:
        (if $draft_tokens > 0 then $accepted / $draft_tokens else 0 end),
      mean_accepted_length: $accepted_length,
      approximate_target_steps_per_second:
        (if $accepted_length > 0 then $tg_rate / $accepted_length else 0 end),
      per_position: $per_position
    }
  ' >"${spec_file}"

jq . "${spec_file}"
echo "Saved ${result_file}"
echo "Saved ${spec_file}"

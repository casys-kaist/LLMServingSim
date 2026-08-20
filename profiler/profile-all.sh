#!/bin/bash
# Sweep the profiler over Qwen3-32B, Qwen3-30B-A3B-Instruct-2507, and
# Llama-3.1-8B at TP=1 and TP=2. Run from inside the vLLM Docker
# (launched via scripts/docker-vllm.sh) at /workspace:
#
#     ./profiler/profile-all.sh
#
# Each run writes to profiler/perf/<HARDWARE>/<MODEL>/<variant>/tp{1,2}/ — see
# ./profiler/profile.sh for the single-model equivalent.
# -----------------------------------------------------------------------------

set -euo pipefail

HARDWARE="${HARDWARE:-RTXPRO6000}"
TP_DEGREES="${TP_DEGREES:-1,2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
# Timed forwards per shot (averaged). N=3 tames single-sample DVFS
# jitter (~15-25% on large GEMMs → ~5%) at ~3x profile time.
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

MODELS=(
    "Qwen/Qwen3-32B"
    "Qwen/Qwen3-30B-A3B-Instruct-2507"
    "meta-llama/Llama-3.1-8B"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for MODEL in "${MODELS[@]}"; do

    cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")
    cmd+=(--tp "$TP_DEGREES")
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
    cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
    cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
    cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
    cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
    [[ -n "${SKIP_SKEW:-}" ]]              && cmd+=(--skip-skew)
    [[ -n "${SKEW_N_FACTOR:-}" ]]          && cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
    [[ -n "${SKEW_PC_FACTOR:-}" ]]         && cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
    [[ -n "${SKEW_KP_FACTOR:-}" ]]         && cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
    [[ -n "${SKEW_KVS_FACTOR:-}" ]]        && cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
    [[ -n "${ONLY_SKEW:-}" ]]              && cmd+=(--only-skew)
    [[ -n "${FORCE:-}" ]]                  && cmd+=(--force)
    [[ -n "${DTYPE:-}" ]]                  && cmd+=(--dtype "$DTYPE")
    [[ -n "${KV_CACHE_DTYPE:-}" ]]         && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "${VARIANT:-}" ]]                && cmd+=(--variant "$VARIANT")
    [[ -n "${VERBOSITY:-}" ]]              && cmd+=($VERBOSITY)

    "${cmd[@]}"
done

echo
echo "All profiles done. Output under profiler/perf/$HARDWARE/:"
for MODEL in "${MODELS[@]}"; do
    echo "  profiler/perf/$HARDWARE/$MODEL/"
done

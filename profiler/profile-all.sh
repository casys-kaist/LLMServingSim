#!/bin/bash
# -----------------------------------------------------------------------------
# Sweep the profiler over several models in one go.
#
# Run from inside the vLLM Docker (scripts/docker-vllm.sh) at /workspace:
#
#     ./profiler/profile-all.sh
#
# Each entry in JOBS is "<model>|<extra flags>". The extras are appended last
# so they override the globals below -- which is not a nicety: MiniMax-M3
# *requires* --block-size 128 (its sparse selection works in 128-token blocks,
# and the platform default of 16 fails outright with "No common block size for
# 16"), while passing 128 globally would describe a paging regime the other
# models are not simulated under.
#
# Output: profiler/perf/<HARDWARE>/<MODEL>/<variant>/tp<N>/
# See ./profiler/profile.sh for the single-model equivalent and the full flag
# list, or https://llmservingsim.ai/docs/profiler/running.
# -----------------------------------------------------------------------------

set -uo pipefail          # not -e: one model failing should not lose the rest

HARDWARE="${HARDWARE:-RTXPRO6000}"

# --- Which models, and what each one needs ----------------------------------
# TP is per job. For a **dense** model TP is the only way to shard, so both
# degrees are real deployments. For the big MoE models it is not: expert
# weights are ~98% of the total and they shard by **EP**, not TP, so a
# max-DP layout runs every instance at tp_size=1 and only tp1/ is ever read.
# Adding TP there costs hours and buys nothing until you actually simulate a
# TP x DP mix.
JOBS=(
    "meta-llama/Llama-3.1-8B|--tp 1,2"
    "Qwen/Qwen3-32B|--tp 1,2"
    "Qwen/Qwen3-30B-A3B-Instruct-2507|--tp 1,2"
    "Qwen/Qwen3.8-27B|--tp 1,2"
    "deepseek-ai/DeepSeek-V3.2-Exp|--tp 1"
    "zai-org/GLM-5|--tp 1"
    "MiniMaxAI/MiniMax-M3|--tp 1 --block-size 128"
)

# --- Globals (every one is a profile.sh variable of the same name) -----------
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
ATTENTION_MAX_KV="${ATTENTION_MAX_KV:-16384}"
ATTENTION_CHUNK_FACTOR="${ATTENTION_CHUNK_FACTOR:-2.0}"
ATTENTION_KV_FACTOR="${ATTENTION_KV_FACTOR:-2.0}"
# Timed forwards per shot (averaged). N=3 tames single-sample DVFS
# jitter (~15-25% on large GEMMs -> ~5%) at ~3x profile time.
MEASUREMENT_ITERATIONS="${MEASUREMENT_ITERATIONS:-3}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failed=()
for job in "${JOBS[@]}"; do
    MODEL="${job%%|*}"
    EXTRA="${job#*|}"
    [[ "$EXTRA" == "$MODEL" ]] && EXTRA=""

    cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
    cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
    cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
    cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
    cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
    # Optional globals, only when set.
    [[ -n "${TP_DEGREES:-}" ]]             && cmd+=(--tp "$TP_DEGREES")
    [[ -n "${BLOCK_SIZE:-}" ]]             && cmd+=(--block-size "$BLOCK_SIZE")
    [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]] && cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    [[ -n "${MAX_MODEL_LEN:-}" ]]          && cmd+=(--max-model-len "$MAX_MODEL_LEN")
    [[ -n "${NUM_HIDDEN_LAYERS:-}" ]]      && cmd+=(--num-hidden-layers "$NUM_HIDDEN_LAYERS")
    [[ -n "${PROFILE_MTP:-}" ]]            && cmd+=(--profile-mtp "$PROFILE_MTP")
    [[ -n "${LINEAR_ATTN_CHUNK:-}" ]]      && cmd+=(--linear-attn-chunk "$LINEAR_ATTN_CHUNK")
    [[ -n "${ATTENTION_DECODE_Q_LENS:-}" ]] && cmd+=(--attention-decode-q-lens "$ATTENTION_DECODE_Q_LENS")
    [[ -n "${SKIP_SKEW:-}" ]]              && cmd+=(--skip-skew)
    [[ -n "${ONLY_SKEW:-}" ]]              && cmd+=(--only-skew)
    [[ -n "${SKEW_N_FACTOR:-}" ]]          && cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
    [[ -n "${SKEW_PC_FACTOR:-}" ]]         && cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
    [[ -n "${SKEW_KP_FACTOR:-}" ]]         && cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
    [[ -n "${SKEW_KVS_FACTOR:-}" ]]        && cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
    [[ -n "${FORCE:-}" ]]                  && cmd+=(--force)
    [[ -n "${DTYPE:-}" ]]                  && cmd+=(--dtype "$DTYPE")
    [[ -n "${KV_CACHE_DTYPE:-}" ]]         && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "${VARIANT:-}" ]]                && cmd+=(--variant "$VARIANT")
    [[ -n "${OUT_ROOT:-}" ]]               && cmd+=(--out-root "$OUT_ROOT")
    [[ -n "${MODEL_CONFIG_ROOT:-}" ]]      && cmd+=(--model-config-root "$MODEL_CONFIG_ROOT")
    # HF_OVERRIDES is an array, one --hf-override per entry. A per-job extra
    # can carry model-specific ones instead.
    for _ovr in "${HF_OVERRIDES[@]:-}"; do
        [[ -n "$_ovr" ]] && cmd+=(--hf-override "$_ovr")
    done
    [[ -n "${VERBOSITY:-}" ]]              && cmd+=($VERBOSITY)
    # Per-job extras last, so they win over the globals.
    # shellcheck disable=SC2206
    [[ -n "$EXTRA" ]] && cmd+=($EXTRA)

    echo
    echo "=== $MODEL ${EXTRA:+($EXTRA)} ==="
    if ! "${cmd[@]}"; then
        echo "!! FAILED: $MODEL" >&2
        failed+=("$MODEL")
    fi
done

echo
echo "Output under profiler/perf/$HARDWARE/:"
for job in "${JOBS[@]}"; do echo "  profiler/perf/$HARDWARE/${job%%|*}/"; done

if (( ${#failed[@]} )); then
    echo
    echo "${#failed[@]} model(s) failed: ${failed[*]}" >&2
    exit 1
fi
echo
echo "All profiles done."

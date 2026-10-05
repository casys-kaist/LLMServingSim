#!/bin/bash
# -----------------------------------------------------------------------------
# Single-run profile script.
#
# This is meant to be edited in place: change the variables below to
# whatever you want to profile right now, then execute:
#
#     ./profiler/profile.sh
#
# The profiler auto-resolves the architecture from the model config's
# ``model_type`` field — you don't specify it here. Make sure the
# matching architecture yaml exists under ``profiler/models/``
# before running.
# -----------------------------------------------------------------------------

set -euo pipefail

# =============================================================================
# EDIT THESE (REQUIRED)
# =============================================================================

# HF-style model id. A raw HuggingFace config.json must exist at
# ``configs/model/<MODEL>.json`` relative to the LLMServingSim root.
# The profiler reads model_type from that config to pick an
# architecture yaml under profiler/models/.
# MODEL="meta-llama/Llama-3.1-8B"
MODEL="Qwen/Qwen3-32B"

# GPU identifier used as an output folder name under ``perf/``.
# Free-form — pick something meaningful for your hardware.
HARDWARE="RTXPRO6000"

# =============================================================================
# EDIT THESE (OPTIONAL — uncomment and adjust as needed)
# =============================================================================
# Unset or empty options are omitted, so the CLI owns every default.
# Commented assignments are examples, not active overrides. See:
#     python3 -m profiler profile --help
# Options can also be supplied through the environment. Boolean flags below
# are enabled by any non-empty value (including "0"); unset them to disable.

# --- TP sweep ---------------------------------------------------------------
# Comma-separated list; must include 1 except with ONLY_SKEW.
# TP_DEGREES="1"                  # use "1,2" to sweep both degrees

# --- Engine kwargs ----------------------------------------------------------
# DTYPE is normally inferred from the model config's ``torch_dtype``
# field (bfloat16 for every model currently in configs/model/). Only
# set it explicitly to force a different weight dtype.
# KV_CACHE_DTYPE defaults to "auto" which inherits DTYPE.
# DTYPE="bfloat16"                 # bfloat16 / float16 / float32 / fp8
# KV_CACHE_DTYPE="fp8"             # auto / fp8 / fp16 / bf16
# MAX_NUM_BATCHED_TOKENS=2048     # vLLM's --max-num-batched-tokens
# MAX_NUM_SEQS=256                # vLLM's --max-num-seqs
# BLOCK_SIZE and GPU_MEMORY_UTILIZATION mirror the simulator's --block-size
# and --npu-memory-utilization; keep them in step with whatever you simulate
# at, since a profile measured under one paging regime doesn't describe
# another. GPU_MEMORY_UTILIZATION also sets the KV block count, which every
# shot-feasibility filter is measured against, so it changes *which* shots the
# sweep contains.
# BLOCK_SIZE=16
# GPU_MEMORY_UTILIZATION=0.9
# MAX_MODEL_LEN caps the engine's context length; lowering it cuts profile
# time on a long-context model.
# MAX_MODEL_LEN=32768

# --- Layer count ------------------------------------------------------------
# Normally leave this alone. The profiler resolves the layer count from the
# checkpoint's config -- 1 for a uniform stack, and for a hybrid the smallest
# count that reaches every distinct block type (4 for Qwen3.8-27B, whose
# layer_types runs gated-DeltaNet x3 before the first full-attention layer) --
# and logs what it chose. Set this only to override that; going below what the
# stack needs leaves layers unprofiled, and the run warns when you do.
# NUM_HIDDEN_LAYERS=4

# --- Model-config overrides -------------------------------------------------
# Override any model-config field without editing configs/model/. Values are
# parsed as JSON when they parse and kept as strings otherwise; dotted keys
# nest. Use this to sweep a shape -- sparse-attention top-k, chunk length,
# expert count -- from the command line.
# HF_OVERRIDES=("index_topk=1024" "num_experts=64")

# --- Linear attention (mamba / gated DeltaNet) ------------------------------
# Chunk length the prefill scan works in, used to place grid points. Left
# unset it resolves from the model config's chunk_size, else from vLLM's
# FLA_CHUNK_SIZE. Measured cost tracks the CHUNK COUNT rather than the token
# count, so the grid samples boundaries and the points just past them.
# LINEAR_ATTN_CHUNK=64

# --- MoE target parallelism -------------------------------------------------
# DP_DEGREES selects native DP+EP component acquisition on one physical GPU.
# Supported targets require DP >= 2; EP is derived as TP * DP, retaining the
# global expert IDs and top-k. Leave MOE_EP_DEGREES unset in this mode; native
# acquisition cannot be combined with ONLY_SKEW, PROFILE_MTP or FORCE. Use a
# new OUT_ROOT to remeasure an immutable native contract.
# DP_DEGREES="2"                  # --dp; comma-separated target DP degrees
# MOE_ROUNDS=3                    # independent native measurement contexts
# Without DP_DEGREES, the retained whole-block path profiles the requested EP
# degrees with a reduced checkpoint shape. This is not native DP+EP coverage.
# MOE_EP_DEGREES="1"              # legacy --moe-ep-degrees; e.g. "1,2,4,8"

# --- Drafter (MTP) ----------------------------------------------------------
# Boot with speculative decoding so vLLM also builds the model's own MTP
# module, and profile it. Set to any non-empty value -- it is a flag, not a
# draft count: the engine boots at num_speculative_tokens=1 so what lands in
# mtp.csv is ONE drafter pass, which is the unit the simulator multiplies by
# its own --num-speculative-tokens. Only models declaring MTP modules support
# this (num_nextn_predict_layers / num_mtp_modules / mtp_num_hidden_layers).
# The sweep is cheap -- one axis, ~40 shots -- but run
# `python -m profiler coverage <model> --profile-mtp` first if the catalog has
# no `mtp:` section yet. MiniMax-M3 additionally needs MAX_MODEL_LEN lowered and
# a smaller GPU_MEMORY_UTILIZATION.
# PROFILE_MTP=1

# --- Attention grid ---------------------------------------------------------
# Bounds the histories used to construct attention shots. Unset/empty uses
# the engine's resolved context bound with room for the decode queries.
# A smaller cap reduces coverage; extrapolation beyond it is not validated.
# ATTENTION_MAX_KV=16384
# Geometric factors: smaller values (>1) make denser, more expensive grids.
# The KV factor controls both prefill-key and decode-KV; costs multiply across
# all four shape axes. All three factors inherit the CLI's sqrt(2) default.
# Existing rows are reusable only with compatible identities and exact keys.
# ATTENTION_CHUNK_FACTOR=1.4142135623730951
# ATTENTION_KV_FACTOR=1.4142135623730951
# ATTENTION_N_FACTOR=1.4142135623730951
# Query tokens per decode sequence. "1" is ordinary decoding. A
# speculative-decoding verification step submits 1 + num_speculative_tokens
# queries per sequence against that sequence's own KV, which is a different
# kernel tile shape rather than a bigger one -- so the simulator falls back to
# the nearest profiled value with a warning instead of interpolating. Profile
# the values you intend to simulate: "1,5" for N=4, "1,4" for N=3. Each extra
# value multiplies the attention grid.
# ATTENTION_DECODE_Q_LENS="1,5"

# --- Measurement averaging --------------------------------------------------
# Timed forwards per ordinary shot, averaged per invocation. More repeats
# increase acquisition cost but do not guarantee a particular error bound.
# MEASUREMENT_ITERATIONS=3

# --- Skew profiling ---------------------------------------------------------
# After the uniform attention grid, profile heterogeneous decode-KV batches
# and compile the alpha lookup. Duration depends on coverage and shot cost.
# Set SKIP_SKEW=1 to disable.
# SKIP_SKEW=1
# Set ONLY_SKEW=1 to refresh skew without the ordinary categories; requires
# compatible existing attention references. Do not combine with SKIP_SKEW.
# ONLY_SKEW=1
#
# Per-axis geometric factors for the skew sweep. 2.0 (default) is
# doubling. Crank higher (e.g. 4.0 on kvs / kp) to coarsen axes
# you don't care about and cut profile time. Lower for denser
# sampling where more accuracy is needed.
# SKEW_N_FACTOR=2.0
# SKEW_PC_FACTOR=2.0
# SKEW_KP_FACTOR=2.0
# SKEW_KVS_FACTOR=2.0
# SKEW_SAMPLES_PER_CELL=32
# SKEW_ROUNDS=3
# SKEW_SEED=0

# --- Resume vs force -------------------------------------------------------
# Default: resume. Existing CSVs are preloaded and only shots whose
# keys aren't already present get fired. Lets you extend an existing
# profile after changing feasibility (e.g. adding pc=2048 cases) in
# minutes instead of hours. Applies to every category plus skew.
# Set FORCE=1 to replace selected category/TP acquisitions. Incompatible
# measurement identities cannot be resumed; use a separate output root to
# preserve old data while remeasuring.
# FORCE=1

# --- Output naming ----------------------------------------------------------
# When omitted, the variant folder is auto-named from the effective
# DTYPE + KV_CACHE_DTYPE — e.g. "bf16" (default), "bf16-kvfp8" (FP8 KV),
# "fp8-kvfp8" (both FP8). DTYPE is pulled from the model's
# ``torch_dtype`` when unset, so you get a meaningful name without
# setting anything. Override VARIANT only for named runs (awq, gptq, ...).
# VARIANT="my_experiment"

# --- Paths ------------------------------------------------------------------
# Where the bundle lands, and where model configs are read from. Only useful
# for writing a throwaway bundle somewhere else, or pointing at a second
# configs/model tree.
# OUT_ROOT="profiler/perf"
# MODEL_CONFIG_ROOT="configs/model"

# --- Verbosity --------------------------------------------------------------
# Default is INFO (progress + TP limits). Use LOG_LEVEL or VERBOSITY, not both.
# LOG_LEVEL="INFO"                 # DEBUG / INFO / WARNING / ERROR
# VERBOSITY="--silent"             # warnings only
# VERBOSITY="--verbose"            # DEBUG + vLLM stdout

# =============================================================================
# EXECUTE — don't usually need to touch below this line.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Build the python command with only the flags that are set.
cmd=(python3 -m profiler profile "$MODEL" --hardware "$HARDWARE")

[[ -n "${TP_DEGREES:-}" ]]             && cmd+=(--tp "$TP_DEGREES")
[[ -n "${DTYPE:-}" ]]                  && cmd+=(--dtype "$DTYPE")
[[ -n "${KV_CACHE_DTYPE:-}" ]]         && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]] && cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[[ -n "${MAX_NUM_SEQS:-}" ]]           && cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
[[ -n "${BLOCK_SIZE:-}" ]]             && cmd+=(--block-size "$BLOCK_SIZE")
[[ -n "${GPU_MEMORY_UTILIZATION:-}" ]] && cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
[[ -n "${MAX_MODEL_LEN:-}" ]]          && cmd+=(--max-model-len "$MAX_MODEL_LEN")
[[ -n "${NUM_HIDDEN_LAYERS:-}" ]]      && cmd+=(--num-hidden-layers "$NUM_HIDDEN_LAYERS")
[[ -n "${MOE_EP_DEGREES:-}" ]]         && cmd+=(--moe-ep-degrees "$MOE_EP_DEGREES")
[[ -n "${DP_DEGREES:-}" ]]             && cmd+=(--dp "$DP_DEGREES")
[[ -n "${MOE_ROUNDS:-}" ]]             && cmd+=(--moe-rounds "$MOE_ROUNDS")
[[ -n "${PROFILE_MTP:-}" ]]            && cmd+=(--profile-mtp)
[[ -n "${LINEAR_ATTN_CHUNK:-}" ]]      && cmd+=(--linear-attn-chunk "$LINEAR_ATTN_CHUNK")
# HF_OVERRIDES is an array, one --hf-override per entry.
for _ovr in "${HF_OVERRIDES[@]:-}"; do
    [[ -n "$_ovr" ]] && cmd+=(--hf-override "$_ovr")
done
[[ -n "${ATTENTION_MAX_KV:-}" ]]       && cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
[[ -n "${ATTENTION_CHUNK_FACTOR:-}" ]] && cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
[[ -n "${ATTENTION_KV_FACTOR:-}" ]]    && cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
[[ -n "${ATTENTION_N_FACTOR:-}" ]]     && cmd+=(--attention-n-factor "$ATTENTION_N_FACTOR")
[[ -n "${ATTENTION_DECODE_Q_LENS:-}" ]] && cmd+=(--attention-decode-q-lens "$ATTENTION_DECODE_Q_LENS")
[[ -n "${MEASUREMENT_ITERATIONS:-}" ]] && cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
[[ -n "${SKIP_SKEW:-}" ]]              && cmd+=(--skip-skew)
[[ -n "${SKEW_N_FACTOR:-}" ]]          && cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
[[ -n "${SKEW_PC_FACTOR:-}" ]]         && cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
[[ -n "${SKEW_KP_FACTOR:-}" ]]         && cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
[[ -n "${SKEW_KVS_FACTOR:-}" ]]        && cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
[[ -n "${SKEW_SAMPLES_PER_CELL:-}" ]] && cmd+=(--skew-samples-per-cell "$SKEW_SAMPLES_PER_CELL")
[[ -n "${SKEW_ROUNDS:-}" ]]            && cmd+=(--skew-rounds "$SKEW_ROUNDS")
[[ -n "${SKEW_SEED:-}" ]]              && cmd+=(--skew-seed "$SKEW_SEED")
[[ -n "${ONLY_SKEW:-}" ]]              && cmd+=(--only-skew)
[[ -n "${FORCE:-}" ]]                  && cmd+=(--force)
[[ -n "${VARIANT:-}" ]]                && cmd+=(--variant "$VARIANT")
[[ -n "${OUT_ROOT:-}" ]]               && cmd+=(--out-root "$OUT_ROOT")
[[ -n "${MODEL_CONFIG_ROOT:-}" ]]      && cmd+=(--model-config-root "$MODEL_CONFIG_ROOT")
[[ -n "${LOG_LEVEL:-}" ]]              && cmd+=(--log-level "$LOG_LEVEL")
if [[ -n "${VERBOSITY:-}" ]]; then
    # Preserve the existing shortcut (or --log-level VALUE) without globbing.
    read -r -a verbosity_args <<< "$VERBOSITY"
    cmd+=("${verbosity_args[@]}")
fi

"${cmd[@]}"

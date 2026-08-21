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

# --- TP sweep ---------------------------------------------------------------
# Comma-separated list; must include 1.
TP_DEGREES="1,2"

# --- Engine kwargs ----------------------------------------------------------
# DTYPE is normally inferred from the model config's ``torch_dtype``
# field (bfloat16 for every model currently in configs/model/). Only
# set it explicitly to force a different weight dtype.
# KV_CACHE_DTYPE defaults to "auto" which inherits DTYPE.
# DTYPE="bfloat16"                 # bfloat16 / float16 / float32 / fp8
# KV_CACHE_DTYPE="fp8"             # auto / fp8 / fp16 / bf16
MAX_NUM_BATCHED_TOKENS=2048      # vLLM's --max-num-batched-tokens
MAX_NUM_SEQS=256                 # vLLM's --max-num-seqs

# --- Attention grid ---------------------------------------------------------
# Upper bound for kv_prefill / kv_decode axes. The grid grows
# geometrically from 512 up to min(this, max_model_len).
ATTENTION_MAX_KV=16384
# Geometric factor for the prefill_chunk axis (grows from 16 up to
# MAX_NUM_BATCHED_TOKENS). 2.0 is doubling; lower for denser sampling
# on the quadratic-cost regime at the cost of longer profile time.
ATTENTION_CHUNK_FACTOR=2.0
# Geometric factor for kv_prefill / kv_decode axes. 2.0 is doubling;
# lower for denser long-context coverage.
ATTENTION_KV_FACTOR=2.0

# --- Measurement averaging --------------------------------------------------
# Timed forwards per shot (averaged by vLLM's layerwise_profile via
# its invocations count). A single sample can swing 15-25% on large
# GEMMs due to DVFS / boost-clock jitter; N=3 (default) cuts that to
# ~5% at ~3x profile time.
MEASUREMENT_ITERATIONS=3

# --- Skew profiling ---------------------------------------------------------
# After the uniform attention grid, also profile heterogeneous
# decode-kv batches (1-2 hours per TP). Required for the alpha
# formula fit that the simulator uses to predict skewed batches.
# Set SKIP_SKEW=1 to disable.
# SKIP_SKEW=1
#
# Per-axis geometric factors for the skew sweep. 2.0 (default) is
# doubling. Crank higher (e.g. 4.0 on kvs / kp) to coarsen axes
# you don't care about and cut profile time. Lower for denser
# sampling where more accuracy is needed.
SKEW_N_FACTOR=2.0
SKEW_PC_FACTOR=2.0
SKEW_KP_FACTOR=2.0
SKEW_KVS_FACTOR=2.0

# --- Resume vs force -------------------------------------------------------
# Default: resume. Existing CSVs are preloaded and only shots whose
# keys aren't already present get fired. Lets you extend an existing
# profile after changing feasibility (e.g. adding pc=2048 cases) in
# minutes instead of hours. Applies to every category plus skew.
# Set FORCE=1 to wipe each CSV and re-profile from scratch.
# FORCE=1

# --- Output naming ----------------------------------------------------------
# When omitted, the variant folder is auto-named from the effective
# DTYPE + KV_CACHE_DTYPE — e.g. "bf16" (default), "bf16-kvfp8" (FP8 KV),
# "fp8-kvfp8" (both FP8). DTYPE is pulled from the model's
# ``torch_dtype`` when unset, so you get a meaningful name without
# setting anything. Override VARIANT only for named runs (awq, gptq, ...).
# VARIANT="my_experiment"

# --- Verbosity --------------------------------------------------------------
# Default is INFO (progress + TP limits). Uncomment one to change:
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
[[ -n "${ATTENTION_MAX_KV:-}" ]]       && cmd+=(--attention-max-kv "$ATTENTION_MAX_KV")
[[ -n "${ATTENTION_CHUNK_FACTOR:-}" ]] && cmd+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
[[ -n "${ATTENTION_KV_FACTOR:-}" ]]    && cmd+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
[[ -n "${MEASUREMENT_ITERATIONS:-}" ]] && cmd+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
[[ -n "${SKIP_SKEW:-}" ]]              && cmd+=(--skip-skew)
[[ -n "${SKEW_N_FACTOR:-}" ]]          && cmd+=(--skew-n-factor "$SKEW_N_FACTOR")
[[ -n "${SKEW_PC_FACTOR:-}" ]]         && cmd+=(--skew-pc-factor "$SKEW_PC_FACTOR")
[[ -n "${SKEW_KP_FACTOR:-}" ]]         && cmd+=(--skew-kp-factor "$SKEW_KP_FACTOR")
[[ -n "${SKEW_KVS_FACTOR:-}" ]]        && cmd+=(--skew-kvs-factor "$SKEW_KVS_FACTOR")
[[ -n "${ONLY_SKEW:-}" ]]              && cmd+=(--only-skew)
[[ -n "${FORCE:-}" ]]                  && cmd+=(--force)
[[ -n "${VARIANT:-}" ]]                && cmd+=(--variant "$VARIANT")
[[ -n "${VERBOSITY:-}" ]]              && cmd+=($VERBOSITY)

"${cmd[@]}"

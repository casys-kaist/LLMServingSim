#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVAL_DIR="${ROOT_DIR}/evaluation"
ART_DIR="${EVAL_DIR}/artifacts"
EXPECTED_DIFF_NOTICE=0

declare -a FIGURES_TO_RUN=()

usage() {
  cat <<'EOF'
Usage:
  bash compare.sh [figure...]

Figures:
  5|6|7|8|9|10
  figure_5|figure_6|figure_7|figure_8|figure_9|figure_10

Examples:
  bash compare.sh
  bash compare.sh 5 6
  bash compare.sh figure_8
EOF
}

normalize_figure() {
  case "$1" in
    5|figure_5) echo "figure_5" ;;
    6|figure_6) echo "figure_6" ;;
    7|figure_7) echo "figure_7" ;;
    8|figure_8) echo "figure_8" ;;
    9|figure_9) echo "figure_9" ;;
    10|figure_10) echo "figure_10" ;;
    *) return 1 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      figure_name=$(normalize_figure "$1") || {
        echo "[Compare] Unknown figure: $1" >&2
        usage
        exit 2
      }
      FIGURES_TO_RUN+=("${figure_name}")
      shift
      ;;
  esac
done

if [[ ${#FIGURES_TO_RUN[@]} -eq 0 ]]; then
  FIGURES_TO_RUN=(figure_5 figure_6 figure_7 figure_8 figure_9 figure_10)
fi

status=0
has_error=0

run_check() {
  local label="$1"
  shift
  echo "[Compare] ${label}"
  "$@"
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "  -> OK"
  elif [[ ${rc} -eq 1 ]]; then
    echo "  -> DIFFERENT"
    status=1
  else
    echo "  -> ERROR (exit ${rc})"
    has_error=1
  fi
}

compare_dir() {
  local lhs="$1"
  local rhs="$2"
  run_check "DIR: ${lhs#${ROOT_DIR}/} vs ${rhs#${ROOT_DIR}/}" diff -qr "${lhs}" "${rhs}"
}

compare_dir_without_sim_time() {
  local lhs="$1"
  local rhs="$2"
  run_check "DIR: ${lhs#${ROOT_DIR}/} vs ${rhs#${ROOT_DIR}/} (excluding *_sim_time.tsv)" \
    diff -qr --exclude='*_sim_time.tsv' "${lhs}" "${rhs}"
}

compare_file() {
  local lhs="$1"
  local rhs="$2"
  run_check "TSV: ${lhs#${ROOT_DIR}/} vs ${rhs#${ROOT_DIR}/}" diff -u "${lhs}" "${rhs}"
}

compare_file_allow_diff() {
  local lhs="$1"
  local rhs="$2"
  local reason="$3"
  echo "[Compare] TSV: ${lhs#${ROOT_DIR}/} vs ${rhs#${ROOT_DIR}/}"
  if diff -q "${lhs}" "${rhs}" >/dev/null 2>&1; then
    echo "  -> OK"
  else
    echo "  -> DIFFERENT (expected: ${reason})"
    EXPECTED_DIFF_NOTICE=1
  fi
}

run_figure_5() {
  local fig_dir="${EVAL_DIR}/figure_5"
  compare_dir "${fig_dir}/A6000/parsed" "${ART_DIR}/figure_5/A6000/parsed"
  compare_dir "${fig_dir}/H100/parsed" "${ART_DIR}/figure_5/H100/parsed"
  for system in MD SD+PC PDD SM; do
    compare_file "${fig_dir}/A6000/parsed/${system}_throughput.tsv" "${ART_DIR}/figure_5/A6000/parsed/${system}_throughput.tsv"
  done
  for system in MD SD+PC PDD SM; do
    compare_file "${fig_dir}/H100/parsed/${system}_throughput.tsv" "${ART_DIR}/figure_5/H100/parsed/${system}_throughput.tsv"
  done
}

run_figure_6() {
  local fig_dir="${EVAL_DIR}/figure_6"
  compare_dir "${fig_dir}/parsed" "${ART_DIR}/figure_6/parsed"
  compare_file "${fig_dir}/parsed/power_tp1.tsv" "${ART_DIR}/figure_6/parsed/power_tp1.tsv"
  compare_file "${fig_dir}/parsed/power_tp2.tsv" "${ART_DIR}/figure_6/parsed/power_tp2.tsv"
  compare_file "${fig_dir}/parsed/component_energy.tsv" "${ART_DIR}/figure_6/parsed/component_energy.tsv"
}

run_figure_7() {
  local fig_dir="${EVAL_DIR}/figure_7"
  compare_dir "${fig_dir}/parsed" "${ART_DIR}/figure_7/parsed"
  compare_file "${fig_dir}/parsed/SD+PC.tsv" "${ART_DIR}/figure_7/parsed/SD+PC.tsv"
  compare_file "${fig_dir}/parsed/MD+PC+PS_inst0.tsv" "${ART_DIR}/figure_7/parsed/MD+PC+PS_inst0.tsv"
  compare_file "${fig_dir}/parsed/MD+PC+PS_inst1.tsv" "${ART_DIR}/figure_7/parsed/MD+PC+PS_inst1.tsv"
  compare_file "${fig_dir}/parsed/MD+PC+PS_shared_cpu.tsv" "${ART_DIR}/figure_7/parsed/MD+PC+PS_shared_cpu.tsv"
}

run_figure_8() {
  local fig_dir="${EVAL_DIR}/figure_8"
  compare_dir_without_sim_time "${fig_dir}/parsed" "${ART_DIR}/figure_8/parsed"
  for scenario in SD MD PDD SM; do
    compare_file "${fig_dir}/parsed/${scenario}_latency.tsv" "${ART_DIR}/figure_8/parsed/${scenario}_latency.tsv"
    compare_file_allow_diff \
      "${fig_dir}/parsed/${scenario}_sim_time.tsv" \
      "${ART_DIR}/figure_8/parsed/${scenario}_sim_time.tsv" \
      "hardware-dependent simulation time"
  done
  echo "[Compare] Figure 8 note: *_sim_time.tsv is hardware-dependent."
  echo "          Differences are expected and do not fail compare."
}

run_figure_9() {
  local fig_dir="${EVAL_DIR}/figure_9"
  compare_dir "${fig_dir}/parsed" "${ART_DIR}/figure_9/parsed"
  compare_file "${fig_dir}/parsed/SD_throughput.tsv" "${ART_DIR}/figure_9/parsed/SD_throughput.tsv"
  compare_file "${fig_dir}/parsed/SD_latency.tsv" "${ART_DIR}/figure_9/parsed/SD_latency.tsv"
}

run_figure_10() {
  local fig_dir="${EVAL_DIR}/figure_10"
  compare_dir "${fig_dir}/parsed" "${ART_DIR}/figure_10/parsed"
  compare_file "${fig_dir}/parsed/gpu_only_b256_throughput.tsv" "${ART_DIR}/figure_10/parsed/gpu_only_b256_throughput.tsv"
  compare_file "${fig_dir}/parsed/pim_b256_throughput.tsv" "${ART_DIR}/figure_10/parsed/pim_b256_throughput.tsv"
  compare_file "${fig_dir}/parsed/pim_sbi_b256_throughput.tsv" "${ART_DIR}/figure_10/parsed/pim_sbi_b256_throughput.tsv"
  compare_file "${fig_dir}/parsed/component_energy.tsv" "${ART_DIR}/figure_10/parsed/component_energy.tsv"
  compare_file "${fig_dir}/parsed/energy_per_token.tsv" "${ART_DIR}/figure_10/parsed/energy_per_token.tsv"
}

for fig in "${FIGURES_TO_RUN[@]}"; do
  echo
  echo "=== ${fig} ==="
  "run_${fig}"
done

if [[ ${has_error} -ne 0 ]]; then
  echo
  echo "[Compare] Completed with errors."
  exit 2
fi

if [[ ${status} -ne 0 ]]; then
  echo
  echo "[Compare] Completed with differences."
  exit 1
fi

echo
if [[ ${EXPECTED_DIFF_NOTICE} -eq 1 ]]; then
  echo "[Compare] All required checks matched."
  echo "          Expected differences were observed in hardware-dependent checks."
else
  echo "[Compare] All checks matched."
fi

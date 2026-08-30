#!/bin/bash
# Exhaustive validation: run every scenario the simulator supports and compare
# each one's total clock count against a recorded baseline. The simulator is
# deterministic, so exact equality is the test -- any difference is a behaviour
# change, intended or not.
#
# This is the counterpart to serving/run.sh, which is a menu of one example per
# feature. Coverage lives here: every cluster config, every parallelism shape,
# every scheduler and caching flag, on several workloads.
#
# Not to be confused with bench/validate.sh, which compares the simulator
# against a recorded *real vLLM* run. That one measures accuracy against ground
# truth; this one measures that nothing moved.
#
#   ./serving/validate.sh                    both stages (this is the one to run)
#   ./serving/validate.sh --clocks-only      skip the accuracy stage
#   ./serving/validate.sh dp moe_dp_pp       only these scenarios, no accuracy stage
#   ./serving/validate.sh --list             print the scenario names
#   ./serving/validate.sh --update           rerun and rewrite the baselines
#   ./serving/validate.sh --update <name>    rewrite just that scenario's baseline
#   ./serving/validate.sh --help             this text
#
# Two stages:
#   1. behaviour -- every scenario's total clock count against a recorded
#      baseline. Catches "my change moved something I did not expect".
#   2. accuracy  -- regenerates each bench/examples entry's outputs/sim.csv and
#      validation/summary.txt and checks both are unchanged. sim.csv is the
#      per-request TTFT/TPOT/latency against a recorded real vLLM run; summary.txt
#      is the error table the docs quote. Digesting both catches drift a total
#      clock could hide, and a summary that no longer describes its sim.csv. The
#      plots are regenerated but not digested -- matplotlib output is not stable
#      across versions, and a check that fails for the wrong reason stops being read.
#
# Anything that differs is printed as a markdown table you can paste into the
# PR. A difference is not automatically a bug -- but it does need a sentence
# saying what changed it and why the new number is the right one.
#
# Run it from the repo root inside the simulator container.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
BASELINES="${BASELINES:-serving/validate-baselines.txt}"
LOG_DIR="${LOG_DIR:-$(mktemp -d)}"
# mktemp -d creates it; an explicitly-set LOG_DIR may not exist yet, and every
# redirect below would fail with the scenarios reporting FAIL for the wrong reason.
mkdir -p "$LOG_DIR" || { echo "cannot create LOG_DIR: $LOG_DIR" >&2; exit 2; }
TIMEOUT="${TIMEOUT:-1800}"

COMMON=(--block-size 16 --log-level WARNING)

C=configs/cluster
TRACE=workloads/example_trace.jsonl
MIXED=workloads/workload_me2_01_mixed.jsonl
SG_L=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl
SG_Q=workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl
SG_Q32=workloads/sharegpt-qwen3-32b-300-sps10.jsonl
SWEBENCH=workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl

# name|args. example_trace.jsonl is 2-22 token prompts, so it exercises control
# flow cheaply but never fills the KV cache or makes DP members drain unevenly;
# the ShareGPT entries are there for the cases that need real sequence lengths.
SCENARIOS=(
    # --- single instance and routing across instances ---
    "single|--cluster-config $C/single_node_single_instance.json --dataset $TRACE --num-reqs 10"
    "single_sharegpt|--cluster-config $C/single_node_single_instance.json --dataset $SG_L --num-reqs 20"
    "single_mixed|--cluster-config $C/single_node_single_instance.json --dataset $MIXED --num-reqs 20"
    "multi|--cluster-config $C/single_node_multi_instance.json --dataset $TRACE --num-reqs 10"
    "multi_sharegpt|--cluster-config $C/single_node_multi_instance.json --dataset $SG_L --num-reqs 20"
    "moe_multi|--cluster-config $C/single_node_moe_multi_instance.json --dataset $TRACE --num-reqs 10"
    "memory_instance|--cluster-config $C/single_node_memory_instance.json --dataset $TRACE --num-reqs 10"

    # --- prefix caching and the memory tiers below it ---
    "prefix_cpu_pool|--cluster-config $C/single_node_multi_instance.json --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU --dataset $TRACE --num-reqs 10"
    "prefix_cpu_pool_sharegpt|--cluster-config $C/single_node_multi_instance.json --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU --dataset $SG_L --num-reqs 20"
    "dual_node_multi|--cluster-config $C/dual_node_multi_instance.json --dataset $TRACE --num-reqs 10"
    "dual_prefix_cpu_pool|--cluster-config $C/dual_node_multi_instance.json --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU --dataset $TRACE --num-reqs 10"
    "prefix_cxl_pool|--cluster-config $C/single_node_cxl_instance.json --enable-prefix-caching --enable-prefix-sharing --prefix-storage CXL --dataset $TRACE --num-reqs 10"
    "cxl|--cluster-config $C/single_node_cxl_instance.json --dataset $TRACE --num-reqs 10"
    "local_offloading|--cluster-config $C/single_node_single_instance.json --enable-local-offloading --dataset $TRACE --num-reqs 10"

    # --- scheduler knobs ---
    "chunked_prefill_512|--cluster-config $C/single_node_single_instance.json --max-num-batched-tokens 512 --dataset $SG_L --num-reqs 20"
    "chunked_prefill_off|--cluster-config $C/single_node_single_instance.json --no-enable-chunked-prefill --max-num-batched-tokens 8192 --dataset $SG_L --num-reqs 20"
    "long_prefill_threshold|--cluster-config $C/single_node_single_instance.json --long-prefill-token-threshold 256 --dataset $SG_L --num-reqs 20"
    "max_num_seqs_8|--cluster-config $C/single_node_single_instance.json --max-num-seqs 8 --dataset $SG_L --num-reqs 20"
    # --skip-prefill only flips is_init, i.e. whether the prefill counts as the
    # first token, so it moves TTFT and not the clock. This scenario checks it
    # still runs; no clock-based case can discriminate it.
    "skip_prefill|--cluster-config $C/single_node_single_instance.json --skip-prefill --dataset $TRACE --num-reqs 10"

    # --- saturated KV cache. Below the ceiling nothing is preempted and the
    # --- capacity is invisible, so these are the only scenarios that exercise
    # --- preemption, recompute and recall from a lower tier. Each flag here
    # --- changes the answer: prefix-off recomputes more, --no-reserve-full-isl
    # --- over-admits (3 preemptions -> 6), a CPU pool recalls instead.
    "saturated|--cluster-config $C/single_node_single_instance.json --npu-memory-utilization 0.18 --dataset $SG_L --num-reqs 20"
    "saturated_prefix_off|--cluster-config $C/single_node_single_instance.json --npu-memory-utilization 0.18 --no-enable-prefix-caching --dataset $SG_L --num-reqs 20"
    "saturated_no_reserve|--cluster-config $C/single_node_single_instance.json --npu-memory-utilization 0.18 --no-reserve-full-isl --dataset $SG_L --num-reqs 20"
    "saturated_cpu_pool|--cluster-config $C/single_node_single_instance.json --npu-memory-utilization 0.18 --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU --dataset $SG_L --num-reqs 20"
    # Block size and batch shape only change the answer once the cache is full:
    # unsaturated they allocate differently and compute exactly the same thing.
    "saturated_block_size_64|--cluster-config $C/single_node_single_instance.json --npu-memory-utilization 0.18 --block-size 64 --dataset $SG_L --num-reqs 20"
    "saturated_wide_batch|--cluster-config $C/single_node_single_instance.json --npu-memory-utilization 0.18 --max-num-batched-tokens 8192 --max-num-seqs 512 --dataset $SG_L --num-reqs 20"

    # --- routing policies ---
    "routing_rr|--cluster-config $C/single_node_multi_instance.json --request-routing-policy RR --dataset $TRACE --num-reqs 10"
    "routing_rand|--cluster-config $C/single_node_multi_instance.json --request-routing-policy RAND --dataset $TRACE --num-reqs 10"
    "expert_routing_rr|--cluster-config $C/single_node_moe_single_instance.json --expert-routing-policy RR --dataset $TRACE --num-reqs 10"
    "expert_routing_rand|--cluster-config $C/single_node_moe_single_instance.json --expert-routing-policy RAND --dataset $TRACE --num-reqs 10"
    "no_block_copy|--cluster-config $C/single_node_moe_single_instance.json --no-enable-block-copy --dataset $TRACE --num-reqs 10"

    # --- dense parallelism ---
    "pp|--cluster-config $C/single_node_pp_instance.json --dataset $TRACE --num-reqs 10"
    "tp_pp|--cluster-config $C/single_node_tp_pp_instance.json --dataset $TRACE --num-reqs 10"
    "dp|--cluster-config $C/single_node_dp_instance.json --dataset $TRACE --num-reqs 10"
    "dp_pp|--cluster-config $C/single_node_dp_pp_instance.json --dataset $TRACE --num-reqs 10"
    "multi_instance_8npu|--cluster-config $C/single_node_4_instance_2TP.json --dataset $TRACE --num-reqs 10"

    # --- MoE parallelism ---
    "moe|--cluster-config $C/single_node_moe_single_instance.json --dataset $TRACE --num-reqs 10"
    "moe_pp|--cluster-config $C/single_node_moe_pp_instance.json --dataset $TRACE --num-reqs 10"
    "moe_dp_ep|--cluster-config $C/single_node_moe_dp_ep_instance.json --dataset $TRACE --num-reqs 10"
    "moe_dp_tp|--cluster-config $C/single_node_moe_dp_tp_instance.json --dataset $TRACE --num-reqs 10"
    "moe_dp_pp|--cluster-config $C/single_node_moe_dp_pp_instance.json --dataset $TRACE --num-reqs 10"
    "moe_dp_tp_pp|--cluster-config $C/single_node_moe_dp_tp_pp_instance.json --dataset $TRACE --num-reqs 10"
    "dual_node_moe_dp_ep|--cluster-config $C/dual_node_moe_dp_ep_intra_inter_instance.json --dataset $TRACE --num-reqs 10"

    # --- DP with uneven drain. The only scenarios that reach an idle DP member
    # --- emitting dummy waves while its peer is still busy (issue #65's hang).
    # --- example_trace.jsonl cannot get there: its members always finish together.
    "dp_pp_uneven|--cluster-config $C/single_node_dp_pp_instance.json --dataset $SG_L --num-reqs 50"
    "moe_dp_pp_uneven|--cluster-config $C/single_node_moe_dp_pp_instance.json --dataset $SG_Q --num-reqs 2"
    "moe_dp_tp_pp_uneven|--cluster-config $C/single_node_moe_dp_tp_pp_instance.json --dataset $SG_Q --num-reqs 10"
    "moe_dp_ep_uneven|--cluster-config $C/single_node_moe_dp_ep_instance.json --dataset $SG_Q --num-reqs 10"

    # --- prefill/decode disaggregation ---
    "pd|--cluster-config $C/single_node_pd_instance.json --dataset $TRACE --num-reqs 10"
    "pd_per_instance_config|--cluster-config $C/single_node_pd_per_instance_config.json --dataset $TRACE --num-reqs 10"
    "pd_sharegpt|--cluster-config $C/single_node_pd_instance.json --dataset $SG_L --num-reqs 20"
    "moe_pd|--cluster-config $C/single_node_moe_pd_instance.json --dataset $TRACE --num-reqs 10"
    "heterogeneous|--cluster-config $C/single_node_heterogeneous.json --dataset $SG_Q32 --num-reqs 10"

    # --- PIM offloading ---
    "pim|--cluster-config $C/single_node_pim_instance.json --enable-attn-offloading --dataset $TRACE --num-reqs 10"
    "pim_sub_batch|--cluster-config $C/single_node_pim_instance.json --enable-attn-offloading --enable-sub-batch-interleaving --dataset $TRACE --num-reqs 10"

    # --- power modelling ---
    "power|--cluster-config $C/single_node_power_instance.json --dataset $TRACE --num-reqs 10 --log-interval 0.1"

    # --- agentic sessions (dependency chains through tool calls) ---
    "agentic_single|--cluster-config $C/single_node_single_instance.json --dataset $SWEBENCH --num-reqs 1"
    "agentic_moe_dp_ep|--cluster-config $C/single_node_moe_dp_ep_instance.json --dataset $SWEBENCH --num-reqs 1"

    # --- a second hardware profile ---
    "rtx4090_single|--cluster-config $C/rtx4090_single_instance.json --dataset $SG_L --num-reqs 20"
    "rtx4090_multi|--cluster-config $C/rtx4090_multi_instance.json --dataset $SG_L --num-reqs 20"
)

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    # The usage block is the header comment above, minus the leading "# ".
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#\{0,1\} \{0,1\}//'
    exit 0
fi

if [[ ${1:-} == --list ]]; then
    for entry in "${SCENARIOS[@]}"; do echo "${entry%%|*}"; done
    exit 0
fi

UPDATE=0
ACCURACY=1
while [[ ${1:-} == --* ]]; do
    case $1 in
        --update)      UPDATE=1 ;;
        --clocks-only) ACCURACY=0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done
WANTED=("$@")
# Running a subset is for iterating on one scenario, so skip the slow stage.
[[ ${#WANTED[@]} -gt 0 ]] && ACCURACY=0

want() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    local n
    for n in "${WANTED[@]}"; do [[ $n == "$1" ]] && return 0; done
    return 1
}

expected_of() {
    [[ -f $BASELINES ]] || return 0
    awk -v k="$1" '$1 == k { print $2 }' "$BASELINES"
}

printf '%-26s %-6s %-14s %s\n' SCENARIO RESULT CLOCKS NOTE
fails=0 ran=0
results=()
changed=()
changed_csv=()

for entry in "${SCENARIOS[@]}"; do
    name=${entry%%|*}
    want "$name" || continue
    read -r -a extra <<< "${entry#*|}"
    log="$LOG_DIR/$name.log"

    timeout "$TIMEOUT" "$PYTHON" -m serving "${COMMON[@]}" "${extra[@]}" \
        --run-id "validate_$name" > "$log" 2>&1
    rc=$?
    clocks=$(grep -oP 'Total clocks \(ns\):\s+\K[0-9]+' "$log" | tail -1)
    ran=$((ran + 1))

    if [[ $rc -ne 0 || -z $clocks ]]; then
        note="exit $rc, see $log"
        [[ $rc -eq 124 ]] && note="TIMED OUT after ${TIMEOUT}s (hang?), see $log"
        printf '%-26s %-6s %-14s %s\n' "$name" FAIL "${clocks:--}" "$note"
        changed+=("$name|$(expected_of "$name")|did not finish|$note")
        fails=$((fails + 1))
        continue
    fi

    results+=("$name $clocks")
    want_clk=$(expected_of "$name")
    if [[ -z $want_clk ]]; then
        printf '%-26s %-6s %-14s %s\n' "$name" NEW "$clocks" "no baseline recorded"
        [[ $UPDATE -eq 0 ]] && fails=$((fails + 1))
    elif [[ $want_clk == "$clocks" ]]; then
        printf '%-26s %-6s %-14s\n' "$name" PASS "$clocks"
    else
        delta=$(awk -v a="$clocks" -v b="$want_clk" 'BEGIN { printf "%+.4f%%", (a - b) / b * 100 }')
        printf '%-26s %-6s %-14s %s\n' "$name" FAIL "$clocks" "expected $want_clk ($delta)"
        changed+=("$name|$want_clk|$clocks|$delta")
        fails=$((fails + 1))
    fi
done

echo
if [[ $fails -eq 0 ]]; then
    echo "Behaviour: $ran/$ran scenarios match their baselines."
else
    echo "Behaviour: $fails/$ran scenarios differ. Logs in $LOG_DIR"
fi

# ---- stage 2: accuracy against the recorded real-vLLM runs ----
if [[ $ACCURACY -eq 1 ]]; then
    echo
    echo "Regenerating bench/examples (per-request accuracy vs recorded vLLM runs)..."
    # run.sh redoes the sim side (sim.csv); validate.sh redoes the comparison
    # against the recorded vLLM run (validation/summary.txt and the plots).
    # Both are needed: sim.csv alone cannot say whether the committed summary --
    # the numbers the docs quote -- still describes it.
    acc_rc=0
    ./bench/examples/run.sh > "$LOG_DIR/bench_examples_run.log" 2>&1 || acc_rc=$?
    if [[ $acc_rc -eq 0 ]]; then
        ./bench/examples/validate.sh > "$LOG_DIR/bench_examples_validate.log" 2>&1 || acc_rc=$?
    fi
    if [[ $acc_rc -ne 0 ]]; then
        echo "Accuracy: FAIL -- bench/examples exited $acc_rc, see $LOG_DIR/bench_examples_*.log"
        fails=$((fails + 1))
    else
        n_ok=0 n_seen=0
        for csv in bench/examples/*/*/outputs/sim.csv; do
            ex=${csv#bench/examples/}; ex=${ex%/outputs/sim.csv}
            # sim.csv is the per-request results; summary.txt is the error table
            # derived from it. Both are deterministic text. The plots are NOT
            # digested: matplotlib output is not stable across versions, and a
            # check that fails for the wrong reason stops being read.
            for pair in "accuracy|$csv|sim.csv" "summary|bench/examples/$ex/validation/summary.txt|summary.txt"; do
                IFS='|' read -r kind f label <<< "$pair"
                [[ -f $f ]] || continue
                got=$(md5sum "$f" | cut -d' ' -f1)
                results+=("$kind:$ex $got")
                n_seen=$((n_seen + 1))
                want_md=$(expected_of "$kind:$ex")
                if [[ -z $want_md ]]; then
                    printf '%-38s %-6s %-14s %s\n' "$ex $label" NEW "${got:0:12}" "no digest recorded"
                    [[ $UPDATE -eq 0 ]] && fails=$((fails + 1))
                elif [[ $want_md == "$got" ]]; then
                    n_ok=$((n_ok + 1))
                else
                    printf '%-38s %-6s %-14s %s\n' "$ex $label" FAIL "${got:0:12}" "changed (was ${want_md:0:12})"
                    changed_csv+=("$ex ($label)")
                    fails=$((fails + 1))
                fi
            done
        done
        if [[ $n_ok -eq $n_seen ]]; then
            echo "Accuracy: all $n_seen sim.csv + summary.txt files are byte-identical."
        fi
    fi
fi

if [[ $UPDATE -eq 1 ]]; then
    ran_clocks=$(printf '%s\n' "${results[@]}")
    {
        echo "# Total clock counts (ns) for serving/validate.sh. The simulator is"
        echo "# deterministic, so these are exact. Regenerate with:"
        echo "#     ./serving/validate.sh --update"
        # Scenarios that just ran win; any other baseline is carried over, and
        # the file keeps SCENARIOS order rather than being sorted.
        for entry in "${SCENARIOS[@]}"; do
            n=${entry%%|*}
            v=$(awk -v k="$n" '$1 == k { print $2 }' <<< "$ran_clocks")
            [[ -z $v ]] && v=$(expected_of "$n")
            [[ -n $v ]] && printf '%s %s\n' "$n" "$v"
        done
        # md5 of each bench/examples sim.csv, the accuracy baseline
        for r in "${results[@]}"; do
            [[ $r == accuracy:* || $r == summary:* ]] && echo "$r"
        done
        [[ ${#results[@]} -eq 0 || ${results[*]} != *accuracy:* ]] && grep -E '^(accuracy|summary):' "$BASELINES" 2>/dev/null
    } > "$BASELINES.tmp"
    mv "$BASELINES.tmp" "$BASELINES"
    echo
    echo "Baselines written to $BASELINES"
    exit 0
fi

# ---- report anything that moved, ready to paste into a PR ----
if [[ ${#changed[@]} -gt 0 || ${#changed_csv[@]} -gt 0 ]]; then
    report="$LOG_DIR/report.md"
    {
        echo "## Validation report"
        echo
        if [[ ${#changed[@]} -gt 0 ]]; then
            echo "Behaviour -- ${#changed[@]}/$ran scenarios changed:"
            echo
            echo "| scenario | baseline | now | delta |"
            echo "| --- | --- | --- | --- |"
            for c in "${changed[@]}"; do
                IFS='|' read -r n b a d <<< "$c"
                echo "| \`$n\` | ${b:-none} | $a | $d |"
            done
            echo
        fi
        if [[ ${#changed_csv[@]} -gt 0 ]]; then
            echo "Accuracy -- these no longer match their recorded digest:"
            echo
            printf -- '- `%s`\n' "${changed_csv[@]}"
            echo
        fi
        echo "For each row above: what in the change moved it, and why the new number"
        echo "is the right one. A difference is not automatically a bug, but it is"
        echo "never self-explanatory."
        echo
        echo "If the change is intended, land the new truth in the same PR:"
        echo
        echo "1. \`./serving/validate.sh --update\`, then commit \`serving/validate-baselines.txt\`."
        if [[ ${#changed_csv[@]} -gt 0 ]]; then
            echo "2. The run above already regenerated \`outputs/sim.csv\` and"
            echo "   \`validation/summary.txt\` in place -- commit them. Commit the three plots"
            echo "   too if git shows them changed; they are regenerated but not digested,"
            echo "   because matplotlib output is not stable across versions. Leaving any of"
            echo "   these behind publishes accuracy numbers for a simulator that no longer"
            echo "   exists."
        fi
    } > "$report"
    echo
    cat "$report"
    echo
    echo "(also written to $report)"
fi

exit $((fails > 0))

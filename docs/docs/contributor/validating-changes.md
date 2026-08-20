---
sidebar_position: 5
title: Validating your changes
---

# Validating your changes

The project does not (yet) ship a unit-test suite. Validation is
done by running the simulator against known scenarios and comparing
results. This page covers the three checks you should run before
opening a PR, in increasing order of cost.

## 1. Smoke run (every PR, ~30 seconds)

The minimum bar: run the smallest bundled scenario and confirm it
still finishes without errors.

```bash
python -m serving \
    --cluster-config configs/cluster/single_node_single_instance.json \
    --dataset workloads/example_trace.jsonl \
    --output outputs/smoke.csv \
    --num-reqs 10
```

What to check:

- Exit code is 0.
- `outputs/smoke.csv` has 10 rows plus a header.
- The final **Throughput Results** block shows non-zero
  `Average prompt throughput (tok/s)` and
  `Average generation throughput (tok/s)`.

:::tip Use `Total clocks (ns)` as the actual regression test
The simulator is deterministic: the same cluster config, workload and
flags reproduce the same makespan **exactly**. So the useful check is
not "does it look sensible" but equality against the value on `main`:

```bash
python -m serving ... 2>&1 | grep 'Total clocks'
```

Any diff in that number means your change altered simulated behaviour.
That is fine when it is the point of the change — say so in the PR, with
the before and after — and a bug when it is not. A refactor that claims
to be behaviour-preserving and moves this number is not
behaviour-preserving.

Capture it for every scenario you touch before you start editing;
recovering the baseline afterwards means stashing your work and
re-running.
:::

If your change is in `serving/`, this is the floor. Don't push a
commit that breaks the smoke run.

## 2. Targeted scenarios (when the change touches related features)

Map your edit to the scenario(s) that exercise it. The bundled
cluster configs cover the major features:

| If you touched... | Run scenario |
| --- | --- |
| `scheduler.py` (any path) | `single_node_single_instance.json` |
| Prefix caching, block pool | `single_node_multi_instance.json` with `--enable-prefix-sharing` |
| KV cache, eviction, memory model | `single_node_memory_instance.json` |
| Multi-instance routing | `single_node_multi_instance.json` |
| Prefill / decode disaggregation | `single_node_pd_instance.json` |
| MoE, expert parallelism | `single_node_moe_single_instance.json` |
| DP+EP wave sync | `single_node_moe_dp_ep_instance.json` |
| CXL placement | `single_node_cxl_instance.json` |
| PIM offload | `single_node_pim_instance.json` |
| Power model | `single_node_power_instance.json` |
| Pipeline parallelism, stage boundaries | `single_node_pp_instance.json`, `single_node_tp_pp_instance.json`, `single_node_moe_pp_instance.json` |
| Trace generator, graph generator | any of the above |

`serving/run.sh` contains ready-to-run commands for all of these.
Pick the relevant ones and confirm they still produce sensible
output.

## 3. Bench validation (changes that affect end-to-end accuracy)

If your change could move the simulator's output relative to real
vLLM (anything in `scheduler.py`, `trace_generator.py`,
`memory_model.py`, profile lookup, MoE accounting), run a bench
validation against a committed reference run.

The bench module captures a real vLLM execution, then compares the
simulator's output for the same dataset:

```bash
# 1. Rerun the sim side of an existing example
./bench/examples/run.sh Llama-3.1-8B

# 2. Compare against the committed vLLM reference
./bench/examples/validate.sh Llama-3.1-8B
```

Output lands in `bench/examples/Llama-3.1-8B/validation/`:

- `summary.txt`: aggregate error on TTFT / TPOT / throughput.
- A handful of PDFs: per-request latency CDF, throughput timeline,
  running-waiting curves.

The committed reference baselines land within 1.5% on TPOT and
end-to-end latency means and within 5.8% on TTFT means — see
**[Validation](/docs/validation)** for the per-configuration table.
**A regression beyond ~5% against those baselines is a blocker.**
Smaller movements need an explanation in the PR description (e.g.,
"this fixes an under-counting bug; the new error is closer to ground
truth than the old").

Compare against the numbers in
`bench/examples/<model>/validation/summary.txt`, not against the ~5%
figure in the abstract: TTFT already sits at -5.8% on the MoE
configuration, so "within 5%" is not a bar it currently clears.

For deeper detail on the validation methodology, see
[`bench/README.md`](https://github.com/casys-kaist/LLMServingSim/blob/main/bench/README.md).

## 4. Profiler-side changes (if you touched `profiler/`)

Profiler changes don't show up in the simulator until you regenerate
the perf bundle. Run a small profile to confirm your edit doesn't
break the pipeline:

```bash
# Inside the vLLM container
MODEL=meta-llama/Llama-3.1-8B HARDWARE=RTXPRO6000 \
    ./profiler/profile.sh
```

Then verify the simulator still loads it cleanly with the smoke
run from step 1.

If you only changed the alpha fit (`fit_alpha.py`), you can use
`SKIP_DENSE=1 SKIP_PER_SEQUENCE=1 SKIP_ATTENTION=1 SKIP_MOE=1
ONLY_SKEW=1 ./profiler/profile.sh` to refresh just `skew_fit.csv`
without rerunning the rest.

## What "this should reproduce" looks like in a PR

In your PR description, include the exact command you ran and the
key number from the output. Examples:

> Validation: `./bench/examples/validate.sh Llama-3.1-8B` →
> TTFT MAPE 2.1% (was 2.3%), TPOT MAPE 1.7% (unchanged), throughput
> 1.2% (was 1.4%).

> Smoke: `python -m serving --cluster-config
> single_node_single_instance.json --dataset example_trace.jsonl
> --num-reqs 10` runs cleanly, output CSV has expected 10 rows.

This gives the reviewer something to rerun, and gives you (and
future readers of the git log) a record of what was checked.

## When the existing scenarios don't cover what you changed

If your contribution adds a feature that no bundled scenario
exercises, **add a new bundled scenario as part of the PR.** Drop
a `configs/cluster/<your_scenario>.json` and add the matching line
to `serving/run.sh`. This makes the feature reproducible for the
next contributor and gives the reviewer something concrete to
exercise.

For features that need a custom workload (a new agentic dataset, a
specific prompt distribution), commit a small JSONL under
`workloads/` and reference it from the cluster config example.
Don't commit anything over a few MB.

## What's next

- **[PR workflow](./pr-workflow)**: how to package the change up.
- **[Reading the output](/docs/simulator/reading-output)**: what the
  per-request CSV columns mean (useful when validating).

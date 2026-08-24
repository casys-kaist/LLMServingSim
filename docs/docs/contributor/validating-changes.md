---
sidebar_position: 5
title: Validating your changes
---

# Validating your changes

The project does not ship a unit-test suite. The simulator is
**deterministic** instead — the same cluster config, workload and flags
reproduce the same makespan exactly — so validation is equality against
recorded results rather than eyeballing plots. `serving/validate.sh`
runs that comparison for you.

## 1. Run the validation script (every PR)

```bash
./serving/validate.sh
```

Two stages, about eight minutes total:

1. **Behaviour** — every scenario in `serving/validate-baselines.txt`,
   compared against its recorded `Total clocks (ns)`. Every cluster
   config, every parallelism shape (TP, PP, DP and their combinations, EP),
   prefix caching and the tiers below it, the scheduler flags, both routing
   policies, PIM, CXL, P/D disaggregation, agentic sessions and two hardware
   profiles.
2. **Accuracy** — regenerates each `bench/examples` entry's `outputs/sim.csv`
   and `validation/summary.txt` and checks both md5s. `sim.csv` is the
   per-request TTFT / TPOT / latency against a recorded real vLLM run;
   `summary.txt` is the error table this site quotes. Digesting both catches
   drift a total clock could hide, and a summary that no longer describes its
   own `sim.csv`. The three plots are regenerated but **not** digested —
   matplotlib output is not stable across versions, and a check that fails for
   the wrong reason stops being read.

A clean run ends with:

```
Behaviour: 58/58 scenarios match their baselines.
Accuracy: all 8 sim.csv + summary.txt files are byte-identical.
```

That is the bar for a change that claims to be behaviour-preserving. A
refactor that moves any of these numbers is not behaviour-preserving.

Useful variations:

```bash
./serving/validate.sh --clocks-only     # skip the slow accuracy stage
./serving/validate.sh dp moe_dp_pp      # just these scenarios, while iterating
./serving/validate.sh --list            # scenario names
./serving/validate.sh --help            # all options
```

Run it from the repo root inside the simulator container.

## 2. If something changed, report it

The script prints a markdown table of everything that moved, and writes
the same thing to `report.md` in its log directory:

```
## Validation report

Behaviour -- 1/58 scenarios changed:

| scenario | baseline | now | delta |
| --- | --- | --- | --- |
| `moe_dp_pp` | 1435561517 | 1435559904 | -0.0001% |
```

**A difference is not automatically a bug — but it is never
self-explanatory.** Paste the table into the PR and add, per row, what in
your change moved it and why the new number is the right one. Reviewers
cannot tell an intended fix from an accidental regression by looking at the
diff.

If the change is intended, land the new truth in the same PR:

1. `./serving/validate.sh --update`, then commit `serving/validate-baselines.txt`.
2. If a `sim.csv` changed, also run `./bench/examples/validate.sh` and commit
   the regenerated `outputs/sim.csv`, `validation/summary.txt` and the three
   plots for each affected example. A changed `sim.csv` makes those plots and
   that summary stale — leaving them behind publishes accuracy numbers for a
   simulator that no longer exists.

:::caution A passing scenario is not proof your case is covered
`workloads/example_trace.jsonl` has 2–22 token prompts, so most scenarios
never fill the KV cache and their DP members always drain together. That is
why issue #65 survived a green `moe_dp_pp`: the bug needed one DP member to
go idle while another was still busy. The `*_uneven` and `saturated_*`
scenarios exist for exactly those regimes. If your change targets a regime
no scenario reaches, see
[the last section](#when-the-existing-scenarios-dont-cover-what-you-changed).
:::

## 3. Bench validation (changes that affect end-to-end accuracy)

Step 1's accuracy stage tells you *whether* `sim.csv` moved. This step
tells you *by how much* — run it when that digest check fails, or when your
change could move the simulator's output relative to real vLLM (anything in
`scheduler.py`, `trace_generator.py`, `memory_model.py`, profile lookup, MoE
accounting) and you want the error numbers before opening the PR.

The bench module captures a real vLLM execution, then compares the
simulator's output for the same dataset:

```bash
# 1. Rerun the sim side of an existing example
./bench/examples/run.sh RTXPRO6000/Llama-3.1-8B

# 2. Compare against the committed vLLM reference
./bench/examples/validate.sh RTXPRO6000/Llama-3.1-8B
```

Output lands in `bench/examples/RTXPRO6000/Llama-3.1-8B/validation/`:

- `summary.txt`: aggregate error on TTFT / TPOT / throughput.
- Three PNGs: `latency.png` (per-request latency CDF), `throughput.png`
  (throughput timeline), `requests.png` (running / waiting curves).

The committed reference baselines land within 1.7% on TPOT means and
2.2% on end-to-end latency means; TTFT means span +1.3% to -13.6%
— see **[Validation](/docs/validation)** for the per-configuration
table.
**A regression beyond ~5% against those baselines is a blocker.**
Smaller movements need an explanation in the PR description (e.g.,
"this fixes an under-counting bug; the new error is closer to ground
truth than the old").

Compare against the numbers in
`bench/examples/<hardware>/<model>/validation/summary.txt`, not against the ~5%
figure in the abstract: TTFT already sits at -13.6% on the MoE
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

Then verify the simulator still loads it cleanly:
`./serving/validate.sh --clocks-only single`.

If you only changed the alpha fit (`fit_alpha.py`), you can use
`SKIP_DENSE=1 SKIP_PER_SEQUENCE=1 SKIP_ATTENTION=1 SKIP_MOE=1
ONLY_SKEW=1 ./profiler/profile.sh` to refresh just `skew_fit.csv`
without rerunning the rest.

## What "this should reproduce" looks like in a PR

In your PR description, include the exact command you ran and the
key number from the output. Examples:

> Validation: `./bench/examples/validate.sh RTXPRO6000/Llama-3.1-8B` →
> TTFT MAPE 2.1% (was 2.3%), TPOT MAPE 1.7% (unchanged), throughput
> 1.2% (was 1.4%).

> Validation: `./serving/validate.sh` → all 58 scenarios match their
> baselines, all 4 `sim.csv` byte-identical.

This gives the reviewer something to rerun, and gives you (and
future readers of the git log) a record of what was checked.

## When the existing scenarios don't cover what you changed

If your contribution adds a feature that no bundled scenario
exercises, **add a scenario as part of the PR.** Add a line to the
`SCENARIOS` list in `serving/validate.sh` (and a
`configs/cluster/<your_scenario>.json` if no bundled config fits), then
record its baseline with `./serving/validate.sh --update <name>` and commit
both. That makes the feature reproducible for the next contributor instead
of relying on them to think of it.

Prefer a scenario that would *fail* without your change. A case whose clock
matches an existing scenario exercises the flag's parsing and nothing else —
check the new number differs from the closest existing one, and if it does
not, find a configuration where the flag actually bites (turning the KV
cache saturated with `--npu-memory-utilization` is usually enough).

`serving/run.sh` is a menu of one example per feature, not a test suite —
adding to it does not get your case validated.

For features that need a custom workload (a new agentic dataset, a
specific prompt distribution), commit a small JSONL under
`workloads/` and reference it from the cluster config example.
Don't commit anything over a few MB.

## What's next

- **[PR workflow](./pr-workflow)**: how to package the change up.
- **[Reading the output](/docs/simulator/reading-output)**: what the
  per-request CSV columns mean (useful when validating).

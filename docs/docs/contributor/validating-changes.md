---
sidebar_position: 5
title: Validating your changes
---

# Validating your changes

The simulator is **deterministic** — the same cluster config, workload and flags
reproduce the same makespan exactly — so validation is equality against
recorded results in addition to focused local checks. `serving/validate.sh`
runs that comparison for you. Temporary scripts and tests written to diagnose
or verify a fix remain local under the [commit policy](./pr-workflow#commit-hygiene).

## 1. Run the validation script (every PR)

For resource-bounded execution, wrap a command with the shared watchdog:

```bash
python3 scripts/monitor_run.py --output outputs/resources/run.csv \
  --max-rss-gib 8 --min-available-gib 16 --max-swap-growth-gib 0.5 \
  --timeout 1800 -- ./serving/validate.sh
```

These are example limits; set them for your host. The CSV records sampled
process-tree RSS and host memory, and a sibling JSON records completion or the
stop reason. Use a new log path for each run. Shared pages may be counted more
than once, and sampling is not a hard memory limit; container limits remain
the backstop. Only the launched command and its descendants are stopped.
Optional GPU telemetry requires both `--gpu-uuid` and `--max-gpu-temp-c`;
it does not allocate a GPU or check other users' ownership. Confirm availability
separately before launching GPU work.

```bash
./serving/validate.sh
```

Two stages, about eight minutes total:

1. **Behaviour** — every scenario in `serving/validate-baselines.txt`,
   compared against its recorded `Total clocks (ns)`. Every cluster
   config, every parallelism shape (TP, PP, DP and their combinations, EP),
   prefix caching and the tiers below it, the scheduler flags, both routing
   policies, PIM, CXL, P/D disaggregation, agentic sessions, two hardware
   profiles — and the **model families**: linear attention, both sparse
   shapes, MLA, heterogeneous stacks and speculative decoding, which nothing
   else reaches because the rest of the list runs Llama-3.1-8B/70B,
   Qwen3-30B-A3B or Qwen3-32B.
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
Behaviour: 70/70 scenarios match their baselines.
Accuracy: all 10 sim.csv + summary.txt files are byte-identical.
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

Run it from the repo root inside the simulator container. In a **fresh**
container run `./scripts/compile.sh` first — it installs the Chakra converter
into the container's site-packages, which does not persist, and without it
every scenario reports "did not finish". Pass `LOG_DIR=<a mounted path>` too,
or the per-scenario logs die with the container.

:::caution[Parts of `profiler/` are simulator inputs]
The simulator shares data and CPU-only helpers with `profiler/`; changes there
can move simulation clocks without changing `serving/`. Important inputs include:

- **`profiler/models/*.yaml`** — the layer order. Merging two catalogs into one
  broke all 16 MoE scenarios exactly this way.
- **`profiler/core/stack.py`** — which block each decoder layer runs, resolved
  from the checkpoint's config. Shared with the profiler on purpose.
- **`profiler/core/catalog_path.py`** — `model_type` → yaml resolution. Also
  shared; adding aliasing to only one side is what broke those 16 scenarios.
- **Attention shape, skew calibration and MoE contract helpers** — geometry,
  correction-table loading and deployment compatibility affect lookup.
- **`profiler/perf/`** — measured data and hardware defaults.

Audit actual imports when deciding the regression scope rather than treating
this list as exhaustive. Shared helpers must remain usable without importing
vLLM or the GPU acquisition environment.
:::

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

:::caution[A passing scenario is not proof your case is covered]
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

- `summary.txt`: mean, median, P90, P95 and P99 errors for TTFT, TPOT and latency.
- Three PNGs: `latency.png` (per-request latency CDF), `throughput.png`
  (throughput timeline), `requests.png` (running / waiting curves).

Compare all fifteen statistics against
`bench/examples/<hardware>/<model>/validation/summary.txt`; do not use only a
mean or the figure in the abstract. The headline Llama and Qwen examples meet
the 5% absolute-error target on all fifteen statistics. The additional reduced
DeepSeek diagnostic does not. See **[Validation](/docs/validation)** for the
current per-configuration results and remaining limitations; these results
do not establish accuracy for unmeasured deployments or workloads.

A matching regression digest establishes reproducibility, not agreement with
vLLM. Explain every intentional movement, including regressions; correcting
execution geometry can expose an independent timing error. Update the
simulator-side examples and their public summaries together, preserving the
recorded vLLM references. When fixed repeated references are available, report
all fifteen statistics against each one. Engine-side variation is not a reason
to omit TTFT or a failing percentile.

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

For a calibration-only change (`skew_calibration.py`), rebuild a **copy** of
the existing bundle with `python -m profiler refit-skew MODEL --hardware HW
--out PROFILE_ROOT`. This CPU-only command does not remeasure skew or alter
attention. `--only-skew`, in contrast, is an acquisition command.
Run the committed bench examples and compare every reported latency statistic;
keep temporary tests, scripts and validation artifacts out of commits.

### If you touched `hardware.yaml`

It feeds the simulator, so a change there can move every clock in
`validate.sh` even though the file lives under `profiler/`.
`python -m profiler hardware` prints the fitted `link_bw` / `link_latency`,
the mean residual and the worst one, plus a per-collective breakdown; the
residual per size is kept in the file. Two things to check:

- **Check each collective against the backend, not just the fit equation.**
  The [calibration contract](../profiler/adding-hardware#calibration-contract)
  specifies Ring phases, GiB/s network units and local reduction costs. At
  two ranks, AllReduce has two network phases but AllGather/ReduceScatter have
  one. A common pair may leave substantial residuals even with the correct
  formula; report them rather than tuning against a model benchmark. CPU
  single-collective traces can verify the equation against the ASTRA binary.
- **Check the timing scope.** `fit.timing` records graphed, isolated or mixed
  inputs. For a captured execution target, verify that graph capture succeeded;
  isolated timings include host launch/synchronisation overhead. Individual
  primitives also do not validate grouped multi-tensor or uneven-rank MoE
  dispatch/combine, which require their own execution-path checks.

Two GPUs are needed for the second one. Without them the command writes the
spec section, records `interconnect: null`, and exits non-zero.

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

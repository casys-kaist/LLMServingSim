---
sidebar_position: 3
title: Codebase tour
---

# Codebase tour

This page answers "I want to add or change X, where do I touch?". It
is a directory map, not a behavior reference. For *what* each piece
does, see the **[Simulator](/docs/simulator/architecture)** and
**[Profiler](/docs/profiler/overview)** sections.

## The five domains

```
LLMServingSim/
├── serving/      Simulator       Python, the core loop
├── profiler/     Profiler        Python, vLLM-based latency capture
├── bench/        Bench           Python, real vLLM run + sim validation
├── workloads/    Workloads       JSONL traces + generators
├── configs/      Configs         JSON: cluster / model / PIM
├── scripts/      Env scripts     Docker launchers + builders
└── astra-sim/    Backend         C++ analytical network simulator
```

Each domain has a clear boundary. **A typical PR touches one or two
of these, not all of them.** If you find yourself editing four
domains for a single change, stop and reconsider the scope.

## Simulator (`serving/`)

Where most contributor work happens.

```
serving/
├── __main__.py              CLI + main loop
└── core/
    ├── scheduler.py         vLLM-style continuous batching
    ├── trace_generator.py   Profile lookup -> text trace
    ├── memory_model.py      KV / weight / CXL byte accounting
    ├── graph_generator.py   Text trace -> Chakra protobuf
    ├── controller.py        ASTRA-Sim subprocess IPC
    ├── router.py            Request routing across instances
    ├── gate_function.py     MoE expert routing
    ├── config_builder.py    Cluster config -> ASTRA-Sim inputs
    ├── power_model.py       Power / energy estimation
    ├── pim_model.py         PIM device model
    ├── request.py           Request / Batch dataclasses
    ├── block_pool.py        Per-tier KV block pool + prefix-cache index
    ├── kv_cache_manager.py  Tiered KV cache manager (block hashing, allocation)
    ├── logger.py            Rich-based logging + stdio capture
    └── utils.py             Model config loading, formatters
```

**Where to touch by intent:**

| Intent | Edit |
| --- | --- |
| Change scheduling policy | `scheduler.py` |
| Change how latency is looked up | `trace_generator.py` (`_lookup_*`) |
| Change byte accounting (KV, weights, prefix cache) | `memory_model.py` |
| Change inter-instance routing | `router.py` |
| Add a new CLI flag | `__main__.py` (argparse), then thread through |
| Change MoE expert distribution | `gate_function.py` |
| Change ASTRA-Sim input generation | `config_builder.py` |
| Add a new power component | `power_model.py` |

## Profiler (`profiler/`)

```
profiler/
├── __main__.py              CLI dispatch (profile / slice)
├── core/                    internals (runner, engine, categories, fit_alpha)
├── models/<model_type>.yaml Architecture catalogs (one per HF model_type)
├── perf/<hw>/<model>/...    Output bundles (CSV per category)
└── profile.sh               Editable user template
```

**Where to touch by intent:**

| Intent | Edit |
| --- | --- |
| Add a new hardware target | Run the profiler with `HARDWARE=` set; output lands in `profiler/perf/<hw>/`. See **[Profiler / Adding hardware](/docs/profiler/adding-hardware)** |
| Add a new model architecture | Drop a YAML in `profiler/models/<model_type>.yaml`. See **[Profiler / Adding model architecture](/docs/profiler/adding-model-architecture)** |
| Change the skew alpha fit | `profiler/core/fit_alpha.py` |
| Change what categories get profiled | `profiler/core/categories.py` + `profiler/core/runner.py` |
| Change output CSV columns | `core/writer.py` (and `_load_perf_db()` in `serving/core/trace_generator.py` to consume them) |

## Bench (`bench/`)

```
bench/
├── __main__.py              CLI (run / validate)
├── core/                    AsyncLLM driver, recorder, validator
├── examples/<model>/        Committed end-to-end runs
└── results/<run_id>/        Output for ad-hoc runs
```

You'll touch this only if you change the validation methodology
itself (how vLLM is driven, what metrics are compared, what plots
are emitted). For day-to-day "did my change regress?" use, see
**[Validating your changes](./validating-changes)**.

## Configs (`configs/`)

```
configs/
├── cluster/<name>.json      Cluster topology (the main thing)
├── model/<org>/<name>.json  Model architecture (subset of HF config.json)
└── pim/<name>.ini           PIM device specs (DRAMSim3 format)
```

Cluster configs are the most edited file outside `serving/`. Adding
a new scenario almost always means dropping a new
`configs/cluster/<scenario>.json` and not touching simulator code at
all. Field-by-field schema lives in
**[Reference / Cluster config](/docs/reference/cluster-config)**.

## Workloads (`workloads/`)

```
workloads/
├── *.jsonl                  Datasets (one request or session per line)
├── generators/              JSONL builders. One subcommand today: `sharegpt`
└── README.md                JSONL format reference
```

Adding a new workload generator is a contained change: a new module
under `generators/`, runnable as
`python -m workloads.generators.<your_module>`. See
**[Workloads / ShareGPT generators](/docs/workloads/sharegpt-generators)**
for the existing pattern.

## ASTRA-Sim (`astra-sim/`)

C++ network simulator, lives as a submodule. **Don't edit unless
the change targets simulator integration.** Most simulator-side
changes never touch this.

The handful of files you might edit:

| File | Why |
| --- | --- |
| `astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py` | New trace `comm_type` syntax, new trace header field, new memory location enum |
| `astra-sim/astra-sim/workload/Workload.cc` | Custom collective issuance, `involved_dim` handling |
| `astra-sim/astra-sim/system/AstraMemoryAPI.hh` | New memory tier enum (paired with `llm_converter.py`) |
| `astra-sim/inputs/...` | Don't edit. Generated by `config_builder.py` on every run |

If you do edit ASTRA-Sim, rerun `./scripts/compile.sh` before
testing. Chakra is *installed* into the container's site-packages, so
editing `llm_converter.py` alone has no effect until you reinstall it
(`cd astra-sim/extern/graph_frontend/chakra && pip3 install .`, which
`compile.sh` does for you).

## Scripts (`scripts/`)

```
scripts/
├── docker-sim.sh            Sim container launcher
├── docker-vllm.sh           vLLM container launcher (profiler / bench)
├── install-vllm.sh          Bare-metal vLLM install (uv venv)
└── compile.sh               ASTRA-Sim + Chakra build
```

You'll touch these rarely. If you add a new entry point, prefer
`python -m <module>` (handled inside the existing containers) over
adding more shell scripts.

## Tests and fixtures

There is **no unit-test suite**. The simulator is deterministic
instead, so validation is exact equality against recorded results:

```bash
./serving/validate.sh
```

58 scenarios compared against recorded `Total clocks (ns)`, then each
`bench/examples` entry's `sim.csv` and `validation/summary.txt` checked
by md5. See **[Validating your changes](./validating-changes)** for
what to do when something differs, and for the case where no scenario
covers what you changed.

When adding a feature that has clean inputs and outputs (a new
`_lookup_*` function, a new memory accounting helper), a scenario in
`serving/validate.sh` is usually the cheapest way to pin it down. A
formal unit-test framework is still an open contribution
opportunity.

## Where docs live

| Audience | Location |
| --- | --- |
| User-facing docs (this site) | `docs/` |
| Per-module developer notes | `<module>/README.md` (each top-level Python module has one) |
| Top-level project README | `README.md` |
| Project context for AI agents | `CLAUDE.md` (mirrors `AGENTS.md`) |

When you change behavior, update the relevant page under
`docs/`. When you add a feature, also update the module's
own `README.md` if it covers something the website doesn't.

## What's next

- **[Coding conventions](./conventions)**: the rules every PR
  follows.
- **[Validating your changes](./validating-changes)**: how to prove
  your change works.

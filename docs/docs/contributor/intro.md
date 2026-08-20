---
sidebar_position: 2
title: Onboarding
---

# Onboarding

This page walks you through the dev environment from a fresh clone
to a working simulator run. The goal: by the end you should be able
to edit a Python file in `serving/`, rerun a simulation, and see
your change reflected in the output CSV.

If you only plan to read code (not run it), skip to **[Codebase
tour](./codebase-tour)** instead.

## Prerequisites

- Linux (Ubuntu 22.04+ tested). macOS works for editing but not for
  running the profiler / bench (those need an NVIDIA GPU).
- Docker (for the simplest path) or the bare-metal vLLM installer
  if you can't use Docker.
- ~5 GB free disk for the simulator container, ~10 GB additional
  if you'll also profile or bench.
- A GitHub account (for the eventual PR).

You do **not** need a GPU just to run the simulator. The bundled
RTXPRO6000 / H100 profile bundles let you simulate without hardware.

## 1. Clone with submodules

ASTRA-Sim lives as a git submodule. Always clone with
`--recurse-submodules`:

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2. Pick your container

Two containers, one per role:

| Container | Image | When you need it |
| --- | --- | --- |
| `scripts/docker-sim.sh` | `astrasim/tutorial-micro2024` + Python deps | Running the simulator. **Always.** |
| `scripts/docker-vllm.sh` | `vllm/vllm-openai:v0.19.0` | Profiling new hardware, running the bench, generating workloads from ShareGPT. **Only if you touch those.** |

For most contributor work (scheduler, memory model, trace generator,
configs), the sim container is all you need:

```bash
./scripts/docker-sim.sh
```

This drops you into a shell at `/app/LLMServingSim` with all Python
deps installed. The repo root is bind-mounted, so edits on your host
are immediately visible inside.

## 3. Build ASTRA-Sim and Chakra

Inside the sim container, on first run:

```bash
./scripts/compile.sh
```

This compiles ASTRA-Sim's analytical backend (used by the simulator)
and installs the Chakra trace converter. Takes a few minutes the
first time, ~30 seconds on incremental rebuilds. Rerun whenever you
touch `astra-sim/` C++ sources.

If the compile fails with a missing dependency, the most common
cause is the submodule not being checked out. Rerun
`git submodule update --init --recursive` from the host and try
again.

## 4. Smoke run

The fastest "is everything working?" check is the bundled
single-instance trace:

```bash
python -m serving \
    --cluster-config configs/cluster/single_node_single_instance.json \
    --dataset workloads/example_trace.jsonl \
    --output outputs/onboarding_smoke.csv \
    --num-reqs 10
```

What you should see:

- A startup banner, then a **KV Cache Initialization** line reporting
  the derived capacity per instance.
- A heartbeat every second: `[N.0s] Avg prompt throughput: … tokens/s,
  Avg generation throughput: … tokens/s`, followed by an indented
  `├─Running Instance[0]: … reqs, Waiting: … reqs, …` branch.
- Ruled **Throughput Results** / **Prefix Caching Results** /
  **Instance [0]** sections at the end, with `Total requests`,
  `Total clocks (ns)`, and per-instance mean / median / P99 TTFT, TPOT
  and ITL in milliseconds.
- `outputs/onboarding_smoke.csv` containing one row per request.

If you got that, the simulator is working. If you got an error, see
**[Troubleshooting](/docs/getting-started/troubleshooting)**.

## 5. Make a real change

Time to actually edit something. A safe first edit: bump the default
log interval so you can see throughput updates more often.

Open `serving/__main__.py` and find the `--log-interval` arg
(it defaults to `1.0`). Change the default to `0.5`, save, and rerun
the smoke command from step 4. You should see twice as many
throughput log lines.

Revert the change (`git checkout serving/__main__.py`) when you're
done playing.

## 6. Read the next pages

You're now set up. Before opening a PR, please skim:

- **[Codebase tour](./codebase-tour)**: where each kind of change
  lives.
- **[Coding conventions](./conventions)**: the small set of rules
  that keep the codebase readable.
- **[Validating your changes](./validating-changes)**: how to know
  your change didn't break anything (we don't have a unit-test
  suite, so this matters).
- **[PR workflow](./pr-workflow)**: branch, commit message style,
  PR template.

## Common setup gotchas

1. **`--recurse-submodules` forgotten** → ASTRA-Sim is missing,
   `compile.sh` fails immediately. Rerun
   `git submodule update --init --recursive`.
2. **Wrong container for the task** → profiler / bench scripts will
   complain about missing CUDA or vLLM. Switch to
   `scripts/docker-vllm.sh`.
3. **Edits not visible inside the container** → check that your edit
   landed under the cloned repo dir (the container mounts the repo
   root, not your full home directory).
4. **Python version mismatch** → both containers ship the right
   Python; don't try to install your own. If you must run
   bare-metal, `scripts/install-vllm.sh` handles the vLLM side.
5. **`docker-sim.sh` says container exists** → either reattach
   (`docker exec -it servingsim_docker bash`) or remove
   (`docker rm -f servingsim_docker`) before rerunning.

## Where to ask for help

- **GitHub Discussions**:
  [casys-kaist/LLMServingSim/discussions](https://github.com/casys-kaist/LLMServingSim/discussions).
  First stop for "how do I…" questions.
- **GitHub Issues**: file under
  [casys-kaist/LLMServingSim/issues](https://github.com/casys-kaist/LLMServingSim/issues)
  with `[contributor]` in the title for setup blockers.
- **Email the main contributors**:
  [jhcho@casys.kaist.ac.kr](mailto:jhcho@casys.kaist.ac.kr?cc=hmchoi@casys.kaist.ac.kr)
  and [hmchoi@casys.kaist.ac.kr](mailto:hmchoi@casys.kaist.ac.kr?cc=jhcho@casys.kaist.ac.kr)
  (CC both whenever possible).

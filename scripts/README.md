# scripts

Shared environment, build and resource-safety entry points. Module-specific run scripts
(e.g. `profiler/profile.sh`, `bench/bench.sh`, `workloads/examples/*.sh`)
live with their module.

## Files

| File | Purpose |
| --- | --- |
| `docker-vllm.sh`  | Launch the vLLM CUDA 13 container (profiler + bench + workloads.generators). Mounts the repo as `/workspace`, pins vLLM 0.31.0, installs `datasets`, `matplotlib`, `pandas` and Torch's pinned NCCL dependency, and applies `scripts/patches/`. |
| `docker-sim.sh`   | Launch the simulator Docker container (ASTRA-Sim + sim Python deps). |
| `install-vllm.sh` | Bare-metal vLLM 0.31.0 install into the repository's Python 3.12 `.venv`. Uses uv's `cu130` PyTorch backend, installs the same extras and NCCL pin, then applies the same source patches as Docker. |
| `compile.sh`      | Build ASTRA-Sim's analytical backend and install the Chakra trace converter. Rerun it after any change under `astra-sim/`, including `llm_converter.py`, which is installed into site-packages rather than imported from the tree. |
| `monitor_run.py` | Run one command with process-tree RSS, host-memory, swap-growth and timeout guards; write CSV telemetry and a JSON completion summary. Optional GPU telemetry targets one explicit physical UUID. |

## Typical first-time setup

Inside Docker (recommended):

```bash
./scripts/docker-vllm.sh   # for profiling, benchmarking, dataset generation
./scripts/docker-sim.sh    # for simulation
./scripts/compile.sh       # one-time ASTRA-Sim + Chakra build (inside docker-sim)
```

Bare metal (vLLM side only):

```bash
./scripts/install-vllm.sh
source .venv/bin/activate
```

Run `bench/bench.sh` after entering the vLLM container or activating the venv;
the benchmark template uses the current environment's `python3` and does not
launch Docker or activate an environment itself.

The installer requires a uv version supporting `--torch-backend cu130` and a
compatible NVIDIA driver. Docker remains the reference environment; a successful
dependency installation does not establish model/backend acquisition coverage.

The retained MiniMax-M3 MTP namespace patch prevents target/drafter layer-name
collisions. The SM120 sparse-MLA fix is upstream in vLLM 0.31.0, so its backport
is removed. Step-shape logging is an explicitly optional CPU-entry diagnostic
(`VLLM_STEP_SHAPE_LOG`), not a GPU timer. Gate statistics are enabled by
`bench --record-gate-stats` or `VLLM_MOE_ACTIVATED_LOG`, require eager execution,
and cover modular routers rather than every MoE backend. Both observers are
disabled by default. The NCCL package pin does not force a communication backend.
Recorded profile bundles and benchmark truths retain their original versions;
changing the launcher does not remeasure or certify them on vLLM 0.31.0.

## Editing notes

For an existing clone, run `git submodule update --init --recursive` before
`./scripts/compile.sh`. The compile script builds and installs the checked-out
ASTRA-Sim and Chakra revisions; it does not fetch newer submodule commits.

Resource limits are configurable; choose them for the host rather than assuming
the defaults fit every machine. See `python3 scripts/monitor_run.py --help`.
Logs must use a fresh path. The watchdog stops its own command and descendants,
not unrelated workloads; container memory limits remain the hard backstop.
GPU telemetry does not reserve devices or check whether another user owns them.
All watchdog options and their defaults are listed in the
[resource-safety reference](../docs/docs/profiler/running.md#resource-safety).

| Group | Watchdog options |
| --- | --- |
| Output | `--output` (required; fresh CSV path) |
| Memory guards | `--max-rss-gib`, `--min-available-gib`, `--max-swap-growth-gib` |
| Timing | `--interval`, `--timeout` |
| GPU telemetry | `--gpu-uuid` and `--max-gpu-temp-c` (supply together) |

Place the child command after `--`.
Before a long skew sweep, preview coverage with `profiler plan-skew`, verify
exclusive GPU availability, and run the acquisition under explicit memory limits.

- Export `HF_TOKEN` in your shell when gated/private resources require it;
  `docker-vllm.sh` forwards the variable. Do not put credentials in the script.
- All GPUs are exposed by default. Select only allocated devices with, for
  example, `VLLM_GPUS='"device=0,1"' ./scripts/docker-vllm.sh`.
- `bench validate` is CPU-only and works in the simulator environment.
  Only recording new vLLM runs requires the GPU environment.
- `docker-sim.sh` includes SciPy for new three-axis native MoE interpolation.
  Its triangulation is prepared at table load; no GPU is needed for lookup.

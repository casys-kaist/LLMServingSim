# scripts

Shared environment, build and resource-safety entry points. Module-specific run scripts
(e.g. `profiler/profile.sh`, `bench/bench.sh`, `workloads/examples/*.sh`)
live with their module.

## Files

| File | Purpose |
| --- | --- |
| `docker-vllm.sh`  | Launch the vLLM Docker container (profiler + bench + workloads.generators). Mounts the repo as `/workspace`, pins vLLM 0.28.0, installs `datasets`, `matplotlib`, `pandas` and the pinned NCCL dependency, and applies `scripts/patches/`. |
| `docker-sim.sh`   | Launch the simulator Docker container (ASTRA-Sim + sim Python deps). |
| `install-vllm.sh` | Bare-metal vLLM install via `uv venv` for environments without Docker. Brings in vLLM 0.28.0 plus `datasets`, `matplotlib` and `pandas`; source patches must be applied separately in the activated environment. |
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
```

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

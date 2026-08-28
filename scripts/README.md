# scripts

Shared environment / build entry points. Module-specific run scripts
(e.g. `profiler/profile.sh`, `bench/bench.sh`, `workloads/examples/*.sh`)
live with their module — only setup and build helpers are here.

## Files

| File | Purpose |
| --- | --- |
| `docker-vllm.sh`  | Launch the vLLM Docker container (profiler + bench + workloads.generators). Mounts repo root as `/workspace`, uses official `vllm/vllm-openai:v0.28.0` image, and pre-installs `datasets` + `matplotlib` on first start. |
| `docker-sim.sh`   | Launch the simulator Docker container (ASTRA-Sim + sim Python deps). |
| `install-vllm.sh` | Bare-metal vLLM install via `uv venv` for environments without Docker. Brings in vLLM 0.28.0 plus `datasets` and `matplotlib`. |
| `compile.sh`      | Build ASTRA-Sim's analytical backend and install the Chakra trace converter. Rerun it after any change under `astra-sim/`, including `llm_converter.py`, which is installed into site-packages rather than imported from the tree. |

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

* `docker-vllm.sh` ships with a placeholder `HF_TOKEN="<your_token>"`.
  Set it to a real HuggingFace token before running so gated configs
  (Llama, etc.) auto-download on first use.
* `--gpus all` is the default; constrain via `--gpus '"device=0,1"'`
  if you want to share the host with other workloads.

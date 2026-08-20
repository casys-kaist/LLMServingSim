---
sidebar_position: 4
title: Troubleshooting
---

# Troubleshooting

Common errors during install and first run, with the quickest fix.

If your issue isn't here, please file a bug at
[github.com/casys-kaist/LLMServingSim/issues](https://github.com/casys-kaist/LLMServingSim/issues)
with the full command, the error output, and your OS / Docker / GPU
versions.

## Submodules are missing

**Symptom:** Build fails with errors about missing files under
`astra-sim/extern/graph_frontend/chakra/` or `astra-sim/build/`.

**Cause:** You cloned without `--recurse-submodules`.

**Fix:**

```bash
git submodule update --init --recursive
```

Then re-run `./scripts/compile.sh`.

## `docker: permission denied`

**Symptom:**

```text
docker: Got permission denied while trying to connect to the
Docker daemon socket
```

**Cause:** Your user isn't in the `docker` group.

**Fix:**

```bash
sudo usermod -aG docker $USER
newgrp docker
# or log out and back in
```

## GPU not detected

**Symptom:** Inside the vLLM container, `nvidia-smi` says
`command not found` or `no devices found`.

**Cause:** NVIDIA Container Toolkit isn't installed or Docker isn't
configured to use it.

**Fix:** install / re-configure the toolkit (see
[Prerequisites](./installation/prerequisites#install-nvidia-container-toolkit))
and restart Docker:

```bash
sudo systemctl restart docker
```

Then verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If the host's `nvidia-smi` works but the container's doesn't, the
toolkit is the problem. If the host's `nvidia-smi` fails too, install
the NVIDIA driver first.

## Hugging Face: gated model / 401 / 403

**Symptom:** When profiling a Llama 3.x or gated Qwen variant:

```text
huggingface_hub.utils._errors.GatedRepoError: Access to model
meta-llama/Llama-3.1-8B is restricted...
```

**Fix:**

1. Accept the license on the model page (one-time, on huggingface.co).
2. Set `HF_TOKEN` in your shell **before** launching the vLLM
   container:

   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
   ./scripts/docker-vllm.sh
   ```

The token gets forwarded into the container automatically. Confirm
with `echo $HF_TOKEN` inside the container.

## ASTRA-Sim build fails

**Symptom:** `./scripts/compile.sh` errors out partway through, often
with a CMake or compiler message.

**Common causes & fixes:**

- **Missing build deps inside the container.** The official
  `astrasim/tutorial-micro2024` image has them by default. If you
  customized the image, ensure `cmake`, `g++`, `protobuf-compiler`,
  `libprotobuf-dev`, and `libboost-dev` are installed.
- **Stale build state.** Wipe the build directories and retry:

  ```bash
  rm -rf astra-sim/build/astra_analytical/build/
  ./scripts/compile.sh
  ```
- **Outside the container.** `compile.sh` is meant to run inside the
  simulator container, not on the host. Use `./scripts/docker-sim.sh`
  first.

## `model_parallel_NPU_group <= 0` or a Chakra `VersionError`

**Symptom:** the Chakra conversion step aborts the run with
`ValueError: model_parallel_NPU_group <= 0`, or with
`google.protobuf.runtime_version.VersionError: Detected incompatible
Protobuf Gencode/Runtime versions`.

**Cause:** Chakra is installed into the container's site-packages by
`scripts/compile.sh`, so an existing container keeps whatever version
was installed when it was first built. After pulling a change to the
converter or to the trace format, that stale copy can no longer read
the traces the simulator writes.

**Fix:** reinstall Chakra inside the simulator container:

```bash
cd astra-sim/extern/graph_frontend/chakra && pip3 install .
```

Rerunning `./scripts/compile.sh` does the same thing along with the
C++ rebuild.

## Container name already in use

**Symptom:**

```text
docker: Error response from daemon: Conflict. The container name
"/servingsim_docker" is already in use by container "abc123..."
```

**Cause:** A previous run left the container around.

**Fix:** either re-attach or remove and recreate.

```bash
# re-attach to existing
docker start -ai servingsim_docker

# or wipe and recreate
docker rm -f servingsim_docker
./scripts/docker-sim.sh
```

Same idea for `vllm_docker`.

## Missing profile data

**Symptom:** Running the simulator with a hardware / model
combination that doesn't have profile data:

```text
FileNotFoundError: Profile variant folder not found:
../profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16-kvfp8. Run the
profiler with matching --dtype / --kv-cache-dtype, or pick an existing
variant under ../profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B
```

**Cause:** The `(hardware, model, dtype, kv_cache_dtype)` tuple has no
profiled bundle. The check is on the **variant folder**, so the message
names the directory rather than a specific CSV. `ls` the parent path it
prints to see which variants you do have.

Note that the variant is *derived*, not chosen: `--kv-cache-dtype fp8`
appends `-kvfp8`, and `--dtype` (or the model config's `torch_dtype`
when you omit it) supplies the prefix. So this fires when you flip a
precision flag without a matching profile run, not only when the
hardware is new.

Two neighbouring errors with different fixes:

```text
FileNotFoundError: meta.yaml missing at <variant>/meta.yaml. Re-run the
profiler to produce it.
```

The folder exists but the bundle is incomplete — usually a profile run
that was interrupted.

```text
FileNotFoundError: Architecture yaml not found for model_type='gemma2'
at profiler/models/gemma2.yaml. Add profiler/models/gemma2.yaml
describing the architecture.
```

The model's family has no catalog yet. See
**[Adding a model architecture](/docs/profiler/adding-model-architecture)**.

**Fix:** either

- pick a hardware / model / precision combo that's already profiled
  (`ls profiler/perf/`), or
- run the **[Profiler](/docs/profiler/overview)** to generate the
  missing bundle yourself.

## `--max-num-batched-tokens` warning at startup

**Symptom:**

```text
max-num-batched-tokens=4096 exceeds profiled 2048 for
RTXPRO6000/meta-llama/Llama-3.1-8B/bf16; attention/dense lookups will
extrapolate
```

There is a matching one for the sequence cap:

```text
max-num-seqs=256 exceeds profiled 128 for
RTXPRO6000/meta-llama/Llama-3.1-8B/bf16; per-sequence lookups will
extrapolate
```

**Cause:** You're running the simulator past the bounds the profiler
swept, taken from `meta.yaml::engine_effective`. Latency lookups
linearly extrapolate past the measured range.

Both are emitted **once per `(hardware, model, variant)`**, not once per
iteration, so seeing them a single time does not mean it happened only
once. And they are silent when `meta.yaml` has no
`engine_effective` entry for the field, so a hand-authored bundle gets
no warning at all.

**Fix:**

- For best accuracy, re-profile at the higher
  `--max-num-batched-tokens` (`MAX_NUM_BATCHED_TOKENS=4096
  ./profiler/profile.sh`).
- Or stay at the profiled bound. Extrapolation is usually fine for
  small overshoots; large ones can drift.

## Simulator stuck / very slow on big workloads

**Symptom:** Simulation runs but takes much longer than expected,
especially with MoE + EP or large prefix caches.

**Common causes & fixes:**

- **Block-copy disabled.** For MoE, keep `--enable-block-copy`
  on (the default). It replays one transformer block's trace across
  every layer instead of re-computing routing per layer. Safe with
  `--expert-routing-policy BALANCED` (default, deterministic);
  `RR`/`RAND` average out per-layer variance.
- **Verbose logging.** `--log-level DEBUG` writes a lot. Drop to
  `--log-level INFO` or `WARNING`.
- **`--log-interval` too small.** Setting it to `0.1` makes the
  logger run every 100 ms; raise to `1.0` (default) or higher.

## Out of memory inside the vLLM container

**Symptom:** Profiler crashes with CUDA OOM partway through the
attention sweep.

**Fix:** lower `MAX_NUM_BATCHED_TOKENS` in `profiler/profile.sh`,
or skip the heavy categories with environment variables (see
[Profiler → Running](/docs/profiler/running)).

## Still stuck?

- **GitHub Issues:** [casys-kaist/LLMServingSim/issues](https://github.com/casys-kaist/LLMServingSim/issues)
- **Discussions:** [casys-kaist/LLMServingSim/discussions](https://github.com/casys-kaist/LLMServingSim/discussions)

When you file a bug, please include:

1. The exact command you ran
2. The full error output
3. Your OS, Docker version, NVIDIA driver, GPU model
4. Whether you're inside the simulator container or the vLLM
   container (or bare metal)

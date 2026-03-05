# LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure

## Current Release: **v1.0.0** (2026-02-25)

### Highlights

- `llm_profile` extended to support MoE architectures (Mixtral-8x7B, Phi-mini-MoE) and
  TPU-v6e-1; scikit-learn-based attention latency predictor replaces tabular lookup for improved
  accuracy across varying batch sizes and sequence lengths
- Multi-instance simulation with configurable request routing (Round Robin, Random, Custom)
  and Prefill/Decode (P/D) disaggregation across separate prefill and decode instances
- MoE simulation with expert parallelism across NPUs and expert offloading to CPU or CXL memory,
  with configurable expert routing policies (Round Robin, Random, Fast, Custom)
- Prefix caching via RadixAttention with optional second-tier prefix pool across CPU and CXL
  memory (`--enable-prefix-caching`, `--enable-prefix-sharing`, `--prefix-storage`)
- Sub-batch interleaving to overlap XPU and PIM computation
  (`--enable-sub-batch-interleaving`)
- Power and energy modeling per node covering NPU, CPU, DRAM, interconnect, NIC, and storage
- CXL memory expansion support with configurable multi-tier memory bandwidth and latency
- Per-request latency metrics: TTFT, TPOT, and ITL with p99 percentile reporting

See full changelog [here](CHANGELOG.md).

## Build LLMServingSim

### 1. Git clone

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
```

### 2. Run Docker

This will configure and run the Docker environment. See `docker.sh` for details.

```bash
./docker.sh
```

### 3. Build ASTRA-Sim and Chakra

This will compile ASTRA-Sim (analytical backend) and install Chakra. See `compile.sh` for details.

```bash
./compile.sh
```

## Run LLMServingSim

### 1. Set input configurations

All configurations for LLMServingSim are generated automatically by
`inference_serving/config_builder.py` from a `cluster_config` file.

The `cluster_config` file specifies node topology, instance layout, hardware type, memory
hierarchy, and interconnect parameters. It also supports per-layer placement rules for weights,
KV cache, and experts, as well as PIM-enabled device configuration.

**Config paths:**
- Cluster config: `cluster_config/{config_name}.json`
- Logical topology config **(ns3 backend only)**: `astra-sim/inputs/logical_topology/{topology_name}.json`

**Dataset path:**
- Dataset: `dataset/{dataset_name}.jsonl`
- Runtime-generated traces: `astra-sim/inputs/trace/`

See `cluster_config/` for example configurations and `cluster_config/README.md` for the
configuration format reference.

### 2. Run LLMServingSim

Test run:

```bash
python main.py \
    --cluster-config 'cluster_config/single_node_single_instance.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/sharegpt_req100_rate10_llama.jsonl' \
    --output 'output/example_single_run.csv' \
    --num-req 100 --log-interval 1.0
```

See `run.sh` for additional examples covering multi-instance, P/D disaggregation, MoE,
prefix caching, CXL memory, PIM, power modeling, and sub-batch interleaving:

```bash
./run.sh
```


## Parameters of `main.py`

The current version supports the following models and hardware:

**Models:** `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-70B`,
`microsoft/Phi-mini-MoE-instruct`, `mistralai/Mixtral-8x7B-v0.1`

**Hardware:** `A6000`, `H100`, `TPU-v6e-1`

New models and hardware can be added using the provided profiler. See
[Adding a New Model & Hardware](#adding-a-new-model--hardware).

| Parameter | Default | Description |
| --- | --- | --- |
| `--cluster-config` | `single_node_single_instance.json` | Node- and instance-level configuration |
| `--max-batch` | `0` | Maximum batch size; `0` means no limit |
| `--max-num-batched-tokens` | `2048` | Maximum tokens processed per iteration |
| `--fp` | `16` | Floating-point precision in bits |
| `--request-routing-policy` | `RR` | Request routing across instances (`RR`, `RAND`, `CUSTOM`) |
| `--expert-routing-policy` | `FAST` | Expert token routing for MoE (`RR`, `RAND`, `FAST`, `CUSTOM`) |
| `--enable-prefix-caching` | `False` | Enable prefix caching via RadixAttention |
| `--enable-prefix-sharing` | `False` | Enable second-tier prefix cache pooling |
| `--prefix-storage` | `None` | Storage tier for the second-tier prefix pool (`None`, `CPU`, `CXL`) |
| `--enable-local-offloading` | `False` | Enable weight offloading to local memory |
| `--enable-attn-offloading` | `False` | Enable attention computation offloading to PIM |
| `--enable-sub-batch-interleaving` | `False` | Enable sub-batch interleaving for XPU/PIM overlap |
| `--enable-attn-prediction` | `False` | Enable real-time attention latency prediction |
| `--prioritize-prefill` | `False` | Prioritize prefill requests in scheduling |
| `--block-size` | `16` | KV cache block size in tokens |
| `--dataset` | `None` | Path to `.jsonl` dataset; if `None`, add requests manually in `main.py` |
| `--output` | `None` | Path for per-request CSV output; if `None`, stdout only |
| `--gen` | `True` | Set to `False` to skip the initiation (prefill) phase |
| `--num-req` | `100` | Number of requests to simulate |
| `--log-interval` | `0.5` | Throughput logging interval in seconds |
| `--log-level` | `WARNING` | Logging verbosity (`WARNING`, `INFO`, `DEBUG`) |
| `--network-backend` | `analytical` | Network simulation backend (`analytical`, `ns3`) |

## Outputs of `main.py`

### 1. Standard output

The simulator reports runtime information through a configurable logger. It logs which requests
are processed at each iteration and periodically reports throughput, memory usage, and power
consumption.

Adjusting `--log-level` to `INFO` or `DEBUG` enables more detailed output, including per-layer
memory load and store activity.

### 2. Output file

`{output_path}.csv` contains per-request latency metrics. An example is provided at
`output/example_run.csv`.

## Adding a New Model & Hardware

### 1. Build a performance model

LLMServingSim uses the PyTorch-based profiler in `llm_profile/` to generate per-layer latency,
attention latency, and power models for a given hardware target. Once profiling is complete,
create a cluster config referencing the new hardware name and run `main.py` as usual.

See [`llm_profile/README.md`](llm_profile/README.md) for full profiling instructions.

### 2. Modify simulator functions (optional)

The current version supports Llama-based model architectures. Models that deviate from this
architecture may require modifications to the following:

**`inference_serving/memory_model.py`** — functions `calculate_sizes` and `get_weight`

`calculate_sizes` computes input, weight, and output tensor sizes for each layer type.
`get_weight` aggregates total model size from `calculate_sizes`.
Modify these according to the target model architecture.

**`inference_serving/trace_generator.py`** — function `synthesize_trace`

This function constructs the per-iteration execution trace by stacking layers according to the
model architecture. When modifying it, ensure:

- The ATTENTION layer is correctly separated per request
- The output size of layer *i* matches the input size of layer *i+1*
- ALLREDUCE operations are correctly placed for tensor-parallel synchronization

## Evaluation

The [`evaluation/`](evaluation/) directory contains the artifact evaluation flow for Figures 5 to 10
from the paper. It includes figure-specific shell scripts, plotting code, parsers, processed
reference inputs, and preserved example outputs under `evaluation/artifacts/`.

Before running artifact evaluation, complete the setup steps above (`./docker.sh` and
`./compile.sh`) and run the evaluation commands inside that environment.

Enter `evaluation/` first:

```bash
cd evaluation
```
Run an individual figure:

```bash
bash figure_5.sh
bash figure_6.sh
bash figure_7.sh
bash figure_8.sh
bash figure_9.sh
bash figure_10.sh
```

To reproduce the full evaluation set in one pass:

```bash
bash run_all.sh
```

To compare generated parsed outputs against preserved artifact snapshots:

```bash
# Compare all figures (5-10)
bash compare.sh
# Compare one figure
bash compare.sh 5
# Compare multiple selected figures
bash compare.sh 5 7 9
# Equivalent single-figure form
bash compare.sh figure_5
```

For visual validation, compare generated PDFs with the corresponding `*_ref.pdf` files in each
figure folder.

See [`evaluation/README.md`](evaluation/README.md) for detailed folder structure, reference-comparison guidance, and per-figure notes.

## Publications

**ISPASS 2026**  
*LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure*  
Jaehong Cho<sup>\*</sup>, Hyunmin Choi<sup>\*</sup>, Guseul Heo, Jongse Park (KAIST) [[Paper]]() (To Appear)                                                                                                                                                       
<sup>\*</sup>Equal contribution                                                                                                                                                                                                                

**CAL 2025**  
*LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure*  
Jaehong Cho, Hyunmin Choi, Guseul Heo, Jongse Park (KAIST)  [[Paper]](https://doi.org/10.1109/LCA.2025.3628325)

**IISWC 2024**  
*LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale*  
Jaehong Cho, Minsu Kim, Hyunmin Choi, Guseul Heo, Jongse Park (KAIST)  [[Paper]](https://doi.org/10.1109/IISWC63097.2024.00012)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12803583.svg)](https://doi.org/10.5281/zenodo.12803583)

## Citation

If you use LLMServingSim in your research, please cite:

```bibtex
@ARTICLE{11224567,
    author={Cho, Jaehong and Choi, Hyunmin and Park, Jongse},
    journal={IEEE Computer Architecture Letters},
    title={{LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving
            Techniques in LLM Infrastructure}},
    year={2025},
    volume={24},
    number={02},
    pages={361-364},
    doi={10.1109/LCA.2025.3628325},
    ISSN={1556-6064},
    publisher={IEEE Computer Society},
    address={Los Alamitos, CA, USA},
    month=jul
}

@INPROCEEDINGS{10763697,
    author={Cho, Jaehong and Kim, Minsu and Choi, Hyunmin and Heo, Guseul and Park, Jongse},
    booktitle={2024 IEEE International Symposium on Workload Characterization (IISWC)},
    title={{LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving
            at Scale}},
    year={2024},
    pages={15-29},
    doi={10.1109/IISWC63097.2024.00012}
}
```

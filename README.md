# LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale

## Publication
Paper: [https://doi.org/10.1109/IISWC63097.2024.00012](https://doi.org/10.1109/IISWC63097.2024.00012)

Authors: Jaehong Cho, Minsu Kim, Hyunmin Choi, Guseul Heo, Jongse Park (KAIST)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12803583.svg)](https://doi.org/10.5281/zenodo.12803583)

## Version Information

Please ask for more features in the issue tab or via email.

### Current Version: `v0.2.1`

- Integrated PyTorch Profiler (`llm-profile`) for accurate and intuitive GPU performance analysis
- Users can easily add models from Hugging Face
- Customizable model configs
- Unified naming convention for consistency and readability

### Previous Versions: 
### `v0.2.0`

- Upgrade ASTRA-Sim and Chakra to the latest version

### `v0.1.0`

- Support GPU with a performance model (TensorRT-LLM)
- Auto config generator (network and memory)
- Verbose option for more detailed log
- More metrics (queuing_delay, TTFT, TPOT)
- Refactored code structure for readability

### `artifact`

- IISWC Artifact for "LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale"
- For the NPU simulator version, please refer to this version

## Build LLMServingSim

### 1. Git clone
```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
```

### 2. `Conda` install (optional)
Conda can be downloaded from the following [link](https://repo.anaconda.com/archive/).
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```

### 3. Install dependency (tested in python 3.9, GCC, G++ 7.5.0)

### Using `conda` environment.yml (Recommended)
```bash
conda env create -p ./env -f ./environment.yml
conda activate ./env
```

### Clean `conda` install 
```bash
conda create -n env_name python=3.9
conda activate env_name
conda install -c conda-forge gcc_linux-64=7.5.0 gxx_linux-64=7.5.0 libprotobuf=3.6.1 cmake=3.22
```

### 4. Build ASTRA-Sim, Chakra

Common issues while building ASTRA-Sim. If error regarding `version of protoc` happens see [here](#common-errors).

```bash
cd astra-sim
./build/astra_analytical/build.sh
cd extern/graph_frontend/chakra
pip install .
cd ../../../..
```

## Run LLMServingSim

### 1. Set input configurations

Now network and remote memory config are automatically set by `inference_serving/config_generator.py`.

Simply passing arguments to `main.py` is enough.
See `inference_serving/config_generator.py` for more details.

**Config & Dataset Path:**

- Network config path: `astra-sim/inputs/network/network.yml`
- System config path: `astra-sim/inputs/system/system.json`
- Remote(Host) memory config path: `astra-sim/inputs/remote_memory/{config_name}.json`
- Dataset path: `dataset/{dataset_name}.tsv`
- Runtime generated traces are located at `astra-sim/inputs/trace`

### 2. Run LLMServingSim

Test Run

```bash
python main.py --model_name 'meta-llama/Llama-3.1-8B-Instruct' --hardware 'RTX3090' --npu_num 1 --npu_group 1 --npu_mem 40 \
    --remote_bw 512 --link_bw 256 --fp 16 --block_size 4 \
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --output 'output/example_run.csv' \
    --verbose --req_num 10
```
or simply use
```bash
./run.sh
```

## Parameters of `main.py`

The current version only supports `meta-llama/Llama-3.1-8B-Instruct` and `RTX3090`. 

You can easily add new models and hardware using the provided PyTorch Profiler.

**Instructions for adding a new model and hardware are located [here](#adding-a-new-model--hardware).**


| Parameters | Supporting Options | Default Value | Notes |
| --- | --- | --- | --- |
| model_name | 'meta-llama/Llama-3.1-8B-Instruct' | 'meta-llama/Llama-3.1-8B-Instruct' |  |
| hardware | 'RTX3090' | 'RTX3090' |  |
| npu_num  | Integer | 16 |  |
| max_batch | Integer | 0 | 0: no limit |
| npu_group | Integer | 1 |  |
| npu_mem | Integer | 40 | GB |
| local_bw | Integer | 1024 | GB/s |
| remote_bw | Integer | 512 | GB/s |
| link_bw | Integer | 256 | GB/s |
| fp | Integer | 16 | bits |
| block_size | Integer | 8 |  |
| dataset | Dataset Path | None | None: manually add requests in main.py |
| output | Output CSV Path | None | None: no csv output only stdout |
| gen | Flag | False | Skip initiation phase On/Off |
| req_num | Integer | 100 |  |
| log_interval | Float | 0.5 | Throughput log interval (s) |
| verbose | Flag | False |  |

## Outputs of `main.py`

### 1. Standard output

The standard output shows which requests are being processed in each iteration of the simulator and displays the measured throughput at regular intervals. 

Additionally, it provides a summary of the simulation at the end.

With `--verbose` option, the log includes more specific information including memory load and store.

### 2. Output file

`{output_filename}.csv` contains the simulation result of each request.

You can find an example in `output/example_run.csv`.

## Adding a New Model & Hardware

### 1. Make a new performance model

We use the PyTorch Profiler to generate performance models.

To profile a new model, follow the instructions provided in the [llm-profile](https://github.com/casys-kaist/llm-profile) repository.

Once profiling is complete, you can easily run LLMServingSim with the following steps:

1. Place the `<Hardware>.csv` file generated by `llm-profile` into the `perf_model` directory.

2. Add a corresponding model config file to the `model_configs` directory, named after your model.
Refer to the existing OPT or Llama configurations for the required format.

3. Run `main.py` with the new model and hardware setup.

If you prefer not to use `llm-profile`, you can measure the latency of each layer using another tool.
Make sure to follow the format of a performance model in `perf_model/RTX3090.csv`.

### 2. Modify functions (optional)

The current version supports OPT and Llama model architectures. If the model architecture does not follow those two, some codes of LLMServingSim should be modified.

1. `inference_serving/memory_model.py`: function `calculate_sizes` & `get_weight`

`calculate_sizes` function calculates the input, weight, and output tensor size for each specific layer.
Change this function according to the model architecture.

`get_weight` function calculates the total model size by retrieving weights from `calculateSizes`.
Also, change this function according to the model architecture.

2. `inference_serving/generate_trace.py`: function `synthsize_trace`

This is the main function that generates trace for each iteration. 
It uses `calculated_sizes` to retrieve input, weight, and output tensor size for each layer.
Then, it stacks layers in the trace according to the model architecture.

While changing this function, there are three important things.

- Make sure ATTENTION layer is well separated for each request

- Make sure ith layer output and i+1th layer input size are matched

- Make sure ALLREDUCE operation is well placed for synchronization

We provide a function to test your trace generation. See `trace_test/` for more details.


## Citation
If you use LLMServingSim for your research, please cite our paper:

```
@INPROCEEDINGS{10763697,
  author={Cho, Jaehong and Kim, Minsu and Choi, Hyunmin and Heo, Guseul and Park, Jongse},
  booktitle={2024 IEEE International Symposium on Workload Characterization (IISWC)}, 
  title={LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale}, 
  year={2024},
  volume={},
  number={},
  pages={15-29},
  keywords={Technological innovation;Program processors;Simulation;Large language models;Heuristic algorithms;Redundancy;Software algorithms;Inference algorithms;Software;System analysis and design;Large language model (LLM);Inference serving;Simulation infrastructure;Neural processing unit (NPU);Processing-in-memory (PIM);Heterogeneous system},
  doi={10.1109/IISWC63097.2024.00012}}
```

## Common Errors

### Error Example
If your error is similar to this, you can use the below solution.
```bash
/home/<user>/LLMServingSim/astra-sim/extern/graph_frontend/chakra/et_def/et_def.pb.h:17:2: error: #error This file was generated by an older version of protoc which is
   17 | #error This file was generated by an older version of protoc which is
      |  ^~~~~
/home/<user>/LLMServingSim/astra-sim/extern/graph_frontend/chakra/et_def/et_def.pb.h:18:2: error: #error incompatible with your Protocol Buffer headers. Please
   18 | #error incompatible with your Protocol Buffer headers.  Please
      |  ^~~~~
/home/<user>/LLMServingSim/astra-sim/extern/graph_frontend/chakra/et_def/et_def.pb.h:19:2: error: #error regenerate this file with a newer version of protoc.
   19 | #error regenerate this file with a newer version of protoc.
      |  ^~~~~
```

### Method 1: Setting Environment Variables

This method explicitly sets the conda environment for CMake to use.

1. **Activate the Conda Environment**:
First, activate the desired conda environment.
    
    ```bash
    conda activate your_env_name
    ```
    
2. **Set the CMAKE_PREFIX_PATH Environment Variable**:
Add the path of the activated conda environment to the `CMAKE_PREFIX_PATH` environment variable.
    
    ```bash
    export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
    ```
    

### Method 2: Setting the Activation Script

1. **Activate the Conda Environment**:
First, activate the conda environment you want to modify.
    
    ```bash
    conda activate your_env_name
    ```
    
2. **Navigate to the Environment's Activation Script Directory**:
The activation scripts are located in the `etc/conda/activate.d` directory within your conda environment. If this directory does not exist, create it along with the deactivation directory.
    
    ```bash
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
    ```
    
3. **Create and Edit the Activation Script**:
Create a script named `set_cmake_prefix.sh` to set the `CMAKE_PREFIX_PATH` when the environment is activated.
    
    ```bash
    nano $CONDA_PREFIX/etc/conda/activate.d/set_cmake_prefix.sh
    ```
    
    Add the following content to this file:
    
    ```bash
    #!/bin/bash
    export OLD_CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
    export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
    ```
    
4. **Create and Edit the Deactivation Script**:
Create a script named `unset_cmake_prefix.sh` to reset the `CMAKE_PREFIX_PATH` when the environment is deactivated.
    
    ```bash
    nano $CONDA_PREFIX/etc/conda/deactivate.d/unset_cmake_prefix.sh
    ```
    
    Add the following content to this file:
    
    ```bash
    #!/bin/bash
    export CMAKE_PREFIX_PATH=$OLD_CMAKE_PREFIX_PATH
    unset OLD_CMAKE_PREFIX_PATH
    ```
    
5. **Set Script Permissions**:
Ensure the scripts are executable.
    
    ```bash
    chmod +x $CONDA_PREFIX/etc/conda/activate.d/set_cmake_prefix.sh
    chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/unset_cmake_prefix.sh

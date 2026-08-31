<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/static/img/llmservingsim_full_primary_dark_transparent.png">
    <img alt="LLMServingSim" src="docs/static/img/llmservingsim_full_primary_transparent.png" width="70%">
  </picture>
</p>

<h3 align="center">
A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure
</h3>

<p align="center">
| <a href="https://llmservingsim.ai"><b>Website</b></a> | <a href="https://llmservingsim.ai/docs/getting-started/overview"><b>Documentation</b></a> | <a href="https://llmservingsim.ai/docs/contributor/welcome"><b>Contribute</b></a> | <a href="https://llmservingsim.ai/contact"><b>Contact</b></a> | <a href="https://llmservingsim.ai/changelog"><b>Changelog</b></a> |
</p>

## Latest News

- [2026/08] Simulation is **~11x faster** with byte-identical results — the four `bench/examples` runs go 16m 40s → 1m 26s. ([#67](https://github.com/casys-kaist/LLMServingSim/pull/67))
- [2026/08] **TP / PP / EP / DP** run in every combination, checked by 58 recorded scenarios (`serving/validate.sh`). ([#68](https://github.com/casys-kaist/LLMServingSim/pull/68))
- [2026/08] **RTX 4090** joins the profile library — within **1%** of a real vLLM run on TTFT / TPOT / latency. ([#59](https://github.com/casys-kaist/LLMServingSim/pull/59))

## About

LLMServingSim is a cycle-level simulator for LLM serving infrastructure. It pairs a Python frontend that mirrors vLLM's continuous-batching scheduler with the ASTRA-Sim C++ analytical network backend, and drives both from per-hardware latency data captured by a vLLM-based layerwise profiler. The result is a unified environment for studying heterogeneous accelerators, disaggregated memory tiers (CPU / CXL / PIM), MoE routing, speculative decoding, and multi-instance parallelism (TP / PP / EP / DP) end-to-end — across dense, MoE, sparse-attention and hybrid linear-attention model families.

## Getting Started

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
./scripts/docker-sim.sh           # launch the simulator container
./scripts/compile.sh              # build ASTRA-Sim + Chakra
./serving/run.sh                  # run the example simulations
```

For installation details, container choices, configuration layout, CLI
flags, and the full set of example workloads, see the
[documentation](https://llmservingsim.ai/docs/getting-started/overview).

## Publications

**ISPASS 2026**  
*LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure*  
Jaehong Cho<sup>\*</sup>, Hyunmin Choi<sup>\*</sup>, Guseul Heo, Jongse Park (KAIST) [[Paper]](https://doi.org/10.1109/ISPASS69572.2026.00012)  
<sup>\*</sup>Equal contribution  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18879965.svg)](https://doi.org/10.5281/zenodo.18879965)

**CAL 2025**  
*LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure*  
Jaehong Cho, Hyunmin Choi, Jongse Park (KAIST)  [[Paper]](https://doi.org/10.1109/LCA.2025.3628325)

**IISWC 2024**  
*LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale*  
Jaehong Cho, Minsu Kim, Hyunmin Choi, Guseul Heo, Jongse Park (KAIST)  [[Paper]](https://doi.org/10.1109/IISWC63097.2024.00012)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12803583.svg)](https://doi.org/10.5281/zenodo.12803583)

## Citation

If you use LLMServingSim in your research, please cite:

```bibtex
@INPROCEEDINGS{11527300,
    author={Cho, Jaehong and Choi, Hyunmin and Heo, Guseul and Park, Jongse},
    booktitle={2026 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)}, 
    title={{LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure}}, 
    year={2026},
    pages={1-14},
    doi={10.1109/ISPASS69572.2026.00012}
}

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

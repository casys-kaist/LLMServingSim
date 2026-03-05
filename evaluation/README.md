# Evaluation README

## Overview

This directory contains the artifact evaluation flow for Figures 5 to 10 in
LLMServingSim 2.0 paper. The working `figure_*` folders are the ones you run and edit. They
contain checked-in configs, reference inputs, plotting code, and reference PDFs. Runtime outputs
such as `logs/`, `results/`, and `parsed/` are created in those folders when you run each figure
script. The `artifacts/` folder keeps frozen copies of previously generated outputs for
comparison.

For figure-specific details such as the figure goal, axis definitions, reference inputs, generated
TSV files, and expected PDFs, see the `README.md` inside each `figure_X/` folder.

## Folder Guide

- `fonts/`: local fonts used by the plotting scripts.
- `parser/`: log-to-TSV parsers for throughput, power, memory, latency, simulation time, and
  energy breakdowns.
- `figure_5.sh` to `figure_10.sh`: per-figure reproduction scripts.
- `figure_5/` to `figure_10/`: figure-specific configs, reference inputs, plotting scripts, and
  `*_ref.pdf` visual references.
- `artifacts/`: preserved outputs from prior runs, including generated `logs/`, `results/`,
  `parsed/`, and figure PDFs.

Within each working `figure_*` folder:

- `config/`: cluster configs used by the figure script.
- `reference/`: processed real-system or prior-work data used as the comparison baseline.
- `figure_X.py`: plotting code.
- `figure_X_ref.pdf` or `figure_Xa_ref.pdf`: reference PDF for visual comparison.

`figure_5/` is split by hardware, so its checked-in inputs are stored under `A6000/` and `H100/`.

## Structure

```text
evaluation/
├── README.md
├── fonts/
├── parser/
├── run_all.sh
├── figure_5.sh ... figure_10.sh
├── figure_5/
│   ├── A6000/{config,reference}/
│   ├── H100/{config,reference}/
│   ├── figure_5.py
│   └── figure_5_ref.pdf
├── figure_6/
│   ├── config/
│   ├── reference/
│   ├── figure_6.py
│   └── figure_6a_ref.pdf, figure_6b_ref.pdf, figure_6c_ref.pdf
├── figure_7/ ... figure_10/
└── artifacts/
    └── figure_5/ ... figure_10/
```

When you run a figure script, it creates figure-local `logs/`, `results/`, and `parsed/`
subdirectories next to the checked-in inputs.

## Running Evaluation

Run one figure from the `evaluation/` folder:

```bash
bash figure_5.sh
```

Run all figure flows:

```bash
bash run_all.sh
```

The scripts already call `main.py`, the required parsers, and the plotting script for that figure.
Most dataset paths are declared near the top of each shell script. If your datasets are stored in a
different location, update those variables before running.

## Comparing Against Reference

### 1) Script Comparison

Use the compare script from the `evaluation/` folder:

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

This script compares generated parsed TSV outputs against the preserved snapshot in
`evaluation/artifacts/` for Figures 5 to 10.

Figure 8 note: `*_sim_time.tsv` is always checked and reported, but simulation-time differences are
treated as expected hardware-dependent variation and do not fail the compare result.

### 2) Visual Comparison

For visual validation, compare generated PDFs with the corresponding `*_ref.pdf` files in each
figure folder.

For figure-specific compare targets and interpretation, see each `figure_X/README.md`.

## Figure Map

- [`figure_5/README.md`](figure_5/README.md): GPU throughput validation against RTX A6000 and H100.
- [`figure_6/README.md`](figure_6/README.md): server power traces and energy breakdown.
- [`figure_7/README.md`](figure_7/README.md): memory usage and prefix-hit validation.
- [`figure_8/README.md`](figure_8/README.md): comparison against prior LLM serving simulators.
- [`figure_9/README.md`](figure_9/README.md): TPU throughput validation and latency error table.
- [`figure_10/README.md`](figure_10/README.md): GPU-only vs. GPU+PIM case study.

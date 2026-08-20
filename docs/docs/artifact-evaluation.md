---
title: Artifact Evaluation
sidebar_position: 7
description: Reproduce the figures and results from published LLMServingSim papers
---

# Artifact Evaluation

Each LLMServingSim paper that ships an artifact lives on its own
branch, frozen at the state submitted to the artifact-evaluation
committee. This page is the entry point for reviewers and readers
who want to reproduce the published figures end to end.

> **Heads up:** the artifact branches are frozen for reproducibility.
> Don't open PRs against them; new development goes to `main`. See
> **[For Contributors → PR workflow](/docs/contributor/pr-workflow)**.

## Available artifacts

| Paper | Venue | Branch | Reproduces |
| --- | --- | --- | --- |
| **LLMServingSim 2.0** | ISPASS 2026 | [`ispass26-artifact`](https://github.com/casys-kaist/LLMServingSim/tree/ispass26-artifact) | Figures 5–10 |
| **LLMServingSim** | IISWC 2024 | [`iiswc24-artifact`](https://github.com/casys-kaist/LLMServingSim/tree/iiswc24-artifact), also released on [Zenodo](https://doi.org/10.5281/zenodo.12803583) | Original paper figures |

The CAL 2025 entry shares the ISPASS 2026 codebase and does not have
its own artifact branch.

## ISPASS 2026 — `ispass26-artifact`

*Cho, Choi, Heo, Park. "LLMServingSim 2.0: A Unified Simulator for
Heterogeneous and Disaggregated LLM Serving Infrastructure", ISPASS
2026. [Zenodo DOI](https://doi.org/10.5281/zenodo.18879965).*

The branch reproduces **Figures 5 through 10** of the paper, plus
the supporting throughput / power / memory / latency parsers under
`evaluation/`.

> The artifact pre-dates the v1.1.0 directory restructure and the
> vLLM-based profiler rewrite, so on `ispass26-artifact` you'll see
> the older layout (`cluster_config/`, `dataset/`, `output/`,
> `inference_serving/`, `main.py`) instead of the `serving/` /
> `configs/` / `workloads/` / `outputs/` paths that the rest of this
> site documents. Follow the branch's own README, not this site's
> Getting Started, while you're inside the artifact.

### 1. Switch to the artifact branch

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim
git checkout ispass26-artifact
```

If you already cloned, just `git checkout ispass26-artifact` and
`git submodule update --init --recursive` to pick up the pinned
ASTRA-Sim submodule.

### 2. Set up the environment

The artifact ships its own Docker launcher and build script (rather
than the two-container split on `main`):

```bash
./docker.sh        # launches the artifact's simulator container
./compile.sh       # builds ASTRA-Sim + Chakra inside the container
```

`docker.sh` mounts the repo at `/app/LLMServingSim`. Run all
subsequent commands from that working directory inside the
container.

### 3. Reproduce a single figure

Each figure has its own driver script under `evaluation/`:

```bash
cd evaluation

bash figure_5.sh        # Hardware coverage (A6000, H100)
bash figure_6.sh        # Multi-instance + P/D disaggregation
bash figure_7.sh        # MoE expert parallelism + offloading
bash figure_8.sh        # Prefix caching across CPU / CXL pools
bash figure_9.sh        # CXL memory expansion
bash figure_10.sh       # Power and energy modeling
```

Each script writes intermediate logs to `evaluation/figure_X/logs/`,
parsed numbers to `evaluation/figure_X/parsed/`, and the final PDF
next to the script.

### 4. Reproduce everything

```bash
cd evaluation
bash run_all.sh
```

This is the same as running all six `figure_*.sh` scripts in
sequence. Expect this to take a few hours on a single workstation;
each figure runs many simulator invocations.

### 5. Compare against the preserved snapshot

Frozen reference outputs live under `evaluation/artifacts/`. To
compare your generated parsed output against those snapshots:

```bash
# Compare every figure
bash compare.sh

# Compare one figure
bash compare.sh 5

# Compare a subset
bash compare.sh 5 7 9
```

For visual confirmation, diff the regenerated `figure_X.pdf` against
the committed `figure_X_ref.pdf` (or `figure_Xa_ref.pdf` for
multi-panel figures) in each folder.

### Per-figure details

Each `evaluation/figure_X/` folder has its own `README.md` with
the figure's goal, axis definitions, reference inputs, expected
TSV files, and the PDF naming convention. Start there if a figure
fails to reproduce or numbers drift outside the comparison
tolerance.

The umbrella reference is
[`evaluation/README.md`](https://github.com/casys-kaist/LLMServingSim/blob/ispass26-artifact/evaluation/README.md),
which lists the parsers, fonts, and folder layout used across all
figures.

## When reproduction fails

A few common cases:

1. **`compile.sh` errors on the submodule**: rerun
   `git submodule update --init --recursive` from the host and try
   again. The submodule pin is part of the artifact.
2. **`figure_X.sh` runs but the parsed output doesn't match**: check
   the corresponding `evaluation/figure_X/README.md` for the
   tolerance band the artifact was certified at; small drift on the
   exact wattage or latency value is expected as long as the
   qualitative trend matches the reference PDF.
3. **A specific simulator command fails on the branch but works on
   `main`**: that's expected. The artifact is frozen at the paper's
   submission state; bug fixes and new features that landed on
   `main` afterward are not back-ported.
4. **You need to extend the artifact** (e.g., add a new GPU to
   Figure 5): we recommend doing the work on `main` instead and
   citing the new result separately. The artifact branch should
   stay reproducible against the paper.

## Reaching the artifact authors

For artifact-specific questions (reproduction failures, environment
setup, requesting a missing reference output), email the main
contributors:

- [jhcho@casys.kaist.ac.kr](mailto:jhcho@casys.kaist.ac.kr?cc=hmchoi@casys.kaist.ac.kr)
- [hmchoi@casys.kaist.ac.kr](mailto:hmchoi@casys.kaist.ac.kr?cc=jhcho@casys.kaist.ac.kr)

CC both whenever possible. See the
[Contact page](/contact) for the full list of channels.

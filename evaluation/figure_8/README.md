# Figure 8

## What This Figure Shows

Figure 8 in the LLMServingSim 2.0 paper compares LLMServingSim 2.0 with prior LLM
serving simulators. It is split into:

- `figure_8a.pdf`: accuracy-oriented comparison for `TPS`, `TTFT`, and `TPOT`, normalized to vLLM
- `figure_8b.pdf`: simulation time comparison

The evaluated scenarios are `SD`, `MD`, `PDD`, and `SM`.

## Important Notice

**Simulation time in Figure 8b depends on the performance of the machine that runs the evaluation.**  
**The absolute `simulation_time_s` values can differ across CPUs, servers, and runtime
environments, even when the configs and datasets are the same.**

Because of this, evaluators should not expect the exact same raw simulation-time numbers unless the
run is performed on the same hardware and software environment as the preserved artifact outputs.  
Use the reference files and PDFs to compare overall behavior and relative trends, but treat the
absolute simulation-time values as hardware-dependent.

## Inputs And Outputs

Checked-in reference measurements are stored in:

- `reference/*_latency.tsv`
- `reference/*_sim_time.tsv`

These files use a `framework` column and contain rows such as `TokenSim`, `Vidur`, `APEX`,
`LLMServingSim`, and `vLLM`.

The figure script generates LLMServingSim 2.0 outputs in:

- `logs/`
- `results/`
- `parsed/`

The parsed simulator values used for plotting are:

- `parsed/*_latency.tsv`
- `parsed/*_sim_time.tsv`

The generated figure outputs are:

- `figure_8a.pdf`
- `figure_8b.pdf`

The visual targets are `figure_8a_ref.pdf` and `figure_8b_ref.pdf`.

The plotting code merges the parsed LLMServingSim 2.0 values with the checked-in reference tables
at plot time.

## Axes And Interpretation

Figure 8a:

- X axis: scenario groups (`SD`, `MD`, `PDD`, `SM`) with per-group metric blocks (`TPS`, `TTFT`,
  `TPOT`)
- Y axis: normalized metric value relative to `vLLM`

Figure 8b:

- X axis: scenario (`Single Dense`, `Multi Dense`, `Prefill-Decode Disaggregated`, `Single MoE`)
- Y axis: simulation time in seconds

Together, the two plots show both output accuracy and runtime cost relative to prior simulators.

## How To Run

From the `evaluation/` folder:

```bash
bash figure_8.sh
```

To compare your output:

1. Compare the parsed TSV files created in `parsed/` with
   `evaluation/artifacts/figure_8/parsed/`.  
   For automated compare from the evaluation folder, run `bash compare.sh <figure_id>`
   (for example, `bash compare.sh 8`).  
   For all options, see `bash compare.sh --help`.
2. Compare `figure_8a.pdf` and `figure_8b.pdf` with `figure_8a_ref.pdf` and `figure_8b_ref.pdf`.

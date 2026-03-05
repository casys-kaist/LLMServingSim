# Figure 5

## What This Figure Shows

Figure 5 in the LLMServingSim 2.0 paper compares throughput over time between real GPU
systems and LLMServingSim 2.0 using the vLLM serving stack. The figure contains eight subplots:
`A6000` and `H100`, each with `MD`, `SD+PC`, `PDD`, and `SM`.

- `MD`: multi-instance dense serving
- `SD+PC`: single dense serving with prefix caching
- `PDD`: prefill/decode disaggregation
- `SM`: single-instance MoE serving

## Inputs And Outputs

Checked-in reference measurements are stored in:

- `A6000/reference/*_throughput.tsv`
- `H100/reference/*_throughput.tsv`

The figure script generates LLMServingSim outputs in:

- `A6000/logs/`, `A6000/results/`, `A6000/parsed/`
- `H100/logs/`, `H100/results/`, `H100/parsed/`

The parsed simulator values used for plotting are:

- `A6000/parsed/*_throughput.tsv`
- `H100/parsed/*_throughput.tsv`

The generated figure output is:

- `figure_5.pdf`

The visual target is `figure_5_ref.pdf`.

## Axes And Interpretation

- X axis: time in seconds
- Y axis: throughput in tokens per second

Each subplot overlays the processed real-system baseline from `reference/` with the parsed
LLMServingSim result from `parsed/`. The goal is to validate how closely the simulated throughput
trajectory matches the measured one over time.

## How To Run

From the `evaluation/` folder:

```bash
bash figure_5.sh
```

To compare your output:

1. Compare the parsed TSV files created in `A6000/parsed/` and `H100/parsed/` with
   `evaluation/artifacts/figure_5/A6000/parsed/` and
   `evaluation/artifacts/figure_5/H100/parsed/`.  
   For automated compare from the evaluation folder, run `bash compare.sh <figure_id>`
   (for example, `bash compare.sh 5`).  
   For all options, see `bash compare.sh --help`.
2. Compare `figure_5.pdf` with `figure_5_ref.pdf`.

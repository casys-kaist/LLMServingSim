# Figure 9

## What This Figure Shows

Figure 9 in the LLMServingSim 2.0 paper validates LLMServingSim 2.0 against a real TPU
system using the vLLM framework. It consists of:

- `figure_9a.pdf`: throughput over time
- `figure_9b.pdf`: error-rate table for latency-related metrics

## Inputs And Outputs

Checked-in reference measurements are stored in:

- `reference/SD_throughput.tsv`
- `reference/SD_latency.tsv`

The figure script generates LLMServingSim outputs in:

- `logs/`
- `results/`
- `parsed/`

The parsed simulator values used for plotting are:

- `parsed/SD_throughput.tsv`
- `parsed/SD_latency.tsv`

The generated figure outputs are:

- `figure_9a.pdf`
- `figure_9b.pdf`

The visual targets are `figure_9a_ref.pdf` and `figure_9b_ref.pdf`.

`SD_latency.tsv` contains `throughput_tok_s`, `mean_ttft_ms`, `mean_tpot_ms`, and `mean_itl_ms`.

## Axes And Interpretation

Figure 9a:

- X axis: time in seconds
- Y axis: throughput in tokens per second

This plot compares the processed TPU baseline in `reference/SD_throughput.tsv` with the parsed
LLMServingSim throughput trace in `parsed/SD_throughput.tsv`.

Figure 9b:

- Rows: `TPS`, `TPOT`, `ITL`, and `Geomean`
- Value: absolute percentage error between the parsed simulator latency TSV and the reference
  latency TSV

This table summarizes how closely LLMServingSim 2.0 reproduces the measured TPU latency behavior.

## How To Run

From the `evaluation/` folder:

```bash
bash figure_9.sh
```

To compare your output:

1. Compare the parsed TSV files created in `parsed/` with
   `evaluation/artifacts/figure_9/parsed/`.  
   For automated compare from the evaluation folder, run `bash compare.sh <figure_id>`
   (for example, `bash compare.sh 9`).  
   For all options, see `bash compare.sh --help`.
2. Compare `figure_9a.pdf` and `figure_9b.pdf` with `figure_9a_ref.pdf` and `figure_9b_ref.pdf`.

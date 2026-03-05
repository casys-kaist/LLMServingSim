# Figure 7

## What This Figure Shows

Figure 7 in the LLMServingSim 2.0 paper compares memory usage and prefix-hit rate
between a real RTX A6000 deployment and LLMServingSim 2.0. It contains:

- single-instance validation with prefix caching
- multi-instance validation with CPU-shared prefix storage

The generated plot is `figure_7.pdf`. The visual reference is `figure_7_ref.pdf`.

## Inputs And Outputs

Checked-in reference measurements are stored in:

- `reference/SD+PC.tsv`
- `reference/MD+PC+PS_inst0.tsv`
- `reference/MD+PC+PS_inst1.tsv`
- `reference/MD+PC+PS_shared_cpu.tsv`

The figure script generates LLMServingSim outputs in:

- `logs/`
- `results/`
- `parsed/`

The parsed simulator values used for plotting are:

- `parsed/SD+PC.tsv`
- `parsed/MD+PC+PS_inst0.tsv`
- `parsed/MD+PC+PS_inst1.tsv`
- `parsed/MD+PC+PS_shared_cpu.tsv`

The generated figure output is:

- `figure_7.pdf`

The visual target is `figure_7_ref.pdf`.

## Axes And Interpretation

Left column, single instance:

- X axis: time in seconds
- Top Y axis: GPU memory usage in GB
- Bottom Y axis: prefix hit rate in percent

Right column, multi instance:

- X axis: time in seconds
- Top Y axis: GPU or shared CPU memory usage in GB
- Bottom Y axis: prefix hit rate in percent

The solid lines come from the processed real-system traces in `reference/`. The dashed lines come
from the parsed LLMServingSim outputs in `parsed/`.

## How To Run

From the `evaluation/` folder:

```bash
bash figure_7.sh
```

To compare your output:

1. Compare the parsed TSV files created in `parsed/` with
   `evaluation/artifacts/figure_7/parsed/`.  
   For automated compare from the evaluation folder, run `bash compare.sh <figure_id>`
   (for example, `bash compare.sh 7`).  
   For all options, see `bash compare.sh --help`.
2. Compare `figure_7.pdf` with `figure_7_ref.pdf`.

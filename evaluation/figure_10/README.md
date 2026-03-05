# Figure 10

## What This Figure Shows

Figure 10 in the LLMServingSim 2.0 paper is a PIM case study that compares:

- `GPU-only`
- `PIM`
- `PIM-sbi`

The generated figure is `figure_10.pdf`, and the visual target is `figure_10_ref.pdf`.

## Inputs And Outputs

Checked-in reference measurements are not required for this figure. It is produced entirely from
LLMServingSim outputs.

The figure script generates LLMServingSim outputs in:

- `logs/`
- `results/`
- `parsed/`

The parsed simulator values used for plotting are:

- `parsed/gpu_only_b256_throughput.tsv`
- `parsed/pim_b256_throughput.tsv`
- `parsed/pim_sbi_b256_throughput.tsv`
- `parsed/component_energy.tsv`
- `parsed/energy_per_token.tsv`

The generated figure output is:

- `figure_10.pdf`

The visual target is `figure_10_ref.pdf`.

## Axes And Interpretation

Left panel:

- X axis: time in seconds
- Y axis: throughput in tokens per second

Right panel:

- X axis: system variant (`GPU-only`, `PIM`, `PIM-sbi`)
- Left Y axis: total energy in joules
- Right Y axis: joules per generated token

The stacked bars come from `parsed/component_energy.tsv`, and the overlaid points/line come from
`parsed/energy_per_token.tsv`.

## How To Run

From the `evaluation/` folder:

```bash
bash figure_10.sh
```

To compare your output:

1. Compare the parsed TSV files created in `parsed/` with
   `evaluation/artifacts/figure_10/parsed/`.  
   For automated compare from the evaluation folder, run `bash compare.sh <figure_id>`
   (for example, `bash compare.sh 10`).  
   For all options, see `bash compare.sh --help`.
2. Compare `figure_10.pdf` with `figure_10_ref.pdf`.

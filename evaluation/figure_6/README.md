# Figure 6

## What This Figure Shows

Figure 6 in the LLMServingSim 2.0 paper validates power and energy modeling on RTX
A6000. It consists of three outputs:

- `figure_6a.pdf`: TP1 power trace
- `figure_6b.pdf`: TP2 power trace
- `figure_6c.pdf`: energy breakdown

## Inputs And Outputs

Checked-in reference measurements are stored in:

- `reference/server_power_tp1.tsv`
- `reference/server_power_tp2.tsv`

The figure script generates LLMServingSim outputs in:

- `logs/`
- `results/`
- `parsed/`

The parsed simulator values used for plotting are:

- `parsed/power_tp1.tsv`
- `parsed/power_tp2.tsv`
- `parsed/component_energy.tsv`

The generated figure outputs are:

- `figure_6a.pdf`
- `figure_6b.pdf`
- `figure_6c.pdf`

The visual targets are `figure_6a_ref.pdf`, `figure_6b_ref.pdf`, and `figure_6c_ref.pdf`.

## Axes And Interpretation

Figure 6a and 6b:

- X axis: time in seconds
- Y axis: power in watts

These line plots compare the real server power trace in `reference/` against the parsed
LLMServingSim power trace in `parsed/`.

Figure 6c:

- X axis: tensor parallel setting (`TP1`, `TP2`)
- Y axis: total energy in joules

This stacked bar chart shows the component-wise energy split parsed from the simulator log.

## How To Run

From the `evaluation/` folder:

```bash
bash figure_6.sh
```

To compare your output:

1. Compare the parsed TSV files created in `parsed/` with
   `evaluation/artifacts/figure_6/parsed/`.  
   For automated compare from the evaluation folder, run `bash compare.sh <figure_id>`
   (for example, `bash compare.sh 6`).  
   For all options, see `bash compare.sh --help`.
2. Compare `figure_6a.pdf`, `figure_6b.pdf`, and `figure_6c.pdf` with the matching `*_ref.pdf`
files.

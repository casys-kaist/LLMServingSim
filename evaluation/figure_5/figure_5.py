#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator


FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parent.parent
FONT_PATH = FIG_DIR.parent / "fonts" / "Tahoma.ttf"
OUTPUT_PATH = FIG_DIR / "figure_5.pdf"

HARDWARES = ("A6000", "H100")
SYSTEMS = ("MD", "SD+PC", "PDD", "SM")

SUBPLOT_WIDTH = 3.25
SUBPLOT_HEIGHT = 1.83

COLOR_GREY = "#BABABA"
COLOR_GREEN = "#319F45"
GRID_GREY = "#D9D9D9"
GREY_WIDTH = 2.75
GREEN_WIDTH = 2.0
FONT_SIZE = 12
TITLE_SIZE = 14


def k_formatter(value, _pos):
    if value == 0:
        return "0"
    return f"{int(value / 1000)}K"


def configure_font():
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Tahoma font not found at {FONT_PATH}")

    font_manager.fontManager.addfont(str(FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["axes.unicode_minus"] = False


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_series(path: Path) -> dict[int, float]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {
            int(row["time_s"]): float(row["throughput"])
            for row in reader
            if row["time_s"] and row["throughput"]
        }


def build_merged_rows(vllm_path: Path, lls_path: Path) -> list[tuple[int, float, float]]:
    vllm_series = load_series(vllm_path)
    lls_series = load_series(lls_path)
    all_times = sorted(set(vllm_series) | set(lls_series))

    rows = []
    for time_s in all_times:
        vllm_value = vllm_series.get(time_s, math.nan)
        lls_value = lls_series.get(time_s, math.nan)
        rows.append((time_s, vllm_value, lls_value))
    return rows


def draw_subplot(ax, rows: list[tuple[int, float, float]], hardware: str, system: str):
    x_values = [row[0] for row in rows]
    vllm_values = [row[1] for row in rows]
    lls_values = [row[2] for row in rows]
    x_max = max(x_values) if x_values else 0
    x_upper = int(math.ceil(x_max / 10.0) * 10) if x_max else 10

    ax.set_axisbelow(True)
    ax.plot(
        x_values,
        vllm_values,
        color=COLOR_GREY,
        linewidth=GREY_WIDTH,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=1,
    )
    ax.plot(
        x_values,
        lls_values,
        color=COLOR_GREEN,
        linewidth=GREEN_WIDTH,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=2,
    )

    ax.set_xlim(0, x_upper)
    ax.set_ylim(0, 4000)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_yticks([0, 1000, 2000, 3000, 4000])
    ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0, y=0)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.set_title(f"{hardware} / {system}", fontsize=TITLE_SIZE, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(True)


def main():
    configure_font()

    figure, axes = plt.subplots(
        len(HARDWARES),
        len(SYSTEMS),
        figsize=(SUBPLOT_WIDTH * len(SYSTEMS), SUBPLOT_HEIGHT * len(HARDWARES)),
        squeeze=False,
    )

    for row_index, hardware in enumerate(HARDWARES):
        hardware_dir = FIG_DIR / hardware
        for col_index, system in enumerate(SYSTEMS):
            vllm_path = hardware_dir / "reference" / f"{system}_throughput.tsv"
            lls_path = hardware_dir / "parsed" / f"{system}_throughput.tsv"

            rows = build_merged_rows(vllm_path, lls_path)
            draw_subplot(axes[row_index][col_index], rows, hardware, system)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {display_path(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator


FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parent.parent
FONT_PATH = FIG_DIR.parent / "fonts" / "Tahoma.ttf"
OUTPUT_PATH = FIG_DIR / "figure_10.pdf"

THROUGHPUT_PATHS = {
    "gpu-only": FIG_DIR / "parsed" / "gpu_only_b256_throughput.tsv",
    "pim": FIG_DIR / "parsed" / "pim_b256_throughput.tsv",
    "pim-sbi": FIG_DIR / "parsed" / "pim_sbi_b256_throughput.tsv",
}
ENERGY_PATH = FIG_DIR / "parsed" / "component_energy.tsv"
ENERGY_PER_TOKEN_PATH = FIG_DIR / "parsed" / "energy_per_token.tsv"

FONT_LABEL = 25
FONT_TICK = 25
LINE_WIDTH = 3
BORDER_WIDTH = 1.5

POWER_COLOR = "#6a00f4"
COLOR_BLUE = "#3763b8"
COLOR_GREEN = "#089f46"
COLOR_ORANGE = "#d52b53"
COLOR_BREAKDOWN = {
    "Others": "#555555",
    "GPU": "#ea4335",
    "CPU": "#fbbc05",
    "Memory": "#4fb158",
    "Link": "#dd6e00",
    "NIC": "#68c4cd",
    "Storage": "#2d4db4",
}

PLOT_ORDER = ["gpu-only", "pim", "pim-sbi"]
DISPLAY_LABELS = ["GPU-only", "PIM", "PIM-sbi"]
THROUGHPUT_MAX_TIME_S = 53


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


def k_formatter(value, _pos):
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value/1000:g}K"
    return f"{int(value)}"


def load_throughput(path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return pd.DataFrame(
        {
            "time_s": pd.to_numeric(df["time_s"]),
            column_name: pd.to_numeric(df["throughput"]),
        }
    )


def merge_throughput_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="time_s", how="outer")
    merged = merged.sort_values("time_s")
    merged = merged.set_index("time_s").reindex(range(THROUGHPUT_MAX_TIME_S + 1)).fillna(0.0)
    merged.index.name = "time_s"
    merged = merged.reset_index()
    return merged


def load_energy_breakdown() -> pd.DataFrame:
    df = pd.read_csv(ENERGY_PATH, sep="\t").set_index("component")
    df = df.reindex(index=COLOR_BREAKDOWN.keys(), columns=PLOT_ORDER).fillna(0.0)
    return df


def load_energy_per_token() -> pd.Series:
    df = pd.read_csv(ENERGY_PER_TOKEN_PATH, sep="\t")
    return df.reindex(columns=PLOT_ORDER).iloc[0].fillna(0.0)


def build_figure():
    throughput_frames = [load_throughput(THROUGHPUT_PATHS[name], name) for name in PLOT_ORDER]
    thr_df = merge_throughput_frames(throughput_frames)
    energy_df = load_energy_breakdown()
    energy_per_token = load_energy_per_token()

    fig, (ax_left, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(14, 4.5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    ax_left.plot(thr_df["time_s"], thr_df["gpu-only"], color=COLOR_BLUE, linewidth=LINE_WIDTH)
    ax_left.plot(thr_df["time_s"], thr_df["pim"], color=COLOR_GREEN, linewidth=LINE_WIDTH)
    ax_left.plot(thr_df["time_s"], thr_df["pim-sbi"], color=COLOR_ORANGE, linewidth=LINE_WIDTH)

    ax_left.set_xlabel("Time (sec)", fontsize=FONT_LABEL)
    ax_left.set_ylabel("Throughput (token/sec)", fontsize=FONT_LABEL)
    ax_left.tick_params(axis="both", labelsize=FONT_TICK)
    ax_left.set_ylim(0, 5000)
    ax_left.yaxis.set_major_locator(MultipleLocator(1000))
    ax_left.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax_left.grid(True, axis="y")

    x_bar = range(len(DISPLAY_LABELS))
    bottom = [0.0, 0.0, 0.0]
    for component in COLOR_BREAKDOWN:
        values = [energy_df.loc[component, name] for name in PLOT_ORDER]
        ax_bar.bar(
            x_bar,
            values,
            bottom=bottom,
            color=COLOR_BREAKDOWN[component],
            zorder=3,
            edgecolor="black",
            linewidth=BORDER_WIDTH,
        )
        bottom = [bottom[i] + values[i] for i in range(len(values))]

    ax_bar.set_xticks(list(x_bar))
    ax_bar.set_xticklabels(DISPLAY_LABELS, fontsize=FONT_TICK)
    ax_bar.set_ylabel("Total Energy (J)", fontsize=FONT_LABEL)
    ax_bar.set_ylim(0, 40000)
    ax_bar.grid(True, axis="y", zorder=0)
    ax_bar.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax_bar.tick_params(axis="y", labelsize=FONT_TICK)

    ax_watt = ax_bar.twinx()
    watt_per_token = [float(energy_per_token[name]) for name in PLOT_ORDER]
    ax_watt.scatter(x_bar, watt_per_token, s=200, zorder=5, color=POWER_COLOR)
    ax_watt.plot(x_bar, watt_per_token, linestyle="--", linewidth=3, color=POWER_COLOR)
    ax_watt.set_ylabel("J / token", fontsize=FONT_LABEL)
    ax_watt.tick_params(axis="y", labelsize=FONT_TICK)
    ax_watt.set_ylim(0, 0.4)
    ax_watt.grid(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    configure_font()
    build_figure()
    print(f"Saved figure to {display_path(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

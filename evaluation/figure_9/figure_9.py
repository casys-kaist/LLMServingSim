#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator


FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parent.parent
TAHOMA_FONT_PATH = FIG_DIR.parent / "fonts" / "Tahoma.ttf"
TIMES_FONT_PATH = FIG_DIR.parent / "fonts" / "Times New Roman.ttf"
TIMES_BOLD_FONT_PATH = FIG_DIR.parent / "fonts" / "Times New Roman Bold.ttf"
PARSED_THROUGHPUT_PATH = FIG_DIR / "parsed" / "SD_throughput.tsv"
VLLM_THROUGHPUT_PATH = FIG_DIR / "reference" / "SD_throughput.tsv"
PARSED_LATENCY_PATH = FIG_DIR / "parsed" / "SD_latency.tsv"
VLLM_LATENCY_PATH = FIG_DIR / "reference" / "SD_latency.tsv"
OUTPUT_PLOT_PATH = FIG_DIR / "figure_9a.pdf"
OUTPUT_ERROR_PATH = FIG_DIR / "figure_9b.pdf"

FIG_WIDTH = 5
FIG_HEIGHT = 3

FONT_LABEL = 18
FONT_TICK = 18
LINE_WIDTH = 3

HEADER_BG = "#666666"
GEOMEAN_BG = "#DDDDDD"
WHITE = "#FFFFFF"
BLACK = "#000000"
TABLE_FONT_SIZE = 24
CELL_TEXT_Y = 0.44

COLORS = {
    "vLLM": "#bababa",
    "LLMServingSim": "#319e44",
}


def k_formatter(value, _pos):
    if value == 0:
        return "0"
    if value % 1000:
        return ""
    return f"{value/1000:.0f}K"


def configure_tahoma_font():
    if not TAHOMA_FONT_PATH.exists():
        raise FileNotFoundError(f"Tahoma font not found at {TAHOMA_FONT_PATH}")

    font_manager.fontManager.addfont(str(TAHOMA_FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(TAHOMA_FONT_PATH)).get_name()
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["axes.unicode_minus"] = False


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def configure_times_font():
    if not TIMES_FONT_PATH.exists():
        raise FileNotFoundError(f"Times New Roman font not found at {TIMES_FONT_PATH}")
    if not TIMES_BOLD_FONT_PATH.exists():
        raise FileNotFoundError(f"Times New Roman bold font not found at {TIMES_BOLD_FONT_PATH}")

    font_manager.fontManager.addfont(str(TIMES_FONT_PATH))
    font_manager.fontManager.addfont(str(TIMES_BOLD_FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(TIMES_FONT_PATH)).get_name()
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["axes.unicode_minus"] = False


def load_series(path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return pd.DataFrame(
        {
            "time_s": pd.to_numeric(df["time_s"]),
            column_name: pd.to_numeric(df["throughput"]),
        }
    )


def merge_series(vllm_df: pd.DataFrame, parsed_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(vllm_df, parsed_df, on="time_s", how="outer").sort_values("time_s")
    merged["vllm"] = merged["vllm"].fillna(0.0)
    merged["llmservingsim"] = merged["llmservingsim"].fillna(0.0)
    return merged


def load_latency_row(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    return df.iloc[0]


def percent_error(observed: float, reference: float) -> float:
    return abs(observed - reference) / reference * 100.0


def geomean(values: list[float]) -> float:
    return math.prod(values) ** (1.0 / len(values))


def build_error_rows() -> list[tuple[str, str]]:
    parsed = load_latency_row(PARSED_LATENCY_PATH)
    vllm = load_latency_row(VLLM_LATENCY_PATH)

    tps_error = percent_error(float(parsed["throughput_tok_s"]), float(vllm["throughput_tok_s"]))
    tpot_error = percent_error(float(parsed["mean_tpot_ms"]), float(vllm["mean_tpot_ms"]))
    itl_error = percent_error(float(parsed["mean_itl_ms"]), float(vllm["mean_itl_ms"]))
    geo = geomean([tps_error, tpot_error, itl_error])

    return [
        ("TPS", f"{tps_error:.5f}"),
        ("TPOT", f"{tpot_error:.5f}"),
        ("ITL", f"{itl_error:.5f}"),
        ("Geomean", f"{geo:.5f}"),
    ]


def build_plot(vllm_df: pd.DataFrame, parsed_df: pd.DataFrame):
    merged_df = merge_series(vllm_df, parsed_df)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(
        merged_df["time_s"],
        merged_df["vllm"],
        color=COLORS["vLLM"],
        linewidth=LINE_WIDTH,
    )
    ax.plot(
        merged_df["time_s"],
        merged_df["llmservingsim"],
        color=COLORS["LLMServingSim"],
        linewidth=LINE_WIDTH,
    )

    ax.set_xlabel("Time (sec)", fontsize=FONT_LABEL)
    ax.set_ylabel("Throughput (token/sec)", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.yaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.set_xlim(merged_df["time_s"].min(), merged_df["time_s"].max())
    ax.set_ylim(0, 3000)
    ax.grid(True, axis="y", alpha=0.6)

    fig.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_error_table(rows: list[tuple[str, str]]) -> None:
    times_regular = font_manager.FontProperties(fname=str(TIMES_FONT_PATH), size=TABLE_FONT_SIZE)
    times_bold = font_manager.FontProperties(fname=str(TIMES_BOLD_FONT_PATH), size=TABLE_FONT_SIZE)

    n_rows = len(rows) + 1
    col_widths = [0.55, 0.45]
    total_width = sum(col_widths)

    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    ax.add_patch(
        plt.Rectangle((0, n_rows - 1), total_width, 1, facecolor=HEADER_BG, edgecolor=BLACK, linewidth=2)
    )
    ax.text(
        total_width / 2,
        n_rows - 1 + CELL_TEXT_Y,
        "Error Rate (%)",
        ha="center",
        va="center",
        color=WHITE,
        fontproperties=times_bold,
        linespacing=1.0,
    )

    for row_index, (label, value) in enumerate(rows):
        y = n_rows - 2 - row_index
        facecolor = GEOMEAN_BG if label == "Geomean" else WHITE
        font_props = times_bold if label == "Geomean" else times_regular

        ax.add_patch(plt.Rectangle((0, y), col_widths[0], 1, facecolor=facecolor, edgecolor=BLACK, linewidth=2))
        ax.add_patch(
            plt.Rectangle((col_widths[0], y), col_widths[1], 1, facecolor=facecolor, edgecolor=BLACK, linewidth=2)
        )

        ax.text(
            col_widths[0] / 2,
            y + CELL_TEXT_Y,
            label,
            ha="center",
            va="center",
            color=BLACK,
            fontproperties=font_props,
            linespacing=1.0,
        )
        ax.text(
            col_widths[0] + col_widths[1] / 2,
            y + CELL_TEXT_Y,
            value,
            ha="center",
            va="center",
            color=BLACK,
            fontproperties=font_props,
            linespacing=1.0,
        )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUTPUT_ERROR_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_tahoma_font()
    vllm_df = load_series(VLLM_THROUGHPUT_PATH, "vllm")
    parsed_df = load_series(PARSED_THROUGHPUT_PATH, "llmservingsim")
    build_plot(vllm_df, parsed_df)

    configure_times_font()
    error_rows = build_error_rows()
    draw_error_table(error_rows)

    print(f"Saved figure to {display_path(OUTPUT_PLOT_PATH)}")
    print(f"Saved figure to {display_path(OUTPUT_ERROR_PATH)}")


if __name__ == "__main__":
    main()

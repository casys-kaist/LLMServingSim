#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parent.parent
FONT_PATH = FIG_DIR.parent / "fonts" / "Tahoma.ttf"

LINE_INPUTS = {
    "figure_6a.pdf": {
        "server": FIG_DIR / "reference" / "server_power_tp1.tsv",
        "parsed": FIG_DIR / "parsed" / "power_tp1.tsv",
    },
    "figure_6b.pdf": {
        "server": FIG_DIR / "reference" / "server_power_tp2.tsv",
        "parsed": FIG_DIR / "parsed" / "power_tp2.tsv",
    },
}
COMPONENT_INPUT_PATH = FIG_DIR / "parsed" / "component_energy.tsv"
COMPONENT_OUTPUT_PATH = FIG_DIR / "figure_6c.pdf"

LINE_FIG_WIDTH = 4.52
LINE_FIG_HEIGHT = 1.69
BAR_FIG_WIDTH = 2.5
BAR_FIG_HEIGHT = 3.0

COLOR_GREY = "#BABABA"
COLOR_GREEN = "#47A942"
GRID_GREY = "#D9D9D9"
GREY_WIDTH = 2.75
GREEN_WIDTH = 2.0

FONT_SIZE = 14
FONT_LABEL = 14
FONT_TICK = 14
BORDER_WIDTH = 1.2

STACK_ORDER = ["Others", "GPU", "CPU", "Memory", "Link", "NIC", "Storage"]
COLOR_BREAKDOWN = {
    "Others": "#555555",
    "GPU": "#ea4335",
    "CPU": "#fbbc05",
    "Memory": "#4fb158",
    "Link": "#dd6e00",
    "NIC": "#68c4cd",
    "Storage": "#2d4db4",
}


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


def load_power_data(path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return pd.DataFrame(
        {
            "time_s": pd.to_numeric(df["time_s"]),
            column_name: pd.to_numeric(df["power_w"]),
        }
    )


def build_line_plot(server_df: pd.DataFrame, parsed_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(LINE_FIG_WIDTH, LINE_FIG_HEIGHT))

    ax.set_axisbelow(True)
    ax.plot(
        server_df["time_s"],
        server_df["server_power"],
        color=COLOR_GREY,
        linewidth=GREY_WIDTH,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=1,
    )
    ax.plot(
        parsed_df["time_s"],
        parsed_df["parsed_power"],
        color=COLOR_GREEN,
        linewidth=GREEN_WIDTH,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=2,
    )

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1000)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0, y=0)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def k_formatter(value, _pos):
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:g}K"
    return f"{int(value)}"


def load_component_data() -> pd.DataFrame:
    if not COMPONENT_INPUT_PATH.exists():
        raise FileNotFoundError(f"Parsed component energy TSV not found at {COMPONENT_INPUT_PATH}")

    df = pd.read_csv(COMPONENT_INPUT_PATH, sep="\t")
    df["component"] = df["component"].astype(str).str.strip()
    return df.set_index("component").reindex(STACK_ORDER)


def build_component_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(BAR_FIG_WIDTH, BAR_FIG_HEIGHT))

    tp_cols = ["TP 1", "TP 2"]
    x = range(len(tp_cols))
    bottom = [0.0] * len(tp_cols)

    for component in STACK_ORDER:
        values = df.loc[component, tp_cols].astype(float).tolist()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=COLOR_BREAKDOWN[component],
            edgecolor="black",
            linewidth=BORDER_WIDTH,
            zorder=3,
        )
        bottom = [bottom[i] + values[i] for i in range(len(tp_cols))]

    ax.set_xticks(list(x))
    ax.set_xticklabels(["TP1", "TP2"], fontsize=FONT_TICK)
    ax.set_ylabel("Total Energy (J)", fontsize=FONT_LABEL)
    ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(True, axis="y", alpha=0.6, zorder=0, linewidth=1.2)
    ax.set_xlim(-0.6, len(tp_cols) - 0.4)

    plt.tight_layout()
    plt.savefig(COMPONENT_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_font()

    for output_name, paths in LINE_INPUTS.items():
        server_df = load_power_data(paths["server"], "server_power")
        parsed_df = load_power_data(paths["parsed"], "parsed_power")
        build_line_plot(server_df, parsed_df, FIG_DIR / output_name)
        print(f"Saved figure to {display_path(FIG_DIR / output_name)}")

    component_df = load_component_data()
    build_component_plot(component_df)
    print(f"Saved figure to {display_path(COMPONENT_OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

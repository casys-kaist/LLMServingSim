#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


FIG_DIR = Path(__file__).resolve().parent
EVAL_DIR = FIG_DIR.parent
REPO_ROOT = FIG_DIR.parent.parent

FONT_PATH = EVAL_DIR / "fonts" / "Tahoma.ttf"
OUTPUT_PATH = FIG_DIR / "figure_7.pdf"

COLOR_SINGLE = "#888888"
COLOR_SINGLE_DARK = "#333333"
COLOR_GREENS = ["#70e000", "#008000"]
COLOR_BLUES = ["#88BCFF", "#3864B9"]
COLOR_REDS = ["#f26a8d", "#880d1e"]

FONT_LABEL = 26
FONT_TICK = 25
LINE_WIDTH = 3


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


def merge_single():
    vllm = pd.read_csv(FIG_DIR / "reference" / "SD+PC.tsv", sep="\t")
    lls = pd.read_csv(FIG_DIR / "parsed" / "SD+PC.tsv", sep="\t")
    return pd.DataFrame(
        {
            "time_s": vllm["time_s"],
            "vllm_mem_mb": pd.to_numeric(vllm["gpu_mem_mb"]),
            "lls_mem_mb": pd.to_numeric(lls["gpu_mem_mb"]),
            "vllm_hit_pct": pd.to_numeric(vllm["prefix_hit_pct"]),
            "lls_hit_pct": pd.to_numeric(lls["prefix_hit_pct"]),
        }
    )


def merge_multi(name):
    vllm = pd.read_csv(FIG_DIR / "reference" / f"{name}.tsv", sep="\t")
    lls = pd.read_csv(FIG_DIR / "parsed" / f"{name}.tsv", sep="\t")

    if "gpu_mem_mb" in lls.columns:
        return pd.DataFrame(
            {
                "time_s": vllm["time_s"],
                "vllm_mem_mb": pd.to_numeric(vllm["gpu_mem_mb"]),
                "lls_mem_mb": pd.to_numeric(lls["gpu_mem_mb"]),
                "vllm_hit_pct": pd.to_numeric(vllm["prefix_hit_pct"]),
                "lls_hit_pct": pd.to_numeric(lls["prefix_hit_pct"]),
            }
        )

    return pd.DataFrame(
        {
            "time_s": vllm["time_s"],
            "vllm_mem_mb": pd.to_numeric(vllm["shared_cpu_mem_mb"]),
            "lls_mem_mb": pd.to_numeric(lls["shared_cpu_mem_mb"]),
            "vllm_hit_pct": pd.to_numeric(vllm["shared_cpu_prefix_hit_pct"]),
            "lls_hit_pct": pd.to_numeric(lls["shared_cpu_prefix_hit_pct"]),
        }
    )


def build_plot():
    single_df = merge_single()
    inst0_df = merge_multi("MD+PC+PS_inst0")
    inst1_df = merge_multi("MD+PC+PS_inst1")
    shared_df = merge_multi("MD+PC+PS_shared_cpu")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    ax_single_mem, ax_multi_mem = axes[0]
    ax_single_hit, ax_multi_hit = axes[1]

    for ax in [ax_single_mem, ax_single_hit, ax_multi_mem, ax_multi_hit]:
        ax.xaxis.set_major_locator(MultipleLocator(20))

    ax_single_mem.plot(
        single_df["time_s"],
        single_df["vllm_mem_mb"] / 1024.0,
        color=COLOR_SINGLE_DARK,
        linewidth=LINE_WIDTH,
    )
    ax_single_mem.plot(
        single_df["time_s"],
        single_df["lls_mem_mb"] / 1024.0,
        color=COLOR_SINGLE,
        linestyle="--",
        linewidth=LINE_WIDTH,
    )
    ax_single_mem.set_ylabel("Memory Usage (GB)", fontsize=FONT_LABEL)
    ax_single_mem.grid(True, axis="y")
    ax_single_mem.tick_params(axis="both", labelsize=FONT_TICK)

    ax_single_hit.plot(
        single_df["time_s"],
        single_df["vllm_hit_pct"],
        color=COLOR_SINGLE_DARK,
        linewidth=LINE_WIDTH,
    )
    ax_single_hit.plot(
        single_df["time_s"],
        single_df["lls_hit_pct"],
        color=COLOR_SINGLE,
        linestyle="--",
        linewidth=LINE_WIDTH,
    )
    ax_single_hit.set_ylabel("Prefix Hit Rate (%)", fontsize=FONT_LABEL)
    ax_single_hit.set_xlabel("Time (sec)", fontsize=FONT_LABEL)
    ax_single_hit.grid(True, axis="y")
    ax_single_hit.tick_params(axis="both", labelsize=FONT_TICK)

    series = [
        (inst0_df, COLOR_BLUES[1], COLOR_BLUES[0]),
        (inst1_df, COLOR_REDS[1], COLOR_REDS[0]),
        (shared_df, COLOR_GREENS[1], COLOR_GREENS[0]),
    ]
    for df, solid, dashed in series:
        ax_multi_mem.plot(
            df["time_s"],
            df["vllm_mem_mb"] / 1024.0,
            color=solid,
            linewidth=LINE_WIDTH,
        )
        ax_multi_mem.plot(
            df["time_s"],
            df["lls_mem_mb"] / 1024.0,
            color=dashed,
            linestyle="--",
            linewidth=LINE_WIDTH,
        )
        ax_multi_hit.plot(
            df["time_s"],
            df["vllm_hit_pct"],
            color=solid,
            linewidth=LINE_WIDTH,
        )
        ax_multi_hit.plot(
            df["time_s"],
            df["lls_hit_pct"],
            color=dashed,
            linestyle="--",
            linewidth=LINE_WIDTH,
        )

    ax_multi_mem.set_ylabel("Memory Usage (GB)", fontsize=FONT_LABEL)
    ax_multi_mem.grid(True, axis="y")
    ax_multi_mem.tick_params(axis="both", labelsize=FONT_TICK)

    ax_multi_hit.set_ylabel("Prefix Hit Rate (%)", fontsize=FONT_LABEL)
    ax_multi_hit.set_xlabel("Time (sec)", fontsize=FONT_LABEL)
    ax_multi_hit.grid(True, axis="y")
    ax_multi_hit.tick_params(axis="both", labelsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    configure_font()
    build_plot()
    print(f"Saved figure to {display_path(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

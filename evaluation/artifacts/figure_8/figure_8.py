#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parent.parent
FONT_PATH = FIG_DIR.parent / "fonts" / "Tahoma.ttf"
PRIOR_WORKS_DIR = FIG_DIR / "reference"
PARSED_DIR = FIG_DIR / "parsed"

OUTPUT_UP = FIG_DIR / "figure_8a.pdf"
OUTPUT_DOWN = FIG_DIR / "figure_8b.pdf"

SCENARIOS = [
    ("SD", "Single Dense"),
    ("MD", "Multi Dense"),
    ("PDD", "Prefill-Decode Disaggregated"),
    ("SM", "Single MoE"),
]

COLORS = {
    "LLMServingSim": "#a4c3b2",
    "TokenSim": "#dee2e6",
    "Vidur": "#6c757d",
    "APEX": "#212529",
    "LLMServingSim2.0": "#319e44",
    "vLLM": "#ffffff",
}

UP_BASELINES = ["TokenSim", "Vidur", "APEX", "LLMServingSim2.0"]
DOWN_BASELINES = ["LLMServingSim", "TokenSim", "Vidur", "APEX", "LLMServingSim2.0"]
BASELINE_OFFSET = {
    "LLMServingSim": -2,
    "TokenSim": -1,
    "Vidur": 0,
    "APEX": 1,
    "LLMServingSim2.0": 2,
}

UP_METRICS = [
    ("throughput_tok_s", "TPS"),
    ("mean_ttft_ms", "TTFT"),
    ("mean_tpot_ms", "TPOT"),
]

FIG_WIDTH = 14.0
UP_FIG_HEIGHT = 3.0
DOWN_FIG_HEIGHT = 3.0
FONT_LABEL = 17
FONT_TICK = 15

UP_Y_MIN = 0.4
UP_Y_MAX = 1.4
DOWN_Y_MAX = 750
COLOR_RED = "#a4161a"


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


def parse_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "X":
        return None
    return float(text)


def load_prior_validation(name: str) -> pd.DataFrame:
    df = pd.read_csv(PRIOR_WORKS_DIR / f"{name}_latency.tsv", sep="\t")
    return df.set_index("framework")


def load_prior_simtime(name: str) -> pd.DataFrame:
    df = pd.read_csv(PRIOR_WORKS_DIR / f"{name}_sim_time.tsv", sep="\t")
    return df.set_index("framework")


def load_parsed_latency(name: str) -> dict[str, float]:
    df = pd.read_csv(PARSED_DIR / f"{name}_latency.tsv", sep="\t")
    row = df.iloc[0]
    return {
        "throughput_tok_s": float(row["throughput_tok_s"]),
        "mean_ttft_ms": float(row["mean_ttft_ms"]),
        "mean_tpot_ms": float(row["mean_tpot_ms"]),
    }


def load_parsed_simtime(name: str) -> float:
    df = pd.read_csv(PARSED_DIR / f"{name}_sim_time.tsv", sep="\t")
    return float(df.iloc[0]["simulation_time_s"])


def build_validation_map():
    validation = {}
    for name, _label in SCENARIOS:
        df = load_prior_validation(name)
        parsed = load_parsed_latency(name)
        df.loc["LLMServingSim2.0"] = parsed
        validation[name] = df
    return validation


def build_simtime_map():
    simtime = {}
    for name, _label in SCENARIOS:
        df = load_prior_simtime(name)
        df.loc["LLMServingSim2.0"] = {"simulation_time_s": load_parsed_simtime(name)}
        simtime[name] = df
    return simtime


def build_up_figure(validation):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, UP_FIG_HEIGHT))

    bar_w = 0.16
    gap_b = 0.02
    gap_m = 0.25
    gap_s = 0.45
    scenario_pad = 0.08
    outer_pad = gap_s * 0.6

    xticks = []
    xticklabels = []
    scenario_centers = []
    scenario_bounds = []

    metric_block_w = len(UP_BASELINES) * bar_w + (len(UP_BASELINES) - 1) * gap_b
    scenario_core_w = len(UP_METRICS) * metric_block_w + (len(UP_METRICS) - 1) * gap_m
    scenario_total_w = scenario_core_w + 2 * scenario_pad

    x = 0.0
    for name, scen_name in SCENARIOS:
        scen_left = x
        scen_right = x + scenario_total_w
        scen_core_left = scen_left + scenario_pad
        vllm_row = validation[name].loc["vLLM"]

        for metric_index, (metric_key, metric_label) in enumerate(UP_METRICS):
            group_left = scen_core_left + metric_index * (metric_block_w + gap_m)
            vllm_value = parse_value(vllm_row[metric_key])

            for baseline_index, baseline in enumerate(UP_BASELINES):
                xpos = group_left + baseline_index * (bar_w + gap_b) + bar_w / 2
                raw_value = parse_value(validation[name].loc[baseline, metric_key])
                normalized_value = None if raw_value is None or vllm_value in (None, 0) else raw_value / vllm_value

                if normalized_value is None:
                    ax.scatter(
                        xpos,
                        UP_Y_MIN + 0.03 * (UP_Y_MAX - UP_Y_MIN) + 0.02,
                        marker="x",
                        s=65,
                        linewidths=2.0,
                        color="black",
                        zorder=6,
                    )
                else:
                    height = min(normalized_value, UP_Y_MAX)
                    ax.bar(
                        xpos,
                        height,
                        width=bar_w,
                        color=COLORS[baseline],
                        edgecolor="black",
                        linewidth=1.0,
                        zorder=3,
                    )
                    if normalized_value > UP_Y_MAX:
                        ax.text(
                            xpos,
                            UP_Y_MAX + 0.03,
                            f"{normalized_value:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=FONT_TICK - 1,
                        )

            group_right = group_left + metric_block_w
            xticks.append((group_left + group_right) / 2)
            xticklabels.append(metric_label)

        scenario_centers.append((scen_left + scen_right) / 2)
        scenario_bounds.append((scen_left, scen_right))
        x = scen_right + gap_s

    ax.set_ylim(UP_Y_MIN, UP_Y_MAX)
    ax.set_ylabel("Normalized Metrics", fontsize=FONT_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(True, axis="y", alpha=0.5, zorder=0)
    ax.axhline(1.0, color=COLOR_RED, linestyle="--", linewidth=1.5, zorder=1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=FONT_TICK)
    ax.tick_params(axis="x", which="both", length=0)

    for (_, name), center in zip(SCENARIOS, scenario_centers):
        ax.text(
            center,
            -0.22,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=FONT_LABEL,
        )

    left_edge = scenario_bounds[0][0] - outer_pad
    right_edge = scenario_bounds[-1][1] + outer_pad
    ax.set_xlim(left_edge, right_edge)

    fig.tight_layout()
    fig.savefig(OUTPUT_UP, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_down_figure(simtime):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, DOWN_FIG_HEIGHT))

    bar_w = 0.16
    gap_b = 0.04
    gap_s = 0.45
    scenario_pad = 0.08
    outer_pad = gap_s * 0.6

    xticks = []
    xticklabels = []
    scenario_bounds = []

    x = 0.0
    group_half_w = max(abs(value) for value in BASELINE_OFFSET.values()) * (bar_w + gap_b)

    for name, scen_name in SCENARIOS:
        group_center = x + group_half_w + scenario_pad

        for baseline in DOWN_BASELINES:
            xpos = group_center + BASELINE_OFFSET[baseline] * (bar_w + gap_b)
            value = None
            if baseline in simtime[name].index:
                value = parse_value(simtime[name].loc[baseline, "simulation_time_s"])

            if value is None:
                ax.scatter(
                    xpos,
                    40,
                    marker="x",
                    s=70,
                    linewidths=2.0,
                    color="black",
                    zorder=6,
                )
            else:
                height = min(value, DOWN_Y_MAX)
                ax.bar(
                    xpos,
                    height,
                    width=bar_w,
                    color=COLORS[baseline],
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=3,
                )
                if value <= DOWN_Y_MAX:
                    ax.text(
                        xpos,
                        height + 20,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=FONT_TICK - 1,
                    )

        left = group_center - group_half_w
        right = group_center + group_half_w
        xticks.append(group_center)
        xticklabels.append(scen_name)
        scenario_bounds.append((left, right))
        x = right + gap_s

    ax.set_ylabel("Simulation Time (s)", fontsize=FONT_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_ylim(0, DOWN_Y_MAX)
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=FONT_LABEL)

    for (_, right) in scenario_bounds[:-1]:
        ax.axvline(right + gap_s / 2, color="black", linewidth=1.0)

    ax.set_xlim(scenario_bounds[0][0] - outer_pad, scenario_bounds[-1][1] + outer_pad)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.28, top=0.83)
    fig.savefig(OUTPUT_DOWN, dpi=300)
    plt.close(fig)


def main():
    configure_font()
    validation = build_validation_map()
    simtime = build_simtime_map()
    build_up_figure(validation)
    build_down_figure(simtime)
    print(f"Saved figure to {display_path(OUTPUT_UP)}")
    print(f"Saved figure to {display_path(OUTPUT_DOWN)}")


if __name__ == "__main__":
    main()

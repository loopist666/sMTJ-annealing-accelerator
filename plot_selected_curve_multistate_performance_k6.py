from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception:
    plt = None


INPUT_DIR = Path("mtj_multistate_sweep_output")
OUTPUT_PATH = Path("selected_curve_multistate_k6_performance.png")

SOURCE_DATASET = "bad"
TRACE_LABEL = "selected trace"
MODEL_K = 6
PBIT_DIMS = 3

INSTANCE_LABELS = {
    "spin_glass_sparse_12": "SG sp-12",
    "spin_glass_dense_12": "SG dn-12",
    "spin_glass_sparse_14": "SG sp-14",
    "spin_glass_dense_14": "SG dn-14",
    "spin_glass_sparse_16": "SG sp-16",
    "spin_glass_dense_16": "SG dn-16",
    "max_cut_14": "MC-14",
    "max_cut_16": "MC-16",
}


def annotate_heatmap(ax, values: np.ndarray, fmt) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                fmt(values[row, col]),
                ha="center",
                va="center",
                fontsize=10,
                color="#202020",
            )


def main() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required to create the performance figure")

    overall = pd.read_csv(INPUT_DIR / "multistate_overall_sweep.csv")
    instances = pd.read_csv(INPUT_DIR / "multistate_instance_sweep.csv")

    selected_overall = overall[
        (overall["dataset"] == SOURCE_DATASET)
        & (overall["n_states"] == MODEL_K)
        & (overall["pbit_dims"] == PBIT_DIMS)
    ]
    selected_instances = instances[
        (instances["dataset"] == SOURCE_DATASET)
        & (instances["n_states"] == MODEL_K)
        & (instances["pbit_dims"] == PBIT_DIMS)
    ].copy()

    if selected_overall.empty or selected_instances.empty:
        raise ValueError("No matching rows found for the selected K-state setting")

    order = list(INSTANCE_LABELS.keys())
    selected_instances["instance"] = pd.Categorical(selected_instances["instance"], categories=order, ordered=True)
    selected_instances = selected_instances.sort_values("instance")

    overall_row = selected_overall.iloc[0]
    mean_gap = float(overall_row["mean_instance_gap"])
    hit_rate = float(overall_row["mean_instance_hit_rate"])
    x_labels = [INSTANCE_LABELS[name] for name in selected_instances["instance"].astype(str)]
    hit_values = selected_instances["optimum_hit_rate"].to_numpy(dtype=float) * 100.0
    gap_values = selected_instances["mean_energy_gap"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(15.5, 9), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.4, wspace=0.34)

    fig.suptitle(
        f"Multistate p-bit performance at model-selected K-state setting (K={MODEL_K}, p-bit dims={PBIT_DIMS})",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.935,
        "8 benchmark instances x 20 runs each; lower gap is better, higher hit rate is better.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    ax_gap = fig.add_subplot(grid[0, 0])
    ax_gap.bar([0], [mean_gap], color="#4da195", width=0.48)
    ax_gap.set_title("Overall mean energy gap", fontsize=13, fontweight="bold")
    ax_gap.set_ylabel("Mean gap to exact optimum")
    ax_gap.set_xticks([0])
    ax_gap.set_xticklabels([TRACE_LABEL])
    ax_gap.set_ylim(0.0, max(1.0, mean_gap * 1.45))
    ax_gap.grid(True, axis="y", alpha=0.25)
    ax_gap.text(0, mean_gap + 0.04, f"{mean_gap:g}", ha="center", va="bottom", fontweight="bold")
    ax_gap.text(
        0,
        ax_gap.get_ylim()[1] * 0.86,
        "Model-selected K stays below one energy unit on average",
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f4f4f4", "edgecolor": "#dddddd"},
    )

    ax_hit = fig.add_subplot(grid[0, 1])
    ax_hit.bar([0], [hit_rate], color="#df704f", width=0.48)
    ax_hit.set_title("Overall optimum hit rate", fontsize=13, fontweight="bold")
    ax_hit.set_ylabel("Hit rate")
    ax_hit.set_xticks([0])
    ax_hit.set_xticklabels([TRACE_LABEL])
    ax_hit.set_ylim(0.0, 1.05)
    ax_hit.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_hit.grid(True, axis="y", alpha=0.25)
    ax_hit.text(0, hit_rate + 0.035, f"{hit_rate * 100:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax_hit.text(
        0,
        0.18,
        "K=6 follows the physical model-selected state count",
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f4f4f4", "edgecolor": "#dddddd"},
    )

    ax_hitmap = fig.add_subplot(grid[1, 0])
    hit_matrix = hit_values.reshape(1, -1)
    hit_img = ax_hitmap.imshow(hit_matrix, cmap="YlGn", vmin=50, vmax=100, aspect="auto")
    ax_hitmap.set_title("Per-instance optimum hit rate", fontsize=13, fontweight="bold")
    ax_hitmap.set_xticks(np.arange(len(x_labels)))
    ax_hitmap.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_hitmap.set_yticks([0])
    ax_hitmap.set_yticklabels([TRACE_LABEL])
    annotate_heatmap(ax_hitmap, hit_matrix, lambda value: f"{value:.0f}%")
    hit_cbar = fig.colorbar(hit_img, ax=ax_hitmap, fraction=0.04, pad=0.02)
    hit_cbar.ax.yaxis.set_major_formatter(PercentFormatter(100.0))

    ax_gapmap = fig.add_subplot(grid[1, 1])
    gap_matrix = gap_values.reshape(1, -1)
    gap_vmax = max(2.5, float(np.max(gap_matrix)))
    gap_img = ax_gapmap.imshow(gap_matrix, cmap="YlOrRd", vmin=0, vmax=gap_vmax, aspect="auto")
    ax_gapmap.set_title("Per-instance mean energy gap", fontsize=13, fontweight="bold")
    ax_gapmap.set_xticks(np.arange(len(x_labels)))
    ax_gapmap.set_xticklabels(x_labels, rotation=45, ha="right")
    ax_gapmap.set_yticks([0])
    ax_gapmap.set_yticklabels([TRACE_LABEL])
    annotate_heatmap(ax_gapmap, gap_matrix, lambda value: f"{value:g}")
    gap_cbar = fig.colorbar(gap_img, ax=ax_gapmap, fraction=0.04, pad=0.02)
    gap_cbar.set_label("Energy gap")

    for ax in [ax_gap, ax_hit, ax_hitmap, ax_gapmap]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

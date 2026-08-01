"""Export Max-Cut and Portfolio comparison views into the benchmark bundle.

The source experiment is the fixed-encoding P1 application ablation.  These
views do not use the TSP-only direct-permutation representation.  They compare
Gray binary p-bit and ordered 1D P-dit device-state interpretations while
keeping each application's native QUBO/Ising encoding fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "results" / "diagnostics" / "p1_fixed_encoding_ablation"
OUTPUT_DIR = ROOT / "results" / "benchmarks" / "tsp_binary_pbit_vs_ordered_1d_mh"

PROBLEMS = {
    "maxcut_10": ("maxcut", "10-node Max-Cut"),
    "portfolio_8": ("portfolio", "8-asset Portfolio"),
}
METHODS = ["gray_binary_pbit", "ordered_1d_pdit"]
METHOD_LABELS = {
    "gray_binary_pbit": "Gray binary p-bit",
    "ordered_1d_pdit": "Ordered 1D P-dit",
}
METHOD_COLORS = {
    "gray_binary_pbit": "#4059AD",
    "ordered_1d_pdit": "#D95F02",
}


def finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return [
        {key: finite_or_none(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def save_metric_plot(summary: pd.DataFrame, slug: str, label: str, output_dir: Path) -> None:
    ordered = summary.set_index("method").reindex(METHODS)
    x = np.arange(len(METHODS))
    colors = [METHOD_COLORS[method] for method in METHODS]
    labels = [METHOD_LABELS[method] for method in METHODS]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    metrics = [
        ("success_probability", "Reference hit rate", 100.0, "%"),
        ("median_first_hit_update_successes", "Median first-hit update", 1.0, "updates"),
        ("mean_best_gap", "Mean best objective gap", 1.0, "gap"),
    ]
    for axis, (column, title, scale, ylabel) in zip(axes, metrics):
        values = ordered[column].astype(float).to_numpy() * scale
        display = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        bars = axis.bar(x, display, color=colors, width=0.62)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            text = "n/a" if not np.isfinite(value) else f"{value:.3g}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(f"{label}: fixed-encoding device-geometry comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{slug}_method_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_success_curve(curves: pd.DataFrame, slug: str, label: str, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    for method in METHODS:
        rows = curves[curves["method"] == method].sort_values("updates")
        axis.plot(
            rows["updates"],
            rows["success_probability"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    axis.set_xscale("log")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Single-spin update budget")
    axis.set_ylabel("Reference hit rate")
    axis.set_title(f"{label}: success versus update budget")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{slug}_success_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(source_dir / "application_comparison_runs.csv")
    summary = pd.read_csv(source_dir / "application_comparison_summary.csv")
    paired = pd.read_csv(source_dir / "application_paired_comparison.csv")
    curves = pd.read_csv(source_dir / "application_success_curves.csv")
    source_summary = json.loads(
        (source_dir / "application_comparison_summary.json").read_text(encoding="utf-8")
    )

    bundle = {
        "scope": "Max-Cut and Portfolio views packaged with the main benchmark artifacts",
        "comparison_design": (
            "Fixed native QUBO/Ising encoding; Gray binary p-bit versus ordered 1D P-dit "
            "device-state interpretation. Direct-permutation encoding is TSP-only."
        ),
        "source_results": str(source_dir.relative_to(ROOT)),
        "physical_dataset": source_summary["comparison_design"]["physical_dataset"],
        "selected_k": source_summary["comparison_design"]["selected_k"],
        "problems": {},
    }

    for problem, (slug, label) in PROBLEMS.items():
        problem_runs = runs[runs["problem"] == problem].copy()
        problem_summary = summary[summary["problem"] == problem].copy()
        problem_paired = paired[paired["problem"] == problem].copy()
        problem_curves = curves[curves["problem"] == problem].copy()
        if len(problem_runs) != 48 or set(problem_summary["method"]) != set(METHODS):
            raise AssertionError(f"Unexpected source rows for {problem}")

        problem_runs.to_csv(output_dir / f"{slug}_comparison_runs.csv", index=False)
        problem_summary.to_csv(output_dir / f"{slug}_comparison_summary.csv", index=False)
        problem_paired.to_csv(output_dir / f"{slug}_paired_comparison.csv", index=False)
        problem_curves.to_csv(output_dir / f"{slug}_success_curves.csv", index=False)
        save_metric_plot(problem_summary, slug, label, output_dir)
        save_success_curve(problem_curves, slug, label, output_dir)

        bundle["problems"][problem] = {
            "label": label,
            "run_count": int(len(problem_runs)),
            "run_file": f"{slug}_comparison_runs.csv",
            "success_curve_file": f"{slug}_success_curves.csv",
            "method_summary": records(problem_summary),
            "paired_comparison": records(problem_paired),
        }

    with open(
        output_dir / "maxcut_portfolio_comparison_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(bundle, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"Max-Cut and Portfolio views written to: {output_dir}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from mtj_hardware_sa_demo import summarize_runs
from mtj_physical_kstate_demo import (
    build_ising_instance,
    build_kstate_profile,
    dwell_dataframe,
    k_selection_dataframe,
    run_kstate_hazard_sa,
    stability_dataframe,
)


CURVE_NAME = "selected_curve"
OUTPUT_DIR = Path("mtj_physical_kstate_selected_curve_output")
RUNS = 40


TECHNICAL_ROUTE = [
    "读取选定的 Time/AI/AV 序列，并将 AI 幅值视为可观测的物理随机状态信号。",
    "对 AI 幅值拟合一维 Gaussian mixture 候选模型，并用 BIC 在 1..8 个隐状态之间选择 K。",
    "用带自保持偏置的 Gaussian HMM 和 Viterbi 解码，把带噪声的连续幅值轨迹分割为稳定的 K 个隐状态。",
    "按各状态的 AI 均值重新排序状态，提取每段 dwell time，并由相邻 dwell 段统计 KxK 转移矩阵。",
    "将 K-state 序列映射到 Gray-code p-bit bias，同时从 AI、跳变强度和局部波动中构造经验随机通道。",
    "把状态退出率作为 hazard rate，驱动 18-spin Ising 随机更新，并用多次运行统计能量 gap、命中率和接受率。",
    "沿外部轴做分箱稳定性检查，观察每个隐状态在不同区间中的占据比例与局部均值漂移。",
]


def save_tables(profile, optimization_rows, summary) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(optimization_rows).to_csv(OUTPUT_DIR / "kstate_optimization_runs.csv", index=False)
    dwell_dataframe(profile).to_csv(OUTPUT_DIR / "kstate_dwell_segments.csv", index=False)
    k_selection_dataframe(profile).to_csv(OUTPUT_DIR / "k_selection_table.csv", index=False)
    stability_dataframe(profile).to_csv(OUTPUT_DIR / "state_stability_bins.csv", index=False)
    with open(OUTPUT_DIR / "kstate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_route_markdown(summary) -> None:
    lines = [
        "# Physical K-State Technical Route",
        "",
        "本脚本只分析选定曲线，并在图例与解释中使用中性命名。",
        "",
        "## 技术路线",
        "",
    ]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(TECHNICAL_ROUTE, start=1))
    lines.extend(
        [
            "",
            "## 关键结果",
            "",
            f"- Selected K: {summary['kstate_model_summary'][CURVE_NAME]['selected_k']}",
            f"- State entropy: {summary['kstate_model_summary'][CURVE_NAME]['state_entropy_bits']:.4f} bits",
            f"- Mean energy gap: {summary['optimization_performance'][CURVE_NAME]['mean_energy_gap']:.4f}",
            f"- Optimum hit rate: {summary['optimization_performance'][CURVE_NAME]['optimum_hit_rate']:.4f}",
        ]
    )
    (OUTPUT_DIR / "technical_route.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def save_plots(profile) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    image = axes[0].imshow(profile.transition_matrix, cmap="viridis", aspect="auto")
    axes[0].set_title("KxK transition matrix")
    axes[0].set_xlabel("To state")
    axes[0].set_ylabel("From state")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    counts = np.bincount(profile.state_labels, minlength=profile.selected_k)
    axes[1].bar(np.arange(profile.selected_k), counts, color="#4c78a8")
    axes[1].set_title("State occupancy")
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kstate_profiles.png", dpi=180)
    plt.close(fig)

    k_table = pd.DataFrame(profile.k_selection_table).sort_values("k")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(k_table["k"], k_table["bic"], marker="o", linewidth=1.6, label="BIC")
    ax.axvline(profile.selected_k, color="#f58518", linestyle="--", linewidth=1.2, label="Selected K")
    ax.set_title("K selection by amplitude model")
    ax.set_xlabel("K")
    ax.set_ylabel("BIC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "k_selection_bic.png", dpi=180)
    plt.close(fig)

    stability = pd.DataFrame(profile.stability_rows)
    pivot = stability.pivot_table(index="bin_idx", columns="state", values="occupancy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for state in pivot.columns:
        ax.plot(pivot.index, pivot[state], marker="o", linewidth=1.3, label=f"State {state}")
    ax.set_title("State occupancy across external-axis bins")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Occupancy")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "state_stability.png", dpi=180)
    plt.close(fig)


def main() -> None:
    profile = build_kstate_profile(CURVE_NAME)
    J, h, optimum = build_ising_instance()

    runs = [
        run_kstate_hazard_sa(profile, J, h, optimum, run_idx=run_idx)
        for run_idx in range(RUNS)
    ]
    optimization_rows = [
        {
            "dataset": profile.name,
            "run_idx": run_idx,
            **result,
        }
        for run_idx, result in enumerate(runs)
    ]

    summary = {
        "assumption": "Analyze one selected MTJ curve with the physical K-state hazard route.",
        "technical_route": TECHNICAL_ROUTE,
        "problem": {
            "type": "18-spin Ising minimization",
            "exact_optimum_energy": optimum,
            "runs_per_curve": RUNS,
        },
        "kstate_model_summary": {
            profile.name: profile.metrics,
        },
        "optimization_performance": {
            profile.name: summarize_runs(runs),
        },
    }

    save_tables(profile, optimization_rows, summary)
    save_route_markdown(summary)
    save_plots(profile)

    print("Physical K-state selected-curve analysis finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

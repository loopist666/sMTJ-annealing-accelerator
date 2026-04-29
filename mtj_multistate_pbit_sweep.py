import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from mtj_hardware_sa_demo import summarize_runs
from mtj_hardware_sa_benchmark import build_benchmark_suite
from mtj_multistate_pbit_demo import build_multistate_profile, run_multidim_pbit_ising


OUTPUT_DIR = Path("mtj_multistate_sweep_output")
STATE_COUNTS = [4, 6, 8]
RUNS_PER_INSTANCE = 20
STEPS = 420


def evaluate_sweep(state_counts: list[int], runs_per_instance: int, steps: int) -> dict:
    suite = build_benchmark_suite()
    profiles = {
        n_states: {
            dataset: build_multistate_profile(dataset, n_states=n_states, steps=steps)
            for dataset in ["good", "bad"]
        }
        for n_states in state_counts
    }

    instance_rows = []
    aggregate = {
        n_states: {dataset: [] for dataset in ["good", "bad"]}
        for n_states in state_counts
    }
    win_counter = {
        n_states: {dataset: 0 for dataset in ["good", "bad"]}
        for n_states in state_counts
    }

    for n_states in state_counts:
        for instance in suite:
            instance_stats = {}
            for dataset, profile in profiles[n_states].items():
                runs = [
                    run_multidim_pbit_ising(profile, instance["J"], instance["h"], instance["optimum"], run_idx=run_idx)
                    for run_idx in range(runs_per_instance)
                ]
                stats = summarize_runs(runs)
                row = {
                    "n_states": n_states,
                    "pbit_dims": profile.pbit_biases.shape[1],
                    "dataset": dataset,
                    "instance": instance["name"],
                    "kind": instance["kind"],
                    "n_spins": instance["n_spins"],
                    "density": instance["density"],
                    "exact_optimum": float(instance["optimum"]),
                    "state_entropy_bits": float(profile.metrics["state_entropy_bits"]),
                    "transition_count": int(profile.metrics["transition_count"]),
                    **stats,
                }
                instance_rows.append(row)
                aggregate[n_states][dataset].append(row)
                instance_stats[dataset] = row

            ranking = sorted(
                ["good", "bad"],
                key=lambda dataset: (
                    instance_stats[dataset]["mean_energy_gap"],
                    -instance_stats[dataset]["optimum_hit_rate"],
                ),
            )
            win_counter[n_states][ranking[0]] += 1

    overall_rows = []
    for n_states in state_counts:
        for dataset in ["good", "bad"]:
            rows = aggregate[n_states][dataset]
            profile = profiles[n_states][dataset]
            overall_rows.append(
                {
                    "n_states": n_states,
                    "pbit_dims": profile.pbit_biases.shape[1],
                    "dataset": dataset,
                    "instances": len(rows),
                    "instance_wins": int(win_counter[n_states][dataset]),
                    "mean_instance_gap": float(np.mean([r["mean_energy_gap"] for r in rows])),
                    "mean_instance_hit_rate": float(np.mean([r["optimum_hit_rate"] for r in rows])),
                    "mean_instance_best_energy": float(np.mean([r["mean_best_energy"] for r in rows])),
                    "mean_instance_acceptance": float(np.mean([r["mean_acceptance_rate"] for r in rows])),
                    "state_entropy_bits": float(profile.metrics["state_entropy_bits"]),
                    "transition_count": int(profile.metrics["transition_count"]),
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    best_by_dataset = {}
    for dataset in ["good", "bad"]:
        sub = overall_df[overall_df["dataset"] == dataset].sort_values(
            ["mean_instance_gap", "mean_instance_hit_rate"],
            ascending=[True, False],
        )
        best_by_dataset[dataset] = {
            "n_states": int(sub.iloc[0]["n_states"]),
            "pbit_dims": int(sub.iloc[0]["pbit_dims"]),
            "mean_instance_gap": float(sub.iloc[0]["mean_instance_gap"]),
            "mean_instance_hit_rate": float(sub.iloc[0]["mean_instance_hit_rate"]),
        }

    return {
        "suite": suite,
        "profiles": profiles,
        "instance_rows": instance_rows,
        "overall_rows": overall_rows,
        "best_by_dataset": best_by_dataset,
        "runs_per_instance": runs_per_instance,
        "steps": steps,
    }


def save_outputs(results: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    instance_df = pd.DataFrame(results["instance_rows"])
    overall_df = pd.DataFrame(results["overall_rows"])
    instance_df.to_csv(OUTPUT_DIR / "multistate_instance_sweep.csv", index=False)
    overall_df.to_csv(OUTPUT_DIR / "multistate_overall_sweep.csv", index=False)

    summary = {
        "assumption": (
            "Sweep multistate segmentation granularity and p-bit dimensionality, then compare "
            "MTJ datasets on a shared Ising/Max-Cut benchmark suite."
        ),
        "state_counts": STATE_COUNTS,
        "runs_per_instance": results["runs_per_instance"],
        "steps": results["steps"],
        "best_by_dataset": results["best_by_dataset"],
    }
    with open(OUTPUT_DIR / "multistate_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    overall_df = overall_df.sort_values(["dataset", "n_states"])
    for dataset in ["good", "bad"]:
        sub = overall_df[overall_df["dataset"] == dataset]
        axes[0].plot(sub["n_states"], sub["mean_instance_gap"], marker="o", label=f"{dataset} mean gap")
        axes[1].plot(sub["n_states"], sub["mean_instance_hit_rate"], marker="o", label=f"{dataset} hit rate")
    axes[0].set_ylabel("Mean instance gap")
    axes[0].set_title("Multistate sweep")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Number of states")
    axes[1].set_ylabel("Mean instance hit rate")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multistate_sweep_plot.png", dpi=180)
    plt.close(fig)


def main() -> None:
    results = evaluate_sweep(STATE_COUNTS, runs_per_instance=RUNS_PER_INSTANCE, steps=STEPS)
    save_outputs(results)

    printable = {
        "state_counts": STATE_COUNTS,
        "runs_per_instance": RUNS_PER_INSTANCE,
        "best_by_dataset": results["best_by_dataset"],
    }
    print("MTJ multistate p-bit sweep finished.")
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from mtj_hardware_sa_demo import (
    GOOD_FILE,
    BAD_FILE,
    build_anneal_profile,
    brute_force_ising_optimum,
    load_dataset,
    run_hardware_sa,
    summarize_runs,
)


OUTPUT_DIR = Path("mtj_benchmark_output")


def build_spin_glass_instance(n_spins: int, density: float, seed: int, field_scale: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    upper = rng.integers(-3, 4, size=(n_spins, n_spins))
    mask = rng.random((n_spins, n_spins)) < density
    upper = np.triu(upper * mask, 1)
    J = upper + upper.T
    h = rng.integers(-field_scale, field_scale + 1, size=n_spins).astype(float)
    return J.astype(float), h


def build_max_cut_instance(n_nodes: int, density: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 6, size=(n_nodes, n_nodes))
    mask = rng.random((n_nodes, n_nodes)) < density
    upper = np.triu(weights * mask, 1)
    graph = upper + upper.T
    # Max-Cut can be written as minimizing -1/2 * s^T J s with J = -W.
    J = -graph.astype(float)
    h = np.zeros(n_nodes, dtype=float)
    return J, h


def build_benchmark_suite() -> list[dict]:
    suite = []
    configs = [
        ("spin_glass_sparse_12", "spin_glass", 12, 0.28, 101),
        ("spin_glass_dense_12", "spin_glass", 12, 0.55, 102),
        ("spin_glass_sparse_14", "spin_glass", 14, 0.25, 103),
        ("spin_glass_dense_14", "spin_glass", 14, 0.50, 104),
        ("spin_glass_sparse_16", "spin_glass", 16, 0.22, 105),
        ("spin_glass_dense_16", "spin_glass", 16, 0.45, 106),
        ("max_cut_14", "max_cut", 14, 0.35, 201),
        ("max_cut_16", "max_cut", 16, 0.30, 202),
    ]

    for name, kind, n, density, seed in configs:
        if kind == "spin_glass":
            J, h = build_spin_glass_instance(n, density, seed)
        else:
            J, h = build_max_cut_instance(n, density, seed)
        optimum = brute_force_ising_optimum(J, h)
        suite.append(
            {
                "name": name,
                "kind": kind,
                "n_spins": n,
                "density": density,
                "seed": seed,
                "J": J,
                "h": h,
                "optimum": optimum,
            }
        )
    return suite


def evaluate_suite(runs_per_instance: int = 30) -> dict:
    profiles = {
        "good": build_anneal_profile(load_dataset(GOOD_FILE, "good")),
        "bad": build_anneal_profile(load_dataset(BAD_FILE, "bad")),
    }
    suite = build_benchmark_suite()

    instance_rows = []
    aggregate = {name: [] for name in profiles}
    win_counter = {name: 0 for name in profiles}

    for instance in suite:
        instance_stats = {}
        for dataset_name, profile in profiles.items():
            runs = [
                run_hardware_sa(profile, instance["J"], instance["h"], instance["optimum"], run_idx=i)
                for i in range(runs_per_instance)
            ]
            stats = summarize_runs(runs)
            stats["dataset"] = dataset_name
            stats["instance"] = instance["name"]
            stats["kind"] = instance["kind"]
            stats["n_spins"] = instance["n_spins"]
            stats["density"] = instance["density"]
            stats["exact_optimum"] = float(instance["optimum"])
            instance_rows.append(stats)
            aggregate[dataset_name].append(stats)
            instance_stats[dataset_name] = stats

        ranking = sorted(
            profiles.keys(),
            key=lambda name: (
                instance_stats[name]["mean_energy_gap"],
                -instance_stats[name]["optimum_hit_rate"],
            ),
        )
        win_counter[ranking[0]] += 1

    overall = {}
    for dataset_name, stats_list in aggregate.items():
        mean_gap = np.mean([s["mean_energy_gap"] for s in stats_list])
        mean_hit = np.mean([s["optimum_hit_rate"] for s in stats_list])
        mean_best = np.mean([s["mean_best_energy"] for s in stats_list])
        mean_accept = np.mean([s["mean_acceptance_rate"] for s in stats_list])
        overall[dataset_name] = {
            "instances": len(stats_list),
            "instance_wins": int(win_counter[dataset_name]),
            "mean_instance_gap": float(mean_gap),
            "mean_instance_hit_rate": float(mean_hit),
            "mean_instance_best_energy": float(mean_best),
            "mean_instance_acceptance": float(mean_accept),
        }

    ranking = sorted(
        overall.keys(),
        key=lambda name: (
            overall[name]["mean_instance_gap"],
            -overall[name]["mean_instance_hit_rate"],
            -overall[name]["instance_wins"],
        ),
    )

    return {
        "profiles": profiles,
        "suite": suite,
        "instance_rows": instance_rows,
        "overall": overall,
        "ranking": ranking,
        "runs_per_instance": runs_per_instance,
    }


def save_outputs(results: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.DataFrame(results["instance_rows"])
    df.to_csv(OUTPUT_DIR / "instance_summary.csv", index=False)

    summary = {
        "assumption": (
            "Benchmark MTJ-derived annealing schedules on multiple exact-solvable Ising-style "
            "optimization instances and compare solution quality across datasets."
        ),
        "runs_per_instance": results["runs_per_instance"],
        "overall": results["overall"],
        "ranking": results["ranking"],
    }
    with open(OUTPUT_DIR / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is None:
        return

    plot_df = df.pivot(index="instance", columns="dataset", values="mean_energy_gap").sort_index()
    ax = plot_df.plot(kind="bar", figsize=(12, 5), rot=25)
    ax.set_ylabel("Mean energy gap")
    ax.set_title("MTJ dataset comparison across benchmark instances")
    ax.grid(True, axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT_DIR / "benchmark_gaps.png", dpi=180)
    plt.close(ax.figure)


def main() -> None:
    results = evaluate_suite(runs_per_instance=30)
    save_outputs(results)

    printable = {
        "runs_per_instance": results["runs_per_instance"],
        "overall": results["overall"],
        "ranking": results["ranking"],
    }
    print("MTJ benchmark suite finished.")
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

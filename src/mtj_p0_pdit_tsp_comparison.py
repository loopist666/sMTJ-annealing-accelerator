"""Compare the original one-hot Ising TSP route with a direct P-dit route.

This is a new comparison script.  Existing application scripts are imported
as the baseline and are not modified.

The P0 route is represented as a categorical permutation: route[position] is
the city at that position.  A random pair of nonfixed positions is selected,
and the current route and the swapped route form a two-state reachable set.
The next state is sampled with the paper's restricted P-dit heat-bath rule
(Duffee et al., PRApplied 24, 044077, Eq. 11).  Every state is therefore a
valid permutation; no Gray code and no one-hot penalty are required.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from mtj_ising_application_demos import (
    build_tsp_problem,
    decode_tsp,
    reference_tsp,
    route_length,
    run_kstate_ising_solver,
    two_opt_route,
)
from mtj_p0_direct_pdit_partition import P0PhysicalSource, build_p0_physical_source
from project_paths import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "mtj_p0_pdit_tsp_comparison_output"
PAPER_DOI = "10.1103/4ngx-cmz7"


@dataclass
class PditTspRun:
    run_idx: int
    best_route: list[int]
    final_route: list[int]
    best_length: float
    final_length: float
    two_opt_route: list[int]
    two_opt_length: float
    clock_ticks: int
    update_attempts: int
    changed_updates: int
    first_reference_hit_update: int | None
    wall_time_seconds: float


def safe_logistic_negative(delta_scaled: float) -> float:
    """Return 1/(1+exp(delta_scaled)) without overflow."""
    if delta_scaled >= 40.0:
        return 0.0
    if delta_scaled <= -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(delta_scaled))


def initialize_permutation(
    n_cities: int,
    source: P0PhysicalSource,
    offset: int,
) -> np.ndarray:
    """Create an sMTJ-keyed permutation with city 0 fixed at position 0."""
    keyed = []
    for city in range(1, n_cities):
        source_idx = (offset + 7 * city) % len(source.step_states)
        channel = city % source.random_u.shape[1]
        key = float(source.random_u[source_idx, channel])
        key += 1e-4 * int(source.step_states[source_idx])
        keyed.append((key, city))
    return np.array([0, *[city for _, city in sorted(keyed)]], dtype=np.int32)


def select_distinct_nonfixed_positions(row: np.ndarray, n_cities: int) -> tuple[int, int]:
    """Select an ordered pair from positions 1..N-1 using two uniforms."""
    movable = n_cities - 1
    first = 1 + min(int(float(row[0]) * movable), movable - 1)
    second_rank = min(int(float(row[1 % len(row)]) * (movable - 1)), movable - 2)
    second = 1 + second_rank
    if second >= first:
        second += 1
    return first, second


def run_permutation_pdit_tsp(
    distances: np.ndarray,
    reference_length: float,
    source: P0PhysicalSource,
    run_idx: int,
    update_budget: int,
    beta_start: float,
    beta_end: float,
    hazard_gain: float,
) -> PditTspRun:
    """Run permutation-preserving pairwise P-dit Glauber updates.

    For a proposed transposition, the reachable set contains the current and
    swapped routes.  The swap probability is

        exp(-beta*E_swap) / [exp(-beta*E_current)+exp(-beta*E_swap)]
      = 1 / [1 + exp(beta*(E_swap-E_current))].

    Random transpositions connect the entire fixed-origin permutation space,
    while the self-loop makes the chain aperiodic.
    """
    start_wall = time.perf_counter()
    n_cities = len(distances)
    source_steps = len(source.step_states)
    offset = (run_idx * 53) % source_steps
    route = initialize_permutation(n_cities, source, offset)
    current_length = route_length(route.tolist(), distances)
    best_length = current_length
    best_route = route.copy()
    first_reference_hit = 0 if current_length <= reference_length + 1e-9 else None
    update_attempts = 0
    changed_updates = 0
    clock_ticks = 0
    max_clock_ticks = max(update_budget * 50, source_steps)

    while update_attempts < update_budget and clock_ticks < max_clock_ticks:
        source_idx = (offset + clock_ticks) % source_steps
        physical_state = int(source.step_states[source_idx])
        row = source.random_u[source_idx]
        clock_ticks += 1

        if hazard_gain > 0.0:
            rate = max(float(source.exit_rates[physical_state]), 1e-12)
            clock_probability = 1.0 - math.exp(-hazard_gain * rate * source.dt_step)
            if float(row[-1]) >= clock_probability:
                continue

        pos_a, pos_b = select_distinct_nonfixed_positions(row, n_cities)
        candidate = route.copy()
        candidate[pos_a], candidate[pos_b] = candidate[pos_b], candidate[pos_a]
        candidate_length = route_length(candidate.tolist(), distances)

        progress = 0.0 if update_budget <= 1 else update_attempts / (update_budget - 1)
        beta = beta_start + (beta_end - beta_start) * progress
        swap_probability = safe_logistic_negative(beta * (candidate_length - current_length))
        accept_idx = (source_idx + 97 + 3 * pos_a + pos_b) % source_steps
        accept_channel = (pos_a + pos_b) % source.random_u.shape[1]
        accept_uniform = float(source.random_u[accept_idx, accept_channel])
        update_attempts += 1

        if accept_uniform < swap_probability:
            route = candidate
            current_length = candidate_length
            changed_updates += 1
            if current_length < best_length - 1e-12:
                best_length = current_length
                best_route = route.copy()
            if first_reference_hit is None and current_length <= reference_length + 1e-9:
                first_reference_hit = update_attempts

    if update_attempts != update_budget:
        raise RuntimeError("Physical clock did not provide enough P-dit updates within the safety limit.")
    if route[0] != 0 or sorted(route.tolist()) != list(range(n_cities)):
        raise AssertionError("P-dit route left the fixed-origin permutation state space.")

    refined_route, refined_length = two_opt_route(best_route.tolist(), distances)
    return PditTspRun(
        run_idx=run_idx,
        best_route=best_route.astype(int).tolist(),
        final_route=route.astype(int).tolist(),
        best_length=float(best_length),
        final_length=float(current_length),
        two_opt_route=refined_route,
        two_opt_length=float(refined_length),
        clock_ticks=clock_ticks,
        update_attempts=update_attempts,
        changed_updates=changed_updates,
        first_reference_hit_update=first_reference_hit,
        wall_time_seconds=float(time.perf_counter() - start_wall),
    )


def run_original_method(
    profile,
    problem,
    runs: int,
    cycles: int,
    reference_length: float,
) -> list[dict]:
    rows = []
    proposal_count = len(profile.step_states) * cycles * profile.state_biases.shape[1]
    for run_idx in range(runs):
        start_wall = time.perf_counter()
        result = run_kstate_ising_solver(profile, problem, run_idx=run_idx, cycles=cycles)
        decoded = decode_tsp(result.best_spins, problem.metadata)
        wall = time.perf_counter() - start_wall
        rows.append(
            {
                "method": "original_onehot_ising",
                "run_idx": run_idx,
                "route_length": float(decoded["route_length"]),
                "two_opt_length": float(decoded["two_opt_route_length"]),
                "route_ratio": float(decoded["route_length"] / reference_length),
                "two_opt_ratio": float(decoded["two_opt_route_length"] / reference_length),
                "raw_assignment_feasible": bool(decoded["feasible_raw_assignment"]),
                "onehot_violation": int(decoded["onehot_violation"]),
                "update_attempts": int(proposal_count),
                "clock_ticks": int(len(profile.step_states) * cycles),
                "changed_updates": int(round(result.acceptance_rate * proposal_count)),
                "change_rate": float(result.acceptance_rate),
                "reference_hit_raw": bool(decoded["route_length"] <= reference_length + 1e-9),
                "reference_hit_two_opt": bool(decoded["two_opt_route_length"] <= reference_length + 1e-9),
                "wall_time_seconds": float(wall),
                "route": decoded["route"],
                "two_opt_route": decoded["two_opt_route"],
            }
        )
    return rows


def run_pdit_method(
    source: P0PhysicalSource,
    distances: np.ndarray,
    runs: int,
    update_budget: int,
    reference_length: float,
    beta_start: float,
    beta_end: float,
    hazard_gain: float,
) -> list[dict]:
    rows = []
    for run_idx in range(runs):
        result = run_permutation_pdit_tsp(
            distances=distances,
            reference_length=reference_length,
            source=source,
            run_idx=run_idx,
            update_budget=update_budget,
            beta_start=beta_start,
            beta_end=beta_end,
            hazard_gain=hazard_gain,
        )
        rows.append(
            {
                "method": "p0_permutation_pdit",
                "run_idx": run_idx,
                "route_length": result.best_length,
                "two_opt_length": result.two_opt_length,
                "route_ratio": result.best_length / reference_length,
                "two_opt_ratio": result.two_opt_length / reference_length,
                "raw_assignment_feasible": True,
                "onehot_violation": 0,
                "update_attempts": result.update_attempts,
                "clock_ticks": result.clock_ticks,
                "changed_updates": result.changed_updates,
                "change_rate": result.changed_updates / max(result.update_attempts, 1),
                "reference_hit_raw": bool(result.best_length <= reference_length + 1e-9),
                "reference_hit_two_opt": bool(result.two_opt_length <= reference_length + 1e-9),
                "wall_time_seconds": result.wall_time_seconds,
                "route": result.best_route,
                "two_opt_route": result.two_opt_route,
                "first_reference_hit_update": result.first_reference_hit_update,
            }
        )
    return rows


def summarize_method(rows: list[dict], n_cities: int, variables: int) -> dict:
    frame = pd.DataFrame(rows)
    return {
        "n_cities": n_cities,
        "method": str(frame["method"].iloc[0]),
        "variables": variables,
        "runs": int(len(frame)),
        "raw_feasible_rate": float(frame["raw_assignment_feasible"].mean()),
        "median_onehot_violation": float(frame["onehot_violation"].median()),
        "best_route_ratio": float(frame["route_ratio"].min()),
        "median_route_ratio": float(frame["route_ratio"].median()),
        "best_two_opt_ratio": float(frame["two_opt_ratio"].min()),
        "median_two_opt_ratio": float(frame["two_opt_ratio"].median()),
        "raw_reference_hit_rate": float(frame["reference_hit_raw"].mean()),
        "two_opt_reference_hit_rate": float(frame["reference_hit_two_opt"].mean()),
        "mean_update_attempts": float(frame["update_attempts"].mean()),
        "mean_clock_ticks": float(frame["clock_ticks"].mean()),
        "mean_change_rate": float(frame["change_rate"].mean()),
        "mean_wall_time_seconds": float(frame["wall_time_seconds"].mean()),
    }


def state_space_comparison(n_cities: int) -> dict:
    fixed_permutations = math.factorial(n_cities - 1)
    log_binary_states = n_cities * n_cities * math.log(2.0)
    log_valid_fraction = math.log(fixed_permutations) - log_binary_states
    return {
        "original_binary_variables": n_cities * n_cities,
        "original_total_binary_states_log10": log_binary_states / math.log(10.0),
        "fixed_origin_valid_permutations": fixed_permutations,
        "original_fixed_origin_feasible_fraction": float(math.exp(log_valid_fraction)),
        "pdit_categorical_variables": n_cities,
        "pdit_reachable_states": fixed_permutations,
        "pdit_feasible_fraction": 1.0,
    }


def save_comparison_plot(summary_rows: list[dict], output_dir: Path) -> None:
    if plt is None:
        return
    frame = pd.DataFrame(summary_rows)
    cities = sorted(frame["n_cities"].unique())
    methods = ["original_onehot_ising", "p0_permutation_pdit"]
    labels = {"original_onehot_ising": "Original one-hot Ising", "p0_permutation_pdit": "P0 categorical P-dit"}
    colors = {"original_onehot_ising": "#64748b", "p0_permutation_pdit": "#d97732"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(cities), dtype=float)
    width = 0.36
    for method_idx, method in enumerate(methods):
        subset = frame[frame["method"] == method].set_index("n_cities").loc[cities]
        offset = (method_idx - 0.5) * width
        axes[0].bar(
            x + offset,
            subset["median_route_ratio"],
            width,
            color=colors[method],
            label=labels[method],
        )
        axes[1].bar(
            x + offset,
            subset["raw_feasible_rate"],
            width,
            color=colors[method],
            label=labels[method],
        )
        axes[2].bar(
            x + offset,
            subset["variables"],
            width,
            color=colors[method],
            label=labels[method],
        )

    axes[0].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Median route length / reference")
    axes[0].set_title("Route quality before 2-opt")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Raw assignment feasible rate")
    axes[1].set_title("Constraint validity")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Represented variables")
    axes[2].set_title("Representation size")
    for axis in axes:
        axis.set_xticks(x, [str(city) for city in cities])
        axis.set_xlabel("Cities")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "tsp_method_comparison.png", dpi=180)
    plt.close(fig)


def save_best_routes_plot(
    all_rows: list[dict],
    problem_metadata: dict,
    n_cities: int,
    output_dir: Path,
) -> None:
    if plt is None:
        return
    coords = np.array(problem_metadata["coords"], dtype=float)
    frame = pd.DataFrame([row for row in all_rows if row["n_cities"] == n_cities])
    methods = ["original_onehot_ising", "p0_permutation_pdit"]
    titles = ["Original one-hot Ising (decoded)", "P0 categorical P-dit (raw feasible)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, method, title in zip(axes, methods, titles):
        best = frame[frame["method"] == method].sort_values("route_length").iloc[0]
        route = list(best["route"])
        closed = route + [route[0]]
        axis.plot(coords[closed, 0], coords[closed, 1], color="#d97732", linewidth=1.8)
        axis.scatter(coords[:, 0], coords[:, 1], color="#2a6f97", s=65, zorder=3)
        for city, (x_val, y_val) in enumerate(coords):
            axis.text(x_val + 0.07, y_val + 0.07, str(city), fontsize=8)
        axis.set_title(f"{title}\nlength={best['route_length']:.3f}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "tsp_best_routes_comparison.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="8,12", help="Comma-separated city counts, each <=24.")
    # Match the published application demo's TSP comparison budget.
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--cycles", type=int, default=45)
    parser.add_argument("--profile-steps", type=int, default=900)
    parser.add_argument("--beta-start", type=float, default=0.08)
    parser.add_argument("--beta-end", type=float, default=4.0)
    parser.add_argument("--hazard-gain", type=float, default=8.0)
    parser.add_argument("--dataset", default="good")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(value) for value in args.sizes.split(",") if value.strip()]
    if not sizes or any(size < 4 or size > 24 for size in sizes):
        raise ValueError("Every TSP size must be between 4 and 24.")
    if args.runs < 1 or args.cycles < 1:
        raise ValueError("runs and cycles must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ``original_profile`` retains the legacy linear-rounded step_states used
    # by the application baseline. ``pdit_source`` uses the new dwell route.
    pdit_source, original_profile = build_p0_physical_source(
        dataset_name=args.dataset,
        steps=args.profile_steps,
    )
    update_budget = (
        len(original_profile.step_states)
        * args.cycles
        * original_profile.state_biases.shape[1]
    )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    reference_rows: list[dict] = []
    metadata_by_size = {}

    for n_cities in sizes:
        problem, _ = build_tsp_problem(n_cities)
        metadata_by_size[n_cities] = problem.metadata
        distances = np.array(problem.metadata["distances"], dtype=float)
        reference = reference_tsp(problem.metadata)
        reference_length = float(reference["route_length"])

        original_rows = run_original_method(
            profile=original_profile,
            problem=problem,
            runs=args.runs,
            cycles=args.cycles,
            reference_length=reference_length,
        )
        pdit_rows = run_pdit_method(
            source=pdit_source,
            distances=distances,
            runs=args.runs,
            update_budget=update_budget,
            reference_length=reference_length,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            hazard_gain=args.hazard_gain,
        )

        for row in [*original_rows, *pdit_rows]:
            all_rows.append({"n_cities": n_cities, "reference_length": reference_length, **row})
        summary_rows.append(summarize_method(original_rows, n_cities, n_cities * n_cities))
        summary_rows.append(summarize_method(pdit_rows, n_cities, n_cities))
        reference_rows.append(
            {
                "n_cities": n_cities,
                "reference_length": reference_length,
                "reference_method": reference["method"],
                "state_space": state_space_comparison(n_cities),
            }
        )

    serializable_rows = []
    for row in all_rows:
        serializable_rows.append(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    pd.DataFrame(serializable_rows).to_csv(output_dir / "tsp_comparison_runs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "tsp_comparison_summary.csv", index=False)

    summary = {
        "stage": "P0 TSP application comparison",
        "paper": {
            "doi": PAPER_DOI,
            "kernel": "restricted reachable-set P-dit heat bath (paper Eq. 11)",
            "swap_probability": "1 / (1 + exp(beta * (L_swap - L_current)))",
        },
        "comparison_design": {
            "dataset": args.dataset,
            "profile_steps": args.profile_steps,
            "runs_per_method": args.runs,
            "cycles": args.cycles,
            "equal_element_update_budget": update_budget,
            "original_representation": "N^2 binary spins with two one-hot penalties",
            "pdit_representation": "N categorical positions in a fixed-origin permutation",
            "pdit_uses_gray_code": False,
            "pdit_uses_onehot_penalty": False,
            "pdit_raw_route_always_feasible": True,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "hazard_gain": args.hazard_gain,
        },
        "resampling_validation": pdit_source.resampling_metrics,
        "references": reference_rows,
        "method_summary": summary_rows,
    }
    with open(output_dir / "tsp_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    save_comparison_plot(summary_rows, output_dir)
    save_best_routes_plot(all_rows, metadata_by_size[max(sizes)], max(sizes), output_dir)

    if not all(row["raw_assignment_feasible"] for row in all_rows if row["method"] == "p0_permutation_pdit"):
        raise AssertionError("A P0 P-dit route became infeasible.")

    print("P0 P-dit TSP comparison finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

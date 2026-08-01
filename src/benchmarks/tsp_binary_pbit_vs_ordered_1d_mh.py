"""End-to-end TSP comparison for binary p-bits and ordered 1D P-dits.

The baseline is the binary p-bit + one-hot Ising/QUBO path.  The proposed
method combines direct-permutation encoding with an ordered 1D
empirical-transition Metropolis-Hastings kernel.  This is intentionally a
complete-method comparison, not a fixed-encoding ablation.

The route is represented directly as a fixed-origin permutation.  The measured
K=6 selected-curve profile supplies the ordered device state and the empirical
proposal rate q_kl = r_k P_kl.  A k -> l proposal selects a symmetric route
transposition whose distance is set by |l-k|, then uses

    min(1, exp[-beta (L_new-L_old)] q_lk/q_kl)

for Metropolis-Hastings acceptance.  Every proposed and accepted route is a
valid permutation; no Gray code or one-hot penalty is used by the P-dit path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
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


OUTPUT_DIR = RESULTS_DIR / "benchmarks" / "tsp_binary_pbit_vs_ordered_1d_mh"
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
    proposal_events: int
    changed_updates: int
    proposed_distance_sum: float
    accepted_distance_sum: float
    initial_device_state: int
    final_device_state: int
    first_reference_hit_update: int | None
    wall_time_seconds: float


@dataclass(frozen=True)
class Ordered1DProposal:
    q_rates: np.ndarray
    dt: float
    event_probabilities: np.ndarray
    event_probability_cap: float


def build_ordered_1d_proposal(
    source: P0PhysicalSource,
    transition_matrix: np.ndarray,
    event_probability_cap: float,
) -> Ordered1DProposal:
    if source.selected_k != 6:
        raise ValueError(
            f"Ordered 1D empirical-transition TSP requires K=6, got K={source.selected_k}."
        )
    if not 0.0 < event_probability_cap <= 1.0:
        raise ValueError("event_probability_cap must be in (0, 1].")
    transition = np.asarray(transition_matrix, dtype=float)
    q_rates = source.exit_rates[:, None] * transition
    np.fill_diagonal(q_rates, 0.0)
    one_way = (q_rates > 0.0) & (q_rates.T <= 0.0)
    if np.any(one_way):
        raise ValueError(
            "Empirical q contains one-way edges; reverse-positive smoothing or edge filtering is required."
        )
    row_rates = q_rates.sum(axis=1)
    maximum_rate = float(np.max(row_rates))
    if maximum_rate <= 0.0:
        raise ValueError("Empirical q has no off-diagonal proposal events.")
    dt = event_probability_cap / maximum_rate
    event_probabilities = dt * row_rates
    return Ordered1DProposal(
        q_rates=q_rates,
        dt=float(dt),
        event_probabilities=event_probabilities,
        event_probability_cap=float(event_probability_cap),
    )


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


def ordered_1d_swap_positions(
    uniform: float,
    n_cities: int,
    state_from: int,
    state_to: int,
) -> tuple[int, int, int]:
    """Map an ordered device jump to a reversible fixed-origin transposition."""
    movable = n_cities - 1
    if movable < 2:
        raise ValueError("At least three cities are required for a nontrivial fixed-origin swap.")
    first = 1 + min(int(float(uniform) * movable), movable - 1)
    raw_distance = max(abs(int(state_to) - int(state_from)), 1)
    distance = 1 + ((raw_distance - 1) % (movable - 1))
    second = 1 + ((first - 1 + distance) % movable)
    if first == second:
        raise AssertionError("Ordered 1D move produced a self-swap.")
    return first, second, distance


def validate_ordered_1d_mh_kernel(
    proposal: Ordered1DProposal,
    city_sizes: list[int],
    beta_start: float,
    beta_end: float,
) -> dict:
    """Validate route-move reversibility and the analytic MH flux identity."""
    move_reversible = True
    for n_cities in city_sizes:
        for state_from in range(len(proposal.q_rates)):
            for state_to in range(len(proposal.q_rates)):
                if state_from == state_to:
                    continue
                for uniform in np.linspace(0.0, 1.0, 41, endpoint=False):
                    forward = ordered_1d_swap_positions(
                        float(uniform), n_cities, state_from, state_to
                    )
                    reverse = ordered_1d_swap_positions(
                        float(uniform), n_cities, state_to, state_from
                    )
                    move_reversible &= forward == reverse

    max_relative_flux_error = 0.0
    checked = 0
    beta_values = [beta_start, 0.5 * (beta_start + beta_end), beta_end]
    delta_values = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0]
    for state_from in range(len(proposal.q_rates)):
        for state_to in range(len(proposal.q_rates)):
            q_forward = float(proposal.q_rates[state_from, state_to])
            q_reverse = float(proposal.q_rates[state_to, state_from])
            if q_forward <= 0.0 or q_reverse <= 0.0:
                continue
            for beta in beta_values:
                for delta in delta_values:
                    ratio = math.exp(float(np.clip(-beta * delta, -745.0, 709.0)))
                    forward_flux = q_forward * min(1.0, ratio * q_reverse / q_forward)
                    reverse_flux = (
                        math.exp(-beta * delta)
                        * q_reverse
                        * min(1.0, (1.0 / ratio) * q_forward / q_reverse)
                    )
                    relative = abs(forward_flux - reverse_flux) / max(
                        forward_flux + reverse_flux, 1e-300
                    )
                    max_relative_flux_error = max(max_relative_flux_error, relative)
                    checked += 1

    adjacent_edges = all(
        proposal.q_rates[state, state + 1] > 0.0
        and proposal.q_rates[state + 1, state] > 0.0
        for state in range(len(proposal.q_rates) - 1)
    )
    result = {
        "route_move_reversible_under_k_l_swap": bool(move_reversible),
        "checked_flux_cases": checked,
        "theoretical_db_max_relative_flux_error": max_relative_flux_error,
        "all_ordered_adjacent_device_edges_bidirectional": bool(adjacent_edges),
        "permutation_ergodicity_argument": (
            "distance-1 swaps occur with positive probability and adjacent transpositions "
            "generate every fixed-origin permutation"
        ),
    }
    if not move_reversible or not adjacent_edges or max_relative_flux_error > 1e-12:
        raise AssertionError(f"Ordered 1D MH kernel validation failed: {result}")
    return result


def run_permutation_pdit_tsp(
    distances: np.ndarray,
    reference_length: float,
    source: P0PhysicalSource,
    proposal: Ordered1DProposal,
    run_idx: int,
    update_budget: int,
    beta_start: float,
    beta_end: float,
) -> PditTspRun:
    """Run permutation-preserving ordered 1D empirical-transition MH."""
    start_wall = time.perf_counter()
    n_cities = len(distances)
    source_steps = len(source.step_states)
    offset = (run_idx * 53) % source_steps
    route = initialize_permutation(n_cities, source, offset)
    current_length = route_length(route.tolist(), distances)
    best_length = current_length
    best_route = route.copy()
    first_reference_hit = 0 if current_length <= reference_length + 1e-9 else None
    initial_device_state = int(source.step_states[offset])
    device_state = initial_device_state
    proposal_events = 0
    changed_updates = 0
    proposed_distance_sum = 0.0
    accepted_distance_sum = 0.0
    clock_ticks = 0

    while clock_ticks < update_budget:
        source_idx = (offset + clock_ticks) % source_steps
        clock_ticks += 1
        q_row = proposal.q_rates[device_state]
        row_rate = float(q_row.sum())
        event_uniform = float(source.random_u[source_idx, 0])
        if row_rate <= 0.0 or event_uniform >= proposal.dt * row_rate:
            continue

        target_idx = (source_idx + 31) % source_steps
        target_uniform = float(source.random_u[target_idx, 1 % source.random_u.shape[1]])
        conditional = q_row / row_rate
        proposed_state = int(
            min(
                np.searchsorted(np.cumsum(conditional), target_uniform, side="right"),
                source.selected_k - 1,
            )
        )
        if proposed_state == device_state:
            raise AssertionError("Empirical off-diagonal proposal selected the current state.")

        position_idx = (source_idx + 67) % source_steps
        position_uniform = float(source.random_u[position_idx, 2 % source.random_u.shape[1]])
        pos_a, pos_b, move_distance = ordered_1d_swap_positions(
            position_uniform,
            n_cities,
            device_state,
            proposed_state,
        )
        candidate = route.copy()
        candidate[pos_a], candidate[pos_b] = candidate[pos_b], candidate[pos_a]
        candidate_length = route_length(candidate.tolist(), distances)
        proposal_events += 1
        proposed_distance_sum += move_distance

        progress = 0.0 if update_budget <= 1 else (clock_ticks - 1) / (update_budget - 1)
        beta = beta_start + (beta_end - beta_start) * progress
        delta = candidate_length - current_length
        q_forward = float(proposal.q_rates[device_state, proposed_state])
        q_reverse = float(proposal.q_rates[proposed_state, device_state])
        boltzmann_ratio = math.exp(float(np.clip(-beta * delta, -745.0, 709.0)))
        acceptance = min(1.0, boltzmann_ratio * q_reverse / max(q_forward, 1e-300))
        accept_idx = (source_idx + 101) % source_steps
        accept_uniform = float(source.random_u[accept_idx, -1])

        if accept_uniform < acceptance:
            route = candidate
            current_length = candidate_length
            device_state = proposed_state
            changed_updates += 1
            accepted_distance_sum += move_distance
            if current_length < best_length - 1e-12:
                best_length = current_length
                best_route = route.copy()
            if first_reference_hit is None and current_length <= reference_length + 1e-9:
                first_reference_hit = clock_ticks

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
        update_attempts=clock_ticks,
        proposal_events=proposal_events,
        changed_updates=changed_updates,
        proposed_distance_sum=proposed_distance_sum,
        accepted_distance_sum=accepted_distance_sum,
        initial_device_state=initial_device_state,
        final_device_state=device_state,
        first_reference_hit_update=first_reference_hit,
        wall_time_seconds=float(time.perf_counter() - start_wall),
    )


def run_binary_pbit_onehot_method(
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
                "method": "binary_pbit_onehot_ising",
                "run_idx": run_idx,
                "route_length": float(decoded["route_length"]),
                "two_opt_length": float(decoded["two_opt_route_length"]),
                "route_ratio": float(decoded["route_length"] / reference_length),
                "two_opt_ratio": float(decoded["two_opt_route_length"] / reference_length),
                "raw_assignment_feasible": bool(decoded["feasible_raw_assignment"]),
                "onehot_violation": int(decoded["onehot_violation"]),
                "update_attempts": int(proposal_count),
                "clock_ticks": int(len(profile.step_states) * cycles),
                "proposal_events": int(proposal_count),
                "changed_updates": int(round(result.acceptance_rate * proposal_count)),
                "change_rate": float(result.acceptance_rate),
                "proposal_event_rate": 1.0,
                "mh_acceptance_given_event": float(result.acceptance_rate),
                "mean_proposed_1d_distance": float("nan"),
                "mean_accepted_1d_distance": float("nan"),
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
    proposal: Ordered1DProposal,
    distances: np.ndarray,
    runs: int,
    update_budget: int,
    reference_length: float,
    beta_start: float,
    beta_end: float,
) -> list[dict]:
    rows = []
    for run_idx in range(runs):
        result = run_permutation_pdit_tsp(
            distances=distances,
            reference_length=reference_length,
            source=source,
            proposal=proposal,
            run_idx=run_idx,
            update_budget=update_budget,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        rows.append(
            {
                "method": "ordered_1d_empirical_mh_pdit",
                "run_idx": run_idx,
                "route_length": result.best_length,
                "two_opt_length": result.two_opt_length,
                "route_ratio": result.best_length / reference_length,
                "two_opt_ratio": result.two_opt_length / reference_length,
                "raw_assignment_feasible": True,
                "onehot_violation": 0,
                "update_attempts": result.update_attempts,
                "clock_ticks": result.clock_ticks,
                "proposal_events": result.proposal_events,
                "changed_updates": result.changed_updates,
                "change_rate": result.changed_updates / max(result.update_attempts, 1),
                "proposal_event_rate": result.proposal_events / max(result.clock_ticks, 1),
                "mh_acceptance_given_event": (
                    result.changed_updates / max(result.proposal_events, 1)
                ),
                "mean_proposed_1d_distance": (
                    result.proposed_distance_sum / max(result.proposal_events, 1)
                ),
                "mean_accepted_1d_distance": (
                    result.accepted_distance_sum / max(result.changed_updates, 1)
                ),
                "reference_hit_raw": bool(result.best_length <= reference_length + 1e-9),
                "reference_hit_two_opt": bool(result.two_opt_length <= reference_length + 1e-9),
                "wall_time_seconds": result.wall_time_seconds,
                "route": result.best_route,
                "two_opt_route": result.two_opt_route,
                "first_reference_hit_update": result.first_reference_hit_update,
                "initial_device_state": result.initial_device_state,
                "final_device_state": result.final_device_state,
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
        "mean_proposal_event_rate": float(frame["proposal_event_rate"].mean()),
        "mean_mh_acceptance_given_event": float(frame["mh_acceptance_given_event"].mean()),
        "mean_proposed_1d_distance": float(frame["mean_proposed_1d_distance"].mean()),
        "mean_accepted_1d_distance": float(frame["mean_accepted_1d_distance"].mean()),
        "mean_wall_time_seconds": float(frame["wall_time_seconds"].mean()),
    }


def state_space_comparison(n_cities: int) -> dict:
    fixed_permutations = math.factorial(n_cities - 1)
    log_binary_states = n_cities * n_cities * math.log(2.0)
    log_valid_fraction = math.log(fixed_permutations) - log_binary_states
    return {
        "binary_onehot_variables": n_cities * n_cities,
        "binary_onehot_total_states_log10": log_binary_states / math.log(10.0),
        "fixed_origin_valid_permutations": fixed_permutations,
        "binary_onehot_fixed_origin_feasible_fraction": float(math.exp(log_valid_fraction)),
        "pdit_categorical_variables": n_cities,
        "pdit_reachable_states": fixed_permutations,
        "pdit_feasible_fraction": 1.0,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def save_comparison_plot(summary_rows: list[dict], output_dir: Path) -> None:
    if plt is None:
        return
    frame = pd.DataFrame(summary_rows)
    cities = sorted(frame["n_cities"].unique())
    methods = ["binary_pbit_onehot_ising", "ordered_1d_empirical_mh_pdit"]
    labels = {
        "binary_pbit_onehot_ising": "Binary p-bit + one-hot Ising",
        "ordered_1d_empirical_mh_pdit": "Ordered 1D empirical-MH P-dit",
    }
    colors = {"binary_pbit_onehot_ising": "#64748b", "ordered_1d_empirical_mh_pdit": "#d97732"}

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
    methods = ["binary_pbit_onehot_ising", "ordered_1d_empirical_mh_pdit"]
    titles = ["Binary p-bit + one-hot Ising (decoded)", "Ordered 1D empirical-MH P-dit"]
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
    parser.add_argument("--event-probability-cap", type=float, default=0.5)
    parser.add_argument("--dataset", default="selected_curve")
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

    # The baseline uses the existing K-state/Gray application solver.  The
    # direct route uses the same K=6 fitted profile through q_kl=r_k P_kl.
    pdit_source, binary_profile = build_p0_physical_source(
        dataset_name=args.dataset,
        steps=args.profile_steps,
    )
    proposal = build_ordered_1d_proposal(
        pdit_source,
        binary_profile.transition_matrix,
        event_probability_cap=args.event_probability_cap,
    )
    kernel_validation = validate_ordered_1d_mh_kernel(
        proposal,
        city_sizes=sizes,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    update_budget = (
        len(binary_profile.step_states)
        * args.cycles
        * binary_profile.state_biases.shape[1]
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

        binary_rows = run_binary_pbit_onehot_method(
            profile=binary_profile,
            problem=problem,
            runs=args.runs,
            cycles=args.cycles,
            reference_length=reference_length,
        )
        pdit_rows = run_pdit_method(
            source=pdit_source,
            proposal=proposal,
            distances=distances,
            runs=args.runs,
            update_budget=update_budget,
            reference_length=reference_length,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        )

        for row in [*binary_rows, *pdit_rows]:
            all_rows.append({"n_cities": n_cities, "reference_length": reference_length, **row})
        summary_rows.append(summarize_method(binary_rows, n_cities, n_cities * n_cities))
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
        "stage": "P1 ordered 1D empirical-transition TSP comparison",
        "paper": {
            "doi": PAPER_DOI,
            "kernel": "ordered 1D empirical-transition Metropolis-Hastings",
            "proposal": "q_kl = r_k P_kl for k != l",
            "acceptance": "min(1, exp(-beta * DeltaL) * q_lk/q_kl)",
        },
        "comparison_design": {
            "dataset": args.dataset,
            "profile_steps": args.profile_steps,
            "runs_per_method": args.runs,
            "cycles": args.cycles,
            "equal_element_update_budget": update_budget,
            "binary_pbit_representation": "N^2 binary spins with two one-hot penalties",
            "pdit_representation": "N categorical positions in a fixed-origin permutation",
            "pdit_uses_gray_code": False,
            "pdit_uses_onehot_penalty": False,
            "pdit_raw_route_always_feasible": True,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "event_probability_cap": args.event_probability_cap,
            "proposal_clock_dt": proposal.dt,
            "event_probabilities_by_state": proposal.event_probabilities.tolist(),
            "ordered_move": (
                "device jump |l-k| sets a cyclic transposition distance among nonfixed positions"
            ),
            "mh_state": "joint route permutation and ordered device state",
        },
        "empirical_transition_proposal": {
            "exit_rates": pdit_source.exit_rates.tolist(),
            "transition_matrix": binary_profile.transition_matrix.tolist(),
            "q_rates": proposal.q_rates.tolist(),
            "all_positive_edges_have_positive_reverse": bool(
                np.all(~((proposal.q_rates > 0.0) & (proposal.q_rates.T <= 0.0)))
            ),
        },
        "kernel_validation": kernel_validation,
        "resampling_validation": pdit_source.resampling_metrics,
        "references": reference_rows,
        "method_summary": summary_rows,
    }
    safe_summary = json_safe(summary)
    with open(output_dir / "tsp_comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(safe_summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    save_comparison_plot(summary_rows, output_dir)
    save_best_routes_plot(all_rows, metadata_by_size[max(sizes)], max(sizes), output_dir)

    if not all(
        row["raw_assignment_feasible"]
        for row in all_rows
        if row["method"] == "ordered_1d_empirical_mh_pdit"
    ):
        raise AssertionError("An ordered 1D empirical-MH P-dit route became infeasible.")

    print("Ordered 1D empirical-transition P-dit TSP comparison finished.")
    print(json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

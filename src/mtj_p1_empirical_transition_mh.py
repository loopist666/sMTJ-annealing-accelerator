"""P1: empirical-transition Metropolis-Hastings for a measured K=6 sMTJ.

This standalone benchmark uses both measured quantities

    q_kl = r_k P_kl

as the actual state-pair proposal rate.  For a proposed k -> l transition the
acceptance probability is

    min(1, exp[-beta (E_l - E_k)] q_lk / q_kl).

The same K=6 physical profile and proposal clock are tested under three
computational state geometries:

1. Gray-coded binary: six vertices of a three-bit Gray path;
2. ordered 1D P-dit: six evenly ordered scalar values;
3. isotropic six-state P-dit: six orthogonal one-hot states.

Each geometry induces a normalized four-variable pairwise test energy.  Its
6^4=1296 states are enumerated exactly, so stationary KL, ground states, and
theoretical/empirical detailed-balance errors are available without an oracle.
Geometry compatibility is also measured directly from q_kl and the three
distance matrices; it is not inferred only from optimization performance.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from mtj_p0_direct_pdit_partition import build_p0_physical_source, stable_softmax
from mtj_physical_kstate_demo import gray_code_table
from project_paths import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "mtj_p1_empirical_transition_mh_output"
GEOMETRY_ORDER = ["gray_binary", "ordered_1d", "isotropic_6state"]
GEOMETRY_LABELS = {
    "gray_binary": "Gray-coded binary",
    "ordered_1d": "Ordered 1D P-dit",
    "isotropic_6state": "Isotropic six-state P-dit",
}
GEOMETRY_COLORS = {
    "gray_binary": "#4059AD",
    "ordered_1d": "#D95F02",
    "isotropic_6state": "#2CA25F",
}


@dataclass(frozen=True)
class P1Config:
    n_variables: int = 4
    n_states: int = 6
    beta: float = 2.5
    coupling: float = 1.0
    field_strength: float = 0.55
    runs: int = 64
    updates: int = 8000
    burn_fraction: float = 0.25
    event_probability_cap: float = 0.50
    seed: int = 20260718
    dataset: str = "selected_curve"


@dataclass
class PhysicalProposalProfile:
    selected_k: int
    transition_matrix: np.ndarray
    exit_rates: np.ndarray
    q_rates: np.ndarray
    dt: float
    event_probabilities: np.ndarray
    dwell_count: int
    state_means: np.ndarray
    resampling_metrics: dict


@dataclass
class GeometryProblem:
    name: str
    embedding: np.ndarray
    distance: np.ndarray
    squared_distance: np.ndarray
    states: np.ndarray
    raw_energies: np.ndarray
    energies: np.ndarray
    energy_scale: float
    probabilities: np.ndarray
    powers: np.ndarray
    ground_energy: float
    ground_indices: np.ndarray


@dataclass
class TraceResult:
    geometry: str
    run_idx: int
    state_index: np.ndarray
    energy: np.ndarray
    best_energy: np.ndarray
    first_hit_update: int | None
    proposal_events: int
    accepted_events: int
    changed_events: int
    proposal_distance_sum: float
    accepted_distance_sum: float
    wall_time_seconds: float


def build_physical_profile(config: P1Config) -> PhysicalProposalProfile:
    source, base = build_p0_physical_source(
        dataset_name=config.dataset,
        steps=max(config.updates, 512),
        max_k=8,
    )
    if source.selected_k != config.n_states:
        raise ValueError(
            f"P1 requires the same K=6 profile, but {config.dataset!r} selected "
            f"K={source.selected_k}."
        )
    transition = np.asarray(base.transition_matrix, dtype=float).copy()
    q_rates = source.exit_rates[:, None] * transition
    np.fill_diagonal(q_rates, 0.0)
    row_rates = q_rates.sum(axis=1)
    dt = config.event_probability_cap / max(float(np.max(row_rates)), 1e-15)
    event_probabilities = dt * row_rates
    if np.max(event_probabilities) > 1.0 + 1e-12:
        raise AssertionError("Discrete proposal event probability exceeds one.")
    return PhysicalProposalProfile(
        selected_k=source.selected_k,
        transition_matrix=transition,
        exit_rates=source.exit_rates.copy(),
        q_rates=q_rates,
        dt=float(dt),
        event_probabilities=event_probabilities,
        dwell_count=len(source.dwell_segments),
        state_means=base.state_means.copy(),
        resampling_metrics=source.resampling_metrics,
    )


def build_embeddings(n_states: int) -> dict[str, np.ndarray]:
    if n_states != 6:
        raise ValueError("The P1 geometry comparison is defined for K=6.")
    gray = gray_code_table(n_states) / math.sqrt(3.0)
    ordered = np.linspace(-1.0, 1.0, n_states, dtype=float)[:, None]
    isotropic = np.eye(n_states, dtype=float)
    return {
        "gray_binary": gray,
        "ordered_1d": ordered,
        "isotropic_6state": isotropic,
    }


def pairwise_distance(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = embedding[:, None, :] - embedding[None, :, :]
    squared = np.sum(delta * delta, axis=2)
    distance = np.sqrt(np.maximum(squared, 0.0))
    max_squared = max(float(np.max(squared)), 1e-15)
    max_distance = max(float(np.max(distance)), 1e-15)
    return distance / max_distance, squared / max_squared


def raw_geometry_energy(
    state: np.ndarray,
    squared_distance: np.ndarray,
    config: P1Config,
) -> float:
    # A four-node ring plus weak opposing chords creates a nontrivial but fully
    # enumerable landscape.  Site preferences remove otherwise large global
    # symmetry degeneracies.  The distance matrix is normalized to [0,1].
    ring_edges = [(idx, (idx + 1) % config.n_variables) for idx in range(config.n_variables)]
    chord_edges = [(0, 2), (1, 3)] if config.n_variables == 4 else []
    preferences = np.asarray([0, 2, 5, 3], dtype=np.int8)[: config.n_variables]
    energy = config.coupling * sum(
        squared_distance[int(state[left]), int(state[right])] for left, right in ring_edges
    )
    energy += 0.35 * config.coupling * sum(
        squared_distance[int(state[left]), int(state[right])] for left, right in chord_edges
    )
    energy += config.field_strength * sum(
        squared_distance[int(category), int(preference)]
        for category, preference in zip(state, preferences)
    )
    return float(energy)


def build_geometry_problem(
    name: str,
    embedding: np.ndarray,
    config: P1Config,
) -> GeometryProblem:
    distance, squared_distance = pairwise_distance(embedding)
    states = np.asarray(
        list(itertools.product(range(config.n_states), repeat=config.n_variables)),
        dtype=np.int8,
    )
    raw_energies = np.asarray(
        [raw_geometry_energy(state, squared_distance, config) for state in states],
        dtype=float,
    )
    raw_minimum = float(np.min(raw_energies))
    energy_scale = float(np.std(raw_energies))
    if energy_scale <= 1e-15:
        raise ValueError(f"Geometry {name} produced a degenerate energy landscape.")
    energies = (raw_energies - raw_minimum) / energy_scale
    probabilities = stable_softmax(-config.beta * energies)
    powers = config.n_states ** np.arange(
        config.n_variables - 1, -1, -1, dtype=np.int64
    )
    ground = float(np.min(energies))
    return GeometryProblem(
        name=name,
        embedding=embedding,
        distance=distance,
        squared_distance=squared_distance,
        states=states,
        raw_energies=raw_energies,
        energies=energies,
        energy_scale=energy_scale,
        probabilities=probabilities,
        powers=powers,
        ground_energy=ground,
        ground_indices=np.flatnonzero(np.isclose(energies, ground, atol=1e-12)),
    )


def state_index(state: np.ndarray, problem: GeometryProblem) -> int:
    return int(np.dot(state.astype(np.int64), problem.powers))


def empirical_transition_mh(
    geometry: GeometryProblem,
    profile: PhysicalProposalProfile,
    config: P1Config,
    initial_state: np.ndarray,
    run_idx: int,
) -> TraceResult:
    """Run q_kl=r_k P_kl as the actual proposal, followed by MH."""
    seed_offset = GEOMETRY_ORDER.index(geometry.name) * 10_000_019
    rng = np.random.default_rng(config.seed + seed_offset + 100_003 * (run_idx + 1))
    state = initial_state.copy()
    current_index = state_index(state, geometry)
    current_energy = float(geometry.energies[current_index])
    trace_index = np.empty(config.updates, dtype=np.int32)
    trace_energy = np.empty(config.updates, dtype=float)
    trace_best = np.empty(config.updates, dtype=float)
    best = current_energy
    first_hit = None
    proposal_events = 0
    accepted_events = 0
    changed_events = 0
    proposal_distance_sum = 0.0
    accepted_distance_sum = 0.0
    start = time.perf_counter()

    for update in range(config.updates):
        variable = int(rng.integers(config.n_variables))
        current_category = int(state[variable])
        q_row = profile.q_rates[current_category]
        row_rate = float(q_row.sum())

        if row_rate > 0.0 and rng.random() < profile.dt * row_rate:
            proposal_events += 1
            conditional = q_row / row_rate
            proposed_category = int(rng.choice(config.n_states, p=conditional))
            if proposed_category != current_category:
                changed_events += 1
                jump_distance = float(
                    geometry.distance[current_category, proposed_category]
                )
                proposal_distance_sum += jump_distance
                proposed_state = state.copy()
                proposed_state[variable] = proposed_category
                proposed_index = state_index(proposed_state, geometry)
                proposed_energy = float(geometry.energies[proposed_index])
                delta_energy = proposed_energy - current_energy
                q_forward = float(profile.q_rates[current_category, proposed_category])
                q_reverse = float(profile.q_rates[proposed_category, current_category])
                boltzmann_ratio = math.exp(
                    float(np.clip(-config.beta * delta_energy, -745.0, 709.0))
                )
                acceptance = min(
                    1.0,
                    boltzmann_ratio * q_reverse / max(q_forward, 1e-300),
                )
                if rng.random() < acceptance:
                    state = proposed_state
                    current_index = proposed_index
                    current_energy = proposed_energy
                    accepted_events += 1
                    accepted_distance_sum += jump_distance

        best = min(best, current_energy)
        if first_hit is None and current_energy <= geometry.ground_energy + 1e-12:
            first_hit = update + 1
        trace_index[update] = current_index
        trace_energy[update] = current_energy
        trace_best[update] = best

    return TraceResult(
        geometry=geometry.name,
        run_idx=run_idx,
        state_index=trace_index,
        energy=trace_energy,
        best_energy=trace_best,
        first_hit_update=first_hit,
        proposal_events=proposal_events,
        accepted_events=accepted_events,
        changed_events=changed_events,
        proposal_distance_sum=proposal_distance_sum,
        accepted_distance_sum=accepted_distance_sum,
        wall_time_seconds=time.perf_counter() - start,
    )


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def geometry_compatibility(
    geometry: GeometryProblem,
    profile: PhysicalProposalProfile,
) -> dict:
    k = profile.selected_k
    mask = ~np.eye(k, dtype=bool)
    q = profile.q_rates.copy()
    q_weights = q[mask] / max(float(q[mask].sum()), 1e-15)
    distances = geometry.distance[mask]
    symmetric_rate = 0.5 * (q + q.T)
    upper = np.triu_indices(k, k=1)
    rate_values = symmetric_rate[upper]
    distance_values = geometry.distance[upper]

    nearest_mass = []
    direction_kl = []
    direction_entropy = []
    for state in range(k):
        offdiag = np.arange(k) != state
        row = q[state, offdiag]
        conditional = row / max(float(row.sum()), 1e-15)
        row_distances = geometry.distance[state, offdiag]
        min_distance = float(np.min(row_distances))
        nearest_mass.append(float(np.sum(conditional[np.isclose(row_distances, min_distance)])))
        uniform = 1.0 / (k - 1)
        positive = conditional > 0.0
        direction_kl.append(
            float(np.sum(conditional[positive] * np.log(conditional[positive] / uniform)))
        )
        direction_entropy.append(
            float(-np.sum(conditional[positive] * np.log(conditional[positive])) / math.log(k - 1))
        )

    mean_distance = float(np.sum(q_weights * distances))
    return {
        "geometry": geometry.name,
        "label": GEOMETRY_LABELS[geometry.name],
        "embedding_dimension": int(geometry.embedding.shape[1]),
        "distinct_nonzero_distances": int(
            len(np.unique(np.round(geometry.distance[mask], decimals=12)))
        ),
        "q_weighted_mean_normalized_distance": mean_distance,
        "proposal_locality_score": 1.0 - mean_distance,
        "mean_nearest_geometry_transition_mass": float(np.mean(nearest_mass)),
        "mean_direction_entropy_fraction": float(np.mean(direction_entropy)),
        "mean_direction_kl_from_uniform": float(np.mean(direction_kl)),
        "symmetric_rate_vs_negative_distance_pearson": safe_correlation(
            rate_values, -distance_values
        ),
        "symmetric_rate_vs_negative_distance_spearman": safe_correlation(
            rankdata_average(rate_values), rankdata_average(-distance_values)
        ),
    }


def integrated_autocorrelation_time(values: np.ndarray, max_lag: int = 1000) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return float("nan")
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered * centered))
    if variance <= 1e-15:
        return 1.0
    n_fft = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=n_fft)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[: len(centered)]
    autocov /= np.arange(len(centered), 0, -1)
    autocorrelation = autocov / autocov[0]
    positive = []
    for lag in range(1, min(max_lag, len(values) - 1) + 1):
        if autocorrelation[lag] <= 0.0:
            break
        positive.append(float(autocorrelation[lag]))
    return float(1.0 + 2.0 * np.sum(positive))


def exact_detailed_balance_validation(
    problem: GeometryProblem,
    profile: PhysicalProposalProfile,
    config: P1Config,
) -> dict:
    max_abs_error = 0.0
    max_relative_error = 0.0
    mean_relative_errors = []
    checked_pairs = 0
    for index, state in enumerate(problem.states):
        pi_left = float(problem.probabilities[index])
        energy_left = float(problem.energies[index])
        for variable in range(config.n_variables):
            current = int(state[variable])
            for proposed in range(config.n_states):
                if proposed == current:
                    continue
                q_forward = float(profile.q_rates[current, proposed])
                q_reverse = float(profile.q_rates[proposed, current])
                if q_forward <= 0.0 or q_reverse <= 0.0:
                    continue
                next_state = state.copy()
                next_state[variable] = proposed
                next_index = state_index(next_state, problem)
                energy_right = float(problem.energies[next_index])
                pi_right = float(problem.probabilities[next_index])
                proposal_forward = profile.dt * q_forward / config.n_variables
                proposal_reverse = profile.dt * q_reverse / config.n_variables
                ratio_forward = math.exp(
                    float(np.clip(-config.beta * (energy_right - energy_left), -745.0, 709.0))
                ) * q_reverse / q_forward
                ratio_reverse = math.exp(
                    float(np.clip(-config.beta * (energy_left - energy_right), -745.0, 709.0))
                ) * q_forward / q_reverse
                flux_forward = pi_left * proposal_forward * min(1.0, ratio_forward)
                flux_reverse = pi_right * proposal_reverse * min(1.0, ratio_reverse)
                error = abs(flux_forward - flux_reverse)
                relative = error / max(flux_forward + flux_reverse, 1e-300)
                max_abs_error = max(max_abs_error, error)
                max_relative_error = max(max_relative_error, relative)
                mean_relative_errors.append(relative)
                checked_pairs += 1
    return {
        "theoretical_db_checked_directed_pairs": checked_pairs,
        "theoretical_db_max_abs_flux_error": max_abs_error,
        "theoretical_db_max_relative_flux_error": max_relative_error,
        "theoretical_db_mean_relative_flux_error": float(np.mean(mean_relative_errors)),
    }


def post_burn_diagnostics(
    traces: list[TraceResult],
    problem: GeometryProblem,
    config: P1Config,
) -> dict:
    burn = int(config.updates * config.burn_fraction)
    counts = np.zeros(len(problem.states), dtype=np.int64)
    transition_counts = np.zeros((len(problem.states), len(problem.states)), dtype=np.int64)
    iats = []
    for trace in traces:
        indices = trace.state_index[burn:]
        counts += np.bincount(indices, minlength=len(problem.states))
        np.add.at(transition_counts, (indices[:-1], indices[1:]), 1)
        iats.append(integrated_autocorrelation_time(trace.energy[burn:]))
    empirical = counts.astype(float) / max(float(counts.sum()), 1.0)
    positive = empirical > 0.0
    kl = float(
        np.sum(empirical[positive] * np.log(empirical[positive] / problem.probabilities[positive]))
    )
    tv = float(0.5 * np.sum(np.abs(empirical - problem.probabilities)))
    flux = transition_counts.astype(float) / max(float(transition_counts.sum()), 1.0)
    empirical_db = float(0.5 * np.sum(np.abs(flux - flux.T)))
    mean_iat = float(np.nanmean(iats))
    total_post_burn = len(traces) * (config.updates - burn)
    return {
        "stationary_kl_empirical_to_exact": kl,
        "stationary_total_variation": tv,
        "empirical_detailed_balance_flux_error": empirical_db,
        "mean_energy_iat_updates": mean_iat,
        "median_energy_iat_updates": float(np.nanmedian(iats)),
        "effective_energy_samples": total_post_burn / max(mean_iat, 1e-15),
        "post_burn_unique_state_fraction": float(np.mean(counts > 0)),
        "post_burn_samples": int(total_post_burn),
        "empirical_distribution": empirical,
    }


def trials_to_solution_99(probability: float) -> float:
    if probability >= 1.0:
        return 1.0
    if probability <= 0.0:
        return float("inf")
    return float(math.ceil(math.log(0.01) / math.log(1.0 - probability)))


def checkpoints(updates: int) -> list[int]:
    values = [10, 25, 50, 100, 200, 500, 1000, 2000, 4000, 8000, updates]
    return sorted({value for value in values if value <= updates} | {updates})


def summarize_geometry(
    problem: GeometryProblem,
    traces: list[TraceResult],
    profile: PhysicalProposalProfile,
    config: P1Config,
) -> tuple[dict, list[dict], list[dict], dict]:
    diagnostics = post_burn_diagnostics(traces, problem, config)
    exact_db = exact_detailed_balance_validation(problem, profile, config)
    success = np.asarray([trace.first_hit_update is not None for trace in traces], dtype=bool)
    success_probability = float(np.mean(success))
    trials_99 = trials_to_solution_99(success_probability)
    first_hits = np.asarray(
        [trace.first_hit_update for trace in traces if trace.first_hit_update is not None],
        dtype=float,
    )
    proposal_events = np.asarray([trace.proposal_events for trace in traces], dtype=float)
    accepted_events = np.asarray([trace.accepted_events for trace in traces], dtype=float)
    proposal_distances = np.asarray(
        [trace.proposal_distance_sum / max(trace.changed_events, 1) for trace in traces]
    )
    accepted_distances = np.asarray(
        [trace.accepted_distance_sum / max(trace.accepted_events, 1) for trace in traces]
    )
    best_gaps = np.asarray([trace.best_energy[-1] - problem.ground_energy for trace in traces])
    final_gaps = np.asarray([trace.energy[-1] - problem.ground_energy for trace in traces])
    wall_times = np.asarray([trace.wall_time_seconds for trace in traces])
    mean_wall_time = float(np.mean(wall_times))
    summary = {
        "geometry": problem.name,
        "label": GEOMETRY_LABELS[problem.name],
        "runs": len(traces),
        "updates_per_run": config.updates,
        "exact_state_count": len(problem.states),
        "ground_state_count": len(problem.ground_indices),
        "energy_scale_raw_std": problem.energy_scale,
        "success_probability": success_probability,
        "trials_to_solution_99": trials_99,
        "mean_trial_wall_time_seconds": mean_wall_time,
        "time_to_solution_99_seconds": mean_wall_time * trials_99,
        "mean_best_energy_gap": float(np.mean(best_gaps)),
        "median_best_energy_gap": float(np.median(best_gaps)),
        "mean_final_energy_gap": float(np.mean(final_gaps)),
        "median_final_energy_gap": float(np.median(final_gaps)),
        "mean_first_hit_update_successes": float(np.mean(first_hits)) if len(first_hits) else float("nan"),
        "median_first_hit_update_successes": float(np.median(first_hits)) if len(first_hits) else float("nan"),
        "between_trace_best_gap_variance": float(np.var(best_gaps, ddof=1)),
        "between_trace_final_gap_variance": float(np.var(final_gaps, ddof=1)),
        "proposal_event_rate": float(np.sum(proposal_events) / (len(traces) * config.updates)),
        "mh_acceptance_given_event": float(np.sum(accepted_events) / max(np.sum(proposal_events), 1.0)),
        "accepted_changes_per_run": float(np.mean(accepted_events)),
        "mean_proposed_geometry_distance": float(np.mean(proposal_distances)),
        "mean_accepted_geometry_distance": float(np.mean(accepted_distances)),
        **{key: value for key, value in diagnostics.items() if key != "empirical_distribution"},
        **exact_db,
    }

    curve_rows = []
    for checkpoint in checkpoints(config.updates):
        reached = [
            trace.best_energy[checkpoint - 1] <= problem.ground_energy + 1e-12
            for trace in traces
        ]
        curve_rows.append(
            {
                "geometry": problem.name,
                "label": GEOMETRY_LABELS[problem.name],
                "updates": checkpoint,
                "success_probability": float(np.mean(reached)),
            }
        )

    run_rows = []
    burn = int(config.updates * config.burn_fraction)
    for trace in traces:
        run_rows.append(
            {
                "geometry": problem.name,
                "label": GEOMETRY_LABELS[problem.name],
                "run_idx": trace.run_idx,
                "success": int(trace.first_hit_update is not None),
                "first_hit_update": trace.first_hit_update,
                "best_energy_gap": float(trace.best_energy[-1] - problem.ground_energy),
                "final_energy_gap": float(trace.energy[-1] - problem.ground_energy),
                "proposal_events": trace.proposal_events,
                "accepted_events": trace.accepted_events,
                "mh_acceptance_given_event": trace.accepted_events / max(trace.proposal_events, 1),
                "energy_iat_updates": integrated_autocorrelation_time(trace.energy[burn:]),
                "wall_time_seconds": trace.wall_time_seconds,
            }
        )
    return summary, curve_rows, run_rows, diagnostics


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


def save_plots(
    problems: dict[str, GeometryProblem],
    profile: PhysicalProposalProfile,
    summaries: pd.DataFrame,
    curves: pd.DataFrame,
    compatibility: pd.DataFrame,
    diagnostics: dict,
    output_dir: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for name in GEOMETRY_ORDER:
        subset = curves[curves["geometry"] == name]
        ax.plot(
            subset["updates"],
            subset["success_probability"],
            marker="o",
            linewidth=2,
            color=GEOMETRY_COLORS[name],
            label=GEOMETRY_LABELS[name],
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("updates per trace")
    ax.set_ylabel(r"$P_{success}(n)$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "p1_success_curves.png", dpi=180)
    plt.close(fig)

    ordered_summary = summaries.set_index("geometry").loc[GEOMETRY_ORDER]
    ordered_compatibility = compatibility.set_index("geometry").loc[GEOMETRY_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.3))
    panels = [
        (ordered_summary, "stationary_kl_empirical_to_exact", "stationary KL"),
        (ordered_summary, "median_energy_iat_updates", "median energy IAT"),
        (ordered_summary, "mh_acceptance_given_event", "MH acceptance / event"),
        (ordered_compatibility, "proposal_locality_score", "proposal locality score"),
    ]
    for ax, (frame, column, title) in zip(axes.ravel(), panels):
        ax.bar(
            range(3),
            frame[column].to_numpy(dtype=float),
            color=[GEOMETRY_COLORS[name] for name in GEOMETRY_ORDER],
        )
        ax.set_xticks(range(3), ["Gray", "1D", "Isotropic"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "p1_sampling_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))
    matrices = [("measured q rates", profile.q_rates)] + [
        (GEOMETRY_LABELS[name], problems[name].distance) for name in GEOMETRY_ORDER
    ]
    for ax, (title, matrix) in zip(axes, matrices):
        image = ax.imshow(matrix, cmap="viridis", aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("to state")
        ax.set_ylabel("from state")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "p1_transition_geometry.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for ax, name in zip(axes, GEOMETRY_ORDER):
        problem = problems[name]
        empirical = diagnostics[name]["empirical_distribution"]
        top = np.argsort(problem.probabilities)[-18:][::-1]
        x = np.arange(len(top))
        ax.plot(x, problem.probabilities[top], "ko-", linewidth=1.4, label="exact")
        ax.bar(x, empirical[top], color=GEOMETRY_COLORS[name], alpha=0.72, label="empirical")
        ax.set_title(GEOMETRY_LABELS[name])
        ax.set_xlabel("top exact states")
        ax.set_ylabel("probability")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "p1_stationary_distributions.png", dpi=180)
    plt.close(fig)


def run_p1(config: P1Config, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = build_physical_profile(config)
    embeddings = build_embeddings(config.n_states)
    problems = {
        name: build_geometry_problem(name, embeddings[name], config)
        for name in GEOMETRY_ORDER
    }
    compatibility_rows = [
        geometry_compatibility(problems[name], profile) for name in GEOMETRY_ORDER
    ]
    compatibility_frame = pd.DataFrame(compatibility_rows)
    compatibility_frame.to_csv(output_dir / "p1_geometry_compatibility.csv", index=False)

    initial_rng = np.random.default_rng(config.seed)
    initial_states = [
        initial_rng.integers(
            0, config.n_states, size=config.n_variables, dtype=np.int8
        )
        for _ in range(config.runs)
    ]
    traces_by_geometry = {name: [] for name in GEOMETRY_ORDER}
    for run_idx, initial_state in enumerate(initial_states):
        for name in GEOMETRY_ORDER:
            traces_by_geometry[name].append(
                empirical_transition_mh(
                    problems[name], profile, config, initial_state, run_idx
                )
            )
        if (run_idx + 1) % max(config.runs // 8, 1) == 0:
            print(f"completed {run_idx + 1}/{config.runs} common initial states", flush=True)

    summaries = []
    curve_rows = []
    run_rows = []
    diagnostics = {}
    for name in GEOMETRY_ORDER:
        summary, geometry_curve, geometry_runs, geometry_diagnostics = summarize_geometry(
            problems[name], traces_by_geometry[name], profile, config
        )
        summaries.append(summary)
        curve_rows.extend(geometry_curve)
        run_rows.extend(geometry_runs)
        diagnostics[name] = geometry_diagnostics

    summary_frame = pd.DataFrame(summaries)
    curve_frame = pd.DataFrame(curve_rows)
    run_frame = pd.DataFrame(run_rows)
    summary_frame.to_csv(output_dir / "p1_summary.csv", index=False)
    curve_frame.to_csv(output_dir / "p1_success_curves.csv", index=False)
    run_frame.to_csv(output_dir / "p1_runs.csv", index=False)

    metadata = {
        "stage": "P1",
        "objective": "empirical-transition MH and K=6 state-geometry comparison",
        "config": config.__dict__,
        "proposal": {
            "definition": "q_kl = r_k P_kl for k != l",
            "event_probability": "dt * sum_l!=k q_kl",
            "acceptance": "min(1, exp(-beta DeltaE) q_lk/q_kl)",
            "variable_selection": "one of four variables uniformly per update",
            "dt": profile.dt,
            "event_probability_cap": config.event_probability_cap,
            "event_probabilities_by_state": profile.event_probabilities.tolist(),
        },
        "physical_profile": {
            "dataset": config.dataset,
            "selected_k": profile.selected_k,
            "dwell_count": profile.dwell_count,
            "state_means": profile.state_means.tolist(),
            "exit_rates": profile.exit_rates.tolist(),
            "transition_matrix": profile.transition_matrix.tolist(),
            "q_rates": profile.q_rates.tolist(),
            "dwell_preserving_resampling": profile.resampling_metrics,
        },
        "geometry_definitions": {
            "gray_binary": "six 3-bit Gray-code vertices, normalized Euclidean/Hamming geometry",
            "ordered_1d": "six evenly spaced scalar states on [-1,1]",
            "isotropic_6state": "six orthogonal one-hot vectors; every distinct pair is equidistant",
            "energy": "normalized ring-plus-chords pair distance and site-preference distance; each geometry divided by its exact raw-energy standard deviation",
            "comparison_note": "Each geometry induces a different exact target. KL and DB compare each sampler to its own target; direct q-versus-distance compatibility provides the target-independent device interpretation test.",
        },
        "geometry_compatibility": compatibility_rows,
        "geometry_ground_states": {
            name: {
                "count": len(problems[name].ground_indices),
                "states": problems[name].states[problems[name].ground_indices].tolist(),
                "raw_energy_min": float(np.min(problems[name].raw_energies)),
                "normalization_scale": problems[name].energy_scale,
            }
            for name in GEOMETRY_ORDER
        },
        "summaries": summaries,
    }
    with (output_dir / "p1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, ensure_ascii=False, indent=2)

    save_plots(
        problems,
        profile,
        summary_frame,
        curve_frame,
        compatibility_frame,
        diagnostics,
        output_dir,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=64)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.updates <= 0:
        raise ValueError("runs and updates must be positive.")
    config = P1Config(
        runs=args.runs,
        updates=args.updates,
        beta=args.beta,
        seed=args.seed,
    )
    metadata = run_p1(config, args.output_dir)
    print("\nP1 empirical-transition MH")
    print(f"output: {args.output_dir.resolve()}")
    for row in metadata["summaries"]:
        print(
            f"{row['label']}: success={row['success_probability']:.3f}, "
            f"accept={row['mh_acceptance_given_event']:.3f}, "
            f"KL={row['stationary_kl_empirical_to_exact']:.4g}, "
            f"IAT={row['median_energy_iat_updates']:.2f}, "
            f"theory_DB={row['theoretical_db_max_relative_flux_error']:.3g}"
        )


if __name__ == "__main__":
    main()

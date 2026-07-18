"""Four-solver P0 benchmark on an exactly enumerable 3-partition problem.

The four solvers are compared with the same objective, beta, update budget,
initial categorical states, and random seeds:

A. physical_gray_pbit: Gray-code variables, empirical sMTJ uniforms, physical
   state bias, and the current dwell-rate hazard gate.
B. software_gray_glauber: the same Gray encoding and invalid-code penalty, but
   standard iid uniforms and an exact single-bit Glauber/heat-bath update.
C. direct_pdit: direct categorical heat-bath update, without Gray code.
D. physical_mh_pdit: measured q_kl = r_k P_kl proposals corrected by
   Metropolis-Hastings, without Gray code.

The small default instance has 3^6 = 729 categorical states.  Consequently,
the exact Boltzmann distribution, ground energy, stationary KL divergence,
and empirical detailed-balance flux error can all be reported without an
optimization oracle.
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

from mtj_hardware_sa_demo import GOOD_FILE, load_dataset
from mtj_p0_direct_pdit_partition import build_p0_physical_source, stable_softmax
from mtj_physical_kstate_demo import (
    build_dwell_transition_matrix,
    build_random_matrix,
    extract_dwell_segments,
    fit_gmm_1d,
    fit_state_dwell_models,
    fit_sticky_gaussian_hmm,
    gray_code_table,
    reorder_states,
)
from project_paths import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "mtj_p0_four_solver_output"
SOLVER_ORDER = [
    "physical_gray_pbit",
    "software_gray_glauber",
    "direct_pdit",
    "physical_mh_pdit",
]
SOLVER_LABELS = {
    "physical_gray_pbit": "A: physical Gray P-bit",
    "software_gray_glauber": "B: software Gray Glauber",
    "direct_pdit": "C: direct P-dit",
    "physical_mh_pdit": "D: physical MH P-dit",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    values: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    n_partitions: int = 3
    beta: float = 2.0
    invalid_penalty: float = 8.0
    runs: int = 64
    updates: int = 4000
    burn_fraction: float = 0.20
    seed: int = 20260716
    physical_dataset: str = "good"
    gray_data_gain: float = 0.35
    gray_hazard_gain: float = 8.0


@dataclass
class ExactProblem:
    values: np.ndarray
    n_partitions: int
    target: float
    states: np.ndarray
    energies: np.ndarray
    state_powers: np.ndarray
    probabilities: np.ndarray
    ground_energy: float
    ground_indices: np.ndarray


@dataclass
class ThreeStatePhysicalProfile:
    transition_matrix: np.ndarray
    exit_rates: np.ndarray
    q_rates: np.ndarray
    dt_proposal: float
    random_u: np.ndarray
    dwell_count: int
    state_means: np.ndarray
    dwell_models: list[dict]


@dataclass
class TraceResult:
    solver: str
    run_idx: int
    energy: np.ndarray
    best_energy: np.ndarray
    state_index: np.ndarray
    valid: np.ndarray
    first_hit_update: int | None
    wall_time_seconds: float
    physical_samples: int
    physical_samples_to_solution: int | None


class PhysicalUniformStream:
    """Deterministic indexed access to empirical-rank uniform channels."""

    def __init__(self, random_u: np.ndarray, run_idx: int, seed: int):
        matrix = np.asarray(random_u, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        self.matrix = np.clip(matrix, 1e-12, 1.0 - 1e-12)
        self.offset = (run_idx * 997 + seed * 37) % len(matrix)
        self.count = 0

    def draw(self, update: int, lane: int) -> float:
        row = (self.offset + update + 137 * lane + 17 * self.count) % len(self.matrix)
        col = (lane + self.count) % self.matrix.shape[1]
        self.count += 1
        return float(self.matrix[row, col])


def build_exact_problem(config: BenchmarkConfig) -> ExactProblem:
    values = np.asarray(config.values, dtype=float)
    total = float(values.sum())
    if not math.isclose(total % config.n_partitions, 0.0, abs_tol=1e-12):
        raise ValueError("The item sum must be divisible by n_partitions.")
    target = total / config.n_partitions
    states = np.asarray(
        list(itertools.product(range(config.n_partitions), repeat=len(values))),
        dtype=np.int8,
    )
    energies = np.asarray(
        [categorical_energy(state, values, config.n_partitions, target) for state in states],
        dtype=float,
    )
    ground = float(np.min(energies))
    logits = -config.beta * (energies - ground)
    probabilities = stable_softmax(logits)
    powers = config.n_partitions ** np.arange(len(values) - 1, -1, -1, dtype=np.int64)
    return ExactProblem(
        values=values,
        n_partitions=config.n_partitions,
        target=target,
        states=states,
        energies=energies,
        state_powers=powers,
        probabilities=probabilities,
        ground_energy=ground,
        ground_indices=np.flatnonzero(np.isclose(energies, ground, atol=1e-12)),
    )


def categorical_energy(
    state: np.ndarray,
    values: np.ndarray,
    n_partitions: int,
    target: float,
) -> float:
    sums = np.bincount(state.astype(np.int64), weights=values, minlength=n_partitions)
    return float(np.sum(((sums - target) / target) ** 2))


def state_index(state: np.ndarray, problem: ExactProblem) -> int:
    return int(np.dot(state.astype(np.int64), problem.state_powers))


def gray_inverse(value: int) -> int:
    decoded = value
    shifted = value
    while shifted:
        shifted >>= 1
        decoded ^= shifted
    return decoded


def categorical_to_gray_bits(state: np.ndarray) -> np.ndarray:
    bits = np.zeros((len(state), 2), dtype=np.int8)
    for idx, category in enumerate(state):
        gray = int(category) ^ (int(category) >> 1)
        bits[idx, 0] = gray & 1
        bits[idx, 1] = (gray >> 1) & 1
    return bits


def decode_gray_bits(bits: np.ndarray, n_partitions: int) -> tuple[np.ndarray, np.ndarray]:
    gray_values = bits[:, 0].astype(np.int64) + 2 * bits[:, 1].astype(np.int64)
    decoded = np.asarray([gray_inverse(int(value)) for value in gray_values], dtype=np.int8)
    valid = decoded < n_partitions
    return decoded, valid


def gray_energy(
    bits: np.ndarray,
    problem: ExactProblem,
    invalid_penalty: float,
) -> tuple[float, np.ndarray, bool]:
    decoded, valid_mask = decode_gray_bits(bits, problem.n_partitions)
    safe_state = np.minimum(decoded, problem.n_partitions - 1)
    base = categorical_energy(safe_state, problem.values, problem.n_partitions, problem.target)
    penalty = invalid_penalty * int(np.sum(~valid_mask))
    return float(base + penalty), decoded, bool(np.all(valid_mask))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 745.0)))
    exp_value = math.exp(max(value, -745.0))
    return exp_value / (1.0 + exp_value)


def initialize_states(config: BenchmarkConfig) -> list[np.ndarray]:
    rng = np.random.default_rng(config.seed)
    return [
        rng.integers(0, config.n_partitions, size=len(config.values), dtype=np.int8)
        for _ in range(config.runs)
    ]


def record_gray_state(
    bits: np.ndarray,
    problem: ExactProblem,
    invalid_penalty: float,
) -> tuple[float, int, bool]:
    energy, decoded, valid = gray_energy(bits, problem, invalid_penalty)
    index = state_index(decoded, problem) if valid else -1
    return energy, index, valid


def run_physical_gray_pbit(
    run_idx: int,
    initial: np.ndarray,
    problem: ExactProblem,
    config: BenchmarkConfig,
    physical_source,
) -> TraceResult:
    bits = categorical_to_gray_bits(initial)
    energy, _, _ = gray_energy(bits, problem, config.invalid_penalty)
    stream = PhysicalUniformStream(physical_source.random_u, run_idx, config.seed)
    codebook = gray_code_table(physical_source.selected_k)
    n_bits = bits.size
    trace_energy = np.empty(config.updates, dtype=float)
    trace_best = np.empty(config.updates, dtype=float)
    trace_index = np.empty(config.updates, dtype=np.int32)
    trace_valid = np.empty(config.updates, dtype=bool)
    best = energy
    first_hit = None
    samples_at_hit = None
    offset = (run_idx * 31) % len(physical_source.step_states)
    start = time.perf_counter()

    for update in range(config.updates):
        physical_idx = (offset + update) % len(physical_source.step_states)
        physical_state = int(physical_source.step_states[physical_idx])
        flat_idx = min(int(stream.draw(update, 0) * n_bits), n_bits - 1)
        variable, bit_idx = divmod(flat_idx, 2)

        original = int(bits[variable, bit_idx])
        bits[variable, bit_idx] = 0
        energy_zero, _, _ = gray_energy(bits, problem, config.invalid_penalty)
        bits[variable, bit_idx] = 1
        energy_one, _, _ = gray_energy(bits, problem, config.invalid_penalty)
        bits[variable, bit_idx] = original

        physical_bias = float(codebook[physical_state, bit_idx % codebook.shape[1]])
        logit = config.beta * (energy_zero - energy_one) + 2.0 * config.gray_data_gain * physical_bias
        proposed = 1 if stream.draw(update, 1) < sigmoid(logit) else 0
        if proposed != original:
            proposed_energy = energy_one if proposed else energy_zero
            delta = proposed_energy - energy
            rate = float(physical_source.exit_rates[physical_state])
            hazard = config.gray_hazard_gain * rate * physical_source.dt_step
            hazard *= math.exp(-config.beta * max(delta, 0.0))
            acceptance = 1.0 - math.exp(-max(hazard, 0.0))
            if stream.draw(update, 2) < acceptance:
                bits[variable, bit_idx] = proposed
                energy = proposed_energy
        else:
            # Consume the same three physical samples per update for a fair
            # physical-randomness accounting independent of acceptance.
            stream.draw(update, 2)

        energy, idx, valid = record_gray_state(bits, problem, config.invalid_penalty)
        best = min(best, energy)
        if first_hit is None and valid and energy <= problem.ground_energy + 1e-12:
            first_hit = update + 1
            samples_at_hit = stream.count
        trace_energy[update] = energy
        trace_best[update] = best
        trace_index[update] = idx
        trace_valid[update] = valid

    return TraceResult(
        solver="physical_gray_pbit",
        run_idx=run_idx,
        energy=trace_energy,
        best_energy=trace_best,
        state_index=trace_index,
        valid=trace_valid,
        first_hit_update=first_hit,
        wall_time_seconds=time.perf_counter() - start,
        physical_samples=stream.count,
        physical_samples_to_solution=samples_at_hit,
    )


def run_software_gray_glauber(
    run_idx: int,
    initial: np.ndarray,
    problem: ExactProblem,
    config: BenchmarkConfig,
) -> TraceResult:
    rng = np.random.default_rng(config.seed + 100_003 * (run_idx + 1))
    bits = categorical_to_gray_bits(initial)
    energy, _, _ = gray_energy(bits, problem, config.invalid_penalty)
    n_bits = bits.size
    trace_energy = np.empty(config.updates, dtype=float)
    trace_best = np.empty(config.updates, dtype=float)
    trace_index = np.empty(config.updates, dtype=np.int32)
    trace_valid = np.empty(config.updates, dtype=bool)
    best = energy
    first_hit = None
    start = time.perf_counter()

    for update in range(config.updates):
        flat_idx = int(rng.integers(n_bits))
        variable, bit_idx = divmod(flat_idx, 2)
        bits[variable, bit_idx] = 0
        energy_zero, _, _ = gray_energy(bits, problem, config.invalid_penalty)
        bits[variable, bit_idx] = 1
        energy_one, _, _ = gray_energy(bits, problem, config.invalid_penalty)
        p_one = sigmoid(config.beta * (energy_zero - energy_one))
        bits[variable, bit_idx] = 1 if rng.random() < p_one else 0
        energy = energy_one if bits[variable, bit_idx] else energy_zero

        energy, idx, valid = record_gray_state(bits, problem, config.invalid_penalty)
        best = min(best, energy)
        if first_hit is None and valid and energy <= problem.ground_energy + 1e-12:
            first_hit = update + 1
        trace_energy[update] = energy
        trace_best[update] = best
        trace_index[update] = idx
        trace_valid[update] = valid

    return TraceResult(
        solver="software_gray_glauber",
        run_idx=run_idx,
        energy=trace_energy,
        best_energy=trace_best,
        state_index=trace_index,
        valid=trace_valid,
        first_hit_update=first_hit,
        wall_time_seconds=time.perf_counter() - start,
        physical_samples=0,
        physical_samples_to_solution=0 if first_hit is not None else None,
    )


def run_direct_pdit(
    run_idx: int,
    initial: np.ndarray,
    problem: ExactProblem,
    config: BenchmarkConfig,
) -> TraceResult:
    rng = np.random.default_rng(config.seed + 200_003 * (run_idx + 1))
    state = initial.copy()
    energy = categorical_energy(state, problem.values, problem.n_partitions, problem.target)
    trace_energy = np.empty(config.updates, dtype=float)
    trace_best = np.empty(config.updates, dtype=float)
    trace_index = np.empty(config.updates, dtype=np.int32)
    trace_valid = np.ones(config.updates, dtype=bool)
    best = energy
    first_hit = None
    start = time.perf_counter()

    for update in range(config.updates):
        variable = int(rng.integers(len(state)))
        candidate_energies = np.empty(problem.n_partitions, dtype=float)
        for category in range(problem.n_partitions):
            state[variable] = category
            candidate_energies[category] = categorical_energy(
                state, problem.values, problem.n_partitions, problem.target
            )
        probabilities = stable_softmax(-config.beta * candidate_energies)
        state[variable] = int(rng.choice(problem.n_partitions, p=probabilities))
        energy = float(candidate_energies[state[variable]])
        best = min(best, energy)
        if first_hit is None and energy <= problem.ground_energy + 1e-12:
            first_hit = update + 1
        trace_energy[update] = energy
        trace_best[update] = best
        trace_index[update] = state_index(state, problem)

    return TraceResult(
        solver="direct_pdit",
        run_idx=run_idx,
        energy=trace_energy,
        best_energy=trace_best,
        state_index=trace_index,
        valid=trace_valid,
        first_hit_update=first_hit,
        wall_time_seconds=time.perf_counter() - start,
        physical_samples=0,
        physical_samples_to_solution=0 if first_hit is not None else None,
    )


def build_three_state_physical_profile(updates: int) -> ThreeStatePhysicalProfile:
    """Refit the good measured trace at K=3 for direct categorical proposals."""
    dataset = load_dataset(GOOD_FILE, "good_K3_for_physical_MH")
    model = fit_gmm_1d(dataset.ai, 3)
    labels, means, vars_, transition = fit_sticky_gaussian_hmm(
        dataset.ai.copy(),
        model["means"].copy(),
        model["vars"].copy(),
        model["weights"].copy(),
    )
    weights = np.bincount(labels, minlength=3).astype(float)
    weights /= max(float(weights.sum()), 1.0)
    labels, means, vars_, weights, transition = reorder_states(
        labels, means, vars_, weights, transition
    )
    dwells = extract_dwell_segments(dataset.time, labels)
    dwell_models, exit_rates = fit_state_dwell_models(dwells, 3)
    dwell_transition = build_dwell_transition_matrix(dwells, 3)
    q_rates = exit_rates[:, None] * dwell_transition
    np.fill_diagonal(q_rates, 0.0)
    max_rate = float(np.max(q_rates.sum(axis=1)))
    dt_proposal = 0.5 / max(max_rate, 1e-12)
    random_u = build_random_matrix(dataset.ai, max(updates, 512), dims=4)
    return ThreeStatePhysicalProfile(
        transition_matrix=dwell_transition,
        exit_rates=exit_rates,
        q_rates=q_rates,
        dt_proposal=dt_proposal,
        random_u=random_u,
        dwell_count=len(dwells),
        state_means=means,
        dwell_models=dwell_models,
    )


def run_physical_mh_pdit(
    run_idx: int,
    initial: np.ndarray,
    problem: ExactProblem,
    config: BenchmarkConfig,
    profile: ThreeStatePhysicalProfile,
) -> TraceResult:
    state = initial.copy()
    energy = categorical_energy(state, problem.values, problem.n_partitions, problem.target)
    stream = PhysicalUniformStream(profile.random_u, run_idx, config.seed + 811)
    trace_energy = np.empty(config.updates, dtype=float)
    trace_best = np.empty(config.updates, dtype=float)
    trace_index = np.empty(config.updates, dtype=np.int32)
    trace_valid = np.ones(config.updates, dtype=bool)
    best = energy
    first_hit = None
    samples_at_hit = None
    start = time.perf_counter()

    for update in range(config.updates):
        variable = min(int(stream.draw(update, 0) * len(state)), len(state) - 1)
        current_category = int(state[variable])
        row = profile.q_rates[current_category]
        row_rate = float(row.sum())
        event_probability = min(profile.dt_proposal * row_rate, 1.0)
        if row_rate > 0.0 and stream.draw(update, 1) < event_probability:
            proposal_u = stream.draw(update, 2)
            proposal_probabilities = row / row_rate
            proposed_category = int(
                np.searchsorted(np.cumsum(proposal_probabilities), proposal_u, side="right")
            )
            proposed_category = min(proposed_category, problem.n_partitions - 1)
            if proposed_category != current_category:
                state[variable] = proposed_category
                proposed_energy = categorical_energy(
                    state, problem.values, problem.n_partitions, problem.target
                )
                q_forward = float(profile.q_rates[current_category, proposed_category])
                q_reverse = float(profile.q_rates[proposed_category, current_category])
                ratio = math.exp(
                    float(np.clip(-config.beta * (proposed_energy - energy), -745.0, 709.0))
                )
                acceptance = min(1.0, ratio * q_reverse / max(q_forward, 1e-300))
                if stream.draw(update, 3) < acceptance:
                    energy = proposed_energy
                else:
                    state[variable] = current_category

        best = min(best, energy)
        if first_hit is None and energy <= problem.ground_energy + 1e-12:
            first_hit = update + 1
            samples_at_hit = stream.count
        trace_energy[update] = energy
        trace_best[update] = best
        trace_index[update] = state_index(state, problem)

    return TraceResult(
        solver="physical_mh_pdit",
        run_idx=run_idx,
        energy=trace_energy,
        best_energy=trace_best,
        state_index=trace_index,
        valid=trace_valid,
        first_hit_update=first_hit,
        wall_time_seconds=time.perf_counter() - start,
        physical_samples=stream.count,
        physical_samples_to_solution=samples_at_hit,
    )


def integrated_autocorrelation_time(values: np.ndarray, max_lag: int = 500) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return float("nan")
    centered = values - float(np.mean(values))
    variance = float(np.dot(centered, centered) / len(centered))
    if variance <= 1e-15:
        return 1.0
    n_fft = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=n_fft)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[: len(centered)]
    autocov /= np.arange(len(centered), 0, -1)
    autocorr = autocov / autocov[0]
    cutoff = min(max_lag, len(autocorr) - 1)
    positive = []
    for lag in range(1, cutoff + 1):
        if autocorr[lag] <= 0.0:
            break
        positive.append(float(autocorr[lag]))
    return float(1.0 + 2.0 * np.sum(positive))


def stationary_diagnostics(
    traces: list[TraceResult],
    problem: ExactProblem,
    burn_fraction: float,
) -> dict:
    burn = int(len(traces[0].energy) * burn_fraction)
    counts = np.zeros(len(problem.states), dtype=np.int64)
    transition_counts = np.zeros((len(problem.states), len(problem.states)), dtype=np.int64)
    valid_samples = 0
    total_samples = 0
    iat_values = []

    for trace in traces:
        indices = trace.state_index[burn:]
        energies = trace.energy[burn:]
        valid_indices = indices[indices >= 0]
        counts += np.bincount(valid_indices, minlength=len(problem.states))
        valid_samples += len(valid_indices)
        total_samples += len(indices)
        iat_values.append(integrated_autocorrelation_time(energies))
        for left, right in zip(indices[:-1], indices[1:]):
            if left >= 0 and right >= 0:
                transition_counts[int(left), int(right)] += 1

    empirical = counts.astype(float) / max(float(counts.sum()), 1.0)
    positive = empirical > 0.0
    kl = float(
        np.sum(empirical[positive] * np.log(empirical[positive] / problem.probabilities[positive]))
    )
    total_transitions = float(transition_counts.sum())
    flux = transition_counts.astype(float) / max(total_transitions, 1.0)
    detailed_balance_error = float(0.5 * np.sum(np.abs(flux - flux.T)))
    total_variation = float(0.5 * np.sum(np.abs(empirical - problem.probabilities)))
    return {
        "stationary_kl_empirical_to_boltzmann": kl,
        "stationary_total_variation": total_variation,
        "detailed_balance_flux_error": detailed_balance_error,
        "post_burn_valid_fraction": valid_samples / max(total_samples, 1),
        "mean_energy_iat_updates": float(np.nanmean(iat_values)),
        "median_energy_iat_updates": float(np.nanmedian(iat_values)),
        "empirical_distribution": empirical,
        "post_burn_valid_samples": int(valid_samples),
        "post_burn_valid_transitions": int(total_transitions),
    }


def safe_trials_to_solution(success_probability: float, confidence: float = 0.99) -> float:
    if success_probability >= 1.0:
        return 1.0
    if success_probability <= 0.0:
        return float("inf")
    return float(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - success_probability)))


def default_checkpoints(updates: int) -> list[int]:
    base = [10, 25, 50, 100, 200, 500, 1000, 2000, 4000, updates]
    return sorted({value for value in base if value <= updates} | {updates})


def summarize_solver(
    solver: str,
    traces: list[TraceResult],
    problem: ExactProblem,
    config: BenchmarkConfig,
) -> tuple[dict, list[dict], list[dict], dict]:
    success = np.asarray([trace.first_hit_update is not None for trace in traces], dtype=bool)
    final_success_probability = float(np.mean(success))
    trials_99 = safe_trials_to_solution(final_success_probability)
    wall_times = np.asarray([trace.wall_time_seconds for trace in traces], dtype=float)
    best_gaps = np.asarray([trace.best_energy[-1] - problem.ground_energy for trace in traces])
    final_gaps = np.asarray([trace.energy[-1] - problem.ground_energy for trace in traces])
    first_hits = np.asarray(
        [trace.first_hit_update for trace in traces if trace.first_hit_update is not None],
        dtype=float,
    )
    physical_samples = np.asarray([trace.physical_samples for trace in traces], dtype=float)
    physical_to_hit = np.asarray(
        [
            trace.physical_samples_to_solution
            for trace in traces
            if trace.physical_samples_to_solution is not None
        ],
        dtype=float,
    )
    diagnostics = stationary_diagnostics(traces, problem, config.burn_fraction)
    valid_fraction = float(np.mean([np.mean(trace.valid) for trace in traces]))
    mean_time = float(np.mean(wall_times))
    tts_99 = mean_time * trials_99 if math.isfinite(trials_99) else float("inf")

    summary = {
        "solver": solver,
        "label": SOLVER_LABELS[solver],
        "runs": len(traces),
        "updates_per_run": config.updates,
        "success_probability": final_success_probability,
        "trials_to_solution_99": trials_99,
        "mean_trial_wall_time_seconds": mean_time,
        "time_to_solution_99_seconds": tts_99,
        "mean_best_energy_gap": float(np.mean(best_gaps)),
        "median_best_energy_gap": float(np.median(best_gaps)),
        "mean_final_energy_gap": float(np.mean(final_gaps)),
        "median_final_energy_gap": float(np.median(final_gaps)),
        "valid_state_fraction": valid_fraction,
        "mean_first_hit_update_successes": float(np.mean(first_hits)) if len(first_hits) else float("nan"),
        "median_first_hit_update_successes": float(np.median(first_hits)) if len(first_hits) else float("nan"),
        "between_trace_best_gap_variance": float(np.var(best_gaps, ddof=1)) if len(best_gaps) > 1 else 0.0,
        "between_trace_final_gap_variance": float(np.var(final_gaps, ddof=1)) if len(final_gaps) > 1 else 0.0,
        "mean_physical_samples_per_run": float(np.mean(physical_samples)),
        "mean_physical_samples_to_solution_successes": (
            float(np.mean(physical_to_hit)) if len(physical_to_hit) else float("nan")
        ),
        **{key: value for key, value in diagnostics.items() if key != "empirical_distribution"},
    }

    curve_rows = []
    for checkpoint in default_checkpoints(config.updates):
        reached = [trace.best_energy[checkpoint - 1] <= problem.ground_energy + 1e-12 for trace in traces]
        curve_rows.append(
            {
                "solver": solver,
                "label": SOLVER_LABELS[solver],
                "updates": checkpoint,
                "success_probability": float(np.mean(reached)),
            }
        )

    run_rows = []
    for trace in traces:
        burn = int(config.updates * config.burn_fraction)
        run_rows.append(
            {
                "solver": solver,
                "label": SOLVER_LABELS[solver],
                "run_idx": trace.run_idx,
                "success": int(trace.first_hit_update is not None),
                "first_hit_update": trace.first_hit_update,
                "best_energy": float(trace.best_energy[-1]),
                "best_energy_gap": float(trace.best_energy[-1] - problem.ground_energy),
                "final_energy": float(trace.energy[-1]),
                "final_energy_gap": float(trace.energy[-1] - problem.ground_energy),
                "valid_state_fraction": float(np.mean(trace.valid)),
                "energy_iat_updates": integrated_autocorrelation_time(trace.energy[burn:]),
                "wall_time_seconds": trace.wall_time_seconds,
                "physical_samples": trace.physical_samples,
                "physical_samples_to_solution": trace.physical_samples_to_solution,
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def create_plots(
    summary_frame: pd.DataFrame,
    curve_frame: pd.DataFrame,
    diagnostics_by_solver: dict,
    problem: ExactProblem,
    output_dir: Path,
) -> None:
    if plt is None:
        return
    colors = dict(zip(SOLVER_ORDER, ["#4059AD", "#6B9AC4", "#2CA25F", "#D95F02"]))

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for solver in SOLVER_ORDER:
        subset = curve_frame[curve_frame["solver"] == solver]
        ax.plot(
            subset["updates"],
            subset["success_probability"],
            marker="o",
            linewidth=2,
            label=SOLVER_LABELS[solver],
            color=colors[solver],
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("updates per trace")
    ax.set_ylabel(r"$P_{success}(n)$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "four_solver_success.png", dpi=180)
    plt.close(fig)

    frame = summary_frame.set_index("solver").loc[SOLVER_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    metrics = [
        ("success_probability", "success probability"),
        ("median_final_energy_gap", "median final energy gap"),
        ("valid_state_fraction", "valid-state fraction"),
        ("median_energy_iat_updates", "median energy IAT (updates)"),
    ]
    for ax, (column, title) in zip(axes.ravel(), metrics):
        values = frame[column].to_numpy(dtype=float)
        ax.bar(range(4), values, color=[colors[name] for name in SOLVER_ORDER])
        ax.set_xticks(range(4), ["A", "B", "C", "D"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "four_solver_metrics.png", dpi=180)
    plt.close(fig)

    top = np.argsort(problem.probabilities)[-20:][::-1]
    x = np.arange(len(top))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.plot(x, problem.probabilities[top], "ko-", label="exact Boltzmann", linewidth=1.6)
    for offset_idx, solver in enumerate(SOLVER_ORDER):
        empirical = diagnostics_by_solver[solver]["empirical_distribution"]
        ax.bar(
            x + (offset_idx - 1.5) * width,
            empirical[top],
            width=width,
            alpha=0.72,
            color=colors[solver],
            label=SOLVER_LABELS[solver],
        )
    ax.set_xlabel("top-20 exact-Boltzmann states")
    ax.set_ylabel("post-burn probability")
    ax.set_xticks(x, [str(index) for index in top], rotation=60, fontsize=7)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "four_solver_stationary.png", dpi=180)
    plt.close(fig)


def run_benchmark(config: BenchmarkConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = build_exact_problem(config)
    physical_steps = max(config.updates, 512)
    physical_source, physical_base = build_p0_physical_source(
        dataset_name=config.physical_dataset,
        steps=physical_steps,
        max_k=8,
    )
    physical_mh_profile = build_three_state_physical_profile(config.updates)
    initial_states = initialize_states(config)

    traces_by_solver: dict[str, list[TraceResult]] = {name: [] for name in SOLVER_ORDER}
    for run_idx, initial in enumerate(initial_states):
        traces_by_solver["physical_gray_pbit"].append(
            run_physical_gray_pbit(
                run_idx, initial, problem, config, physical_source
            )
        )
        traces_by_solver["software_gray_glauber"].append(
            run_software_gray_glauber(run_idx, initial, problem, config)
        )
        traces_by_solver["direct_pdit"].append(
            run_direct_pdit(run_idx, initial, problem, config)
        )
        traces_by_solver["physical_mh_pdit"].append(
            run_physical_mh_pdit(
                run_idx, initial, problem, config, physical_mh_profile
            )
        )
        if (run_idx + 1) % max(1, config.runs // 8) == 0:
            print(f"completed {run_idx + 1}/{config.runs} common initial states", flush=True)

    summaries = []
    curve_rows = []
    run_rows = []
    diagnostics_by_solver = {}
    for solver in SOLVER_ORDER:
        summary, solver_curve, solver_runs, diagnostics = summarize_solver(
            solver, traces_by_solver[solver], problem, config
        )
        summaries.append(summary)
        curve_rows.extend(solver_curve)
        run_rows.extend(solver_runs)
        diagnostics_by_solver[solver] = diagnostics

    summary_frame = pd.DataFrame(summaries)
    curve_frame = pd.DataFrame(curve_rows)
    run_frame = pd.DataFrame(run_rows)
    summary_frame.to_csv(output_dir / "four_solver_summary.csv", index=False)
    curve_frame.to_csv(output_dir / "four_solver_success_curve.csv", index=False)
    run_frame.to_csv(output_dir / "four_solver_runs.csv", index=False)

    metadata = {
        "experiment": "P0 four-solver exact-enumeration comparison",
        "config": config.__dict__,
        "problem": {
            "values": list(config.values),
            "n_partitions": config.n_partitions,
            "target_partition_sum": problem.target,
            "categorical_state_count": len(problem.states),
            "ground_energy": problem.ground_energy,
            "ground_state_count": len(problem.ground_indices),
            "ground_states": problem.states[problem.ground_indices].tolist(),
            "energy_definition": "sum_d ((partition_sum_d-target)/target)^2",
            "gray_invalid_definition": "inverse-Gray category >= 3; penalized per invalid variable",
        },
        "method_notes": {
            "A": "Gray code + empirical sMTJ-rank uniforms + physical Gray bias + dwell-rate hazard gate; dwell-preserving categorical resampling is used.",
            "B": "Same Gray code and invalid penalty as A; iid software uniforms and exact single-bit heat-bath/Glauber update.",
            "C": "Direct categorical P-dit heat-bath update using softmax(-beta E_i(k)); no invalid codes.",
            "D": "Direct categorical measured q_kl=r_k P_kl event proposal with min(1, exp(-beta DeltaE) q_lk/q_kl) correction.",
            "stationary_KL": "For A/B, empirical samples are conditioned on valid Gray states; invalid occupancy is reported separately.",
            "detailed_balance_error": "0.5*sum_ij |F_ij-F_ji| for empirical post-burn valid transition flux F.",
            "time_to_solution": "99% confidence restart estimate from measured final success and mean Python wall time per trace.",
        },
        "physical_gray_source": {
            "dataset": physical_source.name,
            "selected_k": physical_source.selected_k,
            "exit_rates": physical_source.exit_rates.tolist(),
            "dt_step": physical_source.dt_step,
            "dwell_count": len(physical_source.dwell_segments),
            "resampling": physical_source.resampling_metrics,
            "state_means": physical_base.state_means.tolist(),
        },
        "physical_mh_source": {
            "dataset": "good refit at K=3",
            "state_means": physical_mh_profile.state_means.tolist(),
            "exit_rates": physical_mh_profile.exit_rates.tolist(),
            "transition_matrix": physical_mh_profile.transition_matrix.tolist(),
            "q_rates": physical_mh_profile.q_rates.tolist(),
            "dt_proposal": physical_mh_profile.dt_proposal,
            "max_event_probability": float(
                physical_mh_profile.dt_proposal
                * np.max(physical_mh_profile.q_rates.sum(axis=1))
            ),
            "dwell_count": physical_mh_profile.dwell_count,
            "dwell_models": physical_mh_profile.dwell_models,
        },
        "summaries": summaries,
    }
    with (output_dir / "four_solver_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, ensure_ascii=False, indent=2)

    create_plots(summary_frame, curve_frame, diagnostics_by_solver, problem, output_dir)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=64)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--invalid-penalty", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.updates <= 0:
        raise ValueError("runs and updates must both be positive.")
    config = BenchmarkConfig(
        beta=args.beta,
        invalid_penalty=args.invalid_penalty,
        runs=args.runs,
        updates=args.updates,
        seed=args.seed,
    )
    metadata = run_benchmark(config, args.output_dir)
    print("\nFour-solver P0 benchmark")
    print(f"output: {args.output_dir.resolve()}")
    print(f"ground states: {metadata['problem']['ground_state_count']}")
    for row in metadata["summaries"]:
        print(
            f"{row['label']}: success={row['success_probability']:.3f}, "
            f"median_gap={row['median_final_energy_gap']:.4g}, "
            f"valid={row['valid_state_fraction']:.3f}, "
            f"KL={row['stationary_kl_empirical_to_boltzmann']:.4g}, "
            f"DB={row['detailed_balance_flux_error']:.4g}"
        )


if __name__ == "__main__":
    main()

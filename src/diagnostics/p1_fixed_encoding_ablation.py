"""Compare ordered 1D P-dit and Gray binary p-bit application solvers.

The comparison intentionally keeps the application representation fixed.  Both
methods see the same selected-curve K=6 device trace, QUBO/Ising problem,
initial state, physical random stream, and number of single-spin updates.  The
only changed quantity is the embedding used to interpret the six measured
device states:

* ``ordered_1d_pdit``: six ordered scalar levels;
* ``gray_binary_pbit``: six vertices on a three-bit Gray path.

This isolates the state-geometry choice identified by the P1 device analysis.
It is not a comparison against the direct permutation P-dit TSP encoding, which
changes the optimization state space and is benchmarked separately in
``src/benchmarks/tsp_binary_pbit_vs_ordered_1d_mh.py``.
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
    IsingProblem,
    QuboModel,
    build_maxcut_problem,
    build_portfolio_problem,
    build_tsp_problem,
    decode_maxcut,
    decode_portfolio,
    decode_tsp,
    exact_maxcut,
    exact_portfolio,
    qubo_objective_from_spins,
    reference_tsp,
)
from mtj_p0_direct_pdit_partition import P0PhysicalSource, build_p0_physical_source
from mtj_p1_empirical_transition_mh import (
    P1Config,
    build_embeddings,
    build_geometry_problem,
    build_physical_profile,
    geometry_compatibility,
)
from mtj_physical_kstate_demo import compute_energy_scale
from project_paths import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "diagnostics" / "p1_fixed_encoding_ablation"
METHOD_ORDER = ["gray_binary_pbit", "ordered_1d_pdit"]
METHOD_LABELS = {
    "gray_binary_pbit": "Gray binary p-bit",
    "ordered_1d_pdit": "Ordered 1D P-dit",
}
METHOD_COLORS = {
    "gray_binary_pbit": "#4059AD",
    "ordered_1d_pdit": "#D95F02",
}


@dataclass(frozen=True)
class ApplicationCase:
    name: str
    label: str
    problem: IsingProblem
    model: QuboModel
    reference_objective: float
    reference: dict
    update_budget: int
    tolerance: float


@dataclass
class ApplicationRun:
    method: str
    problem: str
    run_idx: int
    best_spins: np.ndarray
    best_objective: float
    final_objective: float
    first_hit_update: int | None
    accepted_updates: int
    checkpoint_best: dict[int, float]
    wall_time_seconds: float


def safe_logistic(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def normalized_embedding(raw: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Center and RMS-normalize each channel under measured occupancy.

    Centering avoids adding a method-dependent static field to the application
    QUBO.  Unit RMS gives the two embeddings the same data-drive scale while
    preserving their topology and relative distances.
    """
    raw = np.asarray(raw, dtype=float)
    occupancy = np.bincount(states, minlength=len(raw)).astype(float)
    occupancy /= max(float(occupancy.sum()), 1.0)
    mean = occupancy @ raw
    centered = raw - mean[None, :]
    rms = np.sqrt(occupancy @ (centered * centered))
    rms = np.where(rms > 1e-12, rms, 1.0)
    return centered / rms[None, :]


def build_method_embeddings(source: P0PhysicalSource) -> dict[str, np.ndarray]:
    raw = build_embeddings(source.selected_k)
    return {
        "gray_binary_pbit": normalized_embedding(raw["gray_binary"], source.step_states),
        "ordered_1d_pdit": normalized_embedding(raw["ordered_1d"], source.step_states),
    }


def checkpoint_updates(update_budget: int) -> list[int]:
    fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    absolute = [10, 25, 50, 100, 200, 500, 1000, 2000, 4000, 8000]
    fractional = {
        max(1, min(update_budget, int(round(update_budget * value))))
        for value in fractions
    }
    return sorted({value for value in absolute if value <= update_budget} | fractional | {update_budget})


def common_initial_spins(
    source: P0PhysicalSource,
    n_variables: int,
    run_idx: int,
) -> np.ndarray:
    """Create a method-independent initial state from the measured uniform stream."""
    rows, channels = source.random_u.shape
    offset = (run_idx * 53) % rows
    spins = np.empty(n_variables, dtype=np.int8)
    for variable in range(n_variables):
        row = (offset + 11 * variable) % rows
        uniform = float(source.random_u[row, variable % channels])
        spins[variable] = 1 if uniform < 0.5 else -1
    return spins


def run_embedded_solver(
    source: P0PhysicalSource,
    embedding: np.ndarray,
    method: str,
    case: ApplicationCase,
    run_idx: int,
    initial_spins: np.ndarray,
    coupling_gain: float,
    data_gain: float,
    hazard_gain: float,
) -> ApplicationRun:
    """Run the application solver with a fixed single-spin update budget."""
    start = time.perf_counter()
    problem = case.problem
    J = problem.J
    h = problem.h
    n_variables = len(h)
    source_steps = len(source.step_states)
    random_channels = source.random_u.shape[1]
    embedding_dims = embedding.shape[1]
    energy_scale = compute_energy_scale(J, h)
    offset = (run_idx * 53) % source_steps

    spins = initial_spins.copy()
    # Cast explicitly because some NumPy/Accelerate combinations emit spurious
    # integer-matmul overflow warnings for an int8 vector and float64 matrix.
    spin_values = spins.astype(float)
    local_fields = np.sum(J * spin_values[None, :], axis=1) + h
    current = float(
        -0.5 * np.sum(spin_values * (local_fields - h))
        - np.sum(h * spin_values)
        + problem.offset
    )
    best = current
    best_spins = spins.copy()
    first_hit = 0 if best <= case.reference_objective + case.tolerance else None
    accepted = 0
    checkpoints = checkpoint_updates(case.update_budget)
    checkpoint_set = set(checkpoints)
    checkpoint_best: dict[int, float] = {}

    for update in range(1, case.update_budget + 1):
        source_idx = (offset + update - 1) % source_steps
        physical_state = int(source.step_states[source_idx])
        component = (update - 1) % embedding_dims
        spin_idx = (offset + update - 1) % n_variables
        local_field = float(local_fields[spin_idx])
        drive = data_gain * float(embedding[physical_state, component])
        drive += coupling_gain * local_field / max(energy_scale, 1e-12)
        proposal_probability = safe_logistic(2.0 * drive)

        proposal_row = (source_idx + 17) % source_steps
        proposal_channel = (update - 1) % random_channels
        proposal_uniform = float(source.random_u[proposal_row, proposal_channel])
        proposal = 1 if proposal_uniform < proposal_probability else -1

        if proposal != spins[spin_idx]:
            old_spin = int(spins[spin_idx])
            delta = float(2.0 * old_spin * local_field)
            if delta <= 0.0:
                accept = True
            else:
                base_rate = max(float(source.exit_rates[physical_state]), 1e-12)
                thermal = math.exp(-delta / max(energy_scale, 1e-12))
                acceptance = 1.0 - math.exp(
                    -hazard_gain * base_rate * source.dt_step * thermal
                )
                accept_row = (source_idx + 97) % source_steps
                accept_uniform = float(source.random_u[accept_row, -1])
                accept = accept_uniform < acceptance
            if accept:
                spins[spin_idx] = proposal
                local_fields += float(proposal - old_spin) * J[:, spin_idx]
                current += delta
                accepted += 1
                if current < best - 1e-12:
                    best = current
                    best_spins = spins.copy()
                    if first_hit is None and best <= case.reference_objective + case.tolerance:
                        first_hit = update

        if update in checkpoint_set:
            checkpoint_best[update] = float(best)

    # Recompute through QUBO for an independent mapping check.
    checked_best = qubo_objective_from_spins(best_spins, case.model)
    checked_final = qubo_objective_from_spins(spins, case.model)
    if not np.isclose(best, checked_best, atol=1e-6, rtol=1e-10):
        raise AssertionError(f"Incremental energy drift for {case.name}/{method}: {best} != {checked_best}")

    return ApplicationRun(
        method=method,
        problem=case.name,
        run_idx=run_idx,
        best_spins=best_spins,
        best_objective=float(checked_best),
        final_objective=float(checked_final),
        first_hit_update=first_hit,
        accepted_updates=accepted,
        checkpoint_best=checkpoint_best,
        wall_time_seconds=float(time.perf_counter() - start),
    )


def build_cases(budget_scale: float) -> list[ApplicationCase]:
    if budget_scale <= 0.0:
        raise ValueError("budget_scale must be positive.")

    tsp_problem, tsp_model = build_tsp_problem(8)
    tsp_reference = reference_tsp(tsp_problem.metadata)
    maxcut_problem, maxcut_model = build_maxcut_problem(10)
    maxcut_reference = exact_maxcut(maxcut_problem.metadata)
    portfolio_problem, portfolio_model = build_portfolio_problem(8)
    portfolio_reference = exact_portfolio(portfolio_model, portfolio_problem.metadata)

    def scaled(value: int) -> int:
        return max(1, int(round(value * budget_scale)))

    return [
        ApplicationCase(
            name="tsp_8",
            label="8-city TSP",
            problem=tsp_problem,
            model=tsp_model,
            reference_objective=float(tsp_reference["route_length"]),
            reference=tsp_reference,
            update_budget=scaled(81_000),
            tolerance=1e-7,
        ),
        ApplicationCase(
            name="maxcut_10",
            label="10-node Max-Cut",
            problem=maxcut_problem,
            model=maxcut_model,
            reference_objective=-float(maxcut_reference["cut_weight"]),
            reference=maxcut_reference,
            update_budget=scaled(54_000),
            tolerance=1e-9,
        ),
        ApplicationCase(
            name="portfolio_8",
            label="8-asset Portfolio",
            problem=portfolio_problem,
            model=portfolio_model,
            reference_objective=float(portfolio_reference["qubo_objective"]),
            reference=portfolio_reference,
            update_budget=scaled(45_000),
            tolerance=1e-7,
        ),
    ]


def decoded_metrics(case: ApplicationCase, run: ApplicationRun) -> dict:
    if case.name.startswith("tsp"):
        decoded = decode_tsp(run.best_spins, case.problem.metadata)
        reference_length = float(case.reference["route_length"])
        return {
            "decoded_quality": float(decoded["route_length"]),
            "quality_ratio_to_reference": float(decoded["route_length"] / reference_length),
            "postprocessed_quality_ratio": float(decoded["two_opt_route_length"] / reference_length),
            "decoded_feasible": bool(decoded["feasible_raw_assignment"]),
            "constraint_violation": int(decoded["onehot_violation"]),
        }
    if case.name.startswith("maxcut"):
        decoded = decode_maxcut(run.best_spins, case.problem.metadata)
        reference_cut = float(case.reference["cut_weight"])
        return {
            "decoded_quality": float(decoded["cut_weight"]),
            "quality_ratio_to_reference": float(decoded["cut_weight"] / reference_cut),
            "postprocessed_quality_ratio": float("nan"),
            "decoded_feasible": True,
            "constraint_violation": 0,
        }
    decoded = decode_portfolio(run.best_spins, case.problem.metadata)
    return {
        "decoded_quality": float(run.best_objective),
        "quality_ratio_to_reference": float("nan"),
        "postprocessed_quality_ratio": float("nan"),
        "decoded_feasible": bool(decoded["feasible_cardinality"]),
        "constraint_violation": abs(
            int(decoded["selected_count"]) - int(case.problem.metadata["choose_k"])
        ),
    }


def trials_to_solution_99(probability: float) -> float:
    if probability >= 1.0:
        return 1.0
    if probability <= 0.0:
        return float("inf")
    return float(math.ceil(math.log(0.01) / math.log(1.0 - probability)))


def summarize_runs(run_frame: pd.DataFrame, cases: list[ApplicationCase]) -> list[dict]:
    case_map = {case.name: case for case in cases}
    rows = []

    def finite_median(series: pd.Series) -> float:
        values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else float("nan")

    for (problem, method), subset in run_frame.groupby(["problem", "method"], sort=False):
        case = case_map[problem]
        success = subset["hit_reference"].to_numpy(dtype=bool)
        probability = float(np.mean(success))
        trials = trials_to_solution_99(probability)
        first_hits = subset.loc[success, "first_hit_update"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "problem": problem,
                "problem_label": case.label,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "variables": len(case.problem.h),
                "runs": len(subset),
                "updates_per_run": case.update_budget,
                "reference_objective": case.reference_objective,
                "success_probability": probability,
                "trials_to_solution_99": trials,
                "fixed_budget_tts99_updates": case.update_budget * trials,
                "median_first_hit_update_successes": (
                    float(np.median(first_hits)) if len(first_hits) else float("nan")
                ),
                "mean_best_gap": float(subset["best_gap"].mean()),
                "median_best_gap": float(subset["best_gap"].median()),
                "best_gap": float(subset["best_gap"].min()),
                "mean_acceptance_rate": float(subset["acceptance_rate"].mean()),
                "mean_wall_time_seconds": float(subset["wall_time_seconds"].mean()),
                "decoded_feasible_rate": float(subset["decoded_feasible"].mean()),
                "median_constraint_violation": float(subset["constraint_violation"].median()),
                "median_quality_ratio_to_reference": finite_median(
                    subset["quality_ratio_to_reference"]
                ),
                "median_postprocessed_quality_ratio": finite_median(
                    subset["postprocessed_quality_ratio"]
                ),
            }
        )
    return rows


def paired_comparisons(run_frame: pd.DataFrame, seed: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed + 991)
    for problem in run_frame["problem"].unique():
        subset = run_frame[run_frame["problem"] == problem]
        pivot = subset.pivot(index="run_idx", columns="method", values="best_gap")
        delta = (
            pivot["ordered_1d_pdit"].to_numpy(dtype=float)
            - pivot["gray_binary_pbit"].to_numpy(dtype=float)
        )
        success_pivot = subset.pivot(index="run_idx", columns="method", values="hit_reference")
        success_delta = (
            success_pivot["ordered_1d_pdit"].to_numpy(dtype=float)
            - success_pivot["gray_binary_pbit"].to_numpy(dtype=float)
        )
        samples = np.empty(4000, dtype=float)
        success_samples = np.empty(4000, dtype=float)
        for idx in range(len(samples)):
            draw = rng.integers(0, len(delta), size=len(delta))
            samples[idx] = float(np.mean(delta[draw]))
            success_samples[idx] = float(np.mean(success_delta[draw]))
        rows.append(
            {
                "problem": problem,
                "metric": "ordered best gap minus binary best gap (negative favors ordered)",
                "paired_runs": len(delta),
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "bootstrap_mean_delta_ci95_low": float(np.quantile(samples, 0.025)),
                "bootstrap_mean_delta_ci95_high": float(np.quantile(samples, 0.975)),
                "ordered_win_rate": float(np.mean(delta < -1e-9)),
                "tie_rate": float(np.mean(np.abs(delta) <= 1e-9)),
                "binary_win_rate": float(np.mean(delta > 1e-9)),
                "ordered_minus_binary_success_probability": float(np.mean(success_delta)),
                "bootstrap_success_delta_ci95_low": float(np.quantile(success_samples, 0.025)),
                "bootstrap_success_delta_ci95_high": float(np.quantile(success_samples, 0.975)),
            }
        )
    return rows


def add_checkpoint_tts(
    summary_rows: list[dict],
    curve_rows: list[dict],
) -> None:
    curve_frame = pd.DataFrame(curve_rows)
    for row in summary_rows:
        subset = curve_frame[
            (curve_frame["problem"] == row["problem"])
            & (curve_frame["method"] == row["method"])
        ]
        candidates = []
        for curve in subset.to_dict("records"):
            trials = trials_to_solution_99(float(curve["success_probability"]))
            candidates.append((float(curve["updates"]) * trials, int(curve["updates"]), trials))
        best_tts, best_checkpoint, best_trials = min(candidates, key=lambda value: value[0])
        row["best_checkpoint_tts99_updates"] = best_tts
        row["best_tts99_checkpoint_updates"] = best_checkpoint
        row["best_tts99_trials"] = best_trials


def build_success_curves(
    runs: list[ApplicationRun],
    cases: list[ApplicationCase],
) -> list[dict]:
    case_map = {case.name: case for case in cases}
    rows = []
    for problem in case_map:
        case = case_map[problem]
        for method in METHOD_ORDER:
            method_runs = [run for run in runs if run.problem == problem and run.method == method]
            for checkpoint in checkpoint_updates(case.update_budget):
                hits = [
                    run.checkpoint_best[checkpoint]
                    <= case.reference_objective + case.tolerance
                    for run in method_runs
                ]
                rows.append(
                    {
                        "problem": problem,
                        "method": method,
                        "updates": checkpoint,
                        "normalized_budget": checkpoint / case.update_budget,
                        "success_probability": float(np.mean(hits)),
                    }
                )
    return rows


def save_plot(summary_frame: pd.DataFrame, curve_frame: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        return
    problems = list(summary_frame["problem"].drop_duplicates())
    labels = [
        str(summary_frame[summary_frame["problem"] == problem]["problem_label"].iloc[0])
        for problem in problems
    ]
    x = np.arange(len(problems), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    for method_index, method in enumerate(METHOD_ORDER):
        subset = summary_frame[summary_frame["method"] == method].set_index("problem").loc[problems]
        offset = (method_index - 0.5) * width
        axes[0].bar(
            x + offset,
            subset["success_probability"],
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        axes[1].bar(
            x + offset,
            subset["mean_best_gap"],
            width,
            color=METHOD_COLORS[method],
        )
        axes[2].bar(
            x + offset,
            subset["mean_wall_time_seconds"],
            width,
            color=METHOD_COLORS[method],
        )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Reference hit probability")
    axes[0].set_ylabel("Probability")
    axes[1].set_title("Best objective gap")
    axes[1].set_ylabel("Mean gap (lower is better)")
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[2].set_title("Software runtime")
    axes[2].set_ylabel("Mean seconds / run")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "application_method_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(problems), figsize=(14.2, 4.2), sharey=True)
    if len(problems) == 1:
        axes = [axes]
    for axis, problem, label in zip(axes, problems, labels):
        for method in METHOD_ORDER:
            subset = curve_frame[
                (curve_frame["problem"] == problem) & (curve_frame["method"] == method)
            ]
            axis.plot(
                subset["normalized_budget"],
                subset["success_probability"],
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(label)
        axis.set_xlabel("Fraction of update budget")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Reference hit probability")
    axes[0].set_ylim(-0.03, 1.03)
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "application_success_curves.png", dpi=180)
    plt.close(fig)


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


def geometry_context(profile_steps: int) -> list[dict]:
    config = P1Config(runs=1, updates=profile_steps)
    profile = build_physical_profile(config)
    embeddings = build_embeddings(config.n_states)
    rows = []
    for name in ["gray_binary", "ordered_1d"]:
        problem = build_geometry_problem(name, embeddings[name], config)
        rows.append(geometry_compatibility(problem, profile))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--profile-steps", type=int, default=900)
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Scale the published demo budgets; use a small value for smoke tests.",
    )
    parser.add_argument("--coupling-gain", type=float, default=3.0)
    parser.add_argument("--data-gain", type=float, default=0.8)
    parser.add_argument("--hazard-gain", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0 or args.profile_steps <= 0:
        raise ValueError("runs and profile_steps must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source, base_profile = build_p0_physical_source(
        dataset_name="selected_curve",
        steps=args.profile_steps,
        max_k=8,
    )
    if source.selected_k != 6:
        raise ValueError(f"The ordered 1D comparison requires K=6, got K={source.selected_k}.")
    embeddings = build_method_embeddings(source)
    cases = build_cases(args.budget_scale)

    all_runs: list[ApplicationRun] = []
    run_rows = []
    for case in cases:
        for run_idx in range(args.runs):
            initial = common_initial_spins(source, len(case.problem.h), run_idx)
            for method in METHOD_ORDER:
                run = run_embedded_solver(
                    source=source,
                    embedding=embeddings[method],
                    method=method,
                    case=case,
                    run_idx=run_idx,
                    initial_spins=initial,
                    coupling_gain=args.coupling_gain,
                    data_gain=args.data_gain,
                    hazard_gain=args.hazard_gain,
                )
                all_runs.append(run)
                decoded = decoded_metrics(case, run)
                run_rows.append(
                    {
                        "problem": case.name,
                        "problem_label": case.label,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "run_idx": run_idx,
                        "updates": case.update_budget,
                        "best_objective": run.best_objective,
                        "final_objective": run.final_objective,
                        "reference_objective": case.reference_objective,
                        "best_gap": run.best_objective - case.reference_objective,
                        "hit_reference": run.first_hit_update is not None,
                        "first_hit_update": run.first_hit_update,
                        "accepted_updates": run.accepted_updates,
                        "acceptance_rate": run.accepted_updates / case.update_budget,
                        "wall_time_seconds": run.wall_time_seconds,
                        **decoded,
                    }
                )
        print(f"completed {case.label}: {args.runs} paired runs", flush=True)

    run_frame = pd.DataFrame(run_rows)
    curve_rows = build_success_curves(all_runs, cases)
    summary_rows = summarize_runs(run_frame, cases)
    add_checkpoint_tts(summary_rows, curve_rows)
    paired_rows = paired_comparisons(run_frame, args.seed)
    summary_frame = pd.DataFrame(summary_rows)
    paired_frame = pd.DataFrame(paired_rows)
    curve_frame = pd.DataFrame(curve_rows)

    run_frame.to_csv(output_dir / "application_comparison_runs.csv", index=False)
    summary_frame.to_csv(output_dir / "application_comparison_summary.csv", index=False)
    paired_frame.to_csv(output_dir / "application_paired_comparison.csv", index=False)
    curve_frame.to_csv(output_dir / "application_success_curves.csv", index=False)

    metadata = {
        "stage": "P1 application comparison",
        "objective": "ordered 1D P-dit versus Gray-coded binary p-bit on practical QUBO/Ising demos",
        "comparison_design": {
            "physical_dataset": "selected_curve",
            "selected_k": source.selected_k,
            "profile_steps": args.profile_steps,
            "runs_per_method_problem": args.runs,
            "common_initial_states": True,
            "common_physical_random_stream": True,
            "equal_single_spin_update_budget": True,
            "fixed_application_encoding": True,
            "embedding_normalization": "measured-occupancy centering and per-channel unit RMS",
            "solver": "single-spin stochastic proposal plus dwell-rate hazard acceptance",
            "coupling_gain": args.coupling_gain,
            "data_gain": args.data_gain,
            "hazard_gain": args.hazard_gain,
            "budget_scale": args.budget_scale,
            "software_runtime_warning": (
                "Wall time measures this Python implementation, not projected MTJ hardware latency."
            ),
        },
        "cases": [
            {
                "name": case.name,
                "label": case.label,
                "variables": len(case.problem.h),
                "updates_per_run": case.update_budget,
                "reference_objective": case.reference_objective,
                "reference": case.reference,
            }
            for case in cases
        ],
        "physical_profile": {
            "state_means": base_profile.state_means,
            "exit_rates": source.exit_rates,
            "transition_matrix": base_profile.transition_matrix,
            "dwell_preserving_resampling": source.resampling_metrics,
        },
        "device_geometry_context": geometry_context(args.profile_steps),
        "method_summary": summary_rows,
        "paired_comparison": paired_rows,
    }
    with (output_dir / "application_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, ensure_ascii=False, indent=2)

    save_plot(summary_frame, curve_frame, output_dir)
    print("P1 application comparison finished.")
    print(summary_frame.to_string(index=False))
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

"""P0: direct categorical P-dit solver driven by an sMTJ K-state trace.

This is a new, standalone implementation.  It intentionally does not modify
the existing Gray-code/Ising demos.

The categorical update follows the isotropic P-dit formulation in:
    Duffee et al., Phys. Rev. Applied 24, 044077 (2025), Eqs. (19)-(22).

Two P0 changes are implemented together:
1. A D-partition variable is represented directly by one categorical P-dit;
   no Gray code and no one-hot penalty are used.
2. The measured discrete K-state trace is resampled by allocating at least one
   sample to every dwell segment.  This preserves dwell order and boundaries
   and cannot create interpolated intermediate states.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from mtj_physical_kstate_demo import (
    DwellSegment,
    KStateProfile,
    build_kstate_profile,
    resample_series,
)
from project_paths import RESULTS_DIR


OUTPUT_DIR = RESULTS_DIR / "mtj_p0_direct_pdit_output"
PAPER_DOI = "10.1103/4ngx-cmz7"


@dataclass
class P0PhysicalSource:
    name: str
    selected_k: int
    step_states: np.ndarray
    exit_rates: np.ndarray
    random_u: np.ndarray
    dt_step: float
    dwell_segments: list[DwellSegment]
    resampling_metrics: dict


@dataclass
class PartitionInstance:
    values: np.ndarray
    n_partitions: int
    target_sum: int
    metadata: dict


def compressed_states(states: np.ndarray) -> list[int]:
    if len(states) == 0:
        return []
    keep = np.r_[True, states[1:] != states[:-1]]
    return states[keep].astype(int).tolist()


def allocate_dwell_samples(dwells: list[DwellSegment], steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample a categorical trace while preserving every dwell boundary.

    Each dwell gets one guaranteed output sample.  Integer counts are then
    adjusted toward their duration-proportional quotas while respecting that
    lower bound.
    Consequently, the compressed output state sequence is exactly the input
    dwell-state sequence.  ``steps`` must be at least the number of dwells.
    """
    if not dwells:
        raise ValueError("At least one dwell segment is required.")
    if steps < len(dwells):
        raise ValueError(
            f"steps={steps} is smaller than dwell_count={len(dwells)}; "
            "all dwell boundaries cannot be preserved."
        )

    durations = np.array([d.duration for d in dwells], dtype=float)
    durations = np.maximum(durations, 0.0)
    if not np.any(durations > 0.0):
        durations = np.ones_like(durations)

    quotas = steps * durations / durations.sum()
    counts = np.maximum(1, np.floor(quotas).astype(int))
    while int(counts.sum()) < steps:
        counts[int(np.argmax(quotas - counts))] += 1
    while int(counts.sum()) > steps:
        eligible = np.flatnonzero(counts > 1)
        remove_idx = int(eligible[np.argmax(counts[eligible] - quotas[eligible])])
        counts[remove_idx] -= 1
    states = np.repeat(np.array([d.state for d in dwells], dtype=np.int32), counts)
    if len(states) != steps:
        raise AssertionError("Dwell apportionment did not produce the requested length.")
    return states, counts


def resampling_validation(base: KStateProfile, new_states: np.ndarray, counts: np.ndarray) -> dict:
    dwell_states = np.array([d.state for d in base.dwell_segments], dtype=np.int32)
    legacy_states = np.rint(
        resample_series(base.state_labels.astype(float), len(new_states))
    ).astype(np.int32)

    expected_path = dwell_states.astype(int).tolist()
    new_path = compressed_states(new_states)
    legacy_path = compressed_states(legacy_states)
    expected_pairs = list(zip(expected_path[:-1], expected_path[1:]))
    expected_pair_set = set(expected_pairs)
    legacy_pairs = list(zip(legacy_path[:-1], legacy_path[1:]))
    new_pairs = list(zip(new_path[:-1], new_path[1:]))

    k = base.selected_k
    durations = np.array([d.duration for d in base.dwell_segments], dtype=float)
    original_occ = np.zeros(k, dtype=float)
    for dwell, duration in zip(base.dwell_segments, durations):
        original_occ[dwell.state] += duration
    original_occ /= max(original_occ.sum(), 1e-12)
    new_occ = np.bincount(new_states, minlength=k).astype(float) / len(new_states)
    legacy_occ = np.bincount(legacy_states, minlength=k).astype(float) / len(legacy_states)

    return {
        "method": "duration-proportional integer apportionment with one-sample dwell lower bound",
        "target_steps": int(len(new_states)),
        "dwell_count": int(len(base.dwell_segments)),
        "all_dwell_boundaries_preserved": bool(new_path == expected_path),
        "new_transition_count": int(len(new_pairs)),
        "expected_dwell_transition_count": int(len(expected_pairs)),
        "new_phantom_transition_count": int(sum(pair not in expected_pair_set for pair in new_pairs)),
        "legacy_transition_count": int(len(legacy_pairs)),
        "legacy_phantom_transition_count": int(sum(pair not in expected_pair_set for pair in legacy_pairs)),
        "legacy_compressed_path_matches_dwells": bool(legacy_path == expected_path),
        "new_occupancy_l1_error": float(np.sum(np.abs(new_occ - original_occ))),
        "legacy_occupancy_l1_error": float(np.sum(np.abs(legacy_occ - original_occ))),
        "original_time_occupancy": original_occ.tolist(),
        "new_sample_occupancy": new_occ.tolist(),
        "legacy_sample_occupancy": legacy_occ.tolist(),
        "min_samples_per_dwell": int(np.min(counts)),
        "max_samples_per_dwell": int(np.max(counts)),
    }


def build_p0_physical_source(
    dataset_name: str = "selected_curve",
    steps: int = 512,
    max_k: int = 8,
) -> tuple[P0PhysicalSource, KStateProfile]:
    """Build the old profile, then replace only its discrete resampling route."""
    base = build_kstate_profile(dataset_name, max_k=max_k, steps=steps)
    states, counts = allocate_dwell_samples(base.dwell_segments, steps=steps)
    metrics = resampling_validation(base, states, counts)
    total_duration = float(sum(d.duration for d in base.dwell_segments))
    source = P0PhysicalSource(
        name=dataset_name,
        selected_k=base.selected_k,
        step_states=states,
        exit_rates=base.exit_rates.copy(),
        random_u=base.random_u.copy(),
        dt_step=total_duration / steps,
        dwell_segments=base.dwell_segments,
        resampling_metrics=metrics,
    )
    return source, base


def make_partition_instance(
    n_partitions: int,
    n_items: int | None,
    max_value: int,
    seed: int,
) -> PartitionInstance:
    if n_partitions not in (3, 6):
        raise ValueError("P0 supports the paper's 3-partition or 6-partition cases.")

    if n_partitions == 3 and (n_items is None or n_items == 14) and max_value == 6:
        # A deterministic 14-number, 1..6 instance with an exact equal partition.
        values = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 3, 3], dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(values)
        source = "deterministic 14-value P-dit3 validation instance"
    else:
        if n_items is None:
            n_items = 1000 if n_partitions == 6 else 14
        rng = np.random.default_rng(seed)
        values = rng.integers(1, max_value + 1, size=n_items, dtype=np.int64)
        # Match the equal-partition target by making the total divisible by D.
        delta = int((-int(values.sum())) % n_partitions)
        if delta:
            candidates = np.flatnonzero(values + delta <= max_value)
            if len(candidates):
                values[int(candidates[-1])] += delta
            else:
                values[-1] -= n_partitions - delta
        source = "seeded random values with total-sum divisibility adjustment"

    total = int(values.sum())
    if total % n_partitions != 0:
        raise AssertionError("Partition instance total must be divisible by D.")
    return PartitionInstance(
        values=values,
        n_partitions=n_partitions,
        target_sum=total // n_partitions,
        metadata={
            "source": source,
            "seed": seed,
            "n_items": int(len(values)),
            "n_partitions": n_partitions,
            "min_value": int(values.min()),
            "max_value": int(values.max()),
            "total_sum": total,
            "target_partition_sum": total // n_partitions,
        },
    )


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / weights.sum()


def categorical_sample(probabilities: np.ndarray, uniform: float) -> int:
    cumulative = np.cumsum(probabilities)
    return int(min(np.searchsorted(cumulative, uniform, side="right"), len(probabilities) - 1))


def partition_sums(values: np.ndarray, states: np.ndarray, n_partitions: int) -> np.ndarray:
    return np.bincount(states, weights=values, minlength=n_partitions).astype(float)


def partition_error(sums: np.ndarray, target: float) -> float:
    return float(np.sum(np.abs(sums - target)))


def partition_energy(sums: np.ndarray) -> float:
    """D-partition objective up to a positive factor and additive constant."""
    return float(np.sum(sums * sums))


def isotropic_pdit_influence(
    item_value: float,
    sums_without_item: np.ndarray,
    total_without_item: float,
) -> np.ndarray:
    """Paper Eqs. (19),(21),(31) with J'_ij = -2*n_i*n_j.

    For candidate dimension d:
      I_i^d = sum_j [J'_ij if z_j=d else -J'_ij]
            = 2*n_i*(total_without_i - 2*S_without_i[d]).
    """
    return 2.0 * item_value * (total_without_item - 2.0 * sums_without_item)


def validate_paper_update_equivalence() -> float:
    """Verify softmax(beta*I) equals heat-bath sampling from paper energy."""
    values = np.array([1.0, 2.0, 4.0, 5.0])
    states = np.array([0, 1, 2, 0], dtype=np.int32)
    item_idx = 1
    beta = 0.03125
    sums = partition_sums(values, states, 3)
    sums_without = sums.copy()
    sums_without[states[item_idx]] -= values[item_idx]
    influence = isotropic_pdit_influence(values[item_idx], sums_without, values.sum() - values[item_idx])
    p_influence = stable_softmax(beta * influence)

    candidate_energy = []
    for candidate in range(3):
        candidate_sums = sums_without.copy()
        candidate_sums[candidate] += values[item_idx]
        # The paper's isotropic coupling energy is 2*sum(S_d^2) plus a constant.
        candidate_energy.append(2.0 * partition_energy(candidate_sums))
    p_energy = stable_softmax(-beta * np.array(candidate_energy))
    return float(np.max(np.abs(p_influence - p_energy)))


def initialize_categorical_states(
    instance: PartitionInstance,
    source: P0PhysicalSource,
    offset: int,
) -> np.ndarray:
    n = len(instance.values)
    d = instance.n_partitions
    states = np.empty(n, dtype=np.int32)
    for item_idx in range(n):
        source_idx = (offset + 7 * item_idx) % len(source.step_states)
        physical_state = int(source.step_states[source_idx])
        uniform = float(source.random_u[source_idx, item_idx % source.random_u.shape[1]])
        states[item_idx] = (physical_state + min(int(uniform * d), d - 1)) % d
    return states


def run_direct_pdit_partition(
    instance: PartitionInstance,
    source: P0PhysicalSource,
    run_idx: int,
    iterations: int,
    checkpoints: list[int],
    beta_start: float,
    beta_end: float,
    hazard_gain: float,
) -> dict:
    """Run asynchronous direct categorical heat-bath updates.

    The physical hazard gate only skips clock ticks.  It is independent of the
    categorical candidate and therefore does not alter the P-dit conditional
    probabilities on ticks where an update is performed.
    """
    d = instance.n_partitions
    n = len(instance.values)
    source_steps = len(source.step_states)
    offset = (run_idx * 53) % source_steps
    states = initialize_categorical_states(instance, source, offset)
    sums = partition_sums(instance.values, states, d)
    total = float(instance.values.sum())
    target = float(instance.target_sum)

    best_error = partition_error(sums, target)
    best_energy = partition_energy(sums)
    first_hit = 0 if best_error == 0.0 else None
    performed_updates = 0
    changed_updates = 0
    checkpoint_set = set(checkpoints)
    checkpoint_rows: list[dict] = []
    best_error_history = np.empty(iterations, dtype=float)

    for iteration in range(1, iterations + 1):
        source_idx = (offset + iteration - 1) % source_steps
        physical_state = int(source.step_states[source_idx])
        row = source.random_u[source_idx]

        if hazard_gain > 0.0:
            rate = max(float(source.exit_rates[physical_state]), 1e-12)
            clock_probability = 1.0 - math.exp(-hazard_gain * rate * source.dt_step)
            if float(row[-1]) >= clock_probability:
                best_error_history[iteration - 1] = best_error
                if iteration in checkpoint_set:
                    checkpoint_rows.append(
                        {
                            "iterations": iteration,
                            "best_error": best_error,
                            "hit_ground": bool(best_error == 0.0),
                            "performed_updates": performed_updates,
                        }
                    )
                continue

        item_idx = min(int(float(row[0]) * n), n - 1)
        old_state = int(states[item_idx])
        value = float(instance.values[item_idx])
        sums_without = sums.copy()
        sums_without[old_state] -= value

        progress = 0.0 if iterations <= 1 else (iteration - 1) / (iterations - 1)
        beta = beta_start + (beta_end - beta_start) * progress
        influence = isotropic_pdit_influence(value, sums_without, total - value)
        probabilities = stable_softmax(beta * influence)
        sample_u = float(row[1 % len(row)])
        new_state = categorical_sample(probabilities, sample_u)
        performed_updates += 1

        if new_state != old_state:
            states[item_idx] = new_state
            sums = sums_without
            sums[new_state] += value
            changed_updates += 1

        error = partition_error(sums, target)
        energy = partition_energy(sums)
        if error < best_error or (error == best_error and energy < best_energy):
            best_error = error
            best_energy = energy
        if first_hit is None and error == 0.0:
            first_hit = iteration
        best_error_history[iteration - 1] = best_error

        if iteration in checkpoint_set:
            checkpoint_rows.append(
                {
                    "iterations": iteration,
                    "best_error": best_error,
                    "hit_ground": bool(best_error == 0.0),
                    "performed_updates": performed_updates,
                }
            )

    return {
        "run_idx": run_idx,
        "best_error": float(best_error),
        "final_error": partition_error(sums, target),
        "best_energy": float(best_energy),
        "final_energy": partition_energy(sums),
        "hit_ground": bool(best_error == 0.0),
        "first_hit_iteration": first_hit,
        "performed_updates": int(performed_updates),
        "changed_updates": int(changed_updates),
        "change_rate": float(changed_updates / max(performed_updates, 1)),
        "final_partition_sums": sums.tolist(),
        "all_variables_valid": bool(np.all((states >= 0) & (states < d))),
        "checkpoint_rows": checkpoint_rows,
        "best_error_history": best_error_history,
    }


def encoding_comparison(n_items: int, n_partitions: int) -> dict:
    gray_dims = int(math.ceil(math.log2(n_partitions)))
    log_direct_states = n_items * math.log(n_partitions)
    log_gray_states = n_items * gray_dims * math.log(2.0)
    log_onehot_states = n_items * n_partitions * math.log(2.0)
    return {
        "direct_pdit_variables": n_items,
        "direct_categories_per_variable": n_partitions,
        "direct_valid_fraction": 1.0,
        "gray_binary_variables": n_items * gray_dims,
        "gray_valid_fraction": float(math.exp(log_direct_states - log_gray_states)),
        "onehot_binary_variables": n_items * n_partitions,
        "onehot_valid_fraction": float(math.exp(log_direct_states - log_onehot_states)),
        "uses_gray_code": False,
        "uses_onehot_penalty": False,
    }


def summarize_checkpoints(checkpoint_rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(checkpoint_rows)
    summary = []
    for iterations, group in frame.groupby("iterations", sort=True):
        summary.append(
            {
                "iterations": int(iterations),
                "runs": int(len(group)),
                "ground_hit_rate": float(group["hit_ground"].mean()),
                "mean_best_error": float(group["best_error"].mean()),
                "median_best_error": float(group["best_error"].median()),
                "mean_performed_updates": float(group["performed_updates"].mean()),
            }
        )
    return summary


def save_plots(
    performance: list[dict],
    base: KStateProfile,
    source: P0PhysicalSource,
    output_dir: Path,
) -> None:
    if plt is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(performance)
    fig, left = plt.subplots(figsize=(8.2, 4.8))
    right = left.twinx()
    left.plot(
        frame["iterations"],
        frame["ground_hit_rate"],
        color="#2a6f97",
        marker="o",
        linewidth=1.8,
        label="Ground hit rate",
    )
    right.plot(
        frame["iterations"],
        frame["median_best_error"],
        color="#d97732",
        marker="s",
        linewidth=1.6,
        label="Median best error",
    )
    left.set_xscale("log", base=2)
    left.set_ylim(-0.03, 1.03)
    left.set_xlabel("Physical clock iterations")
    left.set_ylabel("Ground hit rate", color="#2a6f97")
    right.set_ylabel("Median best partition error", color="#d97732")
    left.grid(True, alpha=0.25)
    handles = left.get_lines() + right.get_lines()
    left.legend(handles, [line.get_label() for line in handles], loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "pdit_partition_performance.png", dpi=180)
    plt.close(fig)

    legacy = np.rint(
        resample_series(base.state_labels.astype(float), len(source.step_states))
    ).astype(np.int32)
    show = min(180, len(source.step_states))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].step(np.arange(show), legacy[:show], where="post", color="#9c4f3d", linewidth=1.2)
    axes[0].set_title("Legacy linear interpolation + rounding")
    axes[1].step(
        np.arange(show),
        source.step_states[:show],
        where="post",
        color="#2a6f97",
        linewidth=1.2,
    )
    axes[1].set_title("P0 dwell-preserving categorical resampling")
    for axis in axes:
        axis.set_ylabel("K-state")
        axis.set_yticks(np.arange(source.selected_k))
        axis.grid(True, alpha=0.2)
    axes[1].set_xlabel("Resampled step")
    fig.tight_layout()
    fig.savefig(output_dir / "dwell_resampling_comparison.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=int, choices=[3, 6], default=3)
    parser.add_argument("--items", type=int, default=None)
    parser.add_argument("--max-value", type=int, default=None)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=2048)
    parser.add_argument("--profile-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20250715)
    parser.add_argument("--hazard-gain", type=float, default=1.0)
    parser.add_argument("--dataset", default="selected_curve")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.items is None:
        args.items = 14 if args.partitions == 3 else 1000
    if args.max_value is None:
        args.max_value = 6 if args.partitions == 3 else 100
    if args.iterations < 64:
        raise ValueError("iterations must be at least 64 for checkpoint reporting.")

    source, base = build_p0_physical_source(
        dataset_name=args.dataset,
        steps=args.profile_steps,
    )
    instance = make_partition_instance(
        n_partitions=args.partitions,
        n_items=args.items,
        max_value=args.max_value,
        seed=args.seed,
    )

    if args.partitions == 3:
        beta_start = beta_end = 1.0 / 32.0
    else:
        beta_start, beta_end = 1.0 / 32.0, 1.0

    powers = [2**power for power in range(6, int(math.log2(args.iterations)) + 1)]
    checkpoints = sorted(set([*powers, args.iterations]))
    runs = [
        run_direct_pdit_partition(
            instance=instance,
            source=source,
            run_idx=run_idx,
            iterations=args.iterations,
            checkpoints=checkpoints,
            beta_start=beta_start,
            beta_end=beta_end,
            hazard_gain=args.hazard_gain,
        )
        for run_idx in range(args.runs)
    ]

    checkpoint_rows = []
    run_rows = []
    for result in runs:
        for row in result["checkpoint_rows"]:
            checkpoint_rows.append({"run_idx": result["run_idx"], **row})
        run_rows.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"checkpoint_rows", "best_error_history"}
            }
        )
    performance = summarize_checkpoints(checkpoint_rows)
    equation_error = validate_paper_update_equivalence()
    if equation_error > 1e-12:
        raise AssertionError(f"P-dit influence/energy update mismatch: {equation_error}")
    if not source.resampling_metrics["all_dwell_boundaries_preserved"]:
        raise AssertionError("P0 resampling failed to preserve the dwell path.")
    if not all(row["all_variables_valid"] for row in run_rows):
        raise AssertionError("A direct P-dit variable left its categorical state space.")

    summary = {
        "stage": "P0",
        "paper": {
            "title": "P-dits: Probabilistic d-dimensional bits for extended-variable probabilistic computing",
            "doi": PAPER_DOI,
            "implemented_equations": [19, 20, 21, 22, 31],
            "update_probability": "P(z_i=d) = softmax_d(beta * I_i^d)",
            "influence": "I_i^d = 2*n_i*(sum_without_i - 2*partition_sum_without_i[d])",
        },
        "p0_changes": {
            "direct_categorical_pdit": True,
            "gray_code_removed": True,
            "onehot_penalty_removed": True,
            "dwell_preserving_resampling": True,
            "existing_scripts_modified": False,
        },
        "instance": instance.metadata,
        "encoding_comparison": encoding_comparison(len(instance.values), instance.n_partitions),
        "physical_source": {
            "dataset": source.name,
            "selected_k": source.selected_k,
            "profile_steps": int(len(source.step_states)),
            "dt_step": source.dt_step,
            "hazard_gain": args.hazard_gain,
            "role": "empirical entropy, variable selection, and optional physical-clock thinning",
        },
        "resampling_validation": source.resampling_metrics,
        "algorithm_validation": {
            "max_softmax_equivalence_error": equation_error,
            "all_runs_categorical_valid": True,
        },
        "configuration": {
            "runs": args.runs,
            "iterations": args.iterations,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "checkpoints": checkpoints,
            "seed": args.seed,
        },
        "performance": performance,
        "final": {
            "ground_hit_rate": float(np.mean([row["hit_ground"] for row in run_rows])),
            "mean_best_error": float(np.mean([row["best_error"] for row in run_rows])),
            "median_best_error": float(np.median([row["best_error"] for row in run_rows])),
            "mean_performed_updates": float(np.mean([row["performed_updates"] for row in run_rows])),
            "mean_change_rate": float(np.mean([row["change_rate"] for row in run_rows])),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(output_dir / "pdit_partition_runs.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(output_dir / "pdit_partition_checkpoints.csv", index=False)
    pd.DataFrame(performance).to_csv(output_dir / "pdit_partition_performance.csv", index=False)
    pd.DataFrame(
        {
            "step": np.arange(len(source.step_states)),
            "p0_dwell_state": source.step_states,
            "legacy_linear_rounded_state": np.rint(
                resample_series(base.state_labels.astype(float), len(source.step_states))
            ).astype(np.int32),
        }
    ).to_csv(output_dir / "resampling_comparison.csv", index=False)
    with open(output_dir / "p0_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_plots(performance, base, source, output_dir)

    print("P0 direct categorical P-dit analysis finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

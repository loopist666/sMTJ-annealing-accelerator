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

from mtj_hardware_sa_demo import (
    BAD_FILE,
    GOOD_FILE,
    build_ising_instance,
    ising_energy,
    load_dataset,
    summarize_runs,
)


OUTPUT_DIR = Path("mtj_multistate_pbit_output")
N_STATES = 4
STEPS = 420


@dataclass
class MultiStateProfile:
    name: str
    state_labels: np.ndarray
    state_centers: np.ndarray
    transition_matrix: np.ndarray
    occupancy: np.ndarray
    pbit_biases: np.ndarray
    random_u: np.ndarray
    temperature: np.ndarray
    metrics: dict


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_mean(x, window)
    mean_sq = rolling_mean(x * x, window)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def resample_series(x: np.ndarray, target_len: int) -> np.ndarray:
    src_idx = np.linspace(0.0, 1.0, num=len(x))
    dst_idx = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(dst_idx, src_idx, x)


def empirical_uniform(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / len(x)


def shannon_entropy(prob: np.ndarray) -> float:
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def kmeans_nd(X: np.ndarray, n_states: int, seed: int = 7, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    choose = np.linspace(0, len(X) - 1, n_states, dtype=int)
    centers = X[choose].copy()
    centers += 1e-4 * rng.normal(size=centers.shape)

    for _ in range(max_iter):
        distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        updated = np.array(
            [X[labels == idx].mean(axis=0) if np.any(labels == idx) else centers[idx] for idx in range(n_states)],
            dtype=float,
        )
        if np.allclose(updated, centers):
            break
        centers = updated

    return labels.astype(np.int32), centers


def gray_code_table(n_states: int) -> np.ndarray:
    dims = int(math.ceil(math.log2(n_states)))
    table = []
    for idx in range(n_states):
        gray = idx ^ (idx >> 1)
        bits = [1 if (gray >> bit) & 1 else -1 for bit in range(dims)]
        table.append(bits)
    return np.array(table, dtype=float)


def build_feature_matrix(ai: np.ndarray) -> np.ndarray:
    abs_diff = np.r_[0.0, np.abs(np.diff(ai))]
    local_jump = rolling_mean(abs_diff, window=15)
    local_std = rolling_std(ai, window=25)
    features = np.column_stack([ai, local_jump, local_std]).astype(float)
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-12)
    return (features - mean) / std


def reorder_states(labels: np.ndarray, centers: np.ndarray, ai: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ai_centers = np.array([ai[labels == idx].mean() if np.any(labels == idx) else np.inf for idx in range(len(centers))])
    order = np.argsort(ai_centers)
    remap = np.zeros_like(order)
    remap[order] = np.arange(len(order))
    new_labels = remap[labels]
    new_centers = centers[order]
    return new_labels, new_centers


def build_transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    for prev_state, next_state in zip(labels[:-1], labels[1:]):
        counts[prev_state, next_state] += 1.0
    row_sum = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(row_sum, 1.0)


def resample_labels(labels: np.ndarray, target_len: int) -> np.ndarray:
    src_idx = np.linspace(0.0, 1.0, num=len(labels))
    dst_idx = np.linspace(0.0, 1.0, num=target_len)
    return np.rint(np.interp(dst_idx, src_idx, labels)).astype(np.int32)


def build_local_occupancy(labels: np.ndarray, n_states: int, target_len: int, window: int = 31) -> np.ndarray:
    resampled = resample_labels(labels, target_len)
    occupancy = np.zeros((target_len, n_states), dtype=float)
    half = window // 2
    for idx in range(target_len):
        left = max(0, idx - half)
        right = min(target_len, idx + half + 1)
        chunk = resampled[left:right]
        occupancy[idx] = np.bincount(chunk, minlength=n_states) / len(chunk)
    return occupancy


def build_temperature(labels: np.ndarray, target_len: int, t_min: float = 0.15, t_max: float = 2.2) -> np.ndarray:
    transitions = np.r_[0.0, (labels[1:] != labels[:-1]).astype(float)]
    activity = rolling_mean(transitions, window=25)
    activity = resample_series(activity, target_len)
    norm = (activity - activity.min()) / (activity.max() - activity.min() + 1e-12)
    envelope = np.maximum.accumulate(norm[::-1])[::-1]
    return t_min + (t_max - t_min) * envelope


def build_random_matrix(ai: np.ndarray, target_len: int, dims: int) -> np.ndarray:
    abs_diff = np.r_[0.0, np.abs(np.diff(ai))]
    local_std = rolling_std(ai, window=25)
    sources = [
        resample_series(ai, target_len),
        resample_series(abs_diff, target_len),
        resample_series(local_std, target_len),
    ]
    channels = []
    for dim in range(dims):
        base = sources[dim % len(sources)]
        rolled = np.roll(base, dim * max(3, target_len // (4 * dims)))
        channels.append(empirical_uniform(rolled))
    return np.column_stack(channels)


def build_pbit_biases(occupancy: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    return occupancy @ codebook


def profile_metrics(
    labels: np.ndarray,
    occupancy: np.ndarray,
    transition_matrix: np.ndarray,
    pbit_biases: np.ndarray,
    temperature: np.ndarray,
) -> dict:
    state_prob = np.bincount(labels, minlength=occupancy.shape[1]).astype(float)
    state_prob /= np.maximum(state_prob.sum(), 1.0)
    trans_entropy = [
        shannon_entropy(row[row > 0]) if np.any(row > 0) else 0.0
        for row in transition_matrix
    ]
    bias_corr = np.corrcoef(pbit_biases.T) if pbit_biases.shape[1] > 1 else np.array([[1.0]])
    return {
        "n_states": int(occupancy.shape[1]),
        "pbit_dims": int(pbit_biases.shape[1]),
        "state_entropy_bits": shannon_entropy(state_prob),
        "state_counts": np.bincount(labels, minlength=occupancy.shape[1]).tolist(),
        "transition_count": int(np.sum(labels[1:] != labels[:-1])),
        "mean_transition_entropy_bits": float(np.mean(trans_entropy)),
        "temperature_start": float(temperature[0]),
        "temperature_end": float(temperature[-1]),
        "temperature_area": float(np.trapezoid(temperature)),
        "mean_abs_pbit_bias": float(np.mean(np.abs(pbit_biases))),
        "pbit_bias_corr": bias_corr.tolist(),
    }


def build_multistate_profile(dataset_name: str, n_states: int = N_STATES, steps: int = STEPS) -> MultiStateProfile:
    dataset = load_dataset(GOOD_FILE if dataset_name == "good" else BAD_FILE, dataset_name)
    features = build_feature_matrix(dataset.ai)
    labels, centers = kmeans_nd(features, n_states=n_states)
    labels, centers = reorder_states(labels, centers, dataset.ai)

    transition_matrix = build_transition_matrix(labels, n_states=n_states)
    occupancy = build_local_occupancy(labels, n_states=n_states, target_len=steps)
    codebook = gray_code_table(n_states)
    pbit_biases = build_pbit_biases(occupancy, codebook)
    random_u = build_random_matrix(dataset.ai, target_len=steps, dims=codebook.shape[1])
    temperature = build_temperature(labels, target_len=steps)
    metrics = profile_metrics(labels, occupancy, transition_matrix, pbit_biases, temperature)

    return MultiStateProfile(
        name=dataset_name,
        state_labels=labels,
        state_centers=centers,
        transition_matrix=transition_matrix,
        occupancy=occupancy,
        pbit_biases=pbit_biases,
        random_u=random_u,
        temperature=temperature,
        metrics=metrics,
    )


def compute_energy_scale(J: np.ndarray, h: np.ndarray) -> float:
    couplings = np.abs(J[np.triu_indices_from(J, k=1)])
    couplings = couplings[couplings > 0]
    coupling_scale = float(np.median(couplings)) if len(couplings) else 1.0
    field_scale = float(np.mean(np.abs(h))) if len(h) else 0.0
    return max(1.0, coupling_scale + field_scale)


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def run_multidim_pbit_ising(
    profile: MultiStateProfile,
    J: np.ndarray,
    h: np.ndarray,
    optimum: float,
    run_idx: int,
    coupling_gain: float = 2.4,
    data_gain: float = 1.2,
) -> dict:
    n = len(h)
    dims = profile.pbit_biases.shape[1]
    energy_scale = compute_energy_scale(J, h)
    offset = (run_idx * 23) % len(profile.random_u)

    init_bias = profile.pbit_biases[offset]
    spins = np.ones(n, dtype=np.int8)
    for idx in range(n):
        spins[idx] = 1 if init_bias[idx % dims] >= 0 else -1

    current = ising_energy(spins, J, h)
    best = current
    accepted = 0
    early_accept = 0
    late_accept = 0
    early_total = 0
    late_total = 0

    for step in range(len(profile.temperature)):
        temp = float(profile.temperature[step])
        for dim in range(dims):
            spin_idx = (step * dims + dim + offset) % n
            local_field = float(J[spin_idx] @ spins + h[spin_idx])
            drive = coupling_gain * local_field / max(energy_scale * temp, 1e-9)
            drive += data_gain * float(profile.pbit_biases[step, dim])
            prob_plus = logistic(2.0 * drive)
            new_spin = 1 if profile.random_u[(offset + step) % len(profile.random_u), dim] < prob_plus else -1

            if step < len(profile.temperature) // 5:
                early_total += 1
                early_accept += int(new_spin != spins[spin_idx])
            if step >= len(profile.temperature) * 4 // 5:
                late_total += 1
                late_accept += int(new_spin != spins[spin_idx])

            if new_spin != spins[spin_idx]:
                spins[spin_idx] = new_spin
                accepted += 1
                current = ising_energy(spins, J, h)
                if current < best:
                    best = current

    return {
        "best_energy": float(best),
        "final_energy": float(current),
        "energy_gap_to_optimum": float(best - optimum),
        "hit_optimum": bool(abs(best - optimum) < 1e-9),
        "acceptance_rate": float(accepted / (len(profile.temperature) * dims)),
        "early_acceptance_rate": float(early_accept / max(early_total, 1)),
        "late_acceptance_rate": float(late_accept / max(late_total, 1)),
    }


def state_dataframe(profile: MultiStateProfile) -> pd.DataFrame:
    rows = []
    for idx, state in enumerate(profile.state_labels):
        rows.append({"dataset": profile.name, "sample_idx": idx, "state": int(state)})
    return pd.DataFrame(rows)


def save_outputs(profiles: list[MultiStateProfile], comparison_rows: list[dict], summary: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "multidim_pbit_runs.csv", index=False)
    pd.concat([state_dataframe(profile) for profile in profiles], ignore_index=True).to_csv(
        OUTPUT_DIR / "multistate_labels.csv",
        index=False,
    )
    with open(OUTPUT_DIR / "multistate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    for profile in profiles:
        axes[0].plot(profile.temperature, label=f"{profile.name} temperature", linewidth=1.4)
        for dim in range(profile.pbit_biases.shape[1]):
            axes[1].plot(
                profile.pbit_biases[:, dim],
                label=f"{profile.name} pbit{dim}",
                linewidth=1.0,
                alpha=0.85,
            )
        state_hist = np.bincount(profile.state_labels, minlength=profile.occupancy.shape[1])
        axes[2].plot(state_hist, marker="o", label=f"{profile.name} state count", linewidth=1.2)

    axes[0].set_title("Multistate-derived temperature")
    axes[0].set_ylabel("Temperature")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Multi-dimensional p-bit biases")
    axes[1].set_ylabel("Bias")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[2].set_title("Multistate occupancy")
    axes[2].set_xlabel("State index")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multistate_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    profiles = [
        build_multistate_profile("good"),
        build_multistate_profile("bad"),
    ]
    J, h, optimum = build_ising_instance()

    comparison_rows = []
    profile_summary = {}
    optimization_summary = {}

    for profile in profiles:
        runs = [
            run_multidim_pbit_ising(profile, J, h, optimum, run_idx=run_idx)
            for run_idx in range(40)
        ]
        profile_summary[profile.name] = profile.metrics
        optimization_summary[profile.name] = summarize_runs(runs)

        for run_idx, result in enumerate(runs):
            comparison_rows.append(
                {
                    "dataset": profile.name,
                    "run_idx": run_idx,
                    **result,
                }
            )

    ranked = sorted(
        profile_summary.keys(),
        key=lambda name: (
            optimization_summary[name]["mean_energy_gap"],
            -optimization_summary[name]["optimum_hit_rate"],
            -profile_summary[name]["state_entropy_bits"],
        ),
    )

    summary = {
        "assumption": (
            "Use multistate segmentation on MTJ traces and convert local state occupancy into a "
            "multi-dimensional p-bit bias vector for stochastic Ising optimization."
        ),
        "configuration": {
            "n_states": N_STATES,
            "pbit_dims": int(math.ceil(math.log2(N_STATES))),
            "steps": STEPS,
            "problem_type": "18-spin Ising minimization",
            "exact_optimum_energy": optimum,
            "runs_per_dataset": 40,
        },
        "multistate_profile": profile_summary,
        "optimization_performance": optimization_summary,
        "overall_ranking": ranked,
    }

    save_outputs(profiles, comparison_rows, summary)

    print("MTJ multistate p-bit demo finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

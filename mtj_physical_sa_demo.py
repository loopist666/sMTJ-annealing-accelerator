import json
import math
from dataclasses import asdict, dataclass
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
    delta_energy_for_flip,
    ising_energy,
    load_dataset,
    summarize_runs,
)


OUTPUT_DIR = Path("mtj_physical_sa_output")


@dataclass
class DwellSegment:
    state: int
    start_time: float
    end_time: float
    duration: float
    start_index: int
    end_index: int


@dataclass
class PhysicalProfile:
    name: str
    state_sequence: np.ndarray
    state_bits: np.ndarray
    random_u: np.ndarray
    base_rate: np.ndarray
    dt_step: float
    low_mean: float
    high_mean: float
    low_threshold: float
    high_threshold: float
    dwell_segments: list[DwellSegment]
    metrics: dict


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def resample_series(x: np.ndarray, target_len: int) -> np.ndarray:
    src_idx = np.linspace(0.0, 1.0, num=len(x))
    dst_idx = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(dst_idx, src_idx, x)


def two_state_kmeans(x: np.ndarray, max_iter: int = 60) -> tuple[float, float]:
    centers = np.array([np.percentile(x, 25), np.percentile(x, 75)], dtype=float)
    for _ in range(max_iter):
        distances = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        updated = np.array(
            [x[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(2)],
            dtype=float,
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    centers.sort()
    return float(centers[0]), float(centers[1])


def infer_two_state_sequence(ai: np.ndarray, hysteresis_fraction: float = 0.35) -> tuple[np.ndarray, dict]:
    low_mean, high_mean = two_state_kmeans(ai)
    gap = high_mean - low_mean
    low_threshold = low_mean + hysteresis_fraction * gap
    high_threshold = high_mean - hysteresis_fraction * gap

    states = np.zeros(len(ai), dtype=np.int8)
    midpoint = 0.5 * (low_mean + high_mean)
    states[0] = int(ai[0] >= midpoint)

    for idx in range(1, len(ai)):
        if ai[idx] <= low_threshold:
            states[idx] = 0
        elif ai[idx] >= high_threshold:
            states[idx] = 1
        else:
            states[idx] = states[idx - 1]

    return states, {
        "low_mean": float(low_mean),
        "high_mean": float(high_mean),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
    }


def extract_dwell_segments(time: np.ndarray, states: np.ndarray) -> list[DwellSegment]:
    sample_dt = float(np.median(np.diff(time)))
    dwells: list[DwellSegment] = []
    start_idx = 0

    for idx in range(1, len(states)):
        if states[idx] != states[idx - 1]:
            start_time = float(time[start_idx])
            end_time = float(time[idx])
            dwells.append(
                DwellSegment(
                    state=int(states[idx - 1]),
                    start_time=start_time,
                    end_time=end_time,
                    duration=max(end_time - start_time, sample_dt),
                    start_index=start_idx,
                    end_index=idx - 1,
                )
            )
            start_idx = idx

    final_start = float(time[start_idx])
    final_end = float(time[-1] + sample_dt)
    dwells.append(
        DwellSegment(
            state=int(states[-1]),
            start_time=final_start,
            end_time=final_end,
            duration=max(final_end - final_start, sample_dt),
            start_index=start_idx,
            end_index=len(states) - 1,
        )
    )
    return dwells


def fit_exponential_duration_model(durations: np.ndarray) -> dict:
    if len(durations) == 0:
        return {
            "count": 0,
            "mean_duration": float("nan"),
            "rate": float("nan"),
            "cv": float("nan"),
            "ks_distance": float("nan"),
        }

    mean_duration = float(np.mean(durations))
    rate = 1.0 / max(mean_duration, 1e-12)
    sorted_d = np.sort(durations)
    empirical_cdf = np.arange(1, len(sorted_d) + 1, dtype=float) / len(sorted_d)
    model_cdf = 1.0 - np.exp(-rate * sorted_d)
    ks_distance = float(np.max(np.abs(empirical_cdf - model_cdf)))

    return {
        "count": int(len(durations)),
        "mean_duration": mean_duration,
        "rate": float(rate),
        "cv": float(np.std(durations) / max(mean_duration, 1e-12)),
        "ks_distance": ks_distance,
    }


def build_time_rescaled_uniforms(dwells: list[DwellSegment], low_rate: float, high_rate: float) -> np.ndarray:
    if not dwells:
        return np.linspace(0.1, 0.9, 32)

    uniforms = []
    for dwell in dwells:
        rate = high_rate if dwell.state == 1 else low_rate
        u = 1.0 - math.exp(-rate * dwell.duration)
        u = min(max(u, 1e-6), 1.0 - 1e-6)
        uniforms.extend([u, 1.0 - u])
    return np.array(uniforms, dtype=float)


def build_local_switch_rate(
    time: np.ndarray,
    states: np.ndarray,
    steps: int,
    prior_rate: float,
    alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    transitions = time[1:][states[1:] != states[:-1]]
    sample_dt = float(np.median(np.diff(time)))
    edges = np.linspace(float(time[0]), float(time[-1] + sample_dt), steps + 1)
    counts, _ = np.histogram(transitions, bins=edges)
    bin_dt = np.diff(edges)
    beta = alpha / max(prior_rate, 1e-9)
    local_rate = (counts + alpha) / np.maximum(bin_dt + beta, 1e-12)
    local_rate = rolling_mean(local_rate, window=max(5, steps // 50))
    local_rate = np.clip(local_rate, 1e-9, None)
    return local_rate, float(np.mean(bin_dt))


def effective_barrier_from_rate(rate: float, attempt_rate: float) -> float:
    return float(max(0.0, math.log(max(attempt_rate, rate) / max(rate, 1e-12))))


def build_physical_profile(dataset_name: str, steps: int = 320) -> PhysicalProfile:
    dataset = load_dataset(GOOD_FILE if dataset_name == "good" else BAD_FILE, dataset_name)
    states, state_info = infer_two_state_sequence(dataset.ai)
    dwell_segments = extract_dwell_segments(dataset.time, states)

    low_durations = np.array([d.duration for d in dwell_segments if d.state == 0], dtype=float)
    high_durations = np.array([d.duration for d in dwell_segments if d.state == 1], dtype=float)
    low_fit = fit_exponential_duration_model(low_durations)
    high_fit = fit_exponential_duration_model(high_durations)

    prior_rate = np.mean([low_fit["rate"], high_fit["rate"]])
    local_rate, dt_step = build_local_switch_rate(dataset.time, states, steps=steps, prior_rate=prior_rate)
    random_u = build_time_rescaled_uniforms(
        dwell_segments,
        low_rate=low_fit["rate"],
        high_rate=high_fit["rate"],
    )
    random_u = resample_series(random_u, steps * 4)

    state_bits = np.rint(resample_series(states.astype(float), steps * 3)).astype(np.int8)
    sample_dt = float(np.median(np.diff(dataset.time)))
    attempt_rate = 10.0 / max(sample_dt, 1e-12)

    metrics = {
        "samples": int(len(dataset.ai)),
        "transition_count": int(np.sum(states[1:] != states[:-1])),
        "state_occupancy_high": float(np.mean(states)),
        "state_occupancy_low": float(1.0 - np.mean(states)),
        "low_state_model": low_fit,
        "high_state_model": high_fit,
        "local_rate_mean": float(np.mean(local_rate)),
        "local_rate_std": float(np.std(local_rate)),
        "local_rate_early_mean": float(np.mean(local_rate[: steps // 4])),
        "local_rate_late_mean": float(np.mean(local_rate[-steps // 4 :])),
        "local_rate_drop_ratio": float(np.mean(local_rate[: steps // 4]) / max(np.mean(local_rate[-steps // 4 :]), 1e-12)),
        "effective_barrier_low_state_log_units": effective_barrier_from_rate(low_fit["rate"], attempt_rate),
        "effective_barrier_high_state_log_units": effective_barrier_from_rate(high_fit["rate"], attempt_rate),
        "random_u_mean": float(np.mean(random_u)),
        "random_u_std": float(np.std(random_u)),
        "sample_dt": sample_dt,
        **state_info,
    }

    return PhysicalProfile(
        name=dataset_name,
        state_sequence=states,
        state_bits=state_bits,
        random_u=random_u,
        base_rate=local_rate,
        dt_step=dt_step,
        low_mean=state_info["low_mean"],
        high_mean=state_info["high_mean"],
        low_threshold=state_info["low_threshold"],
        high_threshold=state_info["high_threshold"],
        dwell_segments=dwell_segments,
        metrics=metrics,
    )


def compute_energy_scale(J: np.ndarray, h: np.ndarray) -> float:
    couplings = np.abs(J[np.triu_indices_from(J, k=1)])
    couplings = couplings[couplings > 0]
    coupling_scale = float(np.median(couplings)) if len(couplings) else 1.0
    field_scale = float(np.mean(np.abs(h))) if len(h) else 0.0
    return max(1.0, coupling_scale + field_scale)


def run_physical_sa(
    profile: PhysicalProfile,
    J: np.ndarray,
    h: np.ndarray,
    optimum: float,
    run_idx: int,
    energy_scale: float,
) -> dict:
    n = len(h)
    offset = (run_idx * 29) % len(profile.random_u)
    spins = np.where(profile.state_bits[offset : offset + n] > 0, 1, -1).astype(np.int8)
    if len(spins) < n:
        extra = np.where(profile.state_bits[: n - len(spins)] > 0, 1, -1).astype(np.int8)
        spins = np.concatenate([spins, extra])

    current = ising_energy(spins, J, h)
    best = current
    accepted = 0
    early_accept = 0
    late_accept = 0
    early_total = 0
    late_total = 0

    for step, base_rate in enumerate(profile.base_rate):
        r_pick = profile.random_u[(offset + 2 * step) % len(profile.random_u)]
        r_accept = profile.random_u[(offset + 2 * step + 1) % len(profile.random_u)]
        flip_idx = min(int(r_pick * n), n - 1)
        delta = delta_energy_for_flip(spins, J, h, flip_idx)

        if delta <= 0.0:
            switch_rate = base_rate
        else:
            switch_rate = base_rate * math.exp(-delta / max(energy_scale, 1e-9))

        accept_prob = 1.0 - math.exp(-switch_rate * profile.dt_step)
        accept = r_accept < accept_prob

        if step < len(profile.base_rate) // 5:
            early_total += 1
            early_accept += int(accept)
        if step >= len(profile.base_rate) * 4 // 5:
            late_total += 1
            late_accept += int(accept)

        if accept:
            spins[flip_idx] *= -1
            current += delta
            accepted += 1
            if current < best:
                best = current

    return {
        "best_energy": float(best),
        "final_energy": float(current),
        "energy_gap_to_optimum": float(best - optimum),
        "hit_optimum": bool(abs(best - optimum) < 1e-9),
        "acceptance_rate": float(accepted / len(profile.base_rate)),
        "early_acceptance_rate": float(early_accept / max(early_total, 1)),
        "late_acceptance_rate": float(late_accept / max(late_total, 1)),
    }


def dwell_dataframe(profile: PhysicalProfile) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": profile.name,
                "state": d.state,
                "start_time": d.start_time,
                "end_time": d.end_time,
                "duration": d.duration,
                "start_index": d.start_index,
                "end_index": d.end_index,
            }
            for d in profile.dwell_segments
        ]
    )


def save_outputs(
    profiles: list[PhysicalProfile],
    comparison_rows: list[dict],
    summary: dict,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "physical_optimization_runs.csv", index=False)
    pd.concat([dwell_dataframe(profile) for profile in profiles], ignore_index=True).to_csv(
        OUTPUT_DIR / "dwell_segments.csv",
        index=False,
    )
    with open(OUTPUT_DIR / "physical_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for profile in profiles:
        axes[0].plot(profile.base_rate, label=f"{profile.name} switch rate", linewidth=1.4)
        axes[1].hist(
            [d.duration for d in profile.dwell_segments],
            bins=18,
            alpha=0.45,
            label=f"{profile.name} dwell",
        )
    axes[0].set_title("MTJ empirical switching rate")
    axes[0].set_ylabel("Rate")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Dwell-time distribution")
    axes[1].set_xlabel("Duration")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "physical_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    profiles = [
        build_physical_profile("good"),
        build_physical_profile("bad"),
    ]
    J, h, optimum = build_ising_instance()
    energy_scale = compute_energy_scale(J, h)

    comparison_rows = []
    physical_summary = {}
    optimization_summary = {}

    for profile in profiles:
        runs = [
            run_physical_sa(profile, J, h, optimum, run_idx=i, energy_scale=energy_scale)
            for i in range(40)
        ]
        physical_summary[profile.name] = profile.metrics
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
        physical_summary.keys(),
        key=lambda name: (
            optimization_summary[name]["mean_energy_gap"],
            -optimization_summary[name]["optimum_hit_rate"],
            -physical_summary[name]["local_rate_drop_ratio"],
        ),
    )

    summary = {
        "assumption": (
            "Use explicit two-state segmentation, dwell-time extraction, exponential dwell models, "
            "and empirical switching rates to drive a hazard-based MTJ annealing demo."
        ),
        "problem": {
            "type": "18-spin Ising minimization",
            "exact_optimum_energy": optimum,
            "runs_per_dataset": 40,
            "energy_scale": energy_scale,
        },
        "physical_model_summary": physical_summary,
        "optimization_performance": optimization_summary,
        "overall_ranking": ranked,
    }

    save_outputs(profiles, comparison_rows, summary)

    print("MTJ physical SA demo finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

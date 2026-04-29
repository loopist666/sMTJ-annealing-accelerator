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


GOOD_FILE = "simulated annealing.xlsx"
BAD_FILE = "simulated annealing_bad.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_DIR = Path("mtj_hardware_sa_output")


@dataclass
class Dataset:
    name: str
    time: np.ndarray
    ai: np.ndarray
    av: np.ndarray


@dataclass
class AnnealProfile:
    name: str
    temperature: np.ndarray
    random_u: np.ndarray
    state_bits: np.ndarray
    jump_intensity: np.ndarray
    metrics: dict


def load_dataset(path: str, name: str) -> Dataset:
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    for col in ["Time", "AI", "AV"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time", "AI", "AV"]).sort_values("Time")
    return Dataset(
        name=name,
        time=df["Time"].to_numpy(dtype=float),
        ai=df["AI"].to_numpy(dtype=float),
        av=df["AV"].to_numpy(dtype=float),
    )


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


def empirical_uniform(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return (ranks + 0.5) / len(x)


def shannon_entropy(prob: np.ndarray) -> float:
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def cooling_metrics(ai: np.ndarray, jump_intensity: np.ndarray, temperature: np.ndarray, random_u: np.ndarray) -> dict:
    n = len(ai)
    quarter = max(8, n // 4)
    early = ai[:quarter]
    late = ai[-quarter:]
    early_jump = np.abs(np.diff(early))
    late_jump = np.abs(np.diff(late))

    bins = 16
    hist, _ = np.histogram(random_u, bins=bins, range=(0.0, 1.0), density=True)
    hist = hist / np.sum(hist)

    temp_diff = np.diff(temperature)
    monotone_ratio = float(np.mean(temp_diff <= 0.0))

    return {
        "samples": int(n),
        "early_std": float(np.std(early)),
        "late_std": float(np.std(late)),
        "std_drop_ratio": float(np.std(early) / (np.std(late) + 1e-12)),
        "early_mean_abs_jump": float(np.mean(early_jump)),
        "late_mean_abs_jump": float(np.mean(late_jump)),
        "jump_drop_ratio": float(np.mean(early_jump) / (np.mean(late_jump) + 1e-12)),
        "cooling_monotone_ratio": monotone_ratio,
        "temperature_start": float(temperature[0]),
        "temperature_end": float(temperature[-1]),
        "temperature_area": float(np.trapezoid(temperature)),
        "random_entropy_bits_16bins": shannon_entropy(hist),
        "jump_intensity_mean": float(np.mean(jump_intensity)),
        "jump_intensity_std": float(np.std(jump_intensity)),
        "ai_mean": float(np.mean(ai)),
    }


def build_anneal_profile(dataset: Dataset, steps: int = 600, t_min: float = 0.02, t_max: float = 2.5) -> AnnealProfile:
    ai = dataset.ai
    diff = np.abs(np.diff(ai))
    if len(diff) < 8:
        raise ValueError(f"{dataset.name} has too few samples for annealing profile.")

    raw_jump = resample_series(diff, steps)
    smooth_jump = rolling_mean(raw_jump, window=max(5, steps // 40))
    jump_min = smooth_jump.min()
    jump_span = smooth_jump.max() - jump_min
    normalized_jump = (smooth_jump - jump_min) / (jump_span + 1e-12)

    # Keep the hardware-derived trend but enforce a non-increasing cooling envelope.
    cooling_envelope = np.maximum.accumulate(normalized_jump[::-1])[::-1]
    temperature = t_min + (t_max - t_min) * cooling_envelope

    ai_resampled = resample_series(ai, steps * 3)
    random_u = empirical_uniform(ai_resampled)

    state_bits = (ai_resampled > np.median(ai)).astype(np.int8)
    metrics = cooling_metrics(ai, smooth_jump, temperature, random_u)

    return AnnealProfile(
        name=dataset.name,
        temperature=temperature,
        random_u=random_u,
        state_bits=state_bits,
        jump_intensity=smooth_jump,
        metrics=metrics,
    )


def build_ising_instance(n_spins: int = 18, seed: int = 11) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    upper = rng.integers(-3, 4, size=(n_spins, n_spins))
    mask = rng.random((n_spins, n_spins)) < 0.38
    upper = np.triu(upper * mask, 1)
    J = upper + upper.T
    h = rng.integers(-2, 3, size=n_spins).astype(float)
    optimum = brute_force_ising_optimum(J, h)
    return J.astype(float), h, float(optimum)


def brute_force_ising_optimum(J: np.ndarray, h: np.ndarray) -> float:
    n = len(h)
    states = np.arange(1 << n, dtype=np.uint32)
    bits = ((states[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8)
    spins = bits * 2 - 1
    energies = -0.5 * np.einsum("bi,ij,bj->b", spins, J, spins) - spins @ h
    return float(np.min(energies))


def ising_energy(spins: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
    return float(-0.5 * spins @ J @ spins - h @ spins)


def delta_energy_for_flip(spins: np.ndarray, J: np.ndarray, h: np.ndarray, idx: int) -> float:
    local_field = J[idx] @ spins + h[idx]
    return float(2.0 * spins[idx] * local_field)


def run_hardware_sa(
    profile: AnnealProfile,
    J: np.ndarray,
    h: np.ndarray,
    optimum: float,
    run_idx: int,
) -> dict:
    n = len(h)
    temp = profile.temperature
    rand = profile.random_u
    states = profile.state_bits

    offset = (run_idx * 37) % len(rand)
    spins = np.where(states[offset : offset + n] > 0, 1, -1).astype(np.int8)
    if len(spins) < n:
        extra = np.where(states[: n - len(spins)] > 0, 1, -1).astype(np.int8)
        spins = np.concatenate([spins, extra])

    current = ising_energy(spins, J, h)
    best = current
    accepted = 0
    early_accept = 0
    late_accept = 0
    early_total = 0
    late_total = 0

    for step, temperature in enumerate(temp):
        r_pick = rand[(offset + 2 * step) % len(rand)]
        r_accept = rand[(offset + 2 * step + 1) % len(rand)]
        flip_idx = min(int(r_pick * n), n - 1)
        delta = delta_energy_for_flip(spins, J, h, flip_idx)

        accept = delta <= 0.0
        if not accept:
            accept_prob = math.exp(-delta / max(float(temperature), 1e-9))
            accept = r_accept < accept_prob

        if step < len(temp) // 5:
            early_total += 1
            early_accept += int(accept)
        if step >= len(temp) * 4 // 5:
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
        "acceptance_rate": float(accepted / len(temp)),
        "early_acceptance_rate": float(early_accept / max(early_total, 1)),
        "late_acceptance_rate": float(late_accept / max(late_total, 1)),
    }


def summarize_runs(results: list[dict]) -> dict:
    best_energies = np.array([r["best_energy"] for r in results], dtype=float)
    gaps = np.array([r["energy_gap_to_optimum"] for r in results], dtype=float)
    acc = np.array([r["acceptance_rate"] for r in results], dtype=float)
    early = np.array([r["early_acceptance_rate"] for r in results], dtype=float)
    late = np.array([r["late_acceptance_rate"] for r in results], dtype=float)
    hit_rate = float(np.mean([r["hit_optimum"] for r in results]))
    return {
        "runs": int(len(results)),
        "mean_best_energy": float(np.mean(best_energies)),
        "std_best_energy": float(np.std(best_energies)),
        "best_energy_seen": float(np.min(best_energies)),
        "mean_energy_gap": float(np.mean(gaps)),
        "median_energy_gap": float(np.median(gaps)),
        "max_energy_gap": float(np.max(gaps)),
        "optimum_hit_rate": hit_rate,
        "mean_acceptance_rate": float(np.mean(acc)),
        "mean_early_acceptance": float(np.mean(early)),
        "mean_late_acceptance": float(np.mean(late)),
    }


def save_artifacts(
    profiles: list[AnnealProfile],
    comparison_rows: list[dict],
    summary: dict,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "optimization_runs.csv", index=False)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for profile in profiles:
        axes[0].plot(profile.temperature, label=f"{profile.name} temperature", linewidth=1.5)
        axes[1].plot(profile.jump_intensity, label=f"{profile.name} jump intensity", linewidth=1.2)
    axes[0].set_title("MTJ-derived cooling schedule")
    axes[0].set_ylabel("Temperature")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].set_title("MTJ jump intensity")
    axes[1].set_xlabel("Annealing step")
    axes[1].set_ylabel("Mean abs jump")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cooling_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    good = load_dataset(GOOD_FILE, "good")
    bad = load_dataset(BAD_FILE, "bad")

    profiles = [
        build_anneal_profile(good),
        build_anneal_profile(bad),
    ]
    J, h, optimum = build_ising_instance()

    comparison_rows = []
    profile_summary = {}
    optimization_summary = {}

    for profile in profiles:
        runs = [
            run_hardware_sa(profile, J, h, optimum, run_idx=i)
            for i in range(40)
        ]
        profile_summary[profile.name] = profile.metrics
        profile_summary[profile.name]["av_mean"] = float(
            np.mean(good.av if profile.name == "good" else bad.av)
        )
        optimization_summary[profile.name] = summarize_runs(runs)

        for i, result in enumerate(runs):
            comparison_rows.append(
                {
                    "dataset": profile.name,
                    "run_idx": i,
                    **result,
                }
            )

    ranking_key = lambda name: (
        optimization_summary[name]["mean_energy_gap"],
        -optimization_summary[name]["optimum_hit_rate"],
        -profile_summary[name]["std_drop_ratio"],
    )
    ranked = sorted(profile_summary.keys(), key=ranking_key)

    summary = {
        "assumption": (
            "Treat MTJ random telegraph-like jumps as a hardware annealing resource. "
            "Local jump intensity defines a cooling schedule, and the resampled signal "
            "provides random numbers and binary states for a small Ising optimization demo."
        ),
        "problem": {
            "type": "18-spin Ising minimization",
            "exact_optimum_energy": optimum,
            "runs_per_dataset": 40,
        },
        "dataset_quality": profile_summary,
        "optimization_performance": optimization_summary,
        "overall_ranking": ranked,
    }

    save_artifacts(profiles, comparison_rows, summary)

    print("MTJ hardware SA demo finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

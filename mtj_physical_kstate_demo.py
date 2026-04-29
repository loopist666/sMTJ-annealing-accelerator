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
    delta_energy_for_flip,
    ising_energy,
    load_dataset,
    summarize_runs,
)


OUTPUT_DIR = Path("mtj_physical_kstate_output")


@dataclass
class DwellSegment:
    state: int
    start_time: float
    end_time: float
    duration: float
    start_index: int
    end_index: int


@dataclass
class KStateProfile:
    name: str
    selected_k: int
    state_labels: np.ndarray
    state_means: np.ndarray
    state_vars: np.ndarray
    state_weights: np.ndarray
    transition_matrix: np.ndarray
    exit_rates: np.ndarray
    dwell_segments: list[DwellSegment]
    step_states: np.ndarray
    state_biases: np.ndarray
    random_u: np.ndarray
    dt_step: float
    k_selection_table: list[dict]
    stability_rows: list[dict]
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


def gaussian_pdf_1d(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    var = np.maximum(var, 1e-8)
    diff = x[:, None] - mean[None, :]
    norm = 1.0 / np.sqrt(2.0 * math.pi * var[None, :])
    return norm * np.exp(-0.5 * diff * diff / var[None, :])


def fit_gmm_1d(x: np.ndarray, k: int, n_init: int = 4, max_iter: int = 100) -> dict:
    x = x.astype(float)
    n = len(x)
    global_var = float(np.var(x) + 1e-6)
    best = None

    for init_idx in range(n_init):
        means = np.quantile(x, np.linspace(0.1, 0.9, k))
        means = means + (init_idx - n_init / 2.0) * 1e-4
        vars_ = np.full(k, global_var, dtype=float)
        weights = np.full(k, 1.0 / k, dtype=float)
        prev_ll = -np.inf

        for _ in range(max_iter):
            pdf = gaussian_pdf_1d(x, means, vars_) * weights[None, :]
            denom = np.maximum(pdf.sum(axis=1, keepdims=True), 1e-12)
            resp = pdf / denom

            nk = np.maximum(resp.sum(axis=0), 1e-9)
            weights = nk / n
            means = (resp * x[:, None]).sum(axis=0) / nk
            vars_ = (resp * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / nk
            vars_ = np.maximum(vars_, global_var * 1e-4)

            pdf = gaussian_pdf_1d(x, means, vars_) * weights[None, :]
            ll = float(np.sum(np.log(np.maximum(pdf.sum(axis=1), 1e-12))))
            if abs(ll - prev_ll) < 1e-7:
                break
            prev_ll = ll

        params = 3 * k - 1
        bic = -2.0 * ll + params * math.log(max(n, 2))
        candidate = {
            "k": k,
            "log_likelihood": ll,
            "bic": float(bic),
            "means": means.copy(),
            "vars": vars_.copy(),
            "weights": weights.copy(),
        }
        if best is None or candidate["bic"] < best["bic"]:
            best = candidate

    return best


def select_k_by_amplitude_model(x: np.ndarray, max_k: int = 8) -> tuple[int, list[dict], dict]:
    results = []
    best_model = None
    for k in range(1, max_k + 1):
        model = fit_gmm_1d(x, k)
        results.append(
            {
                "k": int(k),
                "bic": float(model["bic"]),
                "log_likelihood": float(model["log_likelihood"]),
            }
        )
        if best_model is None or model["bic"] < best_model["bic"]:
            best_model = model
    return int(best_model["k"]), results, best_model


def logsumexp(vec: np.ndarray) -> float:
    vmax = np.max(vec)
    return float(vmax + np.log(np.sum(np.exp(vec - vmax))))


def viterbi_decode(
    x: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
    init_prob: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    emission = np.log(np.maximum(gaussian_pdf_1d(x, means, vars_), 1e-300))
    log_init = np.log(np.maximum(init_prob, 1e-12))
    log_trans = np.log(np.maximum(transition_matrix, 1e-12))

    n = len(x)
    k = len(means)
    score = np.full((n, k), -np.inf, dtype=float)
    back = np.zeros((n, k), dtype=np.int32)
    score[0] = log_init + emission[0]

    for idx in range(1, n):
        for state in range(k):
            candidates = score[idx - 1] + log_trans[:, state]
            prev_state = int(np.argmax(candidates))
            score[idx, state] = candidates[prev_state] + emission[idx, state]
            back[idx, state] = prev_state

    labels = np.zeros(n, dtype=np.int32)
    labels[-1] = int(np.argmax(score[-1]))
    for idx in range(n - 2, -1, -1):
        labels[idx] = back[idx + 1, labels[idx + 1]]
    return labels


def fit_sticky_gaussian_hmm(
    x: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
    weights: np.ndarray,
    n_iter: int = 10,
    self_bias: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = len(means)
    init_prob = weights / np.maximum(np.sum(weights), 1e-12)
    trans = np.full((k, k), 1.0, dtype=float)
    np.fill_diagonal(trans, self_bias)
    trans /= trans.sum(axis=1, keepdims=True)

    labels = np.argmin(np.abs(x[:, None] - means[None, :]), axis=1).astype(np.int32)
    for _ in range(n_iter):
        labels = viterbi_decode(x, means, vars_, init_prob, trans)

        for state in range(k):
            mask = labels == state
            if np.any(mask):
                means[state] = float(np.mean(x[mask]))
                vars_[state] = float(max(np.var(x[mask]), np.var(x) * 1e-4))
                weights[state] = float(np.mean(mask))

        counts = np.ones((k, k), dtype=float)
        counts += np.eye(k) * self_bias
        for prev_state, next_state in zip(labels[:-1], labels[1:]):
            counts[prev_state, next_state] += 1.0
        trans = counts / counts.sum(axis=1, keepdims=True)
        init_prob = np.bincount(labels[: max(1, len(labels) // 20)], minlength=k).astype(float) + 1.0
        init_prob /= init_prob.sum()

    return labels, means, vars_, trans


def reorder_states(
    labels: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
    weights: np.ndarray,
    trans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(means)
    remap = np.zeros_like(order)
    remap[order] = np.arange(len(order))
    new_labels = remap[labels]
    new_means = means[order]
    new_vars = vars_[order]
    new_weights = weights[order]
    new_trans = trans[np.ix_(order, order)]
    return new_labels, new_means, new_vars, new_weights, new_trans


def extract_dwell_segments(time: np.ndarray, labels: np.ndarray) -> list[DwellSegment]:
    sample_dt = float(np.median(np.diff(time)))
    dwells: list[DwellSegment] = []
    start_idx = 0

    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            start_time = float(time[start_idx])
            end_time = float(time[idx])
            dwells.append(
                DwellSegment(
                    state=int(labels[idx - 1]),
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
            state=int(labels[-1]),
            start_time=final_start,
            end_time=final_end,
            duration=max(final_end - final_start, sample_dt),
            start_index=start_idx,
            end_index=len(labels) - 1,
        )
    )
    return dwells


def build_dwell_transition_matrix(dwells: list[DwellSegment], k: int) -> np.ndarray:
    counts = np.ones((k, k), dtype=float)
    for prev_dwell, next_dwell in zip(dwells[:-1], dwells[1:]):
        counts[prev_dwell.state, next_dwell.state] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def fit_state_dwell_models(dwells: list[DwellSegment], k: int) -> tuple[list[dict], np.ndarray]:
    models = []
    exit_rates = np.zeros(k, dtype=float)
    for state in range(k):
        durations = np.array([d.duration for d in dwells if d.state == state], dtype=float)
        if len(durations) == 0:
            model = {
                "state": state,
                "count": 0,
                "mean_duration": float("nan"),
                "rate": float("nan"),
                "cv": float("nan"),
            }
            exit_rates[state] = 1e-6
        else:
            mean_duration = float(np.mean(durations))
            rate = 1.0 / max(mean_duration, 1e-12)
            model = {
                "state": state,
                "count": int(len(durations)),
                "mean_duration": mean_duration,
                "rate": float(rate),
                "cv": float(np.std(durations) / max(mean_duration, 1e-12)),
            }
            exit_rates[state] = rate
        models.append(model)
    return models, exit_rates


def gray_code_table(n_states: int) -> np.ndarray:
    dims = int(math.ceil(math.log2(n_states)))
    table = []
    for idx in range(n_states):
        gray = idx ^ (idx >> 1)
        bits = [1 if (gray >> bit) & 1 else -1 for bit in range(dims)]
        table.append(bits)
    return np.array(table, dtype=float)


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
        rolled = np.roll(base, dim * max(3, target_len // max(4, dims * 2)))
        channels.append(empirical_uniform(rolled))
    return np.column_stack(channels)


def pick_external_axis(dataset) -> tuple[np.ndarray, str]:
    av_unique = np.unique(np.round(dataset.av, decimals=8))
    if len(av_unique) >= 4 and np.std(dataset.av) > 1e-9:
        return dataset.av, "av"
    return dataset.time, "time_window_proxy"


def stability_analysis(dataset, labels: np.ndarray, means: np.ndarray, vars_: np.ndarray, n_bins: int = 5) -> tuple[list[dict], dict]:
    axis, axis_source = pick_external_axis(dataset)
    k = len(means)

    edges = np.linspace(float(np.min(axis)), float(np.max(axis)), n_bins + 1)
    if np.allclose(edges[0], edges[-1]):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        axis = np.linspace(0.0, 1.0, len(labels))
    bin_ids = np.digitize(axis, edges[1:-1], right=False)

    rows = []
    state_presence = np.zeros((k, n_bins), dtype=float)
    state_local_mean = np.full((k, n_bins), np.nan, dtype=float)
    state_local_weight = np.zeros((k, n_bins), dtype=float)

    for bin_idx in range(n_bins):
        mask_bin = bin_ids == bin_idx
        for state in range(k):
            mask = mask_bin & (labels == state)
            occ = float(np.mean(mask_bin & (labels == state))) if np.any(mask_bin) else 0.0
            present = float(occ >= 0.02)
            local_mean = float(np.mean(dataset.ai[mask])) if np.any(mask) else float("nan")
            rows.append(
                {
                    "dataset": dataset.name,
                    "axis_source": axis_source,
                    "bin_idx": bin_idx,
                    "bin_left": float(edges[bin_idx]),
                    "bin_right": float(edges[bin_idx + 1]),
                    "state": state,
                    "occupancy": occ,
                    "present": int(present),
                    "local_ai_mean": local_mean,
                }
            )
            state_presence[state, bin_idx] = present
            state_local_mean[state, bin_idx] = local_mean
            state_local_weight[state, bin_idx] = occ

    state_summary = []
    for state in range(k):
        present_mask = ~np.isnan(state_local_mean[state])
        drift = float(np.nanstd(state_local_mean[state, present_mask])) if np.any(present_mask) else float("nan")
        stable_fraction = float(np.mean(state_presence[state]))
        tolerance = 1.5 * math.sqrt(max(vars_[state], 1e-12))
        stable = bool(stable_fraction >= 0.6 and (math.isnan(drift) or drift <= tolerance))
        state_summary.append(
            {
                "state": state,
                "stable_presence_fraction": stable_fraction,
                "local_mean_drift": drift,
                "tolerance": tolerance,
                "stable_under_axis": stable,
            }
        )

    summary = {
        "axis_source": axis_source,
        "n_bins": n_bins,
        "stable_state_fraction": float(np.mean([row["stable_under_axis"] for row in state_summary])),
        "state_summary": state_summary,
    }
    return rows, summary


def build_kstate_profile(dataset_name: str, max_k: int = 8, steps: int = 360) -> KStateProfile:
    dataset = load_dataset(GOOD_FILE if dataset_name == "good" else BAD_FILE, dataset_name)
    selected_k, k_table, model = select_k_by_amplitude_model(dataset.ai, max_k=max_k)

    labels, means, vars_, trans = fit_sticky_gaussian_hmm(
        dataset.ai.copy(),
        model["means"].copy(),
        model["vars"].copy(),
        model["weights"].copy(),
    )
    weights = np.bincount(labels, minlength=selected_k).astype(float)
    weights /= np.maximum(np.sum(weights), 1.0)
    labels, means, vars_, weights, trans = reorder_states(labels, means, vars_, weights, trans)

    dwells = extract_dwell_segments(dataset.time, labels)
    dwell_models, exit_rates = fit_state_dwell_models(dwells, selected_k)
    dwell_trans = build_dwell_transition_matrix(dwells, selected_k)
    step_states = np.rint(resample_series(labels.astype(float), steps)).astype(np.int32)
    codebook = gray_code_table(selected_k)
    state_biases = codebook[step_states]
    random_u = build_random_matrix(dataset.ai, steps, dims=codebook.shape[1] + 1)
    dt_step = float((dataset.time[-1] - dataset.time[0]) / max(steps - 1, 1))
    stability_rows, stability_summary = stability_analysis(dataset, labels, means, vars_)

    state_prob = np.bincount(labels, minlength=selected_k).astype(float)
    state_prob /= np.maximum(np.sum(state_prob), 1.0)
    metrics = {
        "selected_k": int(selected_k),
        "pbit_dims": int(codebook.shape[1]),
        "state_entropy_bits": shannon_entropy(state_prob),
        "state_counts": np.bincount(labels, minlength=selected_k).tolist(),
        "state_means": means.tolist(),
        "state_std": np.sqrt(vars_).tolist(),
        "sample_transition_count": int(np.sum(labels[1:] != labels[:-1])),
        "dwell_models": dwell_models,
        "transition_matrix": dwell_trans.tolist(),
        "mean_exit_rate": float(np.mean(exit_rates)),
        "stability": stability_summary,
    }

    return KStateProfile(
        name=dataset_name,
        selected_k=selected_k,
        state_labels=labels,
        state_means=means,
        state_vars=vars_,
        state_weights=weights,
        transition_matrix=dwell_trans,
        exit_rates=exit_rates,
        dwell_segments=dwells,
        step_states=step_states,
        state_biases=state_biases,
        random_u=random_u,
        dt_step=dt_step,
        k_selection_table=k_table,
        stability_rows=stability_rows,
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


def run_kstate_hazard_sa(
    profile: KStateProfile,
    J: np.ndarray,
    h: np.ndarray,
    optimum: float,
    run_idx: int,
    coupling_gain: float = 2.0,
    data_gain: float = 1.0,
) -> dict:
    n = len(h)
    dims = profile.state_biases.shape[1]
    energy_scale = compute_energy_scale(J, h)
    offset = (run_idx * 19) % len(profile.step_states)

    init_state = profile.step_states[offset]
    init_bias = profile.state_biases[offset]
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

    for step in range(len(profile.step_states)):
        state = int(profile.step_states[(offset + step) % len(profile.step_states)])
        base_rate = float(max(profile.exit_rates[state], 1e-6))
        state_bias = profile.state_biases[(offset + step) % len(profile.state_biases)]

        for dim in range(dims):
            spin_idx = (step * dims + dim + offset) % n
            local_field = float(J[spin_idx] @ spins + h[spin_idx])
            drive = data_gain * float(state_bias[dim])
            drive += coupling_gain * local_field / max(energy_scale, 1e-9)
            proposal_prob = logistic(2.0 * drive)
            proposal = 1 if profile.random_u[(offset + step) % len(profile.random_u), dim] < proposal_prob else -1

            if proposal != spins[spin_idx]:
                delta = delta_energy_for_flip(spins, J, h, spin_idx)
                rate = base_rate if delta <= 0.0 else base_rate * math.exp(-delta / max(energy_scale, 1e-9))
                accept_prob = 1.0 - math.exp(-rate * profile.dt_step)
                accept_rand = profile.random_u[(offset + step) % len(profile.random_u), -1]
                accept = accept_rand < accept_prob
            else:
                accept = False

            if step < len(profile.step_states) // 5:
                early_total += 1
                early_accept += int(accept)
            if step >= len(profile.step_states) * 4 // 5:
                late_total += 1
                late_accept += int(accept)

            if accept:
                spins[spin_idx] = proposal
                accepted += 1
                current += delta
                if current < best:
                    best = current

    return {
        "best_energy": float(best),
        "final_energy": float(current),
        "energy_gap_to_optimum": float(best - optimum),
        "hit_optimum": bool(abs(best - optimum) < 1e-9),
        "acceptance_rate": float(accepted / max(len(profile.step_states) * dims, 1)),
        "early_acceptance_rate": float(early_accept / max(early_total, 1)),
        "late_acceptance_rate": float(late_accept / max(late_total, 1)),
    }


def dwell_dataframe(profile: KStateProfile) -> pd.DataFrame:
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


def k_selection_dataframe(profile: KStateProfile) -> pd.DataFrame:
    rows = []
    for row in profile.k_selection_table:
        rows.append({"dataset": profile.name, **row})
    return pd.DataFrame(rows)


def stability_dataframe(profile: KStateProfile) -> pd.DataFrame:
    return pd.DataFrame(profile.stability_rows)


def save_outputs(profiles: list[KStateProfile], optimization_rows: list[dict], summary: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    pd.DataFrame(optimization_rows).to_csv(OUTPUT_DIR / "kstate_optimization_runs.csv", index=False)
    pd.concat([dwell_dataframe(profile) for profile in profiles], ignore_index=True).to_csv(
        OUTPUT_DIR / "kstate_dwell_segments.csv",
        index=False,
    )
    pd.concat([k_selection_dataframe(profile) for profile in profiles], ignore_index=True).to_csv(
        OUTPUT_DIR / "k_selection_table.csv",
        index=False,
    )
    pd.concat([stability_dataframe(profile) for profile in profiles], ignore_index=True).to_csv(
        OUTPUT_DIR / "state_stability_bins.csv",
        index=False,
    )
    with open(OUTPUT_DIR / "kstate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, profile in enumerate(profiles):
        axes[0, col].imshow(profile.transition_matrix, cmap="viridis", aspect="auto")
        axes[0, col].set_title(f"{profile.name} KxK transition")
        axes[0, col].set_xlabel("To state")
        axes[0, col].set_ylabel("From state")

        counts = np.bincount(profile.state_labels, minlength=profile.selected_k)
        axes[1, col].bar(np.arange(profile.selected_k), counts)
        axes[1, col].set_title(f"{profile.name} state occupancy")
        axes[1, col].set_xlabel("State")
        axes[1, col].set_ylabel("Count")
        axes[1, col].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kstate_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    profiles = [
        build_kstate_profile("good"),
        build_kstate_profile("bad"),
    ]
    J, h, optimum = build_ising_instance()

    optimization_rows = []
    model_summary = {}
    optimization_summary = {}

    for profile in profiles:
        runs = [
            run_kstate_hazard_sa(profile, J, h, optimum, run_idx=run_idx)
            for run_idx in range(40)
        ]
        model_summary[profile.name] = profile.metrics
        optimization_summary[profile.name] = summarize_runs(runs)

        for run_idx, result in enumerate(runs):
            optimization_rows.append(
                {
                    "dataset": profile.name,
                    "run_idx": run_idx,
                    **result,
                }
            )

    ranked = sorted(
        model_summary.keys(),
        key=lambda name: (
            optimization_summary[name]["mean_energy_gap"],
            -optimization_summary[name]["optimum_hit_rate"],
            -model_summary[name]["stability"]["stable_state_fraction"],
        ),
    )

    summary = {
        "assumption": (
            "Generalize the physical two-state hazard route to a K-state latent-state model: "
            "select K by amplitude-model BIC, segment with a sticky Gaussian HMM, extract dwell "
            "times, build a KxK transition matrix, and evaluate state stability across an external axis."
        ),
        "problem": {
            "type": "18-spin Ising minimization",
            "exact_optimum_energy": optimum,
            "runs_per_dataset": 40,
        },
        "kstate_model_summary": model_summary,
        "optimization_performance": optimization_summary,
        "overall_ranking": ranked,
    }

    save_outputs(profiles, optimization_rows, summary)

    print("MTJ physical K-state demo finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

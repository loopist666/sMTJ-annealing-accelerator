import json
import math
import random
from dataclasses import dataclass, asdict
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
OUTPUT_DIR = Path("mtj_sa_output")


@dataclass
class Dataset:
    name: str
    time: np.ndarray
    ai: np.ndarray
    av: np.ndarray


@dataclass
class FitParams:
    time_scale: float
    time_shift: float
    ai_scale: float
    ai_bias: float


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


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_error": float(np.mean(err)),
        "max_abs_error": float(np.max(np.abs(err))),
        "corr": corr,
    }


def av_summary(good: Dataset, bad: Dataset) -> dict:
    diff = bad.av - good.av.mean()
    return {
        "good_constant": float(np.mean(good.av)),
        "bad_constant": float(np.mean(bad.av)),
        "mean_difference": float(np.mean(diff)),
        "max_abs_difference": float(np.max(np.abs(diff))),
    }


def interpolate_to_good_time(good_time: np.ndarray, ref_time: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    return np.interp(good_time, ref_time, ref_y)


def raw_time_scaled_prediction(good: Dataset, bad: Dataset) -> tuple[np.ndarray, float]:
    naive_scale = (good.time.max() - good.time.min()) / (bad.time.max() - bad.time.min())
    scaled_time = bad.time * naive_scale
    pred = interpolate_to_good_time(good.time, scaled_time, bad.ai)
    return pred, naive_scale


def objective(params: np.ndarray, good: Dataset, bad: Dataset) -> float:
    time_scale, time_shift, ai_scale, ai_bias = params
    transformed_time = bad.time * time_scale + time_shift
    transformed_ai = bad.ai * ai_scale + ai_bias

    lower_gap = max(0.0, transformed_time.min() - good.time.min())
    upper_gap = max(0.0, good.time.max() - transformed_time.max())
    coverage_penalty = 1_000.0 * (lower_gap + upper_gap)

    pred = interpolate_to_good_time(good.time, transformed_time, transformed_ai)
    mse = np.mean((pred - good.ai) ** 2)
    return float(mse + coverage_penalty)


def simulated_annealing(good: Dataset, bad: Dataset, seed: int = 7) -> tuple[FitParams, float]:
    random.seed(seed)
    np.random.seed(seed)

    bounds = np.array(
        [
            [0.15, 0.40],   # time_scale
            [-5.0, 5.0],    # time_shift
            [-3.0, 3.0],    # ai_scale
            [-0.2, 0.2],    # ai_bias
        ],
        dtype=float,
    )

    step_sizes = np.array([0.004, 0.10, 0.04, 0.002], dtype=float)

    def random_point() -> np.ndarray:
        low = bounds[:, 0]
        high = bounds[:, 1]
        return low + np.random.rand(len(bounds)) * (high - low)

    current = random_point()
    current_score = objective(current, good, bad)
    best = current.copy()
    best_score = current_score

    temperature = 0.05
    min_temperature = 1e-6
    iterations = 40_000
    cooling = 0.9998

    for _ in range(iterations):
        candidate = current + np.random.normal(0.0, step_sizes)
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        candidate_score = objective(candidate, good, bad)

        accept = candidate_score < current_score
        if not accept:
            delta = current_score - candidate_score
            accept = random.random() < math.exp(delta / max(temperature, min_temperature))

        if accept:
            current = candidate
            current_score = candidate_score
            if candidate_score < best_score:
                best = candidate.copy()
                best_score = candidate_score

        temperature = max(temperature * cooling, min_temperature)

    return (
        FitParams(
            time_scale=float(best[0]),
            time_shift=float(best[1]),
            ai_scale=float(best[2]),
            ai_bias=float(best[3]),
        ),
        float(best_score),
    )


def fitted_prediction(good: Dataset, bad: Dataset, params: FitParams) -> np.ndarray:
    transformed_time = bad.time * params.time_scale + params.time_shift
    transformed_ai = bad.ai * params.ai_scale + params.ai_bias
    return interpolate_to_good_time(good.time, transformed_time, transformed_ai)


def save_outputs(good: Dataset, raw_pred: np.ndarray, fitted_pred: np.ndarray, metrics: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    compare_df = pd.DataFrame(
        {
            "time": good.time,
            "good_ai": good.ai,
            "raw_bad_ai_interp": raw_pred,
            "fitted_bad_ai_interp": fitted_pred,
            "raw_residual": raw_pred - good.ai,
            "fitted_residual": fitted_pred - good.ai,
        }
    )
    compare_df.to_csv(OUTPUT_DIR / "aligned_comparison.csv", index=False)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if plt is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(good.time, good.ai, label="good AI", linewidth=1.6)
    axes[0].plot(good.time, raw_pred, label="raw aligned bad AI", linewidth=1.1, alpha=0.8)
    axes[0].plot(good.time, fitted_pred, label="SA fitted bad AI", linewidth=1.1, alpha=0.9)
    axes[0].set_ylabel("AI")
    axes[0].set_title("MTJ AI comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(good.time, raw_pred - good.ai, label="raw residual", linewidth=1.1)
    axes[1].plot(good.time, fitted_pred - good.ai, label="SA residual", linewidth=1.1)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Residual")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_plot.png", dpi=180)
    plt.close(fig)


def main() -> None:
    good = load_dataset(GOOD_FILE, "good")
    bad = load_dataset(BAD_FILE, "bad")

    raw_pred, naive_scale = raw_time_scaled_prediction(good, bad)
    sa_params, sa_objective = simulated_annealing(good, bad)
    fitted_pred = fitted_prediction(good, bad, sa_params)

    metrics = {
        "assumption": "将两组 MTJ 数据视为 Time/AI/AV 序列，使用模拟退火寻找 bad 数据映射到 good 数据的最佳时间与幅值变换。",
        "input_files": {
            "good_file": GOOD_FILE,
            "bad_file": BAD_FILE,
        },
        "sample_sizes": {
            "good_rows": int(len(good.time)),
            "bad_rows": int(len(bad.time)),
        },
        "raw_alignment": {
            "naive_time_scale": float(naive_scale),
            "ai_metrics": metric_summary(good.ai, raw_pred),
        },
        "simulated_annealing_fit": {
            "params": asdict(sa_params),
            "objective": sa_objective,
            "ai_metrics": metric_summary(good.ai, fitted_pred),
        },
        "av_difference": av_summary(good, bad),
    }

    save_outputs(good, raw_pred, fitted_pred, metrics)

    print("MTJ simulated annealing demo finished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()

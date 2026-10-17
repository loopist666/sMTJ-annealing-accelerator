"""Paired algorithm experiments. Replayed traces are not hardware repetitions."""
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np

from .measurements import characterize, prepare_replay
from .model import benchmark_instances, exact, cut_weight, denoising_problem, ferromagnetic_minimum
from .solver import MODES, solve


def write_json(path, value):
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o).__name__)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_profiles(records, tau_s=2):
    """A priori rule: all fixed -1 V traces except the explicitly named dead trace."""
    return [prepare_replay(r, tau_s) for r in records if r.time is not None
            and np.allclose(r.voltage, -1, rtol=0, atol=1e-5)
            and "dead" not in r.sheet.lower()]


def run_benchmark(profiles, steps=2048, runs=32, gain=1.5, seed=700):
    if not profiles:
        raise ValueError("No fixed -1 V replay traces found")
    if runs < 1 or steps < 2 or steps > min(len(p["test_z"]) for p in profiles):
        raise ValueError("Invalid run/step budget; replay never wraps, repeats, or resamples")
    instances = benchmark_instances()
    references = {p.name: exact(p) for p in instances}
    rows, examples, mean_paths = [], {}, {}
    clock_start = time.perf_counter()
    for problem in instances:
        optimum, opt_spins = references[problem.name]
        histories = {mode: [] for mode in MODES}
        for pi, profile in enumerate(profiles):
            z = profile["test_z"]
            for run in range(runs):
                # Same source segment, initial state and site proposals for all methods.
                trial_seed = seed+run+pi*10000
                offset = int(np.random.default_rng(trial_seed+9001).integers(len(z)-steps+1))
                segment = z[offset:offset+steps]
                for mode in MODES:
                    start = time.perf_counter()
                    sol = solve(problem, segment, mode, trial_seed, gain, optimum)
                    elapsed = time.perf_counter()-start
                    hit = sol.best_energy <= optimum+1e-8
                    rows.append(dict(instance=problem.name, trace=profile["key"], run=run, seed=trial_seed,
                                     offset_in_test=offset, mode=mode, steps=steps, gain=gain,
                                     best_energy=sol.best_energy, exact_energy=optimum,
                                     gap=max(0., sol.best_energy-optimum), hit=int(hit),
                                     first_hit=sol.first_hit, accepted=sol.accepted,
                                     samples_consumed=sol.samples_consumed, python_wall_s=elapsed,
                                     replay_acquisition_span_s=float(profile["test_time"][offset+steps-1]-profile["test_time"][offset]) if sol.samples_consumed else 0.,
                                     replay_VI_energy_J=float(np.trapezoid(profile["test_power_W"][offset:offset+steps], profile["test_time"][offset:offset+steps])) if sol.samples_consumed else 0.))
                    histories[mode].append(sol.best_history)
                    if pi == 0 and run == 0:
                        examples[f"{problem.name}/{mode}"] = dict(spins=sol.spins, energy=sol.best_energy,
                            gap=sol.best_energy-optimum, cut=cut_weight(problem, sol.spins) if problem.name.startswith("maxcut") else None,
                            history=sol.best_history, current_history=sol.energy_history, thresholds=sol.thresholds)
        mean_paths[problem.name] = {mode: np.mean(v, axis=0) for mode, v in histories.items()}
    summaries, trace_summaries = [], []
    for problem in instances:
        for mode in MODES:
            selected = [r for r in rows if r["instance"] == problem.name and r["mode"] == mode]
            summaries.append(dict(instance=problem.name, mode=mode, n_runs=len(selected),
                hit_rate=float(np.mean([r["hit"] for r in selected])),
                mean_gap=float(np.mean([r["gap"] for r in selected])),
                median_gap=float(np.median([r["gap"] for r in selected])),
                mean_python_wall_ms=float(np.mean([r["python_wall_s"] for r in selected])*1000)))
            for profile in profiles:
                group = [r for r in selected if r["trace"] == profile["key"]]
                trace_summaries.append(dict(instance=problem.name, mode=mode, trace=profile["key"],
                    hit_rate=float(np.mean([r["hit"] for r in group])), mean_gap=float(np.mean([r["gap"] for r in group]))))
    # Paired bootstrap over software seeds only; no claim of hardware-replicate CI.
    comparisons = []
    by_key = {(r["instance"],r["trace"],r["run"],r["mode"]): r for r in rows}
    rng = np.random.default_rng(1990)
    for other in ("shuffled", "gaussian", "constant", "sa", "greedy"):
        differences = []
        for run in range(runs):
            differences.append(np.mean([by_key[(p.name, profile["key"], run, "replay")]["gap"]-
                by_key[(p.name, profile["key"], run, other)]["gap"] for p in instances for profile in profiles]))
        means = np.mean(rng.choice(differences, (2000,runs), replace=True), axis=1)
        comparisons.append(dict(comparison=f"replay minus {other}", metric="mean energy gap (negative favors replay)",
                                estimate=float(np.mean(differences)),
                                software_seed_ci95_low=float(np.quantile(means,.025)),
                                software_seed_ci95_high=float(np.quantile(means,.975))))
    payload = dict(steps=steps, runs_per_trace=runs, gain=gain, seed=seed, profiles=len(profiles),
        filter_tau_s=profiles[0]["filter_tau_s"], train_fraction=.4,
        python=platform.python_version(), numpy=np.__version__, elapsed_s=time.perf_counter()-clock_start,
        statement="Finite recorded-signal software replay. Repeated runs reuse measured records; not independent hardware trials. No hardware acceleration claim.",
        summary=summaries, by_trace=trace_summaries, paired_comparisons=comparisons,
        instances=[dict(name=p.name,J=p.J,h=p.h,exact_energy=references[p.name][0],exact_spins=references[p.name][1]) for p in instances],
        examples=examples, mean_paths=mean_paths)
    return rows, payload


def image_demo(profile, steps=2048, gain=1.5):
    problem, clean, noisy = denoising_problem()
    optimum, exact_spins = ferromagnetic_minimum(problem)
    outputs = {}
    for mode in MODES:
        sol = solve(problem, profile["test_z"][:steps], mode, 123, gain, optimum, initial=noisy.ravel())
        outputs[mode] = dict(image=sol.spins.reshape(clean.shape), energy=sol.best_energy,
                             gap=sol.best_energy-optimum, pixel_accuracy=float(np.mean(sol.spins.reshape(clean.shape)==clean)))
    return dict(clean=clean, noisy=noisy, input_accuracy=float(np.mean(clean==noisy)),
                exact_image=exact_spins.reshape(clean.shape), exact_energy=optimum,
                exact_pixel_accuracy=float(np.mean(exact_spins.reshape(clean.shape)==clean)), methods=outputs,
                note="Synthetic image; exact graph-cut reference. Optimizing MAP energy need not maximize clean-image accuracy.")


def characterize_all(records, out):
    rows = [characterize(r) for r in records]
    write_csv(out/"measurement_summary.csv", rows)
    write_json(out/"measurement_summary.json", rows)
    manifests = [{"source":r.source,"sheet":r.sheet,"sha256":r.sha256,"metadata":r.metadata} for r in records]
    write_json(out/"input_manifest.json", manifests)
    sensitivity = []
    for r in records:
        if r.time is not None:
            for tau in (.5, 2., 10.):
                c = characterize(r, tau)
                sensitivity.append({k:c[k] for k in ("source","sheet","filter_tau_s","residual_std_uA","residual_lag1","tau_effective_s","acf_truncated")})
    write_csv(out/"filter_sensitivity.csv", sensitivity)
    return rows

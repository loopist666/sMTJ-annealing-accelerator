"""Threshold accepting with a sequential scalar disturbance, plus software controls."""
from dataclasses import dataclass
import numpy as np

MODES = ("greedy", "constant", "gaussian", "replay", "shuffled", "sa")


@dataclass
class Solution:
    spins: np.ndarray
    best_energy: float
    final_energy: float
    best_history: np.ndarray
    energy_history: np.ndarray
    thresholds: np.ndarray
    accepted: int
    first_hit: int | None
    samples_consumed: int


def solve(problem, signal, mode="replay", seed=0, gain=1.5, target=None, initial=None):
    """One proposed spin flip per signal sample, no recycling or hidden polish.

    Replay and shuffled differ only in sample order. Software RNG supplies initial
    state and site order. Gaussian/constant controls use test-segment RMS/mean
    respectively (OFFLINE matched controls, not online calibration).
    """
    z = np.asarray(signal, float)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("signal must be a finite 1D sequence of at least 2 samples")
    if mode not in MODES or gain < 0 or not np.isfinite(gain):
        raise ValueError("Invalid mode or gain")
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], problem.n) if initial is None else np.asarray(initial, int).copy()
    current = problem.energy(spins)
    sites = rng.integers(problem.n, size=len(z))
    noise_rng = np.random.default_rng(seed+1000003)
    scale = float(np.median(np.sqrt(np.sum(problem.J**2, axis=1)+problem.h**2)))
    scale = max(scale, np.finfo(float).eps)
    amplitude = gain*2*scale*np.linspace(1., 0., len(z))**2
    if mode == "replay":
        magnitude = np.abs(z)
    elif mode == "shuffled":
        magnitude = np.abs(noise_rng.permutation(z))
    elif mode == "gaussian":
        noise = noise_rng.normal(size=len(z))
        magnitude = np.abs(noise)*np.sqrt(np.mean(z*z)/np.mean(noise*noise))
    elif mode == "constant":
        magnitude = np.full(len(z), np.mean(np.abs(z)))
    elif mode == "sa":
        # P(Exp(1) >= dE/T) = exp(-dE/T), ordinary Metropolis acceptance.
        magnitude = noise_rng.exponential(size=len(z))
    else:
        magnitude = np.zeros(len(z))
    thresholds = amplitude*magnitude
    best, best_s = current, spins.copy()
    best_history, history = np.empty(len(z)+1), np.empty(len(z)+1)
    best_history[0] = history[0] = current
    first_hit = 0 if target is not None and current <= target+1e-8 else None
    accepted = 0
    fields = problem.J@spins+problem.h
    for k, i in enumerate(sites):
        delta = 2*spins[i]*fields[i]
        if delta <= thresholds[k]:
            old = spins[i]
            spins[i] = -old
            fields -= 2*old*problem.J[:, i]
            current += delta
            accepted += 1
            if current < best-1e-10:
                best, best_s = current, spins.copy()
        if target is not None and first_hit is None and best <= target+1e-8:
            first_hit = k+1
        best_history[k+1], history[k+1] = best, current
    # Recompute the reported energies, bounding incremental numerical drift.
    best, current = problem.energy(best_s), problem.energy(spins)
    return Solution(best_s, best, current, best_history, history, thresholds, accepted,
                    first_hit, len(z) if mode in ("replay", "shuffled") else 0)

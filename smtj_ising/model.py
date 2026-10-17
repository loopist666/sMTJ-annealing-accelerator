"""One sign convention: E(s) = -s.T J s / 2 - h.T s, J symmetric, diag=0."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Ising:
    J: np.ndarray
    h: np.ndarray
    name: str = "custom"

    def __post_init__(self):
        self.J, self.h = np.array(self.J, float, copy=True), np.array(self.h, float, copy=True)
        if self.h.ndim != 1:
            raise ValueError("h must be a vector")
        n = len(self.h)
        if n < 1 or self.J.shape != (n, n):
            raise ValueError("J must be square and match h")
        if not np.isfinite(self.J).all() or not np.isfinite(self.h).all():
            raise ValueError("Non-finite J or h")
        if not np.allclose(self.J, self.J.T, rtol=0, atol=1e-12) or not np.allclose(np.diag(self.J), 0, rtol=0, atol=1e-12):
            raise ValueError("J must be symmetric with zero diagonal")
        self.J = (self.J+self.J.T)/2
        np.fill_diagonal(self.J, 0)

    @property
    def n(self):
        return len(self.h)

    def energy(self, spins):
        s = np.asarray(spins)
        if s.shape != self.h.shape or not np.isin(s, [-1, 1]).all():
            raise ValueError("Spins must be an n-vector of -1/+1")
        return float(-.5*s@self.J@s-self.h@s)

    def delta(self, spins, index):
        return float(2*spins[index]*(self.J[index]@spins+self.h[index]))


def maxcut(weights, name="maxcut"):
    w = np.asarray(weights, float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or np.any(w < 0):
        raise ValueError("Max-Cut requires a nonnegative square weight matrix")
    return Ising(-w, np.zeros(len(w)), name)


def cut_weight(problem, spins):
    w = -problem.J
    if np.any(w < 0) or np.any(problem.h):
        raise ValueError("Problem is not a Max-Cut encoding")
    return float(np.sum(np.triu(w*(1-np.outer(spins, spins))/2, 1)))


def qubo(Q, name="qubo"):
    """Full-matrix convention x.T Q x; return Ising and additive QUBO offset."""
    Q = np.asarray(Q, float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not np.isfinite(Q).all():
        raise ValueError("Q must be a finite square matrix")
    Q = (Q+Q.T)/2
    J = -Q/2
    np.fill_diagonal(J, 0)
    h = -Q.sum(axis=1)/2
    offset = (Q.sum()+np.trace(Q))/4
    return Ising(J, h, name), float(offset)


def exact(problem):
    if problem.n > 20:
        raise ValueError("Exact enumeration limited to 20 spins")
    best, best_s = float("inf"), None
    for start in range(0, 1 << problem.n, 4096):
        ids = np.arange(start, min(start+4096, 1 << problem.n), dtype=np.uint64)
        s = 2*((ids[:, None] >> np.arange(problem.n, dtype=np.uint64)) & 1).astype(float)-1
        energies = -.5*np.einsum("bi,ij,bj->b", s, problem.J, s)-s@problem.h
        idx = int(np.argmin(energies))
        if energies[idx] < best:
            best, best_s = float(energies[idx]), s[idx].astype(int)
    return best, best_s


def benchmark_instances():
    result = []
    for seed, density in [(101, .3), (202, .65), (303, 1.)]:
        rng = np.random.default_rng(seed)
        a = np.triu((rng.random((16,16)) < density)*rng.integers(1, 6, (16,16)), 1)
        result.append(maxcut(a+a.T, f"maxcut16_{seed}"))
    rng = np.random.default_rng(404)
    a = np.triu(rng.choice([-2., -1., 1., 2.], (16,16)), 1)
    result.append(Ising(a+a.T, rng.choice([-.5, .5], 16), "spin_glass16_404"))
    return result


def denoising_problem(size=12, seed=51):
    """Synthetic binary-image MAP demo; no device image is segmented."""
    y, x = np.mgrid[:size, :size]
    clean = np.where(((x-size/2)**2+(y-size/2)**2 < (size*.31)**2), 1, -1)
    rng = np.random.default_rng(seed)
    noisy = clean * np.where(rng.random(clean.shape) < .16, -1, 1)
    J = np.zeros((size*size, size*size))
    for r in range(size):
        for c in range(size):
            i = r*size+c
            for rr, cc in [(r+1,c), (r,c+1)]:
                if rr < size and cc < size:
                    j = rr*size+cc
                    J[i,j] = J[j,i] = .75
    return Ising(J, noisy.ravel().astype(float), "binary_image12"), clean, noisy


def ferromagnetic_minimum(problem):
    """Exact s-t cut via Edmonds-Karp; only nonnegative pair couplings."""
    from collections import deque
    if np.any(problem.J < 0):
        raise ValueError("Graph-cut reference requires ferromagnetic J >= 0")
    n = problem.n
    source, sink = n, n+1
    residual = np.zeros((n+2, n+2))
    residual[:n, :n] = 2*problem.J
    residual[source, :n] = 2*np.maximum(problem.h, 0)
    residual[:n, sink] = 2*np.maximum(-problem.h, 0)
    while True:
        parent = np.full(n+2, -1, dtype=int)
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] < 0:
            u = queue.popleft()
            for v in np.flatnonzero(residual[u] > 1e-12):
                if parent[v] < 0:
                    parent[v] = u
                    queue.append(v)
        if parent[sink] < 0:
            spins = np.where(parent[:n] >= 0, 1, -1)
            return problem.energy(spins), spins
        v, amount = sink, float("inf")
        while v != source:
            u = parent[v]
            amount = min(amount, residual[u,v])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[u,v] -= amount
            residual[v,u] += amount
            v = u

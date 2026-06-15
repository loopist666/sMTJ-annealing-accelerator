import itertools
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

from mtj_hardware_sa_demo import ising_energy, summarize_runs
from mtj_physical_kstate_demo import KStateProfile, build_kstate_profile, compute_energy_scale


OUTPUT_DIR = Path("mtj_ising_application_demos_output")

TSP_COORDS_24 = np.array(
    [
        [0.5, 1.0],
        [2.4, 3.7],
        [4.3, 1.1],
        [6.2, 4.2],
        [7.3, 1.4],
        [3.9, 5.5],
        [1.2, 5.8],
        [6.9, 6.1],
        [8.3, 3.6],
        [9.1, 6.8],
        [10.6, 1.2],
        [11.4, 4.6],
        [12.8, 7.3],
        [14.0, 2.1],
        [15.3, 5.8],
        [16.7, 1.4],
        [17.5, 7.0],
        [18.8, 3.3],
        [20.1, 6.1],
        [21.4, 1.0],
        [22.2, 4.5],
        [23.6, 7.5],
        [24.8, 2.6],
        [26.0, 5.9],
    ],
    dtype=float,
)


@dataclass
class QuboModel:
    linear: np.ndarray
    quadratic: np.ndarray
    offset: float = 0.0


@dataclass
class IsingProblem:
    name: str
    J: np.ndarray
    h: np.ndarray
    offset: float
    metadata: dict


@dataclass
class IsingRun:
    problem: str
    run_idx: int
    best_spins: np.ndarray
    final_spins: np.ndarray
    best_energy: float
    final_energy: float
    best_objective: float
    final_objective: float
    acceptance_rate: float
    early_acceptance_rate: float
    late_acceptance_rate: float
    history: np.ndarray
    current_history: np.ndarray


def add_linear(model: QuboModel, idx: int, value: float) -> None:
    model.linear[idx] += value


def add_quadratic(model: QuboModel, i: int, j: int, value: float) -> None:
    if i == j:
        model.linear[i] += value
        return
    a, b = sorted((i, j))
    model.quadratic[a, b] += value


def add_exactly_one(model: QuboModel, indices: list[int], penalty: float) -> None:
    model.offset += penalty
    for idx in indices:
        add_linear(model, idx, -penalty)
    for i, j in itertools.combinations(indices, 2):
        add_quadratic(model, i, j, 2.0 * penalty)


def add_cardinality(model: QuboModel, indices: list[int], target: int, penalty: float) -> None:
    model.offset += penalty * target * target
    for idx in indices:
        add_linear(model, idx, penalty * (1.0 - 2.0 * target))
    for i, j in itertools.combinations(indices, 2):
        add_quadratic(model, i, j, 2.0 * penalty)


def qubo_to_ising(model: QuboModel) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(model.linear)
    J = np.zeros((n, n), dtype=float)
    h = np.zeros(n, dtype=float)
    offset = float(model.offset)

    for idx, coeff in enumerate(model.linear):
        offset += 0.5 * coeff
        h[idx] -= 0.5 * coeff

    for i in range(n):
        for j in range(i + 1, n):
            coeff = model.quadratic[i, j]
            if coeff == 0.0:
                continue
            offset += 0.25 * coeff
            h[i] -= 0.25 * coeff
            h[j] -= 0.25 * coeff
            J[i, j] -= 0.25 * coeff
            J[j, i] = J[i, j]

    return J, h, offset


def binary_from_spins(spins: np.ndarray) -> np.ndarray:
    return ((spins + 1) // 2).astype(np.int8)


def qubo_objective_from_spins(spins: np.ndarray, model: QuboModel) -> float:
    x = binary_from_spins(spins).astype(float)
    return float(model.offset + model.linear @ x + np.sum(model.quadratic * np.outer(x, x)))


def safe_logistic(x: float) -> float:
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def run_kstate_ising_solver(
    profile: KStateProfile,
    problem: IsingProblem,
    run_idx: int,
    cycles: int = 30,
    coupling_gain: float = 3.0,
    data_gain: float = 0.8,
    hazard_gain: float = 8.0,
) -> IsingRun:
    J = problem.J
    h = problem.h
    n = len(h)
    dims = profile.state_biases.shape[1]
    profile_steps = len(profile.step_states)
    total_steps = profile_steps * cycles
    energy_scale = compute_energy_scale(J, h)
    offset_idx = (run_idx * 53) % profile_steps

    spins = np.empty(n, dtype=np.int8)
    for idx in range(n):
        state_row = (offset_idx + idx * 7) % profile_steps
        bias = profile.state_biases[state_row, idx % dims]
        rand = profile.random_u[state_row, idx % profile.random_u.shape[1]]
        spins[idx] = 1 if bias + 0.75 * (rand - 0.5) >= 0.0 else -1

    current = ising_energy(spins, J, h) + problem.offset
    local_fields = J @ spins + h
    best = current
    best_spins = spins.copy()
    accepted = 0
    early_accept = 0
    late_accept = 0
    early_total = 0
    late_total = 0
    history = np.empty(total_steps, dtype=float)
    current_history = np.empty(total_steps, dtype=float)

    for step in range(total_steps):
        profile_idx = (offset_idx + step) % profile_steps
        state = int(profile.step_states[profile_idx])
        base_rate = float(max(profile.exit_rates[state], 1e-9))
        state_bias = profile.state_biases[profile_idx]

        for dim in range(dims):
            spin_idx = (step * dims + dim + offset_idx) % n
            local_field = float(local_fields[spin_idx])
            drive = data_gain * float(state_bias[dim])
            drive += coupling_gain * local_field / max(energy_scale, 1e-12)
            proposal_prob = safe_logistic(2.0 * drive)
            proposal_rand = profile.random_u[profile_idx, dim % profile.random_u.shape[1]]
            proposal = 1 if proposal_rand < proposal_prob else -1

            if proposal == spins[spin_idx]:
                accept = False
                delta = 0.0
            else:
                old_spin = int(spins[spin_idx])
                delta = float(2.0 * old_spin * local_field)
                if delta <= 0.0:
                    accept = True
                else:
                    thermal = math.exp(-delta / max(energy_scale, 1e-12))
                    accept_prob = 1.0 - math.exp(-hazard_gain * base_rate * profile.dt_step * thermal)
                    accept_rand = profile.random_u[profile_idx, -1]
                    accept = accept_rand < accept_prob

            if step < total_steps // 5:
                early_total += 1
                early_accept += int(accept)
            if step >= total_steps * 4 // 5:
                late_total += 1
                late_accept += int(accept)

            if accept:
                old_spin = int(spins[spin_idx])
                spins[spin_idx] = proposal
                local_fields += float(proposal - old_spin) * J[:, spin_idx]
                current += delta
                accepted += 1
                if current < best:
                    best = current
                    best_spins = spins.copy()

        history[step] = best
        current_history[step] = current

    return IsingRun(
        problem=problem.name,
        run_idx=run_idx,
        best_spins=best_spins,
        final_spins=spins.copy(),
        best_energy=float(best),
        final_energy=float(current),
        best_objective=float(best),
        final_objective=float(current),
        acceptance_rate=float(accepted / max(total_steps * dims, 1)),
        early_acceptance_rate=float(early_accept / max(early_total, 1)),
        late_acceptance_rate=float(late_accept / max(late_total, 1)),
        history=history,
        current_history=current_history,
    )


def tsp_var(city: int, position: int, n_cities: int) -> int:
    return city * n_cities + position


def build_tsp_problem(n_cities: int = 8) -> tuple[IsingProblem, QuboModel]:
    if n_cities > len(TSP_COORDS_24):
        raise ValueError(f"n_cities must be <= {len(TSP_COORDS_24)}.")
    coords = TSP_COORDS_24[:n_cities].copy()
    n_cities = len(coords)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    n_vars = n_cities * n_cities
    model = QuboModel(np.zeros(n_vars, dtype=float), np.zeros((n_vars, n_vars), dtype=float))
    penalty = float(10.0 * n_cities * np.max(distances))

    for position in range(n_cities):
        add_exactly_one(model, [tsp_var(city, position, n_cities) for city in range(n_cities)], penalty)
    for city in range(n_cities):
        add_exactly_one(model, [tsp_var(city, position, n_cities) for position in range(n_cities)], penalty)

    # Fix the starting city to remove rotational degeneracy.
    model.offset += penalty
    add_linear(model, tsp_var(0, 0, n_cities), -penalty)

    for position in range(n_cities):
        next_position = (position + 1) % n_cities
        for city_i in range(n_cities):
            for city_j in range(n_cities):
                add_quadratic(
                    model,
                    tsp_var(city_i, position, n_cities),
                    tsp_var(city_j, next_position, n_cities),
                    distances[city_i, city_j],
                )

    J, h, offset = qubo_to_ising(model)
    problem = IsingProblem(
        name=f"tsp_{n_cities}",
        J=J,
        h=h,
        offset=offset,
        metadata={
            "coords": coords.tolist(),
            "distances": distances.tolist(),
            "penalty": penalty,
            "n_cities": n_cities,
        },
    )
    return problem, model


def decode_tsp(spins: np.ndarray, metadata: dict) -> dict:
    n_cities = int(metadata["n_cities"])
    distances = np.array(metadata["distances"], dtype=float)
    x = binary_from_spins(spins).reshape(n_cities, n_cities)
    row_sums = x.sum(axis=1)
    col_sums = x.sum(axis=0)
    feasible = bool(np.all(row_sums == 1) and np.all(col_sums == 1) and x[0, 0] == 1)
    onehot_violation = int(np.sum(np.abs(row_sums - 1)) + np.sum(np.abs(col_sums - 1)) + abs(int(x[0, 0]) - 1))

    route = [-1] * n_cities
    used = set()
    forced = int(np.argmax(x[:, 0]))
    if x[0, 0] == 1:
        forced = 0
    route[0] = forced
    used.add(forced)

    choices = []
    for position in range(1, n_cities):
        for city in range(n_cities):
            choices.append((x[city, position], position, city))
    for _, position, city in sorted(choices, reverse=True):
        if route[position] == -1 and city not in used:
            route[position] = city
            used.add(city)
    for position in range(n_cities):
        if route[position] == -1:
            for city in range(n_cities):
                if city not in used:
                    route[position] = city
                    used.add(city)
                    break

    length = route_length(route, distances)
    refined_route, refined_length = two_opt_route(route, distances)
    return {
        "route": route,
        "route_length": float(length),
        "two_opt_route": refined_route,
        "two_opt_route_length": float(refined_length),
        "feasible_raw_assignment": feasible,
        "onehot_violation": onehot_violation,
        "row_sums": row_sums.astype(int).tolist(),
        "col_sums": col_sums.astype(int).tolist(),
    }


def route_length(route: list[int], distances: np.ndarray) -> float:
    return float(sum(distances[route[i], route[(i + 1) % len(route)]] for i in range(len(route))))


def two_opt_route(route: list[int], distances: np.ndarray) -> tuple[list[int], float]:
    best_route = route.copy()
    best_length = route_length(best_route, distances)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route)):
                if j - i == 1:
                    continue
                candidate = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                candidate_length = route_length(candidate, distances)
                if candidate_length + 1e-12 < best_length:
                    best_route = candidate
                    best_length = candidate_length
                    improved = True

    return best_route, best_length


def exact_tsp(metadata: dict) -> dict:
    n_cities = int(metadata["n_cities"])
    distances = np.array(metadata["distances"], dtype=float)
    best_route = None
    best_length = float("inf")
    for tail in itertools.permutations(range(1, n_cities)):
        route = [0, *tail]
        length = route_length(route, distances)
        if length < best_length:
            best_length = length
            best_route = route
    return {"route": best_route, "route_length": float(best_length)}


def nearest_neighbor_route(distances: np.ndarray, start: int) -> list[int]:
    n_cities = len(distances)
    route = [start]
    unused = set(range(n_cities))
    unused.remove(start)
    while unused:
        last = route[-1]
        nxt = min(unused, key=lambda city: distances[last, city])
        route.append(nxt)
        unused.remove(nxt)
    if start != 0:
        zero_idx = route.index(0)
        route = route[zero_idx:] + route[:zero_idx]
    return route


def reference_tsp(metadata: dict, exact_limit: int = 9) -> dict:
    n_cities = int(metadata["n_cities"])
    distances = np.array(metadata["distances"], dtype=float)
    if n_cities <= exact_limit:
        ref = exact_tsp(metadata)
        ref["method"] = "exact"
        return ref

    best_route = None
    best_length = float("inf")
    for start in range(n_cities):
        route = nearest_neighbor_route(distances, start)
        route, length = two_opt_route(route, distances)
        if length < best_length:
            best_route = route
            best_length = length
    return {
        "route": best_route,
        "route_length": float(best_length),
        "method": "nearest-neighbor all-start + 2-opt",
    }


def build_maxcut_problem(n_nodes: int = 10) -> tuple[IsingProblem, QuboModel]:
    if n_nodes == 10:
        edges = [
            (0, 1, 3.0),
            (0, 3, 2.0),
            (0, 7, 4.0),
            (1, 2, 5.0),
            (1, 4, 3.0),
            (2, 5, 4.0),
            (2, 8, 2.0),
            (3, 4, 6.0),
            (3, 6, 2.0),
            (4, 5, 1.0),
            (4, 7, 4.0),
            (5, 9, 5.0),
            (6, 7, 3.0),
            (6, 9, 4.0),
            (7, 8, 2.0),
            (8, 9, 3.0),
        ]
    else:
        rng = np.random.default_rng(4100 + n_nodes)
        edge_map = {}
        for node in range(n_nodes):
            edge_map[tuple(sorted((node, (node + 1) % n_nodes)))] = float(rng.integers(2, 8))
            edge_map[tuple(sorted((node, (node + 3) % n_nodes)))] = float(rng.integers(1, 7))
        for i, j in itertools.combinations(range(n_nodes), 2):
            if (i, j) not in edge_map and rng.random() < min(0.24, 5.0 / n_nodes):
                edge_map[(i, j)] = float(rng.integers(1, 8))
        edges = [(i, j, weight) for (i, j), weight in sorted(edge_map.items())]

    model = QuboModel(np.zeros(n_nodes, dtype=float), np.zeros((n_nodes, n_nodes), dtype=float))
    for i, j, weight in edges:
        add_linear(model, i, -weight)
        add_linear(model, j, -weight)
        add_quadratic(model, i, j, 2.0 * weight)
    J, h, offset = qubo_to_ising(model)
    problem = IsingProblem(
        name=f"maxcut_{n_nodes}",
        J=J,
        h=h,
        offset=offset,
        metadata={"n_nodes": n_nodes, "edges": edges},
    )
    return problem, model


def decode_maxcut(spins: np.ndarray, metadata: dict) -> dict:
    x = binary_from_spins(spins)
    cut = 0.0
    for i, j, weight in metadata["edges"]:
        if x[i] != x[j]:
            cut += weight
    return {
        "partition_0": [int(i) for i, bit in enumerate(x) if bit == 0],
        "partition_1": [int(i) for i, bit in enumerate(x) if bit == 1],
        "cut_weight": float(cut),
    }


def exact_maxcut(metadata: dict) -> dict:
    n_nodes = int(metadata["n_nodes"])
    best_spins = None
    best_cut = -float("inf")
    for bits in range(1 << n_nodes):
        x = np.array([(bits >> i) & 1 for i in range(n_nodes)], dtype=np.int8)
        cut = 0.0
        for i, j, weight in metadata["edges"]:
            if x[i] != x[j]:
                cut += weight
        if cut > best_cut:
            best_cut = cut
            best_spins = x * 2 - 1
    decoded = decode_maxcut(best_spins, metadata)
    decoded["spins"] = best_spins.astype(int).tolist()
    return decoded


def maxcut_local_search(metadata: dict, starts: int = 160) -> dict:
    n_nodes = int(metadata["n_nodes"])
    edges = metadata["edges"]
    rng = np.random.default_rng(7200 + n_nodes)
    best_x = None
    best_cut = -float("inf")

    for start in range(starts):
        if start == 0:
            x = np.arange(n_nodes, dtype=np.int8) % 2
        else:
            x = rng.integers(0, 2, size=n_nodes, dtype=np.int8)

        improved = True
        while improved:
            improved = False
            for node in rng.permutation(n_nodes):
                delta = 0.0
                for i, j, weight in edges:
                    if i == node or j == node:
                        other = j if i == node else i
                        delta += weight if x[node] == x[other] else -weight
                if delta > 1e-12:
                    x[node] = 1 - x[node]
                    improved = True

        cut = 0.0
        for i, j, weight in edges:
            if x[i] != x[j]:
                cut += weight
        if cut > best_cut:
            best_cut = cut
            best_x = x.copy()

    spins = best_x * 2 - 1
    decoded = decode_maxcut(spins.astype(np.int8), metadata)
    decoded["spins"] = spins.astype(int).tolist()
    decoded["method"] = "multi-start 1-flip local search"
    return decoded


def reference_maxcut(metadata: dict, exact_limit: int = 18) -> dict:
    if int(metadata["n_nodes"]) <= exact_limit:
        ref = exact_maxcut(metadata)
        ref["method"] = "exact"
        return ref
    return maxcut_local_search(metadata)


def build_portfolio_problem(n_assets: int = 8) -> tuple[IsingProblem, QuboModel]:
    if n_assets == 8:
        returns = np.array([0.11, 0.06, 0.14, 0.09, 0.18, 0.07, 0.12, 0.16], dtype=float)
        volatility = np.array([0.20, 0.11, 0.25, 0.16, 0.31, 0.14, 0.22, 0.27], dtype=float)
        corr = np.array(
            [
                [1.00, 0.22, 0.40, 0.15, 0.55, 0.18, 0.35, 0.48],
                [0.22, 1.00, 0.24, 0.45, 0.20, 0.38, 0.28, 0.25],
                [0.40, 0.24, 1.00, 0.30, 0.50, 0.22, 0.46, 0.42],
                [0.15, 0.45, 0.30, 1.00, 0.26, 0.41, 0.25, 0.32],
                [0.55, 0.20, 0.50, 0.26, 1.00, 0.20, 0.44, 0.52],
                [0.18, 0.38, 0.22, 0.41, 0.20, 1.00, 0.27, 0.19],
                [0.35, 0.28, 0.46, 0.25, 0.44, 0.27, 1.00, 0.36],
                [0.48, 0.25, 0.42, 0.32, 0.52, 0.19, 0.36, 1.00],
            ],
            dtype=float,
        )
        covariance = corr * np.outer(volatility, volatility)
        choose_k = 3
    else:
        rng = np.random.default_rng(8300 + n_assets)
        returns = 0.045 + 0.16 * rng.random(n_assets)
        volatility = 0.10 + 0.24 * rng.random(n_assets)
        factors = rng.normal(0.0, 1.0, size=(n_assets, 3)) * volatility[:, None]
        covariance = 0.55 * (factors @ factors.T) / factors.shape[1]
        covariance += np.diag(0.45 * volatility * volatility)
        choose_k = max(3, n_assets // 4)

    risk_weight = 0.8
    return_weight = 1.8
    penalty = 3.0
    model = QuboModel(np.zeros(n_assets, dtype=float), np.zeros((n_assets, n_assets), dtype=float))

    for i in range(n_assets):
        add_linear(model, i, risk_weight * covariance[i, i] - return_weight * returns[i])
    for i, j in itertools.combinations(range(n_assets), 2):
        add_quadratic(model, i, j, 2.0 * risk_weight * covariance[i, j])
    add_cardinality(model, list(range(n_assets)), choose_k, penalty)

    J, h, offset = qubo_to_ising(model)
    problem = IsingProblem(
        name=f"portfolio_{n_assets}",
        J=J,
        h=h,
        offset=offset,
        metadata={
            "returns": returns.tolist(),
            "volatility": volatility.tolist(),
            "covariance": covariance.tolist(),
            "choose_k": choose_k,
            "risk_weight": risk_weight,
            "return_weight": return_weight,
            "penalty": penalty,
        },
    )
    return problem, model


def decode_portfolio(spins: np.ndarray, metadata: dict) -> dict:
    x = binary_from_spins(spins).astype(float)
    returns = np.array(metadata["returns"], dtype=float)
    covariance = np.array(metadata["covariance"], dtype=float)
    selected = [int(i) for i, bit in enumerate(x) if bit > 0.5]
    expected_return = float(returns @ x)
    risk = float(x @ covariance @ x)
    feasible = bool(int(np.sum(x)) == int(metadata["choose_k"]))
    return {
        "selected_assets": selected,
        "selected_count": int(np.sum(x)),
        "feasible_cardinality": feasible,
        "expected_return": expected_return,
        "portfolio_variance": risk,
    }


def exact_portfolio(model: QuboModel, metadata: dict) -> dict:
    n_assets = len(metadata["returns"])
    choose_k = int(metadata["choose_k"])
    best_bits = None
    best_score = float("inf")
    for combo in itertools.combinations(range(n_assets), choose_k):
        x = np.zeros(n_assets, dtype=np.int8)
        x[list(combo)] = 1
        score = float(model.offset + model.linear @ x + np.sum(model.quadratic * np.outer(x, x)))
        if score < best_score:
            best_score = score
            best_bits = x
    spins = best_bits * 2 - 1
    decoded = decode_portfolio(spins.astype(np.int8), metadata)
    decoded["qubo_objective"] = float(best_score)
    return decoded


def run_problem(
    profile: KStateProfile,
    problem: IsingProblem,
    model: QuboModel,
    runs: int,
    cycles: int,
) -> tuple[list[IsingRun], IsingRun]:
    all_runs = []
    for run_idx in range(runs):
        result = run_kstate_ising_solver(profile, problem, run_idx=run_idx, cycles=cycles)
        result.best_objective = qubo_objective_from_spins(result.best_spins, model)
        result.final_objective = qubo_objective_from_spins(result.final_spins, model)
        all_runs.append(result)
    best_run = min(all_runs, key=lambda item: item.best_objective)
    return all_runs, best_run


def plot_tsp(decoded: dict, exact: dict, metadata: dict) -> None:
    if plt is None:
        return
    coords = np.array(metadata["coords"], dtype=float)
    raw_route = decoded["route"]
    refined_route = decoded["two_opt_route"]
    raw_closed = raw_route + [raw_route[0]]
    refined_closed = refined_route + [refined_route[0]]
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    ax.scatter(coords[:, 0], coords[:, 1], s=110, color="#2b6cb0", zorder=3)
    for idx, (x_val, y_val) in enumerate(coords):
        ax.text(x_val + 0.08, y_val + 0.08, str(idx), fontsize=11, weight="bold")
    ax.plot(
        coords[raw_closed, 0],
        coords[raw_closed, 1],
        color="#94a3b8",
        linewidth=1.4,
        linestyle="--",
        alpha=0.8,
        label="raw Ising route",
        zorder=1,
    )
    ax.plot(
        coords[refined_closed, 0],
        coords[refined_closed, 1],
        color="#c2410c",
        linewidth=2.1,
        label="2-opt refined route",
        zorder=2,
    )
    for pos, city in enumerate(refined_route):
        nxt = refined_route[(pos + 1) % len(refined_route)]
        mid = 0.5 * (coords[city] + coords[nxt])
        ax.text(mid[0], mid[1], str(pos + 1), color="#7c2d12", fontsize=9)
    reference_label = "exact" if exact.get("method") == "exact" else "reference"
    ax.set_title(
        f"{len(coords)}-city TSP from MTJ K-state Ising demo\n"
        f"raw={decoded['route_length']:.2f}, refined={decoded['two_opt_route_length']:.2f}, "
        f"{reference_label}={exact['route_length']:.2f}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tsp_route.png", dpi=180)
    plt.close(fig)


def plot_maxcut(decoded: dict, exact: dict, metadata: dict) -> None:
    if plt is None:
        return
    n_nodes = int(metadata["n_nodes"])
    theta = np.linspace(0.0, 2.0 * math.pi, n_nodes, endpoint=False)
    pos = np.column_stack([np.cos(theta), np.sin(theta)])
    x_bits = np.zeros(n_nodes, dtype=int)
    x_bits[decoded["partition_1"]] = 1
    colors = np.where(x_bits > 0, "#c2410c", "#2b6cb0")

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, j, weight in metadata["edges"]:
        cut = x_bits[i] != x_bits[j]
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color="#111827" if cut else "#9ca3af",
            linewidth=0.6 + 0.35 * weight,
            alpha=0.75 if cut else 0.35,
            zorder=1,
        )
    ax.scatter(pos[:, 0], pos[:, 1], s=300, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    for idx in range(n_nodes):
        ax.text(pos[idx, 0], pos[idx, 1], str(idx), color="white", ha="center", va="center", weight="bold")
    ax.set_title(
        "Weighted Max-Cut from MTJ K-state Ising demo\n"
        f"cut={decoded['cut_weight']:.1f}, exact={exact['cut_weight']:.1f}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "maxcut_partition.png", dpi=180)
    plt.close(fig)


def plot_portfolio(decoded: dict, exact: dict, metadata: dict) -> None:
    if plt is None:
        return
    returns = np.array(metadata["returns"], dtype=float)
    volatility = np.array(metadata["volatility"], dtype=float)
    selected = np.zeros(len(returns), dtype=bool)
    selected[decoded["selected_assets"]] = True

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(volatility[~selected], returns[~selected], s=90, color="#64748b", label="not selected")
    axes[0].scatter(volatility[selected], returns[selected], s=130, color="#c2410c", label="selected")
    for idx in range(len(returns)):
        axes[0].text(volatility[idx] + 0.004, returns[idx] + 0.002, str(idx), fontsize=9)
    axes[0].set_xlabel("Volatility")
    axes[0].set_ylabel("Expected return")
    axes[0].set_title("Asset selection")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    labels = [str(i) for i in range(len(returns))]
    axes[1].bar(labels, returns, color=np.where(selected, "#c2410c", "#94a3b8"))
    axes[1].set_title(
        "Portfolio result\n"
        f"return={decoded['expected_return']:.3f}, variance={decoded['portfolio_variance']:.3f}"
    )
    axes[1].set_xlabel("Asset")
    axes[1].set_ylabel("Expected return")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Cardinality-constrained portfolio QUBO "
        f"(exact assets={exact['selected_assets']})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "portfolio_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(scaling_results: list[dict]) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for result in scaling_results:
        n_cities = result["n_cities"]
        runs = result["runs"]
        histories = np.vstack([run.history for run in runs])
        stride = max(1, histories.shape[1] // 900)
        x_axis = np.arange(0, histories.shape[1], stride)
        target = np.min(histories)
        denom = np.maximum(histories[:, :1] - target, 1e-9)
        normalized_gap = np.clip((histories[:, ::stride] - target) / denom, 0.0, None)
        normalized_gap = np.maximum(normalized_gap, 1e-6)
        median_gap = np.quantile(normalized_gap, 0.50, axis=0)
        q25 = np.quantile(normalized_gap, 0.25, axis=0)
        q75 = np.quantile(normalized_gap, 0.75, axis=0)
        progress = x_axis / max(histories.shape[1] - 1, 1)
        ax.plot(progress, median_gap, linewidth=1.6, label=f"{n_cities} cities median")
        ax.fill_between(progress, q25, q75, alpha=0.12)
    ax.set_title("TSP scaling convergence under MTJ K-state stochastic updates")
    ax.set_xlabel("Normalized annealing progress")
    ax.set_ylabel("Normalized QUBO gap to best seen")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence.png", dpi=180)
    plt.close(fig)


def plot_tsp_route_trend(scaling_rows: list[dict]) -> None:
    if plt is None:
        return
    df = pd.DataFrame(scaling_rows).sort_values("n_cities")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(df["n_cities"], df["best_raw_ratio"], marker="o", label="best raw decoded")
    axes[0].plot(df["n_cities"], df["median_raw_ratio"], marker="o", label="median raw decoded")
    axes[0].plot(df["n_cities"], df["best_two_opt_ratio"], marker="s", label="best 2-opt refined")
    axes[0].plot(df["n_cities"], df["median_two_opt_ratio"], marker="s", label="median 2-opt refined")
    axes[0].axhline(1.0, color="#111827", linewidth=1.0, linestyle="--", label="reference")
    axes[0].set_ylabel("Route length / reference")
    axes[0].set_title("Ising-decoded TSP route quality trend")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2)

    axes[1].plot(df["n_cities"], df["feasible_rate"], marker="o", color="#047857")
    axes[1].set_ylabel("Raw one-hot feasible rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(df["n_cities"], df["median_onehot_violation"], marker="o", color="#c2410c", label="median")
    axes[2].plot(df["n_cities"], df["best_onehot_violation"], marker="s", color="#7c2d12", label="best decoded run")
    axes[2].set_xlabel("TSP city count")
    axes[2].set_ylabel("One-hot violation")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tsp_decoded_route_trend.png", dpi=180)
    plt.close(fig)


def run_tsp_scaling(profile: KStateProfile, sizes: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    scaling_results = []
    scaling_rows = []
    run_rows = []

    for n_cities in sizes:
        problem, model = build_tsp_problem(n_cities)
        reference = reference_tsp(problem.metadata)
        runs, best_qubo_run = run_problem(profile, problem, model, runs=24, cycles=45)
        decoded_runs = []

        for run in runs:
            decoded = decode_tsp(run.best_spins, problem.metadata)
            raw_ratio = decoded["route_length"] / reference["route_length"]
            two_opt_ratio = decoded["two_opt_route_length"] / reference["route_length"]
            decoded_runs.append((decoded, run, raw_ratio, two_opt_ratio))
            run_rows.append(
                {
                    "n_cities": n_cities,
                    "variables": len(problem.h),
                    "run_idx": run.run_idx,
                    "best_qubo_objective": run.best_objective,
                    "raw_route_length": decoded["route_length"],
                    "two_opt_route_length": decoded["two_opt_route_length"],
                    "raw_ratio_to_reference": raw_ratio,
                    "two_opt_ratio_to_reference": two_opt_ratio,
                    "raw_feasible": decoded["feasible_raw_assignment"],
                    "onehot_violation": decoded["onehot_violation"],
                    "acceptance_rate": run.acceptance_rate,
                }
            )

        best_decoded, best_decoded_run, best_raw_ratio, best_two_opt_ratio = min(
            decoded_runs,
            key=lambda item: (item[3], item[0]["onehot_violation"], item[2]),
        )
        raw_ratios = np.array([item[2] for item in decoded_runs], dtype=float)
        two_opt_ratios = np.array([item[3] for item in decoded_runs], dtype=float)
        violations = np.array([item[0]["onehot_violation"] for item in decoded_runs], dtype=float)
        feasible = np.array([item[0]["feasible_raw_assignment"] for item in decoded_runs], dtype=float)

        scaling_rows.append(
            {
                "n_cities": n_cities,
                "variables": int(len(problem.h)),
                "reference_length": float(reference["route_length"]),
                "reference_method": reference["method"],
                "best_raw_length": float(np.min([item[0]["route_length"] for item in decoded_runs])),
                "best_two_opt_length": float(best_decoded["two_opt_route_length"]),
                "median_raw_ratio": float(np.median(raw_ratios)),
                "best_raw_ratio": float(np.min(raw_ratios)),
                "median_two_opt_ratio": float(np.median(two_opt_ratios)),
                "best_two_opt_ratio": float(best_two_opt_ratio),
                "feasible_rate": float(np.mean(feasible)),
                "median_onehot_violation": float(np.median(violations)),
                "best_onehot_violation": int(best_decoded["onehot_violation"]),
                "best_decoded_run_idx": int(best_decoded_run.run_idx),
                "best_qubo_objective": float(best_qubo_run.best_objective),
                "best_decoded_solution": best_decoded,
                "reference_solution": reference,
            }
        )
        scaling_results.append(
            {
                "n_cities": n_cities,
                "problem": problem,
                "model": model,
                "reference": reference,
                "runs": runs,
                "best_decoded": best_decoded,
                "best_decoded_run": best_decoded_run,
            }
        )

    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "tsp_scaling_runs.csv", index=False)
    pd.DataFrame(
        [
            {k: v for k, v in row.items() if k not in {"best_decoded_solution", "reference_solution"}}
            for row in scaling_rows
        ]
    ).to_csv(OUTPUT_DIR / "tsp_scaling_summary.csv", index=False)
    return scaling_results, scaling_rows, run_rows


def normalized_gap_curves(histories: np.ndarray, target: float) -> np.ndarray:
    gaps = np.maximum(histories - target, 0.0)
    denom = np.maximum(gaps[:, :1], 1e-9)
    return np.maximum(gaps / denom, 1e-6)


def plot_structured_convergence(
    scaling_results: list[dict],
    output_name: str,
    title: str,
    hit_tolerance: float,
) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for result in scaling_results:
        label = result["label"]
        target = float(result["reference_objective"])
        histories = np.vstack([run.history for run in result["runs"]])
        current_histories = np.vstack([run.current_history for run in result["runs"]])
        stride = max(1, histories.shape[1] // 700)
        progress = np.arange(0, histories.shape[1], stride) / max(histories.shape[1] - 1, 1)

        best_norm = normalized_gap_curves(histories[:, ::stride], target)
        current_norm = normalized_gap_curves(current_histories[:, ::stride], target)
        best_q25, best_q50, best_q75 = np.quantile(best_norm, [0.25, 0.50, 0.75], axis=0)
        current_q50 = np.quantile(current_norm, 0.50, axis=0)
        hit_rate = np.mean(np.maximum(histories[:, ::stride] - target, 0.0) <= hit_tolerance, axis=0)

        axes[0].plot(progress, best_q50, linewidth=1.6, label=label)
        axes[0].fill_between(progress, best_q25, best_q75, alpha=0.12)
        axes[1].plot(progress, current_q50, linewidth=1.4, label=label)
        axes[2].plot(progress, hit_rate, linewidth=1.6, label=label)

    axes[0].set_title(f"{title}: best-so-far convergence")
    axes[0].set_ylabel("Normalized gap")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=2)

    axes[1].set_title("Current-state exploration")
    axes[1].set_ylabel("Median current gap")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)

    axes[2].set_title("Reference hit rate over runs")
    axes[2].set_xlabel("Normalized annealing progress")
    axes[2].set_ylabel("Hit rate")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=180)
    plt.close(fig)


def run_maxcut_scaling(profile: KStateProfile, sizes: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    scaling_results = []
    scaling_rows = []
    run_rows = []

    for n_nodes in sizes:
        problem, model = build_maxcut_problem(n_nodes)
        reference = reference_maxcut(problem.metadata)
        reference_objective = -float(reference["cut_weight"])
        runs, best_run = run_problem(profile, problem, model, runs=24, cycles=30)

        decoded_runs = []
        for run in runs:
            decoded = decode_maxcut(run.best_spins, problem.metadata)
            gap = float(run.best_objective - reference_objective)
            ratio = float(decoded["cut_weight"] / max(reference["cut_weight"], 1e-9))
            decoded_runs.append((decoded, run, gap, ratio))
            run_rows.append(
                {
                    "n_nodes": n_nodes,
                    "edges": len(problem.metadata["edges"]),
                    "run_idx": run.run_idx,
                    "best_objective": run.best_objective,
                    "reference_objective": reference_objective,
                    "gap_to_reference": gap,
                    "cut_weight": decoded["cut_weight"],
                    "cut_ratio_to_reference": ratio,
                    "acceptance_rate": run.acceptance_rate,
                }
            )

        gaps = np.array([item[2] for item in decoded_runs], dtype=float)
        ratios = np.array([item[3] for item in decoded_runs], dtype=float)
        best_decoded, best_decoded_run, best_gap, best_ratio = min(decoded_runs, key=lambda item: item[2])
        scaling_rows.append(
            {
                "n_nodes": n_nodes,
                "variables": int(len(problem.h)),
                "edges": int(len(problem.metadata["edges"])),
                "reference_cut": float(reference["cut_weight"]),
                "reference_method": reference["method"],
                "best_cut": float(best_decoded["cut_weight"]),
                "median_cut_ratio": float(np.median(ratios)),
                "best_cut_ratio": float(best_ratio),
                "median_gap_to_reference": float(np.median(gaps)),
                "best_gap_to_reference": float(best_gap),
                "hit_rate": float(np.mean(gaps <= 1e-9)),
                "best_run_idx": int(best_decoded_run.run_idx),
            }
        )
        scaling_results.append(
            {
                "label": f"{n_nodes} nodes",
                "n": n_nodes,
                "problem": problem,
                "reference": reference,
                "reference_objective": reference_objective,
                "runs": runs,
            }
        )

    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "maxcut_scaling_runs.csv", index=False)
    pd.DataFrame(scaling_rows).to_csv(OUTPUT_DIR / "maxcut_scaling_summary.csv", index=False)
    return scaling_results, scaling_rows, run_rows


def run_portfolio_scaling(profile: KStateProfile, sizes: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    scaling_results = []
    scaling_rows = []
    run_rows = []

    for n_assets in sizes:
        problem, model = build_portfolio_problem(n_assets)
        reference = exact_portfolio(model, problem.metadata)
        reference_objective = float(reference["qubo_objective"])
        runs, best_run = run_problem(profile, problem, model, runs=24, cycles=25)

        decoded_runs = []
        for run in runs:
            decoded = decode_portfolio(run.best_spins, problem.metadata)
            gap = float(run.best_objective - reference_objective)
            decoded_runs.append((decoded, run, gap))
            run_rows.append(
                {
                    "n_assets": n_assets,
                    "choose_k": problem.metadata["choose_k"],
                    "run_idx": run.run_idx,
                    "best_objective": run.best_objective,
                    "reference_objective": reference_objective,
                    "gap_to_reference": gap,
                    "selected_count": decoded["selected_count"],
                    "feasible_cardinality": decoded["feasible_cardinality"],
                    "expected_return": decoded["expected_return"],
                    "portfolio_variance": decoded["portfolio_variance"],
                    "acceptance_rate": run.acceptance_rate,
                }
            )

        gaps = np.array([item[2] for item in decoded_runs], dtype=float)
        feasible = np.array([item[0]["feasible_cardinality"] for item in decoded_runs], dtype=float)
        best_decoded, best_decoded_run, best_gap = min(decoded_runs, key=lambda item: item[2])
        scaling_rows.append(
            {
                "n_assets": n_assets,
                "variables": int(len(problem.h)),
                "choose_k": int(problem.metadata["choose_k"]),
                "reference_objective": reference_objective,
                "reference_assets": reference["selected_assets"],
                "best_objective": float(best_decoded_run.best_objective),
                "best_gap_to_reference": float(best_gap),
                "median_gap_to_reference": float(np.median(gaps)),
                "hit_rate": float(np.mean(gaps <= 1e-7)),
                "feasible_rate": float(np.mean(feasible)),
                "best_selected_assets": best_decoded["selected_assets"],
                "best_run_idx": int(best_decoded_run.run_idx),
            }
        )
        scaling_results.append(
            {
                "label": f"{n_assets} assets",
                "n": n_assets,
                "problem": problem,
                "reference": reference,
                "reference_objective": reference_objective,
                "runs": runs,
            }
        )

    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "portfolio_scaling_runs.csv", index=False)
    pd.DataFrame(scaling_rows).to_csv(OUTPUT_DIR / "portfolio_scaling_summary.csv", index=False)
    return scaling_results, scaling_rows, run_rows


def save_tables(run_rows: list[dict], summary: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "application_demo_runs.csv", index=False)
    with open(OUTPUT_DIR / "application_demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    profile = build_kstate_profile("good", steps=900)
    tsp_scaling_results, tsp_scaling_rows, tsp_scaling_run_rows = run_tsp_scaling(
        profile,
        sizes=[8, 12, 16, 20, 24],
    )
    maxcut_scaling_results, maxcut_scaling_rows, maxcut_scaling_run_rows = run_maxcut_scaling(
        profile,
        sizes=[10, 16, 24, 32],
    )
    portfolio_scaling_results, portfolio_scaling_rows, portfolio_scaling_run_rows = run_portfolio_scaling(
        profile,
        sizes=[8, 12, 16, 24],
    )
    problems_and_models = {
        "maxcut": build_maxcut_problem(),
        "portfolio": build_portfolio_problem(),
    }
    run_settings = {
        "maxcut": {"runs": 50, "cycles": 20},
        "portfolio": {"runs": 50, "cycles": 15},
    }

    exact = {
        "maxcut": exact_maxcut(problems_and_models["maxcut"][0].metadata),
        "portfolio": exact_portfolio(problems_and_models["portfolio"][1], problems_and_models["portfolio"][0].metadata),
    }
    exact_objective = {
        "maxcut": -float(exact["maxcut"]["cut_weight"]),
        "portfolio": float(exact["portfolio"]["qubo_objective"]),
    }
    decoders = {
        "maxcut": decode_maxcut,
        "portfolio": decode_portfolio,
    }

    best_runs = {}
    all_runs = {}
    decoded_results = {}
    run_rows = []
    summary = {
        "assumption": (
            "Each practical optimization task is encoded as a QUBO and then converted to the Ising "
            "energy used by the MTJ physical K-state stochastic solver. The good measured dataset "
            "provides the latent-state sequence, transition/dwell rates, and random numbers."
        ),
        "profile": {
            "dataset": profile.name,
            "selected_k": int(profile.selected_k),
            "pbit_dims": int(profile.metrics["pbit_dims"]),
            "state_entropy_bits": float(profile.metrics["state_entropy_bits"]),
        },
        "tsp_scaling": tsp_scaling_rows,
        "maxcut_scaling": maxcut_scaling_rows,
        "portfolio_scaling": portfolio_scaling_rows,
        "problems": {},
    }

    for name, (problem, model) in problems_and_models.items():
        settings = run_settings[name]
        runs, best_run = run_problem(profile, problem, model, runs=settings["runs"], cycles=settings["cycles"])
        best_runs[name] = best_run
        all_runs[name] = runs
        decoded = decoders[name](best_run.best_spins, problem.metadata)
        decoded_results[name] = decoded

        for result in runs:
            if name == "maxcut":
                decoded_run = decode_maxcut(result.best_spins, problem.metadata)
                quality = decoded_run["cut_weight"]
                feasible = True
            else:
                decoded_run = decode_portfolio(result.best_spins, problem.metadata)
                quality = decoded_run["expected_return"]
                feasible = decoded_run["feasible_cardinality"]

            run_rows.append(
                {
                    "problem": name,
                    "run_idx": result.run_idx,
                    "best_objective": result.best_objective,
                    "final_objective": result.final_objective,
                    "acceptance_rate": result.acceptance_rate,
                    "early_acceptance_rate": result.early_acceptance_rate,
                    "late_acceptance_rate": result.late_acceptance_rate,
                    "decoded_quality": quality,
                    "decoded_feasible": feasible,
                }
            )

        run_dicts = [
            {
                "best_energy": run.best_objective,
                "final_energy": run.final_objective,
                "energy_gap_to_optimum": float(run.best_objective - exact_objective[name]),
                "hit_optimum": bool(abs(run.best_objective - exact_objective[name]) < 1e-7),
                "acceptance_rate": run.acceptance_rate,
                "early_acceptance_rate": run.early_acceptance_rate,
                "late_acceptance_rate": run.late_acceptance_rate,
            }
            for run in runs
        ]
        summary["problems"][name] = {
            "variables": int(len(problem.h)),
            "runs": int(settings["runs"]),
            "cycles_per_run": int(settings["cycles"]),
            "best_objective": float(best_run.best_objective),
            "exact_objective": exact_objective[name],
            "best_gap_to_exact": float(best_run.best_objective - exact_objective[name]),
            "decoded_solution": decoded,
            "exact_reference": exact[name],
            "run_statistics": summarize_runs(run_dicts),
        }

    save_tables(run_rows, summary)
    tsp_24 = tsp_scaling_results[-1]
    plot_tsp(tsp_24["best_decoded"], tsp_24["reference"], tsp_24["problem"].metadata)
    plot_tsp_route_trend(tsp_scaling_rows)
    plot_maxcut(decoded_results["maxcut"], exact["maxcut"], problems_and_models["maxcut"][0].metadata)
    plot_portfolio(decoded_results["portfolio"], exact["portfolio"], problems_and_models["portfolio"][0].metadata)
    plot_convergence(tsp_scaling_results)
    plot_structured_convergence(
        maxcut_scaling_results,
        output_name="maxcut_convergence.png",
        title="Max-Cut scaling",
        hit_tolerance=1e-9,
    )
    plot_structured_convergence(
        portfolio_scaling_results,
        output_name="portfolio_convergence.png",
        title="Portfolio scaling",
        hit_tolerance=1e-7,
    )

    print("MTJ Ising application demos finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math

import numpy as np

from benchmarks import evaluate_benchmark
from methods.registry import register_optimizer


EXPENSIVE_MODES = {
    "ego_mobo",
    "pesmo_bo",
    "mesmo_bo",
    "kg_mobo",
    "turbo_mobo",
    "krvea",
    "mobo_ei",
    "mobo_ucb",
}


def register_variant(key: str, name: str, short_name: str, description: str, mode: str) -> None:
    register_optimizer(
        key,
        {"name": name, "shortName": short_name, "description": description},
        lambda demo, batch_size, selected_mode=mode, method_name=name: run_catalog_variant(demo, batch_size, selected_mode, method_name),
    )


def run_catalog_variant(demo, batch_size: int, mode: str, method_name: str) -> dict:
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")

    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    hv_before = demo.approximate_hypervolume(y, np.array(demo.state.hv_samples, dtype=float))
    archive_x, archive_y = demo.evolution_archive()
    pool = build_pool(demo, archive_x, archive_y, batch_size, mode)
    proxy_y, proxy_std, has_surrogate = proxy_objectives(demo, pool, mode)
    scores = acquisition_scores(demo, mode, pool, proxy_y, proxy_std)
    chosen_idx = demo.diverse_top_k(pool, scores, batch_size)
    chosen_x = pool[chosen_idx]
    chosen_y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in chosen_x])
    chosen_scores = demo.preference_score(chosen_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    uncertainty = np.linalg.norm(proxy_std[chosen_idx], axis=1) if proxy_std is not None else np.full(len(chosen_idx), 0.18)
    return demo.apply_round_results(
        method=method_name,
        strategy=strategy_name(mode),
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        predicted=proxy_y[chosen_idx],
        uncertainty=uncertainty,
        scores=chosen_scores if len(chosen_scores) else scores[chosen_idx],
        hv_before=hv_before,
        surrogate={
            "model": f"{method_name} {'RBF surrogate' if has_surrogate else 'evolutionary proxy'}",
            "containsUncertainty": bool(has_surrogate),
            "calibrated": False,
            "meanCandidateUncertainty": float(np.mean(np.linalg.norm(proxy_std, axis=1))) if proxy_std is not None else 0.18,
            "selectedCandidateUncertainty": float(np.mean(uncertainty)) if len(uncertainty) else None,
            "note": "This catalog demo runner uses lightweight ask-tell proxies; expensive variants use an RBF uncertainty proxy.",
        },
    )


def build_pool(demo, archive_x: np.ndarray, archive_y: np.ndarray, batch_size: int, mode: str) -> np.ndarray:
    input_dim = demo.input_dimension()
    population = int(demo.state.settings["populationSize"])
    random_part = demo.rng.random((max(320, population * 90), input_dim))
    if len(archive_x) == 0:
        archive_x = demo.rng.random((population, input_dim))
        archive_y = np.zeros((population, len(demo.objectives())), dtype=float)
    pareto = archive_x[demo.pareto_mask(archive_y)] if len(archive_y) else archive_x
    focus = pareto if len(pareto) else archive_x
    scale = mode_scale(mode)
    local_parts = [
        np.clip(base + demo.rng.normal(0.0, scale, size=(max(4, population // 2), input_dim)), 0.0, 1.0)
        for base in focus[: max(1, min(len(focus), population * 2))]
    ]
    parts = [random_part, *local_parts]
    if mode in {"pesa2", "epsilon_archive", "grid_moea"}:
        parts.append(grid_archive_candidates(demo, archive_x, archive_y, input_dim))
    if mode in {"mopso", "omopso", "smpso", "lmocso"}:
        parts.append(swarm_candidates(demo, archive_x, archive_y, input_dim, mode))
    if mode in {"ctaea", "two_archive", "vaea"}:
        parts.append(two_archive_candidates(demo, archive_x, archive_y, input_dim))
    if mode in {"mo_cma_es", "moea_dd"}:
        parts.append(covariance_candidates(demo, archive_x, archive_y, input_dim))
    if mode in {"sparse_moea", "lmocso"}:
        parts.append(sparse_candidates(demo, archive_x, input_dim))
    if mode in {"borg", "operator_moea"}:
        parts.append(operator_portfolio_candidates(demo, archive_x, input_dim))
    if mode in {"ar_moea", "r2_ibea", "krvea", "moea_dd", "vaea"}:
        parts.append(reference_vector_candidates(demo, batch_size, input_dim))
    if mode in EXPENSIVE_MODES:
        best = archive_x[int(np.argmax(demo.preference_score(archive_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())))]
        parts.append(np.clip(best + demo.rng.normal(0.0, max(0.06, scale), size=(max(80, batch_size * 20), input_dim)), 0.0, 1.0))
    pool = np.vstack(parts)
    _, idx = np.unique(np.round(pool, 4), axis=0, return_index=True)
    return pool[np.sort(idx)]


def mode_scale(mode: str) -> float:
    return {
        "paes": 0.07,
        "pesa2": 0.08,
        "mopso": 0.11,
        "omopso": 0.14,
        "smpso": 0.08,
        "borg": 0.14,
        "ctaea": 0.10,
        "mo_cma_es": 0.10,
        "ar_moea": 0.11,
        "two_archive": 0.10,
        "hype": 0.09,
        "sparse_moea": 0.18,
        "lmocso": 0.16,
        "r2_ibea": 0.11,
        "momfea": 0.13,
        "moea_dd": 0.10,
        "vaea": 0.12,
        "grid_moea": 0.09,
        "operator_moea": 0.14,
        "krvea": 0.08,
        "turbo_mobo": 0.07,
    }.get(mode, 0.12)


def proxy_objectives(demo, pool: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray | None, bool]:
    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    if len(x) >= max(6, x.shape[1] + 2):
        posterior = demo.RBFPosterior(x, y)
        mean, std = posterior.predict(pool)
        if mode in EXPENSIVE_MODES:
            return mean, std, True
        return mean, np.full_like(mean, 0.12), False
    objective_dim = len(demo.objectives())
    proxy = np.resize(pool, (len(pool), objective_dim))
    return np.clip(proxy, 0.0, 1.0), np.full((len(pool), objective_dim), 0.20), False


def reference_vector_candidates(demo, batch_size: int, input_dim: int) -> np.ndarray:
    objective_dim = max(2, len(demo.objectives()))
    rows = []
    for i in range(max(batch_size * 4, objective_dim * 8)):
        direction = demo.rng.dirichlet(np.ones(objective_dim))
        rows.append(np.clip(np.resize(direction, input_dim) + demo.rng.normal(0.0, 0.11, input_dim), 0.0, 1.0))
    return np.array(rows, dtype=float)


def grid_archive_candidates(demo, archive_x: np.ndarray, archive_y: np.ndarray, input_dim: int) -> np.ndarray:
    eps = 0.08
    if len(archive_x) == 0:
        return demo.rng.random((max(24, input_dim * 6), input_dim))
    boxes = np.floor(np.clip(archive_y, 0.0, 1.0) / eps)
    rows = []
    seen: set[tuple[int, ...]] = set()
    for idx, box in enumerate(boxes):
        key = tuple(int(v) for v in box)
        if key in seen:
            continue
        seen.add(key)
        rows.append(np.clip(archive_x[idx] + demo.rng.normal(0.0, 0.08, input_dim), 0.0, 1.0))
    return np.array(rows, dtype=float) if rows else demo.rng.random((max(24, input_dim * 6), input_dim))


def swarm_candidates(demo, archive_x: np.ndarray, archive_y: np.ndarray, input_dim: int, mode: str) -> np.ndarray:
    scores = demo.preference_score(archive_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    leader = archive_x[int(np.argmax(scores))]
    rows = []
    inertia = {"mopso": 0.58, "omopso": 0.42, "smpso": 0.32, "lmocso": 0.66}.get(mode, 0.52)
    for _ in range(max(28, demo.state.settings["populationSize"] * 4)):
        a = archive_x[int(demo.rng.integers(0, len(archive_x)))]
        b = archive_x[int(demo.rng.integers(0, len(archive_x)))]
        velocity = inertia * (b - a) + demo.rng.random(input_dim) * (leader - a)
        if mode == "omopso":
            velocity += demo.rng.normal(0.0, 0.10, input_dim)
        if mode == "smpso":
            velocity = np.clip(velocity, -0.18, 0.18)
        if mode == "lmocso":
            velocity *= demo.rng.random(input_dim) < min(0.45, 8.0 / max(1, input_dim))
        rows.append(np.clip(a + velocity, 0.0, 1.0))
    return np.array(rows, dtype=float)


def two_archive_candidates(demo, archive_x: np.ndarray, archive_y: np.ndarray, input_dim: int) -> np.ndarray:
    mask = demo.pareto_mask(archive_y)
    convergence = archive_x[mask] if np.any(mask) else archive_x
    spread_order = np.argsort(np.linalg.norm(archive_y - np.mean(archive_y, axis=0), axis=1))
    diversity = archive_x[spread_order[-max(4, min(len(archive_x), demo.state.settings["populationSize"])) :]]
    rows = []
    for _ in range(max(28, demo.state.settings["populationSize"] * 4)):
        a = convergence[int(demo.rng.integers(0, len(convergence)))]
        b = diversity[int(demo.rng.integers(0, len(diversity)))]
        rows.append(np.clip(0.55 * a + 0.45 * b + demo.rng.normal(0.0, 0.08, input_dim), 0.0, 1.0))
    return np.array(rows, dtype=float)


def covariance_candidates(demo, archive_x: np.ndarray, archive_y: np.ndarray, input_dim: int) -> np.ndarray:
    scores = demo.preference_score(archive_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    order = np.argsort(scores)[::-1]
    elite = archive_x[order[: max(4, min(len(order), demo.state.settings["populationSize"]))]]
    mean = elite.mean(axis=0)
    cov = np.cov(elite.T) if len(elite) > 1 else np.eye(input_dim) * 0.04
    cov = np.asarray(cov, dtype=float) + np.eye(input_dim) * 1e-5
    rows = demo.rng.multivariate_normal(mean, cov * 0.45, size=max(28, demo.state.settings["populationSize"] * 4))
    return np.clip(rows, 0.0, 1.0)


def sparse_candidates(demo, archive_x: np.ndarray, input_dim: int) -> np.ndarray:
    rows = []
    active = max(1, min(input_dim, round(math.sqrt(input_dim))))
    for _ in range(max(28, demo.state.settings["populationSize"] * 4)):
        base = archive_x[int(demo.rng.integers(0, len(archive_x)))].copy()
        axes = demo.rng.choice(input_dim, size=active, replace=False)
        base[axes] = np.clip(base[axes] + demo.rng.normal(0.0, 0.18, active), 0.0, 1.0)
        rows.append(base)
    return np.array(rows, dtype=float)


def operator_portfolio_candidates(demo, archive_x: np.ndarray, input_dim: int) -> np.ndarray:
    rows = []
    for i in range(max(32, demo.state.settings["populationSize"] * 5)):
        mode = i % 4
        if mode == 0:
            parent = archive_x[int(demo.rng.integers(0, len(archive_x)))]
            rows.append(np.clip(parent + demo.rng.normal(0.0, 0.11, input_dim), 0.0, 1.0))
        elif mode == 1 and len(archive_x) >= 3:
            ids = demo.rng.choice(len(archive_x), 3, replace=False)
            rows.append(np.clip(archive_x[ids[0]] + 0.65 * (archive_x[ids[1]] - archive_x[ids[2]]), 0.0, 1.0))
        elif mode == 2 and len(archive_x) >= 2:
            ids = demo.rng.choice(len(archive_x), 2, replace=False)
            rows.append(np.clip(0.5 * archive_x[ids[0]] + 0.5 * archive_x[ids[1]] + demo.rng.normal(0.0, 0.04, input_dim), 0.0, 1.0))
        else:
            rows.append(demo.rng.random(input_dim))
    return np.array(rows, dtype=float)


def acquisition_scores(demo, mode: str, pool: np.ndarray, proxy_y: np.ndarray, proxy_std: np.ndarray | None) -> np.ndarray:
    current_y = np.array(demo.state.y, dtype=float)
    current_x = np.array(demo.state.x, dtype=float)
    pareto_y = current_y[demo.pareto_mask(current_y)] if len(current_y) else np.empty((0, proxy_y.shape[1]))
    preference = demo.preference_score(proxy_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    domination = demo.pareto_improvement_proxy(proxy_y, pareto_y) if len(pareto_y) else np.ones(len(pool))
    novelty = np.minimum(1.0, np.linalg.norm(pool[:, None, :] - current_x[None, :, :], axis=2).min(axis=1)) if len(current_x) else np.ones(len(pool))
    crowding = crowding_score(proxy_y)
    reference = reference_alignment(proxy_y)
    hv_proxy = hypervolume_proxy(proxy_y, pareto_y)
    uncertainty = np.linalg.norm(proxy_std, axis=1) if proxy_std is not None else np.full(len(pool), 0.18)
    if mode in {"pesa2", "grid_moea", "epsilon_archive"}:
        return 0.85 * box_novelty(proxy_y) + 0.45 * crowding + 0.25 * novelty
    if mode in {"mopso", "omopso", "smpso", "lmocso"}:
        return 0.58 * preference + 0.58 * domination + 0.34 * novelty
    if mode in {"borg", "operator_moea"}:
        return 0.52 * domination + 0.42 * preference + 0.44 * novelty + 0.22 * crowding
    if mode in {"ctaea", "vaea"}:
        return 0.70 * reference + 0.55 * domination + 0.35 * crowding
    if mode == "mo_cma_es":
        return 0.70 * preference + 0.62 * domination + 0.20 * novelty
    if mode == "ar_moea":
        return 0.75 * reference + 0.42 * age_bias(len(pool)) + 0.28 * novelty
    if mode == "two_archive":
        return 0.62 * domination + 0.60 * crowding + 0.30 * preference
    if mode == "hype":
        return 1.20 * hv_proxy + 0.22 * novelty
    if mode in {"sparse_moea", "moea_dd"}:
        return 0.65 * domination + 0.42 * novelty + 0.30 * preference
    if mode == "r2_ibea":
        return 0.82 * reference + 0.52 * preference + 0.20 * novelty
    if mode == "momfea":
        return 0.52 * domination + 0.52 * novelty + 0.34 * transfer_bias(pool)
    if mode == "ego_mobo":
        return 0.80 * hv_proxy + 0.58 * domination + 0.38 * uncertainty
    if mode == "pesmo_bo":
        return 0.42 * preference + 1.10 * uncertainty + 0.30 * crowding
    if mode == "mesmo_bo":
        return 0.40 * np.max(proxy_y, axis=1) + 1.04 * uncertainty + 0.25 * novelty
    if mode == "kg_mobo":
        return 0.55 * preference + 0.55 * domination + 0.65 * uncertainty
    if mode == "turbo_mobo":
        return 0.88 * preference + 0.42 * uncertainty + 0.22 * local_density(pool, current_x)
    if mode == "krvea":
        return 0.60 * reference + 0.55 * uncertainty + 0.30 * crowding
    if mode == "mobo_ei":
        return 0.75 * np.maximum(preference - np.max(demo.preference_score(current_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())), 0.0) + 0.55 * uncertainty
    if mode == "mobo_ucb":
        return 0.70 * preference + 0.80 * uncertainty + 0.25 * domination
    return preference + 0.35 * novelty + 0.25 * domination


def crowding_score(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    center = np.mean(y, axis=0)
    dist = np.linalg.norm(y - center, axis=1)
    lo, hi = float(np.min(dist)), float(np.max(dist))
    return (dist - lo) / max(1e-9, hi - lo)


def hypervolume_proxy(y: np.ndarray, pareto_y: np.ndarray) -> np.ndarray:
    baseline = np.max(pareto_y, axis=0) if len(pareto_y) else np.zeros(y.shape[1])
    return np.prod(np.clip(y - baseline, 0.0, 1.0) + 1e-3, axis=1)


def reference_alignment(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(19 + y.shape[1])
    directions = rng.dirichlet(np.ones(y.shape[1]), size=len(y))
    return np.sum(y * directions, axis=1)


def box_novelty(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    return np.sum(np.abs(np.floor(y / 0.08) - np.floor(np.mean(y, axis=0) / 0.08)), axis=1)


def age_bias(length: int) -> np.ndarray:
    return np.linspace(0.9, 0.1, length)


def transfer_bias(pool: np.ndarray) -> np.ndarray:
    center_a = np.resize(np.array([0.25, 0.75, 0.50]), pool.shape[1])
    center_b = np.resize(np.array([0.70, 0.30, 0.55]), pool.shape[1])
    scale = max(1.0, math.sqrt(pool.shape[1]))
    return np.maximum(1.0 - np.linalg.norm(pool - center_a, axis=1) / scale, 1.0 - np.linalg.norm(pool - center_b, axis=1) / scale)


def local_density(pool: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if len(observed) == 0:
        return np.ones(len(pool))
    dist = np.linalg.norm(pool[:, None, :] - observed[None, :, :], axis=2).min(axis=1)
    return 1.0 / (1.0 + 12.0 * dist)


def strategy_name(mode: str) -> str:
    return {
        "pesa2": "grid archive selection",
        "mopso": "particle swarm archive attraction",
        "omopso": "turbulence-enhanced swarm search",
        "smpso": "speed-constrained swarm search",
        "borg": "adaptive operator portfolio evolution",
        "ctaea": "constraint-and-reference archive evolution",
        "mo_cma_es": "covariance-guided Pareto search",
        "ar_moea": "angle-regularity guided evolution",
        "two_archive": "convergence-diversity two-archive search",
        "hype": "hypervolume estimation search",
        "sparse_moea": "sparse-variable large-scale mutation",
        "lmocso": "large-scale competitive swarm search",
        "r2_ibea": "R2 indicator-guided evolution",
        "momfea": "multifactorial transfer evolution",
        "moea_dd": "dominance-decomposition guided evolution",
        "vaea": "vector-angle archive evolution",
        "grid_moea": "adaptive grid evolutionary search",
        "operator_moea": "multi-operator evolutionary search",
        "ego_mobo": "expensive hypervolume improvement search",
        "pesmo_bo": "predictive-entropy multi-objective search",
        "mesmo_bo": "max-value entropy multi-objective search",
        "kg_mobo": "knowledge-gradient surrogate search",
        "turbo_mobo": "trust-region expensive multi-objective search",
        "krvea": "surrogate-assisted reference-vector search",
        "mobo_ei": "expected-improvement surrogate search",
        "mobo_ucb": "upper-confidence surrogate exploration",
    }.get(mode, "catalog Pareto search")


for key, name, short_name, description, mode in [
    ("pesa2", "PESA-II", "PESA-II", "Grid-based Pareto envelope selection baseline.", "pesa2"),
    ("mopso", "MOPSO", "MOPSO", "Multi-objective particle swarm baseline with archive attraction.", "mopso"),
    ("omopso", "OMOPSO", "OMOPSO", "Turbulence-enhanced multi-objective PSO baseline.", "omopso"),
    ("smpso", "SMPSO", "SMPSO", "Speed-constrained multi-objective PSO baseline.", "smpso"),
    ("borg", "Borg MOEA", "Borg", "Adaptive operator portfolio MOEA baseline.", "borg"),
    ("ctaea", "C-TAEA", "C-TAEA", "Two-archive reference-guided constrained MOEA baseline.", "ctaea"),
    ("mo_cma_es", "MO-CMA-ES", "MO-CMA", "Covariance-guided Pareto search baseline.", "mo_cma_es"),
    ("ar_moea", "AR-MOEA", "AR-MOEA", "Angle-regularity guided MOEA baseline.", "ar_moea"),
    ("two_archive", "Two-Archive EA", "TAEA", "Convergence-diversity two-archive evolutionary baseline.", "two_archive"),
    ("hype", "HypE", "HypE", "Hypervolume-estimation evolutionary baseline.", "hype"),
    ("sparse_moea", "SparseEA", "SparseEA", "Sparse-variable large-scale MOEA baseline.", "sparse_moea"),
    ("lmocso", "LMOCSO", "LMOCSO", "Large-scale multi-objective competitive swarm baseline.", "lmocso"),
    ("r2_ibea", "R2-IBEA", "R2-IBEA", "R2 indicator based evolutionary baseline.", "r2_ibea"),
    ("momfea", "MOMFEA", "MOMFEA", "Multifactorial evolutionary transfer baseline.", "momfea"),
    ("moea_dd", "MOEA/DD", "MOEA/DD", "Dominance-and-decomposition guided evolutionary baseline.", "moea_dd"),
    ("vaea", "VaEA", "VaEA", "Vector-angle archive evolutionary baseline.", "vaea"),
    ("grid_moea", "Grid-MOEA", "Grid", "Adaptive-grid diversity preserving MOEA baseline.", "grid_moea"),
    ("operator_moea", "Multi-operator MOEA", "MO-MOEA", "Operator portfolio evolutionary baseline.", "operator_moea"),
    ("ego_mobo", "EGO-MOBO", "EGO", "Expensive multi-objective BO baseline using hypervolume-improvement proxies.", "ego_mobo"),
    ("pesmo_bo", "PESMO", "PESMO", "Expensive BO baseline inspired by predictive entropy search.", "pesmo_bo"),
    ("mesmo_bo", "MESMO", "MESMO", "Expensive BO baseline inspired by max-value entropy search.", "mesmo_bo"),
    ("kg_mobo", "MO-KG", "MO-KG", "Expensive BO baseline using a knowledge-gradient style proxy.", "kg_mobo"),
    ("turbo_mobo", "TuRBO-MO", "TuRBO-MO", "Trust-region surrogate optimization baseline for expensive objectives.", "turbo_mobo"),
    ("krvea", "K-RVEA", "K-RVEA", "Surrogate-assisted reference-vector evolutionary baseline.", "krvea"),
    ("mobo_ei", "MOBO-EI", "EI-MOBO", "Expected-improvement style expensive multi-objective baseline.", "mobo_ei"),
    ("mobo_ucb", "MOBO-UCB", "UCB-MOBO", "Upper-confidence style expensive multi-objective baseline.", "mobo_ucb"),
]:
    register_variant(key, name, short_name, description, mode)

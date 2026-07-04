from __future__ import annotations

import math

import numpy as np

from .registry import register_single_method


def archive_xy(demo) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(demo.state.x, dtype=float), np.asarray(demo.state.y, dtype=float)


def best_x(demo) -> np.ndarray:
    x, _ = archive_xy(demo)
    return x[demo.best_index()]


def elite(demo, keep_factor: float = 1.0) -> np.ndarray:
    x, y = archive_xy(demo)
    keep = min(len(x), max(4, int(round(demo.state.settings["populationSize"] * keep_factor))))
    return x[np.argsort(y)[:keep]]


def scale(demo, floor: float = 0.025, factor: float = 1.0) -> float:
    return max(floor, float(demo.state.sigma) * factor)


def sobol_like(demo, batch_size: int):
    dim = demo.state.settings["dimension"]
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], dtype=float)
    offset = len(demo.state.x) + np.arange(batch_size, dtype=float)[:, None] + 1.0
    bases = np.resize(primes, dim)
    rows = (offset / bases[None, :]) % 1.0
    rows = (rows + demo.rng.random((batch_size, dim)) * 0.03) % 1.0
    return rows, "low-discrepancy Sobol-like global scan"


def halton_search(demo, batch_size: int):
    dim = demo.state.settings["dimension"]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def radical_inverse(index: int, base: int) -> float:
        result = 0.0
        factor = 1.0 / base
        while index > 0:
            result += factor * (index % base)
            index //= base
            factor /= base
        return result

    start = len(demo.state.x) + 1
    rows = np.zeros((batch_size, dim), dtype=float)
    for i in range(batch_size):
        for j in range(dim):
            rows[i, j] = radical_inverse(start + i, primes[j % len(primes)])
    return rows, "Halton sequence space-filling search"


def orthogonal_array(demo, batch_size: int):
    dim = demo.state.settings["dimension"]
    levels = np.linspace(0.08, 0.92, max(3, int(math.ceil(math.sqrt(batch_size)))))
    rows = []
    for i in range(batch_size):
        row = np.zeros(dim, dtype=float)
        for j in range(dim):
            row[j] = levels[(i + j * (j + 1)) % len(levels)]
        rows.append(np.clip(row + demo.rng.normal(0.0, 0.015, dim), 0.0, 1.0))
    return np.asarray(rows), "orthogonal-array level design"


def covariance_restart_es(demo, batch_size: int):
    elite_x = elite(demo, 1.5)
    dim = demo.state.settings["dimension"]
    mean = elite_x.mean(axis=0)
    cov = np.cov(elite_x.T) if len(elite_x) > 1 else np.eye(dim) * 0.04
    cov = np.asarray(cov, dtype=float) + np.eye(dim) * 1e-6
    if demo.state.iteration % 5 == 0:
        mean = 0.65 * mean + 0.35 * demo.rng.random(dim)
    rows = demo.rng.multivariate_normal(mean, cov * max(0.04, demo.state.sigma), size=batch_size)
    return np.clip(rows, 0.0, 1.0), "restart covariance evolution strategy"


def cauchy_es(demo, batch_size: int):
    base = best_x(demo)
    step = scale(demo, 0.02, 0.45)
    rows = base + step * demo.rng.standard_cauchy((batch_size, len(base)))
    return np.clip(rows, 0.0, 1.0), "heavy-tailed Cauchy evolution strategy"


def annealed_de(demo, batch_size: int):
    x, y = archive_xy(demo)
    order = np.argsort(y)
    pool = x[order[: max(6, min(len(x), demo.state.settings["populationSize"] * 2))]]
    dim = demo.state.settings["dimension"]
    factor = max(0.25, 0.85 * (0.985 ** demo.state.iteration))
    rows = []
    for _ in range(batch_size):
        ids = demo.rng.choice(len(pool), size=min(3, len(pool)), replace=False)
        if len(ids) < 3:
            rows.append(np.clip(pool[ids[0]] + demo.rng.normal(0.0, scale(demo), dim), 0.0, 1.0))
            continue
        mutant = pool[ids[0]] + factor * (pool[ids[1]] - pool[ids[2]])
        target = pool[int(demo.rng.integers(0, len(pool)))]
        mask = demo.rng.random(dim) < 0.7
        mask[int(demo.rng.integers(0, dim))] = True
        rows.append(np.clip(np.where(mask, mutant, target), 0.0, 1.0))
    return np.asarray(rows), f"annealed differential evolution, F={factor:.2f}"


def adaptive_pso(demo, batch_size: int):
    dim = demo.state.settings["dimension"]
    x = elite(demo, 1.5)
    leader = best_x(demo)
    spread = np.maximum(np.std(x, axis=0), 0.03)
    inertia = max(0.25, 0.78 * (0.99 ** demo.state.iteration))
    rows = []
    for _ in range(batch_size):
        particle = x[int(demo.rng.integers(0, len(x)))]
        velocity = inertia * demo.rng.normal(0.0, spread, dim) + demo.rng.random(dim) * (leader - particle)
        rows.append(np.clip(particle + velocity, 0.0, 1.0))
    return np.asarray(rows), f"adaptive PSO archive sampling, inertia={inertia:.2f}"


def memetic_ga(demo, batch_size: int):
    x, y = archive_xy(demo)
    dim = demo.state.settings["dimension"]

    def tournament() -> np.ndarray:
        ids = demo.rng.choice(len(x), size=min(4, len(x)), replace=False)
        return x[ids[int(np.argmin(y[ids]))]]

    leader = best_x(demo)
    rows = []
    for i in range(batch_size):
        parent_a = tournament()
        parent_b = tournament()
        alpha = demo.rng.beta(2.0, 2.0, size=dim)
        child = alpha * parent_a + (1.0 - alpha) * parent_b
        if i % 2 == 0:
            child = 0.75 * child + 0.25 * leader
        child += demo.rng.normal(0.0, scale(demo, 0.025, 0.55), dim)
        rows.append(np.clip(child, 0.0, 1.0))
    return np.asarray(rows), "memetic genetic search with elite local refinement"


def pattern_restart(demo, batch_size: int):
    base = best_x(demo)
    dim = len(base)
    step = max(0.02, 0.16 * (0.94 ** demo.state.iteration))
    rows = []
    for i in range(batch_size):
        row = base.copy() if i % 4 else demo.rng.random(dim)
        axis = i % dim
        row[axis] = np.clip(row[axis] + demo.rng.choice([-1.0, 1.0]) * step, 0.0, 1.0)
        rows.append(row)
    return np.asarray(rows), f"restarted pattern search, step={step:.3f}"


def surrogate_thompson(demo, batch_size: int):
    x, y = archive_xy(demo)
    dim = demo.state.settings["dimension"]
    if len(x) < max(6, dim + 2):
        return adaptive_pso(demo, batch_size)
    posterior = demo.RBFPosterior(x, y[:, None], lengthscale=0.30, clip_mean=False)
    pool = demo.rng.random((max(700, batch_size * 140), dim))
    local = np.clip(best_x(demo) + demo.rng.normal(0.0, scale(demo, 0.06, 0.9), size=(len(pool) // 3, dim)), 0.0, 1.0)
    pool[: len(local)] = local
    mean, std = posterior.predict(pool)
    sampled = mean[:, 0] + demo.rng.normal(0.0, np.maximum(std[:, 0], 1e-6))
    chosen = demo.diverse_top_k(pool, -sampled, batch_size)
    return pool[chosen], "RBF Thompson sampling proxy"


def surrogate_entropy(demo, batch_size: int):
    x, y = archive_xy(demo)
    dim = demo.state.settings["dimension"]
    if len(x) < max(6, dim + 2):
        return sobol_like(demo, batch_size)
    posterior = demo.RBFPosterior(x, y[:, None], lengthscale=0.38, clip_mean=False)
    pool = demo.rng.random((max(700, batch_size * 140), dim))
    mean, std = posterior.predict(pool)
    novelty = np.minimum(1.0, np.linalg.norm(pool[:, None, :] - x[None, :, :], axis=2).min(axis=1))
    score = std[:, 0] + 0.2 * novelty - 0.05 * mean[:, 0]
    chosen = demo.diverse_top_k(pool, score, batch_size)
    return pool[chosen], "RBF entropy/uncertainty exploration proxy"


def surrogate_knowledge_gradient(demo, batch_size: int):
    x, y = archive_xy(demo)
    dim = demo.state.settings["dimension"]
    if len(x) < max(6, dim + 2):
        return covariance_restart_es(demo, batch_size)
    posterior = demo.RBFPosterior(x, y[:, None], lengthscale=0.28, clip_mean=False)
    pool = demo.rng.random((max(700, batch_size * 140), dim))
    mean, std = posterior.predict(pool)
    best = float(np.min(y))
    improvement = np.maximum(best - mean[:, 0], 0.0)
    score = improvement + 0.65 * std[:, 0]
    chosen = demo.diverse_top_k(pool, score, batch_size)
    return pool[chosen], "RBF knowledge-gradient style ranking"


def surrogate_turbo(demo, batch_size: int):
    x, y = archive_xy(demo)
    dim = demo.state.settings["dimension"]
    if len(x) < max(6, dim + 2):
        return pattern_restart(demo, batch_size)
    center = best_x(demo)
    radius = min(0.35, max(0.045, demo.state.sigma * (0.96 ** demo.state.iteration)))
    pool = np.clip(center + demo.rng.normal(0.0, radius, size=(max(600, batch_size * 120), dim)), 0.0, 1.0)
    posterior = demo.RBFPosterior(x, y[:, None], lengthscale=max(0.18, radius * 1.8), clip_mean=False)
    mean, std = posterior.predict(pool)
    score = -(mean[:, 0] - 1.35 * std[:, 0])
    chosen = demo.diverse_top_k(pool, score, batch_size)
    return pool[chosen], f"trust-region surrogate search, radius={radius:.3f}"


def register(key: str, name: str, short_name: str, description: str, runner) -> None:
    register_single_method(key, {"name": name, "shortName": short_name, "description": description}, runner)


METHODS = [
    ("sobol_like_search", "Sobol-like Search", "Sobol", "Low-discrepancy global design search.", sobol_like),
    ("halton_search", "Halton Search", "Halton", "Halton sequence space-filling baseline.", halton_search),
    ("orthogonal_array_search", "Orthogonal Array Search", "OA", "Level-balanced deterministic design baseline.", orthogonal_array),
    ("covariance_restart_es", "Covariance Restart ES", "CR-ES", "Covariance evolution strategy with periodic restarts.", covariance_restart_es),
    ("cauchy_mutation_es", "Cauchy Mutation ES", "Cauchy-ES", "Heavy-tailed evolution strategy for escaping local basins.", cauchy_es),
    ("annealed_de", "Annealed Differential Evolution", "A-DE", "Differential evolution with a decaying mutation factor.", annealed_de),
    ("adaptive_pso", "Adaptive PSO", "A-PSO", "Archive-spread adaptive particle swarm search.", adaptive_pso),
    ("memetic_ga", "Memetic GA", "M-GA", "Genetic search with local elite refinement.", memetic_ga),
    ("pattern_restart", "Restarted Pattern Search", "R-PS", "Pattern search with periodic random restarts.", pattern_restart),
    ("surrogate_thompson_bo", "Surrogate Thompson BO", "TS-BO", "RBF Thompson-sampling expensive optimization baseline.", surrogate_thompson),
    ("surrogate_entropy_bo", "Surrogate Entropy BO", "ENT-BO", "RBF uncertainty/entropy expensive optimization baseline.", surrogate_entropy),
    ("surrogate_kg_bo", "Surrogate Knowledge Gradient BO", "KG-BO", "RBF knowledge-gradient style expensive optimization baseline.", surrogate_knowledge_gradient),
    ("surrogate_turbo_bo", "Surrogate TuRBO", "TuRBO", "Trust-region RBF expensive optimization baseline.", surrogate_turbo),
]


for item in METHODS:
    register(*item)

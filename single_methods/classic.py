from __future__ import annotations

import math

import numpy as np

from .registry import register_single_method


def call_existing(name: str):
    def runner(demo, batch_size: int):
        return getattr(demo, name)(batch_size)

    return runner


def best_x(demo) -> np.ndarray:
    return np.array(demo.state.x, dtype=float)[demo.best_index()]


def elite_xy(demo) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    order = np.argsort(y)
    keep = min(len(x), max(4, demo.state.settings["populationSize"]))
    return x[order[:keep]], y[order[:keep]]


def local_scale(demo, floor: float = 0.02, factor: float = 1.0) -> float:
    return max(floor, float(demo.state.sigma) * factor)


def propose_one_plus_one_es(demo, batch_size: int):
    base = best_x(demo)
    sigma = local_scale(demo, 0.025, 0.85)
    rows = [np.clip(base + demo.rng.normal(0.0, sigma, size=len(base)), 0.0, 1.0) for _ in range(batch_size)]
    return np.array(rows), f"(1+1)-ES Gaussian mutation, sigma={sigma:.3f}"


def propose_mu_lambda_es(demo, batch_size: int):
    elite, _ = elite_xy(demo)
    dim = demo.state.settings["dimension"]
    ids = demo.rng.integers(0, len(elite), size=batch_size)
    scales = demo.rng.lognormal(mean=math.log(local_scale(demo, 0.03, 0.9)), sigma=0.35, size=batch_size)
    rows = np.array([np.clip(elite[i] + demo.rng.normal(0.0, scales[j], size=dim), 0.0, 1.0) for j, i in enumerate(ids)])
    return rows, "mu+lambda evolution strategy sampling"


def propose_jade_de(demo, batch_size: int):
    archive, _ = elite_xy(demo)
    dim = demo.state.settings["dimension"]
    rows = []
    for _ in range(batch_size):
        ids = demo.rng.choice(len(archive), size=min(3, len(archive)), replace=False)
        if len(ids) < 3:
            rows.append(np.clip(archive[ids[0]] + demo.rng.normal(0.0, 0.08, size=dim), 0.0, 1.0))
            continue
        f = np.clip(demo.rng.normal(0.55, 0.12), 0.15, 0.95)
        cr = np.clip(demo.rng.normal(0.72, 0.10), 0.05, 0.98)
        mutant = archive[ids[0]] + f * (archive[ids[1]] - archive[ids[2]])
        target = archive[int(demo.rng.integers(0, len(archive)))]
        mask = demo.rng.random(dim) < cr
        mask[int(demo.rng.integers(0, dim))] = True
        rows.append(np.clip(np.where(mask, mutant, target), 0.0, 1.0))
    return np.array(rows), "JADE-style adaptive differential mutation"


def propose_current_to_best_de(demo, batch_size: int):
    archive, _ = elite_xy(demo)
    best = best_x(demo)
    dim = len(best)
    rows = []
    for _ in range(batch_size):
        ids = demo.rng.choice(len(archive), size=min(3, len(archive)), replace=False)
        target = archive[ids[0]]
        r1 = archive[ids[1 % len(ids)]]
        r2 = archive[ids[2 % len(ids)]]
        mutant = target + 0.45 * (best - target) + 0.55 * (r1 - r2)
        mask = demo.rng.random(dim) < 0.85
        mask[int(demo.rng.integers(0, dim))] = True
        rows.append(np.clip(np.where(mask, mutant, target), 0.0, 1.0))
    return np.array(rows), "current-to-best differential mutation"


def propose_barebones_pso(demo, batch_size: int):
    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    global_best = x[int(np.argmin(y))]
    elite, _ = elite_xy(demo)
    rows = []
    for _ in range(batch_size):
        personal = elite[int(demo.rng.integers(0, len(elite)))]
        mean = 0.5 * (personal + global_best)
        std = np.maximum(np.abs(personal - global_best), 0.04)
        rows.append(np.clip(demo.rng.normal(mean, std), 0.0, 1.0))
    return np.array(rows), "bare-bones PSO Gaussian attractor"


def propose_quantum_pso(demo, batch_size: int):
    elite, _ = elite_xy(demo)
    attractor = elite.mean(axis=0)
    best = best_x(demo)
    beta = max(0.08, 0.55 * (0.97 ** demo.state.iteration))
    u = np.maximum(demo.rng.random((batch_size, len(best))), 1e-9)
    signs = np.where(demo.rng.random((batch_size, len(best))) < 0.5, -1.0, 1.0)
    rows = attractor + signs * beta * np.abs(best - attractor + 1e-3) * np.log(1.0 / u)
    return np.clip(rows, 0.0, 1.0), f"quantum PSO attractor, beta={beta:.3f}"


def propose_firefly(demo, batch_size: int):
    elite, elite_y = elite_xy(demo)
    best = elite[0]
    rows = []
    for _ in range(batch_size):
        a = elite[int(demo.rng.integers(0, len(elite)))]
        b = elite[int(demo.rng.integers(0, len(elite)))]
        attractiveness = math.exp(-2.0 * float(np.sum((a - b) ** 2)))
        if elite_y[0] <= float(np.mean(elite_y)):
            target = best
        else:
            target = b
        rows.append(np.clip(a + attractiveness * (target - a) + demo.rng.normal(0.0, local_scale(demo, 0.025, 0.7), size=len(a)), 0.0, 1.0))
    return np.array(rows), "firefly attractiveness move"


def propose_bat_algorithm(demo, batch_size: int):
    best = best_x(demo)
    elite, _ = elite_xy(demo)
    rows = []
    for _ in range(batch_size):
        base = elite[int(demo.rng.integers(0, len(elite)))]
        frequency = demo.rng.uniform(0.2, 1.0)
        pulse = demo.rng.normal(0.0, local_scale(demo, 0.02, 0.8), size=len(best))
        rows.append(np.clip(base + frequency * (best - base) + pulse, 0.0, 1.0))
    return np.array(rows), "bat-inspired frequency pulse search"


def propose_harmony_search(demo, batch_size: int):
    elite, _ = elite_xy(demo)
    dim = demo.state.settings["dimension"]
    rows = []
    for _ in range(batch_size):
        row = demo.rng.random(dim)
        memory = demo.rng.random(dim) < 0.88
        for j in range(dim):
            if memory[j]:
                row[j] = elite[int(demo.rng.integers(0, len(elite))), j]
                if demo.rng.random() < 0.35:
                    row[j] = np.clip(row[j] + demo.rng.normal(0.0, local_scale(demo, 0.02, 0.5)), 0.0, 1.0)
        rows.append(row)
    return np.array(rows), "harmony memory consideration and pitch adjustment"


def propose_aco_continuous(demo, batch_size: int):
    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    order = np.argsort(y)
    keep = min(len(order), max(5, demo.state.settings["populationSize"]))
    elite = x[order[:keep]]
    ranks = np.arange(keep, dtype=float)
    weights = np.exp(-(ranks**2) / max(1.0, 2.0 * keep))
    weights = weights / weights.sum()
    ids = demo.rng.choice(keep, size=batch_size, p=weights)
    sigma = np.maximum(elite.std(axis=0), local_scale(demo, 0.025, 0.55))
    rows = np.array([np.clip(demo.rng.normal(elite[i], sigma), 0.0, 1.0) for i in ids])
    return rows, "continuous ant-colony Gaussian archive sampling"


def propose_tabu_search(demo, batch_size: int):
    best = best_x(demo)
    dim = len(best)
    step = local_scale(demo, 0.03, 0.9)
    rows = []
    used_axes = set()
    for i in range(batch_size):
        axis = i % dim
        if axis in used_axes and len(used_axes) < dim:
            axis = next(j for j in range(dim) if j not in used_axes)
        used_axes.add(axis)
        row = best.copy()
        row[axis] = np.clip(row[axis] + demo.rng.choice([-1.0, 1.0]) * step, 0.0, 1.0)
        if demo.rng.random() < 0.35:
            row = np.clip(row + demo.rng.normal(0.0, step * 0.35, size=dim), 0.0, 1.0)
        rows.append(row)
    return np.array(rows), "tabu-style non-repeating coordinate moves"


def propose_spsa(demo, batch_size: int):
    best = best_x(demo)
    dim = len(best)
    step = local_scale(demo, 0.035, 0.75)
    rows = []
    for _ in range((batch_size + 1) // 2):
        delta = demo.rng.choice([-1.0, 1.0], size=dim)
        rows.append(np.clip(best + step * delta, 0.0, 1.0))
        if len(rows) < batch_size:
            rows.append(np.clip(best - step * delta, 0.0, 1.0))
    return np.array(rows[:batch_size]), "SPSA simultaneous perturbation probes"


def propose_direct_search(demo, batch_size: int):
    best = best_x(demo)
    dim = len(best)
    rows = []
    divisions = np.linspace(0.0, 1.0, max(3, int(math.ceil(batch_size ** (1.0 / max(1, min(4, dim)))))))
    for i in range(batch_size):
        row = best.copy()
        axis = i % dim
        row[axis] = divisions[(i // dim) % len(divisions)]
        rows.append(row)
    return np.array(rows), "DIRECT-style axis partition probes"


def propose_lhs_search(demo, batch_size: int):
    dim = demo.state.settings["dimension"]
    rows = np.zeros((batch_size, dim), dtype=float)
    for j in range(dim):
        rows[:, j] = (demo.rng.permutation(batch_size) + demo.rng.random(batch_size)) / max(1, batch_size)
    return rows, "Latin hypercube global sampling"


def propose_opposition_search(demo, batch_size: int):
    x = np.array(demo.state.x, dtype=float)
    elite, _ = elite_xy(demo)
    rows = []
    for i in range(batch_size):
        base = elite[i % len(elite)]
        opposite = 1.0 - base
        trial = 0.65 * opposite + 0.35 * demo.rng.random(len(base))
        if len(x) and demo.rng.random() < 0.5:
            trial = 0.5 * trial + 0.5 * x[int(demo.rng.integers(0, len(x)))]
        rows.append(np.clip(trial, 0.0, 1.0))
    return np.array(rows), "opposition-based elite reflection"


def surrogate_pool(demo, batch_size: int, lengthscale: float = 0.30):
    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)[:, None]
    dim = demo.state.settings["dimension"]
    if len(x) < max(6, dim + 2):
        return None, None, None
    posterior = demo.RBFPosterior(x, y, lengthscale=lengthscale, clip_mean=False)
    pool = demo.rng.random((max(600, batch_size * 120), dim))
    best = x[demo.best_index()]
    local = np.clip(best + demo.rng.normal(0.0, local_scale(demo, 0.05, 0.85), size=(len(pool) // 3, dim)), 0.0, 1.0)
    pool[: len(local)] = local
    mean, std = posterior.predict(pool)
    return pool, mean[:, 0], std[:, 0]


def propose_ei_bo(demo, batch_size: int):
    pool, mean, std = surrogate_pool(demo, batch_size, 0.34)
    if pool is None:
        return propose_mu_lambda_es(demo, batch_size)
    best = float(np.min(demo.state.y))
    z = (best - mean) / np.maximum(std, 1e-9)
    ei_proxy = np.maximum(best - mean, 0.0) + std * np.maximum(z, 0.0)
    chosen = demo.diverse_top_k(pool, ei_proxy, batch_size)
    return pool[chosen], "RBF expected-improvement proxy"


def propose_pi_bo(demo, batch_size: int):
    pool, mean, std = surrogate_pool(demo, batch_size, 0.30)
    if pool is None:
        return propose_mu_lambda_es(demo, batch_size)
    best = float(np.min(demo.state.y))
    score = (best - mean) / np.maximum(std, 1e-9)
    chosen = demo.diverse_top_k(pool, score, batch_size)
    return pool[chosen], "RBF probability-of-improvement proxy"


def propose_ucb_bo(demo, batch_size: int):
    pool, mean, std = surrogate_pool(demo, batch_size, 0.28)
    if pool is None:
        return propose_mu_lambda_es(demo, batch_size)
    beta = 1.2 + 0.25 * math.log1p(max(1, len(demo.state.x)))
    score = -(mean - beta * std)
    chosen = demo.diverse_top_k(pool, score, batch_size)
    return pool[chosen], f"RBF UCB/LCB proxy, beta={beta:.2f}"


def register(key: str, name: str, short: str, description: str, runner) -> None:
    register_single_method(key, {"name": name, "shortName": short, "description": description}, runner)


for key, name, short, description, method_name in [
    ("random_search", "Random search", "RS", "Uniform population sampling baseline.", "propose_random_search"),
    ("hill_climb", "Adaptive hill climb", "HC", "Mutates the current elite and adapts the step size.", "propose_hill_climb"),
    ("de", "Differential evolution", "DE", "Population vector-difference search with greedy selection.", "propose_de"),
    ("pso", "Particle swarm optimization", "PSO", "Classic PSO with inertia, cognitive, and social attraction.", "propose_pso"),
    ("ga", "Genetic algorithm", "GA", "Tournament selection, blend crossover, mutation, and elitist archive sampling.", "propose_ga"),
    ("rbf_bo", "Bayesian optimization (RBF-LCB)", "BO", "RBF surrogate with lower-confidence-bound candidate selection.", "propose_rbf_bo"),
    ("cma_es", "CMA-ES", "CMA-ES", "Covariance-adapted Gaussian search around elite solutions.", "propose_cma_es"),
    ("nelder_mead", "Nelder-Mead", "NM", "Simplex-style local search with reflection and contraction.", "propose_nelder_mead"),
    ("simulated_annealing", "Simulated annealing", "SA", "Temperature-controlled local perturbation search.", "propose_simulated_annealing"),
    ("pattern_search", "Pattern search", "PS", "Coordinate pattern search around the current elite solution.", "propose_pattern_search"),
    ("coordinate_search", "Coordinate search", "CS", "Axis-aligned local search over individual variables.", "propose_coordinate_search"),
    ("cem", "Cross-entropy method", "CEM", "Elite-sampling search that fits a Gaussian to good solutions.", "propose_cem"),
    ("tpe", "TPE-like search", "TPE", "Tree-structured Parzen style density search using good and bad sets.", "propose_tpe"),
    ("trust_region_bo", "Trust-region BO", "TR-BO", "Local surrogate BO with a shrinking trust region around the current best.", "propose_trust_region_bo"),
    ("random_restart_search", "Random restart search", "RR", "Multiple localized restarts around the best solution plus random exploration.", "propose_random_restart_search"),
    ("local_rbf_bo", "Local RBF BO", "LBO", "Local surrogate BO baseline that concentrates sampling around the best point.", "propose_local_rbf_bo"),
]:
    register(key, name, short, description, call_existing(method_name))


EXTRA_METHODS = [
    ("one_plus_one_es", "(1+1)-ES", "1+1-ES", "Single-parent evolution strategy with Gaussian mutation.", propose_one_plus_one_es),
    ("mu_lambda_es", "mu,lambda Evolution Strategy", "mu-lambda ES", "Multi-parent evolution strategy with lognormal step sampling.", propose_mu_lambda_es),
    ("jade_de", "JADE Differential Evolution", "JADE", "Adaptive-DE inspired mutation and crossover baseline.", propose_jade_de),
    ("current_to_best_de", "Current-to-best DE", "ctb-DE", "Differential mutation biased toward the current best solution.", propose_current_to_best_de),
    ("barebones_pso", "Bare-bones PSO", "BBPSO", "Gaussian attractor particle swarm variant.", propose_barebones_pso),
    ("quantum_pso", "Quantum-behaved PSO", "QPSO", "Quantum PSO-style attractor sampling.", propose_quantum_pso),
    ("firefly", "Firefly algorithm", "FA", "Brightness-attraction population search baseline.", propose_firefly),
    ("bat", "Bat algorithm", "BA", "Frequency-pulse swarm search baseline.", propose_bat_algorithm),
    ("harmony_search", "Harmony search", "HS", "Harmony memory and pitch-adjusted sampling baseline.", propose_harmony_search),
    ("continuous_aco", "Continuous ant colony", "ACOR", "Rank-weighted Gaussian archive sampling baseline.", propose_aco_continuous),
    ("tabu_search", "Tabu search", "Tabu", "Local coordinate search with non-repeating move preference.", propose_tabu_search),
    ("spsa", "SPSA", "SPSA", "Simultaneous perturbation stochastic approximation probes.", propose_spsa),
    ("direct_search", "DIRECT-style search", "DIRECT", "Axis partition probes inspired by DIRECT.", propose_direct_search),
    ("lhs_search", "Latin hypercube search", "LHS", "Space-filling Latin hypercube baseline.", propose_lhs_search),
    ("opposition_search", "Opposition-based search", "OBS", "Reflects elite candidates across the domain.", propose_opposition_search),
    ("surrogate_ei_bo", "RBF BO Expected Improvement", "EI-BO", "Expensive optimization baseline using an RBF expected-improvement proxy.", propose_ei_bo),
    ("surrogate_pi_bo", "RBF BO Probability Improvement", "PI-BO", "Expensive optimization baseline using an RBF probability-of-improvement proxy.", propose_pi_bo),
    ("surrogate_ucb_bo", "RBF BO UCB/LCB", "UCB-BO", "Expensive optimization baseline using an uncertainty-weighted RBF acquisition.", propose_ucb_bo),
]

for item in EXTRA_METHODS:
    register(*item)

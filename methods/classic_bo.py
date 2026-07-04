from __future__ import annotations

import math

import numpy as np

from benchmarks import evaluate_benchmark
from methods.registry import register_optimizer


def run_classic_bo(demo, batch_size: int) -> dict:
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")

    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    hv_before = demo.approximate_hypervolume(y, np.array(demo.state.hv_samples, dtype=float))
    posterior = demo.RBFPosterior(x, y)
    pool = candidate_pool(demo, x)
    mean, std = posterior.predict(pool)
    weights = scalarization_weights(demo)
    observed_scalar = y @ weights
    mean_scalar = mean @ weights
    std_scalar = np.maximum(np.sqrt(np.sum((std * weights[None, :]) ** 2, axis=1)), 1e-9)
    incumbent = float(np.max(observed_scalar))
    ei = expected_improvement(mean_scalar, std_scalar, incumbent)
    chosen_idx = demo.diverse_top_k(pool, ei, batch_size)
    chosen_x = pool[chosen_idx]
    chosen_y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in chosen_x])
    chosen_scalar = chosen_y @ weights
    uncertainty = std_scalar[chosen_idx]
    return demo.apply_round_results(
        method="Classic BO",
        strategy="scalarized expected improvement",
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        predicted=mean[chosen_idx],
        uncertainty=uncertainty,
        scores=chosen_scalar,
        hv_before=hv_before,
        surrogate={
            "model": "Classic scalarized RBF-BO",
            "containsUncertainty": True,
            "calibrated": False,
            "meanCandidateUncertainty": float(np.mean(std_scalar)) if len(std_scalar) else None,
            "selectedCandidateUncertainty": float(np.mean(uncertainty)) if len(uncertainty) else None,
            "note": "Classic BO baseline: objectives are collapsed into one weighted scalar objective and Expected Improvement is optimized. It is not Pareto-aware and does not use hypervolume improvement.",
        },
    )


def scalarization_weights(demo) -> np.ndarray:
    active = demo.active_weights_dict()
    objectives = demo.objectives()
    raw = np.array([active.get(obj.get("preferenceKey", "quality"), 0.0) for obj in objectives], dtype=float)
    if float(raw.sum()) <= 1e-9:
        raw = np.ones(len(objectives), dtype=float)
    return raw / raw.sum()


def candidate_pool(demo, x: np.ndarray) -> np.ndarray:
    input_dim = demo.input_dimension()
    random_count = max(260, demo.state.settings["populationSize"] * 90)
    random_part = demo.rng.random((random_count, input_dim))
    local_parts = []
    local_count = max(8, round(demo.state.settings["populationSize"] * 2.0))
    best_count = min(len(x), max(3, demo.state.settings["populationSize"]))
    weights = scalarization_weights(demo)
    y = np.array(demo.state.y, dtype=float)
    order = np.argsort(y @ weights)[::-1][:best_count]
    for idx in order:
        local_parts.append(np.clip(x[idx] + demo.rng.normal(0.0, 0.10, size=(local_count, input_dim)), 0.0, 1.0))
    pool = np.vstack([random_part, *local_parts]) if local_parts else random_part
    _, idx = np.unique(np.round(pool, 4), axis=0, return_index=True)
    return pool[np.sort(idx)]


def expected_improvement(mean: np.ndarray, std: np.ndarray, incumbent: float, xi: float = 0.01) -> np.ndarray:
    improvement = mean - incumbent - xi
    z = improvement / std
    return improvement * normal_cdf(z) + std * normal_pdf(z)


def normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


register_optimizer(
    "classic_bo",
    {
        "name": "Classic BO",
        "shortName": "BO-EI",
        "description": "Python backend: original scalarized Bayesian optimization baseline using RBF posterior uncertainty and Expected Improvement.",
    },
    run_classic_bo,
)

from __future__ import annotations

import math

import numpy as np

from benchmarks import evaluate_benchmark, materialize_benchmark
from methods.registry import register_optimizer


def register_variant(key: str, name: str, short_name: str, description: str, mode: str) -> None:
    register_optimizer(
        key,
        {
            "name": name,
            "shortName": short_name,
            "description": description,
        },
        lambda demo, batch_size, selected_mode=mode, method_name=name: run_variant(demo, batch_size, selected_mode, method_name),
    )


def run_variant(demo, batch_size: int, mode: str, method_name: str) -> dict:
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")

    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    hv_before = demo.approximate_hypervolume(y, np.array(demo.state.hv_samples, dtype=float))
    archive_x, archive_y = demo.evolution_archive()
    pool = candidate_pool(demo, archive_x, archive_y, batch_size, mode)
    posterior = demo.RBFPosterior(x, y) if mode in {"qnehvi_bo", "qparego_bo", "thompson_bo", "entropy_bo"} and len(x) >= max(6, x.shape[1] + 2) else None
    mean = std = None
    if posterior is not None:
        mean, std = posterior.predict(pool)

    scores = acquisition_scores(demo, mode, pool, archive_y, mean, std)
    chosen_idx = demo.diverse_top_k(pool, scores, batch_size)
    chosen_x = pool[chosen_idx]
    chosen_y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in chosen_x])
    chosen_scores = demo.preference_score(chosen_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    uncertainty = np.linalg.norm(std[chosen_idx], axis=1) if std is not None else np.full(len(chosen_idx), 0.18, dtype=float)
    surrogate = {
        "model": f"{method_name} surrogate proxy" if posterior is not None else f"{method_name} evolutionary proxy",
        "containsUncertainty": posterior is not None,
        "calibrated": False,
        "meanCandidateUncertainty": float(np.mean(np.linalg.norm(std, axis=1))) if std is not None else 0.18,
        "selectedCandidateUncertainty": float(np.mean(uncertainty)) if len(uncertainty) else None,
        "note": surrogate_note(mode),
    }
    return demo.apply_round_results(
        method=method_name,
        strategy=strategy_name(mode),
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        predicted=mean[chosen_idx] if mean is not None else chosen_y,
        uncertainty=uncertainty,
        scores=chosen_scores if len(chosen_scores) else scores[chosen_idx],
        hv_before=hv_before,
        surrogate=surrogate,
    )


def strategy_name(mode: str) -> str:
    return {
        "nsga3": "reference-direction guided mating",
        "spea2": "strength-and-density archive selection",
        "paes": "parent-centric local mutation",
        "gde3": "differential mutation with dominance filtering",
        "ibea": "indicator-based environmental selection",
        "smsemoa": "hypervolume contribution selection",
        "rvea": "adaptive reference-vector guidance",
        "mocell": "cellular neighborhood evolution",
        "epsilon_moea": "epsilon-box archive evolution",
        "age_moea": "age-aware archive evolution",
        "qnehvi_bo": "qNEHVI-style uncertainty-guided improvement",
        "qparego_bo": "qParEGO-style random scalarization",
        "thompson_bo": "Thompson-sampled surrogate ranking",
        "entropy_bo": "entropy-seeking surrogate exploration",
    }.get(mode, "Pareto-guided evolution")


def surrogate_note(mode: str) -> str:
    if mode in {"qnehvi_bo", "qparego_bo", "thompson_bo", "entropy_bo"}:
        return "This baseline uses an RBF posterior proxy with uncertainty-aware candidate ranking."
    return "This baseline is deterministic and does not use a calibrated surrogate uncertainty model."


def candidate_pool(demo, archive_x: np.ndarray, archive_y: np.ndarray, batch_size: int, mode: str) -> np.ndarray:
    input_dim = demo.input_dimension()
    random_count = max(280, demo.state.settings["populationSize"] * 96)
    random_part = demo.rng.random((random_count, input_dim))
    pareto_x = archive_x[demo.pareto_mask(archive_y)]
    archive_x = archive_x if len(archive_x) else demo.rng.random((demo.state.settings["populationSize"], input_dim))
    focus = pareto_x if len(pareto_x) else archive_x
    local_scale = {
        "nsga3": 0.12,
        "spea2": 0.10,
        "paes": 0.08,
        "gde3": 0.16,
        "ibea": 0.11,
        "smsemoa": 0.09,
        "rvea": 0.10,
        "mocell": 0.13,
        "epsilon_moea": 0.08,
        "age_moea": 0.15,
        "qnehvi_bo": 0.09,
        "qparego_bo": 0.10,
        "thompson_bo": 0.11,
        "entropy_bo": 0.12,
    }.get(mode, 0.12)
    local_count = max(6, round(demo.state.settings["populationSize"] * 2.2))
    local_parts = []
    for base in focus:
        local_parts.append(np.clip(base + demo.rng.normal(0.0, local_scale, size=(local_count, input_dim)), 0.0, 1.0))
    if mode in {"qnehvi_bo", "qparego_bo", "thompson_bo", "entropy_bo"} and len(archive_x):
        best = archive_x[int(np.argmax(demo.preference_score(archive_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())))]
        local_parts.append(np.clip(best + demo.rng.normal(0.0, local_scale * 0.8, size=(max(16, batch_size * 10), input_dim)), 0.0, 1.0))
    parts = [random_part, *local_parts] if local_parts else [random_part]
    if mode in {"nsga3", "rvea"}:
        parts.append(reference_vector_candidates(demo, batch_size, input_dim))
    if mode == "mocell":
        parts.append(neighborhood_candidates(demo, archive_x, input_dim))
    if mode == "epsilon_moea":
        parts.append(epsilon_box_candidates(demo, archive_x, archive_y, input_dim))
    pool = np.vstack(parts)
    _, idx = np.unique(np.round(pool, 4), axis=0, return_index=True)
    return pool[np.sort(idx)]


def reference_vector_candidates(demo, batch_size: int, input_dim: int) -> np.ndarray:
    directions = []
    obj_count = max(2, len(demo.objectives()))
    for i in range(max(batch_size * 3, obj_count * 4)):
        raw = np.array([((i + 1 + j) % obj_count) + 1 for j in range(obj_count)], dtype=float)
        raw = raw / raw.sum()
        center = np.resize(raw, input_dim)
        directions.append(np.clip(center + demo.rng.normal(0.0, 0.10, size=input_dim), 0.0, 1.0))
    return np.array(directions, dtype=float)


def neighborhood_candidates(demo, archive_x: np.ndarray, input_dim: int) -> np.ndarray:
    if len(archive_x) == 0:
        return demo.rng.random((max(16, input_dim * 6), input_dim))
    rows = []
    for i, base in enumerate(archive_x[: max(6, len(archive_x))]):
        step = 0.06 + 0.01 * (i % 3)
        rows.append(np.clip(base + demo.rng.normal(0.0, step, size=(4, input_dim)), 0.0, 1.0))
    return np.vstack(rows)


def epsilon_box_candidates(demo, archive_x: np.ndarray, archive_y: np.ndarray, input_dim: int) -> np.ndarray:
    if len(archive_x) == 0:
        return demo.rng.random((max(18, input_dim * 8), input_dim))
    eps = 0.08
    box_keys = np.floor(archive_y / eps)
    keep: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for idx, key in enumerate(box_keys):
        tup = tuple(int(v) for v in key)
        if tup in seen:
            continue
        seen.add(tup)
        keep.append(np.clip(archive_x[idx] + demo.rng.normal(0.0, 0.07, size=input_dim), 0.0, 1.0))
    return np.array(keep, dtype=float) if keep else demo.rng.random((max(18, input_dim * 8), input_dim))


def acquisition_scores(demo, mode: str, pool: np.ndarray, archive_y: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    current = np.array(demo.state.y, dtype=float)
    pareto_y = current[demo.pareto_mask(current)] if len(current) else np.empty((0, len(demo.objectives())))
    preference = demo.preference_score(predict_pool(pool, current, demo), demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    novelty = np.minimum(1.0, np.linalg.norm(pool[:, None, :] - np.array(demo.state.x, dtype=float)[None, :, :], axis=2).min(axis=1) if len(demo.state.x) else 1.0)
    domination = demo.pareto_improvement_proxy(predict_pool(pool, current, demo), pareto_y) if len(pareto_y) else np.ones(len(pool), dtype=float)
    crowding = reference_crowding(proxy=pool, y=predict_pool(pool, current, demo))
    if mode == "nsga3":
        return 0.8 * domination + 0.55 * crowding + 0.3 * novelty
    if mode == "spea2":
        return 0.95 * domination + 0.45 * novelty + 0.30 * crowding
    if mode == "paes":
        return 0.65 * preference + 0.75 * novelty + 0.18 * domination
    if mode == "gde3":
        return 0.70 * domination + 0.55 * novelty + 0.30 * preference
    if mode == "ibea":
        return 1.05 * domination + 0.25 * novelty + 0.35 * preference
    if mode == "smsemoa":
        return 1.15 * hypervolume_contribution_proxy(predict_pool(pool, current, demo), pareto_y) + 0.20 * novelty
    if mode == "rvea":
        return 0.65 * reference_alignment_score(predict_pool(pool, current, demo)) + 0.45 * crowding + 0.28 * novelty
    if mode == "mocell":
        return 0.60 * preference + 0.55 * crowding + 0.40 * novelty
    if mode == "epsilon_moea":
        return 0.72 * box_novelty(predict_pool(pool, current, demo)) + 0.52 * novelty + 0.28 * preference
    if mode == "age_moea":
        return 0.55 * preference + 0.35 * novelty + 0.50 * age_bias(pool)
    if mode in {"qnehvi_bo", "qparego_bo", "thompson_bo", "entropy_bo"}:
        predicted = mean if mean is not None else predict_pool(pool, current, demo)
        uncertainty = np.linalg.norm(std, axis=1) if std is not None else np.full(len(pool), 0.18, dtype=float)
        if mode == "qnehvi_bo":
            return 1.05 * hypervolume_contribution_proxy(predicted, pareto_y) + 0.75 * uncertainty + 0.20 * novelty
        if mode == "qparego_bo":
            weights = random_simplex(demo, predicted.shape[1], 1)[0]
            scalar = predicted @ weights
            return scalar + 0.55 * uncertainty + 0.35 * novelty
        if mode == "thompson_bo":
            sampled = predicted + demo.rng.normal(0.0, np.maximum(uncertainty[:, None], 1e-3), size=predicted.shape)
            return demo.preference_score(sampled, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives()) + 0.45 * novelty
        return 0.45 * preference + 0.95 * uncertainty + 0.25 * novelty
    return preference + 0.35 * novelty + 0.25 * domination


def predict_pool(pool: np.ndarray, current: np.ndarray, demo) -> np.ndarray:
    if len(current) == 0:
        return np.array(pool, dtype=float)
    posterior = demo.RBFPosterior(np.array(demo.state.x, dtype=float), current)
    mean, _ = posterior.predict(pool)
    return mean


def reference_crowding(proxy: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.ones(len(proxy), dtype=float)
    center = np.mean(y, axis=0)
    dist = np.linalg.norm(y - center, axis=1)
    order = np.argsort(dist)
    crowd = np.zeros(len(y), dtype=float)
    crowd[order] = np.linspace(1.0, 0.1, len(y))
    return np.interp(np.linspace(0, len(crowd) - 1, len(proxy)), np.arange(len(crowd)), np.sort(crowd))


def hypervolume_contribution_proxy(y: np.ndarray, pareto_y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    baseline = np.max(pareto_y, axis=0) if len(pareto_y) else np.zeros(y.shape[1], dtype=float)
    return np.prod(np.clip(y - baseline, 0.0, 1.0) + 1e-3, axis=1)


def reference_alignment_score(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    directions = random_simplex(None, y.shape[1], len(y))
    return np.sum(y * directions, axis=1)


def box_novelty(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=float)
    return np.sum(np.abs(np.floor(y / 0.08) - np.floor(np.mean(y, axis=0) / 0.08)), axis=1)


def age_bias(pool: np.ndarray) -> np.ndarray:
    if len(pool) == 0:
        return np.array([], dtype=float)
    return np.linspace(0.9, 0.1, len(pool))


def random_simplex(demo, dimensions: int, n: int | None = None) -> np.ndarray:
    count = int(n or max(16, dimensions * 8))
    alpha = np.ones(dimensions, dtype=float)
    rng = demo.rng if demo is not None else np.random.default_rng(5)
    return rng.dirichlet(alpha, size=count)


for key, name, short_name, description, mode in [
    ("nsga3", "NSGA-III", "NSGA-III", "Reference-vector evolutionary baseline inspired by NSGA-III.", "nsga3"),
    ("spea2", "SPEA2", "SPEA2", "Strength-and-density evolutionary baseline inspired by SPEA2.", "spea2"),
    ("paes", "PAES", "PAES", "Pareto-archived evolution strategy baseline.", "paes"),
    ("gde3", "GDE3", "GDE3", "Differential-evolution based multi-objective baseline.", "gde3"),
    ("ibea", "IBEA", "IBEA", "Indicator-based evolutionary baseline.", "ibea"),
    ("smsemoa", "SMS-EMOA", "SMS-EMOA", "Hypervolume-contribution evolutionary baseline.", "smsemoa"),
    ("rvea", "RVEA", "RVEA", "Reference-vector guided evolutionary baseline.", "rvea"),
    ("mocell", "MOCell", "MOCell", "Cellular multi-objective evolutionary baseline.", "mocell"),
    ("epsilon_moea", "epsilon-MOEA", "e-MOEA", "Epsilon-box archive multi-objective baseline.", "epsilon_moea"),
    ("age_moea", "AGE-MOEA", "AGE", "Age-aware diversity-preserving multi-objective baseline.", "age_moea"),
    ("qnehvi_bo", "qNEHVI", "qNEHVI", "Expensive BO baseline using uncertainty-aware hypervolume improvement.", "qnehvi_bo"),
    ("qparego_bo", "qParEGO", "qParEGO", "Expensive BO baseline using randomized scalarization and uncertainty.", "qparego_bo"),
    ("thompson_bo", "Thompson MO-BO", "TS-BO", "Expensive BO baseline using Thompson-style posterior sampling.", "thompson_bo"),
    ("entropy_bo", "Entropy BO", "ENT-BO", "Expensive BO baseline using uncertainty-seeking exploration.", "entropy_bo"),
]:
    register_variant(
        key,
        name,
        short_name,
        description,
        mode,
    )

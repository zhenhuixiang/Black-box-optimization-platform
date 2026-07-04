from __future__ import annotations

import numpy as np

from benchmarks import evaluate_benchmark, materialize_benchmark
from methods.registry import register_optimizer


def run_lilo_bo(demo, batch_size: int) -> dict:
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")

    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    weights = demo.active_weights_array()
    constraints = demo.state.parsed.get("constraints", {})
    objectives = materialize_benchmark(demo.state.benchmark, demo.state.settings["dimension"], demo.state.settings.get("objectiveCount"))["objectives"]
    posterior = demo.RBFPosterior(x, y)

    hv_before = demo.approximate_hypervolume(y, np.array(demo.state.hv_samples, dtype=float))
    candidate_pool = candidate_pool_for(demo, x, y)
    mean, std = posterior.predict(candidate_pool)
    current_pref_best = float(demo.preference_score(y, weights, constraints, objectives).max())
    predicted_pref = demo.preference_score(mean, weights, constraints, objectives)
    uncertainty = np.linalg.norm(std, axis=1)
    strategy = select_strategy(demo, hv_before, y, weights, constraints)
    scores = acquisition_scores(demo, strategy, y, mean, std, predicted_pref, current_pref_best, uncertainty, weights, constraints)
    chosen_idx = demo.diverse_top_k(candidate_pool, scores, batch_size)
    chosen_x = candidate_pool[chosen_idx]
    chosen_y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in chosen_x])
    return demo.apply_round_results(
        method="LILO-BO",
        strategy=strategy,
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        predicted=mean[chosen_idx],
        uncertainty=uncertainty[chosen_idx],
        scores=scores[chosen_idx],
        hv_before=hv_before,
        candidate_uncertainty=uncertainty,
        surrogate=demo.surrogate_report(candidate_uncertainty=uncertainty, selected_uncertainty=uncertainty[chosen_idx]),
    )


def candidate_pool_for(demo, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    input_dim = demo.input_dimension()
    random_count = max(220, demo.state.settings["populationSize"] * 80)
    random_part = demo.rng.random((random_count, input_dim))
    pareto_x = x[demo.pareto_mask(y)]
    local_parts = []
    local_count = max(8, round(demo.state.settings["populationSize"] * 2.5))
    for base in pareto_x:
        local_parts.append(np.clip(base + demo.rng.normal(0.0, 0.09, size=(local_count, input_dim)), 0.0, 1.0))
    parts = [random_part, language_prior_candidates(demo, max(80, demo.state.settings["populationSize"] * 20))]
    if local_parts:
        parts.append(np.vstack(local_parts))
    pool = np.vstack(parts)
    _, idx = np.unique(np.round(pool, 4), axis=0, return_index=True)
    return pool[np.sort(idx)]


def language_prior_candidates(demo, n: int) -> np.ndarray:
    input_dim = demo.input_dimension()
    weights = demo.active_weights_dict()
    focus = max(weights, key=weights.get)
    centers = {
        "quality": np.array([0.72, 0.30, 0.25]),
        "speed": np.array([0.35, 0.80, 0.82]),
        "cost": np.array([0.20, 0.72, 0.68]),
    }
    balanced = np.array([0.48, 0.55, 0.52])
    center = np.resize(0.65 * centers[focus] + 0.35 * balanced, input_dim)
    return np.clip(center + demo.rng.normal(0.0, 0.18, size=(n, input_dim)), 0.0, 1.0)


def select_strategy(demo, hv: float, y: np.ndarray, weights: np.ndarray, constraints: dict[str, float]) -> str:
    pareto_count = int(demo.pareto_mask(y).sum())
    preference_index = {key: idx for idx, key in enumerate(demo.PREFERENCE_KEYS)}
    relevant_weights = np.array([weights[preference_index[key]] for key in demo.benchmark_preference_keys()], dtype=float)
    pref_strength = float(relevant_weights.max() - relevant_weights.min())
    feasible_rate = demo.feasible_rate(y, constraints)
    recent_events = " ".join(demo.state.events[:3])
    stalled = "hypervolume +0.000" in recent_events or "hypervolume +0.001" in recent_events
    if constraints and feasible_rate < 0.55:
        return "constraint-aware preference repair"
    if demo.state.iteration < 2 and pref_strength < 0.10:
        return "qNEHVI-style Pareto expansion"
    if pref_strength > 0.18:
        return "preference-weighted hypervolume improvement"
    if stalled or pareto_count < 4:
        return "uncertainty-guided Pareto exploration"
    if hv > 0.45:
        return "knee-region exploitation"
    return "qParEGO-style randomized scalarization"


def acquisition_scores(demo, strategy, observed_y, pred_mean, pred_std, predicted_pref, current_pref_best, uncertainty, weights, constraints):
    pareto_y = observed_y[demo.pareto_mask(observed_y)]
    domination_margin = demo.pareto_improvement_proxy(pred_mean, pareto_y)
    pref_improvement = np.maximum(0.0, predicted_pref - current_pref_best)
    balanced = pred_mean.min(axis=1)
    feasibility = demo.predicted_feasibility(pred_mean, pred_std, constraints)
    if strategy == "constraint-aware preference repair":
        scores = 1.35 * feasibility + 0.7 * pref_improvement + 0.22 * uncertainty
    elif strategy == "preference-weighted hypervolume improvement":
        scores = 0.9 * domination_margin + 1.4 * pref_improvement + 0.28 * uncertainty
    elif strategy == "uncertainty-guided Pareto exploration":
        scores = 0.55 * domination_margin + 1.2 * uncertainty + 0.25 * balanced
    elif strategy == "knee-region exploitation":
        scores = 0.7 * balanced + 0.7 * predicted_pref + 0.35 * domination_margin
    elif strategy == "qParEGO-style randomized scalarization":
        preference_index = {key: idx for idx, key in enumerate(demo.PREFERENCE_KEYS)}
        objectives = materialize_benchmark(demo.state.benchmark, demo.state.settings["dimension"], demo.state.settings.get("objectiveCount"))["objectives"]
        alpha = np.array([2.0 + 8.0 * weights[preference_index[obj.get("preferenceKey", "quality")]] for obj in objectives], dtype=float)
        scalar = pred_mean @ demo.rng.dirichlet(alpha)
        scores = scalar + 0.35 * uncertainty + 0.45 * domination_margin
    else:
        scores = domination_margin + 0.42 * uncertainty + 0.20 * balanced
    return np.asarray(scores * feasibility, dtype=float)


register_optimizer(
    "lilo_bo",
    {
        "name": "Pareto-aware LILO-BO",
        "shortName": "LILO-BO",
        "description": "Python backend: language-guided Bayesian optimization with an RBF surrogate, Pareto-aware acquisition, and uncertainty exploration.",
    },
    run_lilo_bo,
)

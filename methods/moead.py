from __future__ import annotations

import numpy as np

from benchmarks import evaluate_benchmark
from methods.registry import register_optimizer


def run_moead(demo, batch_size: int) -> dict:
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")
    y = np.array(demo.state.y, dtype=float)
    hv_before = demo.approximate_hypervolume(y, np.array(demo.state.hv_samples, dtype=float))
    archive_x, archive_y = demo.evolution_archive()
    weights = moead_weights(demo, batch_size)
    chosen_x = []
    chosen_y = []
    for weight in weights:
        x_next = propose_candidate(demo, archive_x, archive_y, weight, y)
        chosen_x.append(x_next)
        chosen_y.append(evaluate_benchmark(x_next, demo.state.benchmark))
    chosen_x_arr = np.array(chosen_x, dtype=float)
    chosen_y_arr = np.array(chosen_y, dtype=float)
    scores = demo.preference_score(chosen_y_arr, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    uncertainty = np.full(len(chosen_x_arr), 0.18, dtype=float)
    return demo.apply_round_results(
        method="MOEA/D",
        strategy="decomposition-based scalar subproblems",
        chosen_x=chosen_x_arr,
        chosen_y=chosen_y_arr,
        predicted=chosen_y_arr,
        uncertainty=uncertainty,
        scores=scores,
        hv_before=hv_before,
        surrogate={
            "model": "MOEA/D baseline",
            "containsUncertainty": False,
            "calibrated": False,
            "meanCandidateUncertainty": float(np.mean(uncertainty)),
            "selectedCandidateUncertainty": float(np.mean(uncertainty)),
            "note": "MOEA/D uses decomposition with deterministic scalar subproblem scoring in this demo; it does not use a calibrated surrogate uncertainty model.",
        },
    )


def moead_weights(demo, batch_size: int) -> np.ndarray:
    objectives = demo.objectives()
    active = demo.active_weights_dict()
    base = np.array([active.get(obj.get("preferenceKey", "quality"), 0.0) for obj in objectives], dtype=float)
    base = base / max(float(base.sum()), 1e-9)
    rows = []
    for i in range(batch_size):
        offset = i / max(1, batch_size - 1) - 0.5
        jitter = 0.12 + demo.rng.random(len(objectives)) * 0.45
        row = np.maximum(0.02, base + jitter * offset)
        rows.append(row / row.sum())
    return np.array(rows, dtype=float)


def propose_candidate(demo, archive_x: np.ndarray, archive_y: np.ndarray, weight: np.ndarray, reference_y: np.ndarray) -> np.ndarray:
    ideal = np.max(reference_y, axis=0)
    scores = -np.max(weight[None, :] * np.abs(ideal[None, :] - archive_y), axis=1)
    pref = demo.preference_score(archive_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    order = np.argsort(scores + 0.05 * pref)[::-1]
    first = archive_x[int(order[0])]
    second = archive_x[int(order[1])] if len(order) > 1 else archive_x[int(demo.rng.integers(0, len(archive_x)))]
    return demo.mutate_vector(demo.blend_vectors(first, second, spread=0.7), 0.12)


register_optimizer(
    "moead",
    {
        "name": "MOEA/D",
        "shortName": "MOEA/D",
        "description": "Python backend: decomposition baseline using preference-shaped weight vectors and Tchebycheff-style scalar subproblems.",
    },
    run_moead,
)

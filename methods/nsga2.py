from __future__ import annotations

import numpy as np

from benchmarks import evaluate_benchmark
from methods.registry import register_optimizer


def run_nsga2(demo, batch_size: int) -> dict:
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")
    hv_before = demo.approximate_hypervolume(np.array(demo.state.y, dtype=float), np.array(demo.state.hv_samples, dtype=float))
    archive_x, archive_y = demo.evolution_archive()
    offspring = []
    while len(offspring) < batch_size:
        a, b = demo.tournament_pair(archive_x, archive_y)
        offspring.append(demo.mutate_vector(demo.blend_vectors(archive_x[a], archive_x[b]), 0.16))
        if len(offspring) < batch_size:
            offspring.append(demo.mutate_vector(demo.blend_vectors(archive_x[b], archive_x[a]), 0.16))
    chosen_x = demo.dedupe_vectors(np.array(offspring, dtype=float))[:batch_size]
    while len(chosen_x) < batch_size:
        base = archive_x[int(demo.rng.integers(0, len(archive_x)))]
        chosen_x = np.vstack([chosen_x, demo.mutate_vector(base, 0.22)])
    chosen_y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in chosen_x])
    scores = demo.preference_score(chosen_y, demo.active_weights_array(), demo.state.parsed.get("constraints", {}), demo.objectives())
    uncertainty = np.full(len(chosen_x), 0.18, dtype=float)
    return demo.apply_round_results(
        method="NSGA-II",
        strategy="ask-tell offspring from non-dominated tournament parents",
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        predicted=chosen_y,
        uncertainty=uncertainty,
        scores=scores,
        hv_before=hv_before,
        surrogate={
            "model": "NSGA-II baseline",
            "containsUncertainty": False,
            "calibrated": False,
            "meanCandidateUncertainty": float(np.mean(uncertainty)),
            "selectedCandidateUncertainty": float(np.mean(uncertainty)),
            "note": "NSGA-II uses deterministic ranking plus a fixed mutation rate in this demo; no surrogate uncertainty is modeled.",
        },
    )


register_optimizer(
    "nsga2",
    {
        "name": "NSGA-II",
        "shortName": "NSGA-II",
        "description": "Python backend: evolutionary baseline using non-dominated sorting, crowding distance, crossover, and mutation.",
    },
    run_nsga2,
)

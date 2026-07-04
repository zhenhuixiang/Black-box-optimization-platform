from __future__ import annotations

import numpy as np

from benchmarks import evaluate_benchmark
from methods.registry import register_optimizer


def run_custom_random_local_search(demo, batch_size: int) -> dict:
    """Example ask-tell optimizer.

    The demo object owns the optimization state, RNG, benchmark key, settings,
    Pareto utilities, and the method used to append evaluated candidates.
    """
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")

    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    hv_before = demo.approximate_hypervolume(y, np.array(demo.state.hv_samples, dtype=float))
    pareto_x = x[demo.pareto_mask(y)]
    input_dim = demo.input_dimension()
    if len(pareto_x) == 0:
        base = demo.rng.random((batch_size, input_dim))
    else:
        parent_ids = demo.rng.integers(0, len(pareto_x), size=batch_size)
        base = pareto_x[parent_ids]

    chosen_x = np.clip(base + demo.rng.normal(0.0, 0.12, size=(batch_size, input_dim)), 0.0, 1.0)
    chosen_y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in chosen_x])
    predicted = chosen_y
    uncertainty = np.full(batch_size, 0.0)
    scores = chosen_y.mean(axis=1)
    return demo.apply_round_results(
        method="Custom RLS",
        strategy="Pareto-local random mutation",
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        predicted=predicted,
        uncertainty=uncertainty,
        scores=scores,
        hv_before=hv_before,
        surrogate=demo.surrogate_report(),
    )


register_optimizer(
    "custom_random_local_search",
    {
        "name": "Custom random local search",
        "shortName": "Custom RLS",
        "description": "Example user-defined optimizer: mutates sampled Pareto archive members.",
    },
    run_custom_random_local_search,
)

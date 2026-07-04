from __future__ import annotations

import numpy as np

from benchmarks.registry import input_defs, objective_defs, register_benchmark


def evaluate_custom_sphere_tradeoff(x: np.ndarray) -> np.ndarray:
    """Return normalized maximization objectives in [0, 1]."""
    x = np.asarray(x, dtype=float)
    f1 = 1.0 - np.mean((x - 0.20) ** 2) / 0.64
    f2 = 1.0 - np.mean((x - 0.80) ** 2) / 0.64
    f3 = 1.0 - abs(float(np.mean(x)) - 0.50) / 0.50
    return np.clip(np.array([f1, f2, f3], dtype=float), 0.0, 1.0)


def reference_front(dimension: int, n: int = 512) -> np.ndarray:
    """Optional reference front for IGD/IGD+; rows are objective vectors."""
    t = np.linspace(0.0, 1.0, n)
    designs = np.repeat(t[:, None], max(1, int(dimension)), axis=1)
    values = np.array([evaluate_custom_sphere_tradeoff(row) for row in designs])
    mask = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        others = np.delete(values, i, axis=0)
        if np.any(np.all(others >= point, axis=1) & np.any(others > point, axis=1)):
            mask[i] = False
    return values[mask]


register_benchmark(
    "custom_sphere_tradeoff",
    {
        "name": "Custom sphere tradeoff",
        "description": "Example user-defined 3-objective benchmark with normalized maximization scores.",
        "minDimension": 2,
        "defaultDimension": 5,
        "maxDimension": 30,
        "inputLabels": input_defs,
        "objectives": objective_defs(3),
        "evaluate": evaluate_custom_sphere_tradeoff,
        "referenceFront": reference_front,
    },
)

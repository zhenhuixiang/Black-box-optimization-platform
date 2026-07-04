from __future__ import annotations

import math

import numpy as np

from .registry import input_defs, minimize_to_score, objective_defs, register_benchmark


def dtlz_problem(x: np.ndarray, m: int, variant: int) -> np.ndarray:
    k = len(x) - m + 1
    tail = x[-k:]
    if variant in {1, 3}:
        g = 100.0 * (k + float(np.sum((tail - 0.5) ** 2 - np.cos(20.0 * math.pi * (tail - 0.5)))))
    elif variant == 6:
        g = float(np.sum(tail**0.1))
    else:
        g = float(np.sum((tail - 0.5) ** 2))

    values: list[float] = []
    if variant == 1:
        for i in range(m):
            value = 0.5 * (1.0 + g)
            for j in range(m - i - 1):
                value *= float(x[j])
            if i > 0:
                value *= 1.0 - float(x[m - i - 1])
            values.append(value)
        upper = 0.5 * (1.0 + 225.0 * k)
        return np.array([minimize_to_score(value, 0.0, upper) for value in values], dtype=float)

    if variant == 7:
        f = [float(x[i]) for i in range(m - 1)]
        g7 = 1.0 + 9.0 * float(np.mean(tail))
        h = m - sum((fi / (1.0 + g7)) * (1.0 + math.sin(3.0 * math.pi * fi)) for fi in f)
        values = f + [(1.0 + g7) * h]
        return np.array([minimize_to_score(value, 0.0, 1.0 if idx < m - 1 else 30.0) for idx, value in enumerate(values)], dtype=float)

    alpha = 100.0 if variant == 4 else 1.0
    theta = np.asarray(x[: m - 1], dtype=float)
    if variant == 5:
        theta = np.array([theta[0], *[(1.0 + 2.0 * g * t) / (2.0 * (1.0 + g)) for t in theta[1:]]], dtype=float)
    if variant == 6:
        theta = np.array([theta[0], *[(1.0 + 2.0 * g * t) / (2.0 * (1.0 + g)) for t in theta[1:]]], dtype=float)

    for i in range(m):
        value = 1.0 + g
        for j in range(m - i - 1):
            value *= math.cos(float(theta[j]) ** alpha * math.pi / 2.0)
        if i > 0:
            value *= math.sin(float(theta[m - i - 1]) ** alpha * math.pi / 2.0)
        values.append(value)
    upper = 1.0 + (225.0 * k if variant == 3 else 20.0 if variant == 6 else 0.25 * k)
    return np.array([minimize_to_score(value, 0.0, upper) for value in values], dtype=float)


def register_dtlz(key: str, variant: int, default_dimension: int, description: str) -> None:
    register_benchmark(
        key,
        {
            "name": f"Classic {key.replace('_classic', '').upper()}",
            "description": description,
            "minDimension": 3,
            "defaultDimension": default_dimension,
            "inputLabels": input_defs,
            "objectives": objective_defs(3),
            "evaluate": lambda x, selected=variant: dtlz_problem(x, 3, selected),
        },
    )


register_dtlz("dtlz1_classic", 1, 7, "Three-objective linear Pareto front from the DTLZ benchmark suite, transformed from minimization to normalized maximization scores.")
register_dtlz("dtlz2_classic", 2, 12, "Three-objective concave Pareto surface from the DTLZ benchmark suite, transformed to maximization scores.")
register_dtlz("dtlz3_classic", 3, 12, "Three-objective multimodal DTLZ problem with many local fronts, transformed to maximization scores.")
register_dtlz("dtlz4_classic", 4, 12, "Three-objective biased DTLZ4 problem with a concave Pareto front, transformed to maximization scores.")
register_dtlz("dtlz5_classic", 5, 12, "Three-objective degenerate DTLZ5 problem, transformed to normalized maximization scores.")
register_dtlz("dtlz6_classic", 6, 12, "Three-objective DTLZ6 problem with nonlinear distance term, transformed to maximization scores.")
register_dtlz("dtlz7_classic", 7, 22, "Three-objective disconnected DTLZ7 problem, transformed to normalized maximization scores.")

from __future__ import annotations

import math

import numpy as np

from .registry import input_defs, minimize_to_score, objective_defs, register_benchmark


def zdt_problem(x: np.ndarray, variant: int) -> np.ndarray:
    f1 = float(x[0])
    tail = x[1:]
    if variant == 4:
        raw_tail = -5.0 + 10.0 * tail
        g = 1.0 + 10.0 * len(raw_tail) + float(np.sum(raw_tail**2 - 10.0 * np.cos(4.0 * math.pi * raw_tail)))
    elif variant == 6:
        f1 = 1.0 - math.exp(-4.0 * f1) * (math.sin(6.0 * math.pi * f1) ** 6)
        g = 1.0 + 9.0 * (float(np.sum(tail)) / max(1, len(tail))) ** 0.25
    else:
        g = 1.0 + 9.0 * float(np.sum(tail)) / max(1, len(tail))
    ratio = f1 / g
    if variant == 2:
        h = 1.0 - ratio**2
    elif variant == 3:
        h = 1.0 - math.sqrt(ratio) - ratio * math.sin(10.0 * math.pi * f1)
    elif variant == 6:
        h = 1.0 - ratio**2
    else:
        h = 1.0 - math.sqrt(ratio)
    f2 = g * h
    f2_lower = -1.0 if variant == 3 else 0.0
    f2_upper = 220.0 if variant == 4 else 10.0
    return np.array([minimize_to_score(f1, 0.0, 1.0), minimize_to_score(f2, f2_lower, f2_upper)], dtype=float)


def register_zdt(key: str, variant: int, description: str) -> None:
    register_benchmark(
        key,
        {
            "name": f"Classic {key.upper()}",
            "description": description,
            "minDimension": 2,
            "defaultDimension": 30,
            "inputLabels": input_defs,
            "objectives": objective_defs(2),
            "evaluate": lambda x, selected=variant: zdt_problem(x, selected),
        },
    )


register_zdt("zdt1", 1, "Two-objective convex Pareto front from the ZDT benchmark suite, transformed from minimization to normalized maximization scores.")
register_zdt("zdt2", 2, "Two-objective nonconvex Pareto front from the ZDT benchmark suite, shown as maximization scores.")
register_zdt("zdt3", 3, "Two-objective disconnected Pareto front from the ZDT benchmark suite, shown as maximization scores.")
register_zdt("zdt4", 4, "Two-objective multimodal ZDT problem with many local fronts, transformed to normalized maximization scores.")
register_zdt("zdt6", 6, "Two-objective nonuniform ZDT problem with biased density near the Pareto front, transformed to maximization scores.")

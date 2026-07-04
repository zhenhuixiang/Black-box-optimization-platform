from __future__ import annotations

import math

import numpy as np

from .registry import input_defs, objective_defs, register_benchmark


def unit(x: np.ndarray, length: int | None = None) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    if length is not None and len(z) < length:
        z = np.pad(z, (0, length - len(z)), constant_values=0.5)
    return np.clip(z, 0.0, 1.0)


def resize(x: np.ndarray, length: int) -> np.ndarray:
    return np.resize(unit(x), length)


def score(raw: float, scale: float = 1.0) -> float:
    return float(1.0 / (1.0 + max(0.0, float(raw)) / max(scale, 1e-9)))


def spherical_shape(head: np.ndarray, m: int, alpha: float = 1.0) -> np.ndarray:
    theta = np.resize(np.clip(head, 0.0, 1.0), max(1, m - 1)) ** alpha
    values: list[float] = []
    for i in range(m):
        value = 1.0
        for j in range(m - i - 1):
            value *= math.cos(float(theta[j]) * math.pi / 2.0)
        if i > 0:
            value *= math.sin(float(theta[m - i - 1]) * math.pi / 2.0)
        values.append(value)
    return np.array(values, dtype=float)


def mixed_shape(head: np.ndarray, m: int, variant: int) -> np.ndarray:
    shape = spherical_shape(head, m, alpha=1.0 + 0.08 * (variant % 5))
    if variant % 4 == 1:
        shape = np.sqrt(np.maximum(shape, 0.0))
    elif variant % 4 == 2:
        shape = np.abs(shape - 0.24 * np.sin((variant + 1) * math.pi * shape))
    elif variant % 4 == 3:
        shape = np.maximum(0.0, 1.0 - shape)
    return shape


def register_series(prefix: str, label: str, count: int, default_dimension: int, min_dimension: int, max_dimension: int, objectives: int, evaluator_factory, description_factory) -> None:
    for variant in range(1, count + 1):
        register_benchmark(
            f"{prefix}{variant:02d}",
            {
                "name": f"{label} {variant}",
                "description": description_factory(variant),
                "minDimension": min_dimension,
                "defaultDimension": default_dimension,
                "maxDimension": max_dimension,
                "inputLabels": input_defs,
                "objectives": objective_defs(objectives),
                "evaluate": evaluator_factory(variant),
            },
        )


def maf_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        m = 3 + (1 if variant in {4, 8, 12} else 0)
        z = resize(x, max(m + 5, len(np.asarray(x, dtype=float))))
        head = z[: m - 1]
        tail = z[m - 1 :]
        phase = 0.037 * variant
        if variant in {2, 6, 10, 14}:
            g = float(np.mean(np.abs(tail - 0.5)) + 0.08 * np.mean(np.cos(8.0 * math.pi * (tail + phase)) ** 2))
        elif variant in {3, 7, 11}:
            g = float(np.mean((tail - 0.5) ** 2) + 0.14 * np.mean(np.sin((variant + 1) * math.pi * tail) ** 2))
        elif variant in {5, 9, 13}:
            g = float(np.mean((tail - 0.5 - 0.15 * np.sin(2.0 * math.pi * (tail + phase))) ** 2))
        else:
            g = float(np.mean((tail - 0.5) ** 2))
        raw_shape = mixed_shape(head, m, variant)
        if variant in {6, 12}:
            raw_shape[-1] = abs(raw_shape[-1] - 0.35 * math.sin(5.0 * math.pi * float(np.mean(head)) + phase))
        if variant in {7, 13}:
            raw_shape = np.maximum(0.02, raw_shape * (0.72 + 0.28 * np.sin((variant + 2) * math.pi * raw_shape) ** 2))
        if variant in {8, 14}:
            penalty = 0.25 * max(0.0, 0.30 - float(np.mean(head))) ** 2 + 0.25 * max(0.0, float(np.mean(tail)) - 0.72) ** 2
        else:
            penalty = 0.0
        raw = (1.0 + 2.0 * g + penalty) * (1.0 - np.clip(raw_shape, 0.0, 1.0))
        return np.array([score(v, 1.4 + 0.04 * variant) for v in raw], dtype=float)

    return evaluate


def lsmop_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        m = 3
        z = resize(x, max(20, len(np.asarray(x, dtype=float))))
        head = z[:2]
        groups = np.array_split(z[2:], m)
        g_values = []
        for idx, group in enumerate(groups):
            if len(group) == 0:
                group = np.array([0.5], dtype=float)
            phase = 0.03 * variant + idx * 0.17
            if variant % 4 == 0:
                value = np.mean((group - 0.5) ** 2 + 0.05 * np.cos(10.0 * math.pi * (group + phase)) ** 2)
            elif variant % 4 == 1:
                value = np.mean(np.abs(group - 0.45 - 0.12 * np.sin(2.0 * math.pi * (group + phase))))
            elif variant % 4 == 2:
                value = np.mean((group - np.roll(group, 1)) ** 2 + (group - 0.5) ** 2)
            else:
                value = np.mean((group - 0.5) ** 2) + 0.08 * np.mean(np.sin((idx + variant) * math.pi * group) ** 2)
            g_values.append(float(value))
        shape = spherical_shape(head, m, alpha=1.0 + 0.03 * variant)
        if variant in {5, 10}:
            shape = np.maximum(0.02, 1.0 - shape)
        raw = np.array([(1.0 + 3.0 * g_values[i]) * (1.0 - shape[i]) for i in range(m)], dtype=float)
        return np.array([score(v, 1.6 + 0.05 * variant) for v in raw], dtype=float)

    return evaluate


def imop_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        z = resize(x, max(10, len(np.asarray(x, dtype=float))))
        x1 = float(z[0])
        x2 = float(z[1])
        tail = z[2:] if len(z) > 2 else np.array([0.5], dtype=float)
        phase = 0.061 * variant
        interval = 0.5 + 0.5 * math.sin((variant + 1) * math.pi * x1)
        g = float(np.mean((tail - interval * 0.5) ** 2) + 0.05 * np.mean(np.cos(6.0 * math.pi * (tail + phase)) ** 2))
        raw1 = x1 + 1.5 * g + 0.08 * math.sin((variant + 2) * math.pi * x2) ** 2
        raw2 = 1.0 - math.sqrt(max(0.0, x1)) + 1.8 * g + 0.08 * math.cos((variant + 3) * math.pi * x2) ** 2
        if variant % 3 == 0:
            raw2 += 0.25 * abs(math.sin(5.0 * math.pi * x1))
        return np.array([score(raw1, 1.8), score(raw2, 1.8)], dtype=float)

    return evaluate


def cdtlz_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        m = 3
        z = resize(x, max(12, len(np.asarray(x, dtype=float))))
        tail = z[m - 1 :]
        g = float(np.mean((tail - 0.5) ** 2))
        shape = spherical_shape(z[: m - 1], m, alpha=1.0 + 0.1 * (variant % 4))
        raw = (1.0 + g) * (1.0 - shape)
        radius = float(np.linalg.norm(shape - 0.5))
        penalty = 0.0
        if variant % 2 == 0:
            penalty += 2.0 * max(0.0, 0.28 - radius) ** 2
        else:
            penalty += 2.0 * max(0.0, float(np.sum(shape)) - 1.35) ** 2
        if variant in {3, 6, 9}:
            penalty += 0.3 * max(0.0, 0.45 - float(np.mean(tail))) ** 2
        return np.array([score(v + penalty, 1.4) for v in raw], dtype=float)

    return evaluate


register_series(
    "maf",
    "MaF-style",
    14,
    24,
    3,
    120,
    3,
    maf_evaluator,
    lambda variant: f"MaF-style many-objective benchmark variant {variant}, normalized for maximization.",
)

register_series(
    "lsmop",
    "LSMOP-style",
    10,
    50,
    20,
    300,
    3,
    lsmop_evaluator,
    lambda variant: f"Large-scale multi-objective benchmark variant {variant}, with grouped distance variables.",
)

register_series(
    "imop",
    "IMOP-style",
    8,
    20,
    3,
    120,
    2,
    imop_evaluator,
    lambda variant: f"Irregular Pareto-front benchmark variant {variant}, with interval and disconnected-front behavior.",
)

register_series(
    "cdtlz",
    "Constrained DTLZ-style",
    9,
    12,
    3,
    80,
    3,
    cdtlz_evaluator,
    lambda variant: f"Constrained DTLZ-style benchmark variant {variant}, represented with soft penalties and normalized objective values.",
)

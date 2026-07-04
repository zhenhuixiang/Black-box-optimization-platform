from __future__ import annotations

import math

import numpy as np

from .registry import CURRENT_OBJECTIVE_COUNT, input_defs, objective_defs, register_benchmark


def unit(x: np.ndarray, length: int | None = None) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    if length is not None and len(z) < length:
        z = np.pad(z, (0, length - len(z)), constant_values=0.5)
    return np.clip(z, 0.0, 1.0)


def score(loss: np.ndarray | float, scale: float = 1.0) -> np.ndarray:
    raw = np.asarray(loss, dtype=float)
    return np.clip(1.0 / (1.0 + np.maximum(raw, 0.0) / max(scale, 1e-9)), 0.0, 1.0)


def active_objectives(default: int, maximum: int) -> int:
    value = CURRENT_OBJECTIVE_COUNT.get()
    if value in {None, ""}:
        return default
    return int(np.clip(int(value), 2, maximum))


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
    return np.asarray(values, dtype=float)


def grouped_tail(z: np.ndarray, m: int, variant: int) -> np.ndarray:
    tail = z[m - 1 :] if len(z) >= m else np.array([0.5], dtype=float)
    groups = np.array_split(tail, m)
    values = []
    for idx, group in enumerate(groups):
        if len(group) == 0:
            group = np.array([0.5], dtype=float)
        phase = 0.031 * variant + 0.17 * idx
        if variant % 5 == 0:
            values.append(float(np.mean((group - 0.5) ** 2)))
        elif variant % 5 == 1:
            values.append(float(np.mean(np.abs(group - 0.45 - 0.12 * np.sin(2.0 * math.pi * (group + phase))))))
        elif variant % 5 == 2:
            values.append(float(np.mean((group - np.roll(group, 1)) ** 2 + 0.03 * np.cos(8.0 * math.pi * group) ** 2)))
        elif variant % 5 == 3:
            values.append(float(np.mean((group - 0.5) ** 2 + 0.08 * np.sin((idx + variant + 1) * math.pi * group) ** 2)))
        else:
            values.append(float(np.mean(np.maximum(0.0, group - 0.72) ** 2 + np.maximum(0.0, 0.22 - group) ** 2)))
    return np.asarray(values, dtype=float)


def make_evaluator(family: str, variant: int, default_m: int, max_m: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        m = active_objectives(default_m, max_m)
        z = unit(x, max(m + 4, len(np.asarray(x, dtype=float))))
        head = z[: max(1, m - 1)]
        tail_loss = grouped_tail(z, m, variant)
        phase = 0.047 * variant
        shape = spherical_shape(head, m, alpha=0.75 + 0.04 * (variant % 7))

        if family == "convex":
            loss = (1.0 - shape) ** (1.15 + 0.02 * variant) + 2.4 * tail_loss
        elif family == "concave":
            loss = np.sqrt(np.maximum(1.0 - shape, 0.0)) + 1.8 * tail_loss
        elif family == "disconnected":
            gaps = 0.22 * (np.sin((variant + 2) * math.pi * shape + phase) > 0.35).astype(float)
            loss = np.abs(1.0 - shape + gaps) + 2.1 * tail_loss
        elif family == "degenerate":
            compressed = 0.72 * shape + 0.28 * np.mean(shape)
            compressed[2:] = 0.55 * compressed[2:] + 0.45 * compressed[: len(compressed[2:])]
            loss = np.abs(1.0 - compressed) + 1.9 * tail_loss
        elif family == "sparse":
            active = max(1, int(round(math.sqrt(len(z)))))
            mask = np.zeros(len(z), dtype=float)
            mask[:active] = 1.0
            sparse_penalty = float(np.mean(((z - 0.5) * (1.0 - mask)) ** 2))
            loss = np.abs(1.0 - shape) + 2.2 * tail_loss + sparse_penalty
        elif family == "constrained":
            radius = float(np.linalg.norm(head - 0.5))
            band = 0.18 + 0.025 * (variant % 6)
            violation = max(0.0, band - radius) ** 2 + max(0.0, float(np.mean(z)) - 0.72) ** 2
            loss = np.abs(1.0 - shape) + 1.7 * tail_loss + 3.0 * violation
        elif family == "robust":
            noise_floor = 0.04 * np.sin((variant + 1) * math.pi * z[:m] + phase) ** 2
            loss = np.abs(1.0 - shape) + 1.5 * tail_loss + noise_floor
        elif family == "knee":
            knee = np.exp(-30.0 * (head[0] - (0.22 + 0.04 * (variant % 8))) ** 2)
            loss = np.abs(1.0 - shape) + 2.0 * tail_loss - 0.22 * knee
        else:
            loss = np.abs(1.0 - shape) + 2.0 * tail_loss
        return score(loss, 1.35 + 0.02 * variant)

    return evaluate


def register_family(prefix: str, family: str, label: str, count: int, default_dimension: int, default_m: int, max_m: int, description: str) -> None:
    for variant in range(1, count + 1):
        register_benchmark(
            f"{prefix}{variant:02d}",
            {
                "name": f"{label} {variant}",
                "description": f"{description} Variant {variant}.",
                "minDimension": max(2, default_m),
                "defaultDimension": default_dimension,
                "maxDimension": max(80, default_dimension * 4),
                "minObjectives": 2,
                "defaultObjectives": default_m,
                "maxObjectives": max_m,
                "inputLabels": input_defs,
                "objectives": objective_defs(default_m),
                "evaluate": make_evaluator(family, variant, default_m, max_m),
            },
        )


register_family("bbm_convex", "convex", "BBM Convex Pareto", 8, 16, 3, 10, "Smooth convex-front synthetic benchmark for baseline convergence testing.")
register_family("bbm_concave", "concave", "BBM Concave Pareto", 8, 16, 3, 10, "Concave-front synthetic benchmark with separable and grouped distance variables.")
register_family("bbm_disc", "disconnected", "BBM Disconnected Pareto", 8, 18, 3, 8, "Disconnected-front synthetic benchmark for archive diversity stress tests.")
register_family("bbm_degen", "degenerate", "BBM Degenerate Pareto", 6, 20, 4, 12, "Degenerate many-objective benchmark where objectives share latent structure.")
register_family("bbm_sparse", "sparse", "BBM Sparse Large-scale", 6, 60, 3, 10, "Large-scale sparse-variable benchmark for variable-selection behavior.")
register_family("bbm_const", "constrained", "BBM Constrained Pareto", 8, 18, 3, 8, "Soft-constrained benchmark with infeasible bands and disconnected feasible regions.")
register_family("bbm_robust", "robust", "BBM Robust Design", 6, 14, 3, 10, "Robust-design benchmark with deterministic ripple penalties.")
register_family("bbm_knee", "knee", "BBM Knee Preference", 6, 14, 3, 10, "Knee-point benchmark for preference-aware and recommendation-driven optimizers.")

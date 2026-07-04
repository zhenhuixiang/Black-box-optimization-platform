from __future__ import annotations

import math

import numpy as np

from .registry import clamp, register_benchmark


def fit_vector(x: np.ndarray, length: int) -> np.ndarray:
    return np.resize(np.asarray(x, dtype=float), length)


def ml_system_tuning(x: np.ndarray) -> np.ndarray:
    depth, compression, cache = fit_vector(x, 3)
    quality = (
        0.58
        + 0.28 * math.exp(-((depth - 0.72) ** 2) / 0.045)
        + 0.13 * math.exp(-((compression - 0.28) ** 2) / 0.025)
        - 0.18 * (compression - 0.72) ** 2
        - 0.08 * cache
        + 0.03 * math.sin(6.0 * math.pi * depth)
    )
    speed = 0.34 + 0.48 * compression + 0.22 * cache - 0.30 * depth + 0.06 * math.sin(5.0 * math.pi * cache + 0.5) - 0.05 * abs(compression - cache)
    cost = 0.78 - 0.55 * depth + 0.22 * compression + 0.10 * cache - 0.11 * (cache - 0.6) ** 2 + 0.04 * math.cos(4.0 * math.pi * compression)
    return np.array([clamp(quality), clamp(speed), clamp(cost)], dtype=float)


def dtlz2_like(x: np.ndarray) -> np.ndarray:
    angle_a, angle_b, shape = fit_vector(x, 3)
    scale = 1.0 + (shape - 0.5) ** 2
    f1 = scale * math.cos(angle_a * math.pi / 2.0) * math.cos(angle_b * math.pi / 2.0)
    f2 = scale * math.cos(angle_a * math.pi / 2.0) * math.sin(angle_b * math.pi / 2.0)
    f3 = scale * math.sin(angle_a * math.pi / 2.0)
    return np.clip(np.array([f1, f2, f3], dtype=float) / 1.25, 0.0, 1.0)


def vehicle_safety_like(x: np.ndarray) -> np.ndarray:
    rail, bumper, cabin = fit_vector(x, 3)
    crash_quality = 0.42 + 0.38 * rail + 0.22 * cabin - 0.16 * bumper - 0.12 * (rail - cabin) ** 2 + 0.04 * math.sin(5.0 * math.pi * bumper)
    speed_proxy = 0.82 - 0.36 * rail - 0.30 * cabin + 0.16 * bumper - 0.08 * abs(rail - 0.35)
    material_efficiency = 0.84 - 0.45 * rail - 0.34 * cabin - 0.10 * bumper + 0.08 * math.exp(-((bumper - 0.65) ** 2) / 0.035)
    return np.array([clamp(crash_quality), clamp(speed_proxy), clamp(material_efficiency)], dtype=float)


def rugged_tradeoff(x: np.ndarray) -> np.ndarray:
    x1, x2, x3 = fit_vector(x, 3)
    quality = 0.48 + 0.22 * math.sin(3.0 * math.pi * x1) + 0.16 * math.cos(4.0 * math.pi * x2) + 0.22 * math.exp(-((x1 - 0.72) ** 2 + (x2 - 0.28) ** 2) / 0.035) - 0.10 * x3
    speed = 0.44 + 0.22 * math.sin(3.0 * math.pi * x2 + 0.8) + 0.26 * math.exp(-((x2 - 0.75) ** 2 + (x3 - 0.65) ** 2) / 0.04) + 0.12 * x3 - 0.16 * x1
    cost = 0.72 - 0.28 * x1 + 0.18 * math.cos(5.0 * math.pi * x3) + 0.18 * math.exp(-((x1 - 0.22) ** 2 + (x3 - 0.72) ** 2) / 0.03) - 0.10 * x2
    return np.array([clamp(quality), clamp(speed), clamp(cost)], dtype=float)


register_benchmark(
    "ml_system",
    {
        "name": "ML system tuning",
        "description": "Three-way tradeoff among quality, serving speed, and resource efficiency.",
        "minDimension": 3,
        "defaultDimension": 3,
        "inputs": [{"key": "depth", "label": "Model depth"}, {"key": "compression", "label": "Compression"}, {"key": "cache", "label": "Cache"}],
        "objectives": [
            {"key": "accuracy", "label": "Accuracy value", "goal": "max", "preferenceKey": "quality", "description": "Prediction quality, normalized so higher is better."},
            {"key": "throughput", "label": "Throughput value", "goal": "max", "preferenceKey": "speed", "description": "Serving speed proxy, normalized so higher is better."},
            {"key": "efficiency", "label": "Resource efficiency value", "goal": "max", "preferenceKey": "cost", "description": "Cost/resource efficiency, normalized so higher is better."},
        ],
        "evaluate": ml_system_tuning,
    },
)

register_benchmark(
    "dtlz2",
    {
        "name": "DTLZ2-like smooth front",
        "description": "A smooth concave Pareto surface adapted from the DTLZ2 family.",
        "minDimension": 3,
        "defaultDimension": 3,
        "inputs": [{"key": "angle_a", "label": "Angle A"}, {"key": "angle_b", "label": "Angle B"}, {"key": "shape", "label": "Shape"}],
        "objectives": [
            {"key": "f1", "label": "Objective f1 value", "goal": "max", "preferenceKey": "quality", "description": "First normalized DTLZ2-like objective."},
            {"key": "f2", "label": "Objective f2 value", "goal": "max", "preferenceKey": "speed", "description": "Second normalized DTLZ2-like objective."},
            {"key": "f3", "label": "Objective f3 value", "goal": "max", "preferenceKey": "cost", "description": "Third normalized DTLZ2-like objective."},
        ],
        "evaluate": dtlz2_like,
    },
)

register_benchmark(
    "vehicle",
    {
        "name": "Vehicle safety-like",
        "description": "Engineering-style tradeoff among crash quality, speed proxy, and material efficiency.",
        "minDimension": 3,
        "defaultDimension": 3,
        "inputs": [{"key": "rail", "label": "Rail thickness"}, {"key": "bumper", "label": "Bumper stiffness"}, {"key": "cabin", "label": "Cabin reinforcement"}],
        "objectives": [
            {"key": "crash_safety", "label": "Crash safety value", "goal": "max", "preferenceKey": "quality", "description": "Passenger/crash safety value."},
            {"key": "speed_proxy", "label": "Speed proxy value", "goal": "max", "preferenceKey": "speed", "description": "A lightweight speed proxy."},
            {"key": "material_efficiency", "label": "Material efficiency value", "goal": "max", "preferenceKey": "cost", "description": "Material/cost efficiency value."},
        ],
        "evaluate": vehicle_safety_like,
    },
)

register_benchmark(
    "rugged",
    {
        "name": "Rugged multimodal tradeoff",
        "description": "A nonconvex landscape with several local Pareto regions.",
        "minDimension": 3,
        "defaultDimension": 3,
        "inputs": [{"key": "x1", "label": "Design x1"}, {"key": "x2", "label": "Design x2"}, {"key": "x3", "label": "Design x3"}],
        "objectives": [
            {"key": "mode_a", "label": "Mode A value", "goal": "max", "preferenceKey": "quality", "description": "First rugged objective."},
            {"key": "mode_b", "label": "Mode B value", "goal": "max", "preferenceKey": "speed", "description": "Second rugged objective."},
            {"key": "mode_c", "label": "Mode C value", "goal": "max", "preferenceKey": "cost", "description": "Third rugged objective."},
        ],
        "evaluate": rugged_tradeoff,
    },
)

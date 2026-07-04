from __future__ import annotations

import contextvars
import math
from typing import Any, Callable

import numpy as np


PREFERENCE_KEYS = ["quality", "speed", "cost"]
BENCHMARKS: dict[str, dict[str, Any]] = {}
CURRENT_OBJECTIVE_COUNT: contextvars.ContextVar[int | None] = contextvars.ContextVar("objective_count", default=None)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))


def input_defs(count: int, prefix: str = "x") -> list[dict[str, str]]:
    return [{"key": f"{prefix}{i + 1}", "label": f"{prefix}{i + 1}"} for i in range(count)]


def objective_defs(count: int, ranges: list[tuple[float, float]] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "key": f"f{i + 1}_value",
            "label": f"f{i + 1} value",
            "goal": "max",
            "preferenceKey": PREFERENCE_KEYS[i % len(PREFERENCE_KEYS)],
            "valueRange": [float(ranges[i][0]), float(ranges[i][1])] if ranges and i < len(ranges) else [0.0, 1.0],
            "description": "Classic benchmark objective transformed to a normalized maximization value.",
        }
        for i in range(count)
    ]


def objective_count_bounds(definition: dict[str, Any]) -> tuple[int, int, int, int]:
    native = len(definition.get("objectives", []) or [])
    native = int(definition.get("defaultObjectives") or native or 2)
    minimum = int(definition.get("minObjectives") or 2)
    maximum = int(definition.get("maxObjectives") or max(15, native))
    default = int(definition.get("defaultObjectives") or native)
    minimum = max(1, minimum)
    maximum = max(minimum, maximum)
    default = int(np.clip(default, minimum, maximum))
    native = int(np.clip(native, minimum, maximum))
    return minimum, default, maximum, native


def sanitize_objective_count(definition: dict[str, Any], objective_count: int | None = None) -> int:
    minimum, default, maximum, _native = objective_count_bounds(definition)
    if objective_count in {None, ""}:
        return default
    try:
        value = int(round(float(objective_count)))
    except (TypeError, ValueError):
        value = default
    return int(np.clip(value, minimum, maximum))


def benchmark_objectives(definition: dict[str, Any], objective_count: int | None = None) -> list[dict[str, Any]]:
    count = sanitize_objective_count(definition, objective_count)
    native = list(definition.get("objectives", []) or [])
    if len(native) >= count:
        return native[:count]
    generated = objective_defs(count)
    return native + generated[len(native) :]


def register_benchmark(key: str, definition: dict[str, Any]) -> None:
    if not key or "evaluate" not in definition or not callable(definition["evaluate"]):
        raise ValueError(f"Invalid benchmark registration: {key}")
    inputs = definition.get("inputs", [])
    default_dimension = int(definition.get("defaultDimension") or len(inputs) or definition.get("minDimension") or 1)
    min_dimension = int(definition.get("minDimension") or len(inputs) or 1)
    max_dimension = definition.get("maxDimension")
    min_objectives, default_objectives, max_objectives, native_objectives = objective_count_bounds(definition)
    BENCHMARKS[key] = {
        **definition,
        "key": key,
        "minDimension": min_dimension,
        "defaultDimension": default_dimension,
        "minObjectives": min_objectives,
        "defaultObjectives": default_objectives,
        "maxObjectives": max_objectives,
        "nativeObjectives": native_objectives,
        **({"maxDimension": int(max_dimension)} if max_dimension is not None else {}),
    }


def resolve_benchmark(benchmark: str = "ml_system") -> dict[str, Any]:
    if benchmark in BENCHMARKS:
        return BENCHMARKS[benchmark]
    if "ml_system" in BENCHMARKS:
        return BENCHMARKS["ml_system"]
    return next(iter(BENCHMARKS.values()))


def benchmark_inputs(definition: dict[str, Any], dimension: int | None = None) -> list[dict[str, str]]:
    d = max(
        int(definition.get("minDimension", 1)),
        int(dimension or definition.get("defaultDimension") or len(definition.get("inputs", [])) or 1),
    )
    if definition.get("maxDimension") is not None:
        d = min(d, int(definition["maxDimension"]))
    input_labels: Callable[[int], list[dict[str, str]]] | None = definition.get("inputLabels")
    if input_labels:
        return input_labels(d)
    inputs = definition.get("inputs", [])
    if len(inputs) == d:
        return inputs
    return input_defs(d)


def materialize_benchmark(benchmark: str = "ml_system", dimension: int | None = None, objective_count: int | None = None) -> dict[str, Any]:
    definition = resolve_benchmark(benchmark)
    inputs = benchmark_inputs(definition, dimension)
    objectives = benchmark_objectives(definition, objective_count)
    min_objectives, default_objectives, max_objectives, native_objectives = objective_count_bounds(definition)
    return {
        key: value
        for key, value in {
            **definition,
            "inputs": inputs,
            "objectives": objectives,
            "minDimension": int(definition.get("minDimension") or len(inputs) or 1),
            "defaultDimension": int(definition.get("defaultDimension") or len(inputs) or 1),
            "minObjectives": min_objectives,
            "defaultObjectives": default_objectives,
            "maxObjectives": max_objectives,
            "nativeObjectives": native_objectives,
            **({"maxDimension": int(definition["maxDimension"])} if definition.get("maxDimension") is not None else {}),
        }.items()
        if key not in {"evaluate", "inputLabels", "referenceFront"}
    }


def adapt_objective_values(values: np.ndarray, x: np.ndarray, objective_count: int) -> np.ndarray:
    raw = np.ravel(np.asarray(values, dtype=float))
    if len(raw) == 0:
        raw = np.array([0.0], dtype=float)
    raw = np.clip(raw, 0.0, 1.0)
    if len(raw) >= objective_count:
        return raw[:objective_count]
    z = np.ravel(np.asarray(x, dtype=float))
    if len(z) == 0:
        z = np.array([0.5], dtype=float)
    extras: list[float] = []
    while len(raw) + len(extras) < objective_count:
        idx = len(raw) + len(extras)
        a = float(raw[idx % len(raw)])
        b = float(raw[(idx + 1) % len(raw)])
        feature = float(z[idx % len(z)])
        ripple = 0.5 + 0.5 * math.sin((idx + 1) * math.pi * feature)
        extras.append(float(np.clip(0.52 * a + 0.34 * (1.0 - b) + 0.14 * ripple, 0.0, 1.0)))
    return np.concatenate([raw, np.asarray(extras, dtype=float)])


def evaluate_benchmark(x: np.ndarray, benchmark: str = "ml_system", objective_count: int | None = None) -> np.ndarray:
    definition = resolve_benchmark(benchmark)
    count = sanitize_objective_count(definition, objective_count if objective_count is not None else CURRENT_OBJECTIVE_COUNT.get())
    row = np.asarray(x, dtype=float)
    return adapt_objective_values(np.asarray(definition["evaluate"](row), dtype=float), row, count)


def minimize_to_score(value: float, lower: float, upper: float) -> float:
    return clamp((upper - value) / max(1e-9, upper - lower))

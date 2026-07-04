from __future__ import annotations

from typing import Any


SINGLE_BENCHMARKS: dict[str, dict[str, Any]] = {}


def register_single_benchmark(key: str, definition: dict[str, Any]) -> None:
    if not key or "evaluate" not in definition or not callable(definition["evaluate"]):
        raise ValueError(f"Invalid single-objective benchmark registration: {key}")
    default_dimension = int(definition.get("defaultDimension") or definition.get("minDimension") or 1)
    min_dimension = int(definition.get("minDimension") or 1)
    max_dimension = int(definition.get("maxDimension") or max(default_dimension, min_dimension))
    bounds = definition.get("bounds", [-5.0, 5.0])
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
        bounds = [-5.0, 5.0]
    SINGLE_BENCHMARKS[key] = {
        **definition,
        "key": key,
        "minDimension": min_dimension,
        "defaultDimension": default_dimension,
        "maxDimension": max_dimension,
        "bounds": [float(bounds[0]), float(bounds[1])],
        "optimum": float(definition.get("optimum", 0.0) or 0.0),
    }

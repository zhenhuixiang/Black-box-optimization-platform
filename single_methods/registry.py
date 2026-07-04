from __future__ import annotations

from typing import Any, Callable


SingleMethodRunner = Callable[[Any, int], tuple[Any, str]]
SINGLE_METHODS: dict[str, dict[str, str]] = {}
SINGLE_RUNNERS: dict[str, SingleMethodRunner] = {}


def register_single_method(key: str, metadata: dict[str, str], runner: SingleMethodRunner) -> None:
    if not key or not callable(runner):
        raise ValueError(f"Invalid single-objective method registration: {key}")
    SINGLE_METHODS[key] = metadata
    SINGLE_RUNNERS[key] = runner

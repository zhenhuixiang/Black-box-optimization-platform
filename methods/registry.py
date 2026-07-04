from __future__ import annotations

from typing import Any, Callable


OptimizerRunner = Callable[[Any, int], dict[str, Any]]
OPTIMIZERS: dict[str, dict[str, str]] = {}
RUNNERS: dict[str, OptimizerRunner] = {}


def register_optimizer(key: str, metadata: dict[str, str], runner: OptimizerRunner) -> None:
    if not key or not callable(runner):
        raise ValueError(f"Invalid optimizer registration: {key}")
    OPTIMIZERS[key] = metadata
    RUNNERS[key] = runner


def run_optimizer(demo: Any, batch_size: int) -> dict[str, Any]:
    runner = RUNNERS.get(demo.state.optimizer) or RUNNERS.get("lilo_bo")
    if runner is None:
        raise RuntimeError("No optimizer runner is registered.")
    return runner(demo, batch_size)

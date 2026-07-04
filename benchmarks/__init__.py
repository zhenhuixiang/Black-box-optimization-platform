from __future__ import annotations

from .registry import (
    BENCHMARKS,
    PREFERENCE_KEYS,
    evaluate_benchmark,
    input_defs,
    materialize_benchmark,
    minimize_to_score,
    objective_defs,
    register_benchmark,
)

from . import catalog_extra as _catalog_extra  # noqa: F401
from . import catalog_pack as _catalog_pack  # noqa: F401
from . import classic as _classic  # noqa: F401
from . import dtlz as _dtlz  # noqa: F401
from . import extended_suite as _extended_suite  # noqa: F401
from . import synthetic as _synthetic  # noqa: F401
from . import zdt as _zdt  # noqa: F401

__all__ = [
    "BENCHMARKS",
    "PREFERENCE_KEYS",
    "evaluate_benchmark",
    "input_defs",
    "materialize_benchmark",
    "minimize_to_score",
    "objective_defs",
    "register_benchmark",
]

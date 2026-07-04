from __future__ import annotations

from .registry import SINGLE_BENCHMARKS, register_single_benchmark

from . import classic as _classic  # noqa: F401
from . import extended as _extended  # noqa: F401

__all__ = ["SINGLE_BENCHMARKS", "register_single_benchmark"]

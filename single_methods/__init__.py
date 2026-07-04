from __future__ import annotations

from .registry import SINGLE_METHODS, SINGLE_RUNNERS, register_single_method

from . import classic as _classic  # noqa: F401
from . import extended as _extended  # noqa: F401

__all__ = ["SINGLE_METHODS", "SINGLE_RUNNERS", "register_single_method"]

from __future__ import annotations

from .registry import OPTIMIZERS, RUNNERS, register_optimizer, run_optimizer

from . import classic_bo as _classic_bo  # noqa: F401
from . import catalog_methods as _catalog_methods  # noqa: F401
from . import expanded as _expanded  # noqa: F401
from . import extended_strategies as _extended_strategies  # noqa: F401
from . import lilo_bo as _lilo_bo  # noqa: F401
from . import moead as _moead  # noqa: F401
from . import nsga2 as _nsga2  # noqa: F401
from . import z_method_catalog as _z_method_catalog  # noqa: F401

__all__ = ["OPTIMIZERS", "RUNNERS", "register_optimizer", "run_optimizer"]

from __future__ import annotations

import ast
import hashlib
import json
import importlib
import importlib.util
import math
import mimetypes
import os
import pkgutil
import random
import re
import sys
import time
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from benchmarks.registry import BENCHMARKS, CURRENT_OBJECTIVE_COUNT, PREFERENCE_KEYS, adapt_objective_values, evaluate_benchmark, materialize_benchmark
from methods.registry import OPTIMIZERS, RUNNERS, run_optimizer
from single_benchmarks.registry import SINGLE_BENCHMARKS as REGISTERED_SINGLE_BENCHMARKS
from single_methods.registry import (
    SINGLE_METHODS as REGISTERED_SINGLE_METHODS,
    SINGLE_RUNNERS as REGISTERED_SINGLE_RUNNERS,
)


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
ARGS = [arg for arg in sys.argv[1:] if arg]
OPEN_BROWSER = "--open" in ARGS
PORT_ARGS = [arg for arg in ARGS if arg != "--open"]
PORT = int(os.environ.get("LILO_PORT") or (PORT_ARGS[0] if PORT_ARGS else 8765))
MODULE_SIGNATURE: tuple[tuple[str, int], ...] | None = None
REGISTRY_ERRORS: list[str] = []
EXTERNAL_MODULE_PATHS: dict[str, list[Path]] = {"benchmark": [], "optimizer": []}
EXPERIMENT_LIMITS = {
    "maxRuns": 120,
    "maxRepeats": 30,
    "maxBudget": 1000,
    "maxPopulation": 100,
    "maxWorkers": 8,
}

ALLOWED_CUSTOM_IMPORTS = {
    "__future__",
    "benchmarks",
    "benchmarks.registry",
    "math",
    "methods.registry",
    "numpy",
    "random",
    "statistics",
    "typing",
}
ALLOWED_CUSTOM_IMPORT_PREFIXES = {"numpy"}
FORBIDDEN_CUSTOM_IMPORT_ROOTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shlex",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "urllib",
}
FORBIDDEN_CUSTOM_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_CUSTOM_ATTRS = {
    "chmod",
    "connect",
    "dump",
    "dumps",
    "exec",
    "load",
    "loads",
    "mkdir",
    "open",
    "popen",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "run",
    "save",
    "savez",
    "spawn",
    "system",
    "unlink",
    "write",
    "write_text",
    "write_bytes",
}

METRIC_DIRECTIONS = {
    "hypervolume": 1,
    "recommendedScore": 1,
    "paretoCount": 1,
    "igd": -1,
    "igdPlus": -1,
    "gd": -1,
    "elapsedMs": -1,
    "bestValue": -1,
    "optimumGap": -1,
    "meanValue": -1,
}

METRIC_LABELS = {
    "hypervolume": "Hypervolume",
    "recommendedScore": "Recommended preference score",
    "paretoCount": "Pareto count",
    "igd": "IGD",
    "igdPlus": "IGD+",
    "gd": "Generational distance",
    "elapsedMs": "Elapsed time",
    "bestValue": "Best objective value",
    "optimumGap": "Optimum gap",
    "meanValue": "Mean observed value",
}


def registry_sources() -> list[Path]:
    folders = [ROOT / "benchmarks", ROOT / "methods", ROOT / "single_benchmarks", ROOT / "single_methods"]
    local_sources = [
        path
        for folder in folders
        for path in folder.glob("*.py")
        if path.name not in {"__init__.py", "registry.py"}
    ]
    external_sources = [
        path
        for paths in EXTERNAL_MODULE_PATHS.values()
        for path in paths
        if path.exists() and path.is_file()
    ]
    return [*local_sources, *external_sources]


def current_module_signature() -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(path), path.stat().st_mtime_ns) for path in registry_sources()))


def reload_registry_package(package_name: str, registry: dict[str, Any]) -> None:
    package = importlib.import_module(package_name)
    registry.clear()
    if package_name == "methods":
        RUNNERS.clear()
    if package_name == "single_methods":
        REGISTERED_SINGLE_RUNNERS.clear()
    importlib.invalidate_caches()
    for module_info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        if module_info.name in {"registry"} or module_info.name.startswith("_"):
            continue
        module_name = f"{package_name}.{module_info.name}"
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
        except Exception as exc:
            REGISTRY_ERRORS.append(f"{module_name}: {exc}")


def load_external_module(kind: str, path: Path) -> None:
    package_name = "benchmarks" if kind == "benchmark" else "methods"
    module_stem = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_") or f"external_{kind}"
    module_name = f"{package_name}.external_{abs(hash(str(path.resolve()))) & 0xffffffff:x}_{module_stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def refresh_registries_if_needed(force: bool = False) -> bool:
    global MODULE_SIGNATURE, REGISTRY_ERRORS
    signature = current_module_signature()
    if not force and MODULE_SIGNATURE == signature:
        return False
    REGISTRY_ERRORS = []
    reload_registry_package("benchmarks", BENCHMARKS)
    reload_registry_package("methods", OPTIMIZERS)
    reload_registry_package("single_benchmarks", REGISTERED_SINGLE_BENCHMARKS)
    reload_registry_package("single_methods", REGISTERED_SINGLE_METHODS)
    for kind, paths in EXTERNAL_MODULE_PATHS.items():
        valid_paths: list[Path] = []
        for path in paths:
            if not path.exists() or not path.is_file():
                REGISTRY_ERRORS.append(f"{path}: external {kind} path no longer exists")
                continue
            try:
                load_external_module(kind, path)
                valid_paths.append(path)
            except Exception as exc:
                REGISTRY_ERRORS.append(f"{path}: {exc}")
        EXTERNAL_MODULE_PATHS[kind] = valid_paths
    if "sync_single_objective_registries" in globals():
        sync_single_objective_registries()
    MODULE_SIGNATURE = current_module_signature()
    suffix = f", {len(REGISTRY_ERRORS)} import error(s)" if REGISTRY_ERRORS else ""
    single_suffix = ""
    if "SINGLE_BENCHMARKS" in globals() and "SINGLE_METHODS" in globals():
        single_suffix = f", {len(SINGLE_BENCHMARKS)} single functions, {len(SINGLE_METHODS)} single methods"
    print(f"[{time.strftime('%H:%M:%S')}] Reloaded registries: {len(BENCHMARKS)} benchmarks, {len(OPTIMIZERS)} optimizers{single_suffix}{suffix}.")
    return True


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    samples = np.zeros((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.random(n)) / n
    return samples


def is_dominated(point: np.ndarray, others: np.ndarray) -> bool:
    return bool(np.any(np.all(others >= point, axis=1) & np.any(others > point, axis=1)))


def pareto_mask(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=bool)
    mask = np.ones(len(y), dtype=bool)
    for i, point in enumerate(y):
        if is_dominated(point, np.delete(y, i, axis=0)):
            mask[i] = False
    return mask


def approximate_hypervolume(y: np.ndarray, samples: np.ndarray, reference: np.ndarray | None = None) -> float:
    if len(y) == 0:
        return 0.0
    reference = np.zeros(y.shape[1], dtype=float) if reference is None else reference
    dominated = np.zeros(len(samples), dtype=bool)
    for point in np.maximum(y, reference):
        dominated |= np.all(point >= samples, axis=1)
    return float(dominated.mean() * np.prod(1.0 - reference))


def stable_seed(text: str) -> int:
    return int(sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 1_000_003)


@lru_cache(maxsize=256)
def reference_front_for(benchmark: str, dimension: int, objective_count: int | None = None, sample_count: int = 4096) -> np.ndarray:
    definition = BENCHMARKS.get(benchmark)
    if not definition:
        return np.empty((0, 0), dtype=float)
    materialized = materialize_benchmark(benchmark, dimension, objective_count)
    objective_dim = len(materialized["objectives"])
    reference_front = definition.get("referenceFront")
    if callable(reference_front):
        try:
            front = np.asarray(reference_front(dimension, sample_count), dtype=float)
            if front.ndim == 2 and len(front):
                if front.shape[1] != objective_dim:
                    front = np.array([adapt_objective_values(row, row, objective_dim) for row in front], dtype=float)
                if front.shape[1] == objective_dim:
                    return np.clip(front, 0.0, 1.0)
        except Exception:
            pass

    rng = np.random.default_rng(stable_seed(f"{benchmark}:{dimension}:{objective_dim}:{sample_count}"))
    x = rng.random((sample_count, dimension))
    try:
        token = CURRENT_OBJECTIVE_COUNT.set(objective_dim)
        try:
            y = np.array([evaluate_benchmark(row, benchmark, objective_dim) for row in x], dtype=float)
        finally:
            CURRENT_OBJECTIVE_COUNT.reset(token)
    except Exception:
        return np.empty((0, objective_dim), dtype=float)
    front = np.clip(y[pareto_mask(y)], 0.0, 1.0)
    if len(front) > 512:
        order = np.argsort(front[:, 0])
        keep = np.linspace(0, len(order) - 1, 512).round().astype(int)
        front = front[order[keep]]
    return front


def distance_metrics(y: np.ndarray, benchmark: str, dimension: int, objective_count: int | None = None) -> dict[str, float]:
    if len(y) == 0:
        return {"igd": 1.0, "igdPlus": 1.0, "gd": 1.0}
    approximation = np.clip(y[pareto_mask(y)], 0.0, 1.0)
    reference = reference_front_for(benchmark, dimension, objective_count or y.shape[1])
    if len(approximation) == 0 or len(reference) == 0:
        return {"igd": 1.0, "igdPlus": 1.0, "gd": 1.0}

    diff_ref_to_approx = reference[:, None, :] - approximation[None, :, :]
    igd = float(np.mean(np.min(np.linalg.norm(diff_ref_to_approx, axis=2), axis=1)))
    igd_plus = float(np.mean(np.min(np.linalg.norm(np.maximum(diff_ref_to_approx, 0.0), axis=2), axis=1)))
    diff_approx_to_ref = approximation[:, None, :] - reference[None, :, :]
    gd = float(np.mean(np.min(np.linalg.norm(diff_approx_to_ref, axis=2), axis=1)))
    return {"igd": igd, "igdPlus": igd_plus, "gd": gd}


def preference_score(
    y: np.ndarray,
    weights: np.ndarray,
    constraints: dict[str, float],
    objectives: list[dict[str, Any]],
) -> np.ndarray:
    y = np.atleast_2d(y)
    preference_index = {key: i for i, key in enumerate(PREFERENCE_KEYS)}
    objective_weights = np.array(
        [weights[preference_index[obj.get("preferenceKey", "quality")]] for obj in objectives],
        dtype=float,
    )
    objective_weights = objective_weights / max(float(objective_weights.sum()), 1e-9)
    score = y @ objective_weights
    for key, threshold in constraints.items():
        for idx, obj in enumerate(objectives):
            if obj.get("preferenceKey") == key:
                score -= 2.5 * np.maximum(0.0, threshold - y[:, idx])
    return score


def numeric_setting(raw: dict[str, Any], name: str, fallback: int) -> int:
    value = raw.get(name, fallback)
    if value in {None, ""}:
        return int(fallback)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(fallback)


def parse_language_feedback(text: str) -> dict[str, Any]:
    lower = text.lower()
    weights = np.array([0.34, 0.33, 0.33], dtype=float)
    constraints: dict[str, float] = {}
    notes: list[str] = []
    signals = {
        "quality": ["quality", "accuracy", "accurate", "performance", "safety", "robust"],
        "speed": ["speed", "latency", "fast", "faster", "quick", "delay", "throughput"],
        "cost": ["cost", "cheap", "budget", "memory", "small", "efficient", "resource"],
    }
    for idx, key in enumerate(PREFERENCE_KEYS):
        count = sum(lower.count(token) for token in signals[key])
        if count:
            weights[idx] += 0.22 * count
            notes.append(f"Raised {key} weight from language cues ({count}).")

    threshold_terms = {
        "quality": r"(?:quality|accuracy|performance|safety)",
        "speed": r"(?:speed|latency|throughput)",
        "cost": r"(?:cost|resource|memory|efficiency)",
    }
    threshold_cue = r"(?:>=|>|at\s+least|above|over|no\s+less\s+than)"
    for key, term in threshold_terms.items():
        patterns = [
            rf"{term}\s*(?:should\s+be\s*)?{threshold_cue}\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            rf"{threshold_cue}\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*(?:for\s*)?{term}",
        ]
        match = next((m for pat in patterns if (m := re.search(pat, lower))), None)
        if match:
            value = float(match.group(1))
            constraints[key] = float(np.clip(value / 100.0 if value > 1.0 else value, 0.05, 0.95))
            notes.append(f"Detected {key} target around {constraints[key]:.2f}.")

    if "balanced" in lower or "balance" in lower or "tradeoff" in lower:
        weights = 0.5 * weights + 0.5
        notes.append("Pulled weights toward a balanced tradeoff.")

    weights = np.maximum(weights, 0.05)
    weights = weights / weights.sum()
    if not notes:
        notes.append("No strong preference cue found; using a balanced Pareto search.")
    return {
        "weights": {key: float(weights[i]) for i, key in enumerate(PREFERENCE_KEYS)},
        "constraints": constraints,
        "notes": notes,
        "summary": summarize_preferences(weights, constraints),
    }


def summarize_preferences(weights: np.ndarray, constraints: dict[str, float]) -> str:
    top = PREFERENCE_KEYS[int(np.argmax(weights))]
    floors = ", ".join(f"{k} >= {v:.2f}" for k, v in constraints.items()) if constraints else "no hard floors"
    return f"Preference focus: {top}; {floors}"


@dataclass
class RBFPosterior:
    x: np.ndarray
    y: np.ndarray
    lengthscale: float = 0.28
    noise: float = 1e-4
    clip_mean: bool = True
    alpha: np.ndarray = field(init=False)
    inv_kernel: np.ndarray = field(init=False)
    y_mean: np.ndarray = field(init=False)
    y_std: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.y_mean = self.y.mean(axis=0)
        self.y_std = np.maximum(self.y.std(axis=0), 0.06)
        y_scaled = (self.y - self.y_mean) / self.y_std
        kernel = self._kernel(self.x, self.x) + np.eye(len(self.x)) * self.noise
        try:
            self.inv_kernel = np.linalg.inv(kernel)
        except np.linalg.LinAlgError:
            self.inv_kernel = np.linalg.pinv(kernel)
        self.alpha = self.inv_kernel @ y_scaled

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * sqdist / (self.lengthscale**2))

    def predict(self, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k_star = self._kernel(np.atleast_2d(x_new), self.x)
        mean_scaled = k_star @ self.alpha
        raw_var = 1.0 - np.sum((k_star @ self.inv_kernel) * k_star, axis=1)
        raw_var = np.clip(raw_var, 0.0025, 1.0)
        mean = mean_scaled * self.y_std + self.y_mean
        if self.clip_mean:
            mean = np.clip(mean, 0.0, 1.0)
        std = np.sqrt(raw_var)[:, None] * self.y_std[None, :]
        return mean, std


@dataclass
class DemoState:
    seed: int = 7
    benchmark: str = "ml_system"
    optimizer: str = "lilo_bo"
    settings: dict[str, int] = field(default_factory=lambda: {"populationSize": 9, "maxEvaluations": 60, "dimension": 3})
    x: list[list[float]] = field(default_factory=list)
    y: list[list[float]] = field(default_factory=list)
    feedback: str = "I want a balanced design, but quality should remain strong."
    parsed: dict[str, Any] = field(default_factory=dict)
    manual_weights: dict[str, float] | None = None
    hv_samples: list[list[float]] = field(default_factory=list)
    iteration: int = 0
    last_strategy: str = "initial"
    last_candidates: list[dict[str, Any]] = field(default_factory=list)
    surrogate_report_data: dict[str, Any] = field(default_factory=dict)
    experiment_result: dict[str, Any] | None = None
    events: list[str] = field(default_factory=list)


class ParetoLiloDemo:
    PREFERENCE_KEYS = PREFERENCE_KEYS
    RBFPosterior = RBFPosterior
    pareto_mask = staticmethod(pareto_mask)
    approximate_hypervolume = staticmethod(approximate_hypervolume)
    preference_score = staticmethod(preference_score)

    def __init__(self, auto_reset: bool = True) -> None:
        self.rng = np.random.default_rng(7)
        self.state = DemoState()
        if auto_reset:
            self.reset()

    def sync_registries(self) -> None:
        if not refresh_registries_if_needed():
            return
        if self.state.benchmark not in BENCHMARKS or self.state.optimizer not in OPTIMIZERS:
            benchmark = self.state.benchmark if self.state.benchmark in BENCHMARKS else next(iter(BENCHMARKS))
            optimizer = self.state.optimizer if self.state.optimizer in OPTIMIZERS else next(iter(OPTIMIZERS))
            self.reset(self.state.seed, self.state.feedback, benchmark, optimizer, self.state.settings)
            return
        self.state.events.insert(0, f"Reloaded backend registry: {len(BENCHMARKS)} benchmarks, {len(OPTIMIZERS)} optimizers.")
        self.state.events = self.state.events[:8]

    def sanitize_settings(self, raw: dict[str, Any] | None = None, benchmark: str | None = None) -> dict[str, int]:
        raw = raw or {}
        key = benchmark or self.state.benchmark
        def number(name: str, fallback: int) -> float:
            value = raw.get(name, fallback)
            if value in {None, ""}:
                return float(fallback)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(fallback)

        definition = materialize_benchmark(key, int(number("dimension", self.state.settings.get("dimension", 3))), raw.get("objectiveCount"))
        min_dimension = int(definition.get("minDimension") or 1)
        max_dimension = int(definition.get("maxDimension") or 200)
        default_dimension = int(definition.get("defaultDimension") or len(definition["inputs"]))
        min_objectives = int(definition.get("minObjectives") or 2)
        max_objectives = int(definition.get("maxObjectives") or 15)
        default_objectives = int(definition.get("defaultObjectives") or len(definition["objectives"]) or 2)
        dimension = int(np.clip(round(number("dimension", default_dimension)), min_dimension, max_dimension))
        objective_count = int(np.clip(round(number("objectiveCount", self.state.settings.get("objectiveCount", default_objectives))), min_objectives, max_objectives))
        population = int(np.clip(round(number("populationSize", self.state.settings.get("populationSize", 9))), 1, 200))
        max_evaluations = int(np.clip(round(number("maxEvaluations", self.state.settings.get("maxEvaluations", 60))), population, 5000))
        return {"populationSize": population, "maxEvaluations": max_evaluations, "dimension": dimension, "objectiveCount": objective_count}

    def definition(self) -> dict[str, Any]:
        return materialize_benchmark(self.state.benchmark, self.state.settings["dimension"], self.state.settings.get("objectiveCount"))

    def input_dimension(self) -> int:
        return len(self.definition()["inputs"])

    def objectives(self) -> list[dict[str, Any]]:
        return self.definition()["objectives"]

    def reset(
        self,
        seed: int = 7,
        feedback: str | None = None,
        benchmark: str | None = None,
        optimizer: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        benchmark = benchmark if benchmark in BENCHMARKS else "ml_system"
        optimizer = optimizer if optimizer in OPTIMIZERS else next(iter(OPTIMIZERS))
        clean_settings = self.sanitize_settings(settings, benchmark)
        definition = materialize_benchmark(benchmark, clean_settings["dimension"], clean_settings.get("objectiveCount"))
        input_dim = len(definition["inputs"])
        objective_dim = len(definition["objectives"])
        self.rng = np.random.default_rng(seed)
        initial_x = latin_hypercube(clean_settings["populationSize"], input_dim, self.rng)
        token = CURRENT_OBJECTIVE_COUNT.set(objective_dim)
        try:
            initial_y = np.array([evaluate_benchmark(row, benchmark, objective_dim) for row in initial_x])
            self.state = DemoState(seed=seed, benchmark=benchmark, optimizer=optimizer, settings=clean_settings)
            self.state.x = initial_x.tolist()
            self.state.y = initial_y.tolist()
            self.state.feedback = feedback or self.state.feedback
            self.state.parsed = parse_language_feedback(self.state.feedback)
            self.state.hv_samples = self.rng.random((10000, objective_dim)).tolist()
            self.state.last_strategy = f"{OPTIMIZERS[optimizer]['shortName']} initialization"
            self.state.surrogate_report_data = self.surrogate_report()
            self.state.events = [
                f"Initialized {clean_settings['populationSize']} backend evaluations on {definition['name']}.",
                f"Experiment budget: {clean_settings['maxEvaluations']} max evaluations, d={input_dim}, M={objective_dim}.",
                f"Selected optimizer: {OPTIMIZERS[optimizer]['name']}.",
                "Parsed the first natural-language preference message.",
            ]
            return self.snapshot()
        finally:
            CURRENT_OBJECTIVE_COUNT.reset(token)

    def update_feedback(self, feedback: str) -> dict[str, Any]:
        self.state.feedback = feedback.strip() or self.state.feedback
        self.state.parsed = parse_language_feedback(self.state.feedback)
        self.state.manual_weights = None
        self.state.events.insert(0, "Updated language preference model.")
        self.state.events = self.state.events[:8]
        return self.snapshot()

    def update_weights(self, weights: dict[str, Any]) -> dict[str, Any]:
        relevant = self.benchmark_preference_keys()
        current = self.active_weights_dict()
        for key in relevant:
            if key in weights:
                current[key] = max(0.01, float(weights[key]))
        total = sum(current[key] for key in relevant) or 1.0
        self.state.manual_weights = {key: float(current.get(key, 0.0) / total) if key in relevant else 0.0 for key in PREFERENCE_KEYS}
        self.state.events.insert(0, "Manually adjusted preference weights.")
        self.state.events = self.state.events[:8]
        return self.snapshot()

    def active_weights_dict(self) -> dict[str, float]:
        base = self.state.manual_weights or self.state.parsed.get("weights") or {key: 1.0 / len(PREFERENCE_KEYS) for key in PREFERENCE_KEYS}
        relevant = self.benchmark_preference_keys()
        total = sum(base.get(key, 0.0) for key in relevant) or 1.0
        return {key: float(base.get(key, 0.0) / total) if key in relevant else 0.0 for key in PREFERENCE_KEYS}

    def active_weights_array(self) -> np.ndarray:
        weights = self.active_weights_dict()
        return np.array([weights[key] for key in PREFERENCE_KEYS], dtype=float)

    def benchmark_preference_keys(self) -> list[str]:
        return list(dict.fromkeys(obj.get("preferenceKey", "quality") for obj in self.objectives()))

    def budget_remaining(self) -> int:
        return max(0, self.state.settings["maxEvaluations"] - len(self.state.x))

    def budgeted_batch_size(self, requested: int | None = None) -> int:
        size = requested if requested is not None else self.state.settings["populationSize"]
        return max(0, min(max(1, int(size)), self.budget_remaining()))

    def step(self, batch_size: int | None = None) -> dict[str, Any]:
        actual = self.budgeted_batch_size(batch_size)
        if actual <= 0:
            return self.finish_experiment(0, 0, "Budget already reached.")
        before = len(self.state.x)
        token = CURRENT_OBJECTIVE_COUNT.set(len(self.objectives()))
        try:
            result = run_optimizer(self, actual)
            if len(self.state.x) <= before:
                return self.finish_experiment(0, 0, "Optimizer made no progress.")
            return result
        finally:
            CURRENT_OBJECTIVE_COUNT.reset(token)

    def run_to_budget(self) -> dict[str, Any]:
        started = time.perf_counter()
        rounds = 0
        self.state.experiment_result = None
        while self.budget_remaining() > 0:
            before = len(self.state.x)
            self.step(min(self.state.settings["populationSize"], self.budget_remaining()))
            rounds += 1
            if len(self.state.x) <= before:
                return self.finish_experiment(rounds, int((time.perf_counter() - started) * 1000), "Budget run stopped: optimizer made no progress.")
        return self.finish_experiment(rounds, int((time.perf_counter() - started) * 1000), "Budget run completed.")

    def finish_experiment(self, rounds: int, elapsed_ms: int, note: str) -> dict[str, Any]:
        y = np.array(self.state.y, dtype=float)
        mask = pareto_mask(y)
        scores = preference_score(y, self.active_weights_array(), self.state.parsed.get("constraints", {}), self.objectives())
        best_index = int(np.argmax(scores)) if len(scores) else -1
        self.state.experiment_result = {
            "note": note,
            "rounds": rounds,
            "elapsedMs": elapsed_ms,
            "evaluations": len(self.state.x),
            "maxEvaluations": self.state.settings["maxEvaluations"],
            "paretoCount": int(mask.sum()),
            "hypervolume": approximate_hypervolume(y, np.array(self.state.hv_samples, dtype=float)),
            "recommendedId": best_index,
            "recommendedScore": float(scores[best_index]) if best_index >= 0 else 0.0,
        }
        self.state.events.insert(0, f"{note} {len(self.state.x)}/{self.state.settings['maxEvaluations']} evaluations, {int(mask.sum())} Pareto points.")
        self.state.events = self.state.events[:8]
        return self.snapshot()

    def apply_round_results(
        self,
        method: str,
        strategy: str,
        chosen_x: np.ndarray,
        chosen_y: np.ndarray,
        predicted: np.ndarray,
        uncertainty: np.ndarray,
        scores: np.ndarray,
        hv_before: float,
        surrogate: dict[str, Any],
        candidate_uncertainty: np.ndarray | None = None,
    ) -> dict[str, Any]:
        n = min(len(chosen_x), self.budget_remaining())
        if n <= 0:
            return self.finish_experiment(0, 0, "Budget already reached.")
        chosen_x = np.asarray(chosen_x[:n], dtype=float)
        chosen_y = np.asarray(chosen_y[:n], dtype=float)
        predicted = np.asarray(predicted[:n], dtype=float)
        uncertainty = np.asarray(uncertainty[:n], dtype=float)
        scores = np.asarray(scores[:n], dtype=float)
        self.state.x.extend(chosen_x.tolist())
        self.state.y.extend(chosen_y.tolist())
        self.state.iteration += 1
        self.state.last_strategy = f"{method}: {strategy}"
        self.state.last_candidates = [
            {
                "x": [float(v) for v in chosen_x[i]],
                "predicted": [float(v) for v in predicted[i]],
                "observed": [float(v) for v in chosen_y[i]],
                "uncertainty": float(uncertainty[i]),
                "score": float(scores[i]),
            }
            for i in range(n)
        ]
        self.state.surrogate_report_data = surrogate
        hv_after = approximate_hypervolume(np.array(self.state.y, dtype=float), np.array(self.state.hv_samples, dtype=float))
        self.state.events.insert(0, f"Round {self.state.iteration}: {self.state.last_strategy}; hypervolume +{hv_after - hv_before:.4f}.")
        self.state.events = self.state.events[:8]
        return self.snapshot()

    def set_benchmark(self, benchmark: str) -> dict[str, Any]:
        definition = materialize_benchmark(benchmark, None, None)
        settings = {**self.state.settings, "dimension": definition["defaultDimension"], "objectiveCount": definition.get("defaultObjectives", len(definition["objectives"]))}
        return self.reset(random.randint(1, 99999), self.state.feedback, benchmark, self.state.optimizer, settings)

    def set_optimizer(self, optimizer: str) -> dict[str, Any]:
        return self.reset(random.randint(1, 99999), self.state.feedback, self.state.benchmark, optimizer, self.state.settings)

    def set_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return self.reset(random.randint(1, 99999), self.state.feedback, self.state.benchmark, self.state.optimizer, settings)

    def surrogate_report(self, candidate_uncertainty: np.ndarray | None = None, selected_uncertainty: np.ndarray | None = None) -> dict[str, Any]:
        report = {
            "model": "Fixed-lengthscale RBF posterior proxy",
            "containsUncertainty": True,
            "calibrated": False,
            "meanCandidateUncertainty": None,
            "selectedCandidateUncertainty": None,
            "note": "The Python backend computes an RBF posterior variance proxy. It supports exploration, but it is not a fully calibrated GP fitted by marginal likelihood.",
        }
        if candidate_uncertainty is not None and len(candidate_uncertainty):
            report["meanCandidateUncertainty"] = float(np.mean(candidate_uncertainty))
        if selected_uncertainty is not None and len(selected_uncertainty):
            report["selectedCandidateUncertainty"] = float(np.mean(selected_uncertainty))
        return report

    def feasible_rate(self, y: np.ndarray, constraints: dict[str, float]) -> float:
        if not constraints:
            return 1.0
        mask = np.ones(len(y), dtype=bool)
        for key, threshold in constraints.items():
            for idx, obj in enumerate(self.objectives()):
                if obj.get("preferenceKey") == key:
                    mask &= y[:, idx] >= threshold
        return float(mask.mean())

    def predicted_feasibility(self, pred_mean: np.ndarray, pred_std: np.ndarray, constraints: dict[str, float]) -> np.ndarray:
        if not constraints:
            return np.ones(len(pred_mean), dtype=float)
        feasibility = np.ones(len(pred_mean), dtype=float)
        for key, threshold in constraints.items():
            for idx, obj in enumerate(self.objectives()):
                if obj.get("preferenceKey") == key:
                    z = (pred_mean[:, idx] - threshold) / np.maximum(pred_std[:, idx], 0.04)
                    feasibility *= 1.0 / (1.0 + np.exp(-1.6 * z))
        return feasibility

    def pareto_improvement_proxy(self, pred_mean: np.ndarray, pareto_y: np.ndarray) -> np.ndarray:
        if len(pareto_y) == 0:
            return pred_mean.mean(axis=1)
        diff = pred_mean[:, None, :] - pareto_y[None, :, :]
        return np.max(np.maximum(diff, 0.0).sum(axis=2) - 0.35 * np.maximum(-diff, 0.0).sum(axis=2), axis=1)

    def diverse_top_k(self, pool: np.ndarray, scores: np.ndarray, k: int) -> list[int]:
        order = list(np.argsort(scores)[::-1])
        chosen: list[int] = []
        for idx in order:
            if not chosen or np.linalg.norm(pool[idx] - pool[chosen], axis=1).min() > 0.12:
                chosen.append(int(idx))
            if len(chosen) == k:
                return chosen
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) == k:
                break
        return chosen

    def evolution_archive(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)
        mask = pareto_mask(y)
        cap = min(len(x), max(16, self.state.settings["populationSize"] * 4))
        chosen: list[int] = []

        def add(idx: int) -> None:
            if 0 <= idx < len(x) and idx not in chosen:
                chosen.append(idx)

        for idx, is_pareto in enumerate(mask):
            if is_pareto and len(chosen) < cap:
                add(idx)
        for idx in range(max(0, len(x) - max(1, cap // 3)), len(x)):
            if len(chosen) < cap:
                add(idx)
        for i in range(cap):
            if len(chosen) < cap:
                add(int(i * len(x) / cap))
        return x[chosen], y[chosen]

    def tournament_pair(self, population: np.ndarray, population_y: np.ndarray) -> tuple[int, int]:
        fronts = self.non_dominated_sort(population_y)
        rank = np.zeros(len(population), dtype=int)
        crowd = np.zeros(len(population), dtype=float)
        for front_idx, front in enumerate(fronts):
            rank[front] = front_idx
            crowd[front] = self.crowding_distance(front, population_y)

        def pick() -> int:
            a = int(self.rng.integers(0, len(population)))
            b = int(self.rng.integers(0, len(population)))
            if rank[a] != rank[b]:
                return a if rank[a] < rank[b] else b
            return a if crowd[a] >= crowd[b] else b

        return pick(), pick()

    def non_dominated_sort(self, y: np.ndarray) -> list[np.ndarray]:
        dominates: list[list[int]] = [[] for _ in range(len(y))]
        dominated_by = np.zeros(len(y), dtype=int)
        first: list[int] = []
        for i, point in enumerate(y):
            for j, other in enumerate(y):
                if i == j:
                    continue
                if np.all(point >= other) and np.any(point > other):
                    dominates[i].append(j)
                elif np.all(other >= point) and np.any(other > point):
                    dominated_by[i] += 1
            if dominated_by[i] == 0:
                first.append(i)
        fronts = [np.array(first, dtype=int)] if first else []
        level = 0
        while level < len(fronts) and len(fronts[level]):
            next_front: list[int] = []
            for i in fronts[level]:
                for j in dominates[int(i)]:
                    dominated_by[j] -= 1
                    if dominated_by[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(np.array(next_front, dtype=int))
            level += 1
        return fronts

    def crowding_distance(self, front: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(front) <= 2:
            return np.full(len(front), np.inf)
        distance = np.zeros(len(front), dtype=float)
        for obj in range(y.shape[1]):
            order = np.argsort(y[front, obj])
            distance[order[0]] = np.inf
            distance[order[-1]] = np.inf
            span = max(float(y[front[order[-1]], obj] - y[front[order[0]], obj]), 1e-9)
            for i in range(1, len(order) - 1):
                distance[order[i]] += (y[front[order[i + 1]], obj] - y[front[order[i - 1]], obj]) / span
        return distance

    def blend_vectors(self, a: np.ndarray, b: np.ndarray, spread: float = 0.5) -> np.ndarray:
        alpha = np.clip(0.5 + self.rng.normal(0.0, spread * 0.15, size=len(a)), 0.0, 1.0)
        return np.clip(alpha * a + (1.0 - alpha) * b, 0.0, 1.0)

    def mutate_vector(self, base: np.ndarray, sigma: float = 0.12) -> np.ndarray:
        mask = self.rng.random(len(base)) < 0.35
        return np.clip(base + self.rng.normal(0.0, sigma, size=len(base)) * mask, 0.0, 1.0)

    def dedupe_vectors(self, rows: np.ndarray) -> np.ndarray:
        if len(rows) == 0:
            return rows
        _, idx = np.unique(np.round(rows, 4), axis=0, return_index=True)
        return rows[np.sort(idx)]

    def snapshot(self) -> dict[str, Any]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)
        mask = pareto_mask(y)
        scores = preference_score(y, self.active_weights_array(), self.state.parsed.get("constraints", {}), self.objectives())
        recommended_idx = int(np.argmax(scores)) if len(scores) else -1
        hypervolume = approximate_hypervolume(y, np.array(self.state.hv_samples, dtype=float))
        distances = distance_metrics(y, self.state.benchmark, self.state.settings["dimension"], self.state.settings.get("objectiveCount"))
        recommended_score = float(scores[recommended_idx]) if recommended_idx >= 0 else 0.0
        metric_values = {
            "hypervolume": hypervolume,
            "igd": distances["igd"],
            "igdPlus": distances["igdPlus"],
            "gd": distances["gd"],
            "recommendedScore": recommended_score,
            "paretoCount": int(mask.sum()),
        }
        points = [
            {
                "id": i,
                "x": [float(v) for v in x[i]],
                "y": [float(v) for v in y[i]],
                "pareto": bool(mask[i]),
                "preferenceScore": float(scores[i]),
                "recommended": i == recommended_idx,
            }
            for i in range(len(x))
        ]
        definition = self.definition()
        return {
            "benchmarks": [
                {
                    "key": key,
                    "name": value["name"],
                    "description": value["description"],
                    "defaultDimension": value.get("defaultDimension"),
                    "minDimension": value.get("minDimension"),
                    "maxDimension": value.get("maxDimension"),
                    "defaultObjectives": value.get("defaultObjectives"),
                    "minObjectives": value.get("minObjectives"),
                    "maxObjectives": value.get("maxObjectives"),
                    "nativeObjectives": value.get("nativeObjectives"),
                }
                for key, value in BENCHMARKS.items()
            ],
            "optimizers": [{"key": key, **value} for key, value in OPTIMIZERS.items()],
            "moduleRegistry": {
                "localBenchmarkDir": str(ROOT / "benchmarks"),
                "localOptimizerDir": str(ROOT / "methods"),
                "externalBenchmarks": [str(path) for path in EXTERNAL_MODULE_PATHS["benchmark"]],
                "externalOptimizers": [str(path) for path in EXTERNAL_MODULE_PATHS["optimizer"]],
                "registryErrors": list(REGISTRY_ERRORS),
            },
            "experimentLimits": dict(EXPERIMENT_LIMITS),
            "benchmark": {"key": self.state.benchmark, **definition},
            "optimizer": {"key": self.state.optimizer, **OPTIMIZERS[self.state.optimizer]},
            "objectives": definition["objectives"],
            "inputs": definition["inputs"],
            "dimensions": {"inputs": len(definition["inputs"]), "objectives": len(definition["objectives"])},
            "settings": dict(self.state.settings),
            "budgetRemaining": self.budget_remaining(),
            "iteration": self.state.iteration,
            "nEvaluations": len(points),
            "hypervolume": hypervolume,
            "metrics": metric_values,
            "metricLabels": METRIC_LABELS,
            "metricDirections": {
                key: ("higher_is_better" if direction > 0 else "lower_is_better")
                for key, direction in METRIC_DIRECTIONS.items()
            },
            "paretoCount": int(mask.sum()),
            "points": points,
            "pareto": [points[i] for i in range(len(points)) if mask[i]],
            "recommended": points[recommended_idx] if recommended_idx >= 0 else None,
            "feedback": self.state.feedback,
            "parsed": self.state.parsed,
            "activeWeights": self.active_weights_dict(),
            "manualWeights": self.state.manual_weights,
            "surrogate": self.state.surrogate_report_data or self.surrogate_report(),
            "lastStrategy": self.state.last_strategy,
            "lastCandidates": self.state.last_candidates,
            "experimentResult": self.state.experiment_result,
            "events": self.state.events,
        }


def experiment_settings(raw: dict[str, Any]) -> dict[str, int]:
    population = int(np.clip(numeric_setting(raw, "populationSize", 9), 1, EXPERIMENT_LIMITS["maxPopulation"]))
    max_evaluations = int(np.clip(numeric_setting(raw, "maxEvaluations", 60), population, EXPERIMENT_LIMITS["maxBudget"]))
    return {
        "populationSize": population,
        "maxEvaluations": max_evaluations,
        "dimension": int(np.clip(numeric_setting(raw, "dimension", 3), 1, 200)),
        "objectiveCount": int(np.clip(numeric_setting(raw, "objectiveCount", 3), 2, 15)),
    }


def selected_keys(raw: Any, available: dict[str, Any], fallback: list[str]) -> list[str]:
    if not isinstance(raw, list):
        raw = fallback
    keys = [str(key) for key in raw if str(key) in available]
    return list(dict.fromkeys(keys)) or fallback


def custom_import_allowed(module: str) -> bool:
    module = str(module or "").strip()
    if not module:
        return False
    root = module.split(".", 1)[0]
    if root in FORBIDDEN_CUSTOM_IMPORT_ROOTS:
        return False
    return module in ALLOWED_CUSTOM_IMPORTS or any(module == prefix or module.startswith(f"{prefix}.") for prefix in ALLOWED_CUSTOM_IMPORT_PREFIXES)


def registry_function_name(kind: str) -> str:
    return "register_benchmark" if kind == "benchmark" else "register_optimizer"


def registration_call_key(node: ast.Call, register_name: str) -> str | None:
    if isinstance(node.func, ast.Name) and node.func.id == register_name and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def extract_declared_registration_keys(source: str, kind: str) -> list[str]:
    tree = ast.parse(source)
    register_name = registry_function_name(kind)
    keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            key = registration_call_key(node, register_name)
            if key:
                keys.append(key)
    return keys


def validate_custom_source(kind: str, source: str) -> tuple[ast.Module, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc
    register_name = registry_function_name(kind)
    required = f"{register_name}(...)"
    declared_keys = [
        key
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for key in [registration_call_key(node, register_name)]
        if key
    ]
    if not declared_keys:
        raise ValueError(f"The module must call {required} with a literal string key.")
    errors: list[str] = []
    warnings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not custom_import_allowed(alias.name):
                    errors.append(f"Import '{alias.name}' is not allowed in submitted {kind} code.")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                errors.append("Relative imports are not allowed in submitted code.")
            elif not custom_import_allowed(module):
                errors.append(f"Import from '{module}' is not allowed in submitted {kind} code.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CUSTOM_CALLS:
                errors.append(f"Call '{node.func.id}(...)' is blocked in submitted {kind} code.")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CUSTOM_ATTRS:
                errors.append(f"Attribute call '.{node.func.attr}(...)' is blocked in submitted {kind} code.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"Dunder attribute access '.{node.attr}' is blocked in submitted {kind} code.")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            warnings.append("Global/nonlocal declarations are discouraged in submitted code.")
    if errors:
        unique_errors = list(dict.fromkeys(errors))
        raise ValueError("Safety check failed: " + " ".join(unique_errors[:4]))
    return tree, list(dict.fromkeys(warnings))


def unique_registry_key(base_key: str, available: dict[str, Any], used: set[str]) -> str:
    key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(base_key or "")).strip("_")
    if not key:
        key = "custom_module"
    if not key.startswith("custom_"):
        key = f"custom_{key}"
    candidate = key[:80]
    if candidate not in available and candidate not in used:
        used.add(candidate)
        return candidate
    for index in range(2, 1000):
        candidate = f"{key[:70]}_{index}"
        if candidate not in available and candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError(f"Could not allocate a unique registry key for {base_key}.")


def display_name_kind(kind: str) -> str:
    return "problem" if kind == "benchmark" else "algorithm"


def display_name_article(kind: str) -> str:
    return "A" if kind == "benchmark" else "An"


def normalized_display_name(raw_name: Any) -> str:
    return re.sub(r"\s+", " ", str(raw_name or "").strip())


def registry_display_name_folds(kind: str) -> dict[str, str]:
    registry = BENCHMARKS if kind == "benchmark" else OPTIMIZERS
    names: dict[str, str] = {}
    for value in registry.values():
        if not isinstance(value, dict):
            continue
        name = normalized_display_name(value.get("name", ""))
        if name:
            names.setdefault(name.casefold(), name)
    return names


def clean_requested_display_name(kind: str, raw_name: Any) -> str:
    name = normalized_display_name(raw_name)
    if not name:
        return ""
    if len(name) > 80:
        raise ValueError(f"{display_name_kind(kind).capitalize()} name must be 80 characters or fewer.")
    return name


def numbered_display_name(base_name: str, index: int) -> str:
    suffix = f" ({index})"
    return f"{base_name[: max(1, 80 - len(suffix))].rstrip()}{suffix}"


def requested_display_names(base_name: str, keys: list[str]) -> dict[str, str]:
    if not base_name:
        return {}
    if len(keys) <= 1:
        return {key: base_name for key in keys}
    return {key: (base_name if index == 1 else numbered_display_name(base_name, index)) for index, key in enumerate(keys, start=1)}


def ensure_display_names_available(kind: str, display_names: list[str], existing_folds: dict[str, str] | None = None) -> None:
    label = display_name_kind(kind)
    existing = existing_folds if existing_folds is not None else registry_display_name_folds(kind)
    seen: dict[str, str] = {}
    for raw_name in display_names:
        name = normalized_display_name(raw_name)
        if not name:
            continue
        folded = name.casefold()
        if folded in existing:
            raise ValueError(f"{display_name_article(kind)} {label} named '{existing[folded]}' already exists. Choose a different name.")
        if folded in seen:
            raise ValueError(f"Submitted {label} names must be unique; '{seen[folded]}' appears more than once.")
        seen[folded] = name


def imported_display_name_error(kind: str, keys: list[str], existing_folds: dict[str, str]) -> str:
    registry = BENCHMARKS if kind == "benchmark" else OPTIMIZERS
    label = display_name_kind(kind)
    seen: dict[str, str] = {}
    for key in keys:
        value = registry.get(key)
        if not isinstance(value, dict):
            continue
        name = normalized_display_name(value.get("name", ""))
        if not name:
            continue
        folded = name.casefold()
        if folded in existing_folds:
            return f"{display_name_article(kind)} {label} named '{existing_folds[folded]}' already exists. Choose a different name."
        if folded in seen:
            return f"Submitted {label} names must be unique; '{seen[folded]}' appears more than once."
        seen[folded] = name
    return ""


def append_display_name_overrides(source: str, kind: str, display_names: dict[str, str]) -> str:
    if not display_names:
        return source
    module_name = "benchmarks.registry" if kind == "benchmark" else "methods.registry"
    registry_name = "BENCHMARKS" if kind == "benchmark" else "OPTIMIZERS"
    lines = [
        "",
        "# Display names requested through the BlackBoxLab submit form.",
        f"from {module_name} import {registry_name} as _blackboxlab_registry",
    ]
    for key, name in display_names.items():
        lines.append(f"_blackboxlab_registry[{key!r}]['name'] = {name!r}")
    return f"{source.rstrip()}\n{chr(10).join(lines)}\n"


def rewrite_registration_keys(source: str, kind: str, available: dict[str, Any]) -> tuple[str, list[str], dict[str, str]]:
    tree, warnings = validate_custom_source(kind, source)
    register_name = registry_function_name(kind)
    used: set[str] = set()
    renamed: dict[str, str] = {}

    class RegistrationKeyRenamer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            key = registration_call_key(node, register_name)
            if key is None:
                return node
            new_key = unique_registry_key(key, available, used)
            if new_key != key:
                renamed[key] = new_key
            node.args[0] = ast.copy_location(ast.Constant(new_key), node.args[0])
            return node

    rewritten = RegistrationKeyRenamer().visit(tree)
    ast.fix_missing_locations(rewritten)
    rewritten_source = ast.unparse(rewritten)
    return f"{rewritten_source}\n", warnings, renamed


def safe_custom_filename(kind: str, raw_name: str) -> Path:
    folder = ROOT / ("benchmarks" if kind == "benchmark" else "methods")
    stem = Path(raw_name or "").stem.lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    if not stem:
        stem = f"{kind}_module"
    if not stem.startswith("custom_"):
        stem = f"custom_{stem}"
    return folder / f"{stem[:80]}.py"


def unique_custom_filename(kind: str, raw_name: str, source: str) -> Path:
    target = safe_custom_filename(kind, raw_name)
    suffix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    candidate = target.with_name(f"{target.stem}_{suffix}{target.suffix}")
    if not candidate.exists() or candidate.read_text(encoding="utf-8", errors="ignore") == source:
        return candidate
    for index in range(2, 1000):
        candidate = target.with_name(f"{target.stem}_{suffix}_{index}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError("Could not allocate a unique custom module filename.")


def import_custom_module(body: dict[str, Any]) -> dict[str, Any]:
    kind = str(body.get("kind", "")).strip().lower()
    if kind not in {"benchmark", "optimizer"}:
        raise ValueError("kind must be benchmark or optimizer.")
    filename = str(body.get("filename", "custom_module.py"))
    source = str(body.get("source", ""))
    if not filename.lower().endswith(".py"):
        raise ValueError("Only .py files can be imported.")
    if not source.strip():
        raise ValueError("Selected module is empty.")
    if len(source.encode("utf-8")) > 220_000:
        raise ValueError("Selected module is too large; keep it below 220 KB.")
    registry = BENCHMARKS if kind == "benchmark" else OPTIMIZERS
    existing_display_names = registry_display_name_folds(kind)
    requested_name = clean_requested_display_name(kind, body.get("displayName", ""))
    rewritten_source, safety_warnings, renamed_keys = rewrite_registration_keys(source, kind, registry)
    registered_keys = extract_declared_registration_keys(rewritten_source, kind)
    display_names = requested_display_names(requested_name, registered_keys)
    ensure_display_names_available(kind, list(display_names.values()), existing_display_names)
    rewritten_source = append_display_name_overrides(rewritten_source, kind, display_names)
    before_errors = list(REGISTRY_ERRORS)
    target = unique_custom_filename(kind, filename, rewritten_source)
    target.write_text(rewritten_source, encoding="utf-8")
    reference_front_for.cache_clear()
    refresh_registries_if_needed(force=True)
    new_errors = [error for error in REGISTRY_ERRORS if error not in before_errors]
    own_errors = [error for error in new_errors if target.stem in error or target.name in error]
    if own_errors:
        try:
            target.unlink()
        except OSError:
            pass
        refresh_registries_if_needed(force=True)
        raise ValueError("Import failed after safety checks: " + " ".join(own_errors[:2]))
    display_name_error = imported_display_name_error(kind, registered_keys, existing_display_names)
    if display_name_error:
        try:
            target.unlink()
        except OSError:
            pass
        refresh_registries_if_needed(force=True)
        raise ValueError(display_name_error)
    return {
        "kind": kind,
        "filename": target.name,
        "path": str(target),
        "registeredKeys": registered_keys,
        "displayNames": display_names,
        "renamedKeys": renamed_keys,
        "warnings": safety_warnings,
        "registryErrors": list(REGISTRY_ERRORS),
    }


def submit_custom_problem(body: dict[str, Any]) -> dict[str, Any]:
    source = str(body.get("source", ""))
    if not source.strip():
        raise ValueError("Problem code is empty.")
    if len(source.encode("utf-8")) > 120_000:
        raise ValueError("Problem code is too large; keep it below 120 KB.")
    before = set(BENCHMARKS)
    filename = str(body.get("filename") or "submitted_problem.py")
    result = import_custom_module({"kind": "benchmark", "filename": filename, "source": source, "displayName": body.get("displayName", "")})
    refresh_registries_if_needed(force=True)
    after = set(BENCHMARKS)
    new_keys = sorted(after - before)
    selected = str(body.get("benchmark") or "")
    if selected not in BENCHMARKS:
        selected = next((key for key in reversed(result.get("registeredKeys", [])) if key in BENCHMARKS), "")
    if selected not in BENCHMARKS:
        selected = new_keys[-1] if new_keys else DEMO.state.benchmark
    settings = dict(DEMO.state.settings)
    if body.get("settings"):
        settings.update(dict(body.get("settings", {})))
    state = DEMO.reset(
        seed=int(body.get("seed", random.randint(1, 99999))),
        feedback=body.get("feedback") or DEMO.state.feedback,
        benchmark=selected,
        optimizer=DEMO.state.optimizer,
        settings=settings,
    )
    return {
        "importResult": {**result, "selectedBenchmark": selected},
        "state": state,
    }


def submit_custom_optimizer(body: dict[str, Any]) -> dict[str, Any]:
    source = str(body.get("source", ""))
    if not source.strip():
        raise ValueError("Optimizer code is empty.")
    if len(source.encode("utf-8")) > 220_000:
        raise ValueError("Optimizer code is too large; keep it below 220 KB.")
    before = set(OPTIMIZERS)
    filename = str(body.get("filename") or "submitted_optimizer.py")
    result = import_custom_module({"kind": "optimizer", "filename": filename, "source": source, "displayName": body.get("displayName", "")})
    refresh_registries_if_needed(force=True)
    after = set(OPTIMIZERS)
    new_keys = sorted(after - before)
    selected = str(body.get("optimizer") or "")
    if selected not in OPTIMIZERS:
        selected = next((key for key in reversed(result.get("registeredKeys", [])) if key in OPTIMIZERS), "")
    if selected not in OPTIMIZERS:
        selected = new_keys[-1] if new_keys else DEMO.state.optimizer
    settings = dict(DEMO.state.settings)
    if body.get("settings"):
        settings.update(dict(body.get("settings", {})))
    state = DEMO.reset(
        seed=int(body.get("seed", random.randint(1, 99999))),
        feedback=body.get("feedback") or DEMO.state.feedback,
        benchmark=DEMO.state.benchmark,
        optimizer=selected,
        settings=settings,
    )
    return {
        "importResult": {**result, "selectedOptimizer": selected},
        "state": state,
    }


def import_custom_module_from_path(body: dict[str, Any]) -> dict[str, Any]:
    kind = str(body.get("kind", "")).strip().lower()
    if kind not in {"benchmark", "optimizer"}:
        raise ValueError("kind must be benchmark or optimizer.")
    source_path = Path(str(body.get("path", ""))).expanduser()
    if not source_path.exists() or not source_path.is_file():
        raise ValueError("Local .py file does not exist.")
    if source_path.suffix.lower() != ".py":
        raise ValueError("Only .py files can be imported.")
    try:
        resolved = source_path.resolve()
    except OSError as exc:
        raise ValueError(f"Cannot resolve local path: {exc}") from exc
    if resolved.stat().st_size > 220_000:
        raise ValueError("Local module is too large; keep it below 220 KB.")
    try:
        source = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Local module must be UTF-8 encoded.") from exc
    _, safety_warnings = validate_custom_source(kind, source)
    paths = EXTERNAL_MODULE_PATHS[kind]
    if resolved not in paths:
        paths.append(resolved)
    reference_front_for.cache_clear()
    refresh_registries_if_needed(force=True)
    return {
        "kind": kind,
        "filename": resolved.name,
        "path": str(resolved),
        "mode": "external_path",
        "warnings": safety_warnings,
        "registryErrors": list(REGISTRY_ERRORS),
    }


def refresh_local_modules() -> dict[str, Any]:
    reference_front_for.cache_clear()
    refresh_registries_if_needed(force=True)
    return {
        "benchmarks": len(BENCHMARKS),
        "optimizers": len(OPTIMIZERS),
        "localBenchmarkDir": str(ROOT / "benchmarks"),
        "localOptimizerDir": str(ROOT / "methods"),
        "externalBenchmarks": [str(path) for path in EXTERNAL_MODULE_PATHS["benchmark"]],
        "externalOptimizers": [str(path) for path in EXTERNAL_MODULE_PATHS["optimizer"]],
        "registryErrors": list(REGISTRY_ERRORS),
    }


def experiment_worker_count(raw: dict[str, Any], total_runs: int) -> int:
    default_workers = min(4, max(1, os.cpu_count() or 1), max(1, total_runs))
    workers = numeric_setting(raw, "workers", default_workers)
    return int(np.clip(workers, 1, min(EXPERIMENT_LIMITS["maxWorkers"], max(1, total_runs))))


def experiment_row(
    benchmark_key: str,
    optimizer_key: str,
    seed: int,
    repeat: int,
    snap: dict[str, Any],
    elapsed_ms: int,
) -> dict[str, Any]:
    result = snap.get("experimentResult") or {}
    recommended = snap.get("recommended") or {}
    distances = distance_metrics(
        np.array([point["y"] for point in snap.get("points", [])], dtype=float),
        benchmark_key,
        int(snap["dimensions"]["inputs"]),
        int(snap["dimensions"]["objectives"]),
    )
    return {
        "benchmark": benchmark_key,
        "benchmarkName": snap["benchmark"]["name"],
        "optimizer": optimizer_key,
        "optimizerName": snap["optimizer"]["name"],
        "seed": seed,
        "repeat": repeat,
        "status": "ok",
        "dimension": snap["dimensions"]["inputs"],
        "objectives": snap["dimensions"]["objectives"],
        "populationSize": snap["settings"]["populationSize"],
        "maxEvaluations": snap["settings"]["maxEvaluations"],
        "evaluations": result.get("evaluations", snap["nEvaluations"]),
        "rounds": result.get("rounds", snap["iteration"]),
        "paretoCount": result.get("paretoCount", snap["paretoCount"]),
        "hypervolume": float(result.get("hypervolume", snap["hypervolume"])),
        "igd": distances["igd"],
        "igdPlus": distances["igdPlus"],
        "gd": distances["gd"],
        "recommendedScore": float(result.get("recommendedScore", recommended.get("preferenceScore", 0.0))),
        "recommendedId": int(result.get("recommendedId", recommended.get("id", -1))),
        "elapsedMs": int(result.get("elapsedMs", elapsed_ms)),
    }


def run_experiment_trial(
    benchmark: str,
    optimizer: str,
    seed: int,
    repeat: int,
    settings: dict[str, int],
    feedback: str,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    try:
        demo = ParetoLiloDemo(auto_reset=False)
        snap = demo.reset(seed=seed, feedback=feedback, benchmark=benchmark, optimizer=optimizer, settings=settings)
        snap = demo.run_to_budget()
        return experiment_row(benchmark, optimizer, seed, repeat, snap, int((time.perf_counter() - run_started) * 1000))
    except Exception as exc:  # keep the comparison table useful even if one method fails
        definition = materialize_benchmark(benchmark, settings["dimension"], settings.get("objectiveCount"))
        return {
            "benchmark": benchmark,
            "benchmarkName": definition["name"],
            "optimizer": optimizer,
            "optimizerName": OPTIMIZERS.get(optimizer, {}).get("name", optimizer),
            "seed": seed,
            "repeat": repeat,
            "status": "error",
            "dimension": len(definition["inputs"]),
            "objectives": len(definition["objectives"]),
            "populationSize": settings["populationSize"],
            "maxEvaluations": settings["maxEvaluations"],
            "evaluations": 0,
            "rounds": 0,
            "paretoCount": 0,
            "hypervolume": 0.0,
            "igd": 1.0,
            "igdPlus": 1.0,
            "gd": 1.0,
            "recommendedScore": 0.0,
            "recommendedId": -1,
            "elapsedMs": int((time.perf_counter() - run_started) * 1000),
            "error": str(exc),
        }


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def format_scientific(value: float) -> str:
    return f"{float(value):.4e}"


def sign_test_marker(values: list[float], reference_values: list[float], alpha: float = 0.05) -> dict[str, Any]:
    diffs = [float(value) - float(ref) for value, ref in zip(values, reference_values)]
    wins = sum(diff > 1e-12 for diff in diffs)
    losses = sum(diff < -1e-12 for diff in diffs)
    ties = len(diffs) - wins - losses
    n = wins + losses
    if n == 0:
        return {"marker": "~", "pValue": 1.0, "wins": wins, "ties": ties, "losses": losses}
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    p_value = min(1.0, 2.0 * tail)
    if p_value <= alpha and wins > losses:
        marker = "+"
    elif p_value <= alpha and losses > wins:
        marker = "-"
    else:
        marker = "~"
    return {"marker": marker, "pValue": float(p_value), "wins": wins, "ties": ties, "losses": losses}


def metric_direction(metric: str) -> int:
    return METRIC_DIRECTIONS.get(metric, 1)


def sign_test_marker(values: list[float], reference_values: list[float], metric: str, alpha: float = 0.05) -> dict[str, Any]:
    direction = metric_direction(metric)
    improvements = [direction * (float(value) - float(ref)) for value, ref in zip(values, reference_values)]
    wins = sum(diff > 1e-12 for diff in improvements)
    losses = sum(diff < -1e-12 for diff in improvements)
    ties = len(improvements) - wins - losses
    n = wins + losses
    if n == 0:
        return {"marker": "~", "pValue": 1.0, "wins": wins, "ties": ties, "losses": losses}
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
    p_value = min(1.0, 2.0 * tail)
    if p_value <= alpha and wins > losses:
        marker = "+"
    elif p_value <= alpha and losses > wins:
        marker = "-"
    else:
        marker = "~"
    return {"marker": marker, "pValue": float(p_value), "wins": wins, "ties": ties, "losses": losses}


def paired_metric_values(rows: list[dict[str, Any]], benchmark: str, optimizer: str, metric: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        if row.get("status") == "ok" and row.get("benchmark") == benchmark and row.get("optimizer") == optimizer:
            values[int(row["seed"])] = float(row.get(metric, 0.0))
    return values


def make_paper_table(
    rows: list[dict[str, Any]],
    benchmarks: list[str],
    optimizers: list[str],
    metric: str,
    reference_optimizer: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ref = reference_optimizer if reference_optimizer in optimizers else (optimizers[-1] if optimizers else "")
    optimizer_names = {row["optimizer"]: row["optimizerName"] for row in ok_rows}
    benchmark_names = {row["benchmark"]: row["benchmarkName"] for row in ok_rows}
    footer_counts = {optimizer: {"+": 0, "~": 0, "-": 0} for optimizer in optimizers}
    table_rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        benchmark_rows = [row for row in ok_rows if row["benchmark"] == benchmark]
        first = benchmark_rows[0] if benchmark_rows else {}
        ref_by_seed = paired_metric_values(ok_rows, benchmark, ref, metric)
        cells: dict[str, Any] = {}
        for optimizer in optimizers:
            values_by_seed = paired_metric_values(ok_rows, benchmark, optimizer, metric)
            common_seeds = sorted(set(values_by_seed) & set(ref_by_seed))
            values = [values_by_seed[seed] for seed in sorted(values_by_seed)]
            stats = mean_std(values)
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            if optimizer == ref:
                test = {"marker": "ref", "pValue": 1.0, "wins": 0, "ties": len(values), "losses": 0}
            elif common_seeds:
                test = sign_test_marker([values_by_seed[seed] for seed in common_seeds], [ref_by_seed[seed] for seed in common_seeds], alpha)
                footer_counts[optimizer][test["marker"]] += 1
            else:
                test = {"marker": "~", "pValue": 1.0, "wins": 0, "ties": 0, "losses": 0}
                footer_counts[optimizer]["~"] += 1
            marker = "" if test["marker"] == "ref" else f" {test['marker']}"
            cells[optimizer] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "variance": variance,
                "text": f"{format_scientific(stats['mean'])} ({format_scientific(variance)}){marker}",
                **test,
            }
        table_rows.append(
            {
                "benchmark": benchmark,
                "benchmarkName": benchmark_names.get(benchmark, benchmark),
                "objectives": int(first.get("objectives", 0) or 0),
                "dimension": int(first.get("dimension", 0) or 0),
                "cells": cells,
            }
        )

    footer = {
        optimizer: (
            "reference"
            if optimizer == ref
            else f"+/{footer_counts[optimizer]['+']} ~/{footer_counts[optimizer]['~']} -/{footer_counts[optimizer]['-']}"
        )
        for optimizer in optimizers
    }
    return {
        "metric": metric,
        "metricLabel": {
            "hypervolume": "Hypervolume",
            "recommendedScore": "Recommended preference score",
            "paretoCount": "Pareto count",
            "elapsedMs": "Elapsed time (lower is better; currently reported, not ranked)",
        }.get(metric, metric),
        "referenceOptimizer": ref,
        "referenceName": optimizer_names.get(ref, OPTIMIZERS.get(ref, {}).get("name", ref)),
        "alpha": alpha,
        "optimizers": [{"key": optimizer, "name": optimizer_names.get(optimizer, OPTIMIZERS.get(optimizer, {}).get("name", optimizer))} for optimizer in optimizers],
        "rows": table_rows,
        "footer": footer,
    }


def make_pairwise_matrix(rows: list[dict[str, Any]], optimizers: list[str], metric: str, alpha: float = 0.05) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    optimizer_names = {row["optimizer"]: row["optimizerName"] for row in ok_rows}
    by_trial: dict[tuple[str, int, str], float] = {}
    for row in ok_rows:
        by_trial[(row["benchmark"], int(row["seed"]), row["optimizer"])] = float(row.get(metric, 0.0))
    cells: dict[str, dict[str, Any]] = {}
    for row_optimizer in optimizers:
        row_cells: dict[str, Any] = {}
        for col_optimizer in optimizers:
            if row_optimizer == col_optimizer:
                row_cells[col_optimizer] = {
                    "text": "-",
                    "marker": "self",
                    "pValue": 1.0,
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                    "meanDiff": 0.0,
                    "varianceDiff": 0.0,
                }
                continue
            row_values: list[float] = []
            col_values: list[float] = []
            for benchmark, seed, optimizer in list(by_trial.keys()):
                if optimizer != row_optimizer:
                    continue
                other_key = (benchmark, seed, col_optimizer)
                if other_key in by_trial:
                    row_values.append(by_trial[(benchmark, seed, row_optimizer)])
                    col_values.append(by_trial[other_key])
            diffs = [float(a) - float(b) for a, b in zip(row_values, col_values)]
            mean_diff = float(np.mean(diffs)) if diffs else 0.0
            variance_diff = float(np.var(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            test = sign_test_marker(row_values, col_values, alpha) if row_values else {"marker": "~", "pValue": 1.0, "wins": 0, "ties": 0, "losses": 0}
            row_cells[col_optimizer] = {
                **test,
                "meanDiff": mean_diff,
                "varianceDiff": variance_diff,
                "text": f"{format_scientific(mean_diff)} ({format_scientific(variance_diff)}) {test['marker']}",
                "detail": f"{test['wins']}/{test['ties']}/{test['losses']}",
            }
        cells[row_optimizer] = row_cells
    return {
        "metric": metric,
        "alpha": alpha,
        "optimizers": [{"key": optimizer, "name": optimizer_names.get(optimizer, OPTIMIZERS.get(optimizer, {}).get("name", optimizer))} for optimizer in optimizers],
        "cells": cells,
    }


def summarize_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    wins: dict[tuple[str, str], int] = {}
    groups_by_trial: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups_by_trial.setdefault((row["benchmark"], int(row["seed"])), []).append(row)
    for trial_rows in groups_by_trial.values():
        if not trial_rows:
            continue
        best = max(float(row["hypervolume"]) for row in trial_rows)
        winners = [row for row in trial_rows if abs(float(row["hypervolume"]) - best) <= 1e-12]
        for row in winners:
            key = (row["benchmark"], row["optimizer"])
            wins[key] = wins.get(key, 0) + 1

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault((row["benchmark"], row["optimizer"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (benchmark, optimizer), group in sorted(groups.items()):
        hv = mean_std([float(row["hypervolume"]) for row in group])
        pareto = mean_std([float(row["paretoCount"]) for row in group])
        score = mean_std([float(row["recommendedScore"]) for row in group])
        elapsed = mean_std([float(row["elapsedMs"]) for row in group])
        evals = mean_std([float(row["evaluations"]) for row in group])
        summary.append(
            {
                "benchmark": benchmark,
                "benchmarkName": group[0]["benchmarkName"],
                "optimizer": optimizer,
                "optimizerName": group[0]["optimizerName"],
                "runs": len(group),
                "hypervolumeMean": hv["mean"],
                "hypervolumeStd": hv["std"],
                "paretoCountMean": pareto["mean"],
                "paretoCountStd": pareto["std"],
                "recommendedScoreMean": score["mean"],
                "recommendedScoreStd": score["std"],
                "evaluationsMean": evals["mean"],
                "elapsedMsMean": elapsed["mean"],
                "elapsedMsStd": elapsed["std"],
                "hvWins": wins.get((benchmark, optimizer), 0),
            }
        )

    for benchmark in sorted({row["benchmark"] for row in ok_rows}):
        subset = [item for item in summary if item["benchmark"] == benchmark]
        order = sorted(subset, key=lambda item: item["hypervolumeMean"], reverse=True)
        for rank, item in enumerate(order, start=1):
            item["hypervolumeRank"] = rank
    return summary


def make_paper_table(
    rows: list[dict[str, Any]],
    benchmarks: list[str],
    optimizers: list[str],
    metric: str,
    reference_optimizer: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ref = reference_optimizer if reference_optimizer in optimizers else (optimizers[-1] if optimizers else "")
    optimizer_names = {row["optimizer"]: row["optimizerName"] for row in ok_rows}
    benchmark_names = {row["benchmark"]: row["benchmarkName"] for row in ok_rows}
    footer_counts = {optimizer: {"+": 0, "~": 0, "-": 0} for optimizer in optimizers}
    table_rows: list[dict[str, Any]] = []

    for benchmark in benchmarks:
        benchmark_rows = [row for row in ok_rows if row["benchmark"] == benchmark]
        first = benchmark_rows[0] if benchmark_rows else {}
        ref_by_seed = paired_metric_values(ok_rows, benchmark, ref, metric)
        cells: dict[str, Any] = {}
        for optimizer in optimizers:
            values_by_seed = paired_metric_values(ok_rows, benchmark, optimizer, metric)
            common_seeds = sorted(set(values_by_seed) & set(ref_by_seed))
            values = [values_by_seed[seed] for seed in sorted(values_by_seed)]
            stats = mean_std(values)
            variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
            if optimizer == ref:
                test = {"marker": "ref", "pValue": 1.0, "wins": 0, "ties": len(values), "losses": 0}
            elif common_seeds:
                test = sign_test_marker([values_by_seed[seed] for seed in common_seeds], [ref_by_seed[seed] for seed in common_seeds], metric, alpha)
                footer_counts[optimizer][test["marker"]] += 1
            else:
                test = {"marker": "~", "pValue": 1.0, "wins": 0, "ties": 0, "losses": 0}
                footer_counts[optimizer]["~"] += 1
            marker = "" if test["marker"] == "ref" else f" {test['marker']}"
            cells[optimizer] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "variance": variance,
                "text": f"{format_scientific(stats['mean'])} ({format_scientific(variance)}){marker}",
                **test,
            }
        table_rows.append(
            {
                "benchmark": benchmark,
                "benchmarkName": benchmark_names.get(benchmark, benchmark),
                "objectives": int(first.get("objectives", 0) or 0),
                "dimension": int(first.get("dimension", 0) or 0),
                "cells": cells,
            }
        )

    footer = {
        optimizer: (
            "reference"
            if optimizer == ref
            else f"+/{footer_counts[optimizer]['+']} ~/{footer_counts[optimizer]['~']} -/{footer_counts[optimizer]['-']}"
        )
        for optimizer in optimizers
    }
    direction = "higher_is_better" if metric_direction(metric) > 0 else "lower_is_better"
    return {
        "metric": metric,
        "metricLabel": METRIC_LABELS.get(metric, metric),
        "metricDirection": direction,
        "referenceOptimizer": ref,
        "referenceName": optimizer_names.get(ref, OPTIMIZERS.get(ref, {}).get("name", ref)),
        "alpha": alpha,
        "optimizers": [{"key": optimizer, "name": optimizer_names.get(optimizer, OPTIMIZERS.get(optimizer, {}).get("name", optimizer))} for optimizer in optimizers],
        "rows": table_rows,
        "footer": footer,
    }


def make_pairwise_matrix(rows: list[dict[str, Any]], optimizers: list[str], metric: str, alpha: float = 0.05) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    optimizer_names = {row["optimizer"]: row["optimizerName"] for row in ok_rows}
    direction = metric_direction(metric)
    by_trial: dict[tuple[str, int, str], float] = {}
    for row in ok_rows:
        by_trial[(row["benchmark"], int(row["seed"]), row["optimizer"])] = float(row.get(metric, 0.0))
    cells: dict[str, dict[str, Any]] = {}
    for row_optimizer in optimizers:
        row_cells: dict[str, Any] = {}
        for col_optimizer in optimizers:
            if row_optimizer == col_optimizer:
                row_cells[col_optimizer] = {
                    "text": "-",
                    "marker": "self",
                    "pValue": 1.0,
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                    "meanDiff": 0.0,
                    "varianceDiff": 0.0,
                    "meanImprovement": 0.0,
                    "varianceImprovement": 0.0,
                }
                continue
            row_values: list[float] = []
            col_values: list[float] = []
            for benchmark, seed, optimizer in list(by_trial.keys()):
                if optimizer != row_optimizer:
                    continue
                other_key = (benchmark, seed, col_optimizer)
                if other_key in by_trial:
                    row_values.append(by_trial[(benchmark, seed, row_optimizer)])
                    col_values.append(by_trial[other_key])
            diffs = [float(a) - float(b) for a, b in zip(row_values, col_values)]
            improvements = [direction * diff for diff in diffs]
            mean_diff = float(np.mean(diffs)) if diffs else 0.0
            variance_diff = float(np.var(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            mean_improvement = float(np.mean(improvements)) if improvements else 0.0
            variance_improvement = float(np.var(improvements, ddof=1)) if len(improvements) > 1 else 0.0
            test = sign_test_marker(row_values, col_values, metric, alpha) if row_values else {"marker": "~", "pValue": 1.0, "wins": 0, "ties": 0, "losses": 0}
            row_cells[col_optimizer] = {
                **test,
                "meanDiff": mean_diff,
                "varianceDiff": variance_diff,
                "meanImprovement": mean_improvement,
                "varianceImprovement": variance_improvement,
                "text": f"{format_scientific(mean_improvement)} ({format_scientific(variance_improvement)}) {test['marker']}",
                "detail": f"{test['wins']}/{test['ties']}/{test['losses']}",
            }
        cells[row_optimizer] = row_cells
    return {
        "metric": metric,
        "metricLabel": METRIC_LABELS.get(metric, metric),
        "metricDirection": "higher_is_better" if direction > 0 else "lower_is_better",
        "alpha": alpha,
        "optimizers": [{"key": optimizer, "name": optimizer_names.get(optimizer, OPTIMIZERS.get(optimizer, {}).get("name", optimizer))} for optimizer in optimizers],
        "cells": cells,
    }


def summarize_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    wins: dict[tuple[str, str], int] = {}
    groups_by_trial: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups_by_trial.setdefault((row["benchmark"], int(row["seed"])), []).append(row)
    for trial_rows in groups_by_trial.values():
        if not trial_rows:
            continue
        best = max(float(row["hypervolume"]) for row in trial_rows)
        for row in [item for item in trial_rows if abs(float(item["hypervolume"]) - best) <= 1e-12]:
            key = (row["benchmark"], row["optimizer"])
            wins[key] = wins.get(key, 0) + 1

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault((row["benchmark"], row["optimizer"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (benchmark, optimizer), group in sorted(groups.items()):
        hv = mean_std([float(row["hypervolume"]) for row in group])
        igd = mean_std([float(row["igd"]) for row in group])
        igd_plus = mean_std([float(row["igdPlus"]) for row in group])
        gd = mean_std([float(row["gd"]) for row in group])
        pareto = mean_std([float(row["paretoCount"]) for row in group])
        score = mean_std([float(row["recommendedScore"]) for row in group])
        elapsed = mean_std([float(row["elapsedMs"]) for row in group])
        evals = mean_std([float(row["evaluations"]) for row in group])
        summary.append(
            {
                "benchmark": benchmark,
                "benchmarkName": group[0]["benchmarkName"],
                "optimizer": optimizer,
                "optimizerName": group[0]["optimizerName"],
                "runs": len(group),
                "hypervolumeMean": hv["mean"],
                "hypervolumeStd": hv["std"],
                "igdMean": igd["mean"],
                "igdStd": igd["std"],
                "igdPlusMean": igd_plus["mean"],
                "igdPlusStd": igd_plus["std"],
                "gdMean": gd["mean"],
                "gdStd": gd["std"],
                "paretoCountMean": pareto["mean"],
                "paretoCountStd": pareto["std"],
                "recommendedScoreMean": score["mean"],
                "recommendedScoreStd": score["std"],
                "evaluationsMean": evals["mean"],
                "elapsedMsMean": elapsed["mean"],
                "elapsedMsStd": elapsed["std"],
                "hvWins": wins.get((benchmark, optimizer), 0),
            }
        )

    for benchmark in sorted({row["benchmark"] for row in ok_rows}):
        subset = [item for item in summary if item["benchmark"] == benchmark]
        hv_order = sorted(subset, key=lambda item: item["hypervolumeMean"], reverse=True)
        igd_order = sorted(subset, key=lambda item: item["igdMean"])
        for rank, item in enumerate(hv_order, start=1):
            item["hypervolumeRank"] = rank
        for rank, item in enumerate(igd_order, start=1):
            item["igdRank"] = rank
    return summary


def run_comparison_experiment(body: dict[str, Any]) -> dict[str, Any]:
    refresh_registries_if_needed()
    current = DEMO.snapshot()
    benchmarks = selected_keys(body.get("benchmarks"), BENCHMARKS, [current["benchmark"]["key"]])
    optimizers = selected_keys(body.get("optimizers"), OPTIMIZERS, [current["optimizer"]["key"]])
    repeats = int(np.clip(numeric_setting(body, "repeats", 3), 1, EXPERIMENT_LIMITS["maxRepeats"]))
    settings = experiment_settings(dict(body.get("settings", {})))
    total_runs = len(benchmarks) * len(optimizers) * repeats
    if total_runs > EXPERIMENT_LIMITS["maxRuns"]:
        raise ValueError(f"Experiment grid is too large: {total_runs} runs; limit is {EXPERIMENT_LIMITS['maxRuns']}.")

    seed0 = numeric_setting(body, "seed", 7001)
    workers = experiment_worker_count(body, total_runs)
    metric = str(body.get("metric") or "hypervolume")
    if metric not in METRIC_DIRECTIONS:
        metric = "hypervolume"
    reference_optimizer = str(body.get("referenceOptimizer") or (optimizers[-1] if optimizers else ""))
    if reference_optimizer not in optimizers:
        reference_optimizer = optimizers[-1] if optimizers else ""
    feedback = str(body.get("feedback") or current["feedback"])
    jobs = [
        {
            "order": order,
            "benchmark": benchmark,
            "optimizer": optimizer,
            "seed": seed0 + repeat,
            "repeat": repeat + 1,
        }
        for order, (benchmark, repeat, optimizer) in enumerate(
            (benchmark, repeat, optimizer)
            for benchmark in benchmarks
            for repeat in range(repeats)
            for optimizer in optimizers
        )
    ]
    rows_by_order: dict[int, dict[str, Any]] = {}
    started_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_experiment_trial,
                job["benchmark"],
                job["optimizer"],
                int(job["seed"]),
                int(job["repeat"]),
                settings,
                feedback,
            ): int(job["order"])
            for job in jobs
        }
        for future in as_completed(futures):
            rows_by_order[futures[future]] = future.result()

    rows = [rows_by_order[i] for i in range(len(jobs))]
    summary = summarize_experiment(rows)
    paper_table = make_paper_table(rows, benchmarks, optimizers, metric, reference_optimizer)
    pairwise_matrix = make_pairwise_matrix(rows, optimizers, metric)
    return {
        "limits": dict(EXPERIMENT_LIMITS),
        "settings": settings,
        "benchmarks": benchmarks,
        "optimizers": optimizers,
        "repeats": repeats,
        "seed": seed0,
        "workers": workers,
        "metric": metric,
        "metricDirection": "higher_is_better" if metric_direction(metric) > 0 else "lower_is_better",
        "referenceOptimizer": reference_optimizer,
        "feedback": feedback,
        "rows": rows,
        "summary": summary,
        "paperTable": paper_table,
        "pairwiseMatrix": pairwise_matrix,
        "elapsedMs": int((time.perf_counter() - started_all) * 1000),
    }


def so_sphere(x: np.ndarray) -> float:
    z = 10.0 * np.asarray(x, dtype=float) - 5.0
    return float(np.sum(z**2))


def so_rastrigin(x: np.ndarray) -> float:
    z = 10.24 * np.asarray(x, dtype=float) - 5.12
    return float(10.0 * len(z) + np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z)))


def so_rosenbrock(x: np.ndarray) -> float:
    z = 4.0 * np.asarray(x, dtype=float) - 2.0
    return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))


def so_ackley(x: np.ndarray) -> float:
    z = 10.0 * np.asarray(x, dtype=float) - 5.0
    d = max(1, len(z))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / d)) - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / d) + 20.0 + math.e)


def so_griewank(x: np.ndarray) -> float:
    z = 12.0 * np.asarray(x, dtype=float) - 6.0
    product = np.prod(np.cos(z / np.sqrt(np.arange(1, len(z) + 1))))
    return float(np.sum(z**2) / 4000.0 - product + 1.0)


def so_schwefel(x: np.ndarray) -> float:
    z = 1000.0 * np.asarray(x, dtype=float) - 500.0
    return float(418.9829 * len(z) - np.sum(z * np.sin(np.sqrt(np.abs(z)))))


def so_levy(x: np.ndarray) -> float:
    z = 20.0 * np.asarray(x, dtype=float) - 10.0
    w = 1.0 + (z - 1.0) / 4.0
    first = np.sin(np.pi * w[0]) ** 2
    middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    last = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    return float(first + middle + last)


def so_zakharov(x: np.ndarray) -> float:
    z = 10.0 * np.asarray(x, dtype=float) - 5.0
    idx = np.arange(1, len(z) + 1, dtype=float)
    linear = 0.5 * np.sum(idx * z)
    return float(np.sum(z**2) + linear**2 + linear**4)


def so_ellipsoid(x: np.ndarray) -> float:
    z = 10.0 * np.asarray(x, dtype=float) - 5.0
    if len(z) <= 1:
        return float(z[0] ** 2)
    weights = 10.0 ** (6.0 * np.arange(len(z)) / (len(z) - 1))
    return float(np.sum(weights * z**2))


def so_bent_cigar(x: np.ndarray) -> float:
    z = 10.0 * np.asarray(x, dtype=float) - 5.0
    return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))


def so_discus(x: np.ndarray) -> float:
    z = 10.0 * np.asarray(x, dtype=float) - 5.0
    return float(1e6 * z[0] ** 2 + np.sum(z[1:] ** 2))


def single_input_bounds(definition: dict[str, Any], dimension: int) -> list[list[float]]:
    bounds = definition.get("bounds", [0.0, 1.0])
    lo = float(bounds[0]) if isinstance(bounds, (list, tuple)) and len(bounds) >= 2 else 0.0
    hi = float(bounds[1]) if isinstance(bounds, (list, tuple)) and len(bounds) >= 2 else 1.0
    return [[lo, hi] for _ in range(max(1, int(dimension)))]


SINGLE_BENCHMARKS: dict[str, dict[str, Any]] = {
    "sphere": {"name": "Sphere", "suite": "BBOB basic", "description": "Convex quadratic bowl; useful for sanity checks.", "defaultDimension": 5, "minDimension": 1, "maxDimension": 100, "bounds": [-5.0, 5.0], "evaluate": so_sphere, "optimum": 0.0},
    "rastrigin": {"name": "Rastrigin", "suite": "CEC/BBOB classic", "description": "Highly multimodal separable landscape.", "defaultDimension": 5, "minDimension": 1, "maxDimension": 80, "bounds": [-5.12, 5.12], "evaluate": so_rastrigin, "optimum": 0.0},
    "rosenbrock": {"name": "Rosenbrock", "suite": "CEC/BBOB classic", "description": "Curved valley with strong variable interaction.", "defaultDimension": 5, "minDimension": 2, "maxDimension": 80, "bounds": [-2.0, 2.0], "evaluate": so_rosenbrock, "optimum": 0.0},
    "ackley": {"name": "Ackley", "suite": "CEC classic", "description": "Multimodal basin with many shallow local structures.", "defaultDimension": 5, "minDimension": 1, "maxDimension": 80, "bounds": [-5.0, 5.0], "evaluate": so_ackley, "optimum": 0.0},
    "griewank": {"name": "Griewank", "suite": "CEC classic", "description": "Broad landscape with product interaction ripples.", "defaultDimension": 5, "minDimension": 1, "maxDimension": 80, "bounds": [-6.0, 6.0], "evaluate": so_griewank, "optimum": 0.0},
    "schwefel": {"name": "Schwefel", "suite": "CEC classic", "description": "Deceptive multimodal function with distant global optimum.", "defaultDimension": 10, "minDimension": 2, "maxDimension": 100, "bounds": [-500.0, 500.0], "evaluate": so_schwefel, "optimum": 0.0},
    "levy": {"name": "Levy", "suite": "CEC classic", "description": "Multimodal function with sinusoidal valleys and broad plateaus.", "defaultDimension": 10, "minDimension": 2, "maxDimension": 100, "bounds": [-10.0, 10.0], "evaluate": so_levy, "optimum": 0.0},
    "zakharov": {"name": "Zakharov", "suite": "CEC classic", "description": "Unimodal non-separable function with polynomial conditioning.", "defaultDimension": 10, "minDimension": 2, "maxDimension": 100, "bounds": [-5.0, 5.0], "evaluate": so_zakharov, "optimum": 0.0},
    "bbob_ellipsoid": {"name": "BBOB Ellipsoid", "suite": "BBOB", "description": "Ill-conditioned separable quadratic benchmark.", "defaultDimension": 10, "minDimension": 2, "maxDimension": 100, "bounds": [-5.0, 5.0], "evaluate": so_ellipsoid, "optimum": 0.0},
    "bbob_bent_cigar": {"name": "BBOB Bent Cigar", "suite": "BBOB", "description": "Strongly ill-conditioned single-axis valley.", "defaultDimension": 10, "minDimension": 2, "maxDimension": 100, "bounds": [-5.0, 5.0], "evaluate": so_bent_cigar, "optimum": 0.0},
    "bbob_discus": {"name": "BBOB Discus", "suite": "BBOB", "description": "One dominant stiff axis with many softer directions.", "defaultDimension": 10, "minDimension": 2, "maxDimension": 100, "bounds": [-5.0, 5.0], "evaluate": so_discus, "optimum": 0.0},
}

SINGLE_METHODS: dict[str, dict[str, str]] = {
    "random_search": {"name": "Random search", "shortName": "RS", "description": "Uniform population sampling baseline."},
    "hill_climb": {"name": "Adaptive hill climb", "shortName": "HC", "description": "Mutates the current elite and shrinks steps after failures."},
    "de": {"name": "Differential evolution", "shortName": "DE", "description": "Population vector-difference search with greedy selection."},
    "pso": {"name": "Particle swarm optimization", "shortName": "PSO", "description": "Classic PSO with inertia, cognitive, and social attraction."},
    "ga": {"name": "Genetic algorithm", "shortName": "GA", "description": "Tournament selection, blend crossover, mutation, and elitist archive sampling."},
    "rbf_bo": {"name": "Bayesian optimization (RBF-LCB)", "shortName": "BO", "description": "RBF surrogate with lower-confidence-bound candidate selection."},
    "cma_es": {"name": "CMA-ES", "shortName": "CMA-ES", "description": "Covariance-adapted Gaussian search around elite solutions."},
    "nelder_mead": {"name": "Nelder-Mead", "shortName": "NM", "description": "Simplex-style local search with reflection and contraction."},
    "simulated_annealing": {"name": "Simulated annealing", "shortName": "SA", "description": "Temperature-controlled local perturbation search."},
    "pattern_search": {"name": "Pattern search", "shortName": "PS", "description": "Coordinate pattern search around the current elite solution."},
    "coordinate_search": {"name": "Coordinate search", "shortName": "CS", "description": "Axis-aligned local search over individual variables."},
    "cem": {"name": "Cross-entropy method", "shortName": "CEM", "description": "Elite-sampling search that fits a Gaussian to good solutions."},
    "tpe": {"name": "TPE-like search", "shortName": "TPE", "description": "Tree-structured Parzen style density search using good and bad sets."},
    "trust_region_bo": {"name": "Trust-region BO", "shortName": "TR-BO", "description": "Local surrogate BO with a shrinking trust region around the current best."},
    "random_restart_search": {"name": "Random restart search", "shortName": "RR", "description": "Multiple localized restarts around the best solution plus random exploration."},
    "local_rbf_bo": {"name": "Local RBF BO", "shortName": "LBO", "description": "A local surrogate BO baseline that concentrates sampling around the best point."},
}


CORE_SINGLE_BENCHMARKS = dict(SINGLE_BENCHMARKS)
CORE_SINGLE_METHODS = dict(SINGLE_METHODS)
SINGLE_METHOD_RUNNERS: dict[str, Any] = {}


def sync_single_objective_registries() -> None:
    SINGLE_BENCHMARKS.clear()
    SINGLE_BENCHMARKS.update(CORE_SINGLE_BENCHMARKS)
    SINGLE_BENCHMARKS.update(REGISTERED_SINGLE_BENCHMARKS)
    SINGLE_METHODS.clear()
    SINGLE_METHODS.update(CORE_SINGLE_METHODS)
    SINGLE_METHODS.update(REGISTERED_SINGLE_METHODS)
    SINGLE_METHOD_RUNNERS.clear()
    SINGLE_METHOD_RUNNERS.update(REGISTERED_SINGLE_RUNNERS)


@dataclass
class SingleObjectiveState:
    seed: int = 17
    benchmark: str = "rastrigin"
    method: str = "random_search"
    settings: dict[str, int] = field(default_factory=lambda: {"populationSize": 18, "maxEvaluations": 120, "dimension": 5})
    x: list[list[float]] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    iteration: int = 0
    last_strategy: str = "initial"
    last_candidates: list[dict[str, Any]] = field(default_factory=list)
    sigma: float = 0.18
    pso_velocity: list[list[float]] = field(default_factory=list)
    pso_position: list[list[float]] = field(default_factory=list)
    pso_best_x: list[list[float]] = field(default_factory=list)
    pso_best_y: list[float] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


class SingleObjectiveDemo:
    RBFPosterior = RBFPosterior

    def __init__(self, auto_reset: bool = True) -> None:
        self.rng = np.random.default_rng(17)
        self.state = SingleObjectiveState()
        if auto_reset:
            self.reset()

    def record_event(self, message: str) -> None:
        self.state.events.append(message)

    def should_record_iteration(self, iteration: int, before_evals: int, after_evals: int, before_best: float, after_best: float) -> bool:
        budget = max(1, int(self.state.settings["maxEvaluations"]))
        milestones = {1, 2, 5, 10}
        milestones.update(max(1, round(budget * ratio / max(1, self.state.settings["populationSize"]))) for ratio in (0.25, 0.5, 0.75, 1.0))
        crossed_budget_mark = any(before_evals < round(budget * ratio) <= after_evals for ratio in (0.25, 0.5, 0.75, 1.0))
        delta = before_best - after_best
        significant_improvement = delta > max(1e-12, abs(before_best) * 0.01)
        return significant_improvement or iteration in milestones or crossed_budget_mark

    def sanitize_settings(self, raw: dict[str, Any] | None = None, benchmark: str | None = None) -> dict[str, int]:
        raw = raw or {}
        definition = SINGLE_BENCHMARKS.get(benchmark or self.state.benchmark, SINGLE_BENCHMARKS["rastrigin"])

        def number(name: str, fallback: int) -> float:
            value = raw.get(name, fallback)
            if value in {None, ""}:
                return float(fallback)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(fallback)

        dimension = int(np.clip(round(number("dimension", int(definition["defaultDimension"]))), int(definition["minDimension"]), int(definition["maxDimension"])))
        population = int(np.clip(round(number("populationSize", self.state.settings.get("populationSize", 18))), 1, 300))
        max_evaluations = int(np.clip(round(number("maxEvaluations", self.state.settings.get("maxEvaluations", 120))), population, 10000))
        return {"populationSize": population, "maxEvaluations": max_evaluations, "dimension": dimension}

    def evaluate(self, rows: np.ndarray) -> np.ndarray:
        fn = SINGLE_BENCHMARKS[self.state.benchmark]["evaluate"]
        return np.array([float(fn(row)) for row in np.atleast_2d(rows)], dtype=float)

    def budget_remaining(self) -> int:
        return max(0, self.state.settings["maxEvaluations"] - len(self.state.x))

    def budgeted_batch_size(self, requested: int | None = None) -> int:
        size = requested if requested is not None else self.state.settings["populationSize"]
        return max(0, min(max(1, int(size)), self.budget_remaining()))

    def reset(
        self,
        seed: int = 17,
        benchmark: str | None = None,
        method: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        benchmark = benchmark if benchmark in SINGLE_BENCHMARKS else "rastrigin"
        method = method if method in SINGLE_METHODS else "random_search"
        clean_settings = self.sanitize_settings(settings, benchmark)
        self.rng = np.random.default_rng(seed)
        initial_x = latin_hypercube(clean_settings["populationSize"], clean_settings["dimension"], self.rng)
        self.state = SingleObjectiveState(seed=seed, benchmark=benchmark, method=method, settings=clean_settings)
        self.state.x = initial_x.tolist()
        self.state.y = self.evaluate(initial_x).tolist()
        self.state.pso_position = initial_x.tolist()
        self.state.pso_velocity = self.rng.normal(0.0, 0.08, size=initial_x.shape).tolist()
        self.state.pso_best_x = initial_x.tolist()
        self.state.pso_best_y = list(self.state.y)
        self.state.last_strategy = f"{SINGLE_METHODS[method]['shortName']} initialization"
        self.state.events = [
            f"Initialized {clean_settings['populationSize']} evaluations on {SINGLE_BENCHMARKS[benchmark]['name']}.",
            f"Budget: {clean_settings['maxEvaluations']} evaluations, d={clean_settings['dimension']}.",
            f"Selected method: {SINGLE_METHODS[method]['name']}.",
        ]
        return self.snapshot()

    def set_benchmark(self, benchmark: str) -> dict[str, Any]:
        definition = SINGLE_BENCHMARKS.get(benchmark, SINGLE_BENCHMARKS["rastrigin"])
        settings = {**self.state.settings, "dimension": int(definition["defaultDimension"])}
        return self.reset(random.randint(1, 99999), benchmark, self.state.method, settings)

    def set_method(self, method: str) -> dict[str, Any]:
        return self.reset(random.randint(1, 99999), self.state.benchmark, method, self.state.settings)

    def set_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        return self.reset(random.randint(1, 99999), self.state.benchmark, self.state.method, settings)

    def best_index(self) -> int:
        return int(np.argmin(np.array(self.state.y, dtype=float))) if self.state.y else -1

    def elite_population(self) -> np.ndarray:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)
        keep = min(len(x), max(4, self.state.settings["populationSize"]))
        return x[np.argsort(y)[:keep]]

    def propose_random_search(self, batch_size: int) -> tuple[np.ndarray, str]:
        return self.rng.random((batch_size, self.state.settings["dimension"])), "uniform sampling"

    def propose_hill_climb(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        best = x[self.best_index()]
        spread = max(0.025, float(self.state.sigma))
        probes = np.clip(best + self.rng.normal(0.0, spread, size=(batch_size, len(best))), 0.0, 1.0)
        return probes, f"elite mutation, sigma={spread:.3f}"

    def propose_de(self, batch_size: int) -> tuple[np.ndarray, str]:
        archive = self.elite_population()
        dim = self.state.settings["dimension"]
        if len(archive) < 4:
            return self.propose_hill_climb(batch_size)
        candidates = []
        for _ in range(batch_size):
            ids = self.rng.choice(len(archive), size=3, replace=False)
            mutant = archive[ids[0]] + 0.72 * (archive[ids[1]] - archive[ids[2]])
            target = archive[int(self.rng.integers(0, len(archive)))]
            cross = self.rng.random(dim) < 0.82
            if not np.any(cross):
                cross[int(self.rng.integers(0, dim))] = True
            candidates.append(np.clip(np.where(cross, mutant, target), 0.0, 1.0))
        return np.array(candidates), "elite differential mutation"

    def propose_pso(self, batch_size: int) -> tuple[np.ndarray, str]:
        dim = self.state.settings["dimension"]
        swarm_size = max(batch_size, self.state.settings["populationSize"])
        position = np.array(self.state.pso_position, dtype=float)
        velocity = np.array(self.state.pso_velocity, dtype=float)
        personal_best = np.array(self.state.pso_best_x, dtype=float)
        personal_y = np.array(self.state.pso_best_y, dtype=float)
        if len(position) < swarm_size:
            extra = swarm_size - len(position)
            extra_position = self.rng.random((extra, dim))
            position = np.vstack([position, extra_position])
            velocity = np.vstack([velocity, self.rng.normal(0.0, 0.08, size=(extra, dim))])
            personal_best = np.vstack([personal_best, extra_position])
            personal_y = np.concatenate([personal_y, self.evaluate(extra_position)])
        global_best = personal_best[int(np.argmin(personal_y))]
        r1 = self.rng.random(position.shape)
        r2 = self.rng.random(position.shape)
        velocity = 0.72 * velocity + 1.49 * r1 * (personal_best - position) + 1.49 * r2 * (global_best - position)
        velocity = np.clip(velocity, -0.25, 0.25)
        position = np.clip(position + velocity, 0.0, 1.0)
        self.state.pso_position = position.tolist()
        self.state.pso_velocity = velocity.tolist()
        return position[:batch_size], "inertia+cognitive+social velocity update"

    def propose_ga(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)
        dim = self.state.settings["dimension"]
        elite_count = min(len(x), max(4, self.state.settings["populationSize"]))
        elite = x[np.argsort(y)[:elite_count]]

        def tournament() -> np.ndarray:
            ids = self.rng.choice(len(x), size=min(3, len(x)), replace=False)
            return x[ids[int(np.argmin(y[ids]))]]

        children = []
        if len(elite):
            children.append(elite[0].copy())
        while len(children) < batch_size:
            parent_a = tournament()
            parent_b = tournament()
            alpha = self.rng.random(dim)
            child = alpha * parent_a + (1.0 - alpha) * parent_b
            mutate = self.rng.random(dim) < 0.18
            child = np.clip(child + mutate * self.rng.normal(0.0, max(0.035, self.state.sigma), size=dim), 0.0, 1.0)
            children.append(child)
        return np.array(children[:batch_size]), "tournament selection + blend crossover + mutation"

    def propose_rbf_bo(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)[:, None]
        dim = self.state.settings["dimension"]
        if len(x) < max(6, dim + 2):
            return self.propose_hill_climb(batch_size)
        posterior = RBFPosterior(x, y, lengthscale=0.32, clip_mean=False)
        pool = self.rng.random((max(650, batch_size * 120), dim))
        best = x[self.best_index()]
        local = np.clip(best + self.rng.normal(0.0, max(0.05, self.state.sigma), size=(len(pool) // 3, dim)), 0.0, 1.0)
        pool[: len(local)] = local
        mean, std = posterior.predict(pool)
        score = mean[:, 0] - 1.6 * std[:, 0]
        chosen = self.diverse_top_k(pool, -score, batch_size)
        return pool[chosen], "RBF lower-confidence-bound"

    def propose_cma_es(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)
        best = x[self.best_index()]
        elite = self.elite_population()
        cov = np.cov(elite.T) if len(elite) > 1 else np.eye(len(best)) * 0.05
        cov = np.asarray(cov, dtype=float) + np.eye(len(best)) * 1e-6
        sigma = max(0.04, float(self.state.sigma))
        samples = self.rng.multivariate_normal(best, cov * sigma, size=batch_size)
        return np.clip(samples, 0.0, 1.0), "covariance-adapted Gaussian sampling"

    def propose_nelder_mead(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        best = x[self.best_index()]
        dim = len(best)
        simplex = [best]
        scale = max(0.04, float(self.state.sigma))
        for i in range(dim):
            vertex = best.copy()
            vertex[i] = np.clip(vertex[i] + scale, 0.0, 1.0)
            simplex.append(vertex)
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = np.clip(centroid + (centroid - simplex[-1]), 0.0, 1.0)
        contracted = np.clip(centroid + 0.5 * (simplex[-1] - centroid), 0.0, 1.0)
        candidates = [reflected, contracted]
        while len(candidates) < batch_size:
            candidates.append(np.clip(best + self.rng.normal(0.0, scale, size=dim), 0.0, 1.0))
        return np.array(candidates[:batch_size]), "simplex reflection and contraction"

    def propose_simulated_annealing(self, batch_size: int) -> tuple[np.ndarray, str]:
        best = np.array(self.state.x, dtype=float)[self.best_index()]
        temp = max(0.02, float(self.state.sigma)) * max(0.18, 0.98 ** self.state.iteration)
        rows = np.clip(best + self.rng.normal(0.0, temp, size=(batch_size, len(best))), 0.0, 1.0)
        return rows, f"temperature={temp:.4f}"

    def propose_pattern_search(self, batch_size: int) -> tuple[np.ndarray, str]:
        best = np.array(self.state.x, dtype=float)[self.best_index()]
        dim = len(best)
        step = max(0.03, float(self.state.sigma))
        rows = [best.copy()]
        for i in range(dim):
            plus = best.copy()
            minus = best.copy()
            plus[i] = np.clip(plus[i] + step, 0.0, 1.0)
            minus[i] = np.clip(minus[i] - step, 0.0, 1.0)
            rows.extend([plus, minus])
        while len(rows) < batch_size:
            rows.append(np.clip(best + self.rng.normal(0.0, step, size=dim), 0.0, 1.0))
        return np.array(rows[:batch_size]), "coordinate pattern expansion"

    def propose_coordinate_search(self, batch_size: int) -> tuple[np.ndarray, str]:
        best = np.array(self.state.x, dtype=float)[self.best_index()]
        dim = len(best)
        step = max(0.03, float(self.state.sigma))
        rows = []
        for i in range(batch_size):
            candidate = best.copy()
            axis = i % dim
            direction = -1.0 if (i // dim) % 2 else 1.0
            candidate[axis] = np.clip(candidate[axis] + direction * step, 0.0, 1.0)
            rows.append(candidate)
        return np.array(rows), "axis-wise coordinate sweeps"

    def propose_cem(self, batch_size: int) -> tuple[np.ndarray, str]:
        elite = self.elite_population()
        mean = elite.mean(axis=0)
        cov = np.cov(elite.T) if len(elite) > 1 else np.eye(len(mean)) * 0.04
        cov = np.asarray(cov, dtype=float) + np.eye(len(mean)) * 1e-6
        samples = self.rng.multivariate_normal(mean, cov * max(0.03, self.state.sigma), size=batch_size)
        return np.clip(samples, 0.0, 1.0), "elite distribution fitting"

    def propose_tpe(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)
        cutoff = np.quantile(y, 0.3)
        good = x[y <= cutoff]
        bad = x[y > cutoff]
        if len(good) == 0:
            good = self.elite_population()
        if len(bad) == 0:
            bad = x
        mean_good = good.mean(axis=0)
        mean_bad = bad.mean(axis=0)
        rows = []
        for _ in range(batch_size):
            blend = np.clip(mean_good + self.rng.normal(0.0, max(0.03, self.state.sigma), size=len(mean_good)), 0.0, 1.0)
            if self.rng.random() < 0.4:
                blend = np.clip(blend + 0.25 * (mean_good - mean_bad), 0.0, 1.0)
            rows.append(blend)
        return np.array(rows), "good/bad density ratio search"

    def propose_trust_region_bo(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)[:, None]
        dim = self.state.settings["dimension"]
        if len(x) < max(6, dim + 2):
            return self.propose_rbf_bo(batch_size)
        best = x[self.best_index()]
        radius = min(0.35, max(0.05, float(self.state.sigma) * (0.92 ** self.state.iteration)))
        pool = np.clip(best + self.rng.normal(0.0, radius, size=(max(260, batch_size * 80), dim)), 0.0, 1.0)
        posterior = RBFPosterior(x, y, lengthscale=max(0.2, radius * 1.8), clip_mean=False)
        mean, std = posterior.predict(pool)
        score = mean[:, 0] - 1.5 * std[:, 0] + 0.2 * np.linalg.norm(pool - best, axis=1)
        chosen = self.diverse_top_k(pool, -score, batch_size)
        return pool[chosen], f"trust radius={radius:.4f}"

    def propose_random_restart_search(self, batch_size: int) -> tuple[np.ndarray, str]:
        best = np.array(self.state.x, dtype=float)[self.best_index()]
        rows = []
        for i in range(batch_size):
            if i % 3 == 0:
                rows.append(self.rng.random(len(best)))
            else:
                scale = max(0.03, 0.10 * (0.95 ** self.state.iteration))
                rows.append(np.clip(best + self.rng.normal(0.0, scale, size=len(best)), 0.0, 1.0))
        return np.array(rows), "random restarts around elite"

    def propose_local_rbf_bo(self, batch_size: int) -> tuple[np.ndarray, str]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)[:, None]
        if len(x) < max(6, self.state.settings["dimension"] + 2):
            return self.propose_hill_climb(batch_size)
        best = x[self.best_index()]
        posterior = RBFPosterior(x, y, lengthscale=0.24, clip_mean=False)
        pool = np.clip(best + self.rng.normal(0.0, max(0.06, self.state.sigma), size=(max(350, batch_size * 100), len(best))), 0.0, 1.0)
        mean, std = posterior.predict(pool)
        score = mean[:, 0] - 1.25 * std[:, 0] + 0.25 * np.linalg.norm(pool - best, axis=1)
        chosen = self.diverse_top_k(pool, -score, batch_size)
        return pool[chosen], "local surrogate LCB"

    def diverse_top_k(self, pool: np.ndarray, scores: np.ndarray, k: int) -> list[int]:
        order = list(np.argsort(scores)[::-1])
        chosen: list[int] = []
        for idx in order:
            if not chosen or np.linalg.norm(pool[idx] - pool[chosen], axis=1).min() > 0.08:
                chosen.append(int(idx))
            if len(chosen) == k:
                return chosen
        return chosen + [int(idx) for idx in order if int(idx) not in chosen][: max(0, k - len(chosen))]

    def step(self, batch_size: int | None = None) -> dict[str, Any]:
        actual = self.budgeted_batch_size(batch_size)
        if actual <= 0:
            return self.snapshot()
        before_evals = len(self.state.x)
        before_best = float(np.min(self.state.y))
        registered_runner = SINGLE_METHOD_RUNNERS.get(self.state.method)
        method_map = {
            "hill_climb": self.propose_hill_climb,
            "de": self.propose_de,
            "pso": self.propose_pso,
            "ga": self.propose_ga,
            "rbf_bo": self.propose_rbf_bo,
            "cma_es": self.propose_cma_es,
            "nelder_mead": self.propose_nelder_mead,
            "simulated_annealing": self.propose_simulated_annealing,
            "pattern_search": self.propose_pattern_search,
            "coordinate_search": self.propose_coordinate_search,
            "cem": self.propose_cem,
            "tpe": self.propose_tpe,
            "trust_region_bo": self.propose_trust_region_bo,
            "random_restart_search": self.propose_random_restart_search,
            "local_rbf_bo": self.propose_local_rbf_bo,
        }
        if registered_runner is not None:
            chosen_x, strategy = registered_runner(self, actual)
        else:
            chosen_x, strategy = method_map.get(self.state.method, self.propose_random_search)(actual)
        chosen_x = np.atleast_2d(np.asarray(chosen_x, dtype=float))
        if chosen_x.shape[1] != self.state.settings["dimension"]:
            chosen_x = np.resize(chosen_x, (len(chosen_x), self.state.settings["dimension"]))
        chosen_x = np.clip(chosen_x[:actual], 0.0, 1.0)
        chosen_y = self.evaluate(chosen_x)
        if self.state.method == "pso":
            personal_best = np.array(self.state.pso_best_x, dtype=float)
            personal_y = np.array(self.state.pso_best_y, dtype=float)
            if len(personal_best) < len(chosen_x):
                extra = len(chosen_x) - len(personal_best)
                personal_best = np.vstack([personal_best, chosen_x[-extra:]])
                personal_y = np.concatenate([personal_y, chosen_y[-extra:]])
            for i, value in enumerate(chosen_y):
                if i < len(personal_y) and float(value) < float(personal_y[i]):
                    personal_best[i] = chosen_x[i]
                    personal_y[i] = float(value)
            self.state.pso_best_x = personal_best.tolist()
            self.state.pso_best_y = personal_y.tolist()
        self.state.x.extend(chosen_x.tolist())
        self.state.y.extend(chosen_y.tolist())
        self.state.iteration += 1
        after_best = float(np.min(self.state.y))
        if after_best < before_best:
            self.state.sigma = min(0.35, self.state.sigma * 1.04)
        else:
            self.state.sigma = max(0.025, self.state.sigma * 0.82)
        self.state.last_strategy = f"{SINGLE_METHODS[self.state.method]['shortName']}: {strategy}"
        self.state.last_candidates = [
            {"x": [float(v) for v in chosen_x[i]], "value": float(chosen_y[i]), "improved": float(chosen_y[i]) <= before_best}
            for i in range(len(chosen_x))
        ]
        delta = before_best - after_best
        if self.should_record_iteration(self.state.iteration, before_evals, len(self.state.x), before_best, after_best):
            improvement = f", improved by {delta:.4g}" if delta > 0 else ", no best improvement"
            self.record_event(f"Round {self.state.iteration} ({len(self.state.x)}/{self.state.settings['maxEvaluations']} evals): {self.state.last_strategy}; best={after_best:.6g}{improvement}.")
        return self.snapshot()

    def run_to_budget(self) -> dict[str, Any]:
        while self.budget_remaining() > 0:
            before = len(self.state.x)
            self.step(min(self.state.settings["populationSize"], self.budget_remaining()))
            if len(self.state.x) <= before:
                break
        self.record_event(f"Budget completed: {len(self.state.x)}/{self.state.settings['maxEvaluations']} evaluations.")
        return self.snapshot()

    def convergence(self) -> list[dict[str, float]]:
        best = math.inf
        rows: list[dict[str, float]] = []
        for idx, value in enumerate(self.state.y, start=1):
            best = min(best, float(value))
            rows.append({"evaluation": idx, "value": float(value), "best": float(best)})
        return rows

    def landscape(self, resolution: int = 96) -> dict[str, Any]:
        dim = int(self.state.settings["dimension"])
        resolution = int(np.clip(resolution, 48, 120))
        base = np.full(dim, 0.5, dtype=float)
        best_idx = self.best_index()
        if best_idx >= 0 and self.state.x:
            base = np.array(self.state.x[best_idx], dtype=float)
        if dim <= 1:
            xs = np.linspace(0.0, 1.0, resolution)
            rows = np.repeat(base[None, :], resolution, axis=0)
            rows[:, 0] = xs
            values = self.evaluate(rows)
            return {
                "kind": "1d",
                "dimension": dim,
                "x": [float(v) for v in xs],
                "values": [float(v) for v in values],
                "minValue": float(np.min(values)),
                "maxValue": float(np.max(values)),
                "fixedX": [float(v) for v in base],
            }

        xs = np.linspace(0.0, 1.0, resolution)
        ys = np.linspace(0.0, 1.0, resolution)
        rows = np.repeat(base[None, :], resolution * resolution, axis=0)
        grid_x, grid_y = np.meshgrid(xs, ys)
        rows[:, 0] = grid_x.ravel()
        rows[:, 1] = grid_y.ravel()
        values = self.evaluate(rows).reshape(resolution, resolution)
        return {
            "kind": "2d",
            "dimension": dim,
            "x": [float(v) for v in xs],
            "y": [float(v) for v in ys],
            "values": [[float(v) for v in row] for row in values.tolist()],
            "minValue": float(np.min(values)),
            "maxValue": float(np.max(values)),
            "fixedX": [float(v) for v in base],
            "sliceNote": "x3..xd fixed at the current best vector" if dim > 2 else "full 2D landscape",
        }

    def snapshot(self) -> dict[str, Any]:
        x = np.array(self.state.x, dtype=float)
        y = np.array(self.state.y, dtype=float)
        best_idx = self.best_index()
        best_value = float(y[best_idx]) if best_idx >= 0 else 0.0
        recent_start = max(0, len(x) - self.state.settings["populationSize"])
        points = [
            {
                "id": i,
                "x": [float(v) for v in x[i]],
                "value": float(y[i]),
                "best": i == best_idx,
                "recent": i >= recent_start,
            }
            for i in range(len(x))
        ]
        return {
            "benchmarks": [
                {k: v for k, v in value.items() if k != "evaluate"} | {"key": key}
                for key, value in SINGLE_BENCHMARKS.items()
            ],
            "methods": [{"key": key, **value} for key, value in SINGLE_METHODS.items()],
            "moduleRegistry": {
                "localSingleBenchmarkDir": str(ROOT / "single_benchmarks"),
                "localSingleMethodDir": str(ROOT / "single_methods"),
                "registryErrors": list(REGISTRY_ERRORS),
            },
            "experimentLimits": dict(EXPERIMENT_LIMITS),
            "benchmark": {k: v for k, v in SINGLE_BENCHMARKS[self.state.benchmark].items() if k != "evaluate"} | {"key": self.state.benchmark, "inputBounds": single_input_bounds(SINGLE_BENCHMARKS[self.state.benchmark], self.state.settings["dimension"])},
            "method": {"key": self.state.method, **SINGLE_METHODS[self.state.method]},
            "settings": dict(self.state.settings),
            "dimensions": {"inputs": self.state.settings["dimension"], "objectives": 1},
            "budgetRemaining": self.budget_remaining(),
            "nEvaluations": len(points),
            "iteration": self.state.iteration,
            "bestValue": best_value,
            "bestX": [float(v) for v in x[best_idx]] if best_idx >= 0 else [],
            "meanValue": float(np.mean(y)) if len(y) else 0.0,
            "lastStrategy": self.state.last_strategy,
            "lastCandidates": list(self.state.last_candidates),
            "points": points,
            "convergence": self.convergence(),
            "events": list(self.state.events),
        }


def single_experiment_settings(raw: dict[str, Any]) -> dict[str, int]:
    population = int(np.clip(numeric_setting(raw, "populationSize", 18), 1, EXPERIMENT_LIMITS["maxPopulation"]))
    max_evaluations = int(np.clip(numeric_setting(raw, "maxEvaluations", 120), population, EXPERIMENT_LIMITS["maxBudget"]))
    return {
        "populationSize": population,
        "maxEvaluations": max_evaluations,
        "dimension": int(np.clip(numeric_setting(raw, "dimension", 5), 1, 200)),
    }


def single_experiment_row(
    benchmark_key: str,
    method_key: str,
    seed: int,
    repeat: int,
    snap: dict[str, Any],
    elapsed_ms: int,
) -> dict[str, Any]:
    optimum = float(snap.get("benchmark", {}).get("optimum", 0.0) or 0.0)
    best_value = float(snap.get("bestValue", 0.0))
    return {
        "benchmark": benchmark_key,
        "benchmarkName": snap["benchmark"]["name"],
        "optimizer": method_key,
        "optimizerName": snap["method"]["name"],
        "method": method_key,
        "methodName": snap["method"]["name"],
        "seed": seed,
        "repeat": repeat,
        "status": "ok",
        "dimension": snap["dimensions"]["inputs"],
        "inputBounds": snap["benchmark"].get("inputBounds", []),
        "objectives": 1,
        "populationSize": snap["settings"]["populationSize"],
        "maxEvaluations": snap["settings"]["maxEvaluations"],
        "evaluations": snap["nEvaluations"],
        "rounds": snap["iteration"],
        "bestValue": best_value,
        "optimum": optimum,
        "optimumGap": max(0.0, best_value - optimum),
        "meanValue": float(snap.get("meanValue", 0.0)),
        "elapsedMs": elapsed_ms,
    }


def run_single_experiment_trial(
    benchmark: str,
    method: str,
    seed: int,
    repeat: int,
    settings: dict[str, int],
) -> dict[str, Any]:
    run_started = time.perf_counter()
    try:
        demo = SingleObjectiveDemo(auto_reset=False)
        snap = demo.reset(seed=seed, benchmark=benchmark, method=method, settings=settings)
        snap = demo.run_to_budget()
        return single_experiment_row(benchmark, method, seed, repeat, snap, int((time.perf_counter() - run_started) * 1000))
    except Exception as exc:
        definition = SINGLE_BENCHMARKS.get(benchmark, SINGLE_BENCHMARKS["rastrigin"])
        return {
            "benchmark": benchmark,
            "benchmarkName": definition["name"],
            "optimizer": method,
            "optimizerName": SINGLE_METHODS.get(method, {}).get("name", method),
            "method": method,
            "methodName": SINGLE_METHODS.get(method, {}).get("name", method),
            "seed": seed,
            "repeat": repeat,
            "status": "error",
            "dimension": settings["dimension"],
            "inputBounds": [],
            "objectives": 1,
            "populationSize": settings["populationSize"],
            "maxEvaluations": settings["maxEvaluations"],
            "evaluations": 0,
            "rounds": 0,
            "bestValue": None,
            "optimum": float(definition.get("optimum", 0.0) or 0.0),
            "optimumGap": None,
            "meanValue": None,
            "elapsedMs": int((time.perf_counter() - run_started) * 1000),
            "error": str(exc),
        }


def summarize_single_experiment(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    wins: dict[tuple[str, str], int] = {}
    groups_by_trial: dict[tuple[str, int], list[dict[str, Any]]] = {}
    direction = metric_direction(metric)
    for row in ok_rows:
        groups_by_trial.setdefault((row["benchmark"], int(row["seed"])), []).append(row)
    for trial_rows in groups_by_trial.values():
        if not trial_rows:
            continue
        best_score = max(direction * float(row.get(metric, 0.0)) for row in trial_rows)
        for row in [item for item in trial_rows if abs(direction * float(item.get(metric, 0.0)) - best_score) <= 1e-12]:
            key = (row["benchmark"], row["optimizer"])
            wins[key] = wins.get(key, 0) + 1

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ok_rows:
        groups.setdefault((row["benchmark"], row["optimizer"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (benchmark, method), group in sorted(groups.items()):
        best = mean_std([float(row["bestValue"]) for row in group])
        gap = mean_std([float(row["optimumGap"]) for row in group])
        mean_value = mean_std([float(row["meanValue"]) for row in group])
        elapsed = mean_std([float(row["elapsedMs"]) for row in group])
        evals = mean_std([float(row["evaluations"]) for row in group])
        metric_stats = mean_std([float(row.get(metric, 0.0)) for row in group])
        summary.append(
            {
                "benchmark": benchmark,
                "benchmarkName": group[0]["benchmarkName"],
                "optimizer": method,
                "optimizerName": group[0]["optimizerName"],
                "method": method,
                "methodName": group[0]["methodName"],
                "runs": len(group),
                "bestValueMean": best["mean"],
                "bestValueStd": best["std"],
                "optimumGapMean": gap["mean"],
                "optimumGapStd": gap["std"],
                "meanValueMean": mean_value["mean"],
                "meanValueStd": mean_value["std"],
                "evaluationsMean": evals["mean"],
                "elapsedMsMean": elapsed["mean"],
                "elapsedMsStd": elapsed["std"],
                "selectedMetricMean": metric_stats["mean"],
                "selectedMetricStd": metric_stats["std"],
                "metricWins": wins.get((benchmark, method), 0),
            }
        )

    for benchmark in sorted({row["benchmark"] for row in ok_rows}):
        subset = [item for item in summary if item["benchmark"] == benchmark]
        best_order = sorted(subset, key=lambda item: item["bestValueMean"])
        gap_order = sorted(subset, key=lambda item: item["optimumGapMean"])
        metric_order = sorted(subset, key=lambda item: direction * item["selectedMetricMean"], reverse=True)
        for rank, item in enumerate(best_order, start=1):
            item["bestValueRank"] = rank
        for rank, item in enumerate(gap_order, start=1):
            item["optimumGapRank"] = rank
        for rank, item in enumerate(metric_order, start=1):
            item["selectedMetricRank"] = rank
    return summary


def run_single_comparison_experiment(body: dict[str, Any]) -> dict[str, Any]:
    current = SINGLE_DEMO.snapshot()
    benchmarks = selected_keys(body.get("benchmarks"), SINGLE_BENCHMARKS, [current["benchmark"]["key"]])
    methods = selected_keys(body.get("methods"), SINGLE_METHODS, [current["method"]["key"]])
    repeats = int(np.clip(numeric_setting(body, "repeats", 3), 1, EXPERIMENT_LIMITS["maxRepeats"]))
    settings = single_experiment_settings(dict(body.get("settings", {})))
    total_runs = len(benchmarks) * len(methods) * repeats
    if total_runs > EXPERIMENT_LIMITS["maxRuns"]:
        raise ValueError(f"Experiment grid is too large: {total_runs} runs; limit is {EXPERIMENT_LIMITS['maxRuns']}.")

    seed0 = numeric_setting(body, "seed", 9001)
    workers = experiment_worker_count(body, total_runs)
    metric = str(body.get("metric") or "bestValue")
    if metric not in {"bestValue", "optimumGap", "meanValue", "elapsedMs"}:
        metric = "bestValue"
    reference_method = str(body.get("referenceMethod") or body.get("referenceOptimizer") or (methods[-1] if methods else ""))
    if reference_method not in methods:
        reference_method = methods[-1] if methods else ""
    jobs = [
        {
            "order": order,
            "benchmark": benchmark,
            "method": method,
            "seed": seed0 + repeat,
            "repeat": repeat + 1,
        }
        for order, (benchmark, repeat, method) in enumerate(
            (benchmark, repeat, method)
            for benchmark in benchmarks
            for repeat in range(repeats)
            for method in methods
        )
    ]

    rows_by_order: dict[int, dict[str, Any]] = {}
    started_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_single_experiment_trial,
                job["benchmark"],
                job["method"],
                int(job["seed"]),
                int(job["repeat"]),
                settings,
            ): int(job["order"])
            for job in jobs
        }
        for future in as_completed(futures):
            rows_by_order[futures[future]] = future.result()

    rows = [rows_by_order[i] for i in range(len(jobs))]
    summary = summarize_single_experiment(rows, metric)
    paper_table = make_paper_table(rows, benchmarks, methods, metric, reference_method)
    pairwise_matrix = make_pairwise_matrix(rows, methods, metric)
    return {
        "limits": dict(EXPERIMENT_LIMITS),
        "settings": settings,
        "benchmarks": benchmarks,
        "methods": methods,
        "optimizers": methods,
        "repeats": repeats,
        "seed": seed0,
        "workers": workers,
        "metric": metric,
        "metricDirection": "higher_is_better" if metric_direction(metric) > 0 else "lower_is_better",
        "referenceMethod": reference_method,
        "referenceOptimizer": reference_method,
        "rows": rows,
        "summary": summary,
        "paperTable": paper_table,
        "pairwiseMatrix": pairwise_matrix,
        "elapsedMs": int((time.perf_counter() - started_all) * 1000),
    }


refresh_registries_if_needed(force=True)
DEMO = ParetoLiloDemo()
SINGLE_DEMO = SingleObjectiveDemo()


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "ParetoLiloDemo/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            DEMO.sync_registries()
            self._json(DEMO.snapshot())
            return
        if parsed.path == "/api/single/state":
            refresh_registries_if_needed()
            self._json(SINGLE_DEMO.snapshot())
            return
        if parsed.path == "/api/single/landscape":
            query = parse_qs(parsed.query)
            try:
                resolution = int(query.get("resolution", [96])[0])
            except (TypeError, ValueError):
                resolution = 96
            self._json(SINGLE_DEMO.landscape(resolution=resolution))
            return
        if parsed.path in {"", "/"}:
            self._file(STATIC / "index.html")
            return
        target = (STATIC / parsed.path.lstrip("/")).resolve()
        if STATIC.resolve() not in target.parents and target != STATIC.resolve():
            self.send_error(404)
            return
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return
        self._file(target)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_body()
        DEMO.sync_registries()
        if parsed.path == "/api/reset":
            self._json(
                DEMO.reset(
                    seed=int(body.get("seed", random.randint(1, 99999))),
                    feedback=body.get("feedback"),
                    benchmark=body.get("benchmark"),
                    optimizer=body.get("optimizer"),
                    settings=dict(body.get("settings", {})),
                )
            )
            return
        if parsed.path == "/api/settings":
            self._json(DEMO.set_settings(dict(body.get("settings", {}))))
            return
        if parsed.path == "/api/benchmark":
            self._json(DEMO.set_benchmark(str(body.get("benchmark", ""))))
            return
        if parsed.path == "/api/optimizer":
            self._json(DEMO.set_optimizer(str(body.get("optimizer", ""))))
            return
        if parsed.path == "/api/feedback":
            self._json(DEMO.update_feedback(str(body.get("feedback", ""))))
            return
        if parsed.path == "/api/weights":
            self._json(DEMO.update_weights(dict(body.get("weights", {}))))
            return
        if parsed.path == "/api/step":
            batch_size = body.get("batchSize")
            self._json(DEMO.step(int(batch_size) if batch_size is not None else None))
            return
        if parsed.path == "/api/run_budget":
            self._json(DEMO.run_to_budget())
            return
        if parsed.path == "/api/finish":
            self._json(
                DEMO.finish_experiment(
                    rounds=int(body.get("rounds", 0) or 0),
                    elapsed_ms=int(body.get("elapsedMs", 0) or 0),
                    note=str(body.get("note", "Budget run completed.")),
                )
            )
            return
        if parsed.path == "/api/experiment":
            try:
                self._json(run_comparison_experiment(body))
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/submit_problem":
            try:
                self._json(submit_custom_problem(body))
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/submit_optimizer":
            try:
                self._json(submit_custom_optimizer(body))
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/import":
            try:
                result = import_custom_module(body)
                DEMO.sync_registries()
                self._json({"state": DEMO.snapshot(), "importResult": result})
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/import_path":
            try:
                result = import_custom_module_from_path(body)
                DEMO.sync_registries()
                self._json({"state": DEMO.snapshot(), "importResult": result})
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/refresh":
            result = refresh_local_modules()
            DEMO.sync_registries()
            self._json({"state": DEMO.snapshot(), "refreshResult": result})
            return
        if parsed.path == "/api/single/experiment":
            try:
                self._json(run_single_comparison_experiment(body))
            except ValueError as exc:
                self._json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/single/reset":
            self._json(
                SINGLE_DEMO.reset(
                    seed=int(body.get("seed", random.randint(1, 99999))),
                    benchmark=body.get("benchmark"),
                    method=body.get("method"),
                    settings=dict(body.get("settings", {})),
                )
            )
            return
        if parsed.path == "/api/single/settings":
            self._json(SINGLE_DEMO.set_settings(dict(body.get("settings", {}))))
            return
        if parsed.path == "/api/single/benchmark":
            self._json(SINGLE_DEMO.set_benchmark(str(body.get("benchmark", ""))))
            return
        if parsed.path == "/api/single/method":
            self._json(SINGLE_DEMO.set_method(str(body.get("method", ""))))
            return
        if parsed.path == "/api/single/step":
            batch_size = body.get("batchSize")
            self._json(SINGLE_DEMO.step(int(batch_size) if batch_size is not None else None))
            return
        if parsed.path == "/api/single/run_budget":
            self._json(SINGLE_DEMO.run_to_budget())
            return
        self.send_error(404)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _file(self, path: Path) -> None:
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def make_server(start_port: int) -> tuple[ThreadingHTTPServer, int]:
    if start_port == 0:
        server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
        return server, int(server.server_address[1])
    for port in range(start_port, start_port + 20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), DemoHandler), port
        except OSError as exc:
            if getattr(exc, "winerror", None) != 10048 and getattr(exc, "errno", None) not in {48, 98, 10048}:
                raise
    raise OSError(f"No free localhost port found in {start_port}-{start_port + 19}.")


def open_local_page(url: str) -> None:
    try:
        if os.name == "nt":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            webbrowser.open(url)
    except Exception as exc:
        print(f"Could not open browser automatically: {exc}")
        print(f"Open this page manually: {url}")


def main() -> None:
    server, actual_port = make_server(PORT)
    url = f"http://127.0.0.1:{actual_port}"
    print(f"BlackBoxLab Python backend running at {url}", flush=True)
    if actual_port != PORT:
        print(f"Port {PORT} was busy, so the backend moved to {actual_port}.", flush=True)
    if OPEN_BROWSER:
        threading.Timer(0.8, lambda: open_local_page(url)).start()
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

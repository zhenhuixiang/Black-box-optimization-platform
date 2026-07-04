from __future__ import annotations

import math

import numpy as np

from .registry import register_single_benchmark


def unit(x: np.ndarray, length: int | None = None) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    if length is not None:
        z = np.resize(z, length)
    return np.clip(z, 0.0, 1.0)


def scale(x: np.ndarray, lo: float, hi: float, length: int | None = None) -> np.ndarray:
    return lo + (hi - lo) * unit(x, length)


def beale(x: np.ndarray) -> float:
    z = scale(x, -4.5, 4.5, 2)
    x1, x2 = float(z[0]), float(z[1])
    return float((1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2**2) ** 2 + (2.625 - x1 + x1 * x2**3) ** 2)


def booth(x: np.ndarray) -> float:
    z = scale(x, -10.0, 10.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    return float((x1 + 2.0 * x2 - 7.0) ** 2 + (2.0 * x1 + x2 - 5.0) ** 2)


def matyas(x: np.ndarray) -> float:
    z = scale(x, -10.0, 10.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    return float(0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2)


def mccormick(x: np.ndarray) -> float:
    z = scale(x, -1.5, 4.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    raw = math.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1.0
    return float(raw + 1.9133)


def three_hump_camel(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    return float(2.0 * x1**2 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2**2)


def six_hump_camel(x: np.ndarray) -> float:
    z = scale(x, -3.0, 3.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    raw = (4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2
    return float(raw + 1.031628453)


def branin(x: np.ndarray) -> float:
    z = unit(x, 2)
    x1 = -5.0 + 15.0 * float(z[0])
    x2 = 15.0 * float(z[1])
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    raw = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s
    return float(raw - 0.397887)


def goldstein_price(x: np.ndarray) -> float:
    z = scale(x, -2.0, 2.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    a = 1.0 + (x1 + x2 + 1.0) ** 2 * (19.0 - 14.0 * x1 + 3.0 * x1**2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2**2)
    b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (18.0 - 32.0 * x1 + 12.0 * x1**2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2**2)
    return float(a * b - 3.0)


def forrester(x: np.ndarray) -> float:
    value = float(unit(x, 1)[0])
    raw = (6.0 * value - 2.0) ** 2 * math.sin(12.0 * value - 4.0)
    return float(raw + 6.02074)


def gramacy_lee(x: np.ndarray) -> float:
    value = 0.5 + 2.0 * float(unit(x, 1)[0])
    raw = math.sin(10.0 * math.pi * value) / (2.0 * value) + (value - 1.0) ** 4
    return float(raw + 0.869011)


def bohachevsky(x: np.ndarray) -> float:
    z = scale(x, -100.0, 100.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    return float(x1**2 + 2.0 * x2**2 - 0.3 * math.cos(3.0 * math.pi * x1) - 0.4 * math.cos(4.0 * math.pi * x2) + 0.7)


def schubert(x: np.ndarray) -> float:
    z = scale(x, -10.0, 10.0, 2)
    total = 1.0
    for value in z:
        series = sum(i * math.cos((i + 1) * float(value) + i) for i in range(1, 6))
        total *= series
    return float(total + 186.7309)


def easom(x: np.ndarray) -> float:
    z = scale(x, -100.0, 100.0, 2)
    x1, x2 = float(z[0]), float(z[1])
    raw = -math.cos(x1) * math.cos(x2) * math.exp(-((x1 - math.pi) ** 2 + (x2 - math.pi) ** 2))
    return float(raw + 1.0)


def bukin6(x: np.ndarray) -> float:
    z = unit(x, 2)
    x1 = -15.0 + 10.0 * float(z[0])
    x2 = -3.0 + 6.0 * float(z[1])
    return float(100.0 * math.sqrt(abs(x2 - 0.01 * x1**2)) + 0.01 * abs(x1 + 10.0))


def ridge_chain(x: np.ndarray) -> float:
    z = scale(x, -2.0, 2.0)
    if len(z) <= 1:
        return float(z[0] ** 2)
    return float(np.sum((z[:-1] + 2.0 * z[1:]) ** 2) + 0.1 * np.sum(z**2))


def brown(x: np.ndarray) -> float:
    z = scale(x, -1.0, 4.0)
    if len(z) <= 1:
        return float(z[0] ** 2)
    return float(np.sum((z[:-1] ** 2) ** (z[1:] ** 2 + 1.0) + (z[1:] ** 2) ** (z[:-1] ** 2 + 1.0)))


def exponential(x: np.ndarray) -> float:
    z = scale(x, -1.0, 1.0)
    return float(-math.exp(-0.5 * float(np.sum(z**2))) + 1.0)


def cosine_mixture(x: np.ndarray) -> float:
    z = scale(x, -1.0, 1.0)
    raw = -0.1 * np.sum(np.cos(5.0 * math.pi * z)) + np.sum(z**2)
    return float(raw + 0.1 * len(z))


def plateau_sphere(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    stepped = np.floor(np.abs(z) + 0.5)
    return float(np.sum(stepped**2))


def noisy_quartic_proxy(x: np.ndarray) -> float:
    z = scale(x, -1.28, 1.28)
    idx = np.arange(1, len(z) + 1, dtype=float)
    ripple = 0.02 * np.sum(np.sin(37.0 * z + idx) ** 2)
    return float(np.sum(idx * z**4) + ripple)


def register(key: str, name: str, suite: str, description: str, bounds: tuple[float, float], evaluate, default_dimension: int, min_dimension: int, max_dimension: int) -> None:
    register_single_benchmark(
        key,
        {
            "name": name,
            "suite": suite,
            "description": description,
            "bounds": [float(bounds[0]), float(bounds[1])],
            "defaultDimension": default_dimension,
            "minDimension": min_dimension,
            "maxDimension": max_dimension,
            "optimum": 0.0,
            "evaluate": evaluate,
        },
    )


BENCHMARKS = [
    ("classic_beale", "Beale", "Fixed-dimensional classic", "Two-dimensional polynomial valley benchmark.", (-4.5, 4.5), beale, 2, 2, 2),
    ("classic_booth", "Booth", "Fixed-dimensional classic", "Two-dimensional convex quadratic interaction benchmark.", (-10.0, 10.0), booth, 2, 2, 2),
    ("classic_matyas", "Matyas", "Fixed-dimensional classic", "Two-dimensional smooth bowl with cross coupling.", (-10.0, 10.0), matyas, 2, 2, 2),
    ("classic_mccormick", "McCormick", "Fixed-dimensional classic", "Two-dimensional sinusoidal quadratic benchmark.", (-1.5, 4.0), mccormick, 2, 2, 2),
    ("classic_three_hump_camel", "Three-hump Camel", "Fixed-dimensional classic", "Two-dimensional camel function with polynomial ridges.", (-5.0, 5.0), three_hump_camel, 2, 2, 2),
    ("classic_six_hump_camel", "Six-hump Camel", "Fixed-dimensional classic", "Two-dimensional multimodal camel benchmark shifted to zero optimum.", (-3.0, 3.0), six_hump_camel, 2, 2, 2),
    ("classic_branin", "Branin", "Fixed-dimensional classic", "Two-dimensional Branin function shifted to zero optimum.", (-5.0, 15.0), branin, 2, 2, 2),
    ("classic_goldstein_price", "Goldstein-Price", "Fixed-dimensional classic", "Two-dimensional Goldstein-Price benchmark shifted to zero optimum.", (-2.0, 2.0), goldstein_price, 2, 2, 2),
    ("classic_forrester", "Forrester", "One-dimensional expensive", "One-dimensional expensive optimization benchmark.", (0.0, 1.0), forrester, 1, 1, 1),
    ("classic_gramacy_lee", "Gramacy-Lee", "One-dimensional expensive", "One-dimensional nonlinear benchmark with sharp local variation.", (0.5, 2.5), gramacy_lee, 1, 1, 1),
    ("classic_bohachevsky", "Bohachevsky", "Fixed-dimensional classic", "Two-dimensional separable multimodal quadratic benchmark.", (-100.0, 100.0), bohachevsky, 2, 2, 2),
    ("classic_schubert", "Schubert", "Fixed-dimensional classic", "Two-dimensional product-of-series multimodal benchmark.", (-10.0, 10.0), schubert, 2, 2, 2),
    ("classic_easom", "Easom", "Fixed-dimensional classic", "Needle-like global optimum in a broad flat landscape.", (-100.0, 100.0), easom, 2, 2, 2),
    ("classic_bukin6", "Bukin N.6", "Fixed-dimensional classic", "Narrow curved valley benchmark.", (-15.0, 3.0), bukin6, 2, 2, 2),
    ("extended_ridge_chain", "Ridge Chain", "Scalable synthetic", "Scalable nonseparable quadratic chain benchmark.", (-2.0, 2.0), ridge_chain, 10, 2, 300),
    ("extended_brown", "Brown", "Scalable synthetic", "Scalable nonseparable exponential-power benchmark.", (-1.0, 4.0), brown, 10, 2, 100),
    ("extended_exponential", "Exponential Bowl", "Scalable synthetic", "Smooth bounded bowl with flat tails.", (-1.0, 1.0), exponential, 10, 1, 300),
    ("extended_cosine_mixture", "Cosine Mixture", "Scalable synthetic", "Multimodal cosine mixture with quadratic envelope.", (-1.0, 1.0), cosine_mixture, 10, 1, 200),
    ("extended_plateau_sphere", "Plateau Sphere", "Scalable synthetic", "Stepped sphere with large plateaus.", (-5.0, 5.0), plateau_sphere, 10, 1, 300),
    ("extended_noisy_quartic", "Deterministic Quartic Noise", "Scalable synthetic", "Quartic benchmark with deterministic ripple noise proxy.", (-1.28, 1.28), noisy_quartic_proxy, 10, 1, 200),
]


for item in BENCHMARKS:
    register(*item)

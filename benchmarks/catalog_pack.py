from __future__ import annotations

import math

import numpy as np

from .registry import input_defs, objective_defs, register_benchmark


def unit_vector(x: np.ndarray, length: int | None = None) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if length is not None and len(values) < length:
        values = np.pad(values, (0, length - len(values)), constant_values=0.5)
    return np.clip(values, 0.0, 1.0)


def resize_unit(x: np.ndarray, length: int) -> np.ndarray:
    return np.resize(unit_vector(x), length)


def score_from_raw(raw: float, scale: float = 1.0) -> float:
    return float(1.0 / (1.0 + max(0.0, float(raw)) / max(scale, 1e-9)))


def register_series(
    prefix: str,
    label: str,
    count: int,
    default_dimension: int,
    min_dimension: int,
    max_dimension: int,
    objectives: int,
    evaluator_factory,
    description_factory,
) -> None:
    for variant in range(1, count + 1):
        key = f"{prefix}{variant:02d}"
        register_benchmark(
            key,
            {
                "name": f"{label} {variant}",
                "description": description_factory(variant),
                "minDimension": min_dimension,
                "defaultDimension": default_dimension,
                "maxDimension": max_dimension,
                "inputLabels": input_defs,
                "objectives": objective_defs(objectives),
                "evaluate": evaluator_factory(variant),
            },
        )


def uf_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        z = resize_unit(x, max(2, len(np.asarray(x, dtype=float))))
        base = float(z[0])
        tail = z[1:] if len(z) > 1 else np.array([0.5], dtype=float)
        idx = np.arange(1, len(z), dtype=float)
        phase = 0.09 * variant
        wobble = 0.18 * np.sin(2.0 * math.pi * (idx / (len(z) + 1.0) + phase))
        g = float(np.mean((tail - 0.5 - wobble) ** 2) + 0.08 * np.mean(np.cos(6.0 * math.pi * (tail + phase)) ** 2))
        r1 = base ** (0.55 + 0.03 * (variant % 5)) + 1.2 * g + 0.12 * np.sin((variant + 1) * math.pi * base) ** 2
        r2 = (1.0 - base) ** (0.55 + 0.02 * ((variant + 1) % 5)) + 1.45 * g + 0.08 * np.cos((variant + 2) * math.pi * base) ** 2
        if variant % 3 == 0:
            r1 += 0.10 * np.sin(3.0 * math.pi * base + phase) ** 2
        if variant % 3 == 1:
            r2 += 0.07 * np.cos(5.0 * math.pi * base + phase) ** 2
        return np.array([score_from_raw(r1, 1.6), score_from_raw(r2, 1.6)], dtype=float)

    return evaluate


def cf_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        z = resize_unit(x, max(2, len(np.asarray(x, dtype=float))))
        base = float(z[0])
        tail = z[1:] if len(z) > 1 else np.array([0.5], dtype=float)
        phase = 0.07 * variant
        g = float(np.mean((tail - 0.52 - 0.15 * np.sin(2.0 * math.pi * (tail + phase))) ** 2))
        r1 = base ** 0.9 + 1.1 * g + 0.15 * np.sin((variant + 1) * math.pi * base) ** 2
        r2 = (1.0 - base) ** 0.8 + 1.35 * g + 0.12 * np.cos((variant + 2) * math.pi * base) ** 2
        c1 = base + 0.16 * np.sin((variant + 1) * math.pi * float(np.mean(tail))) - 0.56
        c2 = float(np.mean(tail)) - 0.46 + 0.09 * np.cos((variant + 2) * math.pi * base)
        penalty = 3.8 * (max(0.0, c1) ** 2 + max(0.0, c2) ** 2 + max(0.0, 0.18 - abs(base - 0.5)) ** 2)
        return np.array([score_from_raw(r1 + penalty, 1.7), score_from_raw(r2 + penalty, 1.7)], dtype=float)

    return evaluate


def wfg_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        z = resize_unit(x, max(3, len(np.asarray(x, dtype=float))))
        head = z[:2]
        tail = z[2:] if len(z) > 2 else np.array([0.5], dtype=float)
        phase = 0.05 * variant
        g = float(np.mean((tail - 0.5 - 0.20 * np.sin(2.0 * math.pi * (tail + phase))) ** 2) + 0.05 * np.mean(np.sin((variant + 2) * math.pi * tail) ** 2))
        theta1 = 0.5 * math.pi * np.clip(head[0] ** (0.72 + 0.01 * variant), 0.0, 1.0)
        theta2 = 0.5 * math.pi * np.clip(head[1] ** (0.78 + 0.008 * variant), 0.0, 1.0)
        if variant % 4 == 0:
            raw = np.array([
                (1.0 + g) * (1.0 - math.cos(theta1)) * (1.0 - math.cos(theta2)),
                (1.0 + g) * (1.0 - math.cos(theta1)) * (1.0 - math.sin(theta2)),
                (1.0 + g) * (1.0 - math.sin(theta1) * math.cos(theta2)),
            ])
        elif variant % 4 == 1:
            raw = np.array([
                (1.0 + g) * (1.0 - math.cos(theta1)) * (1.0 - math.sin(theta2)),
                (1.0 + g) * (1.0 - math.cos(theta1)) * (1.0 - math.cos(theta2)),
                (1.0 + g) * (1.0 - math.sin(theta1)),
            ])
        elif variant % 4 == 2:
            raw = np.array([
                (1.0 + g) * (0.8 - math.cos(theta1) * math.cos(theta2) + 0.15 * math.sin(4.0 * math.pi * head[0] + phase) ** 2),
                (1.0 + g) * (0.8 - math.cos(theta1) * math.sin(theta2) + 0.12 * math.cos(4.0 * math.pi * head[1] + phase) ** 2),
                (1.0 + g) * (0.8 - math.sin(theta1) + 0.12 * math.sin(3.0 * math.pi * head[0]) ** 2),
            ])
        else:
            raw = np.array([
                (1.0 + g) * (1.0 - math.sin(theta1) * math.cos(theta2)),
                (1.0 + g) * (1.0 - math.sin(theta1) * math.sin(theta2)),
                (1.0 + g) * (1.0 - math.cos(theta1)) + 0.1 * np.cos(6.0 * math.pi * head[1] + phase) ** 2,
            ])
        return np.array([score_from_raw(v, 1.3 + 0.03 * variant) for v in raw], dtype=float)

    return evaluate


def lz_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        z = resize_unit(x, max(3, len(np.asarray(x, dtype=float))))
        a, b, c = float(z[0]), float(z[1]), float(z[2])
        phase = 0.11 * variant
        ripple = 0.16 * np.sin((variant + 1) * math.pi * (a + b + c)) ** 2
        raw1 = (a - 0.10 * variant / 14.0) ** 2 + 0.48 * (b - 0.35) ** 2 + 0.18 * np.cos(6.0 * math.pi * c + phase) ** 2 + ripple
        raw2 = (1.0 - a) ** 2 + (b - (0.68 - 0.04 * math.sin(phase))) ** 2 + 0.14 * np.sin(4.0 * math.pi * c + phase) ** 2 + 0.08 * np.cos(3.0 * math.pi * a) ** 2
        raw3 = (c - 0.24 * a) ** 2 + 0.52 * (1.0 - b) ** 2 + 0.09 * np.cos(5.0 * math.pi * a + phase) ** 2 + 0.05 * np.sin(5.0 * math.pi * b) ** 2
        return np.array([score_from_raw(raw1, 1.0), score_from_raw(raw2, 1.0), score_from_raw(raw3, 1.0)], dtype=float)

    return evaluate


ENGINEERING_CASES = [
    ("battery", "Battery pack", 5, "Battery pack management trade-off among quality, throughput, and efficiency."),
    ("microgrid", "Microgrid dispatch", 6, "Microgrid dispatch trade-off among quality, responsiveness, and operating cost."),
    ("datacenter", "Datacenter cooling", 6, "Datacenter cooling design trade-off among quality, speed, and power draw."),
    ("drug", "Drug design", 7, "Drug design trade-off among efficacy, synthesis speed, and development cost."),
    ("robot", "Robot gait", 5, "Robot gait tuning trade-off among stability, speed, and energy cost."),
    ("supply", "Supply chain", 6, "Supply chain configuration trade-off among service level, delay, and cost."),
    ("chip", "Chip floorplan", 8, "Chip floorplanning trade-off among performance, frequency, and area efficiency."),
    ("warehouse", "Warehouse flow", 5, "Warehouse flow trade-off among quality, throughput, and labor cost."),
    ("satellite", "Satellite constellation", 6, "Satellite constellation trade-off among quality, latency, and budget."),
    ("hydrogen", "Hydrogen plant", 7, "Hydrogen plant trade-off among output quality, ramp speed, and cost."),
    ("crop", "Crop planning", 5, "Crop planning trade-off among yield quality, timing, and resource use."),
    ("fleet", "Fleet routing", 6, "Fleet routing trade-off among delivery quality, speed, and cost."),
]


def engineering_evaluator(variant: int):
    def evaluate(x: np.ndarray) -> np.ndarray:
        case = ENGINEERING_CASES[variant - 1]
        dimension = case[2]
        z = resize_unit(x, dimension)
        coeff = 0.11 + 0.015 * variant
        mix = 0.03 * np.sin((variant + 1) * 2.0 * math.pi * z)
        q = 0.48 + 0.24 * z[0] + 0.18 * z[min(1, dimension - 1)] - 0.12 * z[min(2, dimension - 1)] + 0.05 * np.sin(5.0 * math.pi * z[0] + coeff) + float(np.mean(mix))
        s = 0.42 + 0.20 * z[min(1, dimension - 1)] + 0.19 * z[min(3, dimension - 1)] - 0.15 * z[0] + 0.06 * np.cos(4.0 * math.pi * z[min(2, dimension - 1)] + coeff)
        c = 0.78 - 0.30 * z[0] + 0.14 * z[min(1, dimension - 1)] + 0.11 * z[min(2, dimension - 1)] - 0.08 * z[min(3, dimension - 1)] + 0.05 * np.sin(3.0 * math.pi * z[min(4, dimension - 1)] + coeff)
        penalty = 0.0
        if dimension > 4:
            penalty += 0.04 * float(np.sum(np.maximum(0.0, z[4:] - 0.78) ** 2))
            penalty += 0.03 * float(np.sum(np.maximum(0.0, 0.16 - z[4:]) ** 2))
        q = np.clip(q, 0.0, 1.0)
        s = np.clip(s, 0.0, 1.0)
        c = np.clip(c, 0.0, 1.0)
        return np.array(
            [
                score_from_raw((1.0 - q) ** 2 + penalty, 0.55),
                score_from_raw((1.0 - s) ** 2 + penalty, 0.55),
                score_from_raw((1.0 - c) ** 2 + penalty, 0.55),
            ],
            dtype=float,
        )

    return evaluate


register_series(
    "uf",
    "UF-style",
    14,
    30,
    2,
    80,
    2,
    uf_evaluator,
    lambda variant: f"UF-style two-objective benchmark variant {variant}, with multimodal distance and nonlinear trade-off structure.",
)

register_series(
    "cf",
    "CF-style",
    14,
    30,
    2,
    80,
    2,
    cf_evaluator,
    lambda variant: f"Constrained CF-style two-objective benchmark variant {variant}, with soft penalties and disconnected feasible regions.",
)

register_series(
    "wfg",
    "WFG-style",
    14,
    24,
    3,
    60,
    3,
    wfg_evaluator,
    lambda variant: f"WFG-style three-objective benchmark variant {variant}, with distortion, shape, and many-to-one trade-offs.",
)

register_series(
    "lz",
    "LZ-style",
    14,
    12,
    3,
    40,
    3,
    lz_evaluator,
    lambda variant: f"LZ-style three-objective benchmark variant {variant}, with rugged multimodal structure and nonconvex fronts.",
)

for variant, (suffix, name, dimension, description) in enumerate(ENGINEERING_CASES, start=1):
    register_benchmark(
        f"eng{variant:02d}",
        {
            "name": name,
            "description": description,
            "minDimension": dimension,
            "defaultDimension": dimension,
            "maxDimension": dimension,
            "inputLabels": input_defs,
            "objectives": objective_defs(3),
            "evaluate": engineering_evaluator(variant),
        },
    )

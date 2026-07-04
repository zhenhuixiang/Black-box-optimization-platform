from __future__ import annotations

import math

import numpy as np

from .registry import input_defs, minimize_to_score, objective_defs, register_benchmark


def scaled(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) * np.asarray(x, dtype=float)


def register_problem(key: str, name: str, description: str, dimension: int, objectives: int, evaluate) -> None:
    register_benchmark(
        key,
        {
            "name": name,
            "description": description,
            "minDimension": dimension,
            "defaultDimension": dimension,
            "maxDimension": dimension,
            "inputLabels": input_defs,
            "objectives": objective_defs(objectives),
            "evaluate": evaluate,
        },
    )


def schaffer_n1(x: np.ndarray) -> np.ndarray:
    z = float(scaled(x[:1], -10.0, 10.0)[0])
    return np.array([minimize_to_score(z**2, 0.0, 100.0), minimize_to_score((z - 2.0) ** 2, 0.0, 144.0)], dtype=float)


def schaffer_n2(x: np.ndarray) -> np.ndarray:
    z = float(scaled(x[:1], -5.0, 10.0)[0])
    if z <= 1.0:
        f1 = -z
    elif z <= 3.0:
        f1 = z - 2.0
    elif z <= 4.0:
        f1 = 4.0 - z
    else:
        f1 = z - 4.0
    f2 = (z - 5.0) ** 2
    return np.array([minimize_to_score(f1, -10.0, 6.0), minimize_to_score(f2, 0.0, 225.0)], dtype=float)


def fonseca_fleming(x: np.ndarray) -> np.ndarray:
    z = scaled(x, -4.0, 4.0)
    n = len(z)
    shift = 1.0 / math.sqrt(n)
    f1 = 1.0 - math.exp(-float(np.sum((z - shift) ** 2)))
    f2 = 1.0 - math.exp(-float(np.sum((z + shift) ** 2)))
    return np.array([minimize_to_score(f1, 0.0, 1.0), minimize_to_score(f2, 0.0, 1.0)], dtype=float)


def kursawe(x: np.ndarray) -> np.ndarray:
    z = scaled(x, -5.0, 5.0)
    f1 = sum(-10.0 * math.exp(-0.2 * math.sqrt(z[i] ** 2 + z[i + 1] ** 2)) for i in range(len(z) - 1))
    f2 = float(np.sum(np.abs(z) ** 0.8 + 5.0 * np.sin(z**3)))
    return np.array([minimize_to_score(f1, -25.0, 0.0), minimize_to_score(f2, -25.0, 35.0)], dtype=float)


def poloni(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:2], -math.pi, math.pi)
    x1, x2 = float(z[0]), float(z[1])
    a1 = 0.5 * math.sin(1) - 2 * math.cos(1) + math.sin(2) - 1.5 * math.cos(2)
    a2 = 1.5 * math.sin(1) - math.cos(1) + 2 * math.sin(2) - 0.5 * math.cos(2)
    b1 = 0.5 * math.sin(x1) - 2 * math.cos(x1) + math.sin(x2) - 1.5 * math.cos(x2)
    b2 = 1.5 * math.sin(x1) - math.cos(x1) + 2 * math.sin(x2) - 0.5 * math.cos(x2)
    f1 = 1 + (a1 - b1) ** 2 + (a2 - b2) ** 2
    f2 = (x1 + 3) ** 2 + (x2 + 1) ** 2
    return np.array([minimize_to_score(f1, 1.0, 80.0), minimize_to_score(f2, 0.0, 80.0)], dtype=float)


def binh_korn(x: np.ndarray) -> np.ndarray:
    z = np.asarray([5.0 * x[0], 3.0 * x[1]], dtype=float)
    f1 = 4.0 * z[0] ** 2 + 4.0 * z[1] ** 2
    f2 = (z[0] - 5.0) ** 2 + (z[1] - 5.0) ** 2
    penalty = 0.0
    if (z[0] - 5) ** 2 + z[1] ** 2 > 25:
        penalty += 20.0
    if (z[0] - 8) ** 2 + (z[1] + 3) ** 2 < 7.7:
        penalty += 20.0
    return np.array([minimize_to_score(f1 + penalty, 0.0, 160.0), minimize_to_score(f2 + penalty, 0.0, 80.0)], dtype=float)


def tanaka(x: np.ndarray) -> np.ndarray:
    z = 2.0 * math.pi * np.asarray(x[:2], dtype=float)
    f1, f2 = float(z[0]), float(z[1])
    penalty = 0.0
    if f1**2 + f2**2 < 1.0 + 0.1 * math.cos(16.0 * math.atan2(f1, f2)):
        penalty += 4.0
    if (f1 - 0.5) ** 2 + (f2 - 0.5) ** 2 > 0.5:
        penalty += 4.0
    return np.array([minimize_to_score(f1 + penalty, 0.0, 11.0), minimize_to_score(f2 + penalty, 0.0, 11.0)], dtype=float)


def viennet2(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:2], -4.0, 4.0)
    x1, x2 = float(z[0]), float(z[1])
    f1 = ((x1 - 2) ** 2) / 2.0 + ((x2 + 1) ** 2) / 13.0 + 3.0
    f2 = ((x1 + x2 - 3) ** 2) / 36.0 + ((-x1 + x2 + 2) ** 2) / 8.0 - 17.0
    f3 = ((x1 + 2 * x2 - 1) ** 2) / 175.0 + ((-x1 + 2 * x2) ** 2) / 17.0 - 13.0
    return np.array([minimize_to_score(f1, 0.0, 12.0), minimize_to_score(f2, -17.0, 5.0), minimize_to_score(f3, -13.0, 5.0)], dtype=float)


def viennet3(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:2], -3.0, 3.0)
    x1, x2 = float(z[0]), float(z[1])
    f1 = 0.5 * (x1**2 + x2**2) + math.sin(x1**2 + x2**2)
    f2 = ((3 * x1 - 2 * x2 + 4) ** 2) / 8.0 + ((x1 - x2 + 1) ** 2) / 27.0 + 15.0
    f3 = 1.0 / (x1**2 + x2**2 + 1.0) - 1.1 * math.exp(-(x1**2 + x2**2))
    return np.array([minimize_to_score(f1, -1.0, 12.0), minimize_to_score(f2, 0.0, 80.0), minimize_to_score(f3, -1.0, 1.0)], dtype=float)


def viennet4(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:2], -4.0, 4.0)
    x1, x2 = float(z[0]), float(z[1])
    f1 = ((x1 - 2) ** 2) / 2.0 + ((x2 + 1) ** 2) / 13.0 + 3.0
    f2 = ((x1 + x2 - 3) ** 2) / 175.0 + ((2 * x2 - x1) ** 2) / 17.0 - 13.0
    f3 = ((3 * x1 - 2 * x2 + 4) ** 2) / 8.0 + ((x1 - x2 + 1) ** 2) / 27.0 + 15.0
    return np.array([minimize_to_score(f1, 0.0, 12.0), minimize_to_score(f2, -13.0, 5.0), minimize_to_score(f3, 0.0, 90.0)], dtype=float)


def viennet5(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:2], -2.0, 2.0)
    x1, x2 = float(z[0]), float(z[1])
    f1 = 0.5 * (x1**2 + x2**2) + math.sin(x1**2 + x2**2)
    f2 = (3 * x1 - 2 * x2 + 4) ** 2 / 8.0 + (x1 - x2 + 1) ** 2 / 27.0 + 15.0
    f3 = 1.0 / (x1**2 + x2**2 + 1.0) - 1.1 * math.exp(-(x1**2 + x2**2))
    return np.array([minimize_to_score(f1, -1.0, 8.0), minimize_to_score(f2, 0.0, 70.0), minimize_to_score(f3, -1.0, 1.0)], dtype=float)


def constr_ex(x: np.ndarray) -> np.ndarray:
    z = np.asarray([0.1 + 0.9 * x[0], 5.0 * x[1]], dtype=float)
    f1 = z[0]
    f2 = (1.0 + z[1]) / z[0]
    penalty = 0.0
    if z[1] + 9.0 * z[0] < 6.0:
        penalty += 5.0
    if -z[1] + 9.0 * z[0] < 1.0:
        penalty += 5.0
    return np.array([minimize_to_score(f1 + penalty, 0.0, 6.0), minimize_to_score(f2 + penalty, 0.0, 60.0)], dtype=float)


def ctp1(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x[:2], dtype=float)
    f1 = z[0]
    g = 1.0 + z[1]
    f2 = g * math.exp(-f1 / g)
    penalty = 0.0
    if f2 / (0.858 * math.exp(-0.541 * f1)) < 1.0:
        penalty += 2.0
    if f2 / (0.728 * math.exp(-0.295 * f1)) < 1.0:
        penalty += 2.0
    return np.array([minimize_to_score(f1 + penalty, 0.0, 5.0), minimize_to_score(f2 + penalty, 0.0, 5.0)], dtype=float)


def water_like(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:3], 0.01, 0.45)
    x1, x2, x3 = [float(v) for v in z]
    f1 = 106780.37 * (x2 + x3) + 61704.67
    f2 = 3000.0 * x1
    f3 = 305700.0 * 2289.0 * x2 / ((0.06 * 2289.0) ** 0.65)
    return np.array([minimize_to_score(f1, 60000.0, 180000.0), minimize_to_score(f2, 0.0, 1400.0), minimize_to_score(f3, 0.0, 3000000.0)], dtype=float)


def welded_beam_like(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:4], 0.1, 2.0)
    h, l, t, b = [float(v) for v in z]
    cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
    deflection = 2.1952 / max(t**3 * b, 1e-9)
    stress = 1.0 / max(h * l, 1e-9) + 1.0 / max(t * b, 1e-9)
    return np.array([minimize_to_score(cost, 0.0, 80.0), minimize_to_score(deflection, 0.0, 30.0), minimize_to_score(stress, 0.0, 40.0)], dtype=float)


def car_side_impact_like(x: np.ndarray) -> np.ndarray:
    z = scaled(x[:7], 0.5, 1.5)
    weight = float(np.sum([1.98, 4.9, 6.67, 6.98, 4.01, 1.78, 2.73] * z))
    safety = float(np.sum((z - 1.0) ** 2) + 0.5 * np.sin(5.0 * z[0]) ** 2)
    intrusion = float(0.8 * z[1] + 0.6 * z[3] + 0.4 * z[5] + 0.2 * z[6])
    return np.array([minimize_to_score(weight, 10.0, 70.0), minimize_to_score(safety, 0.0, 5.0), minimize_to_score(intrusion, 0.0, 4.0)], dtype=float)


register_problem("schaffer_n1", "Schaffer N.1", "Classic one-variable two-objective Schaffer test problem.", 1, 2, schaffer_n1)
register_problem("schaffer_n2", "Schaffer N.2", "Piecewise Schaffer two-objective test problem.", 1, 2, schaffer_n2)
register_problem("fonseca_fleming", "Fonseca-Fleming", "Smooth two-objective test problem with a broad Pareto set.", 3, 2, fonseca_fleming)
register_problem("kursawe", "Kursawe", "Nonconvex multimodal two-objective benchmark.", 3, 2, kursawe)
register_problem("poloni", "Poloni", "Two-variable nonconvex two-objective benchmark.", 2, 2, poloni)
register_problem("binh_korn", "Binh-Korn", "Constrained two-objective engineering benchmark, represented with penalty scores.", 2, 2, binh_korn)
register_problem("tanaka", "Tanaka", "Constrained two-objective benchmark with disconnected feasible regions, represented with penalty scores.", 2, 2, tanaka)
register_problem("viennet2", "Viennet 2", "Three-objective two-variable Viennet benchmark.", 2, 3, viennet2)
register_problem("viennet3", "Viennet 3", "Three-objective nonconvex Viennet benchmark.", 2, 3, viennet3)
register_problem("viennet4", "Viennet 4", "Three-objective Viennet benchmark with differently scaled objectives.", 2, 3, viennet4)
register_problem("viennet5", "Viennet 5", "Three-objective compact-domain Viennet benchmark.", 2, 3, viennet5)
register_problem("constr_ex", "CONSTR-EX", "Constrained two-objective benchmark transformed with soft penalties.", 2, 2, constr_ex)
register_problem("ctp1", "CTP1", "Constrained CTP-style two-objective benchmark transformed with penalties.", 2, 2, ctp1)
register_problem("water_like", "Water Resource-like", "Three-objective water resource planning style benchmark.", 3, 3, water_like)
register_problem("welded_beam_like", "Welded Beam-like", "Three-objective welded beam design style benchmark.", 4, 3, welded_beam_like)
register_problem("car_side_impact_like", "Car Side Impact-like", "Seven-variable three-objective side-impact style benchmark.", 7, 3, car_side_impact_like)

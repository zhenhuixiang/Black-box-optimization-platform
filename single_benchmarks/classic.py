from __future__ import annotations

import math

import numpy as np

from .registry import register_single_benchmark


def unit(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (hi - lo) * unit(x)


def pseudo_rotate(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float).copy()
    if len(z) <= 1:
        return z
    out = z.copy()
    for i in range(1, len(z)):
        out[i] = 0.88 * z[i] + 0.28 * z[i - 1]
    out[0] = 0.88 * z[0] + 0.28 * z[-1]
    return out


def ellipsoid_core(z: np.ndarray) -> float:
    if len(z) <= 1:
        return float(z[0] ** 2)
    weights = 10.0 ** (6.0 * np.arange(len(z), dtype=float) / (len(z) - 1))
    return float(np.sum(weights * z**2))


def rastrigin_core(z: np.ndarray) -> float:
    return float(10.0 * len(z) + np.sum(z**2 - 10.0 * np.cos(2.0 * math.pi * z)))


def rosenbrock_core(z: np.ndarray) -> float:
    if len(z) <= 1:
        return float((z[0] - 1.0) ** 2)
    return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))


def ackley_core(z: np.ndarray) -> float:
    d = max(1, len(z))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / d))
        - np.exp(np.sum(np.cos(2.0 * math.pi * z)) / d)
        + 20.0
        + math.e
    )


def griewank_core(z: np.ndarray) -> float:
    idx = np.arange(1, len(z) + 1, dtype=float)
    return float(np.sum(z**2) / 4000.0 - np.prod(np.cos(z / np.sqrt(idx))) + 1.0)


def schwefel_core(z: np.ndarray) -> float:
    return float(max(0.0, 418.9829 * len(z) - np.sum(z * np.sin(np.sqrt(np.abs(z))))))


def levy_core(z: np.ndarray) -> float:
    w = 1.0 + (z - 1.0) / 4.0
    first = np.sin(math.pi * w[0]) ** 2
    middle = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * w[:-1] + 1.0) ** 2))
    last = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * math.pi * w[-1]) ** 2)
    return float(first + middle + last)


def zakharov_core(z: np.ndarray) -> float:
    idx = np.arange(1, len(z) + 1, dtype=float)
    linear = 0.5 * np.sum(idx * z)
    return float(np.sum(z**2) + linear**2 + linear**4)


def sphere(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    return float(np.sum(z**2))


def sum_squares(x: np.ndarray) -> float:
    z = scale(x, -10.0, 10.0)
    idx = np.arange(1, len(z) + 1, dtype=float)
    return float(np.sum(idx * z**2))


def ellipsoid(x: np.ndarray) -> float:
    return ellipsoid_core(scale(x, -5.0, 5.0))


def discus(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    return float(1e6 * z[0] ** 2 + np.sum(z[1:] ** 2))


def bent_cigar(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))


def sharp_ridge(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    return float(z[0] ** 2 + 100.0 * np.sqrt(np.sum(z[1:] ** 2)))


def different_powers(x: np.ndarray) -> float:
    z = np.abs(scale(x, -1.0, 1.0))
    if len(z) <= 1:
        return float(z[0] ** 2)
    powers = 2.0 + 4.0 * np.arange(len(z), dtype=float) / (len(z) - 1)
    return float(np.sqrt(np.sum(z**powers)))


def rastrigin(x: np.ndarray) -> float:
    return rastrigin_core(scale(x, -5.12, 5.12))


def rotated_rastrigin(x: np.ndarray) -> float:
    return rastrigin_core(pseudo_rotate(scale(x, -5.12, 5.12)))


def buche_rastrigin(x: np.ndarray) -> float:
    z = scale(x, -5.12, 5.12)
    mask = (z > 0.0) & ((np.arange(len(z)) % 2) == 0)
    z = z.copy()
    z[mask] *= 10.0
    return rastrigin_core(z)


def linear_slope(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    weights = np.linspace(1.0, 10.0, len(z))
    return float(np.sum(weights * (z + 5.0)))


def attractive_sector(x: np.ndarray) -> float:
    z = pseudo_rotate(scale(x, -5.0, 5.0))
    weights = np.where(z > 0.0, 100.0, 1.0)
    return float(np.sum(weights * z**2))


def step_ellipsoid(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    stepped = np.sign(z) * np.floor(np.abs(z) + 0.5)
    return 0.1 * abs(z[0]) + ellipsoid_core(stepped)


def rosenbrock(x: np.ndarray) -> float:
    return rosenbrock_core(scale(x, -2.0, 2.0))


def rotated_rosenbrock(x: np.ndarray) -> float:
    return rosenbrock_core(pseudo_rotate(scale(x, -2.0, 2.0)))


def ackley(x: np.ndarray) -> float:
    return ackley_core(scale(x, -32.768, 32.768))


def griewank(x: np.ndarray) -> float:
    return griewank_core(scale(x, -600.0, 600.0))


def schwefel(x: np.ndarray) -> float:
    return schwefel_core(scale(x, -500.0, 500.0))


def levy(x: np.ndarray) -> float:
    return levy_core(scale(x, -10.0, 10.0))


def zakharov(x: np.ndarray) -> float:
    return zakharov_core(scale(x, -5.0, 10.0))


def salomon(x: np.ndarray) -> float:
    z = scale(x, -100.0, 100.0)
    radius = float(np.sqrt(np.sum(z**2)))
    return float(1.0 - np.cos(2.0 * math.pi * radius) + 0.1 * radius)


def alpine1(x: np.ndarray) -> float:
    z = scale(x, -10.0, 10.0)
    return float(np.sum(np.abs(z * np.sin(z) + 0.1 * z)))


def qing(x: np.ndarray) -> float:
    z = scale(x, -500.0, 500.0)
    idx = np.arange(1, len(z) + 1, dtype=float)
    return float(np.sum((z**2 - idx) ** 2))


def dixon_price(x: np.ndarray) -> float:
    z = scale(x, -10.0, 10.0)
    if len(z) <= 1:
        return float((z[0] - 1.0) ** 2)
    idx = np.arange(2, len(z) + 1, dtype=float)
    return float((z[0] - 1.0) ** 2 + np.sum(idx * (2.0 * z[1:] ** 2 - z[:-1]) ** 2))


def styblinski_tang(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    raw = 0.5 * np.sum(z**4 - 16.0 * z**2 + 5.0 * z)
    return float(raw + 39.16616570377142 * len(z))


def michalewicz_shifted(x: np.ndarray) -> float:
    z = math.pi * unit(x)
    idx = np.arange(1, len(z) + 1, dtype=float)
    raw = -np.sum(np.sin(z) * np.sin(idx * z**2 / math.pi) ** 20)
    return float(raw + len(z))


def perm_function(x: np.ndarray) -> float:
    z = scale(x, -1.0, 1.0)
    beta = 0.5
    total = 0.0
    for i in range(1, len(z) + 1):
        inner = 0.0
        for j, value in enumerate(z, start=1):
            inner += (j**i + beta) * ((value / j) ** i - 1.0)
        total += inner**2
    return float(total)


def powell(x: np.ndarray) -> float:
    z = scale(x, -4.0, 5.0)
    padded = np.resize(z, int(math.ceil(len(z) / 4.0)) * 4)
    groups = padded.reshape(-1, 4)
    values = (
        (groups[:, 0] + 10.0 * groups[:, 1]) ** 2
        + 5.0 * (groups[:, 2] - groups[:, 3]) ** 2
        + (groups[:, 1] - 2.0 * groups[:, 2]) ** 4
        + 10.0 * (groups[:, 0] - groups[:, 3]) ** 4
    )
    return float(np.sum(values))


def trid_shifted(x: np.ndarray) -> float:
    z = scale(x, -len(x) ** 2, len(x) ** 2)
    raw = np.sum((z - 1.0) ** 2) - np.sum(z[1:] * z[:-1])
    optimum = -len(z) * (len(z) + 4) * (len(z) - 1) / 6.0
    return float(raw - optimum)


def sum_diff_powers(x: np.ndarray) -> float:
    z = np.abs(scale(x, -1.0, 1.0))
    powers = np.arange(2, len(z) + 2, dtype=float)
    return float(np.sum(z**powers))


def xin_she_yang(x: np.ndarray) -> float:
    z = scale(x, -2.0 * math.pi, 2.0 * math.pi)
    idx = np.arange(1, len(z) + 1, dtype=float)
    return float(np.sum(np.abs(z)) * np.exp(-np.sum(np.sin(z**2))) + 0.01 * np.sum(idx * z**4))


def periodic(x: np.ndarray) -> float:
    z = scale(x, -10.0, 10.0)
    return float(1.0 + np.sum(np.sin(z) ** 2) - 0.1 * np.exp(-np.sum(z**2)))


def happycat(x: np.ndarray) -> float:
    z = scale(x, -2.0, 2.0)
    alpha = 0.125
    norm2 = float(np.sum(z**2))
    return float(abs(norm2 - len(z)) ** (2.0 * alpha) + (0.5 * norm2 + np.sum(z)) / len(z) + 0.5)


def hgbat(x: np.ndarray) -> float:
    z = scale(x, -2.0, 2.0)
    norm2 = float(np.sum(z**2))
    sum_z = float(np.sum(z))
    return float(abs(norm2**2 - sum_z**2) ** 0.5 + (0.5 * norm2 + sum_z) / len(z) + 0.5)


def expanded_scaffer(x: np.ndarray) -> float:
    z = scale(x, -100.0, 100.0)
    pairs_a = z
    pairs_b = np.roll(z, -1)
    sq = pairs_a**2 + pairs_b**2
    return float(np.sum(0.5 + (np.sin(np.sqrt(sq)) ** 2 - 0.5) / (1.0 + 0.001 * sq) ** 2))


def schaffer_f7(x: np.ndarray) -> float:
    z = scale(x, -100.0, 100.0)
    if len(z) <= 1:
        return float(z[0] ** 2)
    s = np.sqrt(z[:-1] ** 2 + z[1:] ** 2)
    values = np.sqrt(s) + np.sqrt(s) * np.sin(50.0 * s**0.2) ** 2
    return float((np.sum(values) / max(1, len(values))) ** 2)


def schaffer_f7_ill(x: np.ndarray) -> float:
    z = scale(x, -100.0, 100.0)
    weights = 10.0 ** (np.linspace(0.0, 1.0, len(z)))
    return schaffer_f7((z * weights + 100.0) / 200.0)


def griewank_rosenbrock(x: np.ndarray) -> float:
    z = scale(x, -3.0, 3.0)
    if len(z) <= 1:
        return float((z[0] - 1.0) ** 2 / 4000.0 - math.cos(z[0] - 1.0) + 1.0)
    a = 100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2
    return float(np.sum(a / 4000.0 - np.cos(a) + 1.0))


def weierstrass(x: np.ndarray) -> float:
    z = scale(x, -0.5, 0.5)
    a = 0.5
    b = 3.0
    kmax = 20
    ks = np.arange(0, kmax + 1, dtype=float)
    term = 0.0
    for value in z:
        term += float(np.sum((a**ks) * np.cos(2.0 * math.pi * (b**ks) * (value + 0.5))))
    constant = len(z) * float(np.sum((a**ks) * np.cos(2.0 * math.pi * (b**ks) * 0.5)))
    return float(term - constant)


def katsuura(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    product = 1.0
    for i, value in enumerate(z, start=1):
        js = np.arange(1, 33, dtype=float)
        frac = np.abs(2.0**js * value - np.round(2.0**js * value)) / (2.0**js)
        product *= (1.0 + i * float(np.sum(frac))) ** (10.0 / (len(z) ** 1.2))
    return float((10.0 / len(z) ** 2) * product - (10.0 / len(z) ** 2))


def lunacek_bi_rastrigin(x: np.ndarray) -> float:
    z = scale(x, -5.0, 5.0)
    mu1 = 2.5
    d = 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(len(z) + 20.0) - 8.2)
    mu2 = -math.sqrt((mu1**2 - d) / s)
    term1 = np.sum((z - mu1) ** 2)
    term2 = d * len(z) + s * np.sum((z - mu2) ** 2)
    return float(min(term1, term2) + 10.0 * (len(z) - np.sum(np.cos(2.0 * math.pi * (z - mu1)))))


def gallagher(seed: int, centers: int):
    def evaluate(x: np.ndarray) -> float:
        z = scale(x, -5.0, 5.0)
        rng = np.random.default_rng(seed + len(z))
        c = rng.uniform(-4.5, 4.5, size=(centers, len(z)))
        c[0] = 0.0
        weights = np.linspace(1.0, 10.0, centers)
        dist = np.sum((z[None, :] - c) ** 2, axis=1)
        return float(np.min(weights * dist))

    return evaluate


def drop_wave(x: np.ndarray) -> float:
    z = scale(np.resize(unit(x), 2), -5.12, 5.12)
    r2 = float(np.sum(z**2))
    raw = -(1.0 + math.cos(12.0 * math.sqrt(r2))) / (0.5 * r2 + 2.0)
    return float(raw + 1.0)


def cross_in_tray(x: np.ndarray) -> float:
    z = scale(np.resize(unit(x), 2), -10.0, 10.0)
    x1, x2 = float(z[0]), float(z[1])
    raw = -0.0001 * (abs(math.sin(x1) * math.sin(x2) * math.exp(abs(100.0 - math.sqrt(x1**2 + x2**2) / math.pi))) + 1.0) ** 0.1
    return float(raw + 2.06261218)


def holder_table(x: np.ndarray) -> float:
    z = scale(np.resize(unit(x), 2), -10.0, 10.0)
    x1, x2 = float(z[0]), float(z[1])
    raw = -abs(math.sin(x1) * math.cos(x2) * math.exp(abs(1.0 - math.sqrt(x1**2 + x2**2) / math.pi)))
    return float(raw + 19.20850257)


def eggholder(x: np.ndarray) -> float:
    z = scale(np.resize(unit(x), 2), -512.0, 512.0)
    x1, x2 = float(z[0]), float(z[1])
    raw = -(x2 + 47.0) * math.sin(math.sqrt(abs(x2 + x1 / 2.0 + 47.0))) - x1 * math.sin(math.sqrt(abs(x1 - (x2 + 47.0))))
    return float(raw + 959.6406627)


def register(
    key: str,
    name: str,
    suite: str,
    description: str,
    bounds: tuple[float, float],
    evaluate,
    default_dimension: int = 10,
    min_dimension: int = 1,
    max_dimension: int = 200,
) -> None:
    register_single_benchmark(
        key,
        {
            "name": name,
            "suite": suite,
            "description": description,
            "defaultDimension": default_dimension,
            "minDimension": min_dimension,
            "maxDimension": max_dimension,
            "bounds": [float(bounds[0]), float(bounds[1])],
            "evaluate": evaluate,
            "optimum": 0.0,
        },
    )


BENCHMARKS = [
    ("bbob_sphere", "BBOB Sphere", "BBOB-inspired", "Convex separable quadratic baseline.", (-5.0, 5.0), sphere, 10, 1, 300),
    ("bbob_sum_squares", "BBOB Sum Squares", "BBOB-inspired", "Weighted separable quadratic with increasing coordinate stiffness.", (-10.0, 10.0), sum_squares, 10, 1, 300),
    ("bbob_ellipsoid_full", "BBOB Ellipsoid Full", "BBOB-inspired", "Severely ill-conditioned separable quadratic.", (-5.0, 5.0), ellipsoid, 10, 2, 300),
    ("bbob_discus_full", "BBOB Discus Full", "BBOB-inspired", "One dominant stiff axis and many soft axes.", (-5.0, 5.0), discus, 10, 2, 300),
    ("bbob_bent_cigar_full", "BBOB Bent Cigar Full", "BBOB-inspired", "Narrow cigar-shaped valley with high conditioning.", (-5.0, 5.0), bent_cigar, 10, 2, 300),
    ("bbob_sharp_ridge", "BBOB Sharp Ridge", "BBOB-inspired", "Sharp ridge structure with one quadratic axis.", (-5.0, 5.0), sharp_ridge, 10, 2, 300),
    ("bbob_different_powers", "BBOB Different Powers", "BBOB-inspired", "Smooth unimodal problem with dimension-dependent exponents.", (-1.0, 1.0), different_powers, 10, 1, 300),
    ("bbob_rastrigin_full", "BBOB Rastrigin Full", "BBOB-inspired", "Highly multimodal separable landscape.", (-5.12, 5.12), rastrigin, 10, 1, 200),
    ("bbob_rotated_rastrigin", "BBOB Rotated Rastrigin", "BBOB-inspired", "Rastrigin with deterministic coordinate mixing.", (-5.12, 5.12), rotated_rastrigin, 10, 2, 200),
    ("bbob_buche_rastrigin", "BBOB Buche-Rastrigin", "BBOB-inspired", "Asymmetric Rastrigin-style multimodal function.", (-5.12, 5.12), buche_rastrigin, 10, 2, 200),
    ("bbob_linear_slope", "BBOB Linear Slope", "BBOB-inspired", "Linear slope problem with optimum at the domain boundary.", (-5.0, 5.0), linear_slope, 10, 1, 300),
    ("bbob_attractive_sector", "BBOB Attractive Sector", "BBOB-inspired", "Sector-weighted quadratic that penalizes one sign region.", (-5.0, 5.0), attractive_sector, 10, 2, 300),
    ("bbob_step_ellipsoid", "BBOB Step Ellipsoid", "BBOB-inspired", "Stepped ill-conditioned ellipsoid with plateaus.", (-5.0, 5.0), step_ellipsoid, 10, 2, 300),
    ("bbob_rosenbrock_full", "BBOB Rosenbrock Full", "BBOB-inspired", "Curved valley benchmark with strong variable interaction.", (-2.0, 2.0), rosenbrock, 10, 2, 200),
    ("bbob_rotated_rosenbrock", "BBOB Rotated Rosenbrock", "BBOB-inspired", "Rosenbrock with deterministic coordinate mixing.", (-2.0, 2.0), rotated_rosenbrock, 10, 2, 200),
    ("bbob_schaffer_f7", "BBOB Schaffer F7", "BBOB-inspired", "Rugged pairwise-distance Schaffer F7 landscape.", (-100.0, 100.0), schaffer_f7, 10, 2, 200),
    ("bbob_schaffer_f7_ill", "BBOB Schaffer F7 Ill-conditioned", "BBOB-inspired", "Schaffer F7 with coordinate scaling.", (-100.0, 100.0), schaffer_f7_ill, 10, 2, 200),
    ("bbob_griewank_rosenbrock", "BBOB Griewank-Rosenbrock", "BBOB-inspired", "Expanded Griewank-Rosenbrock composition.", (-3.0, 3.0), griewank_rosenbrock, 10, 2, 200),
    ("bbob_schwefel_full", "BBOB Schwefel Full", "BBOB-inspired", "Deceptive multimodal function with distant optimum.", (-500.0, 500.0), schwefel, 10, 2, 200),
    ("bbob_gallagher21", "BBOB Gallagher 21 Peaks", "BBOB-inspired", "Deterministic multi-peak Gaussian landscape proxy.", (-5.0, 5.0), gallagher(21, 21), 10, 2, 120),
    ("bbob_gallagher101", "BBOB Gallagher 101 Peaks", "BBOB-inspired", "Dense deterministic multi-peak Gaussian landscape proxy.", (-5.0, 5.0), gallagher(101, 101), 10, 2, 120),
    ("bbob_katsuura", "BBOB Katsuura", "BBOB-inspired", "Fractal-style rugged landscape.", (-5.0, 5.0), katsuura, 10, 1, 100),
    ("bbob_lunacek", "BBOB Lunacek bi-Rastrigin", "BBOB-inspired", "Double-funnel multimodal benchmark.", (-5.0, 5.0), lunacek_bi_rastrigin, 10, 2, 100),
    ("bbob_weierstrass", "BBOB Weierstrass", "BBOB-inspired", "Continuous but highly rugged Weierstrass function.", (-0.5, 0.5), weierstrass, 10, 1, 100),
    ("cec_ackley", "CEC Ackley", "CEC classic", "Broad basin with many shallow local minima.", (-32.768, 32.768), ackley, 10, 1, 200),
    ("cec_griewank", "CEC Griewank", "CEC classic", "Product-interaction multimodal benchmark.", (-600.0, 600.0), griewank, 10, 1, 200),
    ("cec_levy", "CEC Levy", "CEC classic", "Multimodal sinusoidal valley benchmark.", (-10.0, 10.0), levy, 10, 2, 200),
    ("cec_zakharov", "CEC Zakharov", "CEC classic", "Unimodal nonseparable polynomial benchmark.", (-5.0, 10.0), zakharov, 10, 2, 200),
    ("cec_salomon", "CEC Salomon", "CEC classic", "Radial multimodal benchmark.", (-100.0, 100.0), salomon, 10, 1, 200),
    ("cec_alpine1", "CEC Alpine 1", "CEC classic", "Absolute sinusoidal landscape with many local minima.", (-10.0, 10.0), alpine1, 10, 1, 200),
    ("cec_qing", "CEC Qing", "CEC classic", "Quartic shifted target benchmark.", (-500.0, 500.0), qing, 10, 1, 100),
    ("cec_dixon_price", "CEC Dixon-Price", "CEC classic", "Nonseparable chain benchmark.", (-10.0, 10.0), dixon_price, 10, 2, 100),
    ("cec_styblinski_tang", "CEC Styblinski-Tang", "CEC classic", "Shifted quartic multimodal benchmark.", (-5.0, 5.0), styblinski_tang, 10, 1, 100),
    ("cec_michalewicz", "CEC Michalewicz Shifted", "CEC classic", "Steep valleys with dimension-indexed sinusoidal terms.", (0.0, math.pi), michalewicz_shifted, 10, 2, 100),
    ("cec_perm", "CEC Perm", "CEC classic", "Nonseparable perm function with beta=0.5.", (-1.0, 1.0), perm_function, 6, 2, 20),
    ("cec_powell", "CEC Powell", "CEC classic", "Quartic block-structured Powell function.", (-4.0, 5.0), powell, 12, 4, 200),
    ("cec_trid", "CEC Trid Shifted", "CEC classic", "Shifted Trid chain function with zero optimum value.", (-100.0, 100.0), trid_shifted, 10, 2, 60),
    ("cec_sum_diff_powers", "CEC Sum Different Powers", "CEC classic", "Separable function with coordinate-dependent powers.", (-1.0, 1.0), sum_diff_powers, 10, 1, 200),
    ("cec_xin_she_yang", "CEC Xin-She Yang", "CEC classic", "Exponential-sinusoidal rugged function.", (-6.283, 6.283), xin_she_yang, 10, 1, 100),
    ("cec_periodic", "CEC Periodic", "CEC classic", "Periodic multimodal function with a Gaussian depression.", (-10.0, 10.0), periodic, 10, 1, 200),
    ("cec_happycat", "CEC HappyCat", "CEC/GECCO classic", "Nonseparable hybrid benchmark used in real-parameter competitions.", (-2.0, 2.0), happycat, 10, 2, 200),
    ("cec_hgbat", "CEC HGBat", "CEC/GECCO classic", "Hybrid nonseparable benchmark with norm and sum coupling.", (-2.0, 2.0), hgbat, 10, 2, 200),
    ("cec_expanded_scaffer", "CEC Expanded Scaffer F6", "CEC classic", "Expanded pairwise Scaffer F6 function.", (-100.0, 100.0), expanded_scaffer, 10, 2, 200),
    ("cec_drop_wave", "CEC Drop-Wave", "Fixed-dimensional", "Two-dimensional drop-wave function, shifted to zero optimum.", (-5.12, 5.12), drop_wave, 2, 2, 2),
    ("cec_cross_in_tray", "CEC Cross-in-Tray", "Fixed-dimensional", "Two-dimensional cross-in-tray function, shifted to zero optimum.", (-10.0, 10.0), cross_in_tray, 2, 2, 2),
    ("cec_holder_table", "CEC Holder Table", "Fixed-dimensional", "Two-dimensional Holder table function, shifted to zero optimum.", (-10.0, 10.0), holder_table, 2, 2, 2),
    ("cec_eggholder", "CEC Eggholder", "Fixed-dimensional", "Two-dimensional eggholder function, shifted to zero optimum.", (-512.0, 512.0), eggholder, 2, 2, 2),
]


for row in BENCHMARKS:
    register(*row)

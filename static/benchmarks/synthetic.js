(function () {
  const { registerBenchmark, utils } = window.ParetoBenchmarkSuite;
  const { clamp } = utils;

  function fitVector(values, length) {
    return Array.from({ length }, (_, index) => values[index % values.length]);
  }

  function mlSystemTuning(x) {
    const [depth, compression, cache] = fitVector(x, 3);
    const quality =
      0.58 +
      0.28 * Math.exp(-((depth - 0.72) ** 2) / 0.045) +
      0.13 * Math.exp(-((compression - 0.28) ** 2) / 0.025) -
      0.18 * (compression - 0.72) ** 2 -
      0.08 * cache +
      0.03 * Math.sin(6.0 * Math.PI * depth);

    const speed =
      0.34 +
      0.48 * compression +
      0.22 * cache -
      0.3 * depth +
      0.06 * Math.sin(5.0 * Math.PI * cache + 0.5) -
      0.05 * Math.abs(compression - cache);

    const cost =
      0.78 -
      0.55 * depth +
      0.22 * compression +
      0.1 * cache -
      0.11 * (cache - 0.6) ** 2 +
      0.04 * Math.cos(4.0 * Math.PI * compression);

    return [clamp(quality), clamp(speed), clamp(cost)];
  }

  function dtlz2Like(x) {
    const [angleA, angleB, shape] = fitVector(x, 3);
    const scale = 1 + (shape - 0.5) ** 2;
    const f1 = scale * Math.cos((angleA * Math.PI) / 2) * Math.cos((angleB * Math.PI) / 2);
    const f2 = scale * Math.cos((angleA * Math.PI) / 2) * Math.sin((angleB * Math.PI) / 2);
    const f3 = scale * Math.sin((angleA * Math.PI) / 2);
    return [clamp(f1 / 1.25), clamp(f2 / 1.25), clamp(f3 / 1.25)];
  }

  function vehicleSafetyLike(x) {
    const [rail, bumper, cabin] = fitVector(x, 3);
    const crashQuality =
      0.42 + 0.38 * rail + 0.22 * cabin - 0.16 * bumper - 0.12 * (rail - cabin) ** 2 + 0.04 * Math.sin(5 * Math.PI * bumper);
    const speedProxy = 0.82 - 0.36 * rail - 0.3 * cabin + 0.16 * bumper - 0.08 * Math.abs(rail - 0.35);
    const materialEfficiency =
      0.84 - 0.45 * rail - 0.34 * cabin - 0.1 * bumper + 0.08 * Math.exp(-((bumper - 0.65) ** 2) / 0.035);
    return [clamp(crashQuality), clamp(speedProxy), clamp(materialEfficiency)];
  }

  function ruggedTradeoff(x) {
    const [x1, x2, x3] = fitVector(x, 3);
    const quality =
      0.48 +
      0.22 * Math.sin(3 * Math.PI * x1) +
      0.16 * Math.cos(4 * Math.PI * x2) +
      0.22 * Math.exp(-((x1 - 0.72) ** 2 + (x2 - 0.28) ** 2) / 0.035) -
      0.1 * x3;
    const speed =
      0.44 +
      0.22 * Math.sin(3 * Math.PI * x2 + 0.8) +
      0.26 * Math.exp(-((x2 - 0.75) ** 2 + (x3 - 0.65) ** 2) / 0.04) +
      0.12 * x3 -
      0.16 * x1;
    const cost =
      0.72 -
      0.28 * x1 +
      0.18 * Math.cos(5 * Math.PI * x3) +
      0.18 * Math.exp(-((x1 - 0.22) ** 2 + (x3 - 0.72) ** 2) / 0.03) -
      0.1 * x2;
    return [clamp(quality), clamp(speed), clamp(cost)];
  }

  registerBenchmark("ml_system", {
    name: "ML system tuning",
    description: "Three-way tradeoff among quality, serving speed, and resource efficiency.",
    minDimension: 3,
    defaultDimension: 3,
    inputs: ["Model depth", "Compression", "Cache"],
    objectives: [
      { key: "accuracy", label: "Accuracy", preferenceKey: "quality" },
      { key: "throughput", label: "Throughput", preferenceKey: "speed" },
      { key: "efficiency", label: "Resource efficiency", preferenceKey: "cost" },
    ],
    evaluate: mlSystemTuning,
  });

  registerBenchmark("dtlz2", {
    name: "DTLZ2-like smooth front",
    description: "A smooth concave Pareto surface adapted from the DTLZ2 family.",
    minDimension: 3,
    defaultDimension: 3,
    inputs: ["Angle A", "Angle B", "Shape"],
    objectives: [
      { key: "f1", label: "Objective f1", preferenceKey: "quality" },
      { key: "f2", label: "Objective f2", preferenceKey: "speed" },
      { key: "f3", label: "Objective f3", preferenceKey: "cost" },
    ],
    evaluate: dtlz2Like,
  });

  registerBenchmark("vehicle", {
    name: "Vehicle safety-like",
    description: "Engineering-style tradeoff among crash quality, speed proxy, and material efficiency.",
    minDimension: 3,
    defaultDimension: 3,
    inputs: ["Rail thickness", "Bumper stiffness", "Cabin reinforcement"],
    objectives: [
      { key: "crash_safety", label: "Crash safety", preferenceKey: "quality" },
      { key: "speed_proxy", label: "Speed proxy", preferenceKey: "speed" },
      { key: "material_efficiency", label: "Material efficiency", preferenceKey: "cost" },
    ],
    evaluate: vehicleSafetyLike,
  });

  registerBenchmark("rugged", {
    name: "Rugged multimodal tradeoff",
    description: "A nonconvex landscape with several local Pareto regions.",
    minDimension: 3,
    defaultDimension: 3,
    inputs: ["Design x1", "Design x2", "Design x3"],
    objectives: [
      { key: "mode_a", label: "Mode A score", preferenceKey: "quality" },
      { key: "mode_b", label: "Mode B score", preferenceKey: "speed" },
      { key: "mode_c", label: "Mode C score", preferenceKey: "cost" },
    ],
    evaluate: ruggedTradeoff,
  });
})();

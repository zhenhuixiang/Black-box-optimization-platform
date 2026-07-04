(function () {
  const { registerBenchmark, utils } = window.ParetoBenchmarkSuite;
  const { inputLabels, minimizeToScore } = utils;

  function dtlzProblem(x, m, variant) {
    const k = x.length - m + 1;
    const tail = x.slice(x.length - k);
    const g =
      variant === 1 || variant === 3
        ? 100 * (k + tail.reduce((sum, value) => sum + (value - 0.5) ** 2 - Math.cos(20 * Math.PI * (value - 0.5)), 0))
        : tail.reduce((sum, value) => sum + (value - 0.5) ** 2, 0);

    const values = [];
    if (variant === 1) {
      for (let i = 0; i < m; i += 1) {
        let value = 0.5 * (1 + g);
        for (let j = 0; j < m - i - 1; j += 1) value *= x[j];
        if (i > 0) value *= 1 - x[m - i - 1];
        values.push(value);
      }
      const upper = 0.5 * (1 + 225 * k);
      return values.map((value) => minimizeToScore(value, 0, upper));
    }

    for (let i = 0; i < m; i += 1) {
      let value = 1 + g;
      for (let j = 0; j < m - i - 1; j += 1) value *= Math.cos((x[j] * Math.PI) / 2);
      if (i > 0) value *= Math.sin((x[m - i - 1] * Math.PI) / 2);
      values.push(value);
    }
    const upper = variant === 3 ? 1 + 225 * k : 1 + 0.25 * k;
    return values.map((value) => minimizeToScore(value, 0, upper));
  }

  function objectiveDefs(count) {
    const preferenceKeys = ["quality", "speed", "cost"];
    return Array.from({ length: count }, (_, index) => ({
      key: `f${index + 1}_score`,
      label: `f${index + 1} score`,
      preferenceKey: preferenceKeys[index % preferenceKeys.length],
    }));
  }

  function registerDtlz(key, variant, defaultDimension, description) {
    registerBenchmark(key, {
      name: `Classic ${key.replace("_classic", "").toUpperCase()}`,
      description,
      minDimension: 3,
      defaultDimension,
      inputLabels,
      objectives: objectiveDefs(3),
      evaluate: (x) => dtlzProblem(x, 3, variant),
    });
  }

  registerDtlz(
    "dtlz1_classic",
    1,
    7,
    "Three-objective linear Pareto front from the DTLZ benchmark suite, transformed from minimization to normalized maximization scores.",
  );
  registerDtlz(
    "dtlz2_classic",
    2,
    12,
    "Three-objective concave Pareto surface from the DTLZ benchmark suite, transformed to maximization scores.",
  );
  registerDtlz(
    "dtlz3_classic",
    3,
    12,
    "Three-objective multimodal DTLZ problem with many local fronts, transformed to maximization scores.",
  );
})();

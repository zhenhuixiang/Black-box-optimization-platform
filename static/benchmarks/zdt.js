(function () {
  const { registerBenchmark, utils } = window.ParetoBenchmarkSuite;
  const { inputLabels, minimizeToScore } = utils;

  function zdtProblem(x, variant) {
    const f1 = x[0];
    const tail = x.slice(1);
    const g = 1 + (9 * tail.reduce((sum, value) => sum + value, 0)) / Math.max(1, tail.length);
    const ratio = f1 / g;
    let h;
    if (variant === 2) h = 1 - ratio ** 2;
    else if (variant === 3) h = 1 - Math.sqrt(ratio) - ratio * Math.sin(10 * Math.PI * f1);
    else h = 1 - Math.sqrt(ratio);
    const f2 = g * h;
    const f2Lower = variant === 3 ? -1 : 0;
    return [minimizeToScore(f1, 0, 1), minimizeToScore(f2, f2Lower, 10)];
  }

  function registerZdt(key, variant, description) {
    registerBenchmark(key, {
      name: `Classic ${key.toUpperCase()}`,
      description,
      minDimension: 2,
      defaultDimension: 30,
      inputLabels,
      objectives: [
        { key: "f1_score", label: "f1 score", preferenceKey: "quality" },
        { key: "f2_score", label: "f2 score", preferenceKey: "speed" },
      ],
      evaluate: (x) => zdtProblem(x, variant),
    });
  }

  registerZdt(
    "zdt1",
    1,
    "Two-objective convex Pareto front from the ZDT benchmark suite, transformed from minimization to normalized maximization scores.",
  );
  registerZdt(
    "zdt2",
    2,
    "Two-objective nonconvex Pareto front from the ZDT benchmark suite, shown as maximization scores.",
  );
  registerZdt(
    "zdt3",
    3,
    "Two-objective disconnected Pareto front from the ZDT benchmark suite, shown as maximization scores.",
  );
})();

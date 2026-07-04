(function () {
  const benchmarks = {};

  function clamp(value, lo = 0, hi = 1) {
    return Math.min(hi, Math.max(lo, value));
  }

  function inputLabels(count, prefix = "x") {
    return Array.from({ length: count }, (_, index) => `${prefix}${index + 1}`);
  }

  function registerBenchmark(key, definition) {
    if (!key || !definition || typeof definition.evaluate !== "function") {
      throw new Error(`Invalid benchmark registration: ${key}`);
    }
    benchmarks[key] = {
      ...definition,
      key,
      minDimension: definition.minDimension || definition.inputs?.length || 1,
      defaultDimension: definition.defaultDimension || definition.inputs?.length || definition.minDimension || 1,
    };
  }

  function resolveBenchmark(benchmark = "ml_system") {
    return benchmarks[benchmark] || benchmarks.ml_system || Object.values(benchmarks)[0];
  }

  function benchmarkInputs(definition, dimension = null) {
    const d = Math.max(definition.minDimension || 1, Number(dimension || definition.defaultDimension || definition.inputs?.length || 1));
    if (definition.inputLabels) return definition.inputLabels(d);
    if (definition.inputs && definition.inputs.length === d) return definition.inputs;
    return inputLabels(d);
  }

  function materializeBenchmark(benchmark, dimension = null) {
    const definition = resolveBenchmark(benchmark);
    const inputs = benchmarkInputs(definition, dimension);
    return {
      ...definition,
      inputs,
      defaultDimension: definition.defaultDimension || inputs.length,
      minDimension: definition.minDimension || inputs.length,
    };
  }

  function evaluateBenchmark(x, benchmark = "ml_system") {
    const definition = resolveBenchmark(benchmark);
    return definition.evaluate(x);
  }

  window.ParetoBenchmarkSuite = {
    benchmarks,
    registerBenchmark,
    materializeBenchmark,
    evaluateBenchmark,
    utils: {
      clamp,
      inputLabels,
      minimizeToScore(value, lower, upper) {
        return clamp((upper - value) / Math.max(1e-9, upper - lower));
      },
    },
  };
})();

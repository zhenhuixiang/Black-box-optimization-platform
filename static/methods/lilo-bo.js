(function () {
  const suite = window.ParetoOptimizerSuite;

  function runLiloBo(batchSize = 3) {
    const rt = suite.runtime();
    const { demo, RBFPosterior, approximateHypervolume, evaluateBenchmark } = rt;
    const posterior = new RBFPosterior(demo.x, demo.y);
    const hvBefore = approximateHypervolume(demo.y, demo.hvSamples);
    const pool = candidatePool();
    const [mean, std] = posterior.predict(pool);
    const uncertainty = std.map((row) => Math.hypot(...row));
    const strategy = selectStrategy(hvBefore);
    const scores = acquisitionScores(strategy, mean, std);
    const chosen = suite.helpers.diverseTopK(pool, scores, batchSize);
    const chosenX = chosen.map((idx) => pool[idx]);
    const chosenY = chosenX.map((row) => evaluateBenchmark(row, demo.benchmark));
    suite.applyRoundResults(
      "LILO-BO",
      strategy,
      chosenX,
      chosenY,
      chosen.map((idx) => mean[idx]),
      chosen.map((idx) => uncertainty[idx]),
      hvBefore,
      chosen.map((idx) => scores[idx]),
      uncertainty,
    );
  }

  function candidatePool() {
    const rt = suite.runtime();
    const { demo, paretoMask, activeWeights, activeBenchmarkDefinition, clamp } = rt;
    const inputDim = activeBenchmarkDefinition().inputs.length;
    const pool = [];
    const randomCount = Math.max(220, demo.settings.populationSize * 80);
    for (let i = 0; i < randomCount; i += 1) {
      pool.push(Array.from({ length: inputDim }, () => demo.rng.random()));
    }
    const paretoX = demo.x.filter((_, i) => paretoMask(demo.y)[i]);
    const localCount = Math.max(8, Math.round(demo.settings.populationSize * 2.5));
    for (const base of paretoX) {
      for (let i = 0; i < localCount; i += 1) {
        pool.push(base.map((v) => clamp(v + demo.rng.normal(0, 0.09))));
      }
    }
    const weights = activeWeights();
    const focus = Object.entries(weights).sort((a, b) => b[1] - a[1])[0][0];
    const centers = {
      quality: [0.72, 0.3, 0.25],
      speed: [0.35, 0.8, 0.82],
      cost: [0.2, 0.72, 0.68],
    };
    const balanced = [0.48, 0.55, 0.52];
    const center = suite.helpers.fitVector(centers[focus].map((v, i) => 0.65 * v + 0.35 * balanced[i]), inputDim);
    const priorCount = Math.max(80, demo.settings.populationSize * 20);
    for (let i = 0; i < priorCount; i += 1) pool.push(center.map((v) => clamp(v + demo.rng.normal(0, 0.18))));
    return suite.helpers.dedupeVectors(pool);
  }

  function selectStrategy(hv) {
    const rt = suite.runtime();
    const { demo, paretoMask, activeWeights, benchmarkPreferenceKeys } = rt;
    const active = activeWeights();
    const weights = benchmarkPreferenceKeys(demo.benchmark).map((key) => active[key] ?? 0);
    const prefStrength = Math.max(...weights) - Math.min(...weights);
    const constraints = Object.keys(demo.parsed.constraints);
    const feasibleRate = constraints.length ? demo.y.filter((row) => isFeasible(row)).length / demo.y.length : 1;
    const recent = demo.events.slice(0, 3).join(" ");
    if (constraints.length && feasibleRate < 0.55) return "constraint-aware preference repair";
    if (demo.iteration < 2 && prefStrength < 0.1) return "qNEHVI-style Pareto expansion";
    if (prefStrength > 0.18) return "preference-weighted hypervolume improvement";
    if (recent.includes("hypervolume +0.000") || paretoMask(demo.y).filter(Boolean).length < 4) return "uncertainty-guided Pareto exploration";
    if (hv > 0.45) return "knee-region exploitation";
    return "qParEGO-style randomized scalarization";
  }

  function isFeasible(y) {
    const { demo, activeBenchmarkDefinition } = suite.runtime();
    const objectives = activeBenchmarkDefinition().objectives;
    return Object.entries(demo.parsed.constraints).every(([key, threshold]) => {
      const matching = objectives
        .map((obj, index) => [obj, index])
        .filter(([obj]) => obj.preferenceKey === key)
        .map(([, index]) => index);
      return matching.length === 0 || matching.every((index) => y[index] >= threshold);
    });
  }

  function acquisitionScores(strategy, mean, std) {
    const rt = suite.runtime();
    const { demo, paretoMask, activeWeights, activeBenchmarkDefinition, preferenceScore, normalize, dot } = rt;
    const mask = paretoMask(demo.y);
    const paretoY = demo.y.filter((_, i) => mask[i]);
    const weights = activeWeights();
    const currentBest = Math.max(...demo.y.map((row) => preferenceScore(row, weights, demo.parsed.constraints)));
    return mean.map((m, i) => {
      const uncertainty = Math.hypot(...std[i]);
      const domination = paretoImprovementProxy(m, paretoY);
      const pref = preferenceScore(m, weights, demo.parsed.constraints);
      const prefImprovement = Math.max(0, pref - currentBest);
      const balanced = Math.min(...m);
      const feasibility = predictedFeasibility(m, std[i]);
      let score;
      if (strategy === "constraint-aware preference repair") score = 1.35 * feasibility + 0.7 * prefImprovement + 0.22 * uncertainty;
      else if (strategy === "preference-weighted hypervolume improvement") score = 0.9 * domination + 1.4 * prefImprovement + 0.28 * uncertainty;
      else if (strategy === "uncertainty-guided Pareto exploration") score = 0.55 * domination + 1.2 * uncertainty + 0.25 * balanced;
      else if (strategy === "knee-region exploitation") score = 0.7 * balanced + 0.7 * pref + 0.35 * domination;
      else if (strategy === "qParEGO-style randomized scalarization") {
        const objectives = activeBenchmarkDefinition().objectives;
        const sampled = normalize(objectives.map((obj) => 0.15 + (weights[obj.preferenceKey] ?? weights.quality ?? 0.33) + demo.rng.random() * 0.22));
        score = dot(m, sampled) + 0.35 * uncertainty + 0.45 * domination;
      } else score = domination + 0.42 * uncertainty + 0.2 * balanced;
      return score * feasibility;
    });
  }

  function paretoImprovementProxy(point, paretoY) {
    const { average } = suite.runtime();
    if (!paretoY.length) return average(point);
    return Math.max(
      ...paretoY.map((p) => {
        const positive = point.reduce((acc, v, i) => acc + Math.max(0, v - p[i]), 0);
        const negative = point.reduce((acc, v, i) => acc + Math.max(0, p[i] - v), 0);
        return positive - 0.35 * negative;
      }),
    );
  }

  function predictedFeasibility(mean, std) {
    const { demo, activeBenchmarkDefinition } = suite.runtime();
    const objectives = activeBenchmarkDefinition().objectives;
    return Object.entries(demo.parsed.constraints).reduce((acc, [key, threshold]) => {
      const matching = objectives
        .map((obj, index) => [obj, index])
        .filter(([obj]) => obj.preferenceKey === key)
        .map(([, index]) => index);
      return matching.reduce((innerAcc, index) => {
        const z = (mean[index] - threshold) / Math.max(std[index], 0.04);
        return innerAcc * (1 / (1 + Math.exp(-1.6 * z)));
      }, acc);
    }, 1);
  }

  suite.registerOptimizer(
    "lilo_bo",
    {
      name: "Pareto-aware LILO-BO",
      shortName: "LILO-BO",
      description: "Language-guided Bayesian optimization with an RBF surrogate, Pareto-aware acquisition, and uncertainty exploration.",
    },
    runLiloBo,
  );
})();

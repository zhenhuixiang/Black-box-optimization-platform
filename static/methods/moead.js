(function () {
  const suite = window.ParetoOptimizerSuite;

  function runMOEAD(batchSize = 3) {
    const rt = suite.runtime();
    const { demo, approximateHypervolume, preferenceScore, activeWeights } = rt;
    const weights = moeadWeights(batchSize);
    const reference = demo.y.map((row) => row.map((value) => value));
    const archive = suite.helpers.buildEvolutionArchive();
    const parentPool = archive.x;
    const parentY = archive.y;
    const chosen = [];
    const chosenY = [];
    const chosenPred = [];
    const chosenUnc = [];
    for (const weight of weights) {
      const candidate = proposeMOEADCandidate(parentPool, parentY, weight, reference);
      chosen.push(candidate.x);
      chosenY.push(candidate.y);
      chosenPred.push(candidate.y);
      chosenUnc.push(0.18);
    }
    suite.applyRoundResults(
      "MOEA/D",
      "decomposition-based scalar subproblems",
      chosen,
      chosenY,
      chosenPred,
      chosenUnc,
      approximateHypervolume(demo.y, demo.hvSamples),
      chosenY.map((row) => preferenceScore(row, activeWeights(), demo.parsed.constraints)),
    );
  }

  function moeadWeights(batchSize) {
    const rt = suite.runtime();
    const { demo, activeBenchmarkDefinition, activeWeights, normalize } = rt;
    const objectives = activeBenchmarkDefinition().objectives;
    const active = activeWeights();
    const base = normalize(objectives.map((obj) => active[obj.preferenceKey] ?? 1 / objectives.length));
    const weights = [];
    for (let i = 0; i < batchSize; i += 1) {
      const jitter = Array.from({ length: objectives.length }, () => 0.12 + demo.rng.random() * 0.45);
      weights.push(normalize(base.map((value, idx) => Math.max(0.02, value + jitter[idx] * (i / Math.max(1, batchSize - 1) - 0.5)))));
    }
    return weights;
  }

  function proposeMOEADCandidate(parentPool, parentY, weight, reference) {
    const rt = suite.runtime();
    const { demo, evaluateBenchmark, preferenceScore, activeWeights } = rt;
    const parents = parentPool.length ? parentPool : demo.x;
    const parentScores = parentY.length
      ? parentY.map((row) => tchebycheffScore(row, weight, idealPoint(reference)) + 0.05 * preferenceScore(row, activeWeights(), demo.parsed.constraints))
      : [];
    const order = parentScores
      .map((score, index) => [score, index])
      .sort((a, b) => b[0] - a[0])
      .map((pair) => pair[1]);
    const first = parents[order[0] ?? Math.floor(demo.rng.random() * parents.length)];
    const second = parents[order[1] ?? Math.floor(demo.rng.random() * parents.length)];
    const x = suite.helpers.mutateVector(suite.helpers.blendVectors(first, second, 0.7), 0.12);
    const y = evaluateBenchmark(x, demo.benchmark);
    return { x, y };
  }

  function idealPoint(reference) {
    if (!reference.length) return [];
    return reference.reduce((best, row) => row.map((value, i) => Math.max(best[i], value)), Array(reference[0].length).fill(0));
  }

  function tchebycheffScore(row, weight, ideal) {
    if (!ideal.length) return 0;
    return -Math.max(...row.map((value, index) => weight[index] * Math.abs(ideal[index] - value)));
  }

  suite.registerOptimizer(
    "moead",
    {
      name: "MOEA/D",
      shortName: "MOEA/D",
      description: "Decomposition baseline using preference-shaped weight vectors and Tchebycheff scalar subproblems.",
    },
    runMOEAD,
  );
})();

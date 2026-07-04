(function () {
  const suite = window.ParetoOptimizerSuite;

  function runNSGA2(batchSize = 3) {
    const rt = suite.runtime();
    const { demo, evaluateBenchmark, approximateHypervolume, preferenceScore, activeWeights } = rt;
    const archive = suite.helpers.buildEvolutionArchive();
    const population = archive.x;
    const offspring = [];
    while (offspring.length < batchSize) {
      const [a, b] = suite.helpers.tournamentPair(population, archive.y);
      offspring.push(suite.helpers.mutateVector(suite.helpers.blendVectors(population[a], population[b], 0.5), 0.16));
      if (offspring.length < batchSize) {
        offspring.push(suite.helpers.mutateVector(suite.helpers.blendVectors(population[b], population[a], 0.5), 0.16));
      }
    }
    const chosenX = suite.helpers.dedupeVectors(offspring).slice(0, batchSize);
    while (chosenX.length < batchSize) {
      const base = population[Math.floor(demo.rng.random() * population.length)];
      chosenX.push(suite.helpers.mutateVector(base, 0.22));
    }
    const chosenY = chosenX.map((row) => evaluateBenchmark(row, demo.benchmark));
    const uncertainty = chosenX.map(() => 0.18);
    suite.applyRoundResults(
      "NSGA-II",
      "ask-tell offspring from non-dominated tournament parents",
      chosenX,
      chosenY,
      chosenY,
      uncertainty,
      approximateHypervolume(demo.y, demo.hvSamples),
      chosenY.map((row) => preferenceScore(row, activeWeights(), demo.parsed.constraints)),
    );
  }

  suite.registerOptimizer(
    "nsga2",
    {
      name: "NSGA-II",
      shortName: "NSGA-II",
      description: "Evolutionary baseline using non-dominated sorting, crowding distance, crossover, and mutation.",
    },
    runNSGA2,
  );
})();

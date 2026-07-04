(function () {
  const { $, reset, updateFeedback, render, snapshot } = window.ParetoDemo;
  const { runRound, runToBudget } = window.ParetoOptimizerSuite;

  function currentSeed() {
    return Date.now() % 100000;
  }

  function currentFeedback() {
    return $("feedback").value;
  }

  function currentBenchmark() {
    return $("benchmarkSelect").value;
  }

  function currentOptimizer() {
    return $("optimizerSelect").value;
  }

  function currentSettings() {
    return {
      populationSize: $("populationSize").value,
      maxEvaluations: $("maxEvaluations").value,
      dimension: $("dimensionInput").value,
    };
  }

  function resetCurrent() {
    reset(currentSeed(), currentFeedback(), currentBenchmark(), currentOptimizer(), currentSettings());
  }

  function resetForBenchmarkChange() {
    const definition = window.ParetoBenchmarkSuite.materializeBenchmark(currentBenchmark());
    reset(currentSeed(), currentFeedback(), currentBenchmark(), currentOptimizer(), {
      ...currentSettings(),
      dimension: definition.defaultDimension,
    });
  }

  $("applyFeedback").addEventListener("click", () => updateFeedback(currentFeedback()));
  $("step").addEventListener("click", () => runRound());
  $("applySettings").addEventListener("click", resetCurrent);
  $("runBudget").addEventListener("click", () => runToBudget());
  $("benchmarkSelect").addEventListener("change", resetForBenchmarkChange);
  $("optimizerSelect").addEventListener("change", resetCurrent);
  $("objectiveView").addEventListener("change", () => render(snapshot()));
  $("auto").addEventListener("click", async () => {
    for (let i = 0; i < 5; i += 1) {
      runRound();
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  });
  $("reset").addEventListener("click", resetCurrent);

  reset();
})();

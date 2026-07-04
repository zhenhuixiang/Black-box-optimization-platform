(function () {
  const optimizers = {};
  const runners = {};

  function runtime() {
    if (!window.ParetoRuntime) {
      throw new Error("ParetoRuntime is not ready. Load standalone.js before optimizer modules.");
    }
    return window.ParetoRuntime;
  }

  function registerOptimizer(key, metadata, runner) {
    if (!key || typeof runner !== "function") throw new Error(`Invalid optimizer registration: ${key}`);
    optimizers[key] = { ...metadata };
    runners[key] = runner;
  }

  function budgetRemaining() {
    const { demo } = runtime();
    return Math.max(0, demo.settings.maxEvaluations - demo.x.length);
  }

  function budgetedBatchSize(requestedSize = null) {
    const { demo } = runtime();
    const requested = requestedSize === null ? demo.settings.populationSize : requestedSize;
    return Math.max(0, Math.min(Math.max(1, Math.round(requested)), budgetRemaining()));
  }

  function runRound(batchSize = null) {
    const rt = runtime();
    const { demo, render, snapshot } = rt;
    const runner = runners[demo.optimizer];
    const actualBatch = budgetedBatchSize(batchSize);
    if (!runner || actualBatch <= 0) {
      finishExperiment(0, 0, runner ? "Budget already reached." : "No optimizer runner registered.");
      render(snapshot());
      return;
    }
    runner(actualBatch);
  }

  async function runToBudget() {
    const rt = runtime();
    const { demo, render, snapshot } = rt;
    const started = Date.now();
    let rounds = 0;
    demo.experimentResult = null;
    while (budgetRemaining() > 0) {
      const before = demo.x.length;
      runRound(Math.min(demo.settings.populationSize, budgetRemaining()));
      rounds += 1;
      if (demo.x.length <= before) {
        finishExperiment(rounds, Date.now() - started, "Budget run stopped: optimizer made no progress.");
        render(snapshot());
        return;
      }
      if (rounds % 4 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    finishExperiment(rounds, Date.now() - started, "Budget run completed.");
    render(snapshot());
  }

  function finishExperiment(rounds, elapsedMs, note) {
    const rt = runtime();
    const { demo, approximateHypervolume, paretoMask, activeWeights, preferenceScore } = rt;
    const mask = paretoMask(demo.y);
    const weights = activeWeights();
    const scores = demo.y.map((row) => preferenceScore(row, weights, demo.parsed.constraints));
    const bestScore = scores.length ? Math.max(...scores) : 0;
    const bestIndex = scores.indexOf(bestScore);
    demo.experimentResult = {
      note,
      rounds,
      elapsedMs,
      evaluations: demo.x.length,
      maxEvaluations: demo.settings.maxEvaluations,
      paretoCount: mask.filter(Boolean).length,
      hypervolume: approximateHypervolume(demo.y, demo.hvSamples),
      recommendedId: bestIndex,
      recommendedScore: bestScore,
    };
    demo.events.unshift(`${note} ${demo.x.length}/${demo.settings.maxEvaluations} evaluations, ${demo.experimentResult.paretoCount} Pareto points.`);
    demo.events = demo.events.slice(0, 8);
  }

  function applyRoundResults(methodName, strategyLabel, chosenX, chosenY, predicted, uncertainty, hvBefore, scores = null, candidateUncertainty = null) {
    const rt = runtime();
    const { demo, approximateHypervolume, preferenceScore, activeWeights, makeSurrogateReport, average, render, snapshot } = rt;
    const n = Math.min(chosenX.length, budgetRemaining());
    if (n <= 0) {
      finishExperiment(0, 0, "Budget already reached.");
      render(snapshot());
      return;
    }
    const nextX = chosenX.slice(0, n);
    const nextY = chosenY.slice(0, n);
    const hvAfter = approximateHypervolume([...demo.y, ...nextY], demo.hvSamples);
    demo.x.push(...nextX);
    demo.y.push(...nextY);
    demo.iteration += 1;
    demo.lastStrategy = `${methodName}: ${strategyLabel}`;
    demo.lastCandidates = nextX.map((_, i) => ({
      x: nextX[i],
      predicted: predicted?.[i] || nextY[i],
      observed: nextY[i],
      uncertainty: uncertainty?.[i] ?? 0.18,
      score: scores?.[i] ?? preferenceScore(nextY[i], activeWeights(), demo.parsed.constraints),
    }));
    demo.surrogateReport =
      methodName === "LILO-BO"
        ? makeSurrogateReport(candidateUncertainty || uncertainty, uncertainty)
        : {
            model: `${methodName} baseline`,
            containsUncertainty: methodName !== "NSGA-II" ? true : false,
            calibrated: false,
            meanCandidateUncertainty: uncertainty?.length ? average(uncertainty) : null,
            selectedCandidateUncertainty: uncertainty?.length ? average(uncertainty) : null,
            note:
              methodName === "NSGA-II"
                ? "NSGA-II uses deterministic ranking plus a fixed mutation rate in this demo; no surrogate uncertainty is modeled."
                : "MOEA/D uses decomposition with deterministic scalar subproblem scoring in this demo; it does not use a calibrated surrogate uncertainty model.",
          };
    demo.events.unshift(`Round ${demo.iteration}: ${demo.lastStrategy}; hypervolume +${(hvAfter - hvBefore).toFixed(4)}.`);
    demo.events = demo.events.slice(0, 8);
    render(snapshot());
  }

  function fitVector(values, length) {
    return Array.from({ length }, (_, index) => values[index % values.length]);
  }

  function distance(a, b) {
    return Math.hypot(...a.map((v, i) => v - b[i]));
  }

  function diverseTopK(pool, scores, k) {
    const order = scores
      .map((score, i) => [score, i])
      .sort((a, b) => b[0] - a[0])
      .map((pair) => pair[1]);
    const chosen = [];
    for (const idx of order) {
      if (!chosen.length || Math.min(...chosen.map((j) => distance(pool[idx], pool[j]))) > 0.12) chosen.push(idx);
      if (chosen.length === k) return chosen;
    }
    for (const idx of order) {
      if (!chosen.includes(idx)) chosen.push(idx);
      if (chosen.length === k) break;
    }
    return chosen;
  }

  function dedupeVectors(rows) {
    const seen = new Set();
    return rows.filter((row) => {
      const key = row.map((value) => value.toFixed(4)).join(",");
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function buildEvolutionPopulation() {
    return buildEvolutionArchive().x;
  }

  function buildEvolutionArchive() {
    const rt = runtime();
    const { demo, paretoMask, activeBenchmarkDefinition } = rt;
    const inputDim = activeBenchmarkDefinition().inputs.length;
    if (!demo.x.length) {
      return {
        x: [Array.from({ length: inputDim }, () => demo.rng.random())],
        y: [],
      };
    }
    const mask = paretoMask(demo.y);
    const cap = Math.min(demo.x.length, Math.max(16, demo.settings.populationSize * 4));
    const chosen = [];
    const add = (idx) => {
      if (idx >= 0 && idx < demo.x.length && !chosen.includes(idx)) chosen.push(idx);
    };
    mask.forEach((isPareto, idx) => {
      if (isPareto && chosen.length < cap) add(idx);
    });
    for (let idx = Math.max(0, demo.x.length - Math.ceil(cap / 3)); idx < demo.x.length && chosen.length < cap; idx += 1) add(idx);
    for (let i = 0; i < cap && chosen.length < cap; i += 1) add(Math.floor((i * demo.x.length) / cap));
    return {
      x: chosen.map((idx) => demo.x[idx]),
      y: chosen.map((idx) => demo.y[idx]),
    };
  }

  function blendVectors(a, b, spread = 0.5) {
    const { demo, clamp } = runtime();
    return a.map((value, i) => {
      const alpha = clamp(0.5 + demo.rng.normal(0, spread * 0.15), 0, 1);
      return clamp(alpha * value + (1 - alpha) * b[i]);
    });
  }

  function mutateVector(base, sigma = 0.12) {
    const { demo, clamp } = runtime();
    return base.map((value) => {
      const shouldMutate = demo.rng.random() < 0.35;
      return clamp(value + (shouldMutate ? demo.rng.normal(0, sigma) : 0));
    });
  }

  function tournamentPair(population, populationY = null) {
    const { demo, evaluateBenchmark } = runtime();
    const y = populationY?.length === population.length ? populationY : population.map((row) => evaluateBenchmark(row, demo.benchmark));
    const fronts = nonDominatedSort(y);
    const rank = Array(population.length).fill(0);
    fronts.forEach((front, frontIndex) => front.forEach((idx) => (rank[idx] = frontIndex)));
    const crowd = Array(population.length).fill(0);
    fronts.forEach((front) => {
      const values = crowdingDistance(front, y);
      front.forEach((idx, i) => (crowd[idx] = values[i]));
    });
    const pick = () => {
      const a = Math.floor(demo.rng.random() * population.length);
      const b = Math.floor(demo.rng.random() * population.length);
      if (rank[a] !== rank[b]) return rank[a] < rank[b] ? a : b;
      return crowd[a] >= crowd[b] ? a : b;
    };
    return [pick(), pick()];
  }

  function nonDominatedSort(y) {
    const dominates = Array.from({ length: y.length }, () => []);
    const dominatedBy = Array(y.length).fill(0);
    const fronts = [[]];
    for (let i = 0; i < y.length; i += 1) {
      for (let j = 0; j < y.length; j += 1) {
        if (i === j) continue;
        const iDominates = y[i].every((v, k) => v >= y[j][k]) && y[i].some((v, k) => v > y[j][k]);
        if (iDominates) dominates[i].push(j);
        else if (y[j].every((v, k) => v >= y[i][k]) && y[j].some((v, k) => v > y[i][k])) dominatedBy[i] += 1;
      }
      if (dominatedBy[i] === 0) fronts[0].push(i);
    }
    let level = 0;
    while (fronts[level]?.length) {
      const next = [];
      for (const i of fronts[level]) {
        for (const j of dominates[i]) {
          dominatedBy[j] -= 1;
          if (dominatedBy[j] === 0) next.push(j);
        }
      }
      level += 1;
      if (next.length) fronts[level] = next;
    }
    return fronts.filter((front) => front.length);
  }

  function crowdingDistance(front, y) {
    const distance = Array(front.length).fill(0);
    if (front.length <= 2) return distance.map(() => Infinity);
    const m = y[0].length;
    for (let obj = 0; obj < m; obj += 1) {
      const order = front.map((idx, i) => [idx, i]).sort((a, b) => y[a[0]][obj] - y[b[0]][obj]);
      distance[order[0][1]] = Infinity;
      distance[order[order.length - 1][1]] = Infinity;
      const min = y[order[0][0]][obj];
      const max = y[order[order.length - 1][0]][obj];
      const span = Math.max(1e-9, max - min);
      for (let i = 1; i < order.length - 1; i += 1) {
        distance[order[i][1]] += (y[order[i + 1][0]][obj] - y[order[i - 1][0]][obj]) / span;
      }
    }
    return distance;
  }

  window.ParetoOptimizerSuite = {
    optimizers,
    registerOptimizer,
    runRound,
    runToBudget,
    runtime,
    applyRoundResults,
    helpers: {
      fitVector,
      diverseTopK,
      dedupeVectors,
      buildEvolutionPopulation,
      buildEvolutionArchive,
      blendVectors,
      mutateVector,
      tournamentPair,
      nonDominatedSort,
      crowdingDistance,
    },
  };
})();

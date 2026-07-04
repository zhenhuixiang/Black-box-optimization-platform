const colors = {
  quality: "#2563eb",
  speed: "#0f9f8f",
  cost: "#2f8a4c",
};

const $ = (id) => document.getElementById(id);

const preferenceKeys = ["quality", "speed", "cost"];

const { benchmarks, materializeBenchmark, evaluateBenchmark } = window.ParetoBenchmarkSuite;

class RNG {
  constructor(seed = 7) {
    this.seed = seed >>> 0;
    this.nextSpare = null;
  }

  random() {
    this.seed = (1664525 * this.seed + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }

  normal(mean = 0, std = 1) {
    if (this.nextSpare !== null) {
      const value = this.nextSpare;
      this.nextSpare = null;
      return mean + std * value;
    }
    const u = Math.max(this.random(), 1e-12);
    const v = Math.max(this.random(), 1e-12);
    const mag = Math.sqrt(-2 * Math.log(u));
    const z0 = mag * Math.cos(2 * Math.PI * v);
    const z1 = mag * Math.sin(2 * Math.PI * v);
    this.nextSpare = z1;
    return mean + std * z0;
  }
}

const demo = {
  rng: new RNG(7),
  x: [],
  y: [],
  hvSamples: [],
  benchmark: "ml_system",
  optimizer: "lilo_bo",
  feedback: "I want a balanced design, but quality should remain strong.",
  parsed: null,
  manualWeights: null,
  iteration: 0,
  lastStrategy: "initial",
  lastCandidates: [],
  surrogateReport: null,
  experimentResult: null,
  settings: {
    populationSize: 9,
    maxEvaluations: 60,
    dimension: 3,
  },
  events: [],
  view: {
    mode: "2d",
    yaw: -Math.PI / 4,
    pitch: Math.PI / 6,
    drag: null,
  },
};

function clamp(value, lo = 0, hi = 1) {
  return Math.min(hi, Math.max(lo, value));
}

function activeBenchmarkDefinition(benchmark = demo.benchmark, dimension = demo.settings.dimension) {
  return materializeBenchmark(benchmark, dimension);
}

function sanitizeExperimentSettings(raw = {}, benchmark = demo.benchmark) {
  const definition = materializeBenchmark(benchmark, raw.dimension ?? demo.settings.dimension);
  const minDimension = definition.minDimension || 1;
  const defaultDimension = definition.defaultDimension || definition.inputs.length;
  const dimension = Math.max(minDimension, Math.min(200, Math.round(Number(raw.dimension ?? defaultDimension) || defaultDimension)));
  const populationSize = Math.max(1, Math.min(200, Math.round(Number(raw.populationSize ?? demo.settings.populationSize) || 9)));
  const maxEvaluations = Math.max(populationSize, Math.min(5000, Math.round(Number(raw.maxEvaluations ?? demo.settings.maxEvaluations) || 60)));
  return { populationSize, maxEvaluations, dimension };
}

function latinHypercube(n, d, rng) {
  const samples = Array.from({ length: n }, () => Array(d).fill(0));
  for (let j = 0; j < d; j += 1) {
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i -= 1) {
      const r = Math.floor(rng.random() * (i + 1));
      [perm[i], perm[r]] = [perm[r], perm[i]];
    }
    for (let i = 0; i < n; i += 1) samples[i][j] = (perm[i] + rng.random()) / n;
  }
  return samples;
}

function paretoMask(y) {
  return y.map((point, i) => {
    return !y.some((other, j) => {
      if (i === j) return false;
      const allBetter = other.every((v, k) => v >= point[k]);
      const oneStrict = other.some((v, k) => v > point[k]);
      return allBetter && oneStrict;
    });
  });
}

function approximateHypervolume(y, samples) {
  if (!y.length) return 0;
  let dominated = 0;
  for (const sample of samples) {
    if (y.some((point) => point.every((v, k) => v >= sample[k]))) dominated += 1;
  }
  return dominated / samples.length;
}

function parseLanguageFeedback(text) {
  const lower = text.toLowerCase();
  let weights = [0.34, 0.33, 0.33];
  const constraints = {};
  const notes = [];
  const signals = {
    quality: ["quality", "accuracy", "accurate", "性能", "精度", "准确", "效果", "质量", "安全", "鲁棒"],
    speed: ["speed", "latency", "fast", "faster", "quick", "delay", "速度", "延迟", "响应", "快", "实时"],
    cost: ["cost", "cheap", "budget", "memory", "small", "efficient", "resource", "成本", "便宜", "资源", "显存", "内存", "小", "效率"],
  };
  ["quality", "speed", "cost"].forEach((key, idx) => {
    const count = signals[key].reduce((acc, token) => acc + countOf(lower, token), 0);
    if (count) {
      weights[idx] += 0.22 * count;
      notes.push(`Raised ${key} weight from language cues (${count}).`);
    }
  });

  const floors = [
    ["quality", /(quality|accuracy|精度|质量|效果)\s*(>=|>|at\s+least|above|over|no\s+less\s+than|至少|不低于|高于|超过)\s*([0-9]+(?:\.[0-9]+)?)\s*%?/],
    ["speed", /(speed|latency|速度|延迟)\s*(>=|>|at\s+least|above|over|no\s+less\s+than|至少|不低于|高于|超过)\s*([0-9]+(?:\.[0-9]+)?)\s*%?/],
    ["cost", /(cost|成本|资源|显存|内存)\s*(>=|>|at\s+least|above|over|no\s+less\s+than|至少|不低于|高于|超过)\s*([0-9]+(?:\.[0-9]+)?)\s*%?/],
  ];
  for (const [key, pattern] of floors) {
    const match = lower.match(pattern);
    if (match) {
      const raw = Number(match[3]);
      constraints[key] = clamp(raw > 1 ? raw / 100 : raw, 0.05, 0.95);
      notes.push(`Detected ${key} target around ${constraints[key].toFixed(2)}.`);
    }
  }

  if (lower.includes("balanced") || lower.includes("balance") || lower.includes("均衡") || lower.includes("折中")) {
    weights = weights.map((w) => 0.5 * w + 0.5);
    notes.push("Pulled weights toward a balanced tradeoff.");
  }

  weights = normalize(weights.map((w) => Math.max(0.05, w)));
  if (!notes.length) notes.push("No strong preference cue found; using a balanced Pareto search.");
  const labels = ["quality", "speed", "cost"];
  const top = labels[weights.indexOf(Math.max(...weights))];
  const floorsText = Object.keys(constraints).length
    ? `soft floors: ${Object.entries(constraints)
        .map(([k, v]) => `${k} >= ${v.toFixed(2)}`)
        .join(", ")}`
    : "no hard floors";
  return {
    weights: { quality: weights[0], speed: weights[1], cost: weights[2] },
    constraints,
    notes,
    summary: `Preference focus: ${top}; ${floorsText}`,
  };
}

function countOf(text, token) {
  return text.split(token).length - 1;
}

function normalize(values) {
  const total = values.reduce((a, b) => a + b, 0);
  return values.map((v) => v / total);
}

function preferenceScore(y, weights, constraints, objectives = activeBenchmarkDefinition().objectives) {
  const objectiveWeights = normalize(objectives.map((obj) => weights[obj.preferenceKey] ?? weights.quality ?? 1));
  let score = y.reduce((acc, v, i) => acc + v * objectiveWeights[i], 0);
  for (const [key, threshold] of Object.entries(constraints)) {
    objectives.forEach((obj, index) => {
      if (obj.preferenceKey === key) score -= 2.5 * Math.max(0, threshold - y[index]);
    });
  }
  return score;
}

function activeWeights() {
  const base = demo.manualWeights || demo.parsed.weights;
  const relevant = benchmarkPreferenceKeys(demo.benchmark);
  const total = relevant.reduce((sum, key) => sum + (base[key] ?? 0), 0) || 1;
  return Object.fromEntries(preferenceKeys.map((key) => [key, relevant.includes(key) ? (base[key] ?? 0) / total : 0]));
}

function setManualWeight(key, rawValue) {
  const relevant = benchmarkPreferenceKeys(demo.benchmark);
  const current = { ...activeWeights() };
  current[key] = Math.max(0.01, Number(rawValue) / 100);
  const total = relevant.reduce((sum, entryKey) => sum + (current[entryKey] ?? 0), 0) || 1;
  demo.manualWeights = Object.fromEntries(preferenceKeys.map((entryKey) => [entryKey, relevant.includes(entryKey) ? (current[entryKey] ?? 0) / total : 0]));
  demo.events.unshift("Manually adjusted preference weights.");
  demo.events = demo.events.slice(0, 8);
  render(snapshot());
}

function benchmarkPreferenceKeys(benchmark = demo.benchmark) {
  return [...new Set(activeBenchmarkDefinition(benchmark).objectives.map((objective) => objective.preferenceKey || "quality"))];
}

class RBFPosterior {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.lengthscale = 0.28;
    this.noise = 1e-4;
    this.yMean = columnStats(y, "mean");
    this.yStd = columnStats(y, "std").map((v) => Math.max(v, 0.06));
    const scaled = y.map((row) => row.map((v, j) => (v - this.yMean[j]) / this.yStd[j]));
    const kernel = x.map((a, i) => x.map((b, j) => rbf(a, b, this.lengthscale) + (i === j ? this.noise : 0)));
    this.invKernel = inverse(kernel);
    this.alpha = matMul(this.invKernel, scaled);
  }

  predict(pool) {
    const means = [];
    const stds = [];
    for (const row of pool) {
      const k = this.x.map((base) => rbf(row, base, this.lengthscale));
      const meanScaled = vecMat(k, this.alpha);
      const invK = vecMat(k, this.invKernel);
      const rawVar = clamp(1 - dot(invK, k), 0.0025, 1);
      means.push(meanScaled.map((v, j) => clamp(v * this.yStd[j] + this.yMean[j])));
      stds.push(this.yStd.map((s) => Math.sqrt(rawVar) * s));
    }
    return [means, stds];
  }
}

function rbf(a, b, lengthscale) {
  const dist = a.reduce((acc, v, i) => acc + (v - b[i]) ** 2, 0);
  return Math.exp(-0.5 * dist / lengthscale ** 2);
}

function columnStats(values, kind) {
  return values[0].map((_, j) => {
    const col = values.map((row) => row[j]);
    const mean = col.reduce((a, b) => a + b, 0) / col.length;
    if (kind === "mean") return mean;
    const variance = col.reduce((a, b) => a + (b - mean) ** 2, 0) / col.length;
    return Math.sqrt(variance);
  });
}

function inverse(matrix) {
  const n = matrix.length;
  const aug = matrix.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
    }
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]];
    const div = Math.abs(aug[col][col]) < 1e-10 ? 1e-10 : aug[col][col];
    for (let j = 0; j < 2 * n; j += 1) aug[col][j] /= div;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j += 1) aug[row][j] -= factor * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

function matMul(a, b) {
  return a.map((row) => b[0].map((_, j) => row.reduce((acc, v, i) => acc + v * b[i][j], 0)));
}

function vecMat(vec, matrix) {
  return matrix[0].map((_, j) => vec.reduce((acc, v, i) => acc + v * matrix[i][j], 0));
}

function dot(a, b) {
  return a.reduce((acc, v, i) => acc + v * b[i], 0);
}

function makeSurrogateReport(candidateUncertainty = null, selectedUncertainty = null) {
  const report = {
    model: "Fixed-lengthscale RBF posterior proxy",
    containsUncertainty: true,
    calibrated: false,
    meanCandidateUncertainty: null,
    selectedCandidateUncertainty: null,
    note:
      "This demo includes uncertainty through an RBF posterior variance proxy. It is useful for exploration, but it is not a fully calibrated GP: lengthscale/noise are fixed, objectives are independent, and no marginal likelihood fitting is performed.",
  };
  if (candidateUncertainty?.length) report.meanCandidateUncertainty = average(candidateUncertainty);
  if (selectedUncertainty?.length) report.selectedCandidateUncertainty = average(selectedUncertainty);
  return report;
}

function optimizerCatalog() {
  return window.ParetoOptimizerSuite?.optimizers || {};
}

function reset(seed = 7, feedback = demo.feedback, benchmark = demo.benchmark, optimizer = demo.optimizer, settings = {}) {
  const optimizers = optimizerCatalog();
  demo.rng = new RNG(seed);
  demo.benchmark = benchmarks[benchmark] ? benchmark : "ml_system";
  demo.settings = sanitizeExperimentSettings(settings, demo.benchmark);
  demo.optimizer = optimizers[optimizer] ? optimizer : Object.keys(optimizers)[0] || "lilo_bo";
  const optimizerMeta = optimizers[demo.optimizer] || { name: demo.optimizer, shortName: demo.optimizer };
  const benchmarkDefinition = activeBenchmarkDefinition();
  const inputDim = benchmarkDefinition.inputs.length;
  const objectiveDim = benchmarkDefinition.objectives.length;
  demo.x = latinHypercube(demo.settings.populationSize, inputDim, demo.rng);
  demo.y = demo.x.map((row) => evaluateBenchmark(row, demo.benchmark));
  demo.hvSamples = Array.from({ length: 10000 }, () => Array.from({ length: objectiveDim }, () => demo.rng.random()));
  demo.feedback = feedback;
  demo.parsed = parseLanguageFeedback(feedback);
  demo.manualWeights = null;
  demo.iteration = 0;
  demo.lastStrategy = `${optimizerMeta.shortName} initialization`;
  demo.lastCandidates = [];
  demo.surrogateReport = makeSurrogateReport();
  demo.experimentResult = null;
  demo.events = [
    `Initialized ${demo.settings.populationSize} space-filling evaluations on ${benchmarkDefinition.name}.`,
    `Experiment budget: ${demo.settings.maxEvaluations} max evaluations, d=${demo.settings.dimension}.`,
    `Selected optimizer: ${optimizerMeta.name}.`,
    "Parsed the first natural-language preference message.",
  ];
  render(snapshot());
}

function updateFeedback(feedback) {
  demo.feedback = feedback.trim() || demo.feedback;
  demo.parsed = parseLanguageFeedback(demo.feedback);
  demo.manualWeights = null;
  demo.events.unshift("Updated language preference model.");
  demo.events = demo.events.slice(0, 8);
  render(snapshot());
}

function average(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function snapshot() {
  const optimizers = optimizerCatalog();
  const benchmarkDefinition = activeBenchmarkDefinition();
  const optimizerMeta = optimizers[demo.optimizer] || { name: demo.optimizer, description: "" };
  const mask = paretoMask(demo.y);
  const weights = activeWeights();
  const scores = demo.y.map((row) => preferenceScore(row, weights, demo.parsed.constraints));
  const recommendedIdx = scores.indexOf(Math.max(...scores));
  const points = demo.x.map((x, i) => ({
    id: i,
    x,
    y: demo.y[i],
    pareto: mask[i],
    preferenceScore: scores[i],
    recommended: i === recommendedIdx,
  }));
  return {
    benchmarks: Object.entries(benchmarks).map(([key, value]) => ({
      key,
      name: value.name,
      description: value.description,
      defaultDimension: value.defaultDimension,
      minDimension: value.minDimension,
    })),
    optimizers: Object.entries(optimizers).map(([key, value]) => ({
      key,
      name: value.name,
      description: value.description,
    })),
    benchmark: { key: demo.benchmark, ...benchmarkDefinition },
    optimizer: { key: demo.optimizer, ...optimizerMeta },
    inputs: benchmarkDefinition.inputs.map((label, index) => ({ key: `x${index + 1}`, label })),
    objectives: benchmarkDefinition.objectives.map((objective) => ({ ...objective, goal: "max" })),
    dimensions: {
      inputs: benchmarkDefinition.inputs.length,
      objectives: benchmarkDefinition.objectives.length,
    },
    settings: { ...demo.settings },
    budgetRemaining: Math.max(0, demo.settings.maxEvaluations - points.length),
    iteration: demo.iteration,
    nEvaluations: points.length,
    hypervolume: approximateHypervolume(demo.y, demo.hvSamples),
    paretoCount: mask.filter(Boolean).length,
    points,
    pareto: points.filter((p) => p.pareto),
    recommended: points[recommendedIdx],
    feedback: demo.feedback,
    parsed: demo.parsed,
    activeWeights: weights,
    manualWeights: demo.manualWeights,
    surrogate: demo.surrogateReport || makeSurrogateReport(),
    lastStrategy: demo.lastStrategy,
    lastCandidates: demo.lastCandidates,
    experimentResult: demo.experimentResult,
    events: demo.events,
  };
}

function fmt(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function pct(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function objectiveName(index) {
  return (window.currentObjectives && window.currentObjectives[index]) || `Objective ${index + 1}`;
}

function render(data) {
  window.currentInputs = data.inputs.map((item) => item.label);
  window.currentObjectives = data.objectives.map((item) => item.label);
  renderBenchmarkControls(data);
  renderExperimentSettings(data);
  $("feedback").value = data.feedback;
  $("nEvaluations").textContent = data.nEvaluations;
  $("paretoCount").textContent = data.paretoCount;
  $("hypervolume").textContent = fmt(data.hypervolume, 3);
  $("strategy").textContent = data.lastStrategy;
  const uncertainty = data.surrogate.selectedCandidateUncertainty ?? data.surrogate.meanCandidateUncertainty;
  $("uncertainty").textContent = uncertainty === null ? "yes, proxy" : `${fmt(uncertainty)} proxy`;
  $("surrogateNote").textContent = data.surrogate.note;
  renderWeights(data);
  renderEvents(data);
  renderScatter(data);
  renderParetoList(data);
  renderRecommendation(data);
  renderCandidates(data);
  renderExperimentResult(data);
}

function renderBenchmarkControls(data) {
  const select = $("benchmarkSelect");
  if (select.options.length !== data.benchmarks.length) {
    select.innerHTML = data.benchmarks.map((item) => `<option value="${item.key}">${escapeHtml(item.name)}</option>`).join("");
  }
  select.value = data.benchmark.key;
  $("benchmarkSummary").textContent = data.benchmark.description;
  const optimizerSelect = $("optimizerSelect");
  if (optimizerSelect.options.length !== data.optimizers.length) {
    optimizerSelect.innerHTML = data.optimizers.map((item) => `<option value="${item.key}">${escapeHtml(item.name)}</option>`).join("");
  }
  optimizerSelect.value = data.optimizer.key;
  $("optimizerSummary").textContent = data.optimizer.description;
  $("problemDefinition").innerHTML = `
    <div class="definition-row">
      <strong>Decision variables (d=${data.dimensions.inputs})</strong>
      <div class="chips">${data.inputs.map((item) => `<span class="chip">${escapeHtml(item.label)}</span>`).join("")}</div>
    </div>
    <div class="definition-row">
      <strong>Objectives (m=${data.dimensions.objectives}, all normalized for maximization)</strong>
      <div class="chips">${data.objectives.map((item) => `<span class="chip">${escapeHtml(item.label)}</span>`).join("")}</div>
    </div>
  `;
}

function renderExperimentSettings(data) {
  $("populationSize").value = data.settings.populationSize;
  $("maxEvaluations").value = data.settings.maxEvaluations;
  $("dimensionInput").min = data.benchmark.minDimension || 1;
  $("dimensionInput").value = data.settings.dimension;
  $("settingsSummary").textContent = `d=${data.dimensions.inputs}, population=${data.settings.populationSize}, budget=${data.nEvaluations}/${data.settings.maxEvaluations}, remaining=${data.budgetRemaining}.`;
}

function renderWeights(data) {
  const rows = data.objectives.map((objective) => [objective.preferenceKey, objective.label]);
  $("weights").innerHTML = rows
    .map(([key, label]) => {
      const value = data.activeWeights[key];
      return `
        <div class="weight-row">
          <span>${label}</span>
          <input class="weight-slider" type="range" min="1" max="99" value="${Math.round(value * 100)}" data-weight-key="${key}" aria-label="${escapeHtml(label)} preference weight" />
          <strong>${pct(value)}</strong>
        </div>
      `;
    })
    .join("");
  const sliders = $("weights").querySelectorAll ? $("weights").querySelectorAll("[data-weight-key]") : [];
  sliders.forEach((input) => {
    input.addEventListener("input", () => setManualWeight(input.dataset.weightKey, input.value));
  });
  $("preferenceSummary").textContent = data.manualWeights ? "Manual preference weights are active." : data.parsed.summary;
  $("notes").innerHTML = data.parsed.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
}

function renderEvents(data) {
  $("events").innerHTML = data.events.map((event) => `<li>${escapeHtml(event)}</li>`).join("");
}

function renderScatter(data) {
  const select = $("objectiveView");
  const canRender3d = data.objectives.length >= 3;
  if (select) {
    select.disabled = !canRender3d;
    if (!canRender3d && select.value === "3d") select.value = "2d";
  }
  const mode = canRender3d && select?.value === "3d" ? "3d" : "2d";
  demo.view.mode = mode;
  if (mode === "3d") renderScatter3d(data);
  else renderScatter2d(data);
}

function renderScatter2d(data) {
  const svg = $("scatter");
  const objectiveLabels = data.objectives.map((item) => item.label);
  const xLabel = objectiveLabels[0] || "Objective 1";
  const yLabel = objectiveLabels[1] || "Objective 2";
  const colorLabel = objectiveLabels[2] || objectiveLabels[objectiveLabels.length - 1] || "Objective";
  $("chartSubtitle").textContent = `${xLabel} vs ${yLabel}; Pareto points are outlined${objectiveLabels.length > 2 ? `; ${colorLabel} controls color` : ""}`;
  const width = 760;
  const height = 470;
  const margin = { left: 58, right: 34, top: 28, bottom: 58 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const xScale = (v) => margin.left + v * innerW;
  const yScale = (v) => margin.top + (1 - v) * innerH;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const grid = ticks
    .map(
      (t) => `
      <line class="grid" x1="${xScale(t)}" x2="${xScale(t)}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(t)}" y2="${yScale(t)}"></line>
      <text class="tick-label" x="${xScale(t)}" y="${height - 32}" text-anchor="middle">${t.toFixed(2)}</text>
      <text class="tick-label" x="44" y="${yScale(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>
    `,
    )
    .join("");
  const pareto = [...data.pareto].sort((a, b) => a.y[0] - b.y[0]);
  const line =
    pareto.length > 1
      ? `<path class="front-line" d="${pareto.map((p, i) => `${i ? "L" : "M"} ${xScale(p.y[0])} ${yScale(p.y[1])}`).join(" ")}"></path>`
      : "";
  const points = data.points
    .map((p) => {
      const radius = p.recommended ? 8 : p.pareto ? 6.5 : 4.7;
      const klass = `point${p.pareto ? " pareto" : ""}${p.recommended ? " recommended" : ""}`;
      const colorValue = p.y[2] ?? p.y[p.y.length - 1] ?? 0.5;
      return `<circle class="${klass}" cx="${xScale(p.y[0])}" cy="${yScale(p.y[1])}" r="${radius}" fill="${objectiveColor(colorValue)}"><title>${escapeHtml(pointTitle(p))}</title></circle>`;
    })
    .join("");
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${grid}
    <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
    <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
    <text class="axis-label" x="${margin.left + innerW / 2}" y="${height - 13}" text-anchor="middle">${escapeHtml(xLabel)}</text>
    <text class="axis-label" x="18" y="${margin.top + innerH / 2}" transform="rotate(-90 18 ${margin.top + innerH / 2})" text-anchor="middle">${escapeHtml(yLabel)}</text>
    ${line}
    ${points}
    <text class="legend" x="${width - 308}" y="28">${objectiveLabels.length > 2 ? `Color: ${escapeHtml(colorLabel)}, ` : ""}outline: Pareto, red: recommended</text>
  `;
}

function renderScatter3d(data) {
  const svg = $("scatter");
  const labels = data.objectives.map((item) => item.label);
  const shown = [labels[0] || "Objective 1", labels[1] || "Objective 2", labels[2] || "Objective 3"];
  $("chartSubtitle").textContent = `3D objective view: drag to rotate; ${shown.join(", ")}; Pareto points are outlined`;
  const width = 760;
  const height = 470;
  const origin = project3d([0, 0, 0]);
  const axisEnds = [
    { end: project3d([1, 0, 0]), label: shown[0] },
    { end: project3d([0, 1, 0]), label: shown[1] },
    { end: project3d([0, 0, 1]), label: shown[2] },
  ];
  const corners = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
  ].map(project3d);
  const edgePairs = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [1, 5],
    [2, 4],
    [2, 6],
    [3, 5],
    [3, 6],
    [4, 7],
    [5, 7],
    [6, 7],
  ];
  const edges = edgePairs
    .map(([a, b]) => `<line class="axis-3d soft" x1="${corners[a].x}" y1="${corners[a].y}" x2="${corners[b].x}" y2="${corners[b].y}"></line>`)
    .join("");
  const axes = axisEnds
    .map(
      ({ end, label }) => `
      <line class="axis-3d" x1="${origin.x}" y1="${origin.y}" x2="${end.x}" y2="${end.y}"></line>
      <text class="axis-label" x="${end.x}" y="${end.y - 8}" text-anchor="middle">${escapeHtml(label)}</text>
    `,
    )
    .join("");
  const pareto = [...data.pareto]
    .map((p) => ({ ...p, projected: project3d([p.y[0], p.y[1], p.y[2]]) }))
    .sort((a, b) => a.y[0] - b.y[0]);
  const line =
    pareto.length > 1
      ? `<path class="front-line" d="${pareto.map((p, i) => `${i ? "L" : "M"} ${p.projected.x} ${p.projected.y}`).join(" ")}"></path>`
      : "";
  const points = data.points
    .map((p) => {
      const projected = project3d([p.y[0], p.y[1], p.y[2]]);
      return { ...p, projected };
    })
    .sort((a, b) => a.projected.depth - b.projected.depth)
    .map((p) => {
      const radius = p.recommended ? 8.2 : p.pareto ? 6.7 : 4.8;
      const klass = `point${p.pareto ? " pareto" : ""}${p.recommended ? " recommended" : ""}`;
      return `<circle class="${klass}" cx="${p.projected.x}" cy="${p.projected.y}" r="${radius}" fill="${objectiveColor(p.y[2])}"><title>${escapeHtml(pointTitle(p))}</title></circle>`;
    })
    .join("");
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${edges}
    ${axes}
    ${line}
    ${points}
    <text class="legend" x="${width - 390}" y="28">Drag to rotate, dashed: sampled Pareto link, red: recommended</text>
  `;
  bind3dInteractions(svg);
}

function project3d(values) {
  const centered = values.slice(0, 3).map((value) => Number(value ?? 0) - 0.5);
  const angleY = demo.view.yaw;
  const angleX = demo.view.pitch;
  const cosY = Math.cos(angleY);
  const sinY = Math.sin(angleY);
  const cosX = Math.cos(angleX);
  const sinX = Math.sin(angleX);
  const x1 = centered[0] * cosY + centered[2] * sinY;
  const z1 = -centered[0] * sinY + centered[2] * cosY;
  const y1 = centered[1];
  const y2 = y1 * cosX - z1 * sinX;
  const z2 = y1 * sinX + z1 * cosX;
  return {
    x: 380 + x1 * 360,
    y: 248 - y2 * 310,
    depth: z2,
  };
}

function bind3dInteractions(svg) {
  if (!svg || svg.dataset.bound3d === "1") return;
  svg.dataset.bound3d = "1";
  svg.addEventListener("pointerdown", (event) => {
    if (demo.view.mode !== "3d") return;
    demo.view.drag = { x: event.clientX, y: event.clientY, yaw: demo.view.yaw, pitch: demo.view.pitch };
    svg.setPointerCapture?.(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!demo.view.drag) return;
    const dx = event.clientX - demo.view.drag.x;
    const dy = event.clientY - demo.view.drag.y;
    demo.view.yaw = demo.view.drag.yaw + dx * 0.012;
    demo.view.pitch = clamp(demo.view.drag.pitch + dy * 0.012, -1.1, 1.1);
    renderScatter(snapshot());
  });
  const endDrag = () => {
    demo.view.drag = null;
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("pointerleave", endDrag);
}

function objectiveColor(value) {
  return `hsl(${8 + clamp(Number(value ?? 0.5)) * 142}, 68%, 46%)`;
}

function pointTitle(point) {
  const values = point.y.map((value, index) => `${objectiveName(index)} ${fmt(value)}`).join(", ");
  return `#${point.id} ${values}`;
}

function renderParetoList(data) {
  $("paretoList").innerHTML = [...data.pareto]
    .sort((a, b) => b.preferenceScore - a.preferenceScore)
    .slice(0, 9)
    .map((point) => solutionCard(point, point.recommended))
    .join("");
}

function renderRecommendation(data) {
  $("recommendation").innerHTML = solutionCard(data.recommended, true, true);
}

function renderCandidates(data) {
  if (!data.lastCandidates.length) {
    $("candidates").innerHTML = `<p class="summary">Run a BO round to evaluate candidate designs.</p>`;
    return;
  }
  $("candidates").innerHTML = data.lastCandidates
    .map((item, idx) => {
      const point = { id: `new-${idx + 1}`, x: item.x, y: item.observed, preferenceScore: item.score };
      return solutionCard(point, false, false, item.predicted);
    })
    .join("");
}

function renderExperimentResult(data) {
  if (!data.experimentResult) {
    $("experimentResult").textContent = `No budget run yet. Current budget progress is ${data.nEvaluations}/${data.settings.maxEvaluations}.`;
    return;
  }
  const result = data.experimentResult;
  $("experimentResult").textContent =
    `${result.note} evaluations ${result.evaluations}/${result.maxEvaluations}, Pareto points ${result.paretoCount}, ` +
    `hypervolume ${fmt(result.hypervolume, 4)}, recommended #${result.recommendedId} score ${fmt(result.recommendedScore, 4)}, ` +
    `rounds ${result.rounds}, elapsed ${result.elapsedMs} ms.`;
}

function solutionCard(point, recommended = false, expanded = false, predicted = null) {
  const badge = recommended ? `<span class="badge hot">Recommended</span>` : `<span class="badge">#${point.id}</span>`;
  const pred = predicted
    ? `<p class="summary">Predicted: ${predicted.map((value, index) => `${objectiveName(index)} ${fmt(value)}`).join(", ")}</p>`
    : "";
  const inputLabels = window.currentInputs || ["Depth", "Compression", "Cache"];
  const inputs = expanded
    ? `
      <div class="triple">${point.x.map((value, index) => mini(inputLabels[index] || `x${index + 1}`, value)).join("")}</div>
    `
    : "";
  return `
    <article class="solution${recommended ? " recommended" : ""}">
      <div class="solution-head">
        <strong>Preference score ${fmt(point.preferenceScore)}</strong>
        ${badge}
      </div>
      <div class="triple">${point.y.map((value, index) => mini(objectiveName(index), value)).join("")}</div>
      ${inputs}
      ${pred}
    </article>
  `;
}

function mini(label, value) {
  return `
    <div class="mini">
      <span>${label}</span>
      <strong>${fmt(value)}</strong>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.ParetoRuntime = {
  demo,
  benchmarks,
  activeBenchmarkDefinition,
  evaluateBenchmark,
  RBFPosterior,
  approximateHypervolume,
  paretoMask,
  activeWeights,
  benchmarkPreferenceKeys,
  preferenceScore,
  makeSurrogateReport,
  average,
  normalize,
  dot,
  clamp,
  render,
  snapshot,
};

window.ParetoDemo = {
  $,
  demo,
  reset,
  sanitizeExperimentSettings,
  activeBenchmarkDefinition,
  updateFeedback,
  render,
  snapshot,
};

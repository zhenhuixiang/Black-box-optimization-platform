const state = {
  data: null,
  busy: false,
  runningBudget: false,
  runningSingleBudget: false,
  experiment: null,
  singleObjectiveExperiment: null,
  singleObjective: null,
  singleObjectiveLandscape: null,
  singleObjectiveLandscapeLoadingKey: null,
  workspaceView: "singleObjective",
  historyLoaded: false,
  runHistory: [],
  view: {
    mode: "2d",
    yaw: -Math.PI / 4,
    pitch: Math.PI / 6,
    drag: null,
  },
  soView: {
    mode: "2d",
    yaw: -Math.PI / 4,
    pitch: Math.PI / 7,
    drag: null,
  },
};

const colors = {
  quality: "#2563eb",
  speed: "#0f9f8f",
  cost: "#2f8a4c",
};

const $ = (id) => document.getElementById(id);
const historyStorageKey = "blackboxlab.runHistory.v1";
const defaultMoExperimentMethods = ["lilo_bo", "classic_bo", "nsga2", "moead", "nsga3", "spea2", "qnehvi_bo", "trust_region_mobo"];
const expensiveMoExperimentMethods = ["lilo_bo", "classic_bo", "qnehvi_bo", "qparego_bo", "thompson_bo", "entropy_bo", "trust_region_mobo", "knowledge_gradient_mobo"];
const defaultSoExperimentMethods = ["random_search", "de", "pso", "ga", "rbf_bo", "cma_es", "surrogate_entropy_bo", "surrogate_turbo_bo"];
const expensiveSoExperimentMethods = ["rbf_bo", "trust_region_bo", "local_rbf_bo", "surrogate_ei_bo", "surrogate_ucb_bo", "surrogate_entropy_bo", "surrogate_kg_bo", "surrogate_turbo_bo"];

const customProblemExample = `from __future__ import annotations

import numpy as np

from benchmarks.registry import input_defs, objective_defs, register_benchmark


def evaluate_my_problem(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f1 = 1.0 - np.mean((x - 0.20) ** 2) / 0.64
    f2 = 1.0 - np.mean((x - 0.80) ** 2) / 0.64
    f3 = 1.0 - abs(float(np.mean(x)) - 0.50) / 0.50
    return np.clip(np.array([f1, f2, f3]), 0.0, 1.0)


register_benchmark(
    "custom_my_problem",
    {
        "name": "My submitted problem",
        "description": "User-submitted 3-objective optimization problem.",
        "minDimension": 2,
        "defaultDimension": 5,
        "maxDimension": 30,
        "inputLabels": input_defs,
        "objectives": objective_defs(3),
        "evaluate": evaluate_my_problem,
    },
)
`;

const customOptimizerExample = `from __future__ import annotations

import numpy as np

from benchmarks import evaluate_benchmark
from methods.registry import register_optimizer


def run_custom_random_local_search(demo, batch_size: int) -> dict:
    """Example ask-tell optimizer.

    The demo object owns the optimization state, RNG, benchmark key, settings,
    Pareto utilities, and the method used to append evaluated candidates.
    """
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")

    x = np.array(demo.state.x, dtype=float)
    y = np.array(demo.state.y, dtype=float)
    hv_before = demo.approximate_hypervolume(y, np.array(demo.state.hv_samples, dtype=float))
    pareto_x = x[demo.pareto_mask(y)]
    input_dim = demo.input_dimension()
    if len(pareto_x) == 0:
        base = demo.rng.random((batch_size, input_dim))
    else:
        parent_ids = demo.rng.integers(0, len(pareto_x), size=batch_size)
        base = pareto_x[parent_ids]

    chosen_x = np.clip(base + demo.rng.normal(0.0, 0.12, size=(batch_size, input_dim)), 0.0, 1.0)
    chosen_y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in chosen_x])
    predicted = chosen_y
    uncertainty = np.full(batch_size, 0.0)
    scores = chosen_y.mean(axis=1)
    return demo.apply_round_results(
        method="Custom RLS",
        strategy="Pareto-local random mutation",
        chosen_x=chosen_x,
        chosen_y=chosen_y,
        predicted=predicted,
        uncertainty=uncertainty,
        scores=scores,
        hv_before=hv_before,
        surrogate=demo.surrogate_report(),
    )


register_optimizer(
    "custom_random_local_search",
    {
        "name": "Custom random local search",
        "shortName": "Custom RLS",
        "description": "Example user-defined optimizer: mutates sampled Pareto archive members.",
    },
    run_custom_random_local_search,
)
`;

function optionSignature(items) {
  return (items || []).map((item) => item.key).join("|");
}

function optionSearchText(item) {
  return [item?.key, item?.name, item?.shortName, item?.suite, item?.description]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function optionHtml(item) {
  return `<option value="${escapeHtml(item.key)}" data-search="${escapeHtml(optionSearchText(item))}">${escapeHtml(item.name)}</option>`;
}

function setOptions(select, items) {
  select.innerHTML = (items || []).map(optionHtml).join("");
}

function applySelectFilter(selectId, inputId) {
  const select = $(selectId);
  const input = $(inputId);
  if (!select || !input) return;
  const terms = input.value
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  let visible = 0;
  Array.from(select.options).forEach((option) => {
    const haystack = option.dataset.search || `${option.textContent} ${option.value}`.toLowerCase();
    const matched = terms.every((term) => haystack.includes(term));
    const keepVisible = matched || option.selected;
    option.hidden = !keepVisible;
    option.disabled = !keepVisible;
    if (keepVisible) visible += 1;
  });
  input.title = terms.length ? `${visible} matching option(s)` : "";
}

function bindSelectFilter(inputId, selectId) {
  const input = $(inputId);
  if (!input || input.dataset.bound) return;
  input.dataset.bound = "1";
  input.addEventListener("input", () => applySelectFilter(selectId, inputId));
}

function displayNameExists(items, name) {
  const folded = String(name || "").trim().replace(/\s+/g, " ").toLowerCase();
  if (!folded) return false;
  return (items || []).some((item) => String(item.name || "").trim().replace(/\s+/g, " ").toLowerCase() === folded);
}

async function api(path, payload = null) {
  const options = payload
    ? {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    : {};
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = JSON.parse(await response.text());
      message = payload.error || message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }
  return response.json();
}

function fmt(value, digits = 3) {
  return Number(value).toFixed(digits);
}

const singleObjectiveBoundsFallback = {
  sphere: [-5, 5],
  rastrigin: [-5.12, 5.12],
  rosenbrock: [-2, 2],
  ackley: [-5, 5],
  griewank: [-6, 6],
  schwefel: [-500, 500],
  levy: [-10, 10],
  zakharov: [-5, 5],
  bbob_ellipsoid: [-5, 5],
  bbob_bent_cigar: [-5, 5],
  bbob_discus: [-5, 5],
};

function singleObjectiveBounds(data) {
  const raw = data?.benchmark?.bounds || singleObjectiveBoundsFallback[data?.benchmark?.key] || [0, 1];
  const lo = Number(raw?.[0]);
  const hi = Number(raw?.[1]);
  return Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : [0, 1];
}

function singleObjectiveInputBounds(data) {
  const dimension = Number(data?.dimensions?.inputs || data?.settings?.dimension || 1);
  const raw = data?.benchmark?.inputBounds;
  if (Array.isArray(raw) && raw.length === dimension) {
    return raw.map((item) => {
      const lo = Number(item?.[0]);
      const hi = Number(item?.[1]);
      return Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : [0, 1];
    });
  }
  return Array.from({ length: dimension }, () => singleObjectiveBounds(data));
}

function realInputValue(data, index, normalizedValue) {
  const bounds = singleObjectiveInputBounds(data)[index] || singleObjectiveBounds(data);
  const lo = Number(bounds[0]);
  const hi = Number(bounds[1]);
  return lo + clamp(Number(normalizedValue ?? 0)) * (hi - lo);
}

function realInputLabel(data, index, normalizedValue) {
  const bounds = singleObjectiveInputBounds(data)[index] || singleObjectiveBounds(data);
  const lo = Number(bounds[0]);
  const hi = Number(bounds[1]);
  const span = hi - lo;
  const digits = span >= 100 ? 2 : span >= 20 ? 3 : 4;
  return fmt(realInputValue(data, index, normalizedValue), digits);
}

function realDecisionValue(data, normalizedValue) {
  const [lo, hi] = singleObjectiveBounds(data);
  return lo + clamp(Number(normalizedValue ?? 0)) * (hi - lo);
}

function realDecisionVector(data, values) {
  return (values || []).map((value) => realDecisionValue(data, value));
}

function realDecisionMini(data, label, normalizedValue) {
  return mini(label, realDecisionValue(data, normalizedValue));
}

function realAxisTick(data, normalizedValue) {
  const [lo, hi] = singleObjectiveBounds(data);
  const value = lo + Number(normalizedValue) * (hi - lo);
  const span = hi - lo;
  const digits = span >= 100 ? 0 : span >= 20 ? 1 : 2;
  return fmt(value, digits);
}

function realDecisionLabel(data, value) {
  const [lo, hi] = singleObjectiveBounds(data);
  const span = hi - lo;
  const digits = span >= 100 ? 2 : span >= 20 ? 3 : 4;
  return fmt(realDecisionValue(data, value), digits);
}

function pct(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function objectiveName(index) {
  return (window.currentObjectives && window.currentObjectives[index]) || `Objective ${index + 1}`;
}

function render(data) {
  state.data = data;
  window.currentInputs = data.inputs.map((item) => item.label);
  window.currentObjectives = data.objectives.map((item) => item.label);
  renderPlatformStats();
  renderBenchmarkControls(data);
  $("feedback").value = data.feedback;
  $("nEvaluations").textContent = data.nEvaluations;
  $("paretoCount").textContent = data.paretoCount;
  renderSelectedMetric(data);
  $("strategy").textContent = data.lastStrategy;
  const uncertainty = data.surrogate.selectedCandidateUncertainty ?? data.surrogate.meanCandidateUncertainty;
  $("uncertainty").textContent = uncertainty === null ? "yes, proxy" : `${fmt(uncertainty)} proxy`;
  $("surrogateNote").textContent = data.surrogate.note;
  renderExperimentSettings(data);
  renderExperimentControls(data);
  renderWeights(data);
  renderEvents(data);
  renderScatter(data);
  renderParetoList(data);
  renderRecommendation(data);
  renderCandidates(data);
  renderExperimentResult(data);
  syncWorkspaceView();
}

function renderSelectedMetric(data) {
  const key = $("metricSelect")?.value || "hypervolume";
  const value = data.metrics?.[key] ?? data.hypervolume ?? 0;
  const direction = data.metricDirections?.[key] || "higher_is_better";
  $("selectedMetricValue").textContent = key === "paretoCount" ? String(Math.round(value)) : fmt(value, 4);
  $("selectedMetricDirection").textContent = direction === "lower_is_better" ? "lower is better" : "higher is better";
}

function setCustomProblemExample() {
  const editor = $("customProblemCode");
  if (editor && !editor.value.trim()) editor.value = customProblemExample;
  const nameInput = $("customProblemName");
  if (nameInput && !nameInput.value.trim()) nameInput.value = "My submitted problem";
}

function setCustomOptimizerExample() {
  const editor = $("customOptimizerCode");
  if (editor && !editor.value.trim()) editor.value = customOptimizerExample;
  const nameInput = $("customOptimizerName");
  if (nameInput && !nameInput.value.trim()) nameInput.value = "Custom random local search";
}

function importResultDetails(result, selectedLabel) {
  const keys = result.registeredKeys?.length ? ` Registered: ${result.registeredKeys.join(", ")}.` : "";
  const renamedEntries = Object.entries(result.renamedKeys || {});
  const renamed = renamedEntries.length
    ? ` Auto-renamed duplicate key(s): ${renamedEntries.map(([from, to]) => `${from} -> ${to}`).join(", ")}.`
    : "";
  const warnings = result.warnings?.length ? ` Warning: ${result.warnings.join(" ")}` : "";
  return `${selectedLabel}.${keys}${renamed}${warnings}`;
}

function syncWorkspaceView() {
  const singleTab = $("singleRunTab");
  const experimentTab = $("experimentTab");
  const singleObjectiveTab = $("singleObjectiveTab");
  const singleObjectiveExperimentTab = $("singleObjectiveExperimentTab");
  const singleView = $("singleRunView");
  const experimentView = $("experimentView");
  const singleObjectiveView = $("singleObjectiveView");
  const singleObjectiveExperimentView = $("singleObjectiveExperimentView");
  const view = state.workspaceView || "singleObjective";
  if (singleTab) singleTab.classList.toggle("active", view === "single");
  if (experimentTab) experimentTab.classList.toggle("active", view === "experiment");
  if (singleObjectiveTab) singleObjectiveTab.classList.toggle("active", view === "singleObjective");
  if (singleObjectiveExperimentTab) singleObjectiveExperimentTab.classList.toggle("active", view === "singleObjectiveExperiment");
  if (singleView) singleView.classList.toggle("active", view === "single");
  if (experimentView) experimentView.classList.toggle("active", view === "experiment");
  if (singleObjectiveView) singleObjectiveView.classList.toggle("active", view === "singleObjective");
  if (singleObjectiveExperimentView) singleObjectiveExperimentView.classList.toggle("active", view === "singleObjectiveExperiment");
  renderPlatformTools();
}

function renderBenchmarkControls(data) {
  const select = $("benchmarkSelect");
  if (select.dataset.signature !== optionSignature(data.benchmarks)) {
    setOptions(select, data.benchmarks);
    select.dataset.signature = optionSignature(data.benchmarks);
  }
  select.value = data.benchmark.key;
  applySelectFilter("benchmarkSelect", "benchmarkFilter");
  $("benchmarkSummary").textContent = data.benchmark.description;
  const optimizerSelect = $("optimizerSelect");
  if (optimizerSelect.dataset.signature !== optionSignature(data.optimizers)) {
    setOptions(optimizerSelect, data.optimizers);
    optimizerSelect.dataset.signature = optionSignature(data.optimizers);
  }
  optimizerSelect.value = data.optimizer.key;
  applySelectFilter("optimizerSelect", "optimizerFilter");
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

function renderExperimentControls(data) {
  const benchmarkSelect = $("expBenchmarks");
  const optimizerSelect = $("expOptimizers");
  const selectedBenchmarks = selectedValues(benchmarkSelect);
  const selectedOptimizers = selectedValues(optimizerSelect);
  const shouldRefreshBenchmarks = benchmarkSelect.dataset.signature !== optionSignature(data.benchmarks);
  const shouldRefreshOptimizers = optimizerSelect.dataset.signature !== optionSignature(data.optimizers);

  if (shouldRefreshBenchmarks) {
    const fallback = selectedBenchmarks.length ? selectedBenchmarks : [data.benchmark.key];
    setOptions(benchmarkSelect, data.benchmarks);
    benchmarkSelect.dataset.signature = optionSignature(data.benchmarks);
    setSelectedValues(benchmarkSelect, fallback);
  }
  if (shouldRefreshOptimizers) {
    const fallback = selectedOptimizers.length
      ? selectedOptimizers
      : preferredKeys(data.optimizers, defaultMoExperimentMethods, data.optimizer.key);
    setOptions(optimizerSelect, data.optimizers);
    optimizerSelect.dataset.signature = optionSignature(data.optimizers);
    setSelectedValues(optimizerSelect, fallback);
  }
  applySelectFilter("expBenchmarks", "expBenchmarkFilter");
  applySelectFilter("expOptimizers", "expOptimizerFilter");
  const referenceSelect = $("expReference");
  if (referenceSelect.dataset.signature !== optionSignature(data.optimizers)) {
    const currentReference = referenceSelect.value || (data.optimizers.some((item) => item.key === "lilo_bo") ? "lilo_bo" : data.optimizer.key);
    setOptions(referenceSelect, data.optimizers);
    referenceSelect.dataset.signature = optionSignature(data.optimizers);
    referenceSelect.value = currentReference;
  }

  if (!$("expPopulation").dataset.touched) $("expPopulation").value = data.settings.populationSize;
  if (!$("expBudget").dataset.touched) $("expBudget").value = data.settings.maxEvaluations;
  if (!$("expDimension").dataset.touched) $("expDimension").value = data.settings.dimension;
  if (!$("expObjectives").dataset.touched) $("expObjectives").value = data.settings.objectiveCount || data.dimensions.objectives;
  updateExperimentScale("multi");
}

function renderExperimentSettings(data) {
  $("populationSize").value = data.settings.populationSize;
  $("maxEvaluations").value = data.settings.maxEvaluations;
  $("dimensionInput").min = data.benchmark.minDimension || 1;
  $("dimensionInput").max = data.benchmark.maxDimension || 200;
  $("dimensionInput").value = data.settings.dimension;
  $("objectiveCountInput").min = data.benchmark.minObjectives || 2;
  $("objectiveCountInput").max = data.benchmark.maxObjectives || 15;
  $("objectiveCountInput").value = data.settings.objectiveCount || data.dimensions.objectives;
  $("settingsSummary").textContent = `Python backend: d=${data.dimensions.inputs}, M=${data.dimensions.objectives}, population=${data.settings.populationSize}, budget=${data.nEvaluations}/${data.settings.maxEvaluations}, remaining=${data.budgetRemaining}.`;
}

function renderSingleObjective(data) {
  state.singleObjective = data;
  renderPlatformStats();
  if (state.singleObjectiveLandscape && state.singleObjectiveLandscape.baseKey !== singleObjectiveLandscapeBaseKey(data)) {
    state.singleObjectiveLandscape = null;
  }
  renderSingleObjectiveControls(data);
  $("soEvents").innerHTML = data.events.map((event) => `<li>${escapeHtml(event)}</li>`).join("");
  renderSingleObjectiveSearch(data);
  renderSingleObjectiveConvergence(data);
  renderSingleObjectiveBest(data);
  renderSingleObjectiveCandidates(data);
  renderSingleObjectiveExperimentControls(data);
  renderPlatformTools();
}

function renderPlatformStats() {
  const container = $("platformStats");
  if (!container) return;
  const moBenchmarks = state.data?.benchmarks?.length ?? 0;
  const moMethods = state.data?.optimizers?.length ?? 0;
  const soBenchmarks = state.singleObjective?.benchmarks?.length ?? 0;
  const soMethods = state.singleObjective?.methods?.length ?? 0;
  container.innerHTML = `
    <div><span>Multi-objective</span><strong>${moBenchmarks || "-"} problems / ${moMethods || "-"} algorithms</strong></div>
    <div><span>Single-objective</span><strong>${soBenchmarks || "-"} problems / ${soMethods || "-"} algorithms</strong></div>
  `;
}

function registryErrors() {
  return [...new Set([
    ...(state.data?.moduleRegistry?.registryErrors || []),
    ...(state.singleObjective?.moduleRegistry?.registryErrors || []),
  ].filter(Boolean))];
}

function currentPlatformConfig() {
  const view = state.workspaceView || "singleObjective";
  if (view === "single") {
    return {
      page: "multi-objective-test",
      benchmark: $("benchmarkSelect")?.value || state.data?.benchmark?.key,
      method: $("optimizerSelect")?.value || state.data?.optimizer?.key,
      metric: $("metricSelect")?.value || "hypervolume",
      settings: currentSettings(),
      feedback: $("feedback")?.value || "",
      evaluations: state.data?.nEvaluations ?? 0,
    };
  }
  if (view === "experiment") {
    return {
      page: "multi-objective-experiments",
      benchmarks: selectedValues($("expBenchmarks")),
      methods: selectedValues($("expOptimizers")),
      metric: $("expMetric")?.value || "hypervolume",
      repeats: $("expRepeats")?.value,
      seed: $("expSeed")?.value,
      workers: $("expWorkers")?.value,
      settings: experimentSettings(),
    };
  }
  if (view === "singleObjectiveExperiment") {
    return {
      page: "single-objective-experiments",
      benchmarks: selectedValues($("soExpBenchmarks")),
      methods: selectedValues($("soExpMethods")),
      metric: $("soExpMetric")?.value || "bestValue",
      repeats: $("soExpRepeats")?.value,
      seed: $("soExpSeed")?.value,
      workers: $("soExpWorkers")?.value,
      settings: singleObjectiveExperimentSettings(),
    };
  }
  return {
    page: "single-objective-test",
    benchmark: $("soBenchmarkSelect")?.value || state.singleObjective?.benchmark?.key,
    method: $("soMethodSelect")?.value || state.singleObjective?.method?.key,
    settings: singleObjectiveSettings(),
    evaluations: state.singleObjective?.nEvaluations ?? 0,
    bestValue: state.singleObjective?.bestValue,
  };
}

function renderPlatformTools() {
  const health = $("moduleHealth");
  if (health) {
    const errors = registryErrors();
    health.classList.toggle("ok", !errors.length);
    health.classList.toggle("warn", !!errors.length);
    health.textContent = errors.length
      ? `Registry warnings: ${errors.length}. ${errors.slice(0, 2).join(" ")}`
      : "Registry healthy. Local modules are discoverable.";
  }
  const preview = $("configPreview");
  if (preview) {
    preview.textContent = JSON.stringify(currentPlatformConfig(), null, 2);
  }
  renderRunHistory();
}

function loadRunHistory() {
  if (state.historyLoaded) return;
  state.historyLoaded = true;
  try {
    const parsed = JSON.parse(localStorage.getItem(historyStorageKey) || "[]");
    state.runHistory = Array.isArray(parsed) ? parsed.slice(0, 30) : [];
  } catch {
    state.runHistory = [];
  }
}

function saveRunHistory() {
  try {
    localStorage.setItem(historyStorageKey, JSON.stringify(state.runHistory.slice(0, 30)));
  } catch (error) {
    console.warn("Run history could not be saved", error);
  }
}

function addRunHistory(entry) {
  loadRunHistory();
  state.runHistory.unshift({ id: Date.now(), savedAt: new Date().toISOString(), ...entry });
  state.runHistory = state.runHistory.slice(0, 30);
  saveRunHistory();
  renderRunHistory();
}

function renderRunHistory() {
  loadRunHistory();
  const list = $("runHistoryList");
  if (!list) return;
  if (!state.runHistory.length) {
    list.innerHTML = `<p class="summary benchmark-summary">No saved local runs yet.</p>`;
    return;
  }
  list.innerHTML = state.runHistory
    .slice(0, 8)
    .map((item) => {
      const title = item.title || item.kind || "Run";
      const metric = item.metric ? `, metric ${item.metric}` : "";
      const result = item.result ? `, ${item.result}` : "";
      return `
        <article class="history-item">
          <strong>${escapeHtml(title)}</strong>
          <span>${escapeHtml(new Date(item.savedAt).toLocaleString())}${metric}${result}</span>
        </article>
      `;
    })
    .join("");
}

async function applyPreset(name) {
  const presets = {
    smoke: { populationSize: 12, maxEvaluations: 60, dimension: 3, objectiveCount: 3, repeats: 2, workers: 2 },
    expensive: { populationSize: 8, maxEvaluations: 80, dimension: 5, objectiveCount: 3, repeats: 3, workers: 4 },
    paper: { populationSize: 20, maxEvaluations: 200, dimension: 10, objectiveCount: 5, repeats: 10, workers: 4 },
    large: { populationSize: 30, maxEvaluations: 300, dimension: 30, objectiveCount: 8, repeats: 5, workers: 4 },
  };
  const preset = presets[name] || presets.smoke;
  const assign = (id, value) => {
    const element = $(id);
    if (!element) return;
    element.value = value;
    element.dataset.touched = "1";
  };
  ["populationSize", "expPopulation"].forEach((id) => assign(id, preset.populationSize));
  ["soPopulationSize", "soExpPopulation"].forEach((id) => assign(id, preset.populationSize));
  ["maxEvaluations", "expBudget"].forEach((id) => assign(id, preset.maxEvaluations));
  ["soMaxEvaluations", "soExpBudget"].forEach((id) => assign(id, preset.maxEvaluations));
  ["dimensionInput", "expDimension"].forEach((id) => assign(id, preset.dimension));
  ["soDimensionInput", "soExpDimension"].forEach((id) => assign(id, preset.dimension));
  ["objectiveCountInput", "expObjectives"].forEach((id) => assign(id, preset.objectiveCount));
  assign("expRepeats", preset.repeats);
  assign("soExpRepeats", preset.repeats);
  assign("expWorkers", preset.workers);
  assign("soExpWorkers", preset.workers);
  updateExperimentScale("multi");
  updateExperimentScale("single");
  renderPlatformTools();
  const status = $("registryStatus");
  if (status) status.textContent = `Applied ${name} preset to test and experiment settings.`;
  const tasks = [];
  if (state.data) {
    tasks.push(api("/api/settings", { settings: currentSettings() }).then(render));
  }
  if (state.singleObjective) {
    tasks.push(api("/api/single/settings", { settings: singleObjectiveSettings() }).then(renderSingleObjective));
  }
  if (tasks.length) {
    await Promise.all(tasks);
    if (status) status.textContent = `Applied ${name} preset to backend test settings and experiment forms.`;
  }
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function experimentHistoryResult(result) {
  if (!result?.rows?.length) return "no rows";
  const ok = result.rows.filter((row) => row.status === "ok").length;
  return `${ok}/${result.rows.length} ok, ${result.elapsedMs} ms`;
}

function singleRunHistoryResult(data) {
  if (!data) return "";
  return `best ${fmt(data.bestValue ?? 0, 5)}, ${data.nEvaluations}/${data.settings?.maxEvaluations || 0} evals`;
}

function multiRunHistoryResult(data) {
  if (!data) return "";
  const metricKey = $("metricSelect")?.value || "hypervolume";
  const metric = data.metrics?.[metricKey] ?? data.hypervolume ?? 0;
  return `${metricKey} ${fmt(metric, 4)}, ${data.nEvaluations}/${data.settings?.maxEvaluations || 0} evals`;
}

function renderSingleObjectiveControls(data) {
  const benchmarkSelect = $("soBenchmarkSelect");
  if (benchmarkSelect.dataset.signature !== optionSignature(data.benchmarks)) {
    setOptions(benchmarkSelect, data.benchmarks);
    benchmarkSelect.dataset.signature = optionSignature(data.benchmarks);
  }
  benchmarkSelect.value = data.benchmark.key;
  applySelectFilter("soBenchmarkSelect", "soBenchmarkFilter");
  $("soBenchmarkSummary").textContent = `${data.benchmark.suite || "Classic"}: ${data.benchmark.description}`;

  const methodSelect = $("soMethodSelect");
  if (methodSelect.dataset.signature !== optionSignature(data.methods)) {
    setOptions(methodSelect, data.methods);
    methodSelect.dataset.signature = optionSignature(data.methods);
  }
  methodSelect.value = data.method.key;
  applySelectFilter("soMethodSelect", "soMethodFilter");
  $("soMethodSummary").textContent = data.method.description;

  $("soPopulationSize").value = data.settings.populationSize;
  $("soMaxEvaluations").value = data.settings.maxEvaluations;
  $("soDimensionInput").min = data.benchmark.minDimension || 1;
  $("soDimensionInput").max = data.benchmark.maxDimension || 200;
  $("soDimensionInput").value = data.settings.dimension;
  $("soSettingsSummary").textContent = `Python backend: d=${data.dimensions.inputs}, population=${data.settings.populationSize}, budget=${data.nEvaluations}/${data.settings.maxEvaluations}, remaining=${data.budgetRemaining}.`;
  $("soDefinition").innerHTML = `
    <div class="definition-row">
      <strong>Decision variables (d=${data.dimensions.inputs})</strong>
      <div class="chips">${Array.from({ length: Math.min(data.dimensions.inputs, 12) }, (_, index) => `<span class="chip">x${index + 1}</span>`).join("")}${data.dimensions.inputs > 12 ? `<span class="chip">...</span>` : ""}</div>
    </div>
    <div class="definition-row">
      <strong>Objective</strong>
      <div class="chips"><span class="chip">minimize f(x)</span><span class="chip">${escapeHtml(data.benchmark.suite || "Classic")}</span><span class="chip">known optimum ${fmt(data.benchmark.optimum ?? 0, 3)}</span></div>
    </div>
  `;
}

function renderSingleObjectiveExperimentControls(data) {
  const benchmarkSelect = $("soExpBenchmarks");
  const methodSelect = $("soExpMethods");
  if (!benchmarkSelect || !methodSelect) return;
  const selectedBenchmarks = selectedValues(benchmarkSelect);
  const selectedMethods = selectedValues(methodSelect);
  const shouldRefreshBenchmarks = benchmarkSelect.dataset.signature !== optionSignature(data.benchmarks);
  const shouldRefreshMethods = methodSelect.dataset.signature !== optionSignature(data.methods);

  if (shouldRefreshBenchmarks) {
    const fallback = selectedBenchmarks.length ? selectedBenchmarks : [data.benchmark.key];
    setOptions(benchmarkSelect, data.benchmarks);
    benchmarkSelect.dataset.signature = optionSignature(data.benchmarks);
    setSelectedValues(benchmarkSelect, fallback);
  }
  if (shouldRefreshMethods) {
    const fallback = selectedMethods.length
      ? selectedMethods
      : preferredKeys(data.methods, defaultSoExperimentMethods, data.method.key);
    setOptions(methodSelect, data.methods);
    methodSelect.dataset.signature = optionSignature(data.methods);
    setSelectedValues(methodSelect, fallback);
  }
  applySelectFilter("soExpBenchmarks", "soExpBenchmarkFilter");
  applySelectFilter("soExpMethods", "soExpMethodFilter");

  const referenceSelect = $("soExpReference");
  if (referenceSelect.dataset.signature !== optionSignature(data.methods)) {
    const currentReference = referenceSelect.value || (data.methods.some((item) => item.key === "rbf_bo") ? "rbf_bo" : data.method.key);
    setOptions(referenceSelect, data.methods);
    referenceSelect.dataset.signature = optionSignature(data.methods);
    referenceSelect.value = currentReference;
  }

  if (!$("soExpPopulation").dataset.touched) $("soExpPopulation").value = data.settings.populationSize;
  if (!$("soExpBudget").dataset.touched) $("soExpBudget").value = data.settings.maxEvaluations;
  if (!$("soExpDimension").dataset.touched) $("soExpDimension").value = data.settings.dimension;
  updateExperimentScale("single");
}

function singleObjectiveSettings() {
  return {
    populationSize: $("soPopulationSize").value,
    maxEvaluations: $("soMaxEvaluations").value,
    dimension: $("soDimensionInput").value,
  };
}

function valueColor(value, minValue, maxValue) {
  const t = maxValue > minValue ? clamp((value - minValue) / (maxValue - minValue)) : 0;
  return `hsl(${144 - t * 138}, 68%, ${40 + t * 8}%)`;
}

function singleObjectiveHistoryEnabled() {
  return $("soHistoryToggle")?.checked !== false;
}

function singleObjectiveSearchData(data, options = {}) {
  const maxEvaluation = Math.max(1, Math.min(Number(options.maxEvaluation ?? data.nEvaluations ?? data.points.length), data.points.length));
  const populationSize = Math.max(1, Number(data.settings?.populationSize || 1));
  const showHistory = options.history ?? singleObjectiveHistoryEnabled();
  const recentStart = Math.max(0, maxEvaluation - populationSize);
  let points = data.points.filter((point) => Number(point.id ?? 0) < maxEvaluation);
  if (!showHistory) {
    points = points.filter((point) => Number(point.id ?? 0) >= recentStart);
  }
  const bestPoint = points.reduce((best, point) => (!best || Number(point.value) < Number(best.value) ? point : best), null);
  points = points.map((point) => ({
    ...point,
    best: bestPoint ? Number(point.id) === Number(bestPoint.id) : false,
    recent: Number(point.id ?? 0) >= recentStart,
  }));
  return { ...data, nEvaluations: maxEvaluation, points };
}

function renderSingleObjectiveSearch(data, options = {}) {
  const select = $("soSearchView");
  const dimension = Number(data.dimensions?.inputs || 1);
  if (select) {
    select.disabled = dimension < 2;
    if (dimension < 2 && select.value === "3d") select.value = "2d";
  }
  const mode = dimension >= 2 && select?.value === "3d" ? "3d" : "2d";
  if (!options.silent) state.soView.mode = mode;
  if (!options.silent) ensureSingleObjectiveLandscape(data);
  const displayData = singleObjectiveSearchData(data, options);
  if (dimension < 2) {
    renderSingleObjective1dSearch(displayData, options);
  } else if (mode === "3d") {
    renderSingleObjective3dSearch(displayData, options);
  } else {
    renderSingleObjective2dSearch(displayData, options);
  }
}

function singleObjectiveLandscapeEnabled() {
  return !!$("soLandscapeToggle")?.checked;
}

function singleObjectiveLandscapeBaseKey(data) {
  return `${data.benchmark?.key || ""}|${data.dimensions?.inputs || 0}`;
}

function singleObjectiveLandscapeSliceKey(data) {
  const best = (data.bestX || []).slice(0, Math.min(6, data.dimensions?.inputs || 0)).map((value) => fmt(value, 4)).join(",");
  return `${singleObjectiveLandscapeBaseKey(data)}|${best}`;
}

function currentSingleObjectiveLandscape(data) {
  if (!singleObjectiveLandscapeEnabled()) return null;
  const landscape = state.singleObjectiveLandscape;
  return landscape?.baseKey === singleObjectiveLandscapeBaseKey(data) ? landscape : null;
}

function ensureSingleObjectiveLandscape(data) {
  if (!singleObjectiveLandscapeEnabled() || !data) return;
  const baseKey = singleObjectiveLandscapeBaseKey(data);
  const sliceKey = singleObjectiveLandscapeSliceKey(data);
  if (state.singleObjectiveLandscape?.sliceKey === sliceKey || state.singleObjectiveLandscapeLoadingKey) return;
  state.singleObjectiveLandscapeLoadingKey = sliceKey;
  api("/api/single/landscape?resolution=96")
    .then((landscape) => {
      state.singleObjectiveLandscapeLoadingKey = null;
      if (!state.singleObjective) return;
      if (singleObjectiveLandscapeBaseKey(state.singleObjective) !== baseKey) {
        ensureSingleObjectiveLandscape(state.singleObjective);
        return;
      }
      state.singleObjectiveLandscape = { ...landscape, baseKey, sliceKey };
      renderSingleObjectiveSearch(state.singleObjective);
    })
    .catch((error) => {
      state.singleObjectiveLandscapeLoadingKey = null;
      $("soSearchSubtitle").textContent =
        error.message.includes("404") || error.message === "Not Found"
          ? "Landscape endpoint is unavailable. Restart the Python backend, then refresh."
          : `Landscape unavailable: ${error.message}`;
    });
}

function objectiveRange(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  const minValue = finite.length ? Math.min(...finite) : 0;
  const maxValue = finite.length ? Math.max(...finite) : 1;
  const pad = Math.max(1e-9, (maxValue - minValue) * 0.08);
  return maxValue > minValue ? { minValue: minValue - pad, maxValue: maxValue + pad } : { minValue: minValue - 1, maxValue: maxValue + 1 };
}

function landscapeRange(landscape, values) {
  const minValue = Number(landscape?.minValue);
  const maxValue = Number(landscape?.maxValue);
  if (landscape && Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {
    return { minValue, maxValue };
  }
  const fallback = objectiveRange(values);
  return fallback;
}

function landscapeHeatmapImage(landscape, minValue, maxValue, x, y, width, height, inline = false) {
  if (!landscape || landscape.kind !== "2d") return "";
  const nx = landscape.x.length;
  const ny = landscape.y.length;
  if (inline) {
    const cellW = width / Math.max(1, nx);
    const cellH = height / Math.max(1, ny);
    return `<g opacity="0.42">${landscape.values
      .map((row, rowIndex) =>
        row
          .map((value, colIndex) => `<rect x="${x + colIndex * cellW}" y="${y + (ny - 1 - rowIndex) * cellH}" width="${cellW + 0.6}" height="${cellH + 0.6}" fill="${valueColor(value, minValue, maxValue)}"></rect>`)
          .join(""),
      )
      .join("")}</g>`;
  }
  const rangeKey = `${fmt(minValue, 8)}|${fmt(maxValue, 8)}`;
  if (!landscape.heatmapHref || landscape.heatmapRangeKey !== rangeKey) {
    const cells = landscape.values
      .map((row, rowIndex) =>
        row
          .map((value, colIndex) => `<rect x="${colIndex}" y="${ny - 1 - rowIndex}" width="1.08" height="1.08" fill="${valueColor(value, minValue, maxValue)}"></rect>`)
          .join(""),
      )
      .join("");
    const heatmapSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="${nx}" height="${ny}" viewBox="0 0 ${nx} ${ny}" preserveAspectRatio="none"><g opacity="0.42">${cells}</g></svg>`;
    landscape.heatmapHref = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(heatmapSvg)}`;
    landscape.heatmapRangeKey = rangeKey;
  }
  return `<image x="${x}" y="${y}" width="${width}" height="${height}" preserveAspectRatio="none" href="${landscape.heatmapHref}"></image>`;
}

function renderSingleObjective1dSearch(data, options = {}) {
  const svg = options.svg || $("soSearch");
  const width = 760;
  const height = 470;
  const margin = { left: 70, right: 34, top: 24, bottom: 58 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const landscape = currentSingleObjectiveLandscape(data);
  const values = data.points.map((p) => Number(p.value)).concat(landscape?.kind === "1d" ? landscape.values.map(Number) : []);
  const { minValue, maxValue } = objectiveRange(values);
  const xBounds = singleObjectiveInputBounds(data)[0] || [0, 1];
  const xLo = Number(xBounds[0]);
  const xHi = Number(xBounds[1]);
  const xScale = (v) => margin.left + ((Number(v) - xLo) / Math.max(1e-12, xHi - xLo)) * innerW;
  const yScale = (v) => margin.top + (1 - (Number(v) - minValue) / Math.max(1e-12, maxValue - minValue)) * innerH;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const valueTicks = ticks.map((t) => minValue + (maxValue - minValue) * t);
  const grid = [
    ...ticks.map(
      (t) => `
      <line class="grid" x1="${xScale(xLo + (xHi - xLo) * t)}" x2="${xScale(xLo + (xHi - xLo) * t)}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <text class="tick-label" x="${xScale(xLo + (xHi - xLo) * t)}" y="${height - 32}" text-anchor="middle">${realInputLabel(data, 0, t)}</text>
    `,
    ),
    ...valueTicks.map(
      (value) => `
      <line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(value)}" y2="${yScale(value)}"></line>
      <text class="tick-label" x="58" y="${yScale(value) + 4}" text-anchor="end">${fmt(value, 2)}</text>
    `,
    ),
  ].join("");
  const guideLines = [0.2, 0.4, 0.6, 0.8]
    .map((t) => `<line class="guide-line" x1="${xScale(xLo + (xHi - xLo) * t)}" x2="${xScale(xLo + (xHi - xLo) * t)}" y1="${margin.top}" y2="${height - margin.bottom}"></line>`)
    .join("");
  const sorted = [...data.points].sort((a, b) => Number(a.x?.[0] ?? 0) - Number(b.x?.[0] ?? 0));
  const sampledLine =
    sorted.length > 1
      ? `<path class="search-line" d="${sorted.map((p, i) => `${i ? "L" : "M"} ${xScale(realInputValue(data, 0, p.x?.[0] ?? 0))} ${yScale(p.value)}`).join(" ")}"></path>`
      : "";
  const landscapeLine =
    landscape?.kind === "1d"
      ? `<path class="landscape-curve" d="${landscape.x.map((x, i) => `${i ? "L" : "M"} ${xScale(xLo + (xHi - xLo) * x)} ${yScale(landscape.values[i])}`).join(" ")}"></path>`
      : "";
  const points = data.points
    .map((p) => {
      const radius = p.best ? 8 : p.recent ? 5.8 : 4.1;
      const klass = `point${p.best ? " recommended" : ""}${p.recent ? " recent" : ""}`;
      const realX = realInputValue(data, 0, p.x?.[0] ?? 0);
      return `<circle class="${klass}" cx="${xScale(realX)}" cy="${yScale(p.value)}" r="${radius}" fill="${valueColor(p.value, minValue, maxValue)}"><title>#${p.id} x=${realInputLabel(data, 0, p.x?.[0] ?? 0)}, f=${fmt(p.value, 6)}</title></circle>`;
    })
    .join("");
  if (!options.silent) $("soSearchSubtitle").textContent = landscape?.kind === "1d" ? "1D landscape: function curve with evaluated samples" : "1D view: x vs objective value; red outline is current best";
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${grid}
    ${guideLines}
    <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
    <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
    ${landscapeLine}
    ${sampledLine}
    ${points}
    <text class="axis-label" x="${margin.left + innerW / 2}" y="${height - 13}" text-anchor="middle">${escapeHtml(data.inputs?.[0]?.label || "x")}</text>
    <text class="axis-label" x="18" y="${margin.top + innerH / 2}" transform="rotate(-90 18 ${margin.top + innerH / 2})" text-anchor="middle">Objective value f(x)</text>
    <text class="legend" x="${width - 448}" y="28">${landscape?.kind === "1d" ? "Blue: function landscape, thin line: sampled order" : "Line: sampled order by x; color: objective value"}</text>
  `;
}

function renderSingleObjective2dSearch(data, options = {}) {
  const svg = options.svg || $("soSearch");
  const width = 760;
  const height = 470;
  const margin = { left: 58, right: 32, top: 24, bottom: 54 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const xBounds = singleObjectiveInputBounds(data)[0] || singleObjectiveBounds(data);
  const yBounds = singleObjectiveInputBounds(data)[1] || xBounds;
  const xLo = Number(xBounds[0]);
  const xHi = Number(xBounds[1]);
  const yLo = Number(yBounds[0]);
  const yHi = Number(yBounds[1]);
  const xScale = (v) => margin.left + ((Number(v) - xLo) / Math.max(1e-12, xHi - xLo)) * innerW;
  const yScale = (v) => margin.top + (1 - ((Number(v) - yLo) / Math.max(1e-12, yHi - yLo))) * innerH;
  const landscape = currentSingleObjectiveLandscape(data);
  const pointValues = data.points.map((p) => Number(p.value));
  const { minValue, maxValue } = landscape?.kind === "2d" ? landscapeRange(landscape, pointValues) : objectiveRange(pointValues);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const grid = ticks
    .map(
      (t) => `
      <line class="grid" x1="${xScale(xLo + (xHi - xLo) * t)}" x2="${xScale(xLo + (xHi - xLo) * t)}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(yLo + (yHi - yLo) * t)}" y2="${yScale(yLo + (yHi - yLo) * t)}"></line>
      <text class="tick-label" x="${xScale(xLo + (xHi - xLo) * t)}" y="${height - 30}" text-anchor="middle">${realInputLabel(data, 0, t)}</text>
      <text class="tick-label" x="44" y="${yScale(yLo + (yHi - yLo) * t) + 4}" text-anchor="end">${realInputLabel(data, 1, t)}</text>
    `,
    )
    .join("");
  const guideLines = [0.2, 0.4, 0.6, 0.8]
    .map(
      (t) => `
      <line class="guide-line" x1="${xScale(xLo + (xHi - xLo) * t)}" x2="${xScale(xLo + (xHi - xLo) * t)}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <line class="guide-line" x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(yLo + (yHi - yLo) * t)}" y2="${yScale(yLo + (yHi - yLo) * t)}"></line>
    `,
    )
    .join("");
  const heatmap =
    landscape?.kind === "2d"
      ? landscapeHeatmapImage(landscape, minValue, maxValue, margin.left, margin.top, innerW, innerH, options.inlineLandscape)
      : "";
  const points = data.points
    .map((p) => {
      const x = realInputValue(data, 0, p.x?.[0] ?? 0);
      const y = realInputValue(data, 1, p.x?.[1] ?? 0.5);
      const radius = p.best ? 8 : p.recent ? 5.8 : 4.1;
      const klass = `point${p.best ? " recommended" : ""}${p.recent ? " recent" : ""}`;
      return `<circle class="${klass}" cx="${xScale(x)}" cy="${yScale(y)}" r="${radius}" fill="${valueColor(p.value, minValue, maxValue)}"><title>#${p.id} ${escapeHtml(data.inputs?.[0]?.label || "x1")}=${realInputLabel(data, 0, p.x?.[0] ?? 0)}, ${escapeHtml(data.inputs?.[1]?.label || "x2")}=${realInputLabel(data, 1, p.x?.[1] ?? 0.5)}, f=${fmt(p.value, 5)}</title></circle>`;
    })
    .join("");
  if (!options.silent) $("soSearchSubtitle").textContent = landscape?.kind === "2d" ? `2D landscape slice: ${landscape.sliceNote || "x1-x2 plane"}` : data.dimensions.inputs > 2 ? "2D projection: x1 vs x2; color shows objective value" : "2D view: x1 vs x2; color shows objective value";
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${heatmap}
    ${grid}
    ${guideLines}
    <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
    <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
    <text class="axis-label" x="${margin.left + innerW / 2}" y="${height - 12}" text-anchor="middle">${escapeHtml(data.inputs?.[0]?.label || "x1")}</text>
    <text class="axis-label" x="18" y="${margin.top + innerH / 2}" transform="rotate(-90 18 ${margin.top + innerH / 2})" text-anchor="middle">${escapeHtml(data.inputs?.[1]?.label || "x2")}</text>
    ${points}
    <text class="legend" x="${width - 430}" y="26">${landscape?.kind === "2d" ? "Background: function landscape; points: evaluated samples" : "Color: objective value; larger: latest population"}</text>
  `;
}

function renderSingleObjective3dTicks(data, rawMin, rawMax) {
  const ticks = [0, 0.5, 1];
  const valueSpan = Math.max(1e-12, rawMax - rawMin);
  const tickLine = (from, to) => {
    const a = projectSingleObjective3d(from);
    const b = projectSingleObjective3d(to);
    return `<line class="axis-3d soft" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"></line>`;
  };
  const text = (point, label, dx = 0, dy = 0) => {
    const projected = projectSingleObjective3d(point);
    return `<text class="tick-label" x="${projected.x + dx}" y="${projected.y + dy}" text-anchor="middle">${escapeHtml(label)}</text>`;
  };
  const x1Ticks = ticks
    .map((t) => `${tickLine([t, 0, 0], [t, 0, 0.035])}${text([t, 0, 0], realInputLabel(data, 0, t), 0, 20)}`)
    .join("");
  const x2Ticks = ticks
    .map((t) => `${tickLine([0, 0, t], [0.035, 0, t])}${text([0, 0, t], realInputLabel(data, 1, t), -12, 15)}`)
    .join("");
  const fTicks = ticks
    .map((t) => {
      const value = rawMin + valueSpan * t;
      return `${tickLine([0, t, 0], [0.035, t, 0])}${text([0, t, 0], fmt(value, 2), -34, 4)}`;
    })
    .join("");
  return `${x1Ticks}${x2Ticks}${fTicks}`;
}

function renderSingleObjective3dSearch(data, options = {}) {
  const svg = options.svg || $("soSearch");
  const width = 760;
  const height = 470;
  const landscape = currentSingleObjectiveLandscape(data);
  const landscapeValues = landscape?.kind === "2d" ? landscape.values.flat().map(Number) : [];
  const values = data.points.map((p) => Number(p.value)).concat(landscapeValues);
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const valueSpan = Math.max(1e-12, rawMax - rawMin);
  const normalizeValue = (value) => clamp((Number(value) - rawMin) / valueSpan);
  const inputBounds = singleObjectiveInputBounds(data);
  const x1Bounds = inputBounds[0] || [0, 1];
  const x2Bounds = inputBounds[1] || x1Bounds;
  const x1Lo = Number(x1Bounds[0]);
  const x1Hi = Number(x1Bounds[1]);
  const x2Lo = Number(x2Bounds[0]);
  const x2Hi = Number(x2Bounds[1]);
  const normalizeX1 = (value) => (Number(value) - x1Lo) / Math.max(1e-12, x1Hi - x1Lo);
  const normalizeX2 = (value) => (Number(value) - x2Lo) / Math.max(1e-12, x2Hi - x2Lo);
  const origin = projectSingleObjective3d([0, 0, 0]);
  const axisEnds = [
    { end: projectSingleObjective3d([1, 0, 0]), label: data.inputs?.[0]?.label || "x1" },
    { end: projectSingleObjective3d([0, 1, 0]), label: "f(x)" },
    { end: projectSingleObjective3d([0, 0, 1]), label: data.inputs?.[1]?.label || "x2" },
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
  ].map(projectSingleObjective3d);
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
  const guideLines = [0.25, 0.5, 0.75]
    .flatMap((t) => [
      [[t, 0, 0], [t, 1, 0]],
      [[0, t, 0], [1, t, 0]],
      [[0, 0, t], [1, 0, t]],
      [[0, t, 0], [0, t, 1]],
      [[0, 0, t], [0, 1, t]],
      [[t, 0, 0], [t, 0, 1]],
    ])
    .map(([a, b]) => {
      const pa = projectSingleObjective3d(a);
      const pb = projectSingleObjective3d(b);
      return `<line class="guide-line soft-3d" x1="${pa.x}" y1="${pa.y}" x2="${pb.x}" y2="${pb.y}"></line>`;
    })
    .join("");
  const axes = axisEnds
    .map(
      ({ end, label }) => `
      <line class="axis-3d" x1="${origin.x}" y1="${origin.y}" x2="${end.x}" y2="${end.y}"></line>
      <text class="axis-label" x="${end.x}" y="${end.y - 8}" text-anchor="middle">${escapeHtml(label)}</text>
    `,
    )
    .join("");
  const ticks = renderSingleObjective3dTicks(data, rawMin, rawMax);
  const terrain =
    landscape?.kind === "2d"
      ? renderSingleObjectiveTerrainMesh(landscape, normalizeValue)
      : "";
  const points = data.points
    .map((p) => {
      const projected = projectSingleObjective3d([normalizeX1(p.x?.[0] ?? 0), normalizeValue(p.value), normalizeX2(p.x?.[1] ?? 0)]);
      return { ...p, projected };
    })
    .sort((a, b) => a.projected.depth - b.projected.depth)
    .map((p) => {
      const radius = p.best ? 8.2 : p.recent ? 6.1 : 4.8;
      const klass = `point${p.best ? " recommended" : ""}${p.recent ? " recent" : ""}`;
      return `<circle class="${klass}" cx="${p.projected.x}" cy="${p.projected.y}" r="${radius}" fill="${valueColor(p.value, rawMin, rawMax)}"><title>#${p.id} ${escapeHtml(data.inputs?.[0]?.label || "x1")}=${realInputLabel(data, 0, p.x?.[0] ?? 0)}, ${escapeHtml(data.inputs?.[1]?.label || "x2")}=${realInputLabel(data, 1, p.x?.[1] ?? 0)}, f=${fmt(p.value, 6)}</title></circle>`;
    })
    .join("");
  if (!options.silent) $("soSearchSubtitle").textContent = landscape?.kind === "2d" ? "3D landscape: x1, x2, vertical f(x); drag to rotate" : "3D view: x1, x2, vertical f(x); drag to rotate";
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${edges}
    ${guideLines}
    ${axes}
    ${ticks}
    ${terrain}
    ${points}
    <text class="legend" x="${width - 486}" y="28">${landscape?.kind === "2d" ? "Mesh: function landscape; points: evaluated samples" : "Drag to rotate"}; f(x) range [${fmt(rawMin, 3)}, ${fmt(rawMax, 3)}]</text>
  `;
  if (!options.silent) bindSingleObjective3dInteractions(svg);
}

function renderSingleObjectiveTerrainMesh(landscape, normalizeValue) {
  const nx = landscape.x.length;
  const ny = landscape.y.length;
  const stride = Math.max(1, Math.floor(Math.max(nx, ny) / 18));
  const rowLines = [];
  for (let row = 0; row < ny; row += stride) {
    rowLines.push(
      `<path class="landscape-mesh" d="${landscape.x
        .map((x, col) => {
          const p = projectSingleObjective3d([x, normalizeValue(landscape.values[row][col]), landscape.y[row]]);
          return `${col ? "L" : "M"} ${p.x} ${p.y}`;
        })
        .join(" ")}"></path>`,
    );
  }
  const colLines = [];
  for (let col = 0; col < nx; col += stride) {
    colLines.push(
      `<path class="landscape-mesh" d="${landscape.y
        .map((y, row) => {
          const p = projectSingleObjective3d([landscape.x[col], normalizeValue(landscape.values[row][col]), y]);
          return `${row ? "L" : "M"} ${p.x} ${p.y}`;
        })
        .join(" ")}"></path>`,
    );
  }
  return `${rowLines.join("")}${colLines.join("")}`;
}

function projectSingleObjective3d(values) {
  const centered = values.slice(0, 3).map((value) => Number(value ?? 0) - 0.5);
  const angleY = state.soView.yaw;
  const angleX = state.soView.pitch;
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

function bindSingleObjective3dInteractions(svg) {
  if (!svg || svg.dataset.boundSo3d === "1") return;
  svg.dataset.boundSo3d = "1";
  svg.addEventListener("pointerdown", (event) => {
    if (state.soView.mode !== "3d") return;
    state.soView.drag = { x: event.clientX, y: event.clientY, yaw: state.soView.yaw, pitch: state.soView.pitch };
    svg.setPointerCapture?.(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!state.soView.drag || !state.singleObjective) return;
    const dx = event.clientX - state.soView.drag.x;
    const dy = event.clientY - state.soView.drag.y;
    state.soView.yaw = state.soView.drag.yaw + dx * 0.012;
    state.soView.pitch = clamp(state.soView.drag.pitch + dy * 0.012, -1.1, 1.1);
    renderSingleObjectiveSearch(state.singleObjective);
  });
  const endDrag = () => {
    state.soView.drag = null;
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("pointerleave", endDrag);
}

function renderSingleObjectiveConvergence(data, options = {}) {
  const svg = options.svg || $("soConvergence");
  const width = 760;
  const height = 470;
  const margin = { left: 64, right: 34, top: 24, bottom: 54 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const maxEvaluation = Math.max(1, Number(options.maxEvaluation ?? data.nEvaluations ?? data.convergence?.length ?? 1));
  const rows = (data.convergence || []).filter((row) => Number(row.evaluation) <= maxEvaluation);
  if (!rows.length) {
    svg.innerHTML = `<text class="axis-label" x="40" y="40">No data</text>`;
    return;
  }
  const maxEval = Math.max(1, rows[rows.length - 1].evaluation);
  const values = rows.flatMap((row) => [row.value, row.best]);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const pad = Math.max(1e-9, (maxValue - minValue) * 0.08);
  const yMin = minValue - pad;
  const yMax = maxValue + pad;
  const xScale = (v) => margin.left + ((Number(v) - 1) / Math.max(1, maxEval - 1)) * innerW;
  const yScale = (v) => margin.top + (1 - (Number(v) - yMin) / Math.max(1e-12, yMax - yMin)) * innerH;
  const gridTicks = [0, 0.25, 0.5, 0.75, 1];
  const grid = gridTicks
    .map((t) => {
      const yValue = yMin + (yMax - yMin) * t;
      return `
        <line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(yValue)}" y2="${yScale(yValue)}"></line>
        <text class="tick-label" x="54" y="${yScale(yValue) + 4}" text-anchor="end">${fmt(yValue, 2)}</text>
      `;
    })
    .join("");
  const rawPoints = rows.slice(-220).map((row) => `<circle class="point faint" cx="${xScale(row.evaluation)}" cy="${yScale(row.value)}" r="2.4"></circle>`).join("");
  const bestLine = `<path class="convergence-line" d="${rows.map((row, i) => `${i ? "L" : "M"} ${xScale(row.evaluation)} ${yScale(row.best)}`).join(" ")}"></path>`;
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${grid}
    <line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
    <line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
    ${rawPoints}
    ${bestLine}
    <text class="axis-label" x="${margin.left + innerW / 2}" y="${height - 12}" text-anchor="middle">Evaluations</text>
    <text class="axis-label" x="18" y="${margin.top + innerH / 2}" transform="rotate(-90 18 ${margin.top + innerH / 2})" text-anchor="middle">Objective value</text>
    <text class="legend" x="${width - 285}" y="26">Dots: observed, line: best-so-far</text>
  `;
}

function renderSingleObjectiveBest(data) {
  $("soBestSolution").innerHTML = `
    <div class="so-best-meta">
      <div>
        <span>Evaluations</span>
        <strong>${data.nEvaluations}/${data.settings.maxEvaluations}</strong>
      </div>
      <div>
        <span>Current strategy</span>
        <strong>${escapeHtml(data.lastStrategy)}</strong>
      </div>
    </div>
    <article class="solution recommended">
      <div class="solution-section-title">Best solution</div>
      <div class="solution-head">
        <strong>Best value ${fmt(data.bestValue, 6)}</strong>
        <span class="badge hot">Minimum</span>
      </div>
      <div class="triple">${data.bestX.slice(0, 12).map((value, index) => realDecisionMini(data, `x${index + 1}`, value)).join("")}</div>
    </article>
  `;
}

function renderSingleObjectiveCandidates(data) {
  if (!data.lastCandidates?.length) {
    $("soCandidates").innerHTML = `<p class="summary">Run a round to evaluate candidate points.</p>`;
    return;
  }
  $("soCandidates").innerHTML = data.lastCandidates
    .map(
      (item, index) => `
      <article class="solution${item.improved ? " recommended" : ""}">
        <div class="solution-head">
          <strong>Candidate ${index + 1}: f=${fmt(item.value, 5)}</strong>
          <span class="badge${item.improved ? " hot" : ""}">${item.improved ? "Improved" : "Sampled"}</span>
        </div>
        <div class="triple">${item.x.slice(0, 6).map((value, dim) => realDecisionMini(data, `x${dim + 1}`, value)).join("")}</div>
      </article>
    `,
    )
    .join("");
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
  $("weights").querySelectorAll("[data-weight-key]").forEach((input) => {
    input.addEventListener("input", () => updateManualWeight(input.dataset.weightKey, input.value));
  });
  $("preferenceSummary").textContent = data.manualWeights ? "Manual preference weights are active." : data.parsed.summary;
  $("notes").innerHTML = data.parsed.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
}

function normalizedManualWeights(key, rawValue) {
  const current = { ...(state.data?.activeWeights || {}) };
  current[key] = Math.max(0.01, Number(rawValue) / 100);
  const relevant = [...new Set((state.data?.objectives || []).map((obj) => obj.preferenceKey || "quality"))];
  const total = relevant.reduce((sum, entryKey) => sum + (current[entryKey] ?? 0), 0) || 1;
  return Object.fromEntries(Object.entries(current).map(([entryKey, value]) => [entryKey, relevant.includes(entryKey) ? value / total : 0]));
}

function updateManualWeight(key, rawValue) {
  const weights = normalizedManualWeights(key, rawValue);
  withBusy(async () => {
    render(await api("/api/weights", { weights }));
  });
}

function currentSettings() {
  return {
    populationSize: $("populationSize").value,
    maxEvaluations: $("maxEvaluations").value,
    dimension: $("dimensionInput").value,
    objectiveCount: $("objectiveCountInput").value,
  };
}

function experimentSettings() {
  return {
    populationSize: $("expPopulation").value,
    maxEvaluations: $("expBudget").value,
    dimension: $("expDimension").value,
    objectiveCount: $("expObjectives").value,
  };
}

function singleObjectiveExperimentSettings() {
  return {
    populationSize: $("soExpPopulation").value,
    maxEvaluations: $("soExpBudget").value,
    dimension: $("soExpDimension").value,
  };
}

function selectedValues(select) {
  return Array.from(select?.selectedOptions || []).map((option) => option.value);
}

function setSelectedValues(select, values) {
  const allowed = new Set(values);
  Array.from(select.options).forEach((option) => {
    option.selected = allowed.has(option.value);
  });
}

function preferredKeys(items, preferred, fallbackKey = "") {
  const available = new Set((items || []).map((item) => item.key));
  const keys = preferred.filter((key) => available.has(key));
  if (keys.length) return keys;
  return fallbackKey && available.has(fallbackKey) ? [fallbackKey] : (items?.[0]?.key ? [items[0].key] : []);
}

function visibleOptionValues(select) {
  return Array.from(select?.options || [])
    .filter((option) => !option.hidden && !option.disabled)
    .map((option) => option.value);
}

function experimentRunLimit() {
  return Number(state.data?.experimentLimits?.maxRuns || state.singleObjective?.experimentLimits?.maxRuns || 120);
}

function experimentScale(kind) {
  const isSingle = kind === "single";
  const benchmarkSelect = $(isSingle ? "soExpBenchmarks" : "expBenchmarks");
  const methodSelect = $(isSingle ? "soExpMethods" : "expOptimizers");
  const repeatsInput = $(isSingle ? "soExpRepeats" : "expRepeats");
  const benchmarks = selectedValues(benchmarkSelect).length;
  const methods = selectedValues(methodSelect).length;
  const repeats = Math.max(1, Number(repeatsInput?.value || 1));
  const runs = benchmarks * methods * repeats;
  return { benchmarks, methods, repeats, runs, limit: experimentRunLimit() };
}

function updateExperimentScale(kind = "multi") {
  const isSingle = kind === "single";
  const target = $(isSingle ? "soExperimentScale" : "experimentScale");
  const runButton = $(isSingle ? "soRunExperiment" : "runExperiment");
  if (!target) return;
  const scale = experimentScale(kind);
  const hasSelection = scale.benchmarks > 0 && scale.methods > 0;
  const overLimit = scale.runs > scale.limit;
  target.classList.toggle("error", overLimit);
  target.classList.toggle("warn", hasSelection && !overLimit && scale.runs > scale.limit * 0.7);
  target.textContent = hasSelection
    ? `${scale.benchmarks} benchmark(s) x ${scale.methods} method(s) x ${scale.repeats} repeat(s) = ${scale.runs} run(s). Limit ${scale.limit}.`
    : "Select at least one benchmark and one method.";
  if (runButton && !state.busy) runButton.disabled = !hasSelection || overLimit;
}

function syncExperimentReference(selectId, referenceId) {
  const selected = selectedValues($(selectId));
  const reference = $(referenceId);
  if (!reference || !selected.length) return;
  if (!selected.includes(reference.value)) reference.value = selected[0];
}

function applySelectionAction(button) {
  const select = $(button.dataset.selectTarget || "");
  if (!select) return;
  const action = button.dataset.selectAction || "visible";
  let values = [];
  if (action === "clear") {
    values = [];
  } else if (action === "visible") {
    values = visibleOptionValues(select);
  } else if (action === "current") {
    values = select.id.startsWith("so") ? [state.singleObjective?.benchmark?.key].filter(Boolean) : [state.data?.benchmark?.key].filter(Boolean);
  } else if (select.id === "expOptimizers") {
    values = preferredKeys(state.data?.optimizers || [], action === "expensive" ? expensiveMoExperimentMethods : defaultMoExperimentMethods, state.data?.optimizer?.key);
  } else if (select.id === "soExpMethods") {
    values = preferredKeys(state.singleObjective?.methods || [], action === "expensive" ? expensiveSoExperimentMethods : defaultSoExperimentMethods, state.singleObjective?.method?.key);
  }
  setSelectedValues(select, values);
  if (select.id === "expOptimizers") syncExperimentReference("expOptimizers", "expReference");
  if (select.id === "soExpMethods") syncExperimentReference("soExpMethods", "soExpReference");
  updateExperimentScale(select.id.startsWith("so") ? "single" : "multi");
  renderPlatformTools();
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
  state.view.mode = mode;
  if (mode === "3d") renderScatter3d(data);
  else renderScatter2d(data);
}

function objectiveAxisRange(data, index) {
  const values = (data.points || [])
    .map((point) => Number(point.y?.[index]))
    .filter((value) => Number.isFinite(value));
  const configured = data.objectives?.[index]?.valueRange || [0, 1];
  const configuredMin = Number(configured?.[0]);
  const configuredMax = Number(configured?.[1]);
  const hasConfiguredRange = Number.isFinite(configuredMin) && Number.isFinite(configuredMax) && configuredMax > configuredMin;
  let minValue = values.length ? Math.min(...values) : hasConfiguredRange ? configuredMin : 0;
  let maxValue = values.length ? Math.max(...values) : hasConfiguredRange ? configuredMax : 1;
  const span = maxValue - minValue;
  const pad = span > 1e-9 ? span * 0.08 : Math.max(0.04, Math.abs(maxValue) * 0.05);
  minValue -= pad;
  maxValue += pad;
  if (hasConfiguredRange) {
    minValue = Math.max(configuredMin, minValue);
    maxValue = Math.min(configuredMax, maxValue);
    if (maxValue - minValue < 1e-9) {
      minValue = configuredMin;
      maxValue = configuredMax;
    }
  }
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || maxValue <= minValue) {
    return { min: 0, max: 1 };
  }
  return { min: minValue, max: maxValue };
}

function axisTicks(range, count = 5) {
  const steps = Math.max(2, count);
  return Array.from({ length: steps }, (_, index) => range.min + ((range.max - range.min) * index) / (steps - 1));
}

function normalizeAxisValue(value, range) {
  return clamp((Number(value) - range.min) / Math.max(1e-12, range.max - range.min));
}

function axisTickLabel(value, range) {
  const span = Math.abs(range.max - range.min);
  const digits = span < 0.02 ? 4 : span < 0.2 ? 3 : 2;
  return fmt(value, digits);
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
  const margin = { left: 68, right: 34, top: 28, bottom: 58 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const xAxis = objectiveAxisRange(data, 0);
  const yAxis = objectiveAxisRange(data, 1);
  const colorAxis = data.objectives.length > 2 ? objectiveAxisRange(data, 2) : null;
  const xScale = (v) => margin.left + normalizeAxisValue(v, xAxis) * innerW;
  const yScale = (v) => margin.top + (1 - normalizeAxisValue(v, yAxis)) * innerH;

  const xTicks = axisTicks(xAxis, 5);
  const yTicks = axisTicks(yAxis, 5);
  const grid = [
    ...xTicks.map(
      (value) => `
      <line class="grid" x1="${xScale(value)}" x2="${xScale(value)}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <text class="tick-label" x="${xScale(value)}" y="${height - 32}" text-anchor="middle">${axisTickLabel(value, xAxis)}</text>
    `,
    ),
    ...yTicks.map(
      (value) => `
      <line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(value)}" y2="${yScale(value)}"></line>
      <text class="tick-label" x="54" y="${yScale(value) + 4}" text-anchor="end">${axisTickLabel(value, yAxis)}</text>
    `,
    ),
  ].join("");
  const guideLines = [0.2, 0.4, 0.6, 0.8]
    .map(
      (t) => `
      <line class="guide-line" x1="${margin.left + t * innerW}" x2="${margin.left + t * innerW}" y1="${margin.top}" y2="${height - margin.bottom}"></line>
      <line class="guide-line" x1="${margin.left}" x2="${width - margin.right}" y1="${margin.top + (1 - t) * innerH}" y2="${margin.top + (1 - t) * innerH}"></line>
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
      const colorPosition = colorAxis ? normalizeAxisValue(colorValue, colorAxis) : colorValue;
      return `
        <circle class="${klass}" cx="${xScale(p.y[0])}" cy="${yScale(p.y[1])}" r="${radius}" fill="${objectiveColor(colorPosition)}">
          <title>${escapeHtml(pointTitle(p))}</title>
        </circle>
      `;
    })
    .join("");

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${grid}
    ${guideLines}
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
  const axisRanges = [0, 1, 2].map((index) => objectiveAxisRange(data, index));
  const normalizedObjectives = (point) => [
    normalizeAxisValue(point.y?.[0] ?? axisRanges[0].min, axisRanges[0]),
    normalizeAxisValue(point.y?.[1] ?? axisRanges[1].min, axisRanges[1]),
    normalizeAxisValue(point.y?.[2] ?? axisRanges[2].min, axisRanges[2]),
  ];
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
  const guideLines = [0.25, 0.5, 0.75]
    .flatMap((t) => [
      [[t, 0, 0], [t, 1, 0]],
      [[0, t, 0], [1, t, 0]],
      [[0, 0, t], [1, 0, t]],
      [[0, t, 0], [0, t, 1]],
      [[0, 0, t], [0, 1, t]],
      [[t, 0, 0], [t, 0, 1]],
    ])
    .map(([a, b]) => {
      const pa = project3d(a);
      const pb = project3d(b);
      return `<line class="guide-line soft-3d" x1="${pa.x}" y1="${pa.y}" x2="${pb.x}" y2="${pb.y}"></line>`;
    })
    .join("");
  const axes = axisEnds
    .map(
      ({ end, label }) => `
      <line class="axis-3d" x1="${origin.x}" y1="${origin.y}" x2="${end.x}" y2="${end.y}"></line>
      <text class="axis-label" x="${end.x}" y="${end.y - 8}" text-anchor="middle">${escapeHtml(label)}</text>
    `,
    )
    .join("");
  const tickLine = (from, to) => {
    const a = project3d(from);
    const b = project3d(to);
    return `<line class="axis-3d soft" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"></line>`;
  };
  const tickText = (point, label, dx = 0, dy = 0) => {
    const projected = project3d(point);
    return `<text class="tick-label" x="${projected.x + dx}" y="${projected.y + dy}" text-anchor="middle">${escapeHtml(label)}</text>`;
  };
  const ticks = [
    ...axisTicks(axisRanges[0], 3).map((value) => {
      const t = normalizeAxisValue(value, axisRanges[0]);
      return `${tickLine([t, 0, 0], [t, 0, 0.035])}${tickText([t, 0, 0], axisTickLabel(value, axisRanges[0]), 0, 20)}`;
    }),
    ...axisTicks(axisRanges[1], 3).map((value) => {
      const t = normalizeAxisValue(value, axisRanges[1]);
      return `${tickLine([0, t, 0], [0.035, t, 0])}${tickText([0, t, 0], axisTickLabel(value, axisRanges[1]), -34, 4)}`;
    }),
    ...axisTicks(axisRanges[2], 3).map((value) => {
      const t = normalizeAxisValue(value, axisRanges[2]);
      return `${tickLine([0, 0, t], [0.035, 0, t])}${tickText([0, 0, t], axisTickLabel(value, axisRanges[2]), -12, 16)}`;
    }),
  ].join("");
  const pareto = [...data.pareto]
    .map((p) => ({ ...p, projected: project3d(normalizedObjectives(p)) }))
    .sort((a, b) => a.y[0] - b.y[0]);
  const line =
    pareto.length > 1
      ? `<path class="front-line" d="${pareto.map((p, i) => `${i ? "L" : "M"} ${p.projected.x} ${p.projected.y}`).join(" ")}"></path>`
      : "";
  const points = data.points
    .map((p) => {
      const projected = project3d(normalizedObjectives(p));
      return { ...p, projected };
    })
    .sort((a, b) => a.projected.depth - b.projected.depth)
    .map((p) => {
      const radius = p.recommended ? 8.2 : p.pareto ? 6.7 : 4.8;
      const klass = `point${p.pareto ? " pareto" : ""}${p.recommended ? " recommended" : ""}`;
      return `<circle class="${klass}" cx="${p.projected.x}" cy="${p.projected.y}" r="${radius}" fill="${objectiveColor(normalizeAxisValue(p.y[2], axisRanges[2]))}"><title>${escapeHtml(pointTitle(p))}</title></circle>`;
    })
    .join("");
  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfdff"></rect>
    ${edges}
    ${guideLines}
    ${axes}
    ${ticks}
    ${line}
    ${points}
    <text class="legend" x="${width - 390}" y="28">Drag to rotate, dashed: sampled Pareto link, red: recommended</text>
  `;
  bind3dInteractions(svg);
}

function project3d(values) {
  const centered = values.slice(0, 3).map((value) => Number(value ?? 0) - 0.5);
  const angleY = state.view.yaw;
  const angleX = state.view.pitch;
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
    if (state.view.mode !== "3d") return;
    state.view.drag = { x: event.clientX, y: event.clientY, yaw: state.view.yaw, pitch: state.view.pitch };
    svg.setPointerCapture?.(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!state.view.drag || !state.data) return;
    const dx = event.clientX - state.view.drag.x;
    const dy = event.clientY - state.view.drag.y;
    state.view.yaw = state.view.drag.yaw + dx * 0.012;
    state.view.pitch = clamp(state.view.drag.pitch + dy * 0.012, -1.1, 1.1);
    renderScatter(state.data);
  });
  const endDrag = () => {
    state.view.drag = null;
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

function clamp(value, lo = 0, hi = 1) {
  return Math.min(hi, Math.max(lo, value));
}

function renderParetoList(data) {
  const sorted = [...data.pareto].sort((a, b) => b.preferenceScore - a.preferenceScore);
  $("paretoList").innerHTML = sorted
    .slice(0, 9)
    .map((point) => solutionCard(point, point.recommended))
    .join("");
}

function renderRecommendation(data) {
  $("recommendation").innerHTML = solutionCard(data.recommended, true, true);
}

function renderCandidates(data) {
  if (!data.lastCandidates.length) {
    $("candidates").innerHTML = `<p class="summary">Run a round to evaluate candidate designs.</p>`;
    return;
  }
  $("candidates").innerHTML = data.lastCandidates
    .map((item, idx) => {
      const point = {
        id: `new-${idx + 1}`,
        x: item.x,
        y: item.observed,
        preferenceScore: item.score,
      };
      return solutionCard(point, false, false, item.predicted);
    })
    .join("");
}

function renderExperimentResult(data) {
  if (!data.experimentResult) {
    $("experimentResult").textContent = `No budget run yet. Current backend budget progress is ${data.nEvaluations}/${data.settings.maxEvaluations}.`;
    return;
  }
  const result = data.experimentResult;
  const metricKey = $("metricSelect")?.value || "hypervolume";
  const metricLabel = data.metricLabels?.[metricKey] || metricKey;
  const metricValue = data.metrics?.[metricKey] ?? result[metricKey] ?? data.hypervolume ?? 0;
  const metricText = metricKey === "paretoCount" ? String(Math.round(metricValue)) : fmt(metricValue, 4);
  $("experimentResult").textContent =
    `${result.note} evaluations ${result.evaluations}/${result.maxEvaluations}, Pareto points ${result.paretoCount}, ` +
    `${metricLabel} ${metricText}, recommended #${result.recommendedId} score ${fmt(result.recommendedScore, 4)}, ` +
    `rounds ${result.rounds}, elapsed ${result.elapsedMs} ms.`;
}

function renderComparisonExperiment(result) {
  state.experiment = result;
  const hasRows = !!result?.rows?.length;
  ["downloadCsv", "downloadStatsCsv", "downloadMatrixCsv", "downloadReportHtml", "downloadStatsPng", "downloadMatrixPng", "downloadJson"].forEach((id) => {
    $(id).disabled = !hasRows;
  });
  if (!result?.rows?.length) {
    $("paperStatsTable").innerHTML = "";
    $("pairwiseMatrix").innerHTML = "";
    $("experimentSummary").innerHTML = "";
    $("experimentRows").innerHTML = "";
    $("experimentStatus").textContent = "No comparison data yet.";
    return;
  }
  const okRuns = result.rows.filter((row) => row.status === "ok").length;
  const errors = result.rows.length - okRuns;
  $("experimentStatus").textContent =
    `Finished ${okRuns}/${result.rows.length} backend runs with ${result.workers || 1} worker(s) in ${result.elapsedMs} ms` +
    `${errors ? `; ${errors} run(s) returned errors` : ""}.`;
  $("paperStatsTable").innerHTML = paperTableHtml(result.paperTable);
  $("pairwiseMatrix").innerHTML = pairwiseMatrixHtml(result.pairwiseMatrix);

  const summaryColumns = [
    ["benchmarkName", "Benchmark"],
    ["optimizerName", "Method"],
    ["runs", "Runs"],
    ["hypervolumeRank", "HV rank"],
    ["hypervolumeMean", "HV mean"],
    ["hypervolumeStd", "HV std"],
    ["igdRank", "IGD rank"],
    ["igdMean", "IGD mean"],
    ["igdPlusMean", "IGD+ mean"],
    ["gdMean", "GD mean"],
    ["paretoCountMean", "Pareto mean"],
    ["recommendedScoreMean", "Score mean"],
    ["hvWins", "HV wins"],
    ["elapsedMsMean", "Mean ms"],
  ];
  $("experimentSummary").innerHTML = tableHtml(result.summary, summaryColumns, "Summary statistics");

  const rawColumns = [
    ["benchmarkName", "Benchmark"],
    ["optimizerName", "Method"],
    ["seed", "Seed"],
    ["status", "Status"],
    ["evaluations", "Evals"],
    ["paretoCount", "Pareto"],
    ["hypervolume", "HV"],
    ["igd", "IGD"],
    ["igdPlus", "IGD+"],
    ["gd", "GD"],
    ["recommendedScore", "Score"],
    ["elapsedMs", "ms"],
  ];
  $("experimentRows").innerHTML = tableHtml(result.rows.slice(0, 120), rawColumns, `Raw runs (${result.rows.length} total)`);
}

function renderSingleObjectiveComparisonExperiment(result) {
  state.singleObjectiveExperiment = result;
  const hasRows = !!result?.rows?.length;
  [
    "soDownloadCsv",
    "soDownloadStatsCsv",
    "soDownloadMatrixCsv",
    "soDownloadReportHtml",
    "soDownloadStatsPng",
    "soDownloadMatrixPng",
    "soDownloadJson",
  ].forEach((id) => {
    $(id).disabled = !hasRows;
  });
  if (!result?.rows?.length) {
    $("soPaperStatsTable").innerHTML = "";
    $("soPairwiseMatrix").innerHTML = "";
    $("soExperimentSummary").innerHTML = "";
    $("soExperimentRows").innerHTML = "";
    $("soExperimentStatus").textContent = "No single-objective comparison data yet.";
    return;
  }
  const okRuns = result.rows.filter((row) => row.status === "ok").length;
  const errors = result.rows.length - okRuns;
  $("soExperimentStatus").textContent =
    `Finished ${okRuns}/${result.rows.length} backend runs with ${result.workers || 1} worker(s) in ${result.elapsedMs} ms` +
    `${errors ? `; ${errors} run(s) returned errors` : ""}.`;
  $("soPaperStatsTable").innerHTML = paperTableHtml(result.paperTable);
  $("soPairwiseMatrix").innerHTML = pairwiseMatrixHtml(result.pairwiseMatrix);

  const summaryColumns = [
    ["benchmarkName", "Benchmark"],
    ["methodName", "Method"],
    ["runs", "Runs"],
    ["selectedMetricRank", "Metric rank"],
    ["bestValueRank", "Best rank"],
    ["bestValueMean", "Best mean"],
    ["bestValueStd", "Best std"],
    ["optimumGapRank", "Gap rank"],
    ["optimumGapMean", "Gap mean"],
    ["optimumGapStd", "Gap std"],
    ["meanValueMean", "Mean f"],
    ["metricWins", "Metric wins"],
    ["elapsedMsMean", "Mean ms"],
  ];
  $("soExperimentSummary").innerHTML = tableHtml(result.summary, summaryColumns, "Single-objective summary statistics");

  const rawColumns = [
    ["benchmarkName", "Benchmark"],
    ["methodName", "Method"],
    ["seed", "Seed"],
    ["status", "Status"],
    ["evaluations", "Evals"],
    ["bestValue", "Best value"],
    ["optimumGap", "Optimum gap"],
    ["meanValue", "Mean value"],
    ["elapsedMs", "ms"],
  ];
  $("soExperimentRows").innerHTML = tableHtml(result.rows.slice(0, 120), rawColumns, `Single-objective raw runs (${result.rows.length} total)`);
}

function paperTableHtml(table) {
  if (!table?.rows?.length) return `<p class="summary">Paper-style statistics table: no rows.</p>`;
  return `
    <h3>${escapeHtml(table.metricLabel)} paper-style table: mean (variance), ${escapeHtml(table.metricDirection)}, reference ${escapeHtml(table.referenceName)}, alpha=${table.alpha}</h3>
    <table class="paper-stats">
      <thead>
        <tr>
          <th>Problem</th>
          <th>M</th>
          <th>D</th>
          ${table.optimizers.map((optimizer) => `<th>${escapeHtml(optimizer.name)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${table.rows
          .map(
            (row) => `
          <tr>
            <td>${escapeHtml(row.benchmarkName)}</td>
            <td>${escapeHtml(row.objectives)}</td>
            <td>${escapeHtml(row.dimension)}</td>
            ${table.optimizers.map((optimizer) => `<td>${escapeHtml(row.cells?.[optimizer.key]?.text || "")}</td>`).join("")}
          </tr>
        `,
          )
          .join("")}
        <tr class="stats-footer">
          <td colspan="3">+ / ~ / - versus ${escapeHtml(table.referenceName)}</td>
          ${table.optimizers.map((optimizer) => `<td>${escapeHtml(table.footer?.[optimizer.key] || "")}</td>`).join("")}
        </tr>
      </tbody>
    </table>
  `;
}

function pairwiseMatrixHtml(matrix) {
  if (!matrix?.optimizers?.length) return `<p class="summary">Pairwise significance matrix: no rows.</p>`;
  return `
    <h3>Algorithm x algorithm significance matrix: mean difference (variance), row minus column, alpha=${matrix.alpha}</h3>
    <table class="paper-stats">
      <thead>
        <tr>
          <th>Method</th>
          ${matrix.optimizers.map((optimizer) => `<th>${escapeHtml(optimizer.name)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${matrix.optimizers
          .map(
            (rowOptimizer) => `
          <tr>
            <td>${escapeHtml(rowOptimizer.name)}</td>
            ${matrix.optimizers.map((colOptimizer) => `<td>${escapeHtml(matrix.cells?.[rowOptimizer.key]?.[colOptimizer.key]?.text || "")}</td>`).join("")}
          </tr>
        `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function tableHtml(rows, columns, title) {
  if (!rows?.length) return `<p class="summary">${escapeHtml(title)}: no rows.</p>`;
  return `
    <h3>${escapeHtml(title)}</h3>
    <table>
      <thead>
        <tr>${columns.map(([, label]) => `<th>${escapeHtml(label)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
          <tr>
            ${columns.map(([key]) => `<td>${escapeHtml(formatCell(row[key]))}</td>`).join("")}
          </tr>
        `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function formatCell(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.abs(value) >= 100 ? String(Math.round(value)) : fmt(value, 4);
  }
  return value ?? "";
}

function rowsToCsv(rows) {
  if (!rows?.length) return "";
  const columns = Array.from(rows.reduce((keys, row) => {
    Object.keys(row).forEach((key) => keys.add(key));
    return keys;
  }, new Set()));
  const escapeCsv = (value) => {
    const text = String(value ?? "");
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [columns.join(","), ...rows.map((row) => columns.map((key) => escapeCsv(row[key])).join(","))].join("\n");
}

function paperTableToCsv(table) {
  if (!table?.rows?.length) return "";
  const header = ["Problem", "M", "D", ...table.optimizers.map((optimizer) => optimizer.name)];
  const lines = [header];
  table.rows.forEach((row) => {
    lines.push([
      row.benchmarkName,
      row.objectives,
      row.dimension,
      ...table.optimizers.map((optimizer) => row.cells?.[optimizer.key]?.text || ""),
    ]);
  });
  lines.push([
    `+ / ~ / - versus ${table.referenceName}`,
    "",
    "",
    ...table.optimizers.map((optimizer) => table.footer?.[optimizer.key] || ""),
  ]);
  return csvFromArrays(lines);
}

function pairwiseMatrixToCsv(matrix) {
  if (!matrix?.optimizers?.length) return "";
  const lines = [["Method", ...matrix.optimizers.map((optimizer) => optimizer.name)]];
  matrix.optimizers.forEach((rowOptimizer) => {
    lines.push([
      rowOptimizer.name,
      ...matrix.optimizers.map((colOptimizer) => matrix.cells?.[rowOptimizer.key]?.[colOptimizer.key]?.text || ""),
    ]);
  });
  return csvFromArrays(lines);
}

function csvFromArrays(lines) {
  const escapeCsv = (value) => {
    const text = String(value ?? "");
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return lines.map((line) => line.map(escapeCsv).join(",")).join("\n");
}

function experimentReportHtml(result) {
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>BlackBoxLab comparison report</title>
  <style>
    body { font-family: "Times New Roman", serif; color: #111; margin: 24px; }
    h1 { font-size: 18px; text-align: center; text-transform: uppercase; }
    h2 { font-size: 14px; margin-top: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    th, td { border: 1px solid #333; padding: 4px 6px; text-align: center; }
    th { font-weight: 700; }
    .left { text-align: left; }
    .footer td { font-weight: 700; }
  </style>
</head>
<body>
  <h1>Statistical results for ${escapeHtml(result.metric)} comparison</h1>
  ${paperTableHtml(result.paperTable).replaceAll("<h3", "<h2").replaceAll("</h3>", "</h2>")}
  ${pairwiseMatrixHtml(result.pairwiseMatrix).replaceAll("<h3", "<h2").replaceAll("</h3>", "</h2>")}
</body>
</html>`;
}

function downloadBlob(filename, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function paperTableHtml(table) {
  if (!table?.rows?.length) return `<p class="summary">Paper-style statistics table: no rows.</p>`;
  return `
    <h3>${escapeHtml(table.metricLabel)} paper-style table: mean (variance), ${escapeHtml(table.metricDirection)}, reference ${escapeHtml(table.referenceName)}, alpha=${table.alpha}</h3>
    <table class="paper-stats">
      <thead>
        <tr>
          <th>Problem</th>
          <th>M</th>
          <th>D</th>
          ${table.optimizers.map((optimizer) => `<th>${escapeHtml(optimizer.name)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${table.rows
          .map(
            (row) => `
          <tr>
            <td>${escapeHtml(row.benchmarkName)}</td>
            <td>${escapeHtml(row.objectives)}</td>
            <td>${escapeHtml(row.dimension)}</td>
            ${table.optimizers.map((optimizer) => `<td>${escapeHtml(row.cells?.[optimizer.key]?.text || "")}</td>`).join("")}
          </tr>
        `,
          )
          .join("")}
        <tr class="stats-footer">
          <td colspan="3">+ / ~ / - versus ${escapeHtml(table.referenceName)}</td>
          ${table.optimizers.map((optimizer) => `<td>${escapeHtml(table.footer?.[optimizer.key] || "")}</td>`).join("")}
        </tr>
      </tbody>
    </table>
  `;
}

function pairwiseMatrixHtml(matrix) {
  if (!matrix?.optimizers?.length) return `<p class="summary">Pairwise significance matrix: no rows.</p>`;
  return `
    <h3>Algorithm x algorithm significance matrix: mean improvement (variance), row versus column, ${escapeHtml(matrix.metricDirection)}, alpha=${matrix.alpha}</h3>
    <table class="paper-stats">
      <thead>
        <tr>
          <th>Method</th>
          ${matrix.optimizers.map((optimizer) => `<th>${escapeHtml(optimizer.name)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${matrix.optimizers
          .map(
            (rowOptimizer) => `
          <tr>
            <td>${escapeHtml(rowOptimizer.name)}</td>
            ${matrix.optimizers.map((colOptimizer) => `<td>${escapeHtml(matrix.cells?.[rowOptimizer.key]?.[colOptimizer.key]?.text || "")}</td>`).join("")}
          </tr>
        `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function paperTableToCsv(table) {
  if (!table?.rows?.length) return "";
  const lines = [["Problem", "M", "D", ...table.optimizers.map((optimizer) => optimizer.name)]];
  table.rows.forEach((row) => {
    lines.push([
      row.benchmarkName,
      row.objectives,
      row.dimension,
      ...table.optimizers.map((optimizer) => row.cells?.[optimizer.key]?.text || ""),
    ]);
  });
  lines.push([
    `+ / ~ / - versus ${table.referenceName}`,
    "",
    "",
    ...table.optimizers.map((optimizer) => table.footer?.[optimizer.key] || ""),
  ]);
  return csvFromArrays(lines);
}

async function submitCustomProblem() {
  const source = $("customProblemCode").value.trim();
  const displayName = $("customProblemName")?.value.trim() || "";
  if (!source) {
    $("importStatus").textContent = "Paste or load a problem definition first.";
    return;
  }
  if (displayName && displayNameExists(state.data?.benchmarks, displayName)) {
    $("importStatus").textContent = `A problem named "${displayName}" already exists. Choose a different name.`;
    return;
  }
  try {
    $("importStatus").textContent = "Submitting problem to the local Python backend...";
    const result = await api("/api/submit_problem", {
      source,
      displayName,
      filename: "submitted_problem.py",
      feedback: $("feedback").value,
      settings: currentSettings(),
    });
    render(result.state);
    $("importStatus").textContent = importResultDetails(
      result.importResult,
      `Submitted and selected ${result.importResult.selectedBenchmark}. You can now run rounds or run to budget`,
    );
  } catch (error) {
    $("importStatus").textContent =
      error.message.includes("404") || error.message === "Not Found"
        ? "The running backend is old. Restart python server.py, then Ctrl+F5."
        : `Problem submission rejected: ${error.message}`;
  }
}

async function submitCustomOptimizer() {
  const source = $("customOptimizerCode").value.trim();
  const displayName = $("customOptimizerName")?.value.trim() || "";
  if (!source) {
    $("optimizerImportStatus").textContent = "Paste or load an optimizer definition first.";
    return;
  }
  if (displayName && displayNameExists(state.data?.optimizers, displayName)) {
    $("optimizerImportStatus").textContent = `An algorithm named "${displayName}" already exists. Choose a different name.`;
    return;
  }
  try {
    $("optimizerImportStatus").textContent = "Submitting method to the local Python backend...";
    const result = await api("/api/submit_optimizer", {
      source,
      displayName,
      filename: "submitted_optimizer.py",
      feedback: $("feedback").value,
      settings: currentSettings(),
    });
    render(result.state);
    $("optimizerImportStatus").textContent = importResultDetails(
      result.importResult,
      `Submitted and selected ${result.importResult.selectedOptimizer}. You can now run it or compare it in Experiment Lab`,
    );
  } catch (error) {
    $("optimizerImportStatus").textContent =
      error.message.includes("404") || error.message === "Not Found"
        ? "The running backend is old. Restart python server.py, then Ctrl+F5."
        : `Method submission rejected: ${error.message}`;
  }
}

function chartSvgCss() {
  return `
    .axis{stroke:#8c98a8;stroke-width:1}.grid{stroke:#e6ebf1;stroke-width:1}
    .tick-label{fill:#6b7788;font-size:12px}.axis-label{fill:#344255;font-size:13px;font-weight:800}
    .point{stroke:white;stroke-width:1.5}.point.pareto{stroke:#152033;stroke-width:2.2}
    .point.recommended{stroke:#d14b4b;stroke-width:3}.point.recent{stroke:#17202a;stroke-width:1.8}
    .point.faint{fill:#0f9f8f;opacity:.38;stroke:none}.front-line{fill:none;stroke:#152033;stroke-width:2;stroke-dasharray:6 5;opacity:.72}
    .search-line{fill:none;stroke:#2563eb;stroke-width:2;opacity:.72}.convergence-line{fill:none;stroke:#2563eb;stroke-width:2.4}
    .guide-line{stroke:#d8e1ea;stroke-width:1;stroke-dasharray:4 5;opacity:.72}
    .landscape-curve{fill:none;stroke:#0f766e;stroke-width:2.8;opacity:.86}.landscape-cell{opacity:.34;stroke:none}
    .landscape-mesh{fill:none;stroke:#0f766e;stroke-width:1.15;opacity:.38}.soft-3d{stroke:#d8e1ea;stroke-width:1;stroke-dasharray:4 5;opacity:.72}
    .legend{fill:#445267;font-size:12px}.axis-3d{stroke:#708198;stroke-width:1.4}.axis-3d.soft{stroke:#c5ced9;stroke-width:1}
  `;
}

function serializedSvg(svgId, width = 1520, height = 940) {
  const svg = $(svgId);
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = chartSvgCss();
  clone.insertBefore(style, clone.firstChild);
  return new XMLSerializer().serializeToString(clone);
}

function serializedScatterSvg() {
  return serializedSvg("scatter");
}

function downloadCurrentChartSvg() {
  downloadBlob("blackboxlab_objective_tradeoff.svg", serializedScatterSvg(), "image/svg+xml;charset=utf-8");
}

function downloadSvgAsPng(svgText, filename, width = 1520, height = 940) {
  const image = new Image();
  const blob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  image.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(image, 0, 0, width, height);
    URL.revokeObjectURL(url);
    canvas.toBlob((png) => {
      if (!png) return;
      const pngUrl = URL.createObjectURL(png);
      const link = document.createElement("a");
      link.href = pngUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(pngUrl);
    }, "image/png");
  };
  image.src = url;
}

function makeSingleObjectiveFrameSteps(data, maxFrames = 36) {
  const population = Math.max(1, Number(data.settings?.populationSize || 1));
  const first = Math.min(data.points.length, population);
  const steps = [];
  for (let evaluation = first; evaluation <= data.points.length; evaluation += population) {
    steps.push(evaluation);
  }
  if (!steps.includes(data.points.length)) steps.push(data.points.length);
  if (steps.length <= maxFrames) return steps;
  const sampled = [];
  for (let index = 0; index < maxFrames; index += 1) {
    sampled.push(steps[Math.round((index / (maxFrames - 1)) * (steps.length - 1))]);
  }
  return [...new Set(sampled)];
}

function renderSvgToDataUrl(svg) {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", "760");
  clone.setAttribute("height", "470");
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = chartSvgCss();
  clone.insertBefore(style, clone.firstChild);
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(clone))}`;
}

function imageFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = dataUrl;
  });
}

function gifPalette() {
  const base = [
    [255, 255, 255],
    [251, 253, 255],
    [238, 243, 247],
    [226, 232, 240],
    [203, 213, 225],
    [140, 152, 168],
    [68, 82, 103],
    [23, 32, 42],
    [37, 99, 235],
    [15, 159, 143],
    [47, 138, 76],
    [209, 75, 75],
    [242, 184, 75],
  ];
  const palette = [];
  for (let r = 0; r < 6; r += 1) {
    for (let g = 0; g < 6; g += 1) {
      for (let b = 0; b < 6; b += 1) palette.push([r * 51, g * 51, b * 51]);
    }
  }
  for (const color of base) {
    palette.unshift(color);
  }
  return palette.slice(0, 256);
}

function nearestPaletteIndex(r, g, b, palette, cache) {
  const key = `${r >> 3},${g >> 3},${b >> 3}`;
  if (cache.has(key)) return cache.get(key);
  let best = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < palette.length; index += 1) {
    const color = palette[index];
    const dr = r - color[0];
    const dg = g - color[1];
    const db = b - color[2];
    const distance = dr * dr + dg * dg + db * db;
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  cache.set(key, best);
  return best;
}

function lzwEncode(minCodeSize, indices) {
  const clearCode = 1 << minCodeSize;
  const endCode = clearCode + 1;
  const codeSize = minCodeSize + 1;
  const segmentLength = 240;
  const bytes = [];
  let current = 0;
  let bits = 0;
  const writeCode = (code) => {
    current |= code << bits;
    bits += codeSize;
    while (bits >= 8) {
      bytes.push(current & 255);
      current >>= 8;
      bits -= 8;
    }
  };
  writeCode(clearCode);
  let writtenSinceClear = 0;
  for (let i = 0; i < indices.length; i += 1) {
    writeCode(indices[i] & (clearCode - 1));
    writtenSinceClear += 1;
    if (writtenSinceClear >= segmentLength && i < indices.length - 1) {
      writeCode(clearCode);
      writtenSinceClear = 0;
    }
  }
  writeCode(endCode);
  if (bits > 0) bytes.push(current & 255);
  return bytes;
}

function appendBytes(target, values) {
  for (let index = 0; index < values.length; index += 4096) {
    target.push(...values.slice(index, index + 4096));
  }
}

function appendGifSubBlocks(target, bytes) {
  for (let index = 0; index < bytes.length; index += 255) {
    const block = bytes.slice(index, index + 255);
    target.push(block.length);
    appendBytes(target, block);
  }
  target.push(0);
}

function wordBytes(value) {
  return [value & 255, (value >> 8) & 255];
}

function encodeGif(frames, width, height, delayMs = 180) {
  const palette = gifPalette();
  const colorTable = palette.flat();
  while (colorTable.length < 768) colorTable.push(0);
  const bytes = [..."GIF89a"].map((char) => char.charCodeAt(0));
  bytes.push(...wordBytes(width), ...wordBytes(height), 0xf7, 0, 0);
  appendBytes(bytes, colorTable);
  bytes.push(0x21, 0xff, 11, ..."NETSCAPE2.0".split("").map((char) => char.charCodeAt(0)), 3, 1, 0, 0, 0);
  const cache = new Map();
  const delay = Math.max(2, Math.round(delayMs / 10));
  for (const imageData of frames) {
    const indices = new Uint8Array(width * height);
    let pixelIndex = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
      const alpha = imageData.data[i + 3] / 255;
      const r = Math.round(imageData.data[i] * alpha + 255 * (1 - alpha));
      const g = Math.round(imageData.data[i + 1] * alpha + 255 * (1 - alpha));
      const b = Math.round(imageData.data[i + 2] * alpha + 255 * (1 - alpha));
      indices[pixelIndex] = nearestPaletteIndex(r, g, b, palette, cache);
      pixelIndex += 1;
    }
    bytes.push(0x21, 0xf9, 4, 0, ...wordBytes(delay), 0, 0);
    bytes.push(0x2c, 0, 0, 0, 0, ...wordBytes(width), ...wordBytes(height), 0);
    bytes.push(8);
    appendGifSubBlocks(bytes, lzwEncode(8, indices));
  }
  bytes.push(0x3b);
  return new Blob([new Uint8Array(bytes)], { type: "image/gif" });
}

async function renderSingleObjectiveGifFrame(kind, data, evaluation, width, height) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 760 470");
  if (kind === "search") {
    renderSingleObjectiveSearch(data, { svg, maxEvaluation: evaluation, silent: true, history: true, inlineLandscape: true });
  } else {
    renderSingleObjectiveConvergence(data, { svg, maxEvaluation: evaluation });
  }
  const image = await imageFromDataUrl(renderSvgToDataUrl(svg));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

async function downloadSingleObjectiveGif(kind) {
  const data = state.singleObjective;
  const button = kind === "search" ? $("soDownloadSearchGif") : $("soDownloadConvergenceGif");
  if (!data?.points?.length) {
    const original = button.textContent;
    button.textContent = "No data";
    setTimeout(() => {
      button.textContent = original;
    }, 900);
    return;
  }
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "...";
  try {
    const steps = makeSingleObjectiveFrameSteps(data, 18);
    const width = 760;
    const height = 470;
    const frames = [];
    for (let index = 0; index < steps.length; index += 1) {
      button.textContent = `${index + 1}/${steps.length}`;
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const evaluation = steps[index];
      frames.push(await renderSingleObjectiveGifFrame(kind, data, evaluation, width, height));
    }
    button.textContent = "GIF...";
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const blob = encodeGif(frames, width, height, 180);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = kind === "search" ? "blackboxlab_single_objective_search.gif" : "blackboxlab_single_objective_convergence.gif";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1200);
  } catch (error) {
    console.error("GIF export failed", error);
    button.textContent = "Failed";
    await new Promise((resolve) => setTimeout(resolve, 900));
  } finally {
    button.textContent = original;
    button.disabled = false;
  }
}

function downloadElementAsPng(elementId, filename) {
  const element = $(elementId);
  const width = Math.max(1200, element.scrollWidth + 40);
  const height = Math.max(420, element.scrollHeight + 40);
  const html = `<div xmlns="http://www.w3.org/1999/xhtml" style="background:white;color:#111;font-family:Times New Roman,serif;padding:20px;">
    <style>
      h3{font-size:16px;margin:0 0 10px;text-align:center}
      table{width:100%;border-collapse:collapse;font-size:13px}
      th,td{border:1px solid #222;padding:6px 8px;text-align:center;white-space:nowrap}
      th:first-child,td:first-child{text-align:left}
      .stats-footer td{font-weight:bold;border-top:2px solid #222}
    </style>
    ${element.innerHTML}
  </div>`;
  const svgText = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><foreignObject width="100%" height="100%">${html}</foreignObject></svg>`;
  downloadSvgAsPng(svgText, filename, width, height);
}

async function runComparisonExperiment() {
  const benchmarks = selectedValues($("expBenchmarks"));
  const optimizers = selectedValues($("expOptimizers"));
  if (!benchmarks.length || !optimizers.length) {
    $("experimentStatus").textContent = "Select at least one benchmark and one method.";
    return;
  }
  const scale = experimentScale("multi");
  if (scale.runs > scale.limit) {
    $("experimentStatus").textContent = `Experiment grid is too large: ${scale.runs} runs; limit is ${scale.limit}. Use Core, filter+Visible, or reduce repeats.`;
    updateExperimentScale("multi");
    return;
  }
  $("experimentStatus").textContent = "Running backend comparison...";
  $("runExperiment").disabled = true;
  $("downloadCsv").disabled = true;
  $("downloadStatsCsv").disabled = true;
  $("downloadMatrixCsv").disabled = true;
  $("downloadReportHtml").disabled = true;
  $("downloadStatsPng").disabled = true;
  $("downloadMatrixPng").disabled = true;
  $("downloadJson").disabled = true;
  try {
    const result = await api("/api/experiment", {
      benchmarks,
      optimizers,
      repeats: $("expRepeats").value,
      seed: $("expSeed").value,
      workers: $("expWorkers").value,
      metric: $("expMetric").value,
      referenceOptimizer: $("expReference").value,
      settings: experimentSettings(),
      feedback: $("feedback").value,
    });
    renderComparisonExperiment(result);
    addRunHistory({
      kind: "multi-objective-experiment",
      title: "Multi-objective experiment",
      metric: result.metric,
      config: currentPlatformConfig(),
      result: experimentHistoryResult(result),
    });
  } catch (error) {
    $("experimentStatus").textContent = error.message;
  } finally {
    $("runExperiment").disabled = false;
  }
}

async function runSingleObjectiveComparisonExperiment() {
  const benchmarks = selectedValues($("soExpBenchmarks"));
  const methods = selectedValues($("soExpMethods"));
  if (!benchmarks.length || !methods.length) {
    $("soExperimentStatus").textContent = "Select at least one benchmark and one method.";
    return;
  }
  const scale = experimentScale("single");
  if (scale.runs > scale.limit) {
    $("soExperimentStatus").textContent = `Experiment grid is too large: ${scale.runs} runs; limit is ${scale.limit}. Use Core, filter+Visible, or reduce repeats.`;
    updateExperimentScale("single");
    return;
  }
  $("soExperimentStatus").textContent = "Running single-objective backend comparison...";
  $("soRunExperiment").disabled = true;
  [
    "soDownloadCsv",
    "soDownloadStatsCsv",
    "soDownloadMatrixCsv",
    "soDownloadReportHtml",
    "soDownloadStatsPng",
    "soDownloadMatrixPng",
    "soDownloadJson",
  ].forEach((id) => {
    $(id).disabled = true;
  });
  try {
    const result = await api("/api/single/experiment", {
      benchmarks,
      methods,
      repeats: $("soExpRepeats").value,
      seed: $("soExpSeed").value,
      workers: $("soExpWorkers").value,
      metric: $("soExpMetric").value,
      referenceMethod: $("soExpReference").value,
      settings: singleObjectiveExperimentSettings(),
    });
    renderSingleObjectiveComparisonExperiment(result);
    addRunHistory({
      kind: "single-objective-experiment",
      title: "Single-objective experiment",
      metric: result.metric,
      config: currentPlatformConfig(),
      result: experimentHistoryResult(result),
    });
  } catch (error) {
    $("soExperimentStatus").textContent =
      error.message.includes("404") || error.message === "Not Found"
        ? "The running backend is old. Restart python server.py, then Ctrl+F5."
        : error.message;
  } finally {
    $("soRunExperiment").disabled = false;
  }
}

function solutionCard(point, recommended = false, expanded = false, predicted = null) {
  const badge = recommended ? `<span class="badge hot">Recommended</span>` : `<span class="badge">#${point.id}</span>`;
  const x = point.x;
  const y = point.y;
  const pred = predicted
    ? `<p class="summary">Predicted: ${predicted.map((value, index) => `${objectiveName(index)} ${fmt(value)}`).join(", ")}</p>`
    : "";
  const inputLabels = window.currentInputs || ["Depth", "Compression", "Cache"];
  const inputs = expanded
    ? `
      <div class="triple">${x.map((value, index) => mini(inputLabels[index] || `x${index + 1}`, value)).join("")}</div>
    `
    : "";
  return `
    <article class="solution${recommended ? " recommended" : ""}">
      <div class="solution-head">
        <strong>Preference score ${fmt(point.preferenceScore)}</strong>
        ${badge}
      </div>
      <div class="triple">${y.map((value, index) => mini(objectiveName(index), value)).join("")}</div>
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

async function withBusy(fn) {
  if (state.busy) return;
  state.busy = true;
  document.body.style.cursor = "progress";
  try {
    await fn();
  } finally {
    state.busy = false;
    document.body.style.cursor = "";
  }
}

async function runBudgetLive() {
  if (state.runningBudget) {
    state.runningBudget = false;
    $("runBudget").textContent = "Run to budget";
    return;
  }
  state.runningBudget = true;
  state.busy = true;
  $("runBudget").textContent = "Stop run";
  document.body.style.cursor = "progress";
  const started = performance.now();
  let rounds = 0;
  let completedData = null;
  try {
    let data = state.data || (await api("/api/state"));
    while (state.runningBudget && data.budgetRemaining > 0) {
      data = await api("/api/step", {});
      rounds += 1;
      render(data);
      await new Promise((resolve) => setTimeout(resolve, 80));
      if (data.budgetRemaining <= 0) {
        data = await api("/api/finish", {
          rounds,
          elapsedMs: Math.round(performance.now() - started),
          note: "Budget run completed.",
        });
        render(data);
        completedData = data;
      }
    }
  } finally {
    if (completedData) {
      addRunHistory({
        kind: "multi-objective-test",
        title: "Multi-objective budget run",
        metric: $("metricSelect")?.value || "hypervolume",
        config: currentPlatformConfig(),
        result: multiRunHistoryResult(completedData),
      });
    }
    state.runningBudget = false;
    state.busy = false;
    $("runBudget").textContent = "Run to budget";
    document.body.style.cursor = "";
  }
}

async function runSingleBudgetLive() {
  if (state.runningSingleBudget) {
    state.runningSingleBudget = false;
    $("soRunBudget").textContent = "Run to budget";
    return;
  }
  state.runningSingleBudget = true;
  state.busy = true;
  $("soRunBudget").textContent = "Stop run";
  document.body.style.cursor = "progress";
  let latestData = null;
  let lastRenderedEvaluation = state.singleObjective?.nEvaluations ?? -1;
  let lastRenderAt = 0;
  let completedData = null;
  try {
    let data = state.singleObjective || (await api("/api/single/state"));
    while (state.runningSingleBudget && data.budgetRemaining > 0) {
      data = await api("/api/single/step", {});
      latestData = data;
      const now = Date.now();
      if (now - lastRenderAt >= 160 || data.budgetRemaining <= 0) {
        renderSingleObjective(data);
        lastRenderedEvaluation = data.nEvaluations;
        lastRenderAt = now;
      }
      if (data.budgetRemaining <= 0) completedData = data;
      await new Promise((resolve) => setTimeout(resolve, 45));
    }
  } finally {
    if (latestData && latestData.nEvaluations !== lastRenderedEvaluation) {
      renderSingleObjective(latestData);
    }
    if (completedData) {
      addRunHistory({
        kind: "single-objective-test",
        title: "Single-objective budget run",
        metric: "bestValue",
        config: currentPlatformConfig(),
        result: singleRunHistoryResult(completedData),
      });
    }
    state.runningSingleBudget = false;
    state.busy = false;
    $("soRunBudget").textContent = "Run to budget";
    document.body.style.cursor = "";
  }
}

$("applyFeedback").addEventListener("click", () =>
  withBusy(async () => {
    render(await api("/api/feedback", { feedback: $("feedback").value }));
  }),
);

$("step").addEventListener("click", () =>
  withBusy(async () => {
    render(await api("/api/step", {}));
  }),
);

$("applySettings").addEventListener("click", () =>
  withBusy(async () => {
    render(await api("/api/settings", { settings: currentSettings() }));
  }),
);

$("runBudget").addEventListener("click", () =>
  runBudgetLive(),
);

$("benchmarkSelect").addEventListener("change", () =>
  withBusy(async () => {
    render(await api("/api/benchmark", { benchmark: $("benchmarkSelect").value }));
  }),
);

$("optimizerSelect").addEventListener("change", () =>
  withBusy(async () => {
    render(await api("/api/optimizer", { optimizer: $("optimizerSelect").value }));
  }),
);

$("objectiveView").addEventListener("change", () => {
  if (state.data) renderScatter(state.data);
});

$("downloadChartSvg").addEventListener("click", downloadCurrentChartSvg);

$("downloadChartPng").addEventListener("click", () => {
  downloadSvgAsPng(serializedScatterSvg(), "blackboxlab_objective_tradeoff.png");
});

$("soSearchView").addEventListener("change", () => {
  if (state.singleObjective) renderSingleObjectiveSearch(state.singleObjective);
});

$("soLandscapeToggle").addEventListener("change", () => {
  if (!state.singleObjective) return;
  if (!$("soLandscapeToggle").checked) {
    state.singleObjectiveLandscape = null;
  }
  renderSingleObjectiveSearch(state.singleObjective);
});

$("soHistoryToggle").addEventListener("change", () => {
  if (state.singleObjective) renderSingleObjectiveSearch(state.singleObjective);
});

$("soDownloadSearchSvg").addEventListener("click", () => {
  downloadBlob("blackboxlab_single_objective_search.svg", serializedSvg("soSearch"), "image/svg+xml;charset=utf-8");
});

$("soDownloadSearchPng").addEventListener("click", () => {
  downloadSvgAsPng(serializedSvg("soSearch"), "blackboxlab_single_objective_search.png");
});

$("soDownloadSearchGif").addEventListener("click", () => {
  downloadSingleObjectiveGif("search");
});

$("soDownloadConvergenceSvg").addEventListener("click", () => {
  downloadBlob("blackboxlab_single_objective_convergence.svg", serializedSvg("soConvergence"), "image/svg+xml;charset=utf-8");
});

$("soDownloadConvergencePng").addEventListener("click", () => {
  downloadSvgAsPng(serializedSvg("soConvergence"), "blackboxlab_single_objective_convergence.png");
});

$("soDownloadConvergenceGif").addEventListener("click", () => {
  downloadSingleObjectiveGif("convergence");
});

$("singleRunTab").addEventListener("click", () => {
  state.workspaceView = "single";
  syncWorkspaceView();
});

$("experimentTab").addEventListener("click", () => {
  state.workspaceView = "experiment";
  syncWorkspaceView();
});

$("singleObjectiveTab").addEventListener("click", () => {
  state.workspaceView = "singleObjective";
  syncWorkspaceView();
  if (!state.singleObjective) {
    api("/api/single/state").then(renderSingleObjective).catch((error) => {
      $("soSettingsSummary").textContent = error.message;
    });
  }
});

$("singleObjectiveExperimentTab").addEventListener("click", () => {
  state.workspaceView = "singleObjectiveExperiment";
  syncWorkspaceView();
  if (!state.singleObjective) {
    api("/api/single/state").then(renderSingleObjective).catch((error) => {
      $("soExperimentStatus").textContent = error.message;
    });
  }
});

$("metricSelect").addEventListener("change", () => {
  if (state.data) {
    renderSelectedMetric(state.data);
    renderExperimentResult(state.data);
  }
});

$("auto").addEventListener("click", () =>
  withBusy(async () => {
    let data = null;
    for (let i = 0; i < 5; i += 1) {
      data = await api("/api/step", {});
      render(data);
      await new Promise((resolve) => setTimeout(resolve, 160));
    }
  }),
);

$("reset").addEventListener("click", () =>
  withBusy(async () => {
    render(
      await api("/api/reset", {
        feedback: $("feedback").value,
        seed: Date.now() % 100000,
        benchmark: $("benchmarkSelect").value,
        optimizer: $("optimizerSelect").value,
        settings: currentSettings(),
      }),
    );
  }),
);

$("soBenchmarkSelect").addEventListener("change", () =>
  withBusy(async () => {
    renderSingleObjective(await api("/api/single/benchmark", { benchmark: $("soBenchmarkSelect").value }));
  }),
);

$("soMethodSelect").addEventListener("change", () =>
  withBusy(async () => {
    renderSingleObjective(await api("/api/single/method", { method: $("soMethodSelect").value }));
  }),
);

$("soApplySettings").addEventListener("click", () =>
  withBusy(async () => {
    renderSingleObjective(await api("/api/single/settings", { settings: singleObjectiveSettings() }));
  }),
);

$("soRunRound").addEventListener("click", () =>
  withBusy(async () => {
    renderSingleObjective(await api("/api/single/step", {}));
  }),
);

$("soRunBudget").addEventListener("click", () =>
  runSingleBudgetLive(),
);

$("soReset").addEventListener("click", () =>
  withBusy(async () => {
    renderSingleObjective(
      await api("/api/single/reset", {
        seed: Date.now() % 100000,
        benchmark: $("soBenchmarkSelect").value,
        method: $("soMethodSelect").value,
        settings: singleObjectiveSettings(),
      }),
    );
  }),
);

["expPopulation", "expBudget", "expDimension", "expObjectives"].forEach((id) => {
  $(id).addEventListener("input", () => {
    $(id).dataset.touched = "1";
  });
});

["soExpPopulation", "soExpBudget", "soExpDimension"].forEach((id) => {
  $(id).addEventListener("input", () => {
    $(id).dataset.touched = "1";
  });
});

$("runExperiment").addEventListener("click", () =>
  withBusy(runComparisonExperiment),
);

$("soRunExperiment").addEventListener("click", () =>
  withBusy(runSingleObjectiveComparisonExperiment),
);

$("downloadCsv").addEventListener("click", () => {
  if (!state.experiment?.rows?.length) return;
  downloadBlob("blackboxlab_comparison_rows.csv", `\ufeff${rowsToCsv(state.experiment.rows)}`, "text/csv;charset=utf-8");
});

$("downloadStatsCsv").addEventListener("click", () => {
  if (!state.experiment?.paperTable?.rows?.length) return;
  downloadBlob("blackboxlab_paper_statistics.csv", `\ufeff${paperTableToCsv(state.experiment.paperTable)}`, "text/csv;charset=utf-8");
});

$("downloadMatrixCsv").addEventListener("click", () => {
  if (!state.experiment?.pairwiseMatrix?.optimizers?.length) return;
  downloadBlob("blackboxlab_pairwise_significance.csv", `\ufeff${pairwiseMatrixToCsv(state.experiment.pairwiseMatrix)}`, "text/csv;charset=utf-8");
});

$("downloadReportHtml").addEventListener("click", () => {
  if (!state.experiment) return;
  downloadBlob("blackboxlab_comparison_report.html", experimentReportHtml(state.experiment), "text/html;charset=utf-8");
});

$("downloadStatsPng").addEventListener("click", () => {
  if (!state.experiment?.paperTable?.rows?.length) return;
  downloadElementAsPng("paperStatsTable", "blackboxlab_paper_statistics.png");
});

$("downloadMatrixPng").addEventListener("click", () => {
  if (!state.experiment?.pairwiseMatrix?.optimizers?.length) return;
  downloadElementAsPng("pairwiseMatrix", "blackboxlab_pairwise_significance.png");
});

$("downloadJson").addEventListener("click", () => {
  if (!state.experiment) return;
  downloadBlob("blackboxlab_comparison_results.json", JSON.stringify(state.experiment, null, 2), "application/json;charset=utf-8");
});

$("soDownloadCsv").addEventListener("click", () => {
  if (!state.singleObjectiveExperiment?.rows?.length) return;
  downloadBlob("blackboxlab_single_objective_rows.csv", `\ufeff${rowsToCsv(state.singleObjectiveExperiment.rows)}`, "text/csv;charset=utf-8");
});

$("soDownloadStatsCsv").addEventListener("click", () => {
  if (!state.singleObjectiveExperiment?.paperTable?.rows?.length) return;
  downloadBlob("blackboxlab_single_objective_statistics.csv", `\ufeff${paperTableToCsv(state.singleObjectiveExperiment.paperTable)}`, "text/csv;charset=utf-8");
});

$("soDownloadMatrixCsv").addEventListener("click", () => {
  if (!state.singleObjectiveExperiment?.pairwiseMatrix?.optimizers?.length) return;
  downloadBlob("blackboxlab_single_objective_significance.csv", `\ufeff${pairwiseMatrixToCsv(state.singleObjectiveExperiment.pairwiseMatrix)}`, "text/csv;charset=utf-8");
});

$("soDownloadReportHtml").addEventListener("click", () => {
  if (!state.singleObjectiveExperiment) return;
  downloadBlob("blackboxlab_single_objective_report.html", experimentReportHtml(state.singleObjectiveExperiment), "text/html;charset=utf-8");
});

$("soDownloadStatsPng").addEventListener("click", () => {
  if (!state.singleObjectiveExperiment?.paperTable?.rows?.length) return;
  downloadElementAsPng("soPaperStatsTable", "blackboxlab_single_objective_statistics.png");
});

$("soDownloadMatrixPng").addEventListener("click", () => {
  if (!state.singleObjectiveExperiment?.pairwiseMatrix?.optimizers?.length) return;
  downloadElementAsPng("soPairwiseMatrix", "blackboxlab_single_objective_significance.png");
});

$("soDownloadJson").addEventListener("click", () => {
  if (!state.singleObjectiveExperiment) return;
  downloadBlob("blackboxlab_single_objective_results.json", JSON.stringify(state.singleObjectiveExperiment, null, 2), "application/json;charset=utf-8");
});

$("loadProblemExample").addEventListener("click", () => {
  $("customProblemCode").value = customProblemExample;
  $("customProblemName").value = "My submitted problem";
  $("importStatus").textContent = "Example problem loaded. Edit it or submit directly.";
});

$("submitProblem").addEventListener("click", () =>
  withBusy(submitCustomProblem),
);

$("loadOptimizerExample").addEventListener("click", () => {
  $("customOptimizerCode").value = customOptimizerExample;
  $("customOptimizerName").value = "Custom random local search";
  $("optimizerImportStatus").textContent = "Example method loaded. Edit it or submit directly.";
});

$("submitOptimizer").addEventListener("click", () =>
  withBusy(submitCustomOptimizer),
);

$("refreshRegistry")?.addEventListener("click", () =>
  withBusy(async () => {
    const result = await api("/api/refresh", {});
    render(result.state);
    try {
      renderSingleObjective(await api("/api/single/state"));
    } catch (error) {
      console.warn("Single-objective refresh failed", error);
    }
    const errors = result.refreshResult?.registryErrors || result.state?.moduleRegistry?.registryErrors || [];
    $("registryStatus").textContent = errors.length
      ? `Refresh completed with ${errors.length} registry warning(s): ${errors.slice(0, 2).join(" ")}`
      : "Registry refreshed. New local Python modules are available in selectors.";
  }),
);

document.querySelectorAll("[data-jump-view]").forEach((button) => {
  button.addEventListener("click", () => {
    state.workspaceView = button.dataset.jumpView || "singleObjective";
    syncWorkspaceView();
    document.querySelector(".workspace")?.scrollTo({ top: 0, behavior: "smooth" });
  });
});

[
  ["benchmarkFilter", "benchmarkSelect"],
  ["optimizerFilter", "optimizerSelect"],
  ["expBenchmarkFilter", "expBenchmarks"],
  ["expOptimizerFilter", "expOptimizers"],
  ["soBenchmarkFilter", "soBenchmarkSelect"],
  ["soMethodFilter", "soMethodSelect"],
  ["soExpBenchmarkFilter", "soExpBenchmarks"],
  ["soExpMethodFilter", "soExpMethods"],
].forEach(([inputId, selectId]) => bindSelectFilter(inputId, selectId));

document.querySelectorAll("[data-select-action]").forEach((button) => {
  button.addEventListener("click", () => applySelectionAction(button));
});

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset || "smoke"));
});

$("copyConfig")?.addEventListener("click", async () => {
  const button = $("copyConfig");
  const original = button.textContent;
  try {
    await copyTextToClipboard(JSON.stringify(currentPlatformConfig(), null, 2));
    button.textContent = "Copied";
  } catch {
    button.textContent = "Copy failed";
  } finally {
    setTimeout(() => {
      button.textContent = original;
    }, 900);
  }
});

$("downloadConfig")?.addEventListener("click", () => {
  downloadBlob("blackboxlab_config.json", JSON.stringify(currentPlatformConfig(), null, 2), "application/json;charset=utf-8");
});

$("downloadHistory")?.addEventListener("click", () => {
  loadRunHistory();
  downloadBlob("blackboxlab_run_history.json", JSON.stringify(state.runHistory, null, 2), "application/json;charset=utf-8");
});

$("clearHistory")?.addEventListener("click", () => {
  state.runHistory = [];
  saveRunHistory();
  renderRunHistory();
});

[
  "populationSize",
  "maxEvaluations",
  "dimensionInput",
  "objectiveCountInput",
  "expPopulation",
  "expBudget",
  "expDimension",
  "expObjectives",
  "expRepeats",
  "expSeed",
  "expWorkers",
  "expMetric",
  "soPopulationSize",
  "soMaxEvaluations",
  "soDimensionInput",
  "soExpPopulation",
  "soExpBudget",
  "soExpDimension",
  "soExpRepeats",
  "soExpSeed",
  "soExpWorkers",
  "soExpMetric",
].forEach((id) => {
  $(id)?.addEventListener("input", renderPlatformTools);
  $(id)?.addEventListener("change", renderPlatformTools);
});

["expBenchmarks", "expOptimizers", "expRepeats"].forEach((id) => {
  $(id)?.addEventListener("change", () => {
    if (id === "expOptimizers") syncExperimentReference("expOptimizers", "expReference");
    updateExperimentScale("multi");
    renderPlatformTools();
  });
});

["soExpBenchmarks", "soExpMethods", "soExpRepeats"].forEach((id) => {
  $(id)?.addEventListener("change", () => {
    if (id === "soExpMethods") syncExperimentReference("soExpMethods", "soExpReference");
    updateExperimentScale("single");
    renderPlatformTools();
  });
});

setCustomProblemExample();
setCustomOptimizerExample();
loadRunHistory();
renderRunHistory();

api("/api/state").then(render).catch((error) => {
  document.body.innerHTML = `<pre>${escapeHtml(error.stack || error.message)}</pre>`;
});

api("/api/single/state").then(renderSingleObjective).catch((error) => {
  console.warn("Single-objective state failed", error);
});

setInterval(async () => {
  if (state.busy || state.runningBudget || state.runningSingleBudget) return;
  try {
    render(await api("/api/state"));
    if (state.workspaceView === "singleObjective") {
      renderSingleObjective(await api("/api/single/state"));
    }
  } catch (error) {
    console.warn("State refresh failed", error);
  }
}, 4000);

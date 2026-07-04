# BlackBoxLab

BlackBoxLab is a local Python-backed testbed for black-box optimization. It supports single-objective optimization, multi-objective optimization, interactive visual testing, batch comparison experiments, custom problem submission, custom algorithm submission, and downloadable reports for research workflows.

The browser frontend is only the display and control layer. Objective evaluation, optimization steps, Pareto bookkeeping, surrogate modeling, experiment loops, statistics, and registry loading run in the Python backend.

## Highlights

- Single-objective and multi-objective test pages.
- Separate experiment pages for batch comparisons.
- Python backend implementations for objective evaluation and optimizer stepping.
- Interactive 1D, 2D, and 3D visualizations for search points, landscapes, convergence curves, objective spaces, and Pareto fronts.
- Exportable CSV, JSON, HTML reports, significance tables, PNG images, SVG images, and GIF animations.
- Custom benchmark and custom optimizer submission from the UI.
- Adjustable population size, evaluation budget, decision dimension, and multi-objective objective count.
- Multi-objective indicators including hypervolume, IGD, IGD+, GD, recommended score, and Pareto count.
- One-click local startup scripts for Windows and shell environments.

## Quick Start

### Windows

Double-click:

```text
start_blackboxlab.bat
```

or run:

```powershell
cd C:\Users\Administrator\Desktop\LILO\pareto_lilo_demo
.\start_blackboxlab.bat
```

The launcher keeps a terminal open while the server is running. If startup fails, it prints the error and writes `server.start.log`.

### Git Bash, WSL, Linux, or macOS

```bash
cd /path/to/pareto_lilo_demo
./start_blackboxlab.sh
```

Choose a port:

```bash
./start_blackboxlab.sh 8766
```

Debug without opening a browser:

```bash
./start_blackboxlab.sh 8765 --no-browser
```

### Manual startup

```bash
python server.py 8765 --open
```

If port `8765` is busy, the backend automatically tries the next free port and prints the actual URL.

## Requirements

- Python 3.10+.
- NumPy.
- A modern browser.

No Node build step is required. The UI is plain HTML, CSS, and JavaScript served by the Python backend.

## Project Structure

```text
pareto_lilo_demo/
  server.py                    # HTTP server, API routes, optimization state, experiments
  start_blackboxlab.bat         # Windows launcher
  start_blackboxlab.ps1         # PowerShell launcher used by the .bat file
  start_blackboxlab.sh          # shell launcher
  benchmarks/                   # multi-objective benchmark registry and problem modules
  methods/                      # multi-objective optimizer registry and method modules
  single_benchmarks/            # single-objective benchmark registry and function modules
  single_methods/               # single-objective optimizer registry and method modules
  static/
    index.html                  # backend UI
    app.js                      # UI logic and visualization
    styles.css                  # application styling
    docs/                       # user docs for custom problems and algorithms
    examples/                   # example custom Python modules
  standalone.html               # older browser-only demo shell
```

## Main Pages

- `Single test`: run one single-objective method on one benchmark and visualize search points, landscapes, and convergence.
- `Single experiments`: compare single-objective methods across benchmarks and repeats.
- `Multi test`: run one multi-objective method on one benchmark and visualize objective trade-offs, Pareto samples, recommendations, and logs.
- `Multi experiments`: compare multi-objective methods with statistical exports.

## Current Registry Coverage

The local registry is module-based. Counts may change when custom files are added.

| Registry | Count |
| --- | ---: |
| Multi-objective benchmark problems | 201 |
| Multi-objective optimization methods | 197 |
| Single-objective benchmark functions | 78 |
| Single-objective optimization methods | 47 |

Verify your local copy with:

```bash
python -c "import server; print(len(server.BENCHMARKS), len(server.OPTIMIZERS), len(server.SINGLE_BENCHMARKS), len(server.SINGLE_METHODS))"
```

## Benchmark Coverage

The multi-objective registry includes:

- Classic two-objective and three-objective problems.
- ZDT and DTLZ-style suites.
- UF-style, CF-style, WFG-style, LZ-style, MaF-style, LSMOP-style, IMOP-style, constrained-DTLZ-style, and engineering-style synthetic families.
- BlackBoxLab `BBM` synthetic families for convex, concave, disconnected, degenerate, sparse, constrained, robust, and knee-point Pareto structures.
- User-submitted custom benchmarks.

All multi-objective values are represented as normalized maximization scores where larger is better. This keeps dominance, hypervolume, recommendation score, and visual color mapping consistent.

The single-objective registry includes:

- BBOB-style and CEC-style scalable functions.
- Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel, Levy, Zakharov, Schaffer F7, Lunacek, Weierstrass, HappyCat, HGBat, Eggholder, Holder Table, and related functions.
- Fixed-dimensional classics such as Beale, Booth, Matyas, McCormick, Branin, Goldstein-Price, Camel, Forrester, Gramacy-Lee, Bohachevsky, Schubert, Easom, and Bukin N.6.
- User-submitted custom functions.

## Optimization Methods

Multi-objective methods include:

- Pareto-aware LILO-BO.
- Classic scalarized BO.
- NSGA-II, MOEA/D, NSGA-III, SPEA2, PAES, GDE3, IBEA, SMS-EMOA, RVEA, MOCell, epsilon-MOEA, AGE-MOEA.
- Archive, grid, decomposition, reference-vector, indicator, swarm, covariance, sparse-variable, constrained, preference, and operator-portfolio strategies.
- Expensive optimization baselines using RBF surrogate proxies, including qNEHVI-style, qParEGO-style, Thompson, entropy, knowledge-gradient, trust-region, EI-style, UCB-style, and max-value strategies.

Single-objective methods include:

- Random search, hill climbing, DE, PSO, GA, CMA-ES-style search, Nelder-Mead-style search, simulated annealing, pattern search, coordinate search, CEM, TPE-style search, and trust-region BO.
- JADE-style DE, current-to-best DE, firefly, bat, harmony search, continuous ACO, SPSA, DIRECT-style search, LHS, opposition search, low-discrepancy search, orthogonal-array search, memetic GA, adaptive PSO, covariance restart ES, Cauchy ES, and RBF surrogate BO variants.

The included methods are dependency-light research demos and experimental scaffolds. They are not intended to replace audited, full-featured optimization libraries.

## Add a Multi-Objective Benchmark

Create a Python file in `benchmarks/` and call `register_benchmark`.

```python
from __future__ import annotations

import numpy as np

from benchmarks.registry import input_defs, objective_defs, register_benchmark


def evaluate_my_problem(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f1 = 1.0 - np.mean((x - 0.2) ** 2)
    f2 = 1.0 - np.mean((x - 0.8) ** 2)
    return np.clip(np.array([f1, f2]), 0.0, 1.0)


register_benchmark(
    "my_problem",
    {
        "name": "My Problem",
        "description": "A custom two-objective benchmark.",
        "minDimension": 2,
        "defaultDimension": 5,
        "maxDimension": 30,
        "minObjectives": 2,
        "defaultObjectives": 2,
        "maxObjectives": 2,
        "inputLabels": input_defs,
        "objectives": objective_defs(2),
        "evaluate": evaluate_my_problem,
    },
)
```

Restart the backend or use the UI refresh/import controls.

## Add a Multi-Objective Optimizer

Create a Python file in `methods/` and call `register_optimizer`.

```python
from __future__ import annotations

import numpy as np

from benchmarks import evaluate_benchmark
from methods.registry import register_optimizer


def run_my_optimizer(demo, batch_size: int) -> dict:
    batch_size = demo.budgeted_batch_size(batch_size)
    if batch_size <= 0:
        return demo.finish_experiment(0, 0, "Budget already reached.")

    x = demo.rng.random((batch_size, demo.input_dimension()))
    y = np.array([evaluate_benchmark(row, demo.state.benchmark) for row in x])
    return demo.apply_round_results(
        method="My Optimizer",
        strategy="random candidate proposal",
        chosen_x=x,
        chosen_y=y,
        predicted=y,
        uncertainty=np.full(batch_size, 0.0),
        scores=np.mean(y, axis=1),
        hv_before=demo.approximate_hypervolume(
            np.array(demo.state.y, dtype=float),
            np.array(demo.state.hv_samples, dtype=float),
        ),
    )


register_optimizer(
    "my_optimizer",
    {
        "name": "My Optimizer",
        "shortName": "MY",
        "description": "Example custom optimizer.",
    },
    run_my_optimizer,
)
```

## Add a Single-Objective Benchmark

Create a Python file in `single_benchmarks/` and call `register_single_benchmark`.

```python
from __future__ import annotations

import numpy as np

from single_benchmarks.registry import register_single_benchmark


def evaluate(x: np.ndarray) -> float:
    z = 10.0 * np.asarray(x, dtype=float) - 5.0
    return float(np.sum(z ** 2))


register_single_benchmark(
    "my_single_problem",
    {
        "name": "My Single Problem",
        "suite": "Custom",
        "description": "A custom single-objective benchmark.",
        "defaultDimension": 5,
        "minDimension": 1,
        "maxDimension": 100,
        "bounds": [-5.0, 5.0],
        "optimum": 0.0,
        "evaluate": evaluate,
    },
)
```

## Add a Single-Objective Optimizer

Create a Python file in `single_methods/` and call `register_single_method`.

```python
from __future__ import annotations

import numpy as np

from single_methods.registry import register_single_method


def run_my_method(demo, batch_size: int):
    best = np.array(demo.state.x, dtype=float)[demo.best_index()]
    x = np.clip(best + demo.rng.normal(0.0, 0.08, size=(batch_size, len(best))), 0.0, 1.0)
    return x, "local Gaussian perturbation"


register_single_method(
    "my_single_method",
    {
        "name": "My Single Method",
        "shortName": "MY-SO",
        "description": "Example custom single-objective method.",
    },
    run_my_method,
)
```

## Experiment Outputs

Experiment pages can export:

- Raw run rows as CSV.
- Summary statistics tables.
- Pairwise significance matrices.
- HTML reports.
- JSON result bundles.
- PNG chart and table images for paper figures.
- GIF animations for selected visualization panels.

## Notes on LILO-BO

`Pareto-aware LILO-BO` combines:

- A lightweight RBF surrogate.
- A surrogate uncertainty proxy.
- Pareto-aware candidate scoring.
- Preference weights parsed from natural-language feedback or manually adjusted in the UI.

The RBF uncertainty is a proxy, not a calibrated Gaussian process posterior. For paper-grade expensive Bayesian optimization, replace this component with a calibrated probabilistic model and acquisition functions appropriate to your experiment.

## Limitations

- The platform is designed for local experimentation and demos.
- Custom submitted Python code is restricted by a defensive parser, but this is not a hardened sandbox for untrusted users.
- Some built-in catalog methods are runnable strategy proxies rather than full literature-grade implementations.
- Benchmark formulas are normalized to maximization scores in the multi-objective registry, so numeric objective values may differ from original minimization definitions.
- Large experiment grids can be slow because all computations run in the local Python process.

## License

Add your intended open-source license before publishing this repository.

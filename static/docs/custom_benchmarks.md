# Custom Benchmark Modules

Create a Python file that calls `register_benchmark(key, definition)`.

Required fields:

- `name`: display name.
- `description`: shown in the UI.
- `objectives`: list from `objective_defs(m)` or your own objective metadata.
- `evaluate`: function receiving `x: np.ndarray` in `[0, 1]^d` and returning normalized maximization objectives in `[0, 1]^m`.

Recommended fields:

- `minDimension`, `defaultDimension`, `maxDimension`.
- `inputLabels`: usually `input_defs`.
- `referenceFront`: optional function `(dimension, n) -> np.ndarray`; used by IGD and IGD+.

Recommended workflow: save the file in `benchmarks/`, click `Refresh local registry`, then select it in the comparison experiment.

You can also paste its absolute local path into `Local benchmark path` and click `Register benchmark path`; the local Python backend loads that file directly for this session.

See `/examples/custom_benchmark_example.py`.

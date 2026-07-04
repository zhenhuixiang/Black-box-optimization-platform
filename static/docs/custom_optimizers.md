# Custom Optimizer Modules

Create a Python file that calls `register_optimizer(key, metadata, runner)`.

The runner signature is:

```python
def run_my_optimizer(demo, batch_size: int) -> dict:
    ...
```

Useful `demo` APIs:

- `demo.state.x`, `demo.state.y`: evaluated decision vectors and objectives.
- `demo.rng`: seeded NumPy random generator.
- `demo.input_dimension()`, `demo.budgeted_batch_size(batch_size)`.
- `demo.pareto_mask(y)`, `demo.approximate_hypervolume(y, samples)`.
- `demo.apply_round_results(...)`: append evaluated candidates and return a UI snapshot.
- `evaluate_benchmark(x, demo.state.benchmark)`: evaluate the active benchmark.

Recommended workflow: save the file in `methods/`, click `Refresh local registry`, then select it in the comparison experiment.

You can also paste its absolute local path into `Local optimizer path` and click `Register optimizer path`; the local Python backend loads that file directly for this session.

See `/examples/custom_optimizer_example.py`.

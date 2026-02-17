# dagsampler

Minimal Python package for generating synthetic data from causal DAGs.

## What it provides

- `CausalDataGenerator` class for configurable simulation
- Support for `custom` and `random` DAGs
- Mixed continuous/binary/categorical nodes (configurable categorical cardinality)
- Linear/polynomial/interaction functional forms
- Cross-type mechanisms:
  - continuous -> categorical (`categorical_model.name = "threshold"`)
  - categorical -> continuous (`functional_form.name = "stratum_means"`)
- Additive/multiplicative/heteroskedastic noise
- Reproducibility via `seed_structure` and `seed_data`
- Optional d-separation CI oracle output (`store_ci_oracle=true`)

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quick start (Python API)

```python
from causal_simulator import CausalDataGenerator

config = {
    "simulation_params": {"n_samples": 200, "seed": 42},
    "graph_params": {
        "type": "custom",
        "nodes": ["X", "Y", "Z1"],
        "edges": [["X", "Z1"], ["Y", "Z1"]],
    },
}

result = CausalDataGenerator(config).simulate()
data = result["data"]
dag = result["dag"]
params = result["parametrization"]
```

## CLI

The package exposes `dagsampler-generate`.

```bash
dagsampler-generate \
  --config config.json \
  --output dataset.csv \
  --params-out params.json \
  --edges-out edges.json
```

`config.json` must contain the same structure used by `CausalDataGenerator`.

For heteroskedastic noise, use `noise_model.func` from:
- `abs_first_parent`
- `abs_parent_plus_const`
- `mean_abs_plus_const`

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
```

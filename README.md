# dagsampler

Minimal Python package for generating synthetic data from causal DAGs.

[Documentation](https://averinpa.github.io/dagsampler/)

## What it provides

- `CausalDataGenerator` class for configurable simulation
- Support for `custom` and `random` DAGs
- Mixed continuous/binary/categorical nodes (configurable categorical cardinality)
- Linear/polynomial/interaction functional forms
- Cross-type mechanisms:
  - continuous -> categorical (`categorical_model.name = "threshold"`)
  - categorical -> continuous (`functional_form.name = "stratum_means"`)
- Additive/multiplicative/heteroskedastic noise
- Random weight sampling controls (including exclusion band around zero)
- Reproducibility via `seed_structure` and `seed_data`
- Optional d-separation CI oracle output (`store_ci_oracle=true`)

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install "dagsampler @ git+https://github.com/averinpa/dagsampler.git"
```

## Random weights away from zero

For CI benchmark settings where near-zero coefficients can cause practical
faithfulness issues, configure:

```json
{
  "simulation_params": {
    "random_weight_low": -1.5,
    "random_weight_high": 1.5,
    "random_weight_min_abs": 0.1
  }
}
```

This samples random structural weights from:
- `[-1.5, -0.1] U [0.1, 1.5]`

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

# simulation

Minimal Python package for generating synthetic data from causal DAGs.

## What it provides

- `CausalDataGenerator` class for configurable simulation
- Support for `fork`, `chain`, `v_structure`, `custom`, and `random` DAGs
- Mixed continuous/binary nodes
- Linear/polynomial/interaction functional forms
- Additive/multiplicative/heteroskedastic noise
- Reproducibility via `seed_structure` and `seed_data`

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
    "graph_params": {"type": "v_structure", "z_size": 1},
}

result = CausalDataGenerator(config).simulate()
data = result["data"]
dag = result["dag"]
params = result["parametrization"]
```

## CLI

The package exposes `simulation-generate`.

```bash
simulation-generate \
  --config config.json \
  --output dataset.csv \
  --params-out params.json \
  --edges-out edges.json
```

`config.json` must contain the same structure used by `CausalDataGenerator`.

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
```

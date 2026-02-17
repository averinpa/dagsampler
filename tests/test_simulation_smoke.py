import pandas as pd

from causal_simulator import CausalDataGenerator


def test_simulation_smoke_outputs_expected_shapes():
    config = {
        "simulation_params": {"n_samples": 25, "seed": 7},
        "graph_params": {"type": "v_structure", "z_size": 1},
    }
    result = CausalDataGenerator(config).simulate()

    assert set(result.keys()) == {"data", "parametrization", "dag"}
    assert result["data"].shape[0] == 25
    assert list(result["data"].columns) == sorted(result["data"].columns)
    assert list(result["dag"].nodes()) == list(result["data"].columns)


def test_simulation_reproducible_with_explicit_seeds():
    config = {
        "simulation_params": {
            "n_samples": 40,
            "seed_structure": 11,
            "seed_data": 12,
            "binary_proportion": 0.0,
        },
        "graph_params": {"type": "v_structure", "z_size": 1},
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "Y": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "Z1": {
                "type": "continuous",
                "functional_form": {"name": "linear", "weights": {"X": 1.0, "Y": 1.0}},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5},
            },
        },
    }

    first = CausalDataGenerator(config).simulate()
    second = CausalDataGenerator(config).simulate()

    pd.testing.assert_frame_equal(first["data"], second["data"])
    assert list(first["dag"].edges()) == list(second["dag"].edges())

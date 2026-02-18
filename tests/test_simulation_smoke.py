import pandas as pd

from causal_simulator import CausalDataGenerator


def test_simulation_smoke_outputs_expected_shapes():
    config = {
        "simulation_params": {"n_samples": 25, "seed": 7},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y", "Z1"],
            "edges": [("X", "Z1"), ("Y", "Z1")],
        },
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
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y", "Z1"],
            "edges": [("X", "Z1"), ("Y", "Z1")],
        },
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


def test_categorical_generation_with_threshold_and_stratum_means():
    config = {
        "simulation_params": {"n_samples": 120, "seed_structure": 5, "seed_data": 6},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "C", "Y"],
            "edges": [("X", "C"), ("C", "Y")],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "C": {
                "type": "categorical",
                "cardinality": 5,
                "categorical_model": {"name": "threshold", "weights": {"X": 1.0}},
            },
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "stratum_means", "default_mean": 0.0},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.05},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    c_values = result["data"]["C"]
    assert c_values.min() >= 0
    assert c_values.max() <= 4
    assert len(c_values.unique()) >= 3


def test_ci_oracle_and_heteroskedastic_legacy_alias():
    config = {
        "simulation_params": {
            "n_samples": 80,
            "seed_structure": 3,
            "seed_data": 4,
            "store_ci_oracle": True,
            "ci_oracle_max_cond_set": 1,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y", "Z1"],
            "edges": [("X", "Z1"), ("Y", "Z1")],
        },
        "node_params": {
            "Z1": {
                "type": "continuous",
                "functional_form": {"name": "linear", "weights": {"X": 1.0, "Y": 1.0}},
                "noise_model": {
                    "name": "heteroskedastic",
                    "func": "lambda p: 0.5 * np.abs(p.iloc[:, 0])",
                },
            }
        },
    }
    result = CausalDataGenerator(config).simulate()
    assert "ci_oracle" in result
    assert len(result["ci_oracle"]) > 0


def test_endogenous_logistic_categorical_node():
    config = {
        "simulation_params": {"n_samples": 150, "seed_structure": 21, "seed_data": 22},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "B", "C"],
            "edges": [("X", "C"), ("B", "C")],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "B": {"type": "binary", "distribution": {"name": "bernoulli", "p": 0.4}},
            "C": {
                "type": "categorical",
                "cardinality": 3,
                "categorical_model": {
                    "name": "logistic",
                    "intercepts": [0.0, 0.0, 0.0],
                    "weights": {"X": [0.8, -0.2, -0.6], "B": [-0.3, 0.7, -0.4]},
                },
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    c_values = result["data"]["C"]
    assert c_values.min() >= 0
    assert c_values.max() <= 2
    assert len(c_values.unique()) >= 2


def test_random_weight_sampling_excludes_near_zero_band():
    config = {
        "simulation_params": {
            "n_samples": 50,
            "seed_structure": 99,
            "seed_data": 100,
            "random_weight_low": -1.5,
            "random_weight_high": 1.5,
            "random_weight_min_abs": 0.1,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X1", "X2", "Y"],
            "edges": [["X1", "Y"], ["X2", "Y"]],
        },
        "node_params": {
            "X1": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "X2": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    weights = result["parametrization"]["node_params"]["Y"]["functional_form"]["weights"]
    for _, weight in weights.items():
        assert abs(float(weight)) >= 0.1
        assert abs(float(weight)) <= 1.5

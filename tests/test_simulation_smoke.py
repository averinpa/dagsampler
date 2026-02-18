import pandas as pd
import pytest

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


def test_metric_functional_form_rejects_categorical_parent_by_default():
    config = {
        "simulation_params": {"n_samples": 30, "seed_structure": 1, "seed_data": 2},
        "graph_params": {
            "type": "custom",
            "nodes": ["C", "Y"],
            "edges": [["C", "Y"]],
        },
        "node_params": {
            "C": {"type": "categorical", "cardinality": 4},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    with pytest.raises(ValueError, match="metric functional form"):
        CausalDataGenerator(config).simulate()


def test_metric_functional_form_can_redirect_to_stratum_means():
    config = {
        "simulation_params": {
            "n_samples": 30,
            "seed_structure": 1,
            "seed_data": 2,
            "categorical_parent_metric_form_policy": "stratum_means",
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["C", "Y"],
            "edges": [["C", "Y"]],
        },
        "node_params": {
            "C": {"type": "categorical", "cardinality": 4},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    strata_means = result["parametrization"]["node_params"]["Y"]["functional_form"]["strata_means"]
    assert len(strata_means) == 4


def test_mixed_parent_redirect_policy_raises_clear_error():
    config = {
        "simulation_params": {
            "n_samples": 30,
            "seed_structure": 1,
            "seed_data": 2,
            "categorical_parent_metric_form_policy": "stratum_means",
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["C", "X", "Y"],
            "edges": [["C", "Y"], ["X", "Y"]],
        },
        "node_params": {
            "C": {"type": "categorical", "cardinality": 3},
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    with pytest.raises(ValueError, match="requires all parents to be categorical"):
        CausalDataGenerator(config).simulate()


def test_threshold_defaults_do_not_depend_on_realized_sample():
    base = {
        "simulation_params": {"n_samples": 100, "seed_structure": 12},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "C"],
            "edges": [["X", "C"]],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "C": {"type": "categorical", "cardinality": 5, "categorical_model": {"name": "threshold"}},
        },
    }
    cfg_a = dict(base)
    cfg_a["simulation_params"] = dict(base["simulation_params"], seed_data=101)
    cfg_b = dict(base)
    cfg_b["simulation_params"] = dict(base["simulation_params"], seed_data=202)

    res_a = CausalDataGenerator(cfg_a).simulate()
    res_b = CausalDataGenerator(cfg_b).simulate()

    th_a = res_a["parametrization"]["node_params"]["C"]["categorical_model"]["thresholds"]
    th_b = res_b["parametrization"]["node_params"]["C"]["categorical_model"]["thresholds"]
    assert th_a == th_b


def test_threshold_scale_default_is_sampled_and_persisted():
    config = {
        "simulation_params": {"n_samples": 80, "seed_structure": 51, "seed_data": 52},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "C"],
            "edges": [["X", "C"]],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "C": {"type": "categorical", "cardinality": 5, "categorical_model": {"name": "threshold"}},
        },
    }
    result = CausalDataGenerator(config).simulate()
    scale = result["parametrization"]["node_params"]["C"]["categorical_model"]["threshold_scale"]
    assert 0.5 <= float(scale) <= 2.0


def test_logistic_random_weights_respect_min_abs_and_are_persisted():
    config = {
        "simulation_params": {
            "n_samples": 60,
            "seed_structure": 7,
            "seed_data": 8,
            "random_weight_min_abs": 0.2,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "D", "C"],
            "edges": [["X", "C"], ["D", "C"]],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "D": {"type": "categorical", "cardinality": 3},
            "C": {
                "type": "categorical",
                "cardinality": 4,
                "categorical_model": {"name": "logistic", "intercepts": [0.0, 0.0, 0.0, 0.0]},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    weights = result["parametrization"]["node_params"]["C"]["categorical_model"]["weights"]
    flat = []
    for value in weights.values():
        if isinstance(value, list) and value and isinstance(value[0], list):
            for row in value:
                flat.extend(row)
        elif isinstance(value, list):
            flat.extend(value)
    assert flat
    assert all(abs(float(w)) >= 0.2 for w in flat)


def test_stratum_means_preenumerates_all_combinations():
    config = {
        "simulation_params": {"n_samples": 20, "seed_structure": 13, "seed_data": 14},
        "graph_params": {
            "type": "custom",
            "nodes": ["C1", "C2", "Y"],
            "edges": [["C1", "Y"], ["C2", "Y"]],
        },
        "node_params": {
            "C1": {"type": "categorical", "cardinality": 3},
            "C2": {"type": "categorical", "cardinality": 2},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "stratum_means", "default_mean": 0.0},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    strata_means = result["parametrization"]["node_params"]["Y"]["functional_form"]["strata_means"]
    assert len(strata_means) == 6

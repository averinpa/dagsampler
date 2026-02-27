import copy

import numpy as np
import pandas as pd
import pytest

from causal_simulator import (
    CausalDataGenerator,
    chain_config,
    collider_config,
    fork_config,
    indep_config,
    independence_config,
)


def _oracle_lookup(oracle, x: str, y: str, conditioning_set: list[str]) -> bool:
    xy = {x, y}
    target_cond = set(conditioning_set)
    for row in oracle:
        if {row["x"], row["y"]} == xy and set(row["conditioning_set"]) == target_cond:
            return bool(row["is_independent"])
    raise AssertionError(f"CI oracle entry missing for ({x}, {y}) | {conditioning_set}")


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


def test_mixed_parent_redirect_policy_supports_stratum_plus_metric_term():
    config = {
        "simulation_params": {
            "n_samples": 60,
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
    result = CausalDataGenerator(config).simulate()
    ff = result["parametrization"]["node_params"]["Y"]["functional_form"]
    assert "strata_means" in ff
    assert "metric_weights" in ff
    assert "X" in ff["metric_weights"]


def test_stratum_means_requires_at_least_one_categorical_parent():
    config = {
        "simulation_params": {"n_samples": 20, "seed_structure": 11, "seed_data": 12},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "stratum_means"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    with pytest.raises(ValueError, match="at least one categorical parent"):
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


def test_random_graph_extreme_sparsity_produces_no_edges():
    config = {
        "simulation_params": {"n_samples": 25, "seed_structure": 5, "seed_data": 6},
        "graph_params": {"type": "random", "n_nodes": 7, "edge_prob": 0.0},
    }
    result = CausalDataGenerator(config).simulate()
    assert len(list(result["dag"].edges())) == 0
    assert result["data"].shape == (25, 7)


def test_high_cardinality_categorical_node_generation():
    config = {
        "simulation_params": {"n_samples": 400, "seed_structure": 15, "seed_data": 16},
        "graph_params": {
            "type": "custom",
            "nodes": ["C"],
            "edges": [],
        },
        "node_params": {
            "C": {"type": "categorical", "cardinality": 20},
        },
    }
    result = CausalDataGenerator(config).simulate()
    vals = result["data"]["C"]
    assert vals.min() >= 0
    assert vals.max() <= 19
    assert len(vals.unique()) >= 10


def test_imbalanced_categorical_proportions_are_respected():
    config = {
        "simulation_params": {"n_samples": 1200, "seed_structure": 31, "seed_data": 32},
        "graph_params": {
            "type": "custom",
            "nodes": ["C"],
            "edges": [],
        },
        "node_params": {
            "C": {
                "type": "categorical",
                "cardinality": 3,
                "distribution": {"probs": [0.95, 0.04, 0.01]},
            }
        },
    }
    result = CausalDataGenerator(config).simulate()
    p0 = (result["data"]["C"] == 0).mean()
    assert p0 > 0.9


def test_force_uniform_marginals_binary_exogenous():
    config = {
        "simulation_params": {
            "n_samples": 10000,
            "seed_structure": 1001,
            "seed_data": 1002,
            "force_uniform_marginals": True,
        },
        "graph_params": {"type": "custom", "nodes": ["B"], "edges": []},
        "node_params": {"B": {"type": "binary"}},
    }
    result = CausalDataGenerator(config).simulate()
    p = float((result["data"]["B"] == 1).mean())
    assert 0.48 <= p <= 0.52


def test_force_uniform_marginals_categorical_exogenous():
    config = {
        "simulation_params": {
            "n_samples": 10000,
            "seed_structure": 2001,
            "seed_data": 2002,
            "force_uniform_marginals": True,
        },
        "graph_params": {"type": "custom", "nodes": ["C"], "edges": []},
        "node_params": {"C": {"type": "categorical", "cardinality": 4}},
    }
    result = CausalDataGenerator(config).simulate()
    proportions = result["data"]["C"].value_counts(normalize=True).reindex([0, 1, 2, 3], fill_value=0.0)
    for p in proportions.tolist():
        assert 0.23 <= float(p) <= 0.27


def test_independence_config_smoke():
    cfg = independence_config(
        var_specs=[
            {"name": "X", "type": "continuous"},
            {"name": "B", "type": "binary"},
            {"name": "C", "type": "categorical", "cardinality": 4},
        ],
        n_samples=120,
        seed=77,
        force_uniform=True,
    )
    result = CausalDataGenerator(cfg).simulate()
    assert result["data"].shape == (120, 3)
    assert len(list(result["dag"].edges())) == 0


def test_chain_config_smoke_continuous_linear():
    cfg = chain_config(
        var_specs=[
            {"name": "A", "type": "continuous"},
            {"name": "B", "type": "continuous"},
            {"name": "Y", "type": "continuous"},
        ],
        mechanism="linear",
        n_samples=80,
        seed={"structure": 31, "data": 32},
    )
    result = CausalDataGenerator(cfg).simulate()
    assert result["data"].shape == (80, 3)
    assert list(result["dag"].edges()) == [("A", "B"), ("B", "Y")]


def test_fork_config_smoke_mixed_stratum_means():
    cfg = fork_config(
        var_specs={
            "root": {"name": "R", "type": "continuous"},
            "left": {"name": "L", "type": "binary"},
            "right": {"name": "Q", "type": "binary"},
        },
        mechanism="stratum_means",
        n_samples=90,
        seed=55,
    )
    result = CausalDataGenerator(cfg).simulate()
    assert result["data"].shape == (90, 3)
    assert len(list(result["dag"].edges())) == 2


def test_collider_config_smoke_continuous_parents_categorical_collider():
    cfg = collider_config(
        var_specs={
            "left": {"name": "X", "type": "continuous"},
            "right": {"name": "Y", "type": "continuous"},
            "collider": {"name": "C", "type": "categorical", "cardinality": 5},
        },
        mechanism="linear",
        n_samples=110,
        seed=88,
    )
    result = CausalDataGenerator(cfg).simulate()
    assert result["data"].shape == (110, 3)
    assert result["data"]["C"].min() >= 0
    assert result["data"]["C"].max() <= 4


def test_polynomial_functional_form():
    config = {
        "simulation_params": {"n_samples": 120, "seed_structure": 101, "seed_data": 102},
        "graph_params": {"type": "custom", "nodes": ["X", "Y"], "edges": [["X", "Y"]]},
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 1.0}},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "polynomial"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    degrees = result["parametrization"]["node_params"]["Y"]["functional_form"]["degrees"]
    assert 2 <= int(degrees["X"]) <= 4
    assert np.isfinite(result["data"]["Y"].to_numpy()).all()


def test_interaction_functional_form():
    config = {
        "simulation_params": {"n_samples": 90, "seed_structure": 111, "seed_data": 112},
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
                "functional_form": {"name": "interaction"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.1},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    weights = result["parametrization"]["node_params"]["Y"]["functional_form"]["weights"]
    assert "interaction" in weights
    assert np.isfinite(result["data"]["Y"].to_numpy()).all()


def test_exogenous_student_t():
    config = {
        "simulation_params": {"n_samples": 80, "seed_structure": 121, "seed_data": 122},
        "graph_params": {"type": "custom", "nodes": ["X"], "edges": []},
        "node_params": {"X": {"type": "continuous", "distribution": {"name": "student_t"}}},
    }
    result = CausalDataGenerator(config).simulate()
    dist = result["parametrization"]["node_params"]["X"]["distribution"]
    assert "df" in dist


def test_exogenous_gamma():
    config = {
        "simulation_params": {"n_samples": 80, "seed_structure": 131, "seed_data": 132},
        "graph_params": {"type": "custom", "nodes": ["X"], "edges": []},
        "node_params": {"X": {"type": "continuous", "distribution": {"name": "gamma"}}},
    }
    result = CausalDataGenerator(config).simulate()
    assert np.isfinite(result["data"]["X"].to_numpy()).all()


def test_exogenous_exponential():
    config = {
        "simulation_params": {"n_samples": 80, "seed_structure": 141, "seed_data": 142},
        "graph_params": {"type": "custom", "nodes": ["X"], "edges": []},
        "node_params": {"X": {"type": "continuous", "distribution": {"name": "exponential"}}},
    }
    result = CausalDataGenerator(config).simulate()
    assert np.isfinite(result["data"]["X"].to_numpy()).all()


def test_ci_oracle_chain_d_separation():
    config = {
        "simulation_params": {
            "n_samples": 40,
            "seed_structure": 201,
            "seed_data": 202,
            "store_ci_oracle": True,
            "ci_oracle_max_cond_set": 1,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Z", "Y"],
            "edges": [["X", "Z"], ["Z", "Y"]],
        },
    }
    result = CausalDataGenerator(config).simulate()
    oracle = result["ci_oracle"]
    assert _oracle_lookup(oracle, "X", "Y", ["Z"]) is True
    assert _oracle_lookup(oracle, "X", "Y", []) is False


def test_ci_oracle_collider_d_separation():
    config = {
        "simulation_params": {
            "n_samples": 40,
            "seed_structure": 211,
            "seed_data": 212,
            "store_ci_oracle": True,
            "ci_oracle_max_cond_set": 1,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Z", "Y"],
            "edges": [["X", "Z"], ["Y", "Z"]],
        },
    }
    result = CausalDataGenerator(config).simulate()
    oracle = result["ci_oracle"]
    assert _oracle_lookup(oracle, "X", "Y", []) is True
    assert _oracle_lookup(oracle, "X", "Y", ["Z"]) is False


def test_ci_oracle_fork_d_separation():
    config = {
        "simulation_params": {
            "n_samples": 40,
            "seed_structure": 221,
            "seed_data": 222,
            "store_ci_oracle": True,
            "ci_oracle_max_cond_set": 1,
        },
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Z", "Y"],
            "edges": [["Z", "X"], ["Z", "Y"]],
        },
    }
    result = CausalDataGenerator(config).simulate()
    oracle = result["ci_oracle"]
    assert _oracle_lookup(oracle, "X", "Y", ["Z"]) is True
    assert _oracle_lookup(oracle, "X", "Y", []) is False


def test_parametrization_reuse_gives_identical_data():
    config = {
        "simulation_params": {"n_samples": 100, "seed_structure": 301, "seed_data": 302},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y", "Z"],
            "edges": [["X", "Z"], ["Y", "Z"]],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.1, "std": 1.1}},
            "Y": {"type": "continuous", "distribution": {"name": "student_t", "df": 6}},
            "Z": {
                "type": "continuous",
                "functional_form": {"name": "polynomial"},
                "noise_model": {"name": "additive", "dist": "gamma"},
            },
        },
    }
    first = CausalDataGenerator(config).simulate()
    replay_config = copy.deepcopy(first["parametrization"])
    replay_config.setdefault("simulation_params", {})
    replay_config["simulation_params"]["seed_data"] = first["parametrization"]["simulation_params"]["seed_data"]
    second = CausalDataGenerator(replay_config).simulate()
    pd.testing.assert_frame_equal(first["data"], second["data"])


def test_force_uniform_does_not_override_explicit_p():
    config = {
        "simulation_params": {
            "n_samples": 10000,
            "seed_structure": 401,
            "seed_data": 402,
            "force_uniform_marginals": True,
        },
        "graph_params": {"type": "custom", "nodes": ["B"], "edges": []},
        "node_params": {
            "B": {"type": "binary", "distribution": {"name": "bernoulli", "p": 0.3}},
        },
    }
    result = CausalDataGenerator(config).simulate()
    p_hat = float((result["data"]["B"] == 1).mean())
    assert 0.27 <= p_hat <= 0.33


def test_sigmoid_functional_form():
    config = {
        "simulation_params": {"n_samples": 150, "seed_structure": 501, "seed_data": 502},
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
                "functional_form": {"name": "sigmoid"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.05},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    ff = result["parametrization"]["node_params"]["Y"]["functional_form"]
    assert "weights" in ff
    assert "output_weight" in ff
    assert np.isfinite(result["data"]["Y"].to_numpy()).all()


def test_indep_config_force_uniform_marginals_binary():
    cfg = indep_config(
        n_vars=6,
        n_samples=200,
        seed=777,
        force_uniform_marginals=True,
    )
    result = CausalDataGenerator(cfg).simulate()
    data = result["data"]
    for col in data.columns:
        assert abs(float(data[col].mean()) - 0.5) <= 0.05


def test_force_uniform_categorical_balanced_small_sample():
    config = {
        "simulation_params": {
            "n_samples": 11,
            "seed_structure": 601,
            "seed_data": 602,
            "force_uniform_marginals": True,
        },
        "graph_params": {"type": "custom", "nodes": ["C"], "edges": []},
        "node_params": {"C": {"type": "categorical", "cardinality": 2}},
    }
    result = CausalDataGenerator(config).simulate()
    counts = result["data"]["C"].value_counts().to_dict()
    assert set(counts.keys()) <= {0, 1}
    assert abs(counts.get(0, 0) - counts.get(1, 0)) <= 1


# ── Post-nonlinear transform tests ───────────────────────────────


def test_post_transform_tanh():
    """tanh post-transform should produce values in [-1, 1]."""
    config = {
        "simulation_params": {"n_samples": 500, "seed_structure": 900, "seed_data": 901},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 2.0}},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear", "weights": {"X": 1.5}},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0},
                "post_transform": {"name": "tanh"},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    y = result["data"]["Y"].to_numpy()
    assert np.all(y >= -1.0) and np.all(y <= 1.0)
    assert np.isfinite(y).all()


def test_post_transform_none_backward_compat():
    """Without post_transform, values can exceed [-1, 1] (no squashing)."""
    config = {
        "simulation_params": {"n_samples": 500, "seed_structure": 910, "seed_data": 911},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
        },
        "node_params": {
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 2.0}},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear", "weights": {"X": 1.5}},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    y = result["data"]["Y"].to_numpy()
    # With std=2 input and weight=1.5, values should easily exceed [-1, 1]
    assert np.max(np.abs(y)) > 1.0


def test_post_transform_persisted_in_parametrization():
    """Transform name should appear in final_parametrization."""
    config = {
        "simulation_params": {"n_samples": 50, "seed_structure": 920, "seed_data": 921},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
        },
        "node_params": {
            "X": {"type": "continuous"},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5},
                "post_transform": {"name": "sin"},
            },
        },
    }
    result = CausalDataGenerator(config).simulate()
    pt = result["parametrization"]["node_params"]["Y"]["post_transform"]["name"]
    assert pt == "sin"


def test_post_transform_unknown_raises():
    """Unknown transform name should raise ValueError."""
    config = {
        "simulation_params": {"n_samples": 50, "seed_structure": 930, "seed_data": 931},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y"],
            "edges": [["X", "Y"]],
        },
        "node_params": {
            "X": {"type": "continuous"},
            "Y": {
                "type": "continuous",
                "functional_form": {"name": "linear"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 0.5},
                "post_transform": {"name": "unknown_transform"},
            },
        },
    }
    with pytest.raises(ValueError, match="Unknown post_transform"):
        CausalDataGenerator(config).simulate()


def test_post_transform_via_default_endogenous():
    """post_transform in default_endogenous should apply to all endogenous nodes."""
    config = {
        "simulation_params": {"n_samples": 500, "seed_structure": 940, "seed_data": 941},
        "graph_params": {
            "type": "custom",
            "nodes": ["X", "Y", "Z"],
            "edges": [["X", "Y"], ["X", "Z"]],
        },
        "node_params": {
            "default_endogenous": {
                "functional_form": {"name": "linear"},
                "noise_model": {"name": "additive", "dist": "gaussian", "std": 1.0},
                "post_transform": {"name": "tanh"},
            },
            "X": {"type": "continuous", "distribution": {"name": "gaussian", "mean": 0.0, "std": 2.0}},
            "Y": {"type": "continuous"},
            "Z": {"type": "continuous"},
        },
    }
    result = CausalDataGenerator(config).simulate()
    for node in ["Y", "Z"]:
        vals = result["data"][node].to_numpy()
        assert np.all(vals >= -1.0) and np.all(vals <= 1.0)


def test_post_transform_chain_template():
    """post_transform kwarg in chain_config should work."""
    cfg = chain_config(
        var_specs=[
            {"name": "X", "type": "continuous"},
            {"name": "Y", "type": "continuous"},
        ],
        mechanism="linear",
        n_samples=500,
        seed=950,
        post_transform="tanh",
    )
    result = CausalDataGenerator(cfg).simulate()
    y = result["data"]["Y"].to_numpy()
    assert np.all(y >= -1.0) and np.all(y <= 1.0)

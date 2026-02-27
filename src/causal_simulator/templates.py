"""Config template helpers for common DAG structures."""

from __future__ import annotations

from typing import Any, TypeAlias

VarSpec: TypeAlias = dict[str, Any]
SeedSpec: TypeAlias = int | dict[str, int] | None
MechanismName: TypeAlias = str


def _seed_params(seed: SeedSpec) -> dict[str, int]:
    if seed is None:
        return {}
    if isinstance(seed, int):
        return {"seed_structure": seed, "seed_data": seed}
    if isinstance(seed, dict):
        structure = seed.get("structure")
        data = seed.get("data")
        params: dict[str, int] = {}
        if structure is not None:
            params["seed_structure"] = int(structure)
        if data is not None:
            params["seed_data"] = int(data)
        return params
    raise ValueError("seed must be int, dict, or None")


def _normalize_var_spec(spec: VarSpec) -> VarSpec:
    if "name" not in spec or "type" not in spec:
        raise ValueError("each var spec must include 'name' and 'type'")
    var_type = str(spec["type"])
    if var_type not in {"continuous", "binary", "categorical"}:
        raise ValueError(f"unsupported node type: {var_type}")
    out = {"name": str(spec["name"]), "type": var_type}
    if var_type == "categorical":
        out["cardinality"] = int(spec.get("cardinality", 3))
    return out


def _base_config(n_samples: int, seed: SeedSpec) -> dict[str, Any]:
    return {
        "simulation_params": {
            "n_samples": int(n_samples),
            **_seed_params(seed),
        }
    }


def _exogenous_node_params(spec: VarSpec) -> dict[str, Any]:
    node_cfg: dict[str, Any] = {"type": spec["type"]}
    if spec["type"] == "categorical":
        node_cfg["cardinality"] = spec["cardinality"]
    return node_cfg


def _endogenous_node_params(
    spec: VarSpec,
    parents: list[VarSpec],
    mechanism: MechanismName,
    post_transform: str | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {"type": spec["type"]}
    if spec["type"] == "categorical":
        cfg["cardinality"] = spec["cardinality"]
        cfg["categorical_model"] = {"name": "logistic"}
        return cfg

    has_categorical_parent = any(p["type"] == "categorical" for p in parents)
    # stratum_means requires at least one categorical parent in the simulator.
    # Fallback to linear when caller requests stratum_means without one.
    if mechanism == "stratum_means" and has_categorical_parent:
        cfg["functional_form"] = {"name": "stratum_means"}
    elif mechanism in {"linear", "stratum_means", "sigmoid"}:
        form_name = "sigmoid" if mechanism == "sigmoid" else "linear"
        cfg["functional_form"] = {"name": form_name}
    else:
        raise ValueError("mechanism must be 'linear', 'sigmoid', or 'stratum_means'")

    if spec["type"] == "continuous":
        cfg["noise_model"] = {"name": "additive", "dist": "gaussian", "std": 0.5}
    elif spec["type"] == "binary":
        cfg["noise_model"] = {"name": "additive", "dist": "gaussian", "std": 0.5}

    if post_transform is not None and spec["type"] == "continuous":
        cfg["post_transform"] = {"name": post_transform}

    return cfg


def indep_config(
    var_specs: list[VarSpec] | None = None,
    n_samples: int = 100,
    seed: SeedSpec = None,
    force_uniform: bool = True,
    force_uniform_marginals: bool | None = None,
    n_vars: int | None = None,
    node_type: str = "binary",
    cardinality: int = 2,
    prefix: str = "X",
) -> dict[str, Any]:
    """
    Backward-compatible shorthand for independent-node configurations.

    If ``var_specs`` is omitted, generate ``n_vars`` variables of ``node_type``
    (default: binary) named ``{prefix}0..{prefix}{n_vars-1}``.
    """
    if force_uniform_marginals is not None:
        force_uniform = bool(force_uniform_marginals)
    if var_specs is None:
        count = int(n_vars if n_vars is not None else 1)
        if count <= 0:
            raise ValueError("n_vars must be positive when var_specs is not provided")
        var_specs = []
        for i in range(count):
            spec: VarSpec = {"name": f"{prefix}{i}", "type": node_type}
            if node_type == "categorical":
                spec["cardinality"] = int(cardinality)
            var_specs.append(spec)
    return independence_config(
        var_specs=var_specs,
        n_samples=n_samples,
        seed=seed,
        force_uniform=force_uniform,
    )


def independence_config(
    var_specs: list[VarSpec],
    n_samples: int,
    seed: SeedSpec = None,
    force_uniform: bool = True,
) -> dict[str, Any]:
    """
    Config with no edges. All nodes are exogenous.

    var_specs: list of dicts, each with "name" and "type"
               ("continuous", "binary", or "categorical").
               Categorical specs may include "cardinality" (int, default 3).

    seed: int (sets both seed_structure and seed_data) or
          dict {"structure": int, "data": int}.

    force_uniform: passed as force_uniform_marginals in simulation_params.
    """
    specs = [_normalize_var_spec(v) for v in var_specs]
    config = _base_config(n_samples=n_samples, seed=seed)
    config["simulation_params"]["force_uniform_marginals"] = bool(force_uniform)
    config["graph_params"] = {
        "type": "custom",
        "nodes": [s["name"] for s in specs],
        "edges": [],
    }
    config["node_params"] = {s["name"]: _exogenous_node_params(s) for s in specs}
    return config


def chain_config(
    var_specs: list[VarSpec],
    mechanism: MechanismName,
    n_samples: int,
    seed: SeedSpec = None,
    post_transform: str | None = None,
) -> dict[str, Any]:
    """
    Chain: var_specs[0] -> var_specs[1] -> ... -> var_specs[-1].
    var_specs: ordered list of {"name", "type", optionally "cardinality"}.
    mechanism: "linear" (all-continuous/binary parents) or
               "stratum_means" (when any parent is categorical).
    post_transform: optional name of a post-nonlinear transform (e.g. "tanh").
    Root node is exogenous. All others are endogenous with additive
    Gaussian noise (std=0.5) for continuous nodes; no noise for categorical.
    For categorical endogenous nodes, use "logistic_softmax" regardless of
    the mechanism argument.
    """
    specs = [_normalize_var_spec(v) for v in var_specs]
    if len(specs) < 2:
        raise ValueError("chain_config requires at least 2 variables")
    edges = [[specs[i]["name"], specs[i + 1]["name"]] for i in range(len(specs) - 1)]

    config = _base_config(n_samples=n_samples, seed=seed)
    config["graph_params"] = {
        "type": "custom",
        "nodes": [s["name"] for s in specs],
        "edges": edges,
    }

    node_params: dict[str, Any] = {}
    node_params[specs[0]["name"]] = _exogenous_node_params(specs[0])
    for idx in range(1, len(specs)):
        node_params[specs[idx]["name"]] = _endogenous_node_params(
            specs[idx], [specs[idx - 1]], mechanism, post_transform=post_transform
        )
    config["node_params"] = node_params
    return config


def fork_config(
    var_specs: dict[str, VarSpec],
    mechanism: MechanismName,
    n_samples: int,
    seed: SeedSpec = None,
    post_transform: str | None = None,
) -> dict[str, Any]:
    """
    Fork: root -> left, root -> right.
    var_specs: dict with keys "root", "left", "right" — each a node spec
               dict {"name", "type", optionally "cardinality"}.
    mechanism: same rules as chain_config.
    post_transform: optional name of a post-nonlinear transform (e.g. "tanh").
    root is exogenous; left and right are endogenous.
    """
    root = _normalize_var_spec(var_specs["root"])
    left = _normalize_var_spec(var_specs["left"])
    right = _normalize_var_spec(var_specs["right"])
    specs = [root, left, right]

    config = _base_config(n_samples=n_samples, seed=seed)
    config["graph_params"] = {
        "type": "custom",
        "nodes": [s["name"] for s in specs],
        "edges": [[root["name"], left["name"]], [root["name"], right["name"]]],
    }
    config["node_params"] = {
        root["name"]: _exogenous_node_params(root),
        left["name"]: _endogenous_node_params(left, [root], mechanism, post_transform=post_transform),
        right["name"]: _endogenous_node_params(right, [root], mechanism, post_transform=post_transform),
    }
    return config


def collider_config(
    var_specs: dict[str, VarSpec],
    mechanism: MechanismName,
    n_samples: int,
    seed: SeedSpec = None,
    post_transform: str | None = None,
) -> dict[str, Any]:
    """
    Collider: left -> collider, right -> collider.
    var_specs: dict with keys "left", "right", "collider" — each a node spec dict.
    mechanism: same rules as chain_config.
    post_transform: optional name of a post-nonlinear transform (e.g. "tanh").
    left and right are exogenous; collider is endogenous with two parents.
    """
    left = _normalize_var_spec(var_specs["left"])
    right = _normalize_var_spec(var_specs["right"])
    collider = _normalize_var_spec(var_specs["collider"])
    specs = [left, right, collider]

    config = _base_config(n_samples=n_samples, seed=seed)
    config["graph_params"] = {
        "type": "custom",
        "nodes": [s["name"] for s in specs],
        "edges": [[left["name"], collider["name"]], [right["name"], collider["name"]]],
    }
    config["node_params"] = {
        left["name"]: _exogenous_node_params(left),
        right["name"]: _exogenous_node_params(right),
        collider["name"]: _endogenous_node_params(collider, [left, right], mechanism, post_transform=post_transform),
    }
    return config

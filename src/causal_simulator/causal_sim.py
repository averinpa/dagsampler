import itertools
import copy
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.d_separation import is_d_separator
from scipy.special import expit
from scipy.stats import norm

# Safety guards to avoid exploding values that can cause Inf/NaN
SAFE_PARENT_CLIP = 1e3
SAFE_OUTPUT_CLIP = 1e12


def _heteroskedastic_abs_first_parent(parent_data: pd.DataFrame) -> np.ndarray:
    return 0.5 * np.abs(parent_data.iloc[:, 0].to_numpy())


def _heteroskedastic_abs_named_plus_const(parent_data: pd.DataFrame) -> np.ndarray:
    if "X" in parent_data.columns:
        return 0.5 * np.abs(parent_data["X"].to_numpy()) + 0.1
    if "Exo_StudentT" in parent_data.columns:
        return 0.5 * np.abs(parent_data["Exo_StudentT"].to_numpy()) + 0.1
    return 0.5 * np.abs(parent_data.iloc[:, 0].to_numpy()) + 0.1


def _heteroskedastic_mean_abs_plus_const(parent_data: pd.DataFrame) -> np.ndarray:
    return 0.5 * np.mean(np.abs(parent_data.to_numpy()), axis=1) + 0.1


HETEROSKEDASTIC_FN_REGISTRY = {
    "abs_first_parent": _heteroskedastic_abs_first_parent,
    "abs_parent_plus_const": _heteroskedastic_abs_named_plus_const,
    "mean_abs_plus_const": _heteroskedastic_mean_abs_plus_const,
}

LEGACY_HETEROSKEDASTIC_FUNC_ALIASES = {
    "lambda p: 0.5 * np.abs(p.iloc[:, 0])": "abs_first_parent",
    "lambda p: 0.5 * np.abs(p['Exo_StudentT']) + 0.1": "abs_parent_plus_const",
    "lambda p: 0.5 * np.abs(p[\"X\"]) + 0.1": "abs_parent_plus_const",
    "lambda p: 0.5 * np.mean(np.abs(p.values), axis=1) + 0.1": "mean_abs_plus_const",
}

class CausalDataGenerator:
    """
    Generates data from a causal graph based on a detailed configuration.

    This class is designed to be flexible, allowing for the specification
    of graph structures, variable types, functional relationships, and noise
    distributions through a single configuration object. It supports both
    random parameter generation and hardcoded parameters for reproducible experiments.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initializes the simulator with a configuration object.

        Args:
            config (dict): A dictionary specifying the entire simulation setup.
        """
        self.config = config
        self.simulation_params = self.config.get('simulation_params', {})
        self.graph_params = self.config.get('graph_params', {})
        self.node_params = self.config.get('node_params', {})

        # Initialize random number generators for reproducibility
        # Main seed (kept for backward compatibility if provided)
        self.seed = self.simulation_params.get('seed')
        # Derive or respect dedicated seeds for structure/params and data draws
        self.seed_structure = self.simulation_params.get(
            'seed_structure', self.seed
        )
        # If data seed not provided, derive a simple offset when main seed exists
        default_data_seed = self.seed + 1 if isinstance(self.seed, (int, np.integer)) else None
        self.seed_data = self.simulation_params.get('seed_data', default_data_seed)

        # Two independent RNG streams: one for structure/params, one for actual data draws
        self.rng_structure = np.random.default_rng(self.seed_structure)
        self.rng_data = np.random.default_rng(self.seed_data)

        self.graph = None
        self.data = None
        self.node_types = {}
        self.node_cardinalities = {}
        # This will store the exact parameters used, including sampled ones.
        self.final_parametrization = copy.deepcopy(self.config)
        if 'nodes' not in self.final_parametrization:
            self.final_parametrization['nodes'] = {}
        # Persist used seeds for reproducibility
        self.final_parametrization.setdefault('simulation_params', {})
        self.final_parametrization['simulation_params']['seed'] = self.seed
        self.final_parametrization['simulation_params']['seed_structure'] = self.seed_structure
        self.final_parametrization['simulation_params']['seed_data'] = self.seed_data

    def _sanitize_series(self, series: pd.Series) -> pd.Series:
        values = series.to_numpy(copy=False)
        values = np.nan_to_num(values, posinf=SAFE_OUTPUT_CLIP, neginf=-SAFE_OUTPUT_CLIP)
        values = np.clip(values, -SAFE_OUTPUT_CLIP, SAFE_OUTPUT_CLIP)
        return pd.Series(values, index=series.index)

    def _node_type(self, node: str) -> str:
        return self.node_types.get(node, "continuous")

    def _node_cardinality(self, node: str) -> int:
        return int(self.node_cardinalities.get(node, 2))

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray:
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _sample_random_weight(self) -> float:
        """
        Samples a random weight with optional exclusion band around zero.

        Controlled via simulation_params:
            - random_weight_low (default -1.5)
            - random_weight_high (default 1.5)
            - random_weight_min_abs (default 0.0)
        """
        low = float(self.simulation_params.get("random_weight_low", -1.5))
        high = float(self.simulation_params.get("random_weight_high", 1.5))
        min_abs = float(self.simulation_params.get("random_weight_min_abs", 0.0))

        if low >= high:
            raise ValueError("random_weight_low must be strictly less than random_weight_high")
        if min_abs < 0:
            raise ValueError("random_weight_min_abs must be non-negative")

        if min_abs == 0:
            return float(self.rng_structure.uniform(low, high))

        left_low, left_high = low, min(-min_abs, high)
        right_low, right_high = max(min_abs, low), high

        left_len = max(0.0, left_high - left_low)
        right_len = max(0.0, right_high - right_low)
        total = left_len + right_len

        if total <= 0:
            raise ValueError(
                "No valid interval remains after applying random_weight_min_abs. "
                "Adjust random_weight_low/high/min_abs."
            )

        if self.rng_structure.random() < (left_len / total):
            return float(self.rng_structure.uniform(left_low, left_high))
        return float(self.rng_structure.uniform(right_low, right_high))

    def _sample_logistic_weight_array(self, shape, std: float) -> np.ndarray:
        """
        Samples logistic-model weights with optional near-zero exclusion.

        If random_weight_min_abs == 0, uses N(0, std^2).
        If random_weight_min_abs > 0, resamples values with |w| < min_abs.
        """
        min_abs = float(self.simulation_params.get("random_weight_min_abs", 0.0))
        arr = self.rng_structure.normal(0.0, std, size=shape)
        if min_abs <= 0:
            return arr

        mask = np.abs(arr) < min_abs
        attempts = 0
        while np.any(mask):
            attempts += 1
            if attempts > 100:
                # Guaranteed fallback under strict exclusion constraints
                arr[mask] = np.array([self._sample_random_weight() for _ in range(mask.sum())], dtype=float)
                break
            arr[mask] = self.rng_structure.normal(0.0, std, size=mask.sum())
            mask = np.abs(arr) < min_abs
        return arr

    def _get_param(self, path: list, default_sampler: callable, node_type: str = None):
        """
        Gets a parameter from the config. If not found, it tries a default path 
        (for generalized configs) before finally sampling it.
        
        Args:
            path (list): The specific path for the parameter (e.g., ['node_params', 'N1', 'type']).
            default_sampler (callable): A function to sample a value if nothing is found.
            node_type (str, optional): 'exogenous' or 'endogenous' to check for default paths.
        """
        # 1. Try the most specific path first (e.g., node_params.N1.type)
        current_level = self.config
        found = True
        for key in path:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                found = False
                break
        
        if found and (current_level or current_level == 0): # Check for non-empty dict or value
             value = current_level
        else:
            # 2. If not found, try the generalized default path (e.g., node_params.default_endogenous.type)
            if node_type:
                default_path = [path[0], f'default_{node_type}'] + path[2:]
                current_level = self.config
                found_default = True
                for key in default_path:
                    if isinstance(current_level, dict) and key in current_level:
                        current_level = current_level[key]
                    else:
                        found_default = False
                        break
                if found_default and (current_level or current_level == 0):
                    value = current_level
                else:
                    # 3. If still not found, sample the value
                    value = default_sampler()
            else:
                 value = default_sampler()


        # Store the used value back into the final parametrization for full reproducibility
        param_storage = self.final_parametrization
        for key in path[:-1]:
            param_storage = param_storage.setdefault(key, {})
        param_storage[path[-1]] = value
        
        return value

    # --- Graph Generation ---

    def _create_graph(self):
        """Creates the networkx.DiGraph based on 'graph_params' in the config."""
        graph_type = self.graph_params.get('type', 'random')
        self.graph = nx.DiGraph()

        # If explicit nodes/edges are provided, prefer them regardless of type for reproducibility
        provided_edges = self.graph_params.get('edges')
        provided_nodes = self.graph_params.get('nodes')
        if provided_edges is not None and len(provided_edges) > 0:
            self.graph = nx.DiGraph()
            if provided_nodes is not None:
                self.graph.add_nodes_from(provided_nodes)
            self.graph.add_edges_from(provided_edges)
        else:
            if graph_type == 'custom':
                nodes = self.graph_params.get('nodes', [])
                edges = self.graph_params.get('edges', [])
                self.graph = nx.DiGraph(edges)
                self.graph.add_nodes_from(nodes)
            elif graph_type == 'random':
                n_nodes = self.graph_params.get('n_nodes', 5)
                edge_prob = self.graph_params.get('edge_prob', 0.3)
                nodes = [f'N{i}' for i in range(n_nodes)]
                self.graph.add_nodes_from(nodes)
                for u, v in itertools.combinations(nodes, 2):
                    if self.rng_structure.random() < edge_prob:
                        # To ensure DAG, always add edges in one direction
                        self.graph.add_edge(u, v)
            else:
                raise ValueError(
                    f"Unsupported graph type '{graph_type}'. Use 'custom' or 'random'."
                )

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The specified graph is not a DAG.")

        # Persist exact nodes/edges used to enable exact reproducibility later
        self.final_parametrization.setdefault('graph_params', {})
        self.final_parametrization['graph_params']['nodes'] = list(self.graph.nodes())
        self.final_parametrization['graph_params']['edges'] = list(self.graph.edges())

    # --- Data Simulation Engine ---

    def simulate(self) -> dict[str, Any]:
        """Main public method to run the full simulation."""
        self._create_graph()

        n_samples = self.simulation_params.get('n_samples', 100)
        self.data = pd.DataFrame(index=range(n_samples))
        
        for node in nx.topological_sort(self.graph):
            self._generate_node_data(node)
        
        # Sort columns alphabetically to ensure a canonical order for downstream processing.
        # This prevents discrepancies between the data column order and the graph node order.
        sorted_nodes = sorted(self.data.columns)
        self.data = self.data[sorted_nodes]

        # Re-create graph to match the canonical node order of the dataframe.
        ordered_graph = nx.DiGraph()
        ordered_graph.add_nodes_from(sorted_nodes)
        ordered_graph.add_edges_from(self.graph.edges)
        self.graph = ordered_graph
        self.final_parametrization["nodes"] = {
            node: {
                "type": self._node_type(node),
                "cardinality": self._node_cardinality(node),
            }
            for node in sorted_nodes
        }

        result = {
            "data": self.data,
            "parametrization": self.final_parametrization,
            "dag": self.graph
        }
        if self.simulation_params.get("store_ci_oracle", False):
            max_cond_set_size = self.simulation_params.get("ci_oracle_max_cond_set", 2)
            ci_oracle = self._build_ci_oracle(max_cond_set_size=max_cond_set_size)
            result["ci_oracle"] = ci_oracle
            self.final_parametrization["ci_oracle"] = ci_oracle

        return result

    def _generate_node_data(self, node: str):
        """
        Determines a node's type (binary, continuous, or categorical) and dispatches to the
        appropriate generation function.
        """
        parents = list(self.graph.predecessors(node))

        # Determine node type from config or sample from proportions.
        binary_proportion = self.simulation_params.get('binary_proportion', 0.4)
        categorical_proportion = self.simulation_params.get('categorical_proportion', 0.0)
        continuous_proportion = max(0.0, 1.0 - binary_proportion - categorical_proportion)
        prob_sum = binary_proportion + categorical_proportion + continuous_proportion
        probs = np.array(
            [binary_proportion, continuous_proportion, categorical_proportion],
            dtype=float
        )
        probs = probs / (prob_sum if prob_sum > 0 else 1.0)
        node_type_str = self._get_param(
            ['node_params', node, 'type'],
            lambda: self.rng_structure.choice(["binary", "continuous", "categorical"], p=probs),
            node_type='endogenous' if parents else 'exogenous'
        )
        self.node_types[node] = node_type_str

        if node_type_str == "categorical":
            cardinality = self._get_param(
                ['node_params', node, 'cardinality'],
                lambda: int(self.rng_structure.choice([2, 3, 5, 10, 20])),
                node_type='endogenous' if parents else 'exogenous'
            )
            self.node_cardinalities[node] = int(cardinality)
        elif node_type_str == "binary":
            self.node_cardinalities[node] = 2
        else:
            self.node_cardinalities[node] = 0

        if not parents:
            if node_type_str == 'continuous':
                self._generate_exogenous_continuous(node)
            elif node_type_str == 'binary':
                self._generate_exogenous_binary(node)
            elif node_type_str == 'categorical':
                self._generate_exogenous_categorical(node)
            else:
                raise ValueError(f"Unknown node type '{node_type_str}' for node '{node}'")
        else:
            if node_type_str == 'continuous':
                self._generate_endogenous_continuous(node, parents)
            elif node_type_str == 'binary':
                self._generate_endogenous_binary(node, parents)
            elif node_type_str == 'categorical':
                self._generate_endogenous_categorical(node, parents)
            else:
                raise ValueError(f"Unknown node type '{node_type_str}' for node '{node}'")

    def _generate_exogenous_continuous(self, node: str):
        """Generates data for a continuous node with no parents."""
        dist_name = self._get_param(
            ['node_params', node, 'distribution', 'name'], 
            lambda: self.rng_structure.choice(['gaussian', 'student_t', 'gamma', 'exponential']),
            node_type='exogenous'
        )
        n_samples = self.data.shape[0]

        if dist_name == 'gaussian':
            mean = self._get_param(
                ['node_params', node, 'distribution', 'mean'], 
                lambda: self.rng_structure.normal(0, 1), 
                node_type='exogenous'
            )
            std = self._get_param(
                ['node_params', node, 'distribution', 'std'], 
                lambda: self.rng_structure.uniform(0.5, 1.5),
                node_type='exogenous'
            )
            self.data[node] = self.rng_data.normal(mean, std, size=n_samples)
        elif dist_name == 'student_t':
            df = self._get_param(
                ['node_params', node, 'distribution', 'df'], 
                lambda: self.rng_structure.integers(3, 10),
                node_type='exogenous'
            )
            self.data[node] = self.rng_data.standard_t(df, size=n_samples)
        elif dist_name == 'gamma':
            shape = self._get_param(
                ['node_params', node, 'distribution', 'shape'], 
                lambda: self.rng_structure.uniform(1, 5),
                node_type='exogenous'
            )
            scale = self._get_param(
                ['node_params', node, 'distribution', 'scale'], 
                lambda: self.rng_structure.uniform(0.5, 2),
                node_type='exogenous'
            )
            self.data[node] = self.rng_data.gamma(shape, scale, size=n_samples)
        elif dist_name == 'exponential':
            scale = self._get_param(
                ['node_params', node, 'distribution', 'scale'], 
                lambda: self.rng_structure.uniform(0.5, 2),
                node_type='exogenous'
            )
            self.data[node] = self.rng_data.exponential(scale, size=n_samples)
        else:
            raise ValueError(f"Unknown continuous distribution for exogenous node '{node}': {dist_name}")

        # sanitize
        self.data[node] = self._sanitize_series(self.data[node])

    def _generate_exogenous_binary(self, node: str):
        """Generates data for a binary node with no parents."""
        force_uniform = self.simulation_params.get("force_uniform_marginals", False)
        dist_name = self._get_param(
            ['node_params', node, 'distribution', 'name'], 
            lambda: 'bernoulli',
            node_type='exogenous'
        )
        if dist_name != 'bernoulli':
            raise ValueError(f"Binary exogenous node '{node}' must use 'bernoulli' distribution.")
        
        if force_uniform:
            p = self._get_param(
                ['node_params', node, 'distribution', 'p'],
                lambda: 0.5,
                node_type='exogenous'
            )
        else:
            p = self._get_param(
                ['node_params', node, 'distribution', 'p'], 
                lambda: self.rng_structure.uniform(0.1, 0.9),
                node_type='exogenous'
            )
        self.data[node] = self.rng_data.binomial(1, p, size=self.data.shape[0])

    def _generate_exogenous_categorical(self, node: str):
        """Generates data for a categorical exogenous node."""
        force_uniform = self.simulation_params.get("force_uniform_marginals", False)
        n_samples = self.data.shape[0]
        cardinality = self._node_cardinality(node)
        if force_uniform:
            probs = self._get_param(
                ['node_params', node, 'distribution', 'probs'],
                lambda: (np.ones(cardinality) / cardinality).tolist(),
                node_type='exogenous'
            )
        else:
            probs = self._get_param(
                ['node_params', node, 'distribution', 'probs'],
                lambda: self.rng_structure.dirichlet(np.ones(cardinality)).tolist(),
                node_type='exogenous'
            )
        probs = np.asarray(probs, dtype=float)
        if probs.shape[0] != cardinality:
            raise ValueError(
                f"Categorical exogenous node '{node}' expects {cardinality} probabilities, got {probs.shape[0]}"
            )
        probs = probs / probs.sum()
        self.data[node] = self.rng_data.choice(np.arange(cardinality), size=n_samples, p=probs)

    def _generate_endogenous_continuous(self, node: str, parents: list):
        """Generates data for a continuous node with parents."""
        base_value = self._apply_functional_form(node, parents)
        final_value = self._apply_noise_model(node, base_value, parents)
        self.data[node] = self._sanitize_series(final_value)

    def _generate_endogenous_binary(self, node: str, parents: list):
        """Generates data for a binary node with parents, using a sigmoid link."""
        base_value = self._apply_functional_form(node, parents)
        final_value = self._apply_noise_model(node, base_value, parents)
        
        prob = expit(final_value)
        self.data[node] = self.rng_data.binomial(1, p=prob)

    def _generate_endogenous_categorical(self, node: str, parents: list):
        """Generates data for a multi-level categorical node with parent-driven probabilities."""
        n_samples = self.data.shape[0]
        cardinality = self._node_cardinality(node)
        model_name = self._get_param(
            ['node_params', node, 'categorical_model', 'name'],
            lambda: 'logistic',
            node_type='endogenous'
        )

        if model_name == "threshold":
            scores = np.zeros(n_samples, dtype=float)
            weights = self._get_param(
                ['node_params', node, 'categorical_model', 'weights'],
                lambda: {p: self._sample_random_weight() for p in parents},
                node_type='endogenous'
            )
            for parent in parents:
                weight = float(weights.get(parent, 0.0)) if isinstance(weights, dict) else float(weights)
                scores += weight * np.asarray(self.data[parent], dtype=float)

            thresholds = self._get_param(
                ['node_params', node, 'categorical_model', 'thresholds'],
                lambda: (
                    float(self._get_param(
                        ['node_params', node, 'categorical_model', 'threshold_loc'],
                        lambda: 0.0,
                        node_type='endogenous'
                    ))
                    + float(self._get_param(
                        ['node_params', node, 'categorical_model', 'threshold_scale'],
                        lambda: float(self.rng_structure.uniform(0.5, 2.0)),
                        node_type='endogenous'
                    ))
                    * norm.ppf(np.linspace(0, 1, cardinality + 1)[1:-1])
                ).tolist(),
                node_type='endogenous'
            )
            thresholds = np.asarray(thresholds, dtype=float)
            if thresholds.shape[0] != max(0, cardinality - 1):
                raise ValueError(
                    f"Threshold model for '{node}' expects {cardinality - 1} thresholds, got {thresholds.shape[0]}"
                )
            self.data[node] = np.digitize(scores, bins=np.sort(thresholds), right=False)
            return

        if model_name != "logistic":
            raise ValueError(f"Unknown categorical model for node '{node}': {model_name}")

        intercepts = self._get_param(
            ['node_params', node, 'categorical_model', 'intercepts'],
            lambda: self.rng_structure.normal(0, 0.2, size=cardinality).tolist(),
            node_type='endogenous'
        )
        logits = np.tile(np.asarray(intercepts, dtype=float), (n_samples, 1))
        if logits.shape[1] != cardinality:
            raise ValueError(
                f"Categorical logistic model for '{node}' expects {cardinality} intercepts, got {logits.shape[1]}"
            )

        weights = self._get_param(
            ['node_params', node, 'categorical_model', 'weights'],
            lambda: {},
            node_type='endogenous'
        )
        if not isinstance(weights, dict):
            raise ValueError(f"'weights' for categorical node '{node}' must be a dictionary")

        completed_weights = {}
        for parent in parents:
            parent_type = self._node_type(parent)
            parent_values = np.asarray(self.data[parent])
            parent_weight = weights.get(parent)

            if parent_type == "categorical":
                parent_cardinality = self._node_cardinality(parent)
                if parent_weight is None:
                    parent_weight = self._sample_logistic_weight_array(
                        shape=(parent_cardinality, cardinality),
                        std=0.25
                    )
                parent_weight = np.asarray(parent_weight, dtype=float)
                if parent_weight.shape != (parent_cardinality, cardinality):
                    raise ValueError(
                        f"Weights for categorical parent '{parent}' must have shape "
                        f"({parent_cardinality}, {cardinality}), got {parent_weight.shape}"
                    )
                parent_idx = np.clip(parent_values.astype(int), 0, parent_cardinality - 1)
                logits += parent_weight[parent_idx]
                completed_weights[parent] = parent_weight.tolist()
            else:
                if parent_weight is None:
                    parent_weight = self._sample_logistic_weight_array(
                        shape=(cardinality,),
                        std=0.5
                    )
                parent_weight = np.asarray(parent_weight, dtype=float)
                if parent_weight.shape != (cardinality,):
                    raise ValueError(
                        f"Weights for parent '{parent}' must have shape ({cardinality},), got {parent_weight.shape}"
                    )
                logits += np.outer(parent_values.astype(float), parent_weight)
                completed_weights[parent] = parent_weight.tolist()

        param_storage = (
            self.final_parametrization
            .setdefault('node_params', {})
            .setdefault(node, {})
            .setdefault('categorical_model', {})
        )
        param_storage['weights'] = completed_weights

        probs = self._stable_softmax(logits)
        cumulative = np.cumsum(probs, axis=1)
        draws = self.rng_data.random(n_samples)[:, None]
        categories = (draws > cumulative).sum(axis=1)
        self.data[node] = categories.astype(int)

    def _apply_functional_form(self, node: str, parents: list):
        """Computes a node's base value from its parents' data."""
        form_name = self._get_param(
            ['node_params', node, 'functional_form', 'name'], 
            lambda: self.rng_structure.choice(['linear', 'polynomial', 'interaction']),
            node_type='endogenous'
        )
        categorical_parents = [p for p in parents if self._node_type(p) == "categorical"]
        if form_name in {"linear", "polynomial", "interaction"} and categorical_parents:
            policy = str(self.simulation_params.get("categorical_parent_metric_form_policy", "error")).lower()
            if policy == "stratum_means":
                form_name = "stratum_means"
            elif policy == "error":
                raise ValueError(
                    f"Node '{node}' uses metric functional form '{form_name}' with categorical parent(s) "
                    f"{categorical_parents}. Use 'stratum_means' or set "
                    f"simulation_params.categorical_parent_metric_form_policy='stratum_means'."
                )
            else:
                raise ValueError(
                    f"Unknown categorical_parent_metric_form_policy='{policy}'. "
                    "Use 'error' or 'stratum_means'."
                )
        
        # This function will ensure that for any parameter (like weights or degrees),
        # we get a dictionary with a value for every parent.
        def get_parent_param_dict(param_name, default_sampler):
            param = self._get_param(
                ['node_params', node, 'functional_form', param_name],
                default_sampler,
                node_type='endogenous'
            )
            # If the retrieved param is a single number, create a dict for all parents.
            if isinstance(param, (float, int, np.floating, np.integer)):
                return {p: param for p in parents}
            # If it's a dict but is missing some parents, fill them in.
            if isinstance(param, dict):
                # We create a copy to avoid modifying the original config dictionary
                param_copy = param.copy()
                for p in parents:
                    if p not in param_copy:
                        # This can happen in random graphs where parent names are unpredictable.
                        # We sample a value for the missing parent.
                        sampled_values = default_sampler()
                        if isinstance(sampled_values, dict) and p in sampled_values:
                            param_copy[p] = sampled_values[p]
                        elif isinstance(sampled_values, dict):
                             # if the sampler returns a dict but not for the specific parent, sample again
                             param_copy[p] = self._sample_random_weight()
                        else:
                             param_copy[p] = sampled_values

                return param_copy
            return param

        if form_name == 'linear':
            weights = get_parent_param_dict('weights', lambda: {p: self._sample_random_weight() for p in parents})
            return sum(weights[p] * self.data[p] for p in parents)
        
        elif form_name == 'polynomial':
            weights = get_parent_param_dict('weights', lambda: {p: self._sample_random_weight() for p in parents})
            degrees = get_parent_param_dict('degrees', lambda: {p: self.rng_structure.integers(2, 5) for p in parents})
            # Clip parent values to avoid overflow when raising to powers
            return sum(weights[p] * (self.data[p].clip(-SAFE_PARENT_CLIP, SAFE_PARENT_CLIP) ** degrees[p]) for p in parents)

        elif form_name == 'interaction':
            weights = self._get_param(
                ['node_params', node, 'functional_form', 'weights'], 
                lambda: {'interaction': self._sample_random_weight()},
                node_type='endogenous'
            )
            interaction_term = pd.Series(1, index=self.data.index)
            for p in parents:
                interaction_term *= self.data[p].clip(-SAFE_PARENT_CLIP, SAFE_PARENT_CLIP)
            return weights.get('interaction', 1.0) * interaction_term
        elif form_name == 'stratum_means':
            strata_means = self._get_param(
                ['node_params', node, 'functional_form', 'strata_means'],
                lambda: {},
                node_type='endogenous'
            )
            default_mean = self._get_param(
                ['node_params', node, 'functional_form', 'default_mean'],
                lambda: 0.0,
                node_type='endogenous'
            )
            if not isinstance(strata_means, dict):
                raise ValueError(f"'strata_means' for node '{node}' must be a dictionary")
            if not parents:
                raise ValueError(f"'stratum_means' requires at least one parent for node '{node}'")
            categorical_parents = [p for p in parents if self._node_type(p) == "categorical"]
            metric_parents = [p for p in parents if self._node_type(p) != "categorical"]
            if not categorical_parents:
                raise ValueError(
                    f"'stratum_means' requires at least one categorical parent. "
                    f"Node '{node}' has parents: {parents}"
                )

            completed_strata_means = {k: float(v) for k, v in strata_means.items()}
            parent_cardinalities = [self._node_cardinality(p) for p in categorical_parents]
            for combo in itertools.product(*[range(c) for c in parent_cardinalities]):
                key = "|".join([f"{p}={int(v)}" for p, v in zip(categorical_parents, combo)])
                if key not in completed_strata_means:
                    completed_strata_means[key] = float(self.rng_structure.normal(0, 1))

            metric_weights = self._get_param(
                ['node_params', node, 'functional_form', 'metric_weights'],
                lambda: {p: self._sample_random_weight() for p in metric_parents},
                node_type='endogenous'
            )
            if isinstance(metric_weights, (float, int, np.floating, np.integer)):
                metric_weights = {p: float(metric_weights) for p in metric_parents}
            elif not isinstance(metric_weights, dict):
                raise ValueError(f"'metric_weights' for node '{node}' must be a number or dictionary")
            else:
                metric_weights = {**{p: self._sample_random_weight() for p in metric_parents}, **metric_weights}

            cat_parent_df = self.data[categorical_parents]
            parent_values = cat_parent_df.to_numpy(dtype=int)
            keys = [
                "|".join([f"{p}={int(v)}" for p, v in zip(categorical_parents, row)])
                for row in parent_values
            ]
            strata_component = np.array(
                [completed_strata_means.get(k, float(default_mean)) for k in keys],
                dtype=float
            )

            metric_component = np.zeros_like(strata_component)
            for p in metric_parents:
                metric_component += float(metric_weights[p]) * np.asarray(self.data[p], dtype=float)
            output = strata_component + metric_component

            ff_store = (
                self.final_parametrization
                .setdefault('node_params', {})
                .setdefault(node, {})
                .setdefault('functional_form', {})
            )
            ff_store['strata_means'] = completed_strata_means
            ff_store['metric_weights'] = metric_weights
            return pd.Series(output, index=self.data.index)
        else:
            raise ValueError(f"Unknown functional form for node '{node}': {form_name}")

    def _apply_noise_model(self, node: str, base_value, parents: list):
        """Adds noise to the base value computed from parents."""
        noise_name = self._get_param(
            ['node_params', node, 'noise_model', 'name'], 
            lambda: self.rng_structure.choice(['additive', 'multiplicative', 'heteroskedastic']),
            node_type='endogenous'
        )
        n_samples = self.data.shape[0]

        if noise_name == 'additive':
            dist = self._get_param(
                ['node_params', node, 'noise_model', 'dist'], 
                lambda: 'gaussian',
                node_type='endogenous'
            )
            if dist == 'gaussian':
                std = self._get_param(
                    ['node_params', node, 'noise_model', 'std'], 
                    lambda: self.rng_structure.uniform(0.5, 1.5),
                    node_type='endogenous'
                )
                noise = self.rng_data.normal(0, std, size=n_samples)
                return base_value + noise
            elif dist == 'student_t':
                df = self._get_param(
                    ['node_params', node, 'noise_model', 'df'], 
                    lambda: self.rng_structure.integers(3, 10),
                    node_type='endogenous'
                )
                scale = self._get_param(
                    ['node_params', node, 'noise_model', 'scale'], 
                    lambda: self.rng_structure.uniform(0.5, 1.5),
                    node_type='endogenous'
                )
                noise = self.rng_data.standard_t(df, size=n_samples) * scale
                return base_value + noise
            elif dist == 'gamma':
                shape = self._get_param(
                    ['node_params', node, 'noise_model', 'shape'], 
                    lambda: self.rng_structure.uniform(1.0, 5.0),
                    node_type='endogenous'
                )
                scale = self._get_param(
                    ['node_params', node, 'noise_model', 'scale'], 
                    lambda: self.rng_structure.uniform(0.5, 2.0),
                    node_type='endogenous'
                )
                # Center to zero-mean to avoid biasing the signal
                noise = self.rng_data.gamma(shape, scale, size=n_samples) - (shape * scale)
                return base_value + noise
            elif dist == 'exponential':
                scale = self._get_param(
                    ['node_params', node, 'noise_model', 'scale'], 
                    lambda: self.rng_structure.uniform(0.5, 2.0),
                    node_type='endogenous'
                )
                # Center to zero-mean
                noise = self.rng_data.exponential(scale, size=n_samples) - scale
                return base_value + noise
            else:
                raise ValueError(f"Unknown additive noise dist: {dist}")
        elif noise_name == 'multiplicative':
            dist = self._get_param(
                ['node_params', node, 'noise_model', 'dist'], 
                lambda: 'gaussian',
                node_type='endogenous'
            )
            if dist == 'gaussian':
                std = self._get_param(
                    ['node_params', node, 'noise_model', 'std'], 
                    lambda: self.rng_structure.uniform(0.1, 0.5),
                    node_type='endogenous'
                )
                factor = 1.0 + self.rng_data.normal(0, std, size=n_samples)
                # Ensure strictly positive factor
                factor = np.clip(factor, 1e-6, SAFE_OUTPUT_CLIP)
                return base_value * factor
            elif dist == 'student_t':
                df = self._get_param(
                    ['node_params', node, 'noise_model', 'df'], 
                    lambda: self.rng_structure.integers(3, 10),
                    node_type='endogenous'
                )
                scale = self._get_param(
                    ['node_params', node, 'noise_model', 'scale'], 
                    lambda: self.rng_structure.uniform(0.1, 0.5),
                    node_type='endogenous'
                )
                factor = 1.0 + self.rng_data.standard_t(df, size=n_samples) * scale
                factor = np.clip(factor, 1e-6, SAFE_OUTPUT_CLIP)
                return base_value * factor
            elif dist == 'gamma':
                shape = self._get_param(
                    ['node_params', node, 'noise_model', 'shape'], 
                    lambda: self.rng_structure.uniform(1.0, 5.0),
                    node_type='endogenous'
                )
                scale = self._get_param(
                    ['node_params', node, 'noise_model', 'scale'], 
                    lambda: self.rng_structure.uniform(0.5, 2.0),
                    node_type='endogenous'
                )
                # Normalize to mean 1 for a multiplicative factor
                gamma_sample = self.rng_data.gamma(shape, scale, size=n_samples)
                mean = shape * scale
                mean = mean if mean != 0 else 1.0
                factor = gamma_sample / mean
                factor = np.clip(factor, 1e-12, SAFE_OUTPUT_CLIP)
                return base_value * factor
            elif dist == 'exponential':
                scale = self._get_param(
                    ['node_params', node, 'noise_model', 'scale'], 
                    lambda: self.rng_structure.uniform(0.5, 2.0),
                    node_type='endogenous'
                )
                # Normalize to mean 1
                exp_sample = self.rng_data.exponential(scale, size=n_samples)
                factor = exp_sample / scale if scale != 0 else 1.0
                factor = np.clip(factor, 1e-12, SAFE_OUTPUT_CLIP)
                return base_value * factor
            else:
                raise ValueError(f"Unknown multiplicative noise dist: {dist}")
        elif noise_name == 'heteroskedastic':
            func_spec = self._get_param(
                ['node_params', node, 'noise_model', 'func'], 
                lambda: 'abs_first_parent',
                node_type='endogenous'
            )

            if callable(func_spec):
                noise_func = func_spec
            else:
                func_name = LEGACY_HETEROSKEDASTIC_FUNC_ALIASES.get(str(func_spec), str(func_spec))
                noise_func = HETEROSKEDASTIC_FN_REGISTRY.get(func_name)
                if noise_func is None:
                    supported = ", ".join(sorted(HETEROSKEDASTIC_FN_REGISTRY.keys()))
                    raise ValueError(
                        f"Unsupported heteroskedastic func '{func_spec}'. "
                        f"Use one of: {supported}"
                    )

            parent_data = self.data[parents]
            noise_std = noise_func(parent_data)
            noise = self.rng_data.normal(0, 1, size=n_samples) * noise_std
            return base_value + noise
        
        return base_value

    def _build_ci_oracle(self, max_cond_set_size: int = 2) -> list:
        """
        Builds a d-separation oracle from the generated DAG.

        Returns:
            list: Records with keys x, y, conditioning_set, is_independent.
        """
        nodes = list(self.graph.nodes())
        oracle = []
        max_k = max(0, int(max_cond_set_size))

        for i, x in enumerate(nodes):
            for y in nodes[i + 1:]:
                remaining = [n for n in nodes if n not in (x, y)]
                for k in range(0, min(max_k, len(remaining)) + 1):
                    for cond_set in itertools.combinations(remaining, k):
                        is_independent = bool(
                            is_d_separator(self.graph, {x}, {y}, set(cond_set))
                        )
                        oracle.append(
                            {
                                "x": x,
                                "y": y,
                                "conditioning_set": list(cond_set),
                                "is_independent": is_independent,
                            }
                        )

        return oracle

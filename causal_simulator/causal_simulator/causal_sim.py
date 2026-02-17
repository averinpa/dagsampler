import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import expit
import itertools
import copy

# Safety guards to avoid exploding values that can cause Inf/NaN
SAFE_PARENT_CLIP = 1e3
SAFE_OUTPUT_CLIP = 1e12

class CausalDataGenerator:
    """
    Generates data from a causal graph based on a detailed configuration.

    This class is designed to be flexible, allowing for the specification
    of graph structures, variable types, functional relationships, and noise
    distributions through a single configuration object. It supports both
    random parameter generation and hardcoded parameters for reproducible experiments.
    """

    def __init__(self, config: dict):
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
            if graph_type == 'fork':
                z_size = self.graph_params.get('z_size', 1)
                z_nodes = [f'Z{i+1}' for i in range(z_size)]
                self.graph.add_nodes_from(['X', 'Y'] + z_nodes)
                for z in z_nodes:
                    self.graph.add_edge(z, 'X')
                    self.graph.add_edge(z, 'Y')
            elif graph_type == 'chain':
                z_size = self.graph_params.get('z_size', 1)
                nodes = ['X'] + [f'Z{i+1}' for i in range(z_size)] + ['Y']
                nx.add_path(self.graph, nodes)
            elif graph_type == 'v_structure':
                z_size = self.graph_params.get('z_size', 1)
                z_nodes = [f'Z{i+1}' for i in range(z_size)]
                self.graph.add_nodes_from(['X', 'Y'] + z_nodes)
                for z in z_nodes:
                    self.graph.add_edge('X', z)
                    self.graph.add_edge('Y', z)
            elif graph_type == 'custom':
                nodes = self.graph_params.get('nodes', [])
                edges = self.graph_params.get('edges', [])
                self.graph = nx.DiGraph(edges)
                self.graph.add_nodes_from(nodes)
            else: # 'random'
                n_nodes = self.graph_params.get('n_nodes', 5)
                edge_prob = self.graph_params.get('edge_prob', 0.3)
                nodes = [f'N{i}' for i in range(n_nodes)]
                self.graph.add_nodes_from(nodes)
                for u, v in itertools.combinations(nodes, 2):
                    if self.rng_structure.random() < edge_prob:
                        # To ensure DAG, always add edges in one direction
                        self.graph.add_edge(u, v)

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The specified graph is not a DAG.")

        # Persist exact nodes/edges used to enable exact reproducibility later
        self.final_parametrization.setdefault('graph_params', {})
        self.final_parametrization['graph_params']['nodes'] = list(self.graph.nodes())
        self.final_parametrization['graph_params']['edges'] = list(self.graph.edges())

    # --- Data Simulation Engine ---

    def simulate(self) -> dict:
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

        return {
            "data": self.data,
            "parametrization": self.final_parametrization,
            "dag": self.graph
        }

    def _generate_node_data(self, node: str):
        """
        Determines a node's type (binary or continuous) and dispatches to the
        appropriate generation function.
        """
        parents = list(self.graph.predecessors(node))

        # First, determine the node's data type (binary or continuous).
        # This can be hardcoded or will be randomly chosen based on binary_proportion.
        binary_proportion = self.simulation_params.get('binary_proportion', 0.4)
        node_type_str = self._get_param(
            ['node_params', node, 'type'],
            lambda: self.rng_structure.choice(["binary", "continuous"], p=[binary_proportion, 1 - binary_proportion]),
            node_type='endogenous' if parents else 'exogenous'
        )

        if not parents:  # Exogenous node
            if node_type_str == 'continuous':
                self._generate_exogenous_continuous(node)
            else:  # binary
                self._generate_exogenous_binary(node)
        else:  # Endogenous node
            if node_type_str == 'continuous':
                self._generate_endogenous_continuous(node, parents)
            else:  # binary
                self._generate_endogenous_binary(node, parents)

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
        dist_name = self._get_param(
            ['node_params', node, 'distribution', 'name'], 
            lambda: 'bernoulli',
            node_type='exogenous'
        )
        if dist_name != 'bernoulli':
            raise ValueError(f"Binary exogenous node '{node}' must use 'bernoulli' distribution.")
        
        p = self._get_param(
            ['node_params', node, 'distribution', 'p'], 
            lambda: self.rng_structure.uniform(0.1, 0.9),
            node_type='exogenous'
        )
        self.data[node] = self.rng_data.binomial(1, p, size=self.data.shape[0])

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

    def _apply_functional_form(self, node: str, parents: list):
        """Computes a node's base value from its parents' data."""
        form_name = self._get_param(
            ['node_params', node, 'functional_form', 'name'], 
            lambda: self.rng_structure.choice(['linear', 'polynomial', 'interaction']),
            node_type='endogenous'
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
                             param_copy[p] = self.rng_structure.uniform(-1.5, 1.5)
                        else:
                             param_copy[p] = sampled_values

                return param_copy
            return param

        if form_name == 'linear':
            weights = get_parent_param_dict('weights', lambda: {p: self.rng_structure.uniform(-1.5, 1.5) for p in parents})
            return sum(weights[p] * self.data[p] for p in parents)
        
        elif form_name == 'polynomial':
            weights = get_parent_param_dict('weights', lambda: {p: self.rng_structure.uniform(-1.5, 1.5) for p in parents})
            degrees = get_parent_param_dict('degrees', lambda: {p: self.rng_structure.integers(2, 5) for p in parents})
            # Clip parent values to avoid overflow when raising to powers
            return sum(weights[p] * (self.data[p].clip(-SAFE_PARENT_CLIP, SAFE_PARENT_CLIP) ** degrees[p]) for p in parents)

        elif form_name == 'interaction':
            weights = self._get_param(
                ['node_params', node, 'functional_form', 'weights'], 
                lambda: {'interaction': self.rng_structure.uniform(-1.5, 1.5)},
                node_type='endogenous'
            )
            interaction_term = pd.Series(1, index=self.data.index)
            for p in parents:
                interaction_term *= self.data[p].clip(-SAFE_PARENT_CLIP, SAFE_PARENT_CLIP)
            return weights.get('interaction', 1.0) * interaction_term
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
            func_str = self._get_param(
                ['node_params', node, 'noise_model', 'func'], 
                lambda: 'lambda p: 0.5 * np.abs(p.iloc[:, 0])',
                node_type='endogenous'
            )
            # Security warning: eval is used here. Only use with trusted configurations.
            noise_func = eval(func_str) 
            parent_data = self.data[parents]
            noise_std = noise_func(parent_data)
            noise = self.rng_data.normal(0, 1, size=n_samples) * noise_std
            return base_value + noise
        
        return base_value

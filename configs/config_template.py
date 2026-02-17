"""
This file serves as a comprehensive template for the CausalDataGenerator configuration.

It is a Python file instead of a JSON to allow for comments that explain
each available option.

To use this template:
1. Import it into your script or notebook: `from configs.config_template import config`
2. Modify the parameters as needed for your specific experiment.
"""

config = {
    # --------------------------------------------------------------------------
    # 1. Global Simulation Parameters
    # --------------------------------------------------------------------------
    "simulation_params": {
        "n_samples": 1000,
        "seed": 42,
        # Optional: provide dedicated seeds to fully separate structure/params
        # sampling from actual data draws (for strict reproducibility).
        # If omitted, the simulator derives sensible defaults:
        #   seed_structure = seed
        #   seed_data = seed + 1
        "seed_structure": 42,
        "seed_data": 43,
        # The probability that an unspecified node will be of type 'binary'.
        # For example, 0.4 means a 40% chance of being binary.
        "binary_proportion": 0.4
    },

    # --------------------------------------------------------------------------
    # 2. Graph Structure Parameters
    # Choose ONE of the following graph types.
    # The 'custom' graph is used by default to demonstrate all node types below.
    # --------------------------------------------------------------------------
    "graph_params": {
        # --- Option A: Custom Graph (DEFAULT FOR THIS TEMPLATE) ---
        "type": "custom",
        "nodes": [
            "Exo_Gaussian",
            "Exo_StudentT",
            "Exo_Binary",
            "Endo_Linear_Additive",
            "Endo_Poly_Multiplicative",
            "Endo_Interaction_Hetero",
            "Endo_Binary"
        ],
        "edges": [
            ["Exo_Gaussian", "Endo_Linear_Additive"],
            ["Exo_StudentT", "Endo_Linear_Additive"],

            ["Exo_Gaussian", "Endo_Poly_Multiplicative"],
            ["Exo_StudentT", "Endo_Poly_Multiplicative"],

            ["Exo_StudentT", "Endo_Interaction_Hetero"],
            ["Exo_Binary", "Endo_Interaction_Hetero"],

            ["Endo_Linear_Additive", "Endo_Binary"]
        ],
        # Note: If 'edges' is provided and non-empty, the simulator prioritizes
        # these explicit nodes/edges for reproducibility, regardless of 'type'.

        # --- Option B: Simple Fork Structure ---
        # "type": "fork",
        # "z_size": 2, # Creates Z1, Z2 as confounders

        # --- Option C: Simple Chain Structure ---
        # "type": "chain",
        # "z_size": 2, # Creates a chain X -> Z1 -> Z2 -> Y

        # --- Option D: Simple V-Structure ---
        # "type": "v_structure",
        # "z_size": 2, # Creates X -> Z1 <- Y and X -> Z2 <- Y

        # --- Option E: Random DAG ---
        # "type": "random",
        # "n_nodes": 6,
        # "edge_prob": 0.4
    },

    # --------------------------------------------------------------------------
    # 3. Node-Specific Parameters
    # This section defines the behavior of each node in the graph.
    # If you omit the 'type' key for any node, it will be randomly assigned
    # as 'continuous' or 'binary' based on 'binary_proportion' above.
    # --------------------------------------------------------------------------
    "node_params": {
        # You can also set generalized defaults that apply when a specific node does
        # not provide the parameter explicitly. For example:
        # "default_exogenous": { "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
        # "default_endogenous": {
        #     "functional_form": { "name": "linear" },
        #     "noise_model": { "name": "additive", "dist": "gaussian", "std": 1.0 }
        # },
        # Specific node entries below override these defaults.

        # --- EXOGENOUS NODES (no parents) ---
        "Exo_Gaussian": {
            "type": "continuous",
            "distribution": { "name": "gaussian", "mean": 0, "std": 1 }
        },
        "Exo_StudentT": {
            "type": "continuous",
            "distribution": { "name": "student_t", "df": 3 } # Heavy-tailed
        },
        "Exo_Binary": {
            "type": "binary",
            "distribution": { "name": "bernoulli", "p": 0.3 }
        },

        # --- ENDOGENOUS NODES (with parents) ---
        "Endo_Linear_Additive": {
            "type": "continuous",
            "functional_form": {
                "name": "linear",
                "weights": {"Exo_Gaussian": 1.5, "Exo_StudentT": -0.8}
            },
            "noise_model": {
                "name": "additive",
                # Supported: 'gaussian' | 'student_t' | 'gamma' | 'exponential'
                # gaussian: uses 'std'
                # student_t: uses 'df' and 'scale'
                # gamma: uses 'shape' and 'scale' (mean-centered inside simulator)
                # exponential: uses 'scale' (mean-centered inside simulator)
                "dist": "gaussian",
                "std": 0.5
                # Examples:
                # "dist": "student_t", "df": 5, "scale": 0.8
                # "dist": "gamma", "shape": 2.0, "scale": 0.8
                # "dist": "exponential", "scale": 1.0
            }
        },
        "Endo_Poly_Multiplicative": {
            "type": "continuous",
            "functional_form": {
                "name": "polynomial",
                "degrees": {"Exo_Gaussian": 2, "Exo_StudentT": 3},
                "weights": {"Exo_Gaussian": 0.5, "Exo_StudentT": -0.2}
            },
            "noise_model": {
                "name": "multiplicative",
                # Supported: 'gaussian' | 'student_t' | 'gamma' | 'exponential'
                # gaussian: uses 'std' and applies factor = 1 + N(0, std)
                # student_t: uses 'df' and 'scale' with factor = 1 + t(df)*scale
                # gamma: uses 'shape' and 'scale' with factor normalized to mean 1
                # exponential: uses 'scale' with factor normalized to mean 1
                "dist": "gaussian",
                "std": 0.2
                # Examples:
                # "dist": "student_t", "df": 5, "scale": 0.2
                # "dist": "gamma", "shape": 2.0, "scale": 0.8
                # "dist": "exponential", "scale": 1.0
            }
        },
        "Endo_Interaction_Hetero": {
            "type": "continuous",
            "functional_form": {
                "name": "interaction",
                # The weight for the full interaction term Z1*Z2*...
                "weights": {"interaction": 1.0}
            },
            "noise_model": {
                "name": "heteroskedastic",
                # Noise std depends on the value of a parent.
                # Must be a string evaluatable to a lambda that accepts a DataFrame of parents.
                "func": "lambda p: 0.5 * np.abs(p['Exo_StudentT']) + 0.1"
            }
        },
        "Endo_Binary": {
            "type": "binary",
            "functional_form": {
                "name": "linear",
                "weights": {"Endo_Linear_Additive": 2.5}
            },
            "noise_model": {
                # For binary nodes, the result of (functional_form + noise)
                # is passed through a sigmoid. Additive noise here adds
                # randomness *before* the sigmoid is applied.
                "name": "additive",
                # Supported: 'gaussian' | 'student_t' | 'gamma' | 'exponential'
                "dist": "gaussian",  # explicitly set; simulator assumes 'gaussian' if omitted
                "std": 0.1
            }
        }
    }
} 
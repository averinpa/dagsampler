Configuration Examples
======================

This page shows practical config templates for all major simulator options.

Minimal custom DAG
------------------

.. code-block:: json

   {
     "simulation_params": {
       "n_samples": 200,
       "seed": 42
     },
     "graph_params": {
       "type": "custom",
       "nodes": ["X", "Y", "Z1"],
       "edges": [["X", "Z1"], ["Y", "Z1"]]
     }
   }

Random DAG
----------

.. code-block:: json

   {
     "simulation_params": {
       "n_samples": 300,
       "seed_structure": 123,
       "seed_data": 124
     },
     "graph_params": {
       "type": "random",
       "n_nodes": 6,
       "edge_prob": 0.35
     }
   }

Exogenous node distributions
----------------------------

.. code-block:: json

   {
     "simulation_params": {
       "n_samples": 500,
       "seed": 1
     },
     "graph_params": {
       "type": "custom",
       "nodes": ["G", "T", "Ga", "E", "B", "C"],
       "edges": []
     },
     "node_params": {
       "G": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0.0, "std": 1.0 } },
       "T": { "type": "continuous", "distribution": { "name": "student_t", "df": 4 } },
       "Ga": { "type": "continuous", "distribution": { "name": "gamma", "shape": 2.0, "scale": 1.0 } },
       "E": { "type": "continuous", "distribution": { "name": "exponential", "scale": 1.2 } },
       "B": { "type": "binary", "distribution": { "name": "bernoulli", "p": 0.35 } },
       "C": {
         "type": "categorical",
         "cardinality": 5,
         "distribution": { "probs": [0.1, 0.2, 0.3, 0.2, 0.2] }
       }
     }
   }

Continuous child with linear / polynomial / interaction
-------------------------------------------------------

.. code-block:: json

   {
     "simulation_params": { "n_samples": 300, "seed": 10 },
     "graph_params": {
       "type": "custom",
       "nodes": ["X1", "X2", "Y_lin", "Y_poly", "Y_int"],
       "edges": [["X1", "Y_lin"], ["X2", "Y_lin"], ["X1", "Y_poly"], ["X2", "Y_poly"], ["X1", "Y_int"], ["X2", "Y_int"]]
     },
     "node_params": {
       "X1": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
       "X2": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
       "Y_lin": {
         "type": "continuous",
         "functional_form": { "name": "linear", "weights": { "X1": 1.2, "X2": -0.7 } },
         "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.5 }
       },
       "Y_poly": {
         "type": "continuous",
         "functional_form": { "name": "polynomial", "weights": { "X1": 1.0, "X2": 0.6 }, "degrees": { "X1": 3, "X2": 2 } },
         "noise_model": { "name": "additive", "dist": "student_t", "df": 5, "scale": 0.3 }
       },
       "Y_int": {
         "type": "continuous",
         "functional_form": { "name": "interaction", "weights": { "interaction": 0.8 } },
         "noise_model": { "name": "multiplicative", "dist": "gaussian", "std": 0.2 }
       }
     }
   }

Noise model variants
--------------------

.. code-block:: json

   {
     "simulation_params": { "n_samples": 250, "seed": 22 },
     "graph_params": {
       "type": "custom",
       "nodes": ["X", "Y_add", "Y_mult", "Y_hetero"],
       "edges": [["X", "Y_add"], ["X", "Y_mult"], ["X", "Y_hetero"]]
     },
     "node_params": {
       "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
       "Y_add": {
         "type": "continuous",
         "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
         "noise_model": { "name": "additive", "dist": "gamma", "shape": 2.0, "scale": 0.6 }
       },
       "Y_mult": {
         "type": "continuous",
         "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
         "noise_model": { "name": "multiplicative", "dist": "exponential", "scale": 1.0 }
       },
       "Y_hetero": {
         "type": "continuous",
         "functional_form": { "name": "linear", "weights": { "X": 1.0 } },
         "noise_model": { "name": "heteroskedastic", "func": "abs_parent_plus_const" }
       }
     }
   }

Binary child
------------

.. code-block:: json

   {
     "simulation_params": { "n_samples": 300, "seed": 33 },
     "graph_params": {
       "type": "custom",
       "nodes": ["X", "Z", "B"],
       "edges": [["X", "B"], ["Z", "B"]]
     },
     "node_params": {
       "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
       "Z": { "type": "binary", "distribution": { "name": "bernoulli", "p": 0.4 } },
       "B": {
         "type": "binary",
         "functional_form": { "name": "linear", "weights": { "X": 1.3, "Z": 0.9 } },
         "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.5 }
       }
     }
   }

Categorical child (logistic softmax)
------------------------------------

.. code-block:: json

   {
     "simulation_params": { "n_samples": 400, "seed_structure": 40, "seed_data": 41 },
     "graph_params": {
       "type": "custom",
       "nodes": ["X", "B", "C"],
       "edges": [["X", "C"], ["B", "C"]]
     },
     "node_params": {
       "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
       "B": { "type": "binary", "distribution": { "name": "bernoulli", "p": 0.5 } },
       "C": {
         "type": "categorical",
         "cardinality": 3,
         "categorical_model": {
           "name": "logistic",
           "intercepts": [0.0, 0.0, 0.0],
           "weights": {
             "X": [0.9, -0.2, -0.7],
             "B": [-0.4, 0.8, -0.3]
           }
         }
       }
     }
   }

Continuous to categorical (threshold)
-------------------------------------

.. code-block:: json

   {
     "simulation_params": { "n_samples": 350, "seed": 50 },
     "graph_params": {
       "type": "custom",
       "nodes": ["X", "C"],
       "edges": [["X", "C"]]
     },
     "node_params": {
       "X": { "type": "continuous", "distribution": { "name": "gaussian", "mean": 0, "std": 1 } },
       "C": {
         "type": "categorical",
         "cardinality": 5,
         "categorical_model": {
           "name": "threshold",
           "weights": { "X": 1.0 },
           "thresholds": [-1.0, -0.2, 0.4, 1.1]
         }
       }
     }
   }

Categorical to continuous (stratum-specific means)
--------------------------------------------------

.. code-block:: json

   {
     "simulation_params": { "n_samples": 300, "seed": 60 },
     "graph_params": {
       "type": "custom",
       "nodes": ["C1", "C2", "Y"],
       "edges": [["C1", "Y"], ["C2", "Y"]]
     },
     "node_params": {
       "C1": { "type": "categorical", "cardinality": 3 },
       "C2": { "type": "categorical", "cardinality": 2 },
       "Y": {
         "type": "continuous",
         "functional_form": {
           "name": "stratum_means",
           "default_mean": 0.0,
           "strata_means": {
             "C1=0|C2=0": -1.5,
             "C1=1|C2=0": 0.2,
             "C1=2|C2=1": 1.8
           }
         },
         "noise_model": { "name": "additive", "dist": "gaussian", "std": 0.15 }
       }
     }
   }

CI oracle output
----------------

.. code-block:: json

   {
     "simulation_params": {
       "n_samples": 250,
       "seed": 70,
       "store_ci_oracle": true,
       "ci_oracle_max_cond_set": 2
     },
     "graph_params": {
       "type": "custom",
       "nodes": ["X", "Y", "Z"],
       "edges": [["X", "Z"], ["Y", "Z"]]
     }
   }

When ``store_ci_oracle`` is enabled, ``simulate()`` also returns a ``ci_oracle``
list with entries of the form:

.. code-block:: json

   {
     "x": "X",
     "y": "Y",
     "conditioning_set": ["Z"],
     "is_independent": false
   }

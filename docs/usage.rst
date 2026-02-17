Usage
=====

Install
-------

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install "dagsampler @ git+https://github.com/averinpa/dagsampler.git"

Python API
----------

.. code-block:: python

   from causal_simulator import CausalDataGenerator

   config = {
       "simulation_params": {"n_samples": 200, "seed": 42},
       "graph_params": {
           "type": "custom",
           "nodes": ["X", "Y", "Z1"],
           "edges": [["X", "Z1"], ["Y", "Z1"]],
       },
   }

   result = CausalDataGenerator(config).simulate()

CLI
---

.. code-block:: bash

   dagsampler-generate \
     --config config.json \
     --output dataset.csv \
     --params-out params.json \
     --edges-out edges.json

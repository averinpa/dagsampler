Overview
========

The ``dagsampler`` package provides a configurable causal data generator.

Main features:

* ``custom`` and ``random`` DAG generation
* Continuous, binary, and categorical variables
* Multiple structural forms and noise models
* Random structural weight controls:
  * ``random_weight_low``
  * ``random_weight_high``
  * ``random_weight_min_abs`` (excludes near-zero coefficients)
* Reproducible sampling with separate structure/data seeds
* Optional d-separation CI oracle output

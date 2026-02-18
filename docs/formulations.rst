Model Formulations
==================

This page describes the mathematical structure implemented by the simulator and
the valid combinations of node types, structural equations, and noise models.

Graph Model
-----------

The simulator generates a DAG :math:`G = (V, E)` using one of:

* ``custom``: user-defined node and edge sets
* ``random``: random acyclic edges over ordered nodes

For any node :math:`j \in V`, let :math:`\mathrm{Pa}(j)` denote its parents.

Node Types
----------

Supported node types:

* Continuous
* Binary (values in ``{0, 1}``)
* Categorical (values in ``{0, \dots, K-1}``, configurable cardinality :math:`K`)

Exogenous Nodes (:math:`\mathrm{Pa}(j)=\varnothing`)
----------------------------------------------------

Continuous exogenous node:

.. math::

   X_j \sim \mathcal{D}_j

where :math:`\mathcal{D}_j` is one of Gaussian, Student-t, Gamma, or Exponential.

Binary exogenous node:

.. math::

   X_j \sim \mathrm{Bernoulli}(p_j)

Categorical exogenous node:

.. math::

   X_j \sim \mathrm{Categorical}(\pi_{j,0}, \dots, \pi_{j,K-1}), \quad \sum_k \pi_{j,k}=1

Endogenous Continuous Nodes
---------------------------

General form:

.. math::

   X_j = f_j(X_{\mathrm{Pa}(j)}) + \epsilon_j

Supported structural forms :math:`f_j`:

Linear:

.. math::

   f_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p

Polynomial:

.. math::

   f_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p^{d_{jp}}

Interaction:

.. math::

   f_j = w_j \prod_{p \in \mathrm{Pa}(j)} X_p

Stratum-specific means (categorical parents to continuous child):

.. math::

   f_j = \mu_{s(\mathbf{x}_{\mathrm{Pa}(j)})}

where :math:`s(\cdot)` indexes the categorical parent stratum.

Random structural weights
-------------------------

When ``weights`` are omitted for ``linear``, ``polynomial``, or ``interaction``,
the simulator samples weights from a configurable interval:

.. math::

   w \sim \mathrm{Uniform}(L, H)

where ``L=random_weight_low`` and ``H=random_weight_high``.

If ``random_weight_min_abs = m > 0``, values in :math:`(-m, m)` are excluded
and weights are sampled from:

.. math::

   [L, -m] \cup [m, H]

This is useful for CI benchmark settings where near-zero coefficients can cause
practical faithfulness issues via cancellation.

Noise models:

Additive:

.. math::

   X_j = f_j + \epsilon_j

Multiplicative:

.. math::

   X_j = f_j \cdot (1 + \epsilon_j')

Heteroskedastic:

.. math::

   X_j = f_j + \sigma_j(X_{\mathrm{Pa}(j)}) z, \quad z \sim \mathcal{N}(0,1)

with registered :math:`\sigma_j(\cdot)` choices:

* ``abs_first_parent``
* ``abs_parent_plus_const``
* ``mean_abs_plus_const``

Endogenous Binary Nodes
-----------------------

Binary children use a logistic link on the latent signal:

.. math::

   \eta_j = f_j(X_{\mathrm{Pa}(j)}) + \epsilon_j

.. math::

   \Pr(X_j=1 \mid X_{\mathrm{Pa}(j)}) = \sigma(\eta_j), \quad
   \sigma(t)=\frac{1}{1+e^{-t}}

.. math::

   X_j \sim \mathrm{Bernoulli}\!\left(\sigma(\eta_j)\right)

Endogenous Categorical Nodes
----------------------------

Two models are supported.

1. Logistic (multinomial softmax)

.. math::

   \ell_{jk} = b_{jk} + \sum_{p \in \mathrm{Pa}(j)} g_{jpk}(X_p)

.. math::

   \Pr(X_j=k \mid X_{\mathrm{Pa}(j)}) =
   \frac{\exp(\ell_{jk})}{\sum_{m=0}^{K-1} \exp(\ell_{jm})}

where :math:`g_{jpk}` depends on parent type:

* continuous/binary parent: linear contribution per class
* categorical parent: class-specific lookup via parent-category weight matrix

2. Threshold (continuous-to-categorical)

.. math::

   s_j = \sum_{p \in \mathrm{Pa}(j)} w_{jp} X_p

.. math::

   X_j = \mathrm{digitize}(s_j; \tau_{j1}, \dots, \tau_{j(K-1)})

If thresholds are not provided, defaults are fixed from a theoretical Gaussian
quantile grid (optionally shifted/scaled by ``threshold_loc`` and
``threshold_scale``), not from realized sample quantiles.

Compatibility Matrix
--------------------

.. list-table:: Supported combinations
   :header-rows: 1
   :widths: 16 24 28 32

   * - Child type
     - Parent types
     - Structural model
     - Noise / link
   * - Continuous
     - Continuous, binary, categorical, or mixed
     - ``linear``, ``polynomial``, ``interaction``, ``stratum_means``
     - ``additive``, ``multiplicative``, ``heteroskedastic``
   * - Binary
     - Continuous, binary, categorical, or mixed
     - ``linear``, ``polynomial``, ``interaction``, ``stratum_means``
     - Latent signal + noise, then logistic link and Bernoulli draw
   * - Categorical
     - Continuous, binary, categorical, or mixed
     - ``categorical_model = logistic`` or ``categorical_model = threshold``
     - Softmax sampling (logistic) or threshold digitization

For random structural weights, additional controls are:
``random_weight_low``, ``random_weight_high``, and ``random_weight_min_abs``.
The same ``random_weight_min_abs`` exclusion is applied to auto-sampled
categorical logistic weights as well.

Categorical parents in metric forms
-----------------------------------

Using categorical parents with ``linear``, ``polynomial``, or ``interaction``
is blocked by default (``categorical_parent_metric_form_policy = "error"``),
because treating category codes as metric values can distort the intended DGP.

Set ``categorical_parent_metric_form_policy = "stratum_means"`` to auto-redirect
such cases to ``stratum_means``.

Stratum means reproducibility
-----------------------------

For ``stratum_means`` with multiple categorical parents, all strata are
pre-enumerated and assigned means upfront, ensuring stable DGP parameters even
for rare/unseen strata in a particular sample.

CI Oracle (Ground Truth)
------------------------

If ``simulation_params.store_ci_oracle = true``, the simulator stores conditional
independence truth values from DAG d-separation:

.. math::

   X \perp\!\!\!\perp Y \mid S \iff S \text{ is a d-separator of } X \text{ and } Y \text{ in } G

for conditioning sets up to ``ci_oracle_max_cond_set``.

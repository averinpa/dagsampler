# Simulator TODO

## Open Bugs

### High priority

- **Mixed-parent nodes unhandled for `policy='stratum_means'`**: When a node has both categorical and continuous parents and `categorical_parent_metric_form_policy='stratum_means'` is set, `_apply_functional_form` redirects to `stratum_means`, which then raises an error because it requires all parents to be categorical. Fix: either (a) support mixed parents in `stratum_means` by adding a linear term over continuous parents on top of the categorical stratum mean; or (b) make the policy redirect check whether all parents are categorical first and raise a clear error if not.

### Medium priority

- **Threshold default cut-points may produce imbalanced classes**: `threshold_loc` and `threshold_scale` default to fixed constants `0.0` and `1.0`, placing cut-points at N(0,1) quantiles. If the actual score distribution has a different scale, classes will be heavily imbalanced. Fix: sample `threshold_scale` from `rng_structure` (e.g. uniform over a reasonable range), or derive it from the parent weight magnitudes.

### Low priority

- **No interaction terms in categorical logistic model**: Logits are a sum of individual parent contributions — no cross-parent interaction terms. Consider adding an `interaction` option to the categorical logistic model.

---

## Planned Work

### Hardening and cleanup

- Add docstrings and type hints to all public classes and methods
- Add comprehensive pytest suite covering: noise models, graph generation, mixed-type edge cases, high cardinality, extreme sparsity, imbalanced categories

### Validation

- Validate that known CI relations are recoverable from generated mixed-type data
- Sanity-check marginal distributions, cross-type dependencies, categorical proportions
- Verify backward compatibility: existing experiment configs still work
- Produce validation notebook with diagnostic plots

# Simulator TODO

## Open Bugs

### High priority

- **Mixed-parent nodes not supported**: A continuous/binary child with both categorical and continuous parents cannot be generated — `stratum_means` requires all-categorical parents, and metric forms (`linear`, `polynomial`, `interaction`) are blocked when any parent is categorical. The error message is now clear, but the case is simply unsupported. Fix: support mixed parents in `stratum_means` by computing a linear term over continuous parents and adding it to the categorical stratum mean, i.e. `output = stratum_mean[cat_key] + Σ w_p * x_p` for continuous parents.

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

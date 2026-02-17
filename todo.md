## Q2: Extend Simulator + Implementation (May – Jul 2026)

**Objective:** Extend the existing causal simulator (`/Users/pavelaverin/Projects/simulation`) into a proper Python package with mixed-type support. Implement all CI tests. Run preliminary experiments to validate the setup.

**Note:** The core simulator already exists with DAG generation, continuous + binary data, multiple functional forms/noise models, config-driven experiments, and PC algorithm integration. Weeks 13–15 extend it rather than building from scratch.

### Week 13 (May 11–17) — Extend simulator: multi-level categorical variables

**DO**
- Add categorical variables with configurable cardinality (2, 3, 5, 10, 20 levels)
- Implement logistic structural equations for categorical nodes (parents determine category probabilities)
- Add cross-type mechanisms:
  - Continuous → categorical (threshold-based)
  - Categorical → continuous (stratum-specific means)
- Add ground truth CI relation storage (oracle for Type I / power evaluation)

**OUTPUT**
- Extended simulator with full mixed-type support

---

### Week 14 (May 18–24) — Package hardening and cleanup

**DO**
- Fix `eval()` security issue — replace with function registry
- Add docstrings and type hints to all public classes/methods
- Add comprehensive pytest suite (noise models, graph generation, edge cases, mixed types)
- Rename package if needed (more distinctive than `causal_simulator`)
- Clean up pyproject.toml: proper metadata, versioning, dependencies

**OUTPUT**
- Publication-ready Python package

---

### Week 15 (May 25–31) — Validation of extended simulator

**DO**
- Validate: recover known CI relations from generated mixed-type data
- Sanity-check marginal distributions, cross-type dependencies, categorical proportions
- Test edge cases: high cardinality, extreme sparsity, imbalanced categories
- Verify backward compatibility: existing experiment configs still work

**OUTPUT**
- Validation notebook with diagnostic plots

---

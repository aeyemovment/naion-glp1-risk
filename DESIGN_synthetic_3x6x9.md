# Synthetic digital-twin mixing experiment — 3×6×9 → ∞ lattice

Research prototype. synthetic_only applies to twin rows; real rows = published-count
reconstruction from PubMed. compliance_ref=Addendum_5_DT9. NOT for clinical use.

## Stage 0 — feature selection (prerequisite)
Train the PubMed-only models on 10 disjoint random 10% folds. Rank features by mean |SHAP|
across the 10 folds and keep those stable across folds (in the top-k in >= 8/10 folds).
Only these features define the digital-patient profile.

## Lattice (adapted from Grok brain `run_cusp_dt_3x6x9_infinity_antiblackhole_meiosis.py`,
policy CUSP-DT-3X6X9-ANTIBH-MEIOSIS-001: 3 parents × 6 levels × 9 variants = 162 cells → ∞)

| Axis | Count | Meaning here |
|---|---|---|
| 3 parents | 3 | twin generators: (P1) independent marginals calibrated to published aggregates; (P2) Gaussian copula keeping feature correlations; (P3) structural risk model, with outcome drawn from SHAP-selected features and their effects |
| 6 levels | 6 | twin volume relative to the real training size: 1×, 3×, 6×, 9×, 27×, 100× |
| 9 variants | 9 | synthetic share of the training mix: 10, 20, …, 90 % (0 % and 100 % run as anchors) |

Each cell is crossed with the real held-out fraction h ∈ {10, 20, …, 100 %}.
- The rows not held out are the real data available for training and for fitting the generator.
- h = 100 % means no real training data. Generators are then calibrated only on external
  priors (RCT-pooled data, meta-analyses, case reports), and the model is tested on all real
  PubMed rows.
- The evaluation set is always real, held-out PubMed data. No twin is ever scored.
- No leakage: generators are fit only on each cell's real training split.

**→ ∞ horizon:** the best cells (top AUROC, with calibration, by Brier score, as tie-breaker)
seed a child lattice. The child uses a finer grid around that cell's ratio and volume
(e.g. ±5 % share, a finer volume ladder). Recursion repeats until the gain is under 0.005
AUROC or depth reaches 3. In practice "∞" means recursive refinement to convergence, not
unbounded compute.

## Readouts
- AUROC, PR-AUC, Brier score, calibration slope/intercept, and SHAP rank agreement with the
  real-only model (Spearman).
- Heatmap: synthetic share × real held-out fraction, one per volume level and generator.
- Best real/synthetic mix, plus the point where adding more twins stops helping (diminishing
  returns).

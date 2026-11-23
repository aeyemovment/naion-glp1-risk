# Model card - NAION risk in GLP-1RA users (PubMed-only models) v1.0.0

**Research prototype. NOT for clinical, diagnostic, treatment-selection or regulatory use.** compliance_ref=Addendum_5_DT9.

- Training data: count-weighted pseudo-individual data reconstructed from 11 published cohorts (3,271,499 patients; 1,425 NAION/ION events). No patient-level health records were used.
- Features (29 columns): therapy_dpp4i, therapy_glp1ra_class, therapy_no_glp1ra, therapy_other_antidiabetic, therapy_other_antiobesity, therapy_sema_or_tirz, therapy_semaglutide, therapy_sglt2i, indication_iih, indication_obesity, indication_t2d, region_nordic, region_uk, region_us, source_type_claims, source_type_ehr_network, source_type_ehr_primary_care, source_type_national_registry, source_type_single_center, source_type_va_ehr, outcome_def_ion, outcome_def_naion, glp1ra_exposed, followup_years, mean_age, pct_female, followup_years_missing, mean_age_missing, pct_female_missing. Values are arm- or study-level (ecological), not individual measurements.
- Files: `xgb_pubmed_only.json/.ubj` (XGBoost 300 trees, depth 3, lr 0.05; load with `xgboost.Booster().load_model`),
  `logreg_pubmed_only.json` (L2 logistic regression on standardized features: p = sigmoid(intercept + coef . (x - mu)/sd)), `rf_pubmed_only.joblib` (scikit-learn RF).
- Validation (from the paper): XGBoost AUROC 0.656 +/- 0.011 over ten disjoint 10% training folds; leave-one-study-out AUROC ~0.53; PPV at top-10% threshold ~0.087% vs prevalence 0.044%.
- Known limitations: discrimination is driven mainly by study setting; transport to unseen settings is near chance; individual-level risk factors (cup-to-disc ratio, sleep apnoea, hypertension) are absent.
- Intended use: research enrichment and hypothesis generation only.

# Data dictionary

## real/reconstructed_patient_counts.csv (count-weighted pseudo-individual patient data)
| column | meaning |
|---|---|
| study | PMID of the source cohort |
| glp1ra_exposed | 1 = GLP-1RA (or GIP/GLP-1RA) arm, 0 = comparator arm |
| therapy | semaglutide, glp1ra_class, sema_or_tirz (exposed) or sglt2i, dpp4i, no_glp1ra, other_antidiabetic, other_antiobesity (comparator) |
| indication | t2d, obesity, iih (study population) |
| region | us, nordic, uk |
| source_type | ehr_network, claims, va_ehr, national_registry, ehr_primary_care, single_center |
| outcome_def | naion, or ion where only ischemic optic neuropathy was reported |
| followup_years, mean_age, pct_female | study-level values (median-imputed; *_missing = 1 when imputed) |
| y | 1 = NAION/ION event, 0 = no event |
| n | number of patients with this exact row (weights) |

Each row is a covariate pattern; expanding rows n times gives individual pseudo-patients. Values are arm- or study-level, not individual measurements.

## real/extracted_*.csv
One row per extracted unit (patient, study arm/contrast, signal, pooled estimate). Every row carries source_pmid,
evidence_level (fulltext / abstract / webfetch) and evidence_quote (<=200 characters). Missing = not reported.

## synthetic/twins_*.csv (synthetic_only = true)
Individual synthetic patients with the same columns as the reconstructed data plus twin_id, generator, prior_source.
- P1_marginal: features sampled independently from training marginals; risk = therapy-stratum rate
- P2_copula: joint covariate patterns + Gaussian-copula jitter on numeric features; risk = Beta-posterior draw of the source pattern
- P3_structural: P2 features; risk from a logistic model on SHAP-stable features
- prior_source = real: fitted to all reconstructed counts; prior: published pooled estimates only (no real outcome data)
Synthetic records are not real patients and must never be presented or analysed as such.

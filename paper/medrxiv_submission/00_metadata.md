# medRxiv submission metadata

**Title:** A Foundation Model for GLP-1 Receptor Agonist-Associated NAION Risk Stratification from Reconstructed Evidence in PubMed-Indexed Publications

**Article type:** New Results / original research (machine-learning prediction study using published aggregate data and synthetic data)

**Subject category (choose one):** Ophthalmology (primary); alternative: Health Informatics

**Keywords:** nonarteritic anterior ischemic optic neuropathy; GLP-1 receptor agonists; semaglutide; machine learning; synthetic data; digital twins; graph neural networks; risk prediction; TRIPOD+AI

**Abstract (246 words; paste as plain text):**

Background. Observational studies and pharmacovigilance reports link glucagon-like peptide-1 receptor agonists (GLP-1RAs), particularly semaglutide, to nonarteritic anterior ischemic optic neuropathy (NAION), but no tool stratifies individual risk. Whether machine learning (ML) on published evidence, with or without synthetic digital twins, can do so is unknown.
Methods. We extracted data from 73 of 169 PubMed publications and reconstructed count-weighted patient-level data from 11 non-overlapping cohorts (3,271,499 patients; 1,425 events). Logistic regression, random forest, XGBoost, graph neural networks and ensembles were trained on 10% and tested on 90%, across 10 disjoint 10% folds, across training fractions of 10% to 90%, and by leave-one-study-out validation. Synthetic twins were added in a factorial grid. SHAP values, permutation tests and Benjamini-Hochberg correction were used.
Results. Within studies, GLP-1RA exposure was associated with NAION (odds ratio 1.59 (95% CI, 1.42 to 1.77)); semaglutide 1.73 (95% CI, 1.44 to 2.08); obesity indication 2.70 (95% CI, 1.64 to 4.44). XGBoost discriminated modestly (AUROC 0.656 ± 0.011; observed/expected 1.00) but fell to 0.534 in unseen studies. At the top-10% threshold, PPV was 0.087% against a prevalence of 0.044%. Neither graph or convolutional networks, ensembles nor added vascular-score features outperformed PubMed-only XGBoost; synthetic digital twins mainly regularized weaker learners, and the best hybrid XGBoost gained 0.0015 AUROC. Simulated addition of optic-disc imaging raised AUROC to about 0.82 to 0.84.
Conclusions. Published evidence supports a reproducible exposure signal but only modest, setting-dependent risk discrimination. The score suits research enrichment, not treatment selection; patient-level prospective data are required.

**Licence (choose on medRxiv):** CC BY-NC 4.0 (non-commercial)

**Files to upload**
1. `01_manuscript_IEEE.pdf` — MAIN manuscript, IEEE two-column format
   (alternative single-column version: `01_manuscript_preprint.pdf`)
2. `02_Supplementary_Workbook_S1.xlsx` — all 511 feature combinations × models, odds ratios, Shapley values
3. `03_TRIPOD_AI_checklist.docx` — reporting checklist
4. Data/code: https://github.com/aeyemovment/naion-glp1-risk (release v1.0.0) — cited in the data-availability statement
5. `04_IEEE_LaTeX_Overleaf.zip` — LaTeX source (not uploaded to medRxiv; for Overleaf editing)

**Author:** Kemar E. Green — 1 Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, USA; 2 Johns Hopkins Data Science and AI Institute, Baltimore, MD, USA; 3 NeuroAgent AI, Inc., Baltimore, MD, USA. Corresponding: kemarearlgreen@neuroagentai.org

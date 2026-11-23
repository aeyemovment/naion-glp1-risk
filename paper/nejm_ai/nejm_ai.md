---
title: "Machine Learning on Reconstructed Published Evidence to Stratify Risk of Nonarteritic Anterior Ischemic Optic Neuropathy with GLP-1 Receptor Agonists: Supervised, Graph, and Digital-Twin Approaches"
subtitle: "Original Article — draft for author review (NEJM AI format)"
date: "November 22, 2026"
---

[Author list — to be completed and approved by Dr. Kemar Green before submission]  
Corresponding author: Dr. Kemar Green, [affiliation, address, e-mail]

**Research-prototype notice.** This work uses only published aggregate data and computationally generated synthetic records (synthetic_only twins; research_prototype=true; compliance reference Addendum 5, DT#9). The models are not validated for clinical, diagnostic, treatment-selection or regulatory use, and no patient-level health information was accessed.

## Abstract

**Background.** Observational studies and pharmacovigilance reports link glucagon-like peptide-1 receptor agonists (GLP-1RAs), particularly semaglutide, to nonarteritic anterior ischemic optic neuropathy (NAION), but no tool stratifies individual risk. Whether machine learning (ML) on published evidence, with or without synthetic digital twins, can do so is unknown.

**Methods.** We extracted data from 73 of 169 PubMed publications and reconstructed count-weighted patient-level data from 11 non-overlapping cohorts (3,271,499 patients; 1,425 events). Logistic regression, random forest, XGBoost, graph neural networks and ensembles were trained on 10% and tested on 90%, across 10 disjoint 10% folds, across training fractions of 10% to 90%, and by leave-one-study-out validation. Synthetic twins were added in a factorial lattice. SHAP values, permutation tests and Benjamini-Hochberg correction were used.

**Results.** Within studies, GLP-1RA exposure was associated with NAION (odds ratio 1.59 (95% CI, 1.42 to 1.77)); semaglutide 1.73 (95% CI, 1.44 to 2.08); obesity indication 2.70 (95% CI, 1.64 to 4.44). XGBoost discriminated modestly (AUROC 0.656 ± 0.011; observed/expected 1.00) but fell to 0.534 in unseen studies. At the top-10% threshold, PPV was 0.087% against a prevalence of 0.044%. Neither graph or convolutional networks, ensembles nor added vascular-score features outperformed PubMed-only XGBoost; synthetic digital twins mainly regularized weaker learners, and the best hybrid XGBoost gained 0.0015 AUROC. Simulated addition of optic-disc imaging raised AUROC to about 0.82 to 0.84.

**Conclusions.** Published evidence supports a reproducible exposure signal but only modest, setting-dependent risk discrimination. The score suits research enrichment, not treatment selection; patient-level prospective data are required.

## Introduction

Nonarteritic anterior ischemic optic neuropathy (NAION) is the most common acute optic nerve disease of adults older than 50 years [@p9186104], with an annual incidence of roughly 2 to 10 per 100,000 in that age group [@p8032479; @p9186104]. Vision loss is usually permanent, and the fellow eye is affected in about 15 to 21% of patients over several years [@p12208242; @p38358828]. The leading anatomic risk factor is a small, crowded optic disc [@p6514298; @p3684223; @p38211825], and NAION is attributed to infarction of the laminar and retrolaminar optic nerve head, with nocturnal hypotension, impaired autoregulation and a compartment syndrome in a crowded disc among proposed contributors [@p19596112]. No treatment is proven [@p25763979; @p35270411]; decompression surgery was ineffective and possibly harmful [@p7844872], and evidence that aspirin protects the fellow eye is limited to conflicting observational studies [@p9427177; @p9186127; @p12208242].

Glucagon-like peptide-1 receptor agonists (GLP-1RAs) and the dual GIP/GLP-1 receptor agonist tirzepatide produce large reductions in glycated hemoglobin and body weight [@p33567185; @p35658024] and reduce cardiovascular events [@p27633186; @p37952131]. Since 2024, cohort studies have reported hazard ratios of about 1.8 to 2.4 for NAION with semaglutide [@p38958939; @p39696569; @p41678180; @p40146102], whereas other cohorts and meta-analyses found no consistent increase [@p39491755; @p39976940; @p41692115]. Absolute risks are low [@p41678180; @p42132708], and proposed mechanisms — rapid glycemic lowering, blood-pressure and volume changes, and intracranial-pressure effects — remain hypotheses [@p40055951; @p41857480; @p42866022; @p42547311]. Consensus statements recommend shared decision-making and call for identifying which patients are susceptible [@p42132708; @p41058255; @p41022666].

No individual risk tool exists, and patient-level data remain siloed. We assessed whether machine learning applied to reconstructed published evidence can stratify NAION risk among GLP-1RA users, which features matter, and whether graph neural networks, ensembles, convolutional networks, synthetic digital twins or transfer from non-users improve on standard models.

## Methods

### Data sources and extraction

We searched PubMed for GLP-1RA or GIP/GLP-1RA terms combined with optic neuropathy terms and retrieved 169 records, 60 with open-access full text. Three extraction agents worked in parallel under a fixed schema: patient-level rows from case reports, arm-level counts and effect estimates from observational studies, and signal statistics and pooled estimates from pharmacovigilance studies, pooled trials and meta-analyses. Every value carries its PMID, an evidence level (full text, abstract or publisher page) and a supporting quotation; unreported values were left missing. One retracted article was excluded [@p41144820; @p42468029].

### Reconstruction of patient-level data

For each of 11 cohorts with event counts in both arms we selected one primary, non-overlapping contrast (Table 1); overlapping Danish registry analyses and a case-control study were excluded from reconstruction. Each arm became two rows (events, non-events) weighted by their counts, carrying arm-level therapy and exposure and study-level indication, region, data source, outcome definition, follow-up, mean age and percentage female (study medians imputed with missingness indicators). Because splitting patients uniformly at random is equivalent to binomial or multinomial thinning of these counts, every split below operates on individual patients without enumerating millions of rows.

### Models and validation

We trained L2-regularized logistic regression, random forest and XGBoost [@chen2016] with count weights. Validation used (i) a single 10% training / 90% test split; (ii) ten disjoint folds, each training on a different random 10% of patients and testing on the remaining 90%; (iii) training fractions from 10% to 90% in 10-point steps with 10 repeats each; and (iv) leave-one-study-out validation. We report AUROC, PR-AUC, Brier score, calibration slope and observed-to-expected (O/E) ratio, and sensitivity, specificity, accuracy, PPV and NPV at three thresholds fixed on training data (Youden-optimal, top 10%, top 20% of predicted risk), with net benefit [@vickers2006].

### Graph neural networks

Each distinct covariate pattern became a node; edges joined each node to its k nearest neighbours in standardized feature space and linked the exposed and comparator arms of the same study. A GraphSAGE network [@hamilton2017] was trained with a count-weighted binomial loss on training counts only. We varied depth (1 to 6 layers), neighbourhood size (5, 10, 20), edge weighting (uniform or Gaussian kernels) and the weight of same-study links, and compared learned edge weights through graph attention [@velickovic2018]; configurations were selected on folds 0 to 4 and confirmed on untouched folds 5 to 9. Attributions used integrated gradients [@sundararajan2017].

### Synthetic digital twins

Twins were generated in a lattice adapted from a 3×6×9 design: three generators (independent marginals; Gaussian copula preserving feature correlations; a structural model on SHAP-stable features) and their combinations, three prior sources (fitted to the real training split, published external priors only, or a blend), six volumes (1 to 100 times the real training size) and synthetic shares of 10% to 100%, crossed with real training fractions of 0% to 90%, four models and 10 repeats. Twins were never used for testing, and generators never saw test data. Recursive refinement around the best cell continued while AUROC improved by at least 0.005.

### Ensembles

All combinations of two or more of logistic regression, random forest, XGBoost and the GNN were combined by averaging, by weights from inner cross-validation, and by stacking with a logistic meta-learner fitted on inner out-of-fold predictions, and a cross-family ensemble averaged the best PubMed-only, hybrid and synthetic-only models.

### Statistical analysis

Within-study effects were estimated with count-weighted logistic regression with study fixed effects; marginal associations used cluster-robust standard errors by study. SHAP values [@lundberg2017] were tested against null distributions from 500 refits with events reallocated at random across arms, globally and within studies. Published ratio estimates were pooled with DerSimonian-Laird random effects [@dersimonian1986]. AUROC was tested against 0.5 with the Hanley-McNeil method [@hanley1982]; models were compared with the Nadeau-Bengio corrected resampled t-test [@nadeau2003] on paired folds and with exact McNemar tests; sensitivity, specificity, PPV, NPV and accuracy were tested against chance with exact binomial tests. P values within each family were adjusted with the Benjamini-Hochberg procedure [@benjamini1995]. Reporting follows TRIPOD+AI [@collins2024].

**Stage 15 — compute, energy and token use.** Energy was estimated with the Green Algorithms method (runtime × [cores × per-core power + memory × 0.3725 W/GB] × PUE) for an Apple M3 Pro laptop (PUE 1.0, package power 25 to 45 W), with run times taken from each stage's log where available, and converted to CO2e at 0.10 to 0.70 (central 0.367) kgCO2e/kWh. Inference time was measured per 1,000 predictions. Tokens consumed by the AI assistant were recorded where metered.

**Stage 16 — what-if: multimodal ocular imaging.** For each reconstructed patient, imaging features were drawn conditional on the observed outcome (P[feature | NAION] ∝ prior × odds ratio; P[feature | no NAION] = prior), independently across features: cup-to-disc ratio categories with the published control distribution and odds ratios (fundus photography with 25% and OCT with 5% adjacent-category misclassification), and hypothetical optic disc drusen (prevalence 2%, odds ratio 2 to 8), OCT angiography hypoperfusion (20%, 1.5 to 3) and serial on-treatment imaging change (5%, 2 to 10), with sleep apnoea (27%, 3.28) as a non-imaging comparator. XGBoost was trained with and without these features on five patient-level 10%/90% folds for each of three generation seeds, and the number of events needed to estimate AUROC within ±0.03 in a prospective cohort was derived from the Hanley-McNeil variance.

## Results

### Evidence base

Of 169 records, 73 publications contributed data: 29 observational studies, 17 case reports or series (18 NAION patients), 11 pharmacovigilance studies and 16 pooled-trial or meta-analytic reports (Figure 1). Reconstruction yielded 3,271,499 patients and 1,425 events (prevalence 0.044%) from 11 cohorts in the United States, the Nordic countries and the United Kingdom (Table 1). No source provided patient-level imaging or video; case reports described fundus photography and optical coherence tomography only as figures.

### Associations

With study fixed effects, GLP-1RA exposure was associated with NAION (odds ratio 1.59 (95% CI, 1.42 to 1.77); q<0.001). Estimates were 1.73 (95% CI, 1.44 to 2.08) for semaglutide, 2.70 (95% CI, 1.64 to 4.44) when the indication was obesity and 1.62 (95% CI, 1.45 to 1.82) for type 2 diabetes. The association depended on the comparator: 1.81 (95% CI, 1.54 to 2.11) against SGLT2 inhibitors and 2.35 (95% CI, 1.74 to 3.16) against other antidiabetic drugs, but 1.03 (95% CI, 0.57 to 1.85) against DPP-4 inhibitors and 1.14 (95% CI, 0.94 to 1.39) against untreated comparators. In patients with idiopathic intracranial hypertension the direction reversed (0.43 (95% CI, 0.22 to 0.84)). The exposure effect was larger in studies with older participants (interaction odds ratio 1.99 (95% CI, 1.30 to 3.04) per 10 years; q=0.002) and did not vary with the percentage of women (q=0.943). Random-effects pooling of published estimates gave a ratio of 1.34 (95% CI, 1.16 to 1.55; I² = 87%) (Figure 3). Patients in case reports were younger than exposed cohort participants (median 47 vs 60 years; q=0.016); the excess of men was not significant (q=0.274).

### Discrimination, calibration and explainability

Trained on 10% of patients, XGBoost reached an AUROC of 0.657 on the remaining 90% with calibration slope 0.97; across ten disjoint 10% training folds the AUROC was 0.656 ± 0.011 and O/E 1.00 (Figure 2). Logistic regression (0.616) and random forest (0.646) did not differ significantly from XGBoost (all corrected q≥0.27). AUROC rose with training fraction to a plateau near 0.67 at 40% to 60%. In unseen studies, AUROC fell to 0.534. SHAP rankings were stable across folds; permutation tests confirmed signal for exposure (q=0.004) and therapy (q=0.015), while data source and region — markers of setting rather than patient risk — carried the largest attributions.

### Threshold performance

At the top-10% threshold, XGBoost had sensitivity 0.35, specificity 0.82, PPV 0.087% and NPV 99.97% (positive likelihood ratio 2.01); at the Youden threshold sensitivity was 0.59 and specificity 0.61. Sensitivity and PPV exceeded chance (P<0.001), whereas specificity did not exceed the value expected at the same flag rate and accuracy remained below the no-information rate of 99.96%. Net benefit exceeded treat-all and treat-none strategies only at threshold probabilities below 0.1%. No metric differed between models (all q≥0.99).

### Graph neural networks, ensembles and digital twins

The GraphSAGE GNN reached an AUROC of 0.649 ± 0.014 across the 10 folds, not different from XGBoost on identical splits (difference -0.0077; P=0.779), with O/E 1.01. In leave-one-study-out validation its AUROC was 0.531 (P=0.034 vs 0.5). Integrated-gradient attributions ranked therapy, data source and region highest, mirroring SHAP.

Nested tuning over 366 GNN configurations selected a SAGE network of depth 6 with k=20 neighbours, uniform edge weighting and same-study weight 5.0; on the untouched folds it differed from the default network by 0.0006 AUROC (P=0.948).

Among 11 member combinations × 3 combiners, the best ensemble (cross_family(pubmed+hybrid+synthetic), mean) reached AUROC 0.657, a difference of 0.0010 from the best single model (q=1.000).

In the XGBoost lattice (5,310 fits), no hybrid cell improved AUROC over PubMed-only training (0 of 162 cells with a confidence interval above zero); the best hybrid changed AUROC by 0.0009 (95% CI, -0.0004 to 0.0022), synthetic-only training with twins fitted to real data lost 0.010, and twins built only from published priors reached AUROC values near 0.55. Adding twins beyond about 9 to 27 times the real training size reduced performance.

Across the full factorial grid (45,360 configurations), 4452 hybrid or synthetic configurations improved discrimination over PubMed-only training after Benjamini-Hochberg correction.

Table 2 summarizes the main model families.

### Transport, fine-tuning and convolutional networks

**Background-risk model transported to GLP-1RA users.** Trained on 1,983,956 non-GLP patients (740 events), background models discriminated within non-users (AUROC 0.626 to 0.666) but much less among 1,287,543 GLP-1RA users (685 events): AUROC 0.533 to 0.597 (best, ensemble). Observed events among GLP-1RA users exceeded background predictions (O/E 1.41 to 1.63), an independent estimate of excess risk consistent with the within-study odds ratio. Traditional and setting features therefore explain little of which GLP-1RA users develop NAION.

**Fine-tuning.** The best update (pooled refit; pretraining synthetic, fine-tuning hybrid, 10% of GLP-1RA patients) reached AUROC 0.618 in held-out GLP-1RA users with O/E 1.08. 49 configurations improved on the pretrained background model after Benjamini-Hochberg correction. Recalibration corrected under-prediction without changing discrimination, as expected; synthetic fine-tuning data did not outperform real data.

**Convolutional network.** The 1D-CNN reached AUROC 0.653 ± 0.011 across ten folds, not different from XGBoost (difference -0.0037; P=0.782), and 0.514 in leave-one-study-out validation. Convolutions add nothing when features lack spatial or sequential structure. Two- and three-dimensional CNNs and multimodal fusion were not trainable: the corpus contains one usable modality (structured tabular data derived from text), imaging only as published figures for about six patients, and no video.

### Incidence, rare-event and anomaly models

**Incidence.** NAION occurred at 32.0 per 100,000 person-years among GLP-1RA users and 17.9 among comparators (incidence rate ratio 1.79; 95% CI, 1.61 to 1.99; Mantel-Haenszel ratio stratified by study 1.73), a rate difference of 14.1 per 100,000 person-years. Both rates lie within or above the published background range for adults aged 50 years or older [@p8032479; @p9186104]. The ratio was 1.84 in type 2 diabetes and 0.43 in idiopathic intracranial hypertension, and Poisson models with study fixed effects showed heterogeneity of the exposure effect by study age band (P=<0.001), indication (P=<0.001), drug (P=0.002), comparator (P=0.026) and follow-up (P=<0.001) but not by the percentage of women (P=0.756). Because age is a study-level attribute and person-years were reported by only two studies, these strata describe settings rather than individual patients.

**Rare-event and anomaly-detection models.** A Poisson rate model with a person-time offset (AUROC 0.652), King-Zeng corrected logistic regression (0.655) and incidence-anchored XGBoost (0.657) matched standard XGBoost (0.656); class weighting required prior correction to restore calibration, and a focal-loss network was miscalibrated. Anomaly detectors trained only on patients without NAION performed near chance (AUROC 0.413 to 0.575), because patients with and without NAION from the same study arm share identical features in reconstructed aggregate data; anomaly methods need individual-level covariates.

### Detection within GLP-1RA users and non-users

**Performance within GLP-1RA users and within non-users.** Trained on mixed data, every model discriminated better among non-users (AUROC 0.577 to 0.674) than among GLP-1RA users (0.511 to 0.600; best, xgb). The within-group difference in AUROC was significant after correction for 0 of 9 models. Together with the under-prediction of a background model in GLP-1RA users, this indicates that the features available in published data capture background NAION risk but little of what determines which GLP-1RA users develop NAION.

### Vascular score features and a proposed NAION-adapted score

**Validated vascular score as a feature.** Only three ABCD2/ABCD3-I items could be derived: age 60 years or older (7 of 11 studies, from study mean age), blood pressure (5 of 11, using hypertension prevalence as a proxy) and diabetes (11 of 11); clinical features, symptom duration, dual events and imaging are specific to transient ischemic attack. Adding the partial items to XGBoost changed AUROC by -0.0019 (P=0.917). The partial score alone, in studies with all items, reached AUROC 0.545.

**Proposed NAION-adapted points score.** Fitting point weights to the reconstructed data produced implausible protective signs for hypertension, diabetes and GLP-1RA exposure, because these items vary only between studies; we therefore report this fit only as a demonstration of ecological bias. A literature-informed version assigned points from published patient-level ratios (male sex, hypertension, hyperlipidemia, diabetes, coronary heart disease and sleep apnea [@p34671609]) and from our within-study GLP-1RA estimates (indication-specific), with age carried from the ABCD2 convention: male sex 2 points, hypertension 1 point, hyperlipidemia 1 point, diabetes 2 points, coronary heart disease 2 points, sleep apnea 5 points, age ge60 1 point, cdr le 0.20 post exposure fellow eye 16 points, cdr 0.21 0.40 post exposure fellow eye 8 points, glp1ra t2d 2 points, glp1ra obesity 4 points. On setting-level data it ranked cohorts poorly (AUROC 0.417; difference from XGBoost -0.240), as expected when settings with more vascular risk factors recorded fewer events; the score requires prospective, patient-level calibration, including disc morphology.

**Limitations of the partial score.** only A, B and D of ABCD2/ABCD3-I are derivable; C, duration, dual-TIA and imaging items are TIA-specific; A and B are study-level proxies (age from study mean with assumed SD 12; hypertension diagnosis for BP >= 140/90); A available for 7/11 and B for 5/11 studies; remaining values imputed and flagged; both arms of a study share the same A and B, so the score cannot separate exposed from unexposed patients within a study; any apparent importance reflects between-setting differences, not individual patient risk.

**Cup-to-disc ratio.** No source reported a pre-treatment (pre-GLP-1RA) cup-to-disc ratio. In the case reports, disc status was recorded at or after NAION diagnosis, usually in the unaffected fellow eye. The largest source, a 16-center case-control study of 1,812 unaffected fellow eyes, found a fellow-eye ratio of 0.20 or less in 54% of patients with NAION versus 9% of controls (odds ratio 46.3; 95% CI, 22.9 to 93.4; ratio 0.21 to 0.40: 6.8; 3.5 to 13.2) and no difference between semaglutide users, other GLP-1RA users and unexposed patients [@p42556433]. We therefore entered the cup-to-disc ratio as the highest-weighted item of the proposed score but labelled it a post-exposure fellow-eye measurement: disc anatomy is structurally stable and the fellow eye is the accepted proxy, yet this is weaker evidence than a baseline measured before GLP-1RA initiation, and it could not be validated in the cohort data. The same limitation applies to NAION without GLP-1RA exposure: the classic and recent disc-at-risk studies measured the cup-to-disc ratio after the event, in fellow eyes or in affected eyes after edema resolved [@p6514298; @p3684223; @p38211825; @p42556433], and none had a measurement taken before NAION. Because both groups share this post-event design, the comparison between exposed and unexposed NAION is balanced, but no source supports a truly prospective (pre-event) estimate of disc-related risk. Baseline optic disc photography or OCT before GLP-1RA initiation in a prospective cohort would close this gap.

**With versus without ABCD features across all experiments.** Over 455 paired comparisons using identical splits and seeds, adding the partial ABCD items changed AUROC by 0.0016 on average; 0 comparisons differed after Benjamini-Hochberg correction.

### Feature combinations and the compounding hypothesis

**Feature combinations and the compounding hypothesis.** Across all 511 combinations of nine feature groups, the largest Shapley contributions to XGBoost AUROC came from setting (0.0350), duration (0.0290), sex (0.0270), drug (0.0212). For the hypothesis that individually negligible features contribute in combination, no feature group met the prespecified criteria; no pair showed significant positive interaction. For an optional-input calculator, a single XGBoost trained with random feature-group dropout differed from a bank of dedicated sub-models by -0.0121 AUROC on average across input patterns.

### Compute footprint and imaging what-if

**Compute, energy and token use.** All model training and inference ran locally on one laptop (Apple M3 Pro, 11 cores, 18 GB). Across every experiment the analyses used 7.6 wall-clock hours and 62.6 core-hours (87% of run time measured from logs, the rest estimated), consuming an estimated 0.25 kWh (range 0.19 to 0.31) and emitting 92 g CO2e at the US average grid intensity (range 19 to 215 g), about 0.5 km of car travel. The largest single consumer was the full-factorial twins (2.4 h). Inference was negligible: logreg 0.75 ms, rf 26.68 ms, xgb 1.69 ms per 1,000 patients. The AI assistant used for retrieval, extraction and drafting consumed at least 764,562 tokens in the four metered extraction and literature agents; tokens of the main session were not metered, and assistant inference energy, which occurs in remote data centres, is not included in these totals.

**What-if: adding ocular imaging (simulation).** Imaging values were simulated for the reconstructed patients conditional on their observed outcome. Adding cup-to-disc ratio from fundus photographs, with published odds ratios [@p42556433] and 25% photographic misclassification, raised AUROC from 0.653 to 0.770; OCT-measured ratio reached 0.818. Adding hypothetical effects for optic disc drusen, OCT angiography perfusion and serial on-treatment imaging reached 0.840 (range across hypothetical effect sizes 0.825 to 0.876), and 0.854 with sleep apnoea added; sleep apnoea alone reached 0.708. Positive predictive value at the top-10% threshold rose from 0.098% to 0.262%, still below 1% because NAION is rare. A prospective cohort would need about 263 NAION events, roughly 604,000 GLP-1RA initiators at the observed incidence, to estimate AUROC within ±0.03; a case-control design enriched for NAION would need far fewer imaged patients. These figures are simulations that assume conditional independence of features and, for drusen, OCT angiography and serial imaging, effect sizes that have not been measured.

## Discussion

Published evidence on GLP-1RAs and NAION contains a consistent exposure signal: within the same studies, exposed patients had roughly 1.6 times the odds of NAION, and the estimate was higher for semaglutide and for obesity indications. Machine learning on the same evidence, however, discriminated only modestly, and most of what it learned was where a patient was treated rather than who the patient was. Discrimination collapsed toward chance in studies the model had not seen, and the strongest attributions went to data source and region. More flexible learners — graph neural networks, ensembles and synthetic digital twins — did not change this picture.

The threshold metrics show what this means in practice. Sensitivity and PPV beat chance, so the score enriches for cases roughly twofold, which is an advantage for selecting patients for closer surveillance or for enrolment in research. Absolute PPV, however, stayed below 0.1%: about one in 1,100 flagged patients would develop NAION. Specificity added nothing beyond the flag rate, accuracy was uninformative because the outcome is rare, and the high NPV mainly reflected the low prevalence. Net benefit was positive only at threshold probabilities suited to low-risk actions such as a dilated examination or optical coherence tomography before treatment, counselling, or slower dose escalation, consistent with recent consensus recommendations [@p42132708; @p41058255].

The intended clinical question is whether a validated score can identify patients who should start antiplatelet therapy. These data cannot answer it. Risk prediction is not treatment-effect prediction; none of the extracted cohorts or case reports recorded antiplatelet exposure, NAION is not primarily thromboembolic, and the evidence for aspirin after NAION is limited. At the observed PPV, treating every flagged patient would expose roughly a thousand people to bleeding risk per event that occurs, before any benefit is assumed. Answering the question requires prospective validation with patient-level predictors (disc morphology, sleep apnoea, nocturnal hypotension, dose escalation) followed by a risk-stratified trial with bleeding as a co-primary safety outcome.

Synthetic twins helped only weak learners. In the full factorial experiment, thousands of hybrid configurations improved logistic regression, consistent with regularization, but the best hybrid XGBoost gained only 0.0015 AUROC; twins built from published priors alone produced little more than chance discrimination, and fine-tuning a background model worked only with real GLP-1RA data. Twins cannot create patient-level information that the source data do not contain.

The limiting factor is the feature set, not the learner. In simulation, adding cup-to-disc ratio from fundus photographs or OCT, with published effect sizes, raised AUROC from about 0.65 to 0.77 to 0.82, and plausible additional imaging reached about 0.84; even then positive predictive value stayed below 1%. A prospective cohort of GLP-1RA initiators with baseline optic-disc imaging, which no published study has reported, is therefore the most informative next step. The entire analysis used about 0.25 kWh of local compute, so the environmental cost of the modelling itself is negligible relative to the data collection it motivates.

We see this work as one module of a proposed multimodal neuro-ophthalmic foundation model rather than a foundation model itself. Such a system would combine a published-evidence risk component like the one developed here, encoders for optic-disc photography and OCT, and separately pretrained components such as an oculomotor world model (GrokEyeWorld, which is trained on synthetic eye-movement dynamics and shares no variables with the present data). No transfer between these components was performed or evaluated here; it would require patients with linked risk factors, imaging and oculomotor recordings.

This study has limitations. Predictors were arm- or study-level, so associations with age, sex and setting are ecological. Reconstructed patients share covariate patterns, which makes thresholds coarse. Outcome definitions differed (one cohort reported ischemic optic neuropathy), TriNetX cells below 10 were masked, several studies were available only as abstracts, and overlapping registries were handled by selection rather than by individual linkage. No imaging was available, so multimodal and computer-vision models could not be trained. All results are retrospective and in silico.

In conclusion, ML on published evidence confirms a modest, reproducible association between GLP-1RA exposure and NAION and yields a research-grade enrichment score, not a treatment-selection tool. Patient-level prospective data are the necessary next step.

**Use of AI tools.** Literature retrieval, data extraction, analysis code, figures and the first draft of this manuscript were produced with an AI assistant (Claude, Anthropic) working under the direction of the authors. Every extracted value carries a source PMID and evidence quote. The authors are responsible for verifying all content before submission. AI tools are not authors.

## Tables and figures

Table: Cohorts reconstructed into count-weighted patient-level data (one primary, non-overlapping contrast per study).

| Study                       | Setting           | Region   | Population   | Exposure                   | Comparator         | Exposed, events/N   | Comparator, events/N   |
|:----------------------------|:------------------|:---------|:-------------|:---------------------------|:-------------------|:--------------------|:-----------------------|
| Hathaway 2024 [@p38958939]  | Single center     | US       | T2D          | Semaglutide                | Other antidiabetic | 46/555              | 23/1,134               |
| Grauslund 2024 [@p39696569] | National registry | Nordic   | T2D          | Semaglutide                | No GLP-1RA         | 67/106,454          | 151/317,698            |
| Simonsen 2025 [@p40098249]  | National registry | Nordic   | T2D          | Semaglutide                | SGLT2i             | 8/16,860            | 7/34,153               |
| Hsu 2025 [@p40146102]       | EHR network       | US       | T2D          | Semaglutide                | Other antidiabetic | 45/174,584          | 29/174,584             |
| Wang 2025 [@p40788646]      | EHR network       | US       | T2D          | Semaglutide or tirzepatide | Other antidiabetic | 35/79,699           | 19/79,699              |
| Ramsey 2025 [@p40788647]    | EHR network       | US       | T2D          | GLP-1RA class              | No GLP-1RA         | 96/185,066          | 79/185,066             |
| Tesfaye 2026 [@p41104517]   | Claims            | US       | T2D          | GLP-1RA class              | SGLT2i             | 270/482,912         | 144/482,912            |
| Heberer 2026 [@p41678180]   | VA EHR            | US       | T2D          | Semaglutide                | SGLT2i             | 30/11,478           | 143/90,883             |
| Noh 2026 [@p41701611]       | Primary-care EHR  | UK       | T2D          | GLP-1RA class              | DPP-4i             | 14/106,858          | 53/416,369             |
| Al-Harbi 2026 [@p41994958]  | EHR network       | US       | IIH          | GLP-1RA class              | No GLP-1RA         | 12/15,559           | 28/15,560              |
| Ueda 2026 [@p42441962]      | National registry | Nordic   | T2D          | GLP-1RA class              | SGLT2i             | 62/107,518          | 64/185,898             |

Table: Model performance by family. AUROC values for 10×10% folds are mean ± SD; thresholds fixed on training data; PPV and NPV as percentages. LOSO, leave-one-study-out; O/E, observed-to-expected events.

| Model                                                           | Training data         | AUROC 10/90   | AUROC 10×10% folds            | AUROC LOSO   | O/E   | Sens/Spec top 10%   | PPV/NPV top 10%   |
|:----------------------------------------------------------------|:----------------------|:--------------|:------------------------------|:-------------|:------|:--------------------|:------------------|
| Logistic regression (L2)                                        | PubMed only           | 0.627         | 0.616 ± 0.010                 | 0.528        | 1.01  | 0.26/0.85           | 0.075%/99.96%     |
| Random forest                                                   | PubMed only           | 0.642         | 0.646 ± 0.014                 | 0.531        | 1.00  | 0.31/0.85           | 0.096%/99.96%     |
| XGBoost                                                         | PubMed only           | 0.657         | 0.656 ± 0.011                 | 0.534        | 1.00  | 0.35/0.82           | 0.087%/99.97%     |
| Mean ensemble (LR+RF+XGB)                                       | PubMed only           | 0.657         | 0.656 ± 0.011                 | —            | —     | 0.31/0.86           | 0.101%/99.97%     |
| GNN (GraphSAGE, 2 layers)                                       | PubMed only           | 0.647         | 0.649 ± 0.014                 | 0.531        | 1.01  | 0.35/0.82           | 0.086%/99.97%     |
| GNN tuned (SAGE, depth 6, k=20, uniform, study weight 5.0)      | PubMed only           | —             | 0.657 (untouched folds)       | 0.541        | 1.03  | —                   | —                 |
| xgb — pubmed only                                               | PubMed only           | —             | 0.667 (all fractions)         | —            | 1.00  | —                   | —                 |
| xgb — hybrid (P2+P3, prior priors, 100×, 10% synthetic)         | Hybrid                | —             | 0.669 (all fractions)         | —            | 1.01  | —                   | —                 |
| ensemble — synthetic only (P2, real priors, 9×, 100% synthetic) | Synthetic only        | —             | 0.664 (all fractions)         | —            | 1.01  | —                   | —                 |
| ensemble — synthetic only, 0% real (P2, 100×)                   | Published priors only | —             | 0.569 (all real data as test) | —            | 1.11  | —                   | —                 |
| Ensemble cross_family(pubmed+hybrid+synthetic) (mean)           | PubMed only           | —             | 0.657 ± 0.010                 | —            | 1.01  | —                   | —                 |
| Ensemble logreg+xgb (mean)                                      | PubMed only           | —             | 0.657 ± 0.008                 | —            | 1.00  | —                   | —                 |

![Study flow and validation design.](paper/figures/fig1_study_flow.png){width=100%}

![Discrimination and calibration of PubMed-only models: (top) receiver-operating-characteristic, precision-recall and decile calibration on the 90% hold-out after training on 10%; (middle) AUROC and PR-AUC across ten disjoint 10% training folds; (bottom) AUROC, calibration-in-the-large and SHAP-rank stability across training fractions of 10% to 90%.](paper/figures/fig2_performance.png){width=100%}

![(Top) Within-study odds ratios for NAION with study fixed effects (blue, q<0.05 after Benjamini-Hochberg correction). (Bottom) SHAP importance (mean ± SD over ten folds) for three model types.](paper/figures/fig3_effects_explainability.png){width=100%}

The Supplementary Appendix (the full preprint) contains all additional tables and figures: data inventory and modalities; case-level phenotyping and meta-regression; every P value (within-study and marginal associations, SHAP permutation tests, meta-analysis, demographics, ML experiments); threshold metrics for all models; training-fraction sweeps; GNN depth and edge-weight tuning; ensembles; the full digital-twin factorial; non-GLP to GLP transport and fine-tuning; CNN and imaging feasibility; incidence, rare-event and anomaly models; ABCD2 features and the with/without-ABCD ablation; all 511 feature combinations (Supplementary Workbook S1); compute and energy accounting; the imaging what-if simulation; and the TRIPOD+AI checklist.

## Data and code availability

Extraction files (every value with PMID and evidence quotation), reconstructed count-weighted patient-level data, synthetic digital-twin datasets, analysis code and result files are provided in the data release accompanying the preprint [repository DOI to be added].

## References
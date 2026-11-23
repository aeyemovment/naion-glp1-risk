---
title: "Toward a Foundation Model for NAION Risk Stratification in GLP-1 Receptor Agonist Users: Machine Learning, Graph Neural Networks and Synthetic Digital Twins Built from PubMed-Indexed Evidence"
subtitle: "Original Article"
date: "November 23, 2026"
---

| **Kemar Earl Green, DO, FANA**^1,2^
| *^1^Johns Hopkins Data Science and AI Institute, Baltimore, MD, USA*
| *^2^NeuroAgent AI, Inc., Baltimore, MD, USA*
| kemarearlgreen@neuroagentai.org

**Research-prototype notice.** This work uses only published aggregate data and computationally generated synthetic records (synthetic_only twins; research_prototype=true; compliance reference Addendum 5, DT#9). The models are not validated for clinical, diagnostic, treatment-selection or regulatory use, and no patient-level health information was accessed.

## Abstract

**Background.** Observational studies and pharmacovigilance reports link glucagon-like peptide-1 receptor agonists (GLP-1RAs), particularly semaglutide, to nonarteritic anterior ischemic optic neuropathy (NAION), but no tool stratifies individual risk. Whether machine learning (ML) on published evidence, with or without synthetic digital twins, can do so is unknown.

**Methods.** We extracted data from 73 of 169 PubMed publications and reconstructed count-weighted patient-level data from 11 non-overlapping cohorts (3,271,499 patients; 1,425 events). Logistic regression, random forest, XGBoost, graph neural networks and ensembles were trained on 10% and tested on 90%, across 10 disjoint 10% folds, across training fractions of 10% to 90%, and by leave-one-study-out validation. Synthetic twins were added in a factorial grid. SHAP values, permutation tests and Benjamini-Hochberg correction were used.

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

Twins were generated in a factorial grid [@montgomery2017] with local refinement [@bergstra2012]: three generators (independent marginals; Gaussian copula preserving feature correlations [@patki2016]; a structural model on SHAP-stable features) and their combinations, three prior sources (fitted to the real training split, published external priors only, or a blend), six volumes (1 to 100 times the real training size) and synthetic shares of 10% to 100%, crossed with real training fractions of 0% to 90%, four models and 10 repeats. Twins were never used for testing, and generators never saw test data. Local refinement around the best cell continued while AUROC improved by at least 0.005.

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

In the XGBoost grid (5,310 fits), no hybrid cell improved AUROC over PubMed-only training (0 of 162 cells with a confidence interval above zero); the best hybrid changed AUROC by 0.0009 (95% CI, -0.0004 to 0.0022), synthetic-only training with twins fitted to real data lost 0.010, and twins built only from published priors reached AUROC values near 0.55. Adding twins beyond about 9 to 27 times the real training size reduced performance.

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

We describe this work as a foundation machine learning model in the sense of a base layer: it fixes the outcome definition, the reconstructed evidence base, the validation protocol and a benchmark (AUROC about 0.66 within settings, about 0.53 across settings) against which every later module must be judged. It is not a large pretrained model. The intended next layers are encoders for optic-disc photography and OCT and separately pretrained components such as an oculomotor world model, a learned generative model of eye-movement dynamics adapted from Ha and Schmidhuber [@ha2018], trained on synthetic eye-movement data and sharing no variables with the present data, combined in a multimodal neuro-ophthalmic foundation model. No transfer between components was performed or evaluated here; it requires patients with linked risk factors, imaging and oculomotor recordings.

This study has limitations. Predictors were arm- or study-level, so associations with age, sex and setting are ecological. Reconstructed patients share covariate patterns, which makes thresholds coarse. Outcome definitions differed (one cohort reported ischemic optic neuropathy), TriNetX cells below 10 were masked, several studies were available only as abstracts, and overlapping registries were handled by selection rather than by individual linkage. No imaging was available, so multimodal and computer-vision models could not be trained. All results are retrospective and in silico.

## Conclusion

This study asked whether published evidence can stratify the risk of NAION in GLP-1RA users, which features carry that risk, and which model architecture predicts it best.

**Risk features.** Within studies, GLP-1RA exposure was associated with NAION (OR 1.59, 95% CI 1.42 to 1.77, P<0.001, q<0.001). The highest-risk subgroup was GLP-1RA use for obesity (OR 2.70, 95% CI 1.64 to 4.44, P<0.001, q<0.001), followed by semaglutide (OR 1.73, 95% CI 1.44 to 2.08, P<0.001, q<0.001) and comparison with SGLT2 inhibitors (OR 1.81, 95% CI 1.54 to 2.11, P<0.001, q<0.001); the exposure effect was larger in studies with older participants (OR 1.99, 95% CI 1.30 to 3.04, P=0.001, q=0.002 per 10 years of mean age). Exposure was the most stable single feature: its odds ratio exceeded 1 in all 256 feature combinations that contained it (median 1.59, range 1.04 to 1.92).

**Feature combinations.** The best-discriminating combination was age + exposure + setting + sex (XGBoost, AUROC 0.660 ± 0.008), no better than all nine groups together (0.655 ± 0.012). The largest contribution came from study setting (region, data source and outcome definition; Shapley value 0.035 AUROC, q=0.059), which describes where a patient was studied rather than the patient; among the remaining features, the largest contribution came from follow-up duration (time at risk) (Shapley value 0.029 AUROC, q=0.006). No pair of feature groups was synergistic, so the hypothesis that weak features compound into strong risk was not supported.

**Best architecture.** PubMed-only XGBoost discriminated best (AUROC 0.656 ± 0.011 over ten folds), but its advantage over random forest (+0.011 AUROC, P=0.745) and logistic regression (+0.041, P=0.218) was not significant, and graph, convolutional, ensemble, anomaly-detection and synthetic digital-twin variants did not improve on it. In studies the model had not seen, AUROC fell to 0.534 (P=0.021 versus chance).

**Answer.** Published evidence supports a reproducible association between GLP-1RA exposure and NAION, strongest with obesity indications, but it does not support individual risk prediction good enough to select patients for treatment, including antiplatelet therapy. The model is a research tool for enriching cohorts, and its main product is a benchmark and protocol for the patient-level, imaging-linked prospective data that the question requires.

**Use of AI tools.** Literature retrieval, data extraction, analysis code, figures and the first draft of this manuscript were produced with an AI assistant (Claude, Anthropic) working under the direction of the authors. Every extracted value carries a source PMID and evidence quote. The authors are responsible for verifying all content before submission. AI tools are not authors.

## Tables and figures

Table: Cohorts reconstructed into count-weighted patient-level data (one primary, non-overlapping contrast per study). Abbreviations and variable names are defined in the Nomenclature appendix.

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

Table: Model performance by family. AUROC values for 10×10% folds are mean ± SD; thresholds fixed on training data; PPV and NPV as percentages. LOSO, leave-one-study-out; O/E, observed-to-expected events. Abbreviations and variable names are defined in the Nomenclature appendix.

| Model                                                           | Training data         | AUROC 10/90   | AUROC 10×10% folds            | AUROC LOSO   | O/E   | Sens/Spec top 10%   | PPV/NPV top 10%   |
|:----------------------------------------------------------------|:----------------------|:--------------|:------------------------------|:-------------|:------|:--------------------|:------------------|
| Logistic regression (L2)                                        | PubMed only           | 0.627         | 0.616 ± 0.010                 | 0.528        | 1.01  | 0.26/0.85           | 0.075%/99.96%     |
| Random forest                                                   | PubMed only           | 0.642         | 0.646 ± 0.014                 | 0.531        | 1.00  | 0.31/0.85           | 0.096%/99.96%     |
| XGBoost                                                         | PubMed only           | 0.657         | 0.656 ± 0.011                 | 0.534        | 1.00  | 0.35/0.82           | 0.087%/99.97%     |
| Mean ensemble (LR+RF+XGB)                                       | PubMed only           | 0.657         | 0.656 ± 0.011                 | —            | —     | 0.31/0.86           | 0.101%/99.97%     |
| GNN (GraphSAGE, 2 layers)                                       | PubMed only           | 0.647         | 0.649 ± 0.014                 | 0.531        | 1.01  | 0.35/0.82           | 0.086%/99.97%     |
| GNN tuned (SAGE, depth 6, k=20, uniform, study weight 5.0)      | PubMed only           | —             | 0.657 (untouched folds)       | 0.541        | 1.03  | —                   | —                 |
| xgb — pubmed only                                               | PubMed only           | —             | 0.667 (all fractions)         | —            | 1.00  | —                   | —                 |
| xgb — hybrid (copula + structural generators, published priors, 100×, 10% synthetic)         | Hybrid                | —             | 0.669 (all fractions)         | —            | 1.01  | —                   | —                 |
| ensemble — synthetic only (copula generator, real-data priors, 9×, 100% synthetic) | Synthetic only        | —             | 0.664 (all fractions)         | —            | 1.01  | —                   | —                 |
| ensemble — synthetic only, 0% real (copula generator, 100×)                   | Published priors only | —             | 0.569 (all real data as test) | —            | 1.11  | —                   | —                 |
| Ensemble cross_family(pubmed+hybrid+synthetic) (mean)           | PubMed only           | —             | 0.657 ± 0.010                 | —            | 1.01  | —                   | —                 |
| Ensemble logreg+xgb (mean)                                      | PubMed only           | —             | 0.657 ± 0.008                 | —            | 1.00  | —                   | —                 |

![Study flow and validation design. Abbreviations and variable names are defined in the Nomenclature appendix.](paper/figures/fig1_study_flow.png){width=100%}

![Discrimination and calibration of PubMed-only models: (top) receiver-operating-characteristic, precision-recall and decile calibration on the 90% hold-out after training on 10%; (middle) AUROC and PR-AUC across ten disjoint 10% training folds; (bottom) AUROC, calibration-in-the-large and SHAP-rank stability across training fractions of 10% to 90%. Abbreviations and variable names are defined in the Nomenclature appendix.](paper/figures/fig2_performance.png){width=100%}

![(Top) Within-study odds ratios for NAION with study fixed effects (blue, q<0.05 after Benjamini-Hochberg correction). (Bottom) SHAP importance (mean ± SD over ten folds) for three model types. Abbreviations and variable names are defined in the Nomenclature appendix.](paper/figures/fig3_effects_explainability.png){width=100%}

The Supplementary Appendix (the full preprint) contains all additional tables and figures: data inventory and modalities; case-level phenotyping and meta-regression; every P value (within-study and marginal associations, SHAP permutation tests, meta-analysis, demographics, ML experiments); threshold metrics for all models; training-fraction sweeps; GNN depth and edge-weight tuning; ensembles; the full digital-twin factorial; non-GLP to GLP transport and fine-tuning; CNN and imaging feasibility; incidence, rare-event and anomaly models; ABCD2 features and the with/without-ABCD ablation; all 511 feature combinations (Supplementary Workbook S1); compute and energy accounting; the imaging what-if simulation; and the TRIPOD+AI checklist.

## Data and code availability

Code, trained model weights, extraction files (every value with PMID and evidence quotation), reconstructed count-weighted patient-level data and synthetic digital-twin datasets are available at https://github.com/aeyemovment/naion-glp1-risk (release v1.0.0). A research-only web calculator is at https://naion-glp1-calculator.vercel.app. Copyright (c) 2026 NeuroAgent AI, Inc. All materials are released for non-commercial research use only, with attribution (citation of this article): code and model weights under a research-only, non-commercial licence included in the repository, data under CC BY-NC 4.0. Commercial use, and clinical, diagnostic or treatment-selection use, are not permitted without prior written permission (kemarearlgreen@neuroagentai.org).

## Supplementary Appendix: Nomenclature


Abbreviations are also defined at first use in the text. Variable and label names are those printed in tables and figures; "…" stands for any suffix or prefix (for example, therapy_… covers therapy_semaglutide and therapy_sglt2i).

Table: Abbreviations.

| Abbreviation | Definition |
|:--|:--|
| ABCD² | Age, Blood pressure, Clinical features, Duration of symptoms and Diabetes score for stroke risk after transient ischemic attack |
| ABCD³-I | ABCD² extended with dual transient ischemic attack and Imaging findings |
| Acc | accuracy |
| AE | autoencoder (anomaly detector) |
| AI | artificial intelligence |
| AION | anterior ischemic optic neuropathy |
| AUROC | area under the receiver operating characteristic curve |
| B | number of permutations |
| BP | blood pressure |
| CDR | cup-to-disc ratio |
| CI | confidence interval |
| CNN | convolutional neural network (CNN-1D, one-dimensional) |
| CO2e | carbon dioxide equivalent |
| CV | cross-validation |
| DL | deep learning |
| DPP-4i | dipeptidyl peptidase-4 inhibitor |
| EBGM | empirical Bayes geometric mean (pharmacovigilance disproportionality) |
| EHR | electronic health record |
| Ens. | ensemble (mean of member models) |
| FAERS | US Food and Drug Administration Adverse Event Reporting System |
| FE | fixed effects |
| GAT | graph attention network |
| GB | gigabyte |
| GCIPL | ganglion cell-inner plexiform layer |
| GIP | glucose-dependent insulinotropic polypeptide |
| GLP-1 | glucagon-like peptide-1 |
| GLP-1RA | glucagon-like peptide-1 receptor agonist |
| GNN | graph neural network |
| Grad-CAM | gradient-weighted class activation mapping |
| GraphSAGE | graph sample-and-aggregate network |
| IC025 | lower 95% credibility bound of the information component (pharmacovigilance) |
| IForest | isolation forest (anomaly detector) |
| IIH | idiopathic intracranial hypertension |
| ION | ischemic optic neuropathy |
| IONDT | Ischemic Optic Neuropathy Decompression Trial |
| IPD | individual patient data |
| IRR | incidence rate ratio |
| I² | percentage of between-study variance due to heterogeneity |
| k | number of estimates (meta-analysis) or neighbours per node (graph) |
| L2 | ridge (squared-coefficient) penalty |
| LOSO | leave-one-study-out validation |
| LR | logistic regression |
| LR+ | positive likelihood ratio |
| MH | Mantel-Haenszel |
| ML | machine learning |
| MRI | magnetic resonance imaging |
| NAION | nonarteritic anterior ischemic optic neuropathy |
| NB | net benefit |
| NIR | no-information rate |
| NPV | negative predictive value |
| O/E | observed-to-expected event ratio |
| OCT | optical coherence tomography |
| OCTA | optical coherence tomography angiography |
| OR | odds ratio |
| OSA | obstructive sleep apnea |
| PC | principal component |
| PCA | principal component analysis |
| PMC | PubMed Central |
| PMID | PubMed identifier |
| PPV | positive predictive value |
| PR-AUC | area under the precision-recall curve |
| PRR | proportional reporting ratio |
| PUE | power usage effectiveness |
| PY | person-years |
| q | Benjamini-Hochberg adjusted P value |
| QPI-1007 | intravitreal small interfering RNA against caspase-2 (investigational) |
| RCT | randomized controlled trial |
| RF | random forest |
| RNFL | retinal nerve fiber layer |
| ROC | receiver operating characteristic |
| ROR | reporting odds ratio |
| SD | standard deviation |
| SE | standard error |
| Sens | sensitivity |
| SGLT2i | sodium-glucose cotransporter-2 inhibitor |
| SHAP | Shapley additive explanations |
| Spec | specificity |
| SUSTAIN-6 | Trial to Evaluate Cardiovascular and Other Long-term Outcomes with Semaglutide in Subjects with Type 2 Diabetes |
| SVM | support vector machine |
| T2D | type 2 diabetes |
| TIA | transient ischemic attack |
| TRIPOD+AI | Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis, AI extension |
| VA | US Department of Veterans Affairs |
| XGB | XGBoost (extreme gradient boosting) |

Table: Variable and label names used in tables and figures.

| Name | Definition |
|:--|:--|
| therapy_… | drug arm of the patient (one-hot): semaglutide; sema_or_tirz = semaglutide or tirzepatide; glp1ra_class = any or unspecified GLP-1RA; comparators sglt2i, dpp4i, no_glp1ra (no GLP-1RA), other_antidiabetic, other_antiobesity |
| indication_… | study population: t2d = type 2 diabetes, obesity = obesity without diabetes, iih = idiopathic intracranial hypertension |
| region_… | study region: us = United States, nordic = Nordic countries, uk = United Kingdom |
| source_type_… | data source: claims = insurance claims, ehr_network = multi-institution EHR network, ehr_primary_care = primary-care EHR, national_registry = national registry, single_center = single-center records, va_ehr = Veterans Affairs EHR |
| outcome_def_… | outcome definition: naion = NAION, ion = ischemic optic neuropathy (when NAION was not reported separately) |
| source_type | data source (see source_type_…) |
| outcome_def | outcome definition (see outcome_def_…) |
| case_control | case-control design |
| target_trial | target-trial emulation design |
| variable_level | naming pattern of one-hot features: variable name, underscore, level |
| glp1ra_exposed | 1 if the patient was in a GLP-1RA arm, 0 if in a comparator arm |
| followup_years | mean study follow-up (years) |
| mean_age | mean age of the study arm (years) |
| pct_female | percentage of women in the study arm |
| …_missing | 1 if the preceding variable was not reported and was replaced by the median across studies |
| semaglutide | semaglutide |
| sema_or_tirz | semaglutide or tirzepatide |
| glp1ra_class | any or unspecified GLP-1RA |
| no_glp1ra | no GLP-1RA (comparator) |
| sglt2i | SGLT2 inhibitor |
| dpp4i | DPP-4 inhibitor |
| other_antidiabetic | other glucose-lowering drug |
| other_antiobesity | other anti-obesity drug |
| ehr_network | multi-institution EHR network |
| ehr_primary_care | primary-care EHR |
| national_registry | national registry |
| single_center | single-center records |
| va_ehr | Veterans Affairs EHR |
| claims | insurance claims |
| any_glp1ra | any GLP-1RA |
| t2d_or_obesity | type 2 diabetes or obesity |
| exposed_vs_… | GLP-1RA exposure versus the named comparator (within-study contrast) |
| exposed_x_… | interaction of GLP-1RA exposure with the named study-level covariate |
| exposed_… | GLP-1RA exposure within the named drug or indication subgroup |
| age_per_10y | per 10-year higher mean age |
| female_per_10pct | per 10-percentage-point higher share of women |
| age_band | stratum of study mean age |
| female_band | stratum of the share of women |
| fu_band | stratum of follow-up duration |
| dataset_… | data source of a published estimate (e.g. danish = Danish registries, trinetx = TriNetX network, medicare = Medicare claims) |
| design_… | study design of a published estimate (cohort, target_trial = target-trial emulation, case_control, other) |
| comparator_cls_… | comparator class of a published estimate (sglt2, unexposed, other_antidiabetic, ...) |
| exposure_drug_… | exposure drug of a published estimate (semaglutide, any_glp1ra, tirzepatide, ...) |
| followup | reported follow-up of a published estimate (years) |
| followup_missing | 1 if follow-up was not reported |
| population | study population of a published estimate |
| adjusted_f | 1 if the estimate was covariate-adjusted |
| abcd_… | partial ABCD² component derived from study-level data: A = age, B = blood pressure (hypertension prevalence), D = diabetes |
| logreg | L2-penalized logistic regression |
| rf | random forest |
| xgb | XGBoost |
| gnn | graph neural network |
| cnn1d | one-dimensional convolutional neural network |
| ensemble | ensemble (mean of member predictions unless stated) |
| cross_family | ensemble combining PubMed-only, synthetic and hybrid models |
| focal_mlp | multilayer perceptron trained with focal loss |
| poisson_offset | Poisson regression with log person-time offset |
| logreg_kingzeng | logistic regression with King-Zeng rare-event correction |
| xgb_standard | XGBoost, unweighted |
| xgb_balanced_raw | XGBoost with class-balanced weights, uncorrected |
| xgb_balanced_priorcorrected | XGBoost with class-balanced weights, prior-corrected probabilities |
| xgb_incidence_anchored | XGBoost with predictions anchored to published incidence |
| xgb_continue | XGBoost fine-tuned by continued boosting |
| isolation_forest | isolation forest (anomaly detection) |
| one_class_svm | one-class support vector machine (anomaly detection) |
| lof_novelty | local outlier factor in novelty mode (anomaly detection) |
| gnn_joint | graph neural network trained jointly on both groups |
| pooled_refit | model refitted on pooled source and target data |
| pretrain_only | model trained on non-users only, applied unchanged |
| recal_intercept | intercept-only recalibration |
| recal_logistic | logistic (intercept and slope) recalibration |
| glp_only | model trained on GLP-1RA users only |
| internal_nonGLP | internal validation in non-users |
| external_GLP | external validation in GLP-1RA users |
| …_mean | mean across folds or repeats |
| …_sd | standard deviation across folds or repeats |
| auroc | AUROC |
| pr_auc | PR-AUC |
| cal_slope | calibration slope |
| o_e | observed-to-expected event ratio |
| lr_pos | positive likelihood ratio |
| flagged_frac | fraction of patients flagged as high risk |
| train_frac | fraction of patients used for training |
| T1_youden | threshold maximizing Youden's J (sensitivity + specificity - 1) |
| T2_top10pct | threshold flagging the top 10% of training risk |
| T3_top20pct | threshold flagging the top 20% of training risk |
| brier | Brier score (mean squared error of predicted risk) |
| uniform | graph edges all weighted 1 |
| gauss… | graph edges weighted by a Gaussian kernel of feature distance with the stated bandwidth |
| threshold | decision threshold (see T1_youden, T2_top10pct, T3_top20pct) |
| synthetic_only | 1 for computationally generated synthetic records |
| research_prototype | research-prototype flag (not for clinical use) |

## References
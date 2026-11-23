---
title: "Development of a Foundation Model for GLP-1 Receptor Agonist-Associated NAION Risk Stratification Using Reconstructed Evidence in PubMed-Indexed Publications"
subtitle: "Preprint with all experiments and results"
date: "November 23, 2026"
---

| **Kemar Earl Green, DO, FANA**^1,2,3^
| *^1^Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, USA*
| *^2^Johns Hopkins Data Science and AI Institute, Baltimore, MD, USA*
| *^3^NeuroAgent AI, Inc., Baltimore, MD, USA*
| kemarearlgreen@neuroagentai.org

**Research-prototype notice.** This work uses only published aggregate data and computationally generated synthetic records (synthetic_only twins; research_prototype=true; compliance reference Addendum 5, DT#9). The models are not validated for clinical, diagnostic, treatment-selection or regulatory use, and no patient-level health information was accessed.

## Abstract

**Background.** Observational studies and pharmacovigilance reports link glucagon-like peptide-1 receptor agonists (GLP-1RAs), particularly semaglutide, to nonarteritic anterior ischemic optic neuropathy (NAION), but no tool stratifies individual risk. Whether machine learning (ML) on published evidence, with or without synthetic digital twins, can do so is unknown.

**Methods.** We extracted data from 73 of 169 PubMed publications and reconstructed count-weighted patient-level data from 11 non-overlapping cohorts (3,271,499 patients; 1,425 events). Logistic regression, random forest, XGBoost, graph neural networks and ensembles were trained on 10% and tested on 90%, across 10 disjoint 10% folds, across training fractions of 10% to 90%, and by leave-one-study-out validation. Synthetic twins were added in a factorial grid. SHAP values, permutation tests and Benjamini-Hochberg correction were used.

**Results.** Within studies, GLP-1RA exposure was associated with NAION (odds ratio 1.59 (95% CI, 1.42 to 1.77)); semaglutide 1.73 (95% CI, 1.44 to 2.08); obesity indication 2.70 (95% CI, 1.64 to 4.44). XGBoost discriminated modestly (AUROC 0.656 ± 0.011; observed/expected 1.00) but fell to 0.534 in unseen studies. At the top-10% threshold, PPV was 0.087% against a prevalence of 0.044%. Neither graph or convolutional networks, ensembles nor added vascular-score features outperformed PubMed-only XGBoost; synthetic digital twins mainly regularized weaker learners, and the best hybrid XGBoost gained 0.0015 AUROC. Simulated addition of optic-disc imaging raised AUROC to about 0.82 to 0.84.

**Conclusions.** Published evidence supports a reproducible exposure signal but only modest, setting-dependent risk discrimination. The score suits research enrichment, not treatment selection; patient-level prospective data are required.

## Introduction

Nonarteritic anterior ischemic optic neuropathy (NAION) is the most common acute optic nerve disease of adults older than 50 years [@p9186104]. Population-based estimates of its annual incidence in people aged 50 years or older range from 2.30 per 100,000 in Missouri and Los Angeles County [@p8032479] to approximately 10 per 100,000 in Olmsted County, Minnesota, where incidence has remained stable over four decades [@p9186104; @p38358828]. The disorder presents as abrupt, painless monocular vision loss with optic disc edema [@p37987292], and visual symptoms are frequently first noticed on waking, as reported by 42% of patients in the Ischemic Optic Neuropathy Decompression Trial (IONDT) cohort [@p8906027]. About half of affected eyes present with near-normal visual acuity, and visual acuity and fields change mainly during the first 6 months [@p17698200]. The fellow eye is affected in roughly 15–21% of patients over several years [@p12208242; @p38358828], with diabetes and worse first-eye visual function predicting second-eye involvement [@p12208242; @p36502803], whereas recurrence in the same eye is uncommon [@p11704035]. Because the visual deficit is usually permanent and bilateral disease is not rare, even small changes in incidence carry meaningful population consequences.

The leading anatomic risk factor is a small, crowded optic disc with little or no physiologic cup, the so-called disc at risk [@p6514298; @p3684223]. In the 729-participant Quark207 trial cohort, a small fellow-eye cup-to-disc ratio was the most prevalent risk factor across countries, races and sexes, whereas cardiovascular risk factors were not typically more prevalent than in the general population [@p38211825]. Optic disc drusen are found in most younger patients with NAION and may act as an independent risk factor [@p32298654; @p31972711]. Systemic associations include hypertension, diabetes, hyperlipidemia, coronary heart disease, male sex and obstructive sleep apnea, as summarized in a meta-analysis of 49 studies [@p34671609] and in cohort and case-control studies [@p7977604; @p9366670; @p27659545]; sleep apnea carries one of the largest relative risks in pooled analyses [@p26443989; @p34671609]. Nocturnal arterial hypotension has been proposed as a final insult that lowers optic nerve head perfusion below a critical level in susceptible patients [@p8172267] and has been associated with same-eye recurrence [@p11704035]. Other reported precipitants include cataract surgery, after which relative risk rises within the first year although absolute risk remains low [@p36166807; @p32818451]; perioperative blood loss, hypotension and anemia [@p18358851]; and hemodialysis-related hypotension and anemia [@p11730278]. Drug associations remain debated: pooled observational data did not show a significant association with phosphodiesterase-5 inhibitors [@p31559705], although a case-crossover study reported a 2-fold increase in risk [@p26720519]; a causal link with amiodarone has never been firmly established [@p16148635]; and interferon alfa is classed as a "possible" association [@p20630599]. Evidence on smoking [@p17398323] and thrombophilia is inconsistent, with one case-control study finding no association with thrombophilic markers [@p10201595], while a managed-care cohort [@p27659545] and small case-control data [@p15834600] reported positive associations. The optic nerve head is supplied by the short posterior ciliary arteries, and NAION is attributed to infarction of the laminar and retrolaminar optic nerve head [@p19596112]. Proposed contributors include nocturnal hypotension, impaired autoregulation, vasculopathic occlusion and venous insufficiency [@p19596112]. Clinicopathologic and blood-flow data point to vasculopathy within or distal to the paraoptic branches, with diabetes the most consistently identified vasculopathic risk factor [@p12782932]. A widely cited model holds that ischemic axonal swelling within a crowded disc produces a compartment syndrome [@p19596112]; three-dimensional reconstruction of a human infarct did not correspond to a single vascular territory, consistent with this view [@p14522783]. Hayreh argued that vascular insufficiency of the optic nerve head underlies AION while noting conflicting interpretations of its blood supply and of measurement methods [@p11470451]. Rodent and primate photoembolic models reproduce disc edema followed by retinal ganglion cell loss [@p14507856; @p18326695], but much of the pathophysiology remains unknown [@p26959135].

No treatment for NAION is proven [@p25763979; @p35270411]. The IONDT, a randomized trial of optic nerve decompression surgery, was stopped early; surgery was not effective and may have been harmful [@p7844872], and 24-month follow-up confirmed no benefit [@p10865316]. Spontaneous improvement among patients in the careful follow-up group was more common than previously reported [@p7844872]. A large nonrandomized cohort reported better acuity outcomes with systemic corticosteroids [@p18404273], but a small randomized trial [@p29705054] and meta-analyses [@p31725628; @p35270411] did not show significant visual acuity benefit. A 725-participant sham-controlled trial of intravitreal QPI-1007 did not meet its primary endpoint [@p40816607], and evidence for levodopa comes from small or nonrandomized studies [@p8554083; @p10711891]. Antiplatelet evidence is limited to observational data for fellow-eye protection. Retrospective studies reported lower second-eye NAION with aspirin [@p9427177; @p10624433], but a larger retrospective cohort found possible short-term and little or no long-term benefit [@p9186127], and aspirin use was not associated with fellow-eye NAION in the prospectively followed IONDT cohort [@p12208242]. No randomized evidence was identified for aspirin in any risk-defined subgroup, including drug-associated NAION; this gap is relevant to whether antiplatelet therapy could be targeted at high-risk patients.

GLP-1 is an intestinal L-cell hormone that augments insulin and suppresses glucagon secretion, slows gastrointestinal motility and reduces food intake, and is inactivated within minutes by dipeptidyl peptidase IV [@p17928588; @p29617641; @p22945360]. GLP-1RAs range from short-acting agents (exenatide twice daily, lixisenatide) to once-daily liraglutide and once-weekly exenatide, dulaglutide and semaglutide, and oral semaglutide is also available [@p22945360; @p33068776]. Semaglutide has a half-life of approximately 1 week [@p27633186], and tirzepatide is a dual GIP and GLP-1 receptor agonist [@p35658024]. In placebo-controlled trials, semaglutide 2.4 mg reduced body weight by a mean 14.9% at 68 weeks [@p33567185], and tirzepatide reduced it by 15.0–20.9% at 72 weeks [@p35658024]. Tirzepatide lowered HbA1c by more than semaglutide 1 mg in type 2 diabetes [@p34170647]. Cardiovascular outcome trials showed reductions in major adverse cardiovascular events with liraglutide, semaglutide and dulaglutide [@p27295427; @p27633186; @p31189511; @p31422062], including in people with obesity without diabetes [@p37952131], and semaglutide reduced kidney outcomes in chronic kidney disease [@p38785209]. Across trials, GLP-1RAs modestly lower systolic blood pressure and slightly raise heart rate [@p26358202]. Ocular and neural findings are directly relevant to NAION. In SUSTAIN-6, semaglutide increased diabetic retinopathy complications (hazard ratio 1.76) [@p27633186], an effect attributed mainly to the magnitude and speed of early HbA1c reduction [@p29178519]. GLP-1 receptor expression in the normal human eye is low and confined to a small fraction of ganglion-cell-layer neurons [@p29707863]. GLP-1 receptor agonism lowered intracranial pressure in a rodent hydrocephalus model [@p28835515] and in a randomized trial of exenatide in idiopathic intracranial hypertension [@p36907221], although central receptor activation increased cerebrospinal fluid secretion and intracranial pressure in another rat study [@p40275284].

After a neuro-ophthalmology cohort reported higher NAION risk among patients prescribed semaglutide [@p38958939; @p40055951], a nationwide Danish cohort [@p39696569], a Danish nested case-control study [@p42126975], a US veterans target trial emulation [@p41678180] and US electronic health record cohorts [@p40146102; @p40788646] reported hazard ratios of about 1.8–2.4. In contrast, a multinational cohort found no significant association [@p39491755], and a 14-database network found mostly null active-comparator estimates with a modest self-controlled signal [@p39976940]. Meta-analyses range from a significant pooled risk [@p42060879; @p40810985] to signals confined to observational diabetes studies [@p41475544] or no consistent increase with low-certainty evidence [@p41692115]. Absolute risks are low [@p41678180; @p42132708]. The European Medicines Agency classifies NAION as a very rare adverse effect of semaglutide [@p41857480], and joint neuro-ophthalmology/ophthalmology and European multidisciplinary consensus statements recommend shared decision-making rather than class-wide restriction [@p42132708; @p41058255]. Proposed mechanisms remain hypotheses. They include rapid glycemic lowering analogous to early worsening of retinopathy [@p40055951; @p41857480], blood-pressure and nocturnal-pressure lowering [@p41857480], volume depletion from gastrointestinal effects or rapid weight loss [@p41678180; @p42773534], and impaired optic nerve head autoregulation [@p41678180]. Other proposals are intracranial-pressure-mediated changes in retinal perfusion [@p42335513; @p42866022], a "two-hit" model combining anatomic susceptibility with hypohydration [@p42547311], and exposure-dependent pharmacokinetic thresholds favoring high-exposure subcutaneous semaglutide [@p42770962]. Several of these proposals are speculative or mutually inconsistent; the systemic vascular effects of GLP-1RAs would be expected to reduce ischemic risk [@p42866022], and blood pressure, dehydration and autonomic effects are described as theoretical [@p42890826]. Crowded discs appear equally prevalent in exposed and unexposed NAION [@p42556433], and reviewers have called for identifying which patient characteristics (disc anatomy, blood-pressure change, weight loss, glycemic improvement) mark higher risk [@p41022666].

This evidence leaves several questions open: whether the signal is causal, whether it differs by agent, dose, indication and duration [@p42126975; @p42770962], and which patients are susceptible. The aim of this study was to assess whether machine learning applied to published evidence can stratify NAION risk among GLP-1RA users and which features matter.

Beyond conventional supervised learning, we examined whether graph neural networks that share information across similar patient profiles, ensembles, convolutional networks and synthetic digital twins improve on simpler models; whether a model of background NAION risk learned without any GLP-1RA exposure transports to GLP-1RA users; and whether fine-tuning such a model with real, synthetic or hybrid GLP-1RA data improves risk stratification. Every comparison was tested statistically with correction for multiple testing.

## Methods

### Data sources and extraction

We searched PubMed for GLP-1RA or GIP/GLP-1RA terms combined with optic neuropathy terms and retrieved 169 records, 60 with open-access full text. Three extraction agents worked in parallel under a fixed schema: patient-level rows from case reports, arm-level counts and effect estimates from observational studies, and signal statistics and pooled estimates from pharmacovigilance studies, pooled trials and meta-analyses. Every value carries its PMID, an evidence level (full text, abstract or publisher page) and a supporting quotation; unreported values were left missing. One retracted article was excluded [@p41144820; @p42468029].

### Reconstruction of patient-level data

For each of 11 cohorts with event counts in both arms we selected one primary, non-overlapping contrast (Reconstructed cohorts table, Section A); overlapping Danish registry analyses and a case-control study were excluded from reconstruction. Each arm became two rows (events, non-events) weighted by their counts, carrying arm-level therapy and exposure and study-level indication, region, data source, outcome definition, follow-up, mean age and percentage female (study medians imputed with missingness indicators). Because splitting patients uniformly at random is equivalent to binomial or multinomial thinning of these counts, every split below operates on individual patients without enumerating millions of rows.

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

### Detailed derivation of every dataset

**Search corpus.** A single PubMed query combined drug terms (semaglutide, tirzepatide, liraglutide, dulaglutide, exenatide, GLP-1, glucagon-like peptide) with outcome terms (NAION, ischemic/ischaemic optic neuropathy, optic neuropathy). Metadata, abstracts and PubMed Central open-access full texts, including embedded tables, were retrieved programmatically; publication types were recorded.

**Extraction datasets.** Four structured datasets were produced by three independent extraction agents working from the same rules: (1) a case-level table with one row per published patient (age, sex, drug, formulation, dose, recent escalation, time to onset, indication, comorbidities, disc morphology, drusen, dehydration, laterality, visual acuity converted to logMAR, imaging available, treatment and recovery); (2) a cohort-level table with one row per study, contrast and subgroup (design, data source, population, exposure, comparator, arm sizes and events, person-years, effect measure with CI, adjustment, follow-up and population characteristics); (3) a pharmacovigilance table (database, drug, formulation, subgroup, case counts, ROR, PRR, IC025, EBGM); and (4) a pooled-trial and meta-analysis table (pooled estimates by subgroup and risk factors named by consensus statements). Values not reported were left missing; every row carries the PMID, evidence level and a supporting quotation of at most 200 characters, and possible duplicate cohorts were flagged.

**Reconstructed patient-level dataset (pseudo-IPD).** From the cohort table we kept contrasts with event counts in both arms and selected one primary, non-overlapping contrast per study, preferring crude full-cohort counts and excluding overlapping national-registry analyses and a case-control design. Each arm produced an event row and a non-event row weighted by their counts. Arm-level variables were exposure and therapy; study-level variables were indication, region, data source, outcome definition, follow-up, mean age and percentage female, with medians imputed and missingness flagged. Because uniform random sampling of patients is equivalent to binomial (single split) or multinomial (k-way partition) thinning of counts, every split is an exact patient-level split.

**Non-GLP and GLP sub-datasets.** For transfer experiments the reconstructed data were divided into comparator-arm patients (no GLP-1RA) and exposed-arm patients. Background-risk models used only features defined in both populations (exposure and therapy excluded).

**Synthetic datasets (digital twins [@kamelboulos2021]).** Three generators produced twin profiles: a marginal generator sampled each feature independently from training marginals with risk equal to the therapy-stratum rate; a Gaussian-copula generator [@patki2016] sampled joint covariate patterns with copula jitter on numeric features and drew each twin's risk from the Beta posterior of its source pattern; a structural generator used the copula generator's features with risk from a logistic model restricted to SHAP-stable features. Generators were fitted to the real training split only (source = real), to published priors only (source = prior: a non-user baseline of about 1.35 per 10,000 person-years derived from a pooled incidence of 85 per 100,000 GLP-1RA users and a pooled ratio of 2.52 [@p42060879], with drug-specific ratios of 1.93 and 1.70 [@p42612051; @p41498755]), or to a 50/50 blend. Twin counts were distributed over a bank of 5,000 profiles by multinomial sampling and outcomes drawn binomially.

**Released datasets.** The search records, the four extraction tables, the reconstructed counts, an exact 1% patient-level expansion, the Introduction evidence table and eight synthetic sets of 200,000 twins each (three generators × two prior sources, plus non-GLP and GLP populations) are released with a data dictionary and checksums.

### Detailed description of every experiment

**Stage 1 — PubMed-only supervised learning.** Logistic regression (weights rescaled to the number of events so that the L2 penalty is meaningful), random forest and XGBoost were evaluated with a 10%/90% split, ten disjoint 10% training folds and leave-one-study-out validation; SHAP stability was summarized across folds. **Stage 1b** clustered case-report patients (k-means, Gaussian mixture, Ward; bootstrap co-assignment) and fitted a random-forest meta-regression of published log ratios with grouped cross-validation. **Stage 1c** repeated Stage 1 at training fractions of 10% to 90% (10 repeats each) and measured SHAP-rank agreement with Kendall's W.

**Stage 2 — synthetic-data grid with local refinement.** XGBoost was trained on mixtures defined by three generators, six volumes (1, 3, 6, 9, 27 and 100 times the real training size) and synthetic shares of 10% to 100%, crossed with real held-out fractions of 10% to 100% and three seeds (5,310 fits). The training set of size V×R contained (1−s)×V×R real patients (sampled without replacement when available, otherwise bootstrapped) and s×V×R twins. The grid was refined around the best cell with halving steps while AUROC improved by ≥0.005 (maximum depth 3).

**Stage 3 — full factorial.** Four models (logistic regression, random forest, XGBoost and their mean) trained on identical data per cell; real training fractions 0% to 90%; seven generator sets (every non-empty combination of the marginal, copula and structural generators); three prior sources; six volumes; ten shares; ten repeats. Differences from PubMed-only training on the same split were tested per configuration (one-sided paired t) with Benjamini-Hochberg correction; robust configurations were ranked across all training fractions under a calibration constraint (|O/E−1|<0.10, slope 0.9 to 1.1).

**Stage 4 — ensembles.** Eleven member subsets of logistic regression, random forest, XGBoost and the GNN, three combiners (mean, inner-CV AUROC weights, logistic stacking on inner out-of-fold logits) and a cross-family ensemble, under the same split paradigms.

**Stage 5 — inference.** Count-weighted logistic models (marginal with cluster-robust errors; within-study with study fixed effects, including exposure by drug, indication, comparator and study-level age, sex and follow-up); SHAP permutation tests (500 refits, global and within-study nulls); random-effects meta-analysis with subgroup tests; demographic contrasts (exact binomial, Wilcoxon, ratio-of-ratios z tests); and threshold metrics with tests against chance, McNemar and corrected resampled t comparisons, and net benefit.

**Stage 6 — graph neural networks.** GraphSAGE on a pattern graph (k-nearest neighbours plus same-study links), with integrated-gradient attributions, under all split paradigms; **Stage 6c** tuned depth (1 to 6), k, edge weighting and same-study weight (360 configurations) and learned attention weights (graph attention network, depths 1 to 6), selecting on folds 0 to 4, confirming on folds 5 to 9 and testing the top configurations by leave-one-study-out validation.

**Stage 7 — background-risk model with external validation in GLP-1RA users.** Models trained only on non-GLP patients (internal splits as above) were applied to all GLP-1RA-exposed patients to discriminate NAION from no NAION; observed-to-expected ratios with exact Poisson intervals quantify how much a background model under-predicts GLP-1RA users. A GLP-aware model trained on both arms served as reference.

**Stage 8 — fine-tuning.** Within each of five repeats, non-GLP patients were split 80/20 (pretraining pool, forgetting test) and GLP patients 50/10/40 (fine-tuning pool, validation, test). Pretraining sources (none, real, synthetic, hybrid) were crossed with fine-tuning sources (real, synthetic, hybrid) and GLP fractions (0%, 5%, 10%, 25%, 50%). Update methods were recalibration in the large, logistic recalibration, continued XGBoost boosting from the pretrained margin (rounds selected on validation), GNN joint training with GLP up-weighting (selected on validation), pooled refitting and GLP-only training. Three additional split schemes tested transport in each direction and the standard mixed split.

**Stage 10 — incidence, rare-event and anomaly models.** Incidence rates per 100,000 person-years used reported person-years where available and otherwise N × follow-up; rates were stratified by drug, indication, comparator, region, data source, study age band, percentage female and follow-up, with exact conditional confidence intervals for incidence rate ratios, Mantel-Haenszel ratios stratified by study and Poisson models with study fixed effects, exposure-by-stratum terms and a log person-time offset (Wald tests of heterogeneity). Rare-event learners were a Poisson rate model with offset, class-weighted XGBoost with prior correction, King-Zeng corrected logistic regression, a focal-loss network and incidence-anchored XGBoost recalibrated to observed incidence within exposure-by-age strata. Anomaly detectors (Isolation Forest, one-class SVM, Local Outlier Factor and an autoencoder) were trained on non-event patients of the training split only.

**Stage 11 — detection within GLP-1RA users and within non-users.** Every model (logistic regression, random forest, XGBoost, GNN, 1D-CNN, mean ensemble, Poisson rate model, Isolation Forest and autoencoder) was trained on mixed data in each of the ten folds and evaluated separately within the exposed and unexposed parts of the test set (AUROC, PR-AUC, O/E, calibration slope and threshold metrics at the training top-10% threshold); group differences were tested with corrected paired t tests and Benjamini-Hochberg correction.

**Stage 12 — validated vascular score and a proposed NAION-adapted score.** The ABCD2 score and its newest extension ABCD3-I were derived where possible (age 60 years or older as the expected share from study mean age with an assumed SD of 12 years; blood pressure from hypertension prevalence; diabetes from the study population) and used only as features. A points score was fitted to the data and, separately, built from published patient-level ratios and our within-study GLP-1RA estimates, with points equal to log ratio divided by log 1.28 (hypertension = 1 point), recalibrated on training data and validated with the same splits. **Stage 12b** repeated every experiment with and without the ABCD features using identical splits and seeds, so that all differences are paired.

**Stage 14 — feature combinations and the compounding hypothesis.** Because a calculator should accept whatever inputs are available, models were trained on every non-empty combination of nine feature groups (exposure, drug, indication, age, sex, duration, hypertension, diabetes and setting; 511 combinations) with logistic regression, random forest and XGBoost on all ten folds and with the GNN, 1D-CNN and mean ensemble on three folds. Exact Shapley values and pairwise Shapley interaction indices of AUROC quantified each group's contribution across all coalitions; the hypothesis that individually negligible groups contribute in combination was tested on fold-level values with corrected t tests and Benjamini-Hochberg correction. A single XGBoost trained with random feature-group dropout was compared with a bank of dedicated sub-models for each input pattern.

**Stage 9 — convolutional networks and multimodal feasibility.** A one-dimensional CNN treated the encoded feature vector as a sequence (two convolution layers, global max pooling and a skip connection) under the same paradigms. A two- and three-dimensional imaging pipeline (fundus and OCT volume encoders, late fusion with tabular features and Grad-CAM) was implemented and verified on random tensors, but could not be trained because no analysable images exist in the published corpus.

**Stage 15 — compute, energy and token use.** Energy was estimated with the Green Algorithms method (runtime × [cores × per-core power + memory × 0.3725 W/GB] × PUE) for an Apple M3 Pro laptop (PUE 1.0, package power 25 to 45 W), with run times taken from each stage's log where available, and converted to CO2e at 0.10 to 0.70 (central 0.367) kgCO2e/kWh. Inference time was measured per 1,000 predictions. Tokens consumed by the AI assistant were recorded where metered.

**Stage 16 — what-if: multimodal ocular imaging.** For each reconstructed patient, imaging features were drawn conditional on the observed outcome (P[feature | NAION] ∝ prior × odds ratio; P[feature | no NAION] = prior), independently across features: cup-to-disc ratio categories with the published control distribution and odds ratios (fundus photography with 25% and OCT with 5% adjacent-category misclassification), and hypothetical optic disc drusen (prevalence 2%, odds ratio 2 to 8), OCT angiography hypoperfusion (20%, 1.5 to 3) and serial on-treatment imaging change (5%, 2 to 10), with sleep apnoea (27%, 3.28) as a non-imaging comparator. XGBoost was trained with and without these features on five patient-level 10%/90% folds for each of three generation seeds, and the number of events needed to estimate AUROC within ±0.03 in a prospective cohort was derived from the Hanley-McNeil variance.

## Results

### Evidence base

Of 169 records, 73 publications contributed data: 29 observational studies, 17 case reports or series (18 NAION patients), 11 pharmacovigilance studies and 16 pooled-trial or meta-analytic reports (study flow figure, Section A). Reconstruction yielded 3,271,499 patients and 1,425 events (prevalence 0.044%) from 11 cohorts in the United States, the Nordic countries and the United Kingdom (Reconstructed cohorts table, Section A). No source provided patient-level imaging or video; case reports described fundus photography and optical coherence tomography only as figures.

### Associations

With study fixed effects, GLP-1RA exposure was associated with NAION (odds ratio 1.59 (95% CI, 1.42 to 1.77); q<0.001). Estimates were 1.73 (95% CI, 1.44 to 2.08) for semaglutide, 2.70 (95% CI, 1.64 to 4.44) when the indication was obesity and 1.62 (95% CI, 1.45 to 1.82) for type 2 diabetes. The association depended on the comparator: 1.81 (95% CI, 1.54 to 2.11) against SGLT2 inhibitors and 2.35 (95% CI, 1.74 to 3.16) against other antidiabetic drugs, but 1.03 (95% CI, 0.57 to 1.85) against DPP-4 inhibitors and 1.14 (95% CI, 0.94 to 1.39) against untreated comparators. In patients with idiopathic intracranial hypertension the direction reversed (0.43 (95% CI, 0.22 to 0.84)). The exposure effect was larger in studies with older participants (interaction odds ratio 1.99 (95% CI, 1.30 to 3.04) per 10 years; q=0.002) and did not vary with the percentage of women (q=0.943). Random-effects pooling of published estimates gave a ratio of 1.34 (95% CI, 1.16 to 1.55; I² = 87%) (published-estimate forest plot, Section C). Patients in case reports were younger than exposed cohort participants (median 47 vs 60 years; q=0.016); the excess of men was not significant (q=0.274).

### Discrimination, calibration and explainability

Trained on 10% of patients, XGBoost reached an AUROC of 0.657 on the remaining 90% with calibration slope 0.97; across ten disjoint 10% training folds the AUROC was 0.656 ± 0.011 and O/E 1.00 (hold-out ROC, precision-recall and calibration figure, Section B). Logistic regression (0.616) and random forest (0.646) did not differ significantly from XGBoost (all corrected q≥0.27). AUROC rose with training fraction to a plateau near 0.67 at 40% to 60%. In unseen studies, AUROC fell to 0.534. SHAP rankings were stable across folds; permutation tests confirmed signal for exposure (q=0.004) and therapy (q=0.015), while data source and region — markers of setting rather than patient risk — carried the largest attributions.

### Threshold performance

At the top-10% threshold, XGBoost had sensitivity 0.35, specificity 0.82, PPV 0.087% and NPV 99.97% (positive likelihood ratio 2.01); at the Youden threshold sensitivity was 0.59 and specificity 0.61. Sensitivity and PPV exceeded chance (P<0.001), whereas specificity did not exceed the value expected at the same flag rate and accuracy remained below the no-information rate of 99.96%. Net benefit exceeded treat-all and treat-none strategies only at threshold probabilities below 0.1%. No metric differed between models (all q≥0.99).

### Graph neural networks, ensembles and digital twins

The GraphSAGE GNN reached an AUROC of 0.649 ± 0.014 across the 10 folds, not different from XGBoost on identical splits (difference -0.0077; P=0.779), with O/E 1.01. In leave-one-study-out validation its AUROC was 0.531 (P=0.034 vs 0.5). Integrated-gradient attributions ranked therapy, data source and region highest, mirroring SHAP.

Nested tuning over 366 GNN configurations selected a SAGE network of depth 6 with k=20 neighbours, uniform edge weighting and same-study weight 5.0; on the untouched folds it differed from the default network by 0.0006 AUROC (P=0.948).

Among 11 member combinations × 3 combiners, the best ensemble (cross_family(pubmed+hybrid+synthetic), mean) reached AUROC 0.657, a difference of 0.0010 from the best single model (q=1.000).

In the XGBoost grid (5,310 fits), no hybrid cell improved AUROC over PubMed-only training (0 of 162 cells with a confidence interval above zero); the best hybrid changed AUROC by 0.0009 (95% CI, -0.0004 to 0.0022), synthetic-only training with twins fitted to real data lost 0.010, and twins built only from published priors reached AUROC values near 0.55. Adding twins beyond about 9 to 27 times the real training size reduced performance.

Across the full factorial grid (45,360 configurations), 4452 hybrid or synthetic configurations improved discrimination over PubMed-only training after Benjamini-Hochberg correction.

The model-summary table (Section I) summarizes the main model families.

### Transport, fine-tuning and convolutional networks

**Background-risk model transported to GLP-1RA users.** Trained on 1,983,956 non-GLP patients (740 events), background models discriminated within non-users (AUROC 0.626 to 0.666) but much less among 1,287,543 GLP-1RA users (685 events): AUROC 0.533 to 0.597 (best, ensemble). Observed events among GLP-1RA users exceeded background predictions (O/E 1.41 to 1.63), an independent estimate of excess risk consistent with the within-study odds ratio. Traditional and setting features therefore explain little of which GLP-1RA users develop NAION.

**Fine-tuning.** The best update (pooled refit; pretraining synthetic, fine-tuning hybrid, 10% of GLP-1RA patients) reached AUROC 0.618 in held-out GLP-1RA users with O/E 1.08. 49 configurations improved on the pretrained background model after Benjamini-Hochberg correction. Recalibration corrected under-prediction without changing discrimination, as expected; synthetic fine-tuning data did not outperform real data.

**Convolutional network.** The 1D-CNN reached AUROC 0.653 ± 0.011 across ten folds, not different from XGBoost (difference -0.0037; P=0.782), and 0.514 in leave-one-study-out validation. Convolutions add nothing when features lack spatial or sequential structure. Two- and three-dimensional CNNs and multimodal fusion were not trainable: the corpus contains one usable modality (structured tabular data derived from text), imaging only as published figures for about six patients, and no video.

### Incidence, rare-event and anomaly models

**Incidence.** NAION occurred at 32.0 per 100,000 person-years among GLP-1RA users and 17.9 among comparators (incidence rate ratio 1.79; 95% CI, 1.61 to 1.99; Mantel-Haenszel ratio stratified by study 1.73), a rate difference of 14.1 per 100,000 person-years. Both rates lie within or above the published background range for adults aged 50 years or older [@p8032479; @p9186104]. The ratio was 1.84 in type 2 diabetes and 0.43 in idiopathic intracranial hypertension, and Poisson models with study fixed effects showed heterogeneity of the exposure effect by study age band (P=<0.001), indication (P=<0.001), drug (P=0.002), comparator (P=0.026) and follow-up (P=<0.001) but not by the percentage of women (P=0.756). Because age is a study-level attribute and person-years were reported by only two studies, these strata describe settings rather than individual patients.

**Rare-event and anomaly-detection models.** A Poisson rate model with a person-time offset (AUROC 0.652), King-Zeng corrected logistic regression (0.655) and incidence-anchored XGBoost (0.657) matched standard XGBoost (0.656); class weighting required prior correction to restore calibration, and a focal-loss network was miscalibrated. Anomaly detectors trained only on patients without NAION performed near chance (AUROC 0.413 to 0.575), because patients with and without NAION from the same study arm share identical features in reconstructed aggregate data; anomaly methods need individual-level covariates.

Table: NAION incidence per 100,000 person-years in GLP-1RA users and comparators, by stratum (IRR, incidence rate ratio; MH, Mantel-Haenszel by study; q, Benjamini-Hochberg).

| Stratifier   | Level              |   Studies |   GLP events |   GLP rate |   Non-GLP events |   Non-GLP rate |   IRR |   95% CI low |   95% CI high |   MH IRR | q      |
|:-------------|:-------------------|----------:|-------------:|-----------:|-----------------:|---------------:|------:|-------------:|--------------:|---------:|:-------|
| overall      | all                |        11 |          685 |       32   |              740 |           17.9 |  1.79 |         1.61 |          1.99 |     1.73 | <0.001 |
| drug         | glp1ra_class       |         5 |          454 |       51.8 |              368 |           29.1 |  1.78 |         1.55 |          2.05 |     1.5  | <0.001 |
| drug         | sema_or_tirz       |         1 |           35 |       22   |               19 |           11.9 |  1.84 |         1.03 |          3.41 |     1.84 | 0.052  |
| drug         | semaglutide        |         5 |          196 |       17.8 |              353 |           13   |  1.36 |         1.14 |          1.63 |     2.29 | 0.001  |
| indication   | iih                |         1 |           12 |       37.6 |               28 |           87.8 |  0.43 |         0.2  |          0.87 |     0.43 | 0.023  |
| indication   | t2d                |        10 |          673 |       31.9 |              712 |           17.4 |  1.84 |         1.65 |          2.05 |     1.81 | <0.001 |
| comparator   | dpp4i              |         1 |           14 |       13.1 |               53 |           12.7 |  1.03 |         0.53 |          1.88 |     1.03 | 0.880  |
| comparator   | no_glp1ra          |         3 |          175 |       25.1 |              258 |           12.8 |  1.97 |         1.62 |          2.4  |     1.44 | <0.001 |
| comparator   | other_antidiabetic |         3 |          126 |       14.7 |               71 |            8.2 |  1.78 |         1.32 |          2.41 |     2.25 | <0.001 |
| comparator   | sglt2i             |         4 |          370 |       77.6 |              358 |           43   |  1.8  |         1.56 |          2.09 |     1.84 | <0.001 |
| region       | nordic             |         3 |          137 |       28.2 |              222 |           11.2 |  2.51 |         2.01 |          3.12 |     2.07 | <0.001 |
| region       | uk                 |         1 |           14 |       13.1 |               53 |           12.7 |  1.03 |         0.53 |          1.88 |     1.03 | 0.880  |
| region       | us                 |         7 |          534 |       34.5 |              465 |           26.7 |  1.29 |         1.14 |          1.47 |     1.67 | <0.001 |
| source_type  | claims             |         1 |          270 |      103.5 |              144 |           55.2 |  1.88 |         1.53 |          2.31 |     1.88 | <0.001 |
| source_type  | ehr_network        |         4 |          188 |       14.9 |              155 |           12.3 |  1.21 |         0.98 |          1.51 |     1.21 | 0.094  |
| source_type  | ehr_primary_care   |         1 |           14 |       13.1 |               53 |           12.7 |  1.03 |         0.53 |          1.88 |     1.03 | 0.880  |
| source_type  | national_registry  |         3 |          137 |       28.2 |              222 |           11.2 |  2.51 |         2.01 |          3.12 |     2.07 | <0.001 |
| source_type  | single_center      |         1 |           46 |     2762.8 |               23 |          676.1 |  4.09 |         2.43 |          7.06 |     4.09 | <0.001 |
| source_type  | va_ehr             |         1 |           30 |      122.9 |              143 |           66.5 |  1.85 |         1.2  |          2.75 |     1.85 | 0.005  |
| age_band     | 55-59              |         4 |          206 |       16.4 |              270 |           18.7 |  0.88 |         0.73 |          1.06 |     1.47 | 0.194  |
| age_band     | <55                |         1 |           12 |       37.6 |               28 |           87.8 |  0.43 |         0.2  |          0.87 |     0.43 | 0.023  |
| age_band     | >=60               |         1 |          270 |      103.5 |              144 |           55.2 |  1.88 |         1.53 |          2.31 |     1.88 | <0.001 |
| age_band     | unknown            |         5 |          197 |       33.1 |              298 |           12.4 |  2.66 |         2.21 |          3.2  |     2.13 | <0.001 |
| female_band  | 40-54%             |         5 |          513 |       28.8 |              422 |           13.6 |  2.12 |         1.86 |          2.42 |     1.76 | <0.001 |
| female_band  | <40%               |         1 |           30 |      122.9 |              143 |           66.5 |  1.85 |         1.2  |          2.75 |     1.85 | 0.005  |
| female_band  | >=55%              |         2 |           58 |      172.8 |               51 |          144.5 |  1.2  |         0.81 |          1.78 |     1.71 | 0.432  |
| female_band  | unknown            |         3 |           84 |       28.1 |              124 |           16   |  1.75 |         1.31 |          2.33 |     1.52 | <0.001 |
| fu_band      | 1-2.5 y            |         4 |          173 |       29.5 |              269 |           34.6 |  0.85 |         0.7  |          1.04 |     1.27 | 0.123  |
| fu_band      | <=1 y              |         3 |          346 |       72.8 |              261 |           30.2 |  2.41 |         2.04 |          2.84 |     1.74 | <0.001 |
| fu_band      | >2.5 y             |         4 |          166 |       15.4 |              210 |            8.4 |  1.83 |         1.48 |          2.25 |     2.43 | <0.001 |

### Detection within GLP-1RA users and non-users

**Performance within GLP-1RA users and within non-users.** Trained on mixed data, every model discriminated better among non-users (AUROC 0.577 to 0.674) than among GLP-1RA users (0.511 to 0.600; best, xgb). The within-group difference in AUROC was significant after correction for 0 of 9 models. Together with the under-prediction of a background model in GLP-1RA users, this indicates that the features available in published data capture background NAION risk but little of what determines which GLP-1RA users develop NAION.

Table: Model performance within GLP-1RA users and within non-users (mean over ten folds; threshold = top 10% of training risk).

| Model            | Group         |   AUROC |   PR-AUC | O/E    |   Sens |   Spec |    Acc |    PPV |    NPV |
|:-----------------|:--------------|--------:|---------:|:-------|-------:|-------:|-------:|-------:|-------:|
| autoencoder      | GLP-1RA users |  0.5136 |   0.0037 | —      | 0.2349 | 0.8168 | 0.8165 | 0.0008 | 0.9995 |
| autoencoder      | non-users     |  0.6114 |   0.0008 | —      | 0.1617 | 0.9475 | 0.9472 | 0.0012 | 0.9997 |
| cnn1d            | GLP-1RA users |  0.5946 |   0.006  | 1.0018 | 0.3845 | 0.7143 | 0.7141 | 0.0008 | 0.9995 |
| cnn1d            | non-users     |  0.6731 |   0.0017 | 0.9998 | 0.3112 | 0.8918 | 0.8916 | 0.0014 | 0.9997 |
| ensemble         | GLP-1RA users |  0.5977 |   0.0061 | 1.0674 | 0.3662 | 0.7374 | 0.7372 | 0.0012 | 0.9995 |
| ensemble         | non-users     |  0.6719 |   0.0015 | 0.9581 | 0.315  | 0.8917 | 0.8914 | 0.0014 | 0.9997 |
| gnn              | GLP-1RA users |  0.5942 |   0.0061 | 1.0097 | 0.3943 | 0.706  | 0.7059 | 0.0008 | 0.9995 |
| gnn              | non-users     |  0.6673 |   0.0015 | 1.0100 | 0.3099 | 0.8911 | 0.8909 | 0.0014 | 0.9997 |
| isolation_forest | GLP-1RA users |  0.511  |   0.0006 | —      | 0.2234 | 0.7991 | 0.7988 | 0.0006 | 0.9995 |
| isolation_forest | non-users     |  0.5767 |   0.0007 | —      | 0.2405 | 0.9292 | 0.9289 | 0.0013 | 0.9997 |
| logreg           | GLP-1RA users |  0.5612 |   0.0008 | 1.1752 | 0.331  | 0.7352 | 0.735  | 0.0008 | 0.9995 |
| logreg           | non-users     |  0.635  |   0.0007 | 0.8887 | 0.1925 | 0.9213 | 0.9211 | 0.0012 | 0.9997 |
| poisson_offset   | GLP-1RA users |  0.5972 |   0.006  | 1.0153 | 0.3636 | 0.7426 | 0.7424 | 0.001  | 0.9996 |
| poisson_offset   | non-users     |  0.6706 |   0.0016 | 1.0112 | 0.3254 | 0.8817 | 0.8815 | 0.0013 | 0.9997 |
| rf               | GLP-1RA users |  0.5902 |   0.0009 | 1.0958 | 0.3449 | 0.7536 | 0.7534 | 0.0008 | 0.9995 |
| rf               | non-users     |  0.6632 |   0.0008 | 0.9392 | 0.263  | 0.9181 | 0.9178 | 0.0015 | 0.9997 |
| xgb              | GLP-1RA users |  0.5995 |   0.0062 | 1.0111 | 0.3937 | 0.7106 | 0.7104 | 0.0008 | 0.9995 |
| xgb              | non-users     |  0.674  |   0.0017 | 1.0089 | 0.315  | 0.8917 | 0.8914 | 0.0014 | 0.9997 |

### Vascular score features and a proposed NAION-adapted score

**Validated vascular score as a feature.** Only three ABCD2/ABCD3-I items could be derived: age 60 years or older (7 of 11 studies, from study mean age), blood pressure (5 of 11, using hypertension prevalence as a proxy) and diabetes (11 of 11); clinical features, symptom duration, dual events and imaging are specific to transient ischemic attack. Adding the partial items to XGBoost changed AUROC by -0.0019 (P=0.917). The partial score alone, in studies with all items, reached AUROC 0.545.

**Proposed NAION-adapted points score.** Fitting point weights to the reconstructed data produced implausible protective signs for hypertension, diabetes and GLP-1RA exposure, because these items vary only between studies; we therefore report this fit only as a demonstration of ecological bias. A literature-informed version assigned points from published patient-level ratios (male sex, hypertension, hyperlipidemia, diabetes, coronary heart disease and sleep apnea [@p34671609]) and from our within-study GLP-1RA estimates (indication-specific), with age carried from the ABCD2 convention: male sex 2 points, hypertension 1 point, hyperlipidemia 1 point, diabetes 2 points, coronary heart disease 2 points, sleep apnea 5 points, age ge60 1 point, cdr le 0.20 post exposure fellow eye 16 points, cdr 0.21 0.40 post exposure fellow eye 8 points, glp1ra t2d 2 points, glp1ra obesity 4 points. On setting-level data it ranked cohorts poorly (AUROC 0.417; difference from XGBoost -0.240), as expected when settings with more vascular risk factors recorded fewer events; the score requires prospective, patient-level calibration, including disc morphology.

**Limitations of the partial score.** only A, B and D of ABCD2/ABCD3-I are derivable; C, duration, dual-TIA and imaging items are TIA-specific; A and B are study-level proxies (age from study mean with assumed SD 12; hypertension diagnosis for BP >= 140/90); A available for 7/11 and B for 5/11 studies; remaining values imputed and flagged; both arms of a study share the same A and B, so the score cannot separate exposed from unexposed patients within a study; any apparent importance reflects between-setting differences, not individual patient risk.

**Cup-to-disc ratio.** No source reported a pre-treatment (pre-GLP-1RA) cup-to-disc ratio. In the case reports, disc status was recorded at or after NAION diagnosis, usually in the unaffected fellow eye. The largest source, a 16-center case-control study of 1,812 unaffected fellow eyes, found a fellow-eye ratio of 0.20 or less in 54% of patients with NAION versus 9% of controls (odds ratio 46.3; 95% CI, 22.9 to 93.4; ratio 0.21 to 0.40: 6.8; 3.5 to 13.2) and no difference between semaglutide users, other GLP-1RA users and unexposed patients [@p42556433]. We therefore entered the cup-to-disc ratio as the highest-weighted item of the proposed score but labelled it a post-exposure fellow-eye measurement: disc anatomy is structurally stable and the fellow eye is the accepted proxy, yet this is weaker evidence than a baseline measured before GLP-1RA initiation, and it could not be validated in the cohort data. The same limitation applies to NAION without GLP-1RA exposure: the classic and recent disc-at-risk studies measured the cup-to-disc ratio after the event, in fellow eyes or in affected eyes after edema resolved [@p6514298; @p3684223; @p38211825; @p42556433], and none had a measurement taken before NAION. Because both groups share this post-event design, the comparison between exposed and unexposed NAION is balanced, but no source supports a truly prospective (pre-event) estimate of disc-related risk. Baseline optic disc photography or OCT before GLP-1RA initiation in a prospective cohort would close this gap.

**With versus without ABCD features across all experiments.** Over 455 paired comparisons using identical splits and seeds, adding the partial ABCD items changed AUROC by 0.0016 on average; 0 comparisons differed after Benjamini-Hochberg correction.

Table: AUROC with and without partial ABCD2 features in every experiment (paired, identical splits; q, Benjamini-Hochberg).

| Experiment                              | Model                                          |   Without ABCD |   With ABCD |   Difference |   q |
|:----------------------------------------|:-----------------------------------------------|---------------:|------------:|-------------:|----:|
| Stage 1: 10 x 10% folds                 | logreg                                         |         0.6159 |      0.6202 |       0.0043 |   1 |
| Stage 1: leave-one-study-out            | logreg                                         |         0.5097 |      0.5353 |       0.0256 |   1 |
| Stage 1: 10 x 10% folds                 | rf                                             |         0.6458 |      0.6507 |       0.005  |   1 |
| Stage 1: leave-one-study-out            | rf                                             |         0.5315 |      0.5315 |       0      |   1 |
| Stage 1: 10 x 10% folds                 | xgb                                            |         0.6565 |      0.6542 |      -0.0023 |   1 |
| Stage 1: leave-one-study-out            | xgb                                            |         0.5353 |      0.5441 |       0.0088 |   1 |
| Stage 1c: training fractions 10-90%     | logreg                                         |         0.6289 |      0.6408 |       0.0119 |   1 |
| Stage 1c: training fractions 10-90%     | rf                                             |         0.6593 |      0.663  |       0.0037 |   1 |
| Stage 1c: training fractions 10-90%     | xgb                                            |         0.6669 |      0.667  |       0.0001 |   1 |
| Stage 6: GNN folds                      | gnn                                            |         0.6487 |      0.6508 |       0.002  |   1 |
| Stage 6: GNN LOSO                       | gnn                                            |         0.5309 |      0.5397 |       0.0088 |   1 |
| Stage 9: CNN folds                      | cnn1d                                          |         0.6527 |      0.6523 |      -0.0004 |   1 |
| Stage 9: CNN LOSO                       | cnn1d                                          |         0.5137 |      0.5262 |       0.0125 |   1 |
| Stage 10: rare-event / anomaly folds    | poisson_offset                                 |         0.6523 |      0.6525 |       0.0002 |   1 |
| Stage 10: rare-event / anomaly folds    | xgb_balanced_raw                               |         0.6481 |      0.6481 |       0      |   1 |
| Stage 10: rare-event / anomaly folds    | xgb_balanced_priorcorrected                    |         0.6481 |      0.6481 |       0      |   1 |
| Stage 10: rare-event / anomaly folds    | logreg_kingzeng                                |         0.6551 |      0.6579 |       0.0027 |   1 |
| Stage 10: rare-event / anomaly folds    | xgb_incidence_anchored                         |         0.6572 |      0.6542 |      -0.0029 |   1 |
| Stage 10: rare-event / anomaly folds    | focal_mlp                                      |         0.5749 |      0.5722 |      -0.0027 |   1 |
| Stage 10: rare-event / anomaly folds    | xgb_standard                                   |         0.6565 |      0.6542 |      -0.0023 |   1 |
| Stage 10: rare-event / anomaly folds    | isolation_forest                               |         0.5411 |      0.5364 |      -0.0048 |   1 |
| Stage 10: rare-event / anomaly folds    | one_class_svm                                  |         0.4133 |      0.4162 |       0.0028 |   1 |
| Stage 10: rare-event / anomaly folds    | lof_novelty                                    |         0.4211 |      0.4138 |      -0.0073 |   1 |
| Stage 10: rare-event / anomaly folds    | autoencoder                                    |         0.5747 |      0.5847 |       0.01   |   1 |
| Stage 11: within GLP-1RA users          | autoencoder                                    |         0.5136 |      0.5208 |       0.0072 |   1 |
| Stage 11: within all                    | autoencoder                                    |         0.5747 |      0.5847 |       0.01   |   1 |
| Stage 11: within non-users              | autoencoder                                    |         0.6114 |      0.6099 |      -0.0014 |   1 |
| Stage 11: within GLP-1RA users          | cnn1d                                          |         0.5946 |      0.5935 |      -0.001  |   1 |
| Stage 11: within all                    | cnn1d                                          |         0.6527 |      0.6523 |      -0.0004 |   1 |
| Stage 11: within non-users              | cnn1d                                          |         0.6731 |      0.6734 |       0.0003 |   1 |
| Stage 11: within GLP-1RA users          | ensemble                                       |         0.5977 |      0.5976 |      -0.0001 |   1 |
| Stage 11: within all                    | ensemble                                       |         0.6559 |      0.6554 |      -0.0005 |   1 |
| Stage 11: within non-users              | ensemble                                       |         0.6719 |      0.6728 |       0.0009 |   1 |
| Stage 11: within GLP-1RA users          | gnn                                            |         0.5942 |      0.5931 |      -0.0011 |   1 |
| Stage 11: within all                    | gnn                                            |         0.6491 |      0.6479 |      -0.0011 |   1 |
| Stage 11: within non-users              | gnn                                            |         0.6673 |      0.6668 |      -0.0005 |   1 |
| Stage 11: within GLP-1RA users          | isolation_forest                               |         0.511  |      0.511  |       0      |   1 |
| Stage 11: within all                    | isolation_forest                               |         0.5411 |      0.5364 |      -0.0048 |   1 |
| Stage 11: within non-users              | isolation_forest                               |         0.5767 |      0.5646 |      -0.0121 |   1 |
| Stage 11: within GLP-1RA users          | logreg                                         |         0.5612 |      0.576  |       0.0148 |   1 |
| Stage 11: within all                    | logreg                                         |         0.6159 |      0.6202 |       0.0043 |   1 |
| Stage 11: within non-users              | logreg                                         |         0.635  |      0.6377 |       0.0027 |   1 |
| Stage 11: within GLP-1RA users          | poisson_offset                                 |         0.5972 |      0.5972 |       0      |   1 |
| Stage 11: within all                    | poisson_offset                                 |         0.6523 |      0.6525 |       0.0002 |   1 |
| Stage 11: within non-users              | poisson_offset                                 |         0.6706 |      0.6704 |      -0.0002 |   1 |
| Stage 11: within GLP-1RA users          | rf                                             |         0.5902 |      0.594  |       0.0038 |   1 |
| Stage 11: within all                    | rf                                             |         0.6458 |      0.6507 |       0.005  |   1 |
| Stage 11: within non-users              | rf                                             |         0.6632 |      0.6682 |       0.0049 |   1 |
| Stage 11: within GLP-1RA users          | xgb                                            |         0.5995 |      0.5946 |      -0.005  |   1 |
| Stage 11: within all                    | xgb                                            |         0.6565 |      0.6542 |      -0.0023 |   1 |
| Stage 11: within non-users              | xgb                                            |         0.674  |      0.6724 |      -0.0017 |   1 |
| Stage 7: non-GLP model, external_GLP    | ensemble                                       |         0.5966 |      0.5835 |      -0.0131 |   1 |
| Stage 7: non-GLP model, internal_nonGLP | ensemble                                       |         0.6665 |      0.6617 |      -0.0048 |   1 |
| Stage 7: non-GLP model, external_GLP    | gnn                                            |         0.5893 |      0.5795 |      -0.0098 |   1 |
| Stage 7: non-GLP model, internal_nonGLP | gnn                                            |         0.6641 |      0.6592 |      -0.0049 |   1 |
| Stage 7: non-GLP model, external_GLP    | logreg                                         |         0.5328 |      0.5632 |       0.0305 |   1 |
| Stage 7: non-GLP model, internal_nonGLP | logreg                                         |         0.6257 |      0.6357 |       0.0101 |   1 |
| Stage 7: non-GLP model, external_GLP    | rf                                             |         0.5784 |      0.5886 |       0.0101 |   1 |
| Stage 7: non-GLP model, internal_nonGLP | rf                                             |         0.6518 |      0.6599 |       0.0082 |   1 |
| Stage 7: non-GLP model, external_GLP    | xgb                                            |         0.5929 |      0.5856 |      -0.0073 |   1 |
| Stage 7: non-GLP model, internal_nonGLP | xgb                                            |         0.6639 |      0.6633 |      -0.0006 |   1 |
| Stage 4: ensembles folds                | cross_family(pubmed+hybrid+synthetic) (mean)   |         0.6574 |      0.6563 |      -0.0011 |   1 |
| Stage 4: ensembles folds                | gnn (single)                                   |         0.6525 |      0.6487 |      -0.0037 |   1 |
| Stage 4: ensembles folds                | logreg (single)                                |         0.6159 |      0.6202 |       0.0043 |   1 |
| Stage 4: ensembles folds                | logreg+gnn (mean)                              |         0.6527 |      0.6506 |      -0.0021 |   1 |
| Stage 4: ensembles folds                | logreg+gnn (stacked)                           |         0.6528 |      0.6493 |      -0.0035 |   1 |
| Stage 4: ensembles folds                | logreg+gnn (weighted)                          |         0.6519 |      0.6496 |      -0.0022 |   1 |
| Stage 4: ensembles folds                | logreg+rf (mean)                               |         0.6455 |      0.6488 |       0.0034 |   1 |
| Stage 4: ensembles folds                | logreg+rf (stacked)                            |         0.6457 |      0.6506 |       0.0049 |   1 |
| Stage 4: ensembles folds                | logreg+rf (weighted)                           |         0.6449 |      0.65   |       0.0051 |   1 |
| Stage 4: ensembles folds                | logreg+rf+gnn (mean)                           |         0.6539 |      0.6541 |       0.0002 |   1 |
| Stage 4: ensembles folds                | logreg+rf+gnn (stacked)                        |         0.6519 |      0.6499 |      -0.002  |   1 |
| Stage 4: ensembles folds                | logreg+rf+gnn (weighted)                       |         0.6536 |      0.6531 |      -0.0005 |   1 |
| Stage 4: ensembles folds                | logreg+rf+xgb (mean)                           |         0.6556 |      0.6558 |       0.0002 |   1 |
| Stage 4: ensembles folds                | logreg+rf+xgb (stacked)                        |         0.6561 |      0.6543 |      -0.0018 |   1 |
| Stage 4: ensembles folds                | logreg+rf+xgb (weighted)                       |         0.656  |      0.6558 |      -0.0002 |   1 |
| Stage 4: ensembles folds                | logreg+rf+xgb+gnn (mean)                       |         0.656  |      0.6551 |      -0.0009 |   1 |
| Stage 4: ensembles folds                | logreg+rf+xgb+gnn (stacked)                    |         0.6556 |      0.6525 |      -0.0031 |   1 |
| Stage 4: ensembles folds                | logreg+rf+xgb+gnn (weighted)                   |         0.6556 |      0.6551 |      -0.0005 |   1 |
| Stage 4: ensembles folds                | logreg+xgb (mean)                              |         0.6571 |      0.6549 |      -0.0021 |   1 |
| Stage 4: ensembles folds                | logreg+xgb (stacked)                           |         0.6565 |      0.6539 |      -0.0025 |   1 |
| Stage 4: ensembles folds                | logreg+xgb (weighted)                          |         0.6569 |      0.6546 |      -0.0023 |   1 |
| Stage 4: ensembles folds                | logreg+xgb+gnn (mean)                          |         0.6564 |      0.6537 |      -0.0027 |   1 |
| Stage 4: ensembles folds                | logreg+xgb+gnn (stacked)                       |         0.6557 |      0.6528 |      -0.0029 |   1 |
| Stage 4: ensembles folds                | logreg+xgb+gnn (weighted)                      |         0.6554 |      0.6538 |      -0.0016 |   1 |
| Stage 4: ensembles folds                | rf (single)                                    |         0.6458 |      0.6507 |       0.005  |   1 |
| Stage 4: ensembles folds                | rf+gnn (mean)                                  |         0.6547 |      0.6533 |      -0.0014 |   1 |
| Stage 4: ensembles folds                | rf+gnn (stacked)                               |         0.6529 |      0.6485 |      -0.0045 |   1 |
| Stage 4: ensembles folds                | rf+gnn (weighted)                              |         0.654  |      0.6534 |      -0.0006 |   1 |
| Stage 4: ensembles folds                | rf+xgb (mean)                                  |         0.6563 |      0.656  |      -0.0003 |   1 |
| Stage 4: ensembles folds                | rf+xgb (stacked)                               |         0.6568 |      0.6551 |      -0.0017 |   1 |
| Stage 4: ensembles folds                | rf+xgb (weighted)                              |         0.6561 |      0.656  |      -0.0001 |   1 |
| Stage 4: ensembles folds                | rf+xgb+gnn (mean)                              |         0.6566 |      0.6544 |      -0.0022 |   1 |
| Stage 4: ensembles folds                | rf+xgb+gnn (stacked)                           |         0.6545 |      0.652  |      -0.0025 |   1 |
| Stage 4: ensembles folds                | rf+xgb+gnn (weighted)                          |         0.6559 |      0.6541 |      -0.0018 |   1 |
| Stage 4: ensembles folds                | xgb (single)                                   |         0.6565 |      0.6542 |      -0.0023 |   1 |
| Stage 4: ensembles folds                | xgb+gnn (mean)                                 |         0.6557 |      0.6539 |      -0.0018 |   1 |
| Stage 4: ensembles folds                | xgb+gnn (stacked)                              |         0.655  |      0.6527 |      -0.0023 |   1 |
| Stage 4: ensembles folds                | xgb+gnn (weighted)                             |         0.6557 |      0.6539 |      -0.0017 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.05 / glp_only              |         0.5841 |      0.5834 |      -0.0007 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.05 / gnn_joint             |         0.6039 |      0.5954 |      -0.0085 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.05 / pooled_refit          |         0.5947 |      0.5972 |       0.0025 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.05 / pretrain_only         |         0.5861 |      0.5906 |       0.0046 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.05 / recal_intercept       |         0.5861 |      0.5906 |       0.0046 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.05 / recal_logistic        |         0.5861 |      0.5906 |       0.0046 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.05 / xgb_continue          |         0.5961 |      0.5965 |       0.0004 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.1 / glp_only               |         0.6059 |      0.6058 |      -0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.1 / gnn_joint              |         0.609  |      0.6091 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.1 / pooled_refit           |         0.6045 |      0.6057 |       0.0012 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.1 / pretrain_only          |         0.5869 |      0.5901 |       0.0032 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.1 / recal_intercept        |         0.5869 |      0.5901 |       0.0032 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.1 / recal_logistic         |         0.5869 |      0.5901 |       0.0032 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.1 / xgb_continue           |         0.6052 |      0.6094 |       0.0042 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.25 / glp_only              |         0.6083 |      0.6087 |       0.0004 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.25 / gnn_joint             |         0.6107 |      0.6103 |      -0.0004 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.25 / pooled_refit          |         0.6086 |      0.607  |      -0.0016 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.25 / pretrain_only         |         0.5886 |      0.5801 |      -0.0085 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.25 / recal_intercept       |         0.5886 |      0.5801 |      -0.0085 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.25 / recal_logistic        |         0.5886 |      0.5801 |      -0.0085 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.25 / xgb_continue          |         0.6084 |      0.6062 |      -0.0022 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.5 / glp_only               |         0.6118 |      0.612  |       0.0002 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.5 / gnn_joint              |         0.6131 |      0.6116 |      -0.0015 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.5 / pooled_refit           |         0.6146 |      0.6129 |      -0.0017 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.5 / pretrain_only          |         0.5761 |      0.5826 |       0.0065 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.5 / recal_intercept        |         0.5761 |      0.5826 |       0.0065 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.5 / recal_logistic         |         0.5761 |      0.5826 |       0.0065 |   1 |
| Stage 8: fine-tuning                    | hybrid / hybrid / 0.5 / xgb_continue           |         0.6135 |      0.6148 |       0.0013 |   1 |
| Stage 8: fine-tuning                    | hybrid / none / 0.0 / pretrain_only            |         0.5701 |      0.5846 |       0.0145 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.05 / glp_only                |         0.5655 |      0.5556 |      -0.0099 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.05 / gnn_joint               |         0.5676 |      0.5737 |       0.006  |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.05 / pooled_refit            |         0.5856 |      0.586  |       0.0005 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.05 / pretrain_only           |         0.5804 |      0.5872 |       0.0068 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.05 / recal_intercept         |         0.5804 |      0.5872 |       0.0068 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.05 / recal_logistic          |         0.5804 |      0.5872 |       0.0068 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.05 / xgb_continue            |         0.5915 |      0.5985 |       0.007  |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.1 / glp_only                 |         0.5953 |      0.5948 |      -0.0005 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.1 / gnn_joint                |         0.6012 |      0.6002 |      -0.001  |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.1 / pooled_refit             |         0.6046 |      0.6065 |       0.0019 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.1 / pretrain_only            |         0.5749 |      0.5703 |      -0.0046 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.1 / recal_intercept          |         0.5749 |      0.5703 |      -0.0046 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.1 / recal_logistic           |         0.5749 |      0.5703 |      -0.0046 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.1 / xgb_continue             |         0.6053 |      0.5994 |      -0.0059 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.25 / glp_only                |         0.608  |      0.6081 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.25 / gnn_joint               |         0.6083 |      0.6082 |      -0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.25 / pooled_refit            |         0.6138 |      0.6084 |      -0.0054 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.25 / pretrain_only           |         0.5774 |      0.5898 |       0.0125 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.25 / recal_intercept         |         0.5774 |      0.5898 |       0.0125 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.25 / recal_logistic          |         0.5774 |      0.5898 |       0.0125 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.25 / xgb_continue            |         0.6094 |      0.61   |       0.0005 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.5 / glp_only                 |         0.6113 |      0.6113 |       0      |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.5 / gnn_joint                |         0.613  |      0.6118 |      -0.0012 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.5 / pooled_refit             |         0.6126 |      0.6125 |      -0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.5 / pretrain_only            |         0.579  |      0.5791 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.5 / recal_intercept          |         0.579  |      0.5791 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.5 / recal_logistic           |         0.579  |      0.5791 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | hybrid / real / 0.5 / xgb_continue             |         0.6089 |      0.6119 |       0.003  |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.05 / glp_only           |         0.5224 |      0.5235 |       0.0011 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.05 / gnn_joint          |         0.5671 |      0.5591 |      -0.0081 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.05 / pooled_refit       |         0.5646 |      0.5775 |       0.0128 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.05 / pretrain_only      |         0.5983 |      0.5898 |      -0.0084 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.05 / recal_intercept    |         0.5983 |      0.5898 |      -0.0084 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.05 / recal_logistic     |         0.5983 |      0.5898 |      -0.0084 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.05 / xgb_continue       |         0.5983 |      0.5829 |      -0.0154 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.1 / glp_only            |         0.5338 |      0.5569 |       0.0232 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.1 / gnn_joint           |         0.6018 |      0.5948 |      -0.007  |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.1 / pooled_refit        |         0.6099 |      0.6112 |       0.0013 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.1 / pretrain_only       |         0.5926 |      0.5862 |      -0.0063 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.1 / recal_intercept     |         0.5926 |      0.5862 |      -0.0063 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.1 / recal_logistic      |         0.5926 |      0.5862 |      -0.0063 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.1 / xgb_continue        |         0.6025 |      0.601  |      -0.0015 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.25 / glp_only           |         0.5495 |      0.5548 |       0.0053 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.25 / gnn_joint          |         0.5963 |      0.6024 |       0.0062 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.25 / pooled_refit       |         0.6059 |      0.6055 |      -0.0004 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.25 / pretrain_only      |         0.5953 |      0.5976 |       0.0023 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.25 / recal_intercept    |         0.5953 |      0.5976 |       0.0023 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.25 / recal_logistic     |         0.5953 |      0.5976 |       0.0023 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.25 / xgb_continue       |         0.6037 |      0.6063 |       0.0025 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.5 / glp_only            |         0.5721 |      0.5792 |       0.007  |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.5 / gnn_joint           |         0.5998 |      0.6081 |       0.0082 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.5 / pooled_refit        |         0.6098 |      0.61   |       0.0002 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.5 / pretrain_only       |         0.5861 |      0.5759 |      -0.0102 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.5 / recal_intercept     |         0.5861 |      0.5759 |      -0.0102 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.5 / recal_logistic      |         0.5861 |      0.5759 |      -0.0102 |   1 |
| Stage 8: fine-tuning                    | hybrid / synthetic / 0.5 / xgb_continue        |         0.5996 |      0.6089 |       0.0093 |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.05 / glp_only                |         0.5985 |      0.596  |      -0.0025 |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.05 / pooled_refit            |         0.5985 |      0.596  |      -0.0025 |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.1 / glp_only                 |         0.5963 |      0.5945 |      -0.0018 |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.1 / pooled_refit             |         0.5963 |      0.5945 |      -0.0018 |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.25 / glp_only                |         0.6121 |      0.6143 |       0.0022 |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.25 / pooled_refit            |         0.6121 |      0.6143 |       0.0022 |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.5 / glp_only                 |         0.6149 |      0.6139 |      -0.001  |   1 |
| Stage 8: fine-tuning                    | none / hybrid / 0.5 / pooled_refit             |         0.6149 |      0.6139 |      -0.001  |   1 |
| Stage 8: fine-tuning                    | none / real / 0.05 / glp_only                  |         0.5736 |      0.5656 |      -0.008  |   1 |
| Stage 8: fine-tuning                    | none / real / 0.05 / pooled_refit              |         0.5736 |      0.5656 |      -0.008  |   1 |
| Stage 8: fine-tuning                    | none / real / 0.1 / glp_only                   |         0.5958 |      0.5891 |      -0.0067 |   1 |
| Stage 8: fine-tuning                    | none / real / 0.1 / pooled_refit               |         0.5958 |      0.5891 |      -0.0067 |   1 |
| Stage 8: fine-tuning                    | none / real / 0.25 / glp_only                  |         0.6114 |      0.6114 |       0      |   1 |
| Stage 8: fine-tuning                    | none / real / 0.25 / pooled_refit              |         0.6114 |      0.6114 |       0      |   1 |
| Stage 8: fine-tuning                    | none / real / 0.5 / glp_only                   |         0.6113 |      0.6113 |       0      |   1 |
| Stage 8: fine-tuning                    | none / real / 0.5 / pooled_refit               |         0.6113 |      0.6113 |       0      |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.05 / glp_only             |         0.5489 |      0.5361 |      -0.0128 |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.05 / pooled_refit         |         0.5489 |      0.5361 |      -0.0128 |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.1 / glp_only              |         0.5172 |      0.5109 |      -0.0063 |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.1 / pooled_refit          |         0.5172 |      0.5109 |      -0.0063 |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.25 / glp_only             |         0.5783 |      0.5739 |      -0.0044 |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.25 / pooled_refit         |         0.5783 |      0.5739 |      -0.0044 |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.5 / glp_only              |         0.5683 |      0.5774 |       0.0091 |   1 |
| Stage 8: fine-tuning                    | none / synthetic / 0.5 / pooled_refit          |         0.5683 |      0.5774 |       0.0091 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.05 / glp_only                |         0.5817 |      0.578  |      -0.0037 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.05 / gnn_joint               |         0.5809 |      0.5835 |       0.0027 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.05 / pooled_refit            |         0.5951 |      0.5969 |       0.0019 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.05 / pretrain_only           |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.05 / recal_intercept         |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.05 / recal_logistic          |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.05 / xgb_continue            |         0.5964 |      0.5965 |       0.0002 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.1 / glp_only                 |         0.6069 |      0.6071 |       0.0002 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.1 / gnn_joint                |         0.6124 |      0.6112 |      -0.0012 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.1 / pooled_refit             |         0.6057 |      0.6089 |       0.0032 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.1 / pretrain_only            |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.1 / recal_intercept          |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.1 / recal_logistic           |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.1 / xgb_continue             |         0.5969 |      0.5963 |      -0.0006 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.25 / glp_only                |         0.6108 |      0.6113 |       0.0005 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.25 / gnn_joint               |         0.6093 |      0.611  |       0.0016 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.25 / pooled_refit            |         0.6163 |      0.6141 |      -0.0022 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.25 / pretrain_only           |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.25 / recal_intercept         |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.25 / recal_logistic          |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.25 / xgb_continue            |         0.61   |      0.6118 |       0.0019 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.5 / glp_only                 |         0.6099 |      0.61   |       0.0001 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.5 / gnn_joint                |         0.6121 |      0.6109 |      -0.0011 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.5 / pooled_refit             |         0.611  |      0.6084 |      -0.0026 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.5 / pretrain_only            |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.5 / recal_intercept          |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.5 / recal_logistic           |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / hybrid / 0.5 / xgb_continue             |         0.6104 |      0.6091 |      -0.0013 |   1 |
| Stage 8: fine-tuning                    | real / none / 0.0 / pretrain_only              |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.05 / glp_only                  |         0.5709 |      0.5627 |      -0.0082 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.05 / gnn_joint                 |         0.5751 |      0.5751 |       0      |   1 |
| Stage 8: fine-tuning                    | real / real / 0.05 / pooled_refit              |         0.5723 |      0.572  |      -0.0002 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.05 / pretrain_only             |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.05 / recal_intercept           |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.05 / recal_logistic            |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.05 / xgb_continue              |         0.594  |      0.5818 |      -0.0122 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.1 / glp_only                   |         0.6057 |      0.6046 |      -0.0011 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.1 / gnn_joint                  |         0.6075 |      0.6058 |      -0.0017 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.1 / pooled_refit               |         0.6085 |      0.6082 |      -0.0003 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.1 / pretrain_only              |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.1 / recal_intercept            |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.1 / recal_logistic             |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.1 / xgb_continue               |         0.6078 |      0.6065 |      -0.0013 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.25 / glp_only                  |         0.6127 |      0.6127 |       0      |   1 |
| Stage 8: fine-tuning                    | real / real / 0.25 / gnn_joint                 |         0.6127 |      0.6127 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.25 / pooled_refit              |         0.6139 |      0.6139 |       0      |   1 |
| Stage 8: fine-tuning                    | real / real / 0.25 / pretrain_only             |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.25 / recal_intercept           |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.25 / recal_logistic            |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.25 / xgb_continue              |         0.6101 |      0.6123 |       0.0021 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.5 / glp_only                   |         0.6113 |      0.6113 |       0      |   1 |
| Stage 8: fine-tuning                    | real / real / 0.5 / gnn_joint                  |         0.6115 |      0.6115 |       0      |   1 |
| Stage 8: fine-tuning                    | real / real / 0.5 / pooled_refit               |         0.6118 |      0.6114 |      -0.0005 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.5 / pretrain_only              |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.5 / recal_intercept            |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.5 / recal_logistic             |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / real / 0.5 / xgb_continue               |         0.6113 |      0.612  |       0.0007 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.05 / glp_only             |         0.5485 |      0.5496 |       0.0011 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.05 / gnn_joint            |         0.5608 |      0.5858 |       0.0249 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.05 / pooled_refit         |         0.581  |      0.5905 |       0.0095 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.05 / pretrain_only        |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.05 / recal_intercept      |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.05 / recal_logistic       |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.05 / xgb_continue         |         0.6026 |      0.602  |      -0.0006 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.1 / glp_only              |         0.5328 |      0.5239 |      -0.0089 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.1 / gnn_joint             |         0.5619 |      0.5723 |       0.0103 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.1 / pooled_refit          |         0.6129 |      0.6078 |      -0.0051 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.1 / pretrain_only         |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.1 / recal_intercept       |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.1 / recal_logistic        |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.1 / xgb_continue          |         0.5965 |      0.5888 |      -0.0077 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.25 / glp_only             |         0.5285 |      0.5293 |       0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.25 / gnn_joint            |         0.6171 |      0.6151 |      -0.0021 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.25 / pooled_refit         |         0.6087 |      0.6126 |       0.0038 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.25 / pretrain_only        |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.25 / recal_intercept      |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.25 / recal_logistic       |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.25 / xgb_continue         |         0.6004 |      0.6064 |       0.0059 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.5 / glp_only              |         0.5717 |      0.5738 |       0.0021 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.5 / gnn_joint             |         0.6104 |      0.6155 |       0.0051 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.5 / pooled_refit          |         0.6051 |      0.6099 |       0.0048 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.5 / pretrain_only         |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.5 / recal_intercept       |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.5 / recal_logistic        |         0.5878 |      0.587  |      -0.0008 |   1 |
| Stage 8: fine-tuning                    | real / synthetic / 0.5 / xgb_continue          |         0.6069 |      0.6088 |       0.0019 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.05 / glp_only           |         0.5635 |      0.563  |      -0.0005 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.05 / gnn_joint          |         0.555  |      0.5684 |       0.0134 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.05 / pooled_refit       |         0.5657 |      0.5633 |      -0.0024 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.05 / pretrain_only      |         0.5011 |      0.5108 |       0.0097 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.05 / recal_intercept    |         0.5011 |      0.5108 |       0.0097 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.05 / recal_logistic     |         0.5011 |      0.5108 |       0.0097 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.05 / xgb_continue       |         0.5579 |      0.5622 |       0.0042 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.1 / glp_only            |         0.6146 |      0.6158 |       0.0012 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.1 / gnn_joint           |         0.6131 |      0.6182 |       0.0052 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.1 / pooled_refit        |         0.6178 |      0.6191 |       0.0014 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.1 / pretrain_only       |         0.5078 |      0.5271 |       0.0193 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.1 / recal_intercept     |         0.5078 |      0.5271 |       0.0193 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.1 / recal_logistic      |         0.5078 |      0.5271 |       0.0193 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.1 / xgb_continue        |         0.6134 |      0.6138 |       0.0004 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.25 / glp_only           |         0.6107 |      0.6114 |       0.0006 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.25 / gnn_joint          |         0.612  |      0.612  |      -0      |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.25 / pooled_refit       |         0.6088 |      0.6112 |       0.0024 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.25 / pretrain_only      |         0.5461 |      0.5509 |       0.0048 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.25 / recal_intercept    |         0.5461 |      0.5509 |       0.0048 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.25 / recal_logistic     |         0.5461 |      0.5509 |       0.0048 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.25 / xgb_continue       |         0.6088 |      0.6089 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.5 / glp_only            |         0.6104 |      0.6084 |      -0.002  |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.5 / gnn_joint           |         0.6109 |      0.6099 |      -0.001  |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.5 / pooled_refit        |         0.6142 |      0.6085 |      -0.0057 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.5 / pretrain_only       |         0.5143 |      0.5305 |       0.0162 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.5 / recal_intercept     |         0.5143 |      0.5305 |       0.0162 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.5 / recal_logistic      |         0.5143 |      0.5305 |       0.0162 |   1 |
| Stage 8: fine-tuning                    | synthetic / hybrid / 0.5 / xgb_continue        |         0.6123 |      0.6094 |      -0.0029 |   1 |
| Stage 8: fine-tuning                    | synthetic / none / 0.0 / pretrain_only         |         0.5182 |      0.542  |       0.0238 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.05 / glp_only             |         0.5958 |      0.5967 |       0.0008 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.05 / gnn_joint            |         0.6059 |      0.6034 |      -0.0025 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.05 / pooled_refit         |         0.6131 |      0.6043 |      -0.0088 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.05 / pretrain_only        |         0.5599 |      0.5703 |       0.0105 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.05 / recal_intercept      |         0.5599 |      0.5703 |       0.0105 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.05 / recal_logistic       |         0.5599 |      0.5703 |       0.0105 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.05 / xgb_continue         |         0.6013 |      0.6049 |       0.0035 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.1 / glp_only              |         0.6048 |      0.605  |       0.0002 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.1 / gnn_joint             |         0.6052 |      0.6    |      -0.0053 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.1 / pooled_refit          |         0.6066 |      0.6082 |       0.0015 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.1 / pretrain_only         |         0.5217 |      0.5516 |       0.0299 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.1 / recal_intercept       |         0.5217 |      0.5516 |       0.0299 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.1 / recal_logistic        |         0.5217 |      0.5516 |       0.0299 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.1 / xgb_continue          |         0.5906 |      0.5983 |       0.0078 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.25 / glp_only             |         0.6086 |      0.6086 |       0      |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.25 / gnn_joint            |         0.6091 |      0.6097 |       0.0005 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.25 / pooled_refit         |         0.6114 |      0.6124 |       0.001  |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.25 / pretrain_only        |         0.5778 |      0.5651 |      -0.0127 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.25 / recal_intercept      |         0.5778 |      0.5651 |      -0.0127 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.25 / recal_logistic       |         0.5778 |      0.5651 |      -0.0127 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.25 / xgb_continue         |         0.6111 |      0.6119 |       0.0008 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.5 / glp_only              |         0.6113 |      0.6113 |       0      |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.5 / gnn_joint             |         0.6116 |      0.6116 |       0.0001 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.5 / pooled_refit          |         0.6153 |      0.6092 |      -0.0061 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.5 / pretrain_only         |         0.5388 |      0.5414 |       0.0026 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.5 / recal_intercept       |         0.5388 |      0.5414 |       0.0026 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.5 / recal_logistic        |         0.5388 |      0.5414 |       0.0026 |   1 |
| Stage 8: fine-tuning                    | synthetic / real / 0.5 / xgb_continue          |         0.6075 |      0.6082 |       0.0008 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.05 / glp_only        |         0.5108 |      0.5126 |       0.0018 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.05 / gnn_joint       |         0.5474 |      0.5119 |      -0.0355 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.05 / pooled_refit    |         0.5378 |      0.5523 |       0.0146 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.05 / pretrain_only   |         0.5153 |      0.5423 |       0.027  |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.05 / recal_intercept |         0.5153 |      0.5423 |       0.027  |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.05 / recal_logistic  |         0.5153 |      0.5423 |       0.027  |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.05 / xgb_continue    |         0.5315 |      0.558  |       0.0266 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.1 / glp_only         |         0.5146 |      0.5305 |       0.0159 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.1 / gnn_joint        |         0.5717 |      0.5499 |      -0.0218 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.1 / pooled_refit     |         0.5469 |      0.577  |       0.0301 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.1 / pretrain_only    |         0.5005 |      0.4999 |      -0.0006 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.1 / recal_intercept  |         0.5005 |      0.4999 |      -0.0006 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.1 / recal_logistic   |         0.5005 |      0.4999 |      -0.0006 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.1 / xgb_continue     |         0.528  |      0.5458 |       0.0178 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.25 / glp_only        |         0.5571 |      0.5599 |       0.0028 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.25 / gnn_joint       |         0.5517 |      0.5557 |       0.004  |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.25 / pooled_refit    |         0.5812 |      0.5868 |       0.0055 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.25 / pretrain_only   |         0.5355 |      0.5617 |       0.0262 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.25 / recal_intercept |         0.5355 |      0.5617 |       0.0262 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.25 / recal_logistic  |         0.5355 |      0.5617 |       0.0262 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.25 / xgb_continue    |         0.5741 |      0.5853 |       0.0112 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.5 / glp_only         |         0.5634 |      0.5663 |       0.0028 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.5 / gnn_joint        |         0.5909 |      0.5746 |      -0.0164 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.5 / pooled_refit     |         0.5853 |      0.5972 |       0.0119 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.5 / pretrain_only    |         0.5289 |      0.545  |       0.0161 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.5 / recal_intercept  |         0.5289 |      0.545  |       0.0161 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.5 / recal_logistic   |         0.5289 |      0.545  |       0.0161 |   1 |
| Stage 8: fine-tuning                    | synthetic / synthetic / 0.5 / xgb_continue     |         0.5821 |      0.5745 |      -0.0076 |   1 |
| Stage 3: full factorial                 | all 45744 configurations                       |         0.6406 |      0.6417 |       0.0011 |   1 |

### Feature combinations and the compounding hypothesis

**Feature combinations and the compounding hypothesis.** Across all 511 combinations of nine feature groups, the largest Shapley contributions to XGBoost AUROC came from setting (0.0350), duration (0.0290), sex (0.0270), drug (0.0212). For the hypothesis that individually negligible features contribute in combination, no feature group met the prespecified criteria; no pair showed significant positive interaction. For an optional-input calculator, a single XGBoost trained with random feature-group dropout differed from a bank of dedicated sub-models by -0.0121 AUROC on average across input patterns.

Table: Best feature-group combinations by number of groups (two best per size; mean over folds; threshold = training top 10%; q, Benjamini-Hochberg over all combinations and models; full results for all 511 combinations and every model in Supplementary Workbook S1).

| Feature groups                                                          | Model   |   AUROC |      SD | q (AUROC>0.5)   |   Sens |   Spec |    Acc |     PPV |    NPV | q (PPV>prev)   |
|:------------------------------------------------------------------------|:--------|--------:|--------:|:----------------|-------:|-------:|-------:|--------:|-------:|:---------------|
| duration                                                                | gnn     |  0.6322 | 0.00684 | <0.001          | 0.3989 | 0.7579 | 0.7578 | 0.00074 | 0.9997 | <0.001         |
| duration                                                                | xgb     |  0.6306 | 0.00753 | <0.001          | 0.3758 | 0.7804 | 0.7803 | 0.00077 | 0.9997 | <0.001         |
| drug|setting                                                            | xgb     |  0.6532 | 0.00967 | <0.001          | 0.3266 | 0.8378 | 0.8375 | 0.0009  | 0.9997 | <0.001         |
| drug|setting                                                            | cnn1d   |  0.6488 | 0.00993 | <0.001          | 0.3199 | 0.8452 | 0.845  | 0.00095 | 0.9997 | <0.001         |
| exposure|setting|sex                                                    | xgb     |  0.6594 | 0.00976 | <0.001          | 0.3424 | 0.8343 | 0.8341 | 0.00094 | 0.9997 | <0.001         |
| duration|exposure|setting                                               | xgb     |  0.6576 | 0.00878 | <0.001          | 0.3386 | 0.8359 | 0.8357 | 0.00092 | 0.9997 | <0.001         |
| age|exposure|setting|sex                                                | xgb     |  0.6601 | 0.00764 | <0.001          | 0.347  | 0.831  | 0.8308 | 0.00092 | 0.9997 | <0.001         |
| exposure|hypertension|setting|sex                                       | xgb     |  0.6595 | 0.00811 | <0.001          | 0.3429 | 0.8336 | 0.8334 | 0.00094 | 0.9997 | <0.001         |
| age|exposure|hypertension|setting|sex                                   | xgb     |  0.6595 | 0.00757 | <0.001          | 0.3529 | 0.8267 | 0.8265 | 0.00092 | 0.9997 | <0.001         |
| diabetes|exposure|indication|setting|sex                                | xgb     |  0.6591 | 0.0132  | <0.001          | 0.3292 | 0.8453 | 0.8451 | 0.00096 | 0.9997 | <0.001         |
| age|duration|exposure|hypertension|setting|sex                          | xgb     |  0.6591 | 0.00841 | <0.001          | 0.3432 | 0.8338 | 0.8336 | 0.00093 | 0.9997 | <0.001         |
| diabetes|exposure|hypertension|indication|setting|sex                   | xgb     |  0.6584 | 0.0108  | <0.001          | 0.3291 | 0.8452 | 0.8449 | 0.00096 | 0.9997 | <0.001         |
| age|diabetes|exposure|hypertension|indication|setting|sex               | xgb     |  0.6566 | 0.0111  | <0.001          | 0.3436 | 0.8311 | 0.8309 | 0.00093 | 0.9997 | <0.001         |
| age|diabetes|drug|exposure|indication|setting|sex                       | xgb     |  0.6565 | 0.0103  | <0.001          | 0.3526 | 0.8199 | 0.8197 | 0.00087 | 0.9997 | <0.001         |
| age|diabetes|drug|duration|exposure|indication|setting|sex              | xgb     |  0.6558 | 0.0104  | <0.001          | 0.3513 | 0.8209 | 0.8207 | 0.00087 | 0.9997 | <0.001         |
| age|diabetes|duration|exposure|hypertension|indication|setting|sex      | xgb     |  0.6551 | 0.0118  | <0.001          | 0.3474 | 0.8228 | 0.8226 | 0.00087 | 0.9997 | <0.001         |
| age|diabetes|drug|duration|exposure|hypertension|indication|setting|sex | xgb     |  0.6551 | 0.012   | <0.001          | 0.3492 | 0.8209 | 0.8207 | 0.00087 | 0.9997 | <0.001         |
| age|diabetes|drug|duration|exposure|hypertension|indication|setting|sex | rf      |  0.6513 | 0.0128  | <0.001          | 0.321  | 0.8412 | 0.841  | 0.0009  | 0.9996 | <0.001         |

Table: Odds-ratio reliability of each feature across all combinations that contain it (count-weighted logistic regression with cluster-robust standard errors by study; study-level covariates are ecological and imprecise with 11 clusters).

| Feature                    |   Models |   Median OR |   Min OR | Max OR   | OR>1   | q<0.05   |
|:---------------------------|---------:|------------:|---------:|:---------|:-------|:---------|
| therapy_dpp4i              |      256 |        0.73 |     0.24 | 1.86     | 25%    | 100%     |
| source_type_single_center  |      256 |      254.36 |     0.01 | >10^4    | 99%    | 91%      |
| therapy_other_antiobesity  |      256 |        0.19 |     0.05 | 32.57    | 16%    | 91%      |
| pct_female                 |      256 |        0.61 |     0.06 | 1.42     | 13%    | 84%      |
| indication_obesity         |      256 |       34.12 |     0.11 | 182.78   | 56%    | 78%      |
| region_uk                  |      256 |        0.34 |     0.01 | 49.64    | 16%    | 75%      |
| glp1ra_exposed             |      256 |        1.59 |     1.04 | 1.92     | 100%   | 74%      |
| indication_iih             |      256 |        2.99 |     0    | >10^4    | 69%    | 67%      |
| source_type_va_ehr         |      224 |        4.99 |     0    | 298.92   | 73%    | 65%      |
| therapy_sglt2i             |      128 |        0.58 |     0.52 | 0.97     | 0%     | 62%      |
| abcd_D                     |      128 |        0.29 |     0.01 | 8.11     | 27%    | 62%      |
| abcd_B                     |      256 |        1.33 |     0.96 | 8.92     | 98%    | 61%      |
| region_nordic              |      256 |        1.33 |     0    | 153.49   | 70%    | 57%      |
| outcome_def_ion            |      216 |        1.08 |     0.16 | >10^4    | 53%    | 56%      |
| source_type_ehr_network    |      256 |        1.38 |     0    | 96.85    | 66%    | 50%      |
| mean_age                   |      256 |        0    |     0    | >10^4    | 30%    | 44%      |
| abcd_A                     |      256 |        9.69 |     0    | >10^4    | 69%    | 39%      |
| followup_years             |      256 |        0.98 |     0.67 | 22.81    | 47%    | 36%      |
| therapy_no_glp1ra          |      256 |        0.99 |     0.86 | 1.82     | 49%    | 29%      |
| therapy_other_antidiabetic |      256 |        0.93 |     0.36 | 6.10     | 45%    | 12%      |
| therapy_sema_or_tirz       |      256 |        1.07 |     0.53 | 7.63     | 71%    | 0%       |
| therapy_semaglutide        |      256 |        1.12 |     0.77 | 2.47     | 89%    | 0%       |

### Compute footprint and imaging what-if

**Compute, energy and token use.** All model training and inference ran locally on one laptop (Apple M3 Pro, 11 cores, 18 GB). Across every experiment the analyses used 7.6 wall-clock hours and 62.6 core-hours (87% of run time measured from logs, the rest estimated), consuming an estimated 0.25 kWh (range 0.19 to 0.31) and emitting 92 g CO2e at the US average grid intensity (range 19 to 215 g), about 0.5 km of car travel. The largest single consumer was the full-factorial twins (2.4 h). Inference was negligible: logreg 0.75 ms, rf 26.68 ms, xgb 1.69 ms per 1,000 patients. The AI assistant used for retrieval, extraction and drafting consumed at least 764,562 tokens in the four metered extraction and literature agents; tokens of the main session were not metered, and assistant inference energy, which occurs in remote data centres, is not included in these totals.

**What-if: adding ocular imaging (simulation).** Imaging values were simulated for the reconstructed patients conditional on their observed outcome. Adding cup-to-disc ratio from fundus photographs, with published odds ratios [@p42556433] and 25% photographic misclassification, raised AUROC from 0.653 to 0.770; OCT-measured ratio reached 0.818. Adding hypothetical effects for optic disc drusen, OCT angiography perfusion and serial on-treatment imaging reached 0.840 (range across hypothetical effect sizes 0.825 to 0.876), and 0.854 with sleep apnoea added; sleep apnoea alone reached 0.708. Positive predictive value at the top-10% threshold rose from 0.098% to 0.262%, still below 1% because NAION is rare. A prospective cohort would need about 263 NAION events, roughly 604,000 GLP-1RA initiators at the observed incidence, to estimate AUROC within ±0.03; a case-control design enriched for NAION would need far fewer imaged patients. These figures are simulations that assume conditional independence of features and, for drusen, OCT angiography and serial imaging, effect sizes that have not been measured.

![Simulated AUROC after adding ocular imaging modalities and sleep apnoea to the published-feature model; black bars show the range across hypothetical effect sizes.](results/stage16/fig_imaging_whatif.png){width=100%}

![Stand-alone gain vs Shapley value of each feature group (XGBoost) and pairwise Shapley interactions across all 511 feature combinations.](results/stage14/fig_feature_combinations.png){width=100%}

![AUROC and PPV of each model within non-users and within GLP-1RA users (mean ± SD over ten folds).](results/stage11/fig_subgroups.png){width=100%}

![NAION incidence per 100,000 person-years by study age band, indication and drug; grey band = published background incidence in adults aged 50 years or older.](results/stage10/fig_incidence_strata.png){width=100%}

## Complete experimental results

### A. Data inventory and modalities

Table: Data sources by type and modality. Structured tabular data (categorical and numeric) were the only modality usable for modelling; imaging existed only as figures in case reports and no video was located.

| Source                                                                | Data type                                                | Modalities                            |   No. of modalities |   Records | Patients   | Used in                                                           |
|:----------------------------------------------------------------------|:---------------------------------------------------------|:--------------------------------------|--------------------:|----------:|:-----------|:------------------------------------------------------------------|
| PubMed bibliographic records                                          | unstructured text (title, abstract) + metadata           | text                                  |                   1 |       169 | —          | screening, extraction                                             |
| PMC open-access full texts                                            | unstructured text + embedded tables                      | text, tabular                         |                   2 |        60 | —          | extraction                                                        |
| Cohort / case-control studies (extracted)                             | structured tabular: categorical + numeric counts/effects | categorical, numeric                  |                   2 |       166 | —          | pseudo-IPD (Stage 1), meta-regression, twin generators            |
| Reconstructed pseudo-IPD                                              | structured tabular (count-weighted patient rows)         | categorical, numeric, binary outcome  |                   2 |        48 | 3,271,499  | Stage 1 supervised ML, Stage 2 test sets                          |
| Case reports / series (extracted)                                     | structured tabular from narrative text                   | categorical, numeric, free-text notes |                   3 |        20 | 20         | unsupervised phenotyping                                          |
| Case-report ophthalmic imaging (figures only, not analysed as pixels) | image (fundus photo, OCT RNFL/GCIPL, visual fields, MRI) | image                                 |                   1 |        17 | 8          | inventory only (pre/post pairs counted); insufficient n for CV/DL |
| Pharmacovigilance (FAERS, VigiBase, EudraVigilance)                   | structured tabular (disproportionality statistics)       | categorical, numeric                  |                   2 |        25 | —          | external priors, discussion                                       |
| RCT-pooled analyses and meta-analyses                                 | structured tabular (pooled estimates)                    | categorical, numeric                  |                   2 |        44 | —          | external priors for synthetic-only (h=100%) generators            |
| Video                                                                 | -                                                        | none located                          |                   0 |         0 | 0          | -                                                                 |

Table: Reconstructed cohorts.

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

![Study flow and validation design.](paper/figures/fig1_study_flow.png){width=100%}

### B. PubMed-only supervised models

![ROC, precision-recall and calibration on the 90% hold-out.](results/stage1/fig_holdout_roc_pr_cal.png){width=100%}

![AUROC and PR-AUC across ten disjoint 10% training folds.](results/stage1/fig_kfold.png){width=100%}

![Training-fraction sweep (10% to 90%, 10 repeats each): AUROC, O/E and SHAP-rank stability (Kendall W).](results/stage1c/fig_train_fraction_curves.png){width=100%}

Table: Training-fraction sweep (mean over 10 repeats).

| model   | train_frac   |   auroc |   auroc_sd |   pr_auc |   brier |   cal_slope |    o_e |   o_e_sd |
|:--------|:-------------|--------:|-----------:|---------:|--------:|------------:|-------:|---------:|
| logreg  | 10%          |  0.6122 |     0.0127 |   0.0007 |  0.0004 |      0.9831 | 1.0125 |   0.0471 |
| logreg  | 20%          |  0.6257 |     0.0174 |   0.0011 |  0.0004 |      0.9851 | 0.9944 |   0.0779 |
| logreg  | 30%          |  0.6259 |     0.0093 |   0.001  |  0.0004 |      0.9849 | 1.0013 |   0.0861 |
| logreg  | 40%          |  0.628  |     0.0085 |   0.0009 |  0.0004 |      1.0021 | 0.8647 |   0.0313 |
| logreg  | 50%          |  0.6394 |     0.0131 |   0.0008 |  0.0004 |      0.9942 | 0.93   |   0.0418 |
| logreg  | 60%          |  0.6374 |     0.0087 |   0.0009 |  0.0004 |      1.0002 | 0.879  |   0.0345 |
| logreg  | 70%          |  0.626  |     0.0125 |   0.0013 |  0.0004 |      1.0084 | 0.8244 |   0.0422 |
| logreg  | 80%          |  0.6345 |     0.0109 |   0.0015 |  0.0004 |      1.005  | 0.8498 |   0.0365 |
| logreg  | 90%          |  0.6313 |     0.0118 |   0.0035 |  0.0004 |      1.0018 | 0.8719 |   0.0921 |
| rf      | 10%          |  0.6475 |     0.0074 |   0.0008 |  0.0004 |      0.9823 | 1.0141 |   0.0565 |
| rf      | 20%          |  0.6567 |     0.0063 |   0.001  |  0.0004 |      0.9864 | 0.9907 |   0.0697 |
| rf      | 30%          |  0.6625 |     0.0082 |   0.0009 |  0.0004 |      0.9869 | 0.9895 |   0.0842 |
| rf      | 40%          |  0.6618 |     0.0069 |   0.0009 |  0.0004 |      0.9844 | 1.0004 |   0.0509 |
| rf      | 50%          |  0.6623 |     0.0071 |   0.0009 |  0.0004 |      0.9877 | 0.9742 |   0.031  |
| rf      | 60%          |  0.6656 |     0.0062 |   0.0009 |  0.0004 |      0.9861 | 0.9876 |   0.0443 |
| rf      | 70%          |  0.6592 |     0.0137 |   0.0009 |  0.0004 |      0.9856 | 0.9914 |   0.0632 |
| rf      | 80%          |  0.6647 |     0.0115 |   0.0009 |  0.0004 |      0.9874 | 0.9786 |   0.048  |
| rf      | 90%          |  0.6538 |     0.0217 |   0.0009 |  0.0004 |      0.9891 | 0.9658 |   0.1082 |
| xgb     | 10%          |  0.6542 |     0.0097 |   0.0043 |  0.0004 |      0.9792 | 1.0126 |   0.06   |
| xgb     | 20%          |  0.6617 |     0.0058 |   0.0044 |  0.0004 |      0.9848 | 0.9914 |   0.069  |
| xgb     | 30%          |  0.667  |     0.0066 |   0.0047 |  0.0004 |      0.9864 | 0.9898 |   0.0844 |
| xgb     | 40%          |  0.6704 |     0.0041 |   0.0048 |  0.0004 |      0.9833 | 1.0011 |   0.0513 |
| xgb     | 50%          |  0.6695 |     0.0053 |   0.0047 |  0.0004 |      0.9873 | 0.9737 |   0.0304 |
| xgb     | 60%          |  0.674  |     0.0063 |   0.0054 |  0.0004 |      0.9856 | 0.9872 |   0.0439 |
| xgb     | 70%          |  0.6679 |     0.015  |   0.0051 |  0.0004 |      0.9843 | 0.99   |   0.063  |
| xgb     | 80%          |  0.6733 |     0.0099 |   0.0053 |  0.0004 |      0.9867 | 0.9799 |   0.0475 |
| xgb     | 90%          |  0.6641 |     0.0227 |   0.0053 |  0.0004 |      0.9885 | 0.9621 |   0.1082 |

![SHAP beeswarm, XGBoost on the 90% hold-out.](results/stage1/fig_shap_beeswarm_xgb.png){width=85%}

![SHAP importance across ten folds.](results/stage1/fig_shap_kfold_importance.png){width=85%}

### C. Statistical inference (all P values, Benjamini-Hochberg q within family)

Table: Within-study (study fixed effects) associations.

| Term                          | Model                                                                            |    OR |   95% CI low |   95% CI high | P      | q      |
|:------------------------------|:---------------------------------------------------------------------------------|------:|-------------:|--------------:|:-------|:-------|
| glp1ra_exposed                | A2 study fixed effects: any GLP-1RA vs comparator                                | 1.587 |        1.422 |         1.772 | <0.001 | <0.001 |
| exposed_semaglutide           | A2 study FE: exposure by drug                                                    | 1.73  |        1.439 |         2.08  | <0.001 | <0.001 |
| exposed_glp1ra_class          | A2 study FE: exposure by drug                                                    | 1.496 |        1.3   |         1.721 | <0.001 | <0.001 |
| exposed_sema_or_tirz          | A2 study FE: exposure by drug                                                    | 1.842 |        1.054 |         3.221 | 0.032  | 0.044  |
| exposed_t2d                   | A2 study FE: exposure by indication                                              | 1.622 |        1.448 |         1.817 | <0.001 | <0.001 |
| exposed_obesity               | A2 study FE: exposure by indication                                              | 2.697 |        1.637 |         4.441 | <0.001 | <0.001 |
| exposed_iih                   | A2 study FE: exposure by indication                                              | 0.428 |        0.218 |         0.842 | 0.014  | 0.021  |
| exposed_vs_dpp4i              | A2 study FE: exposure by comparator class                                        | 1.029 |        0.571 |         1.855 | 0.924  | 0.943  |
| exposed_vs_no_glp1ra          | A2 study FE: exposure by comparator class                                        | 1.143 |        0.936 |         1.394 | 0.189  | 0.236  |
| exposed_vs_other_antidiabetic | A2 study FE: exposure by comparator class                                        | 2.348 |        1.743 |         3.164 | <0.001 | <0.001 |
| exposed_vs_sglt2i             | A2 study FE: exposure by comparator class                                        | 1.806 |        1.544 |         2.113 | <0.001 | <0.001 |
| glp1ra_exposed                | A2 study FE: exposure effect modification (study-level age, % female, follow-up) | 1.411 |        1.244 |         1.602 | <0.001 | <0.001 |
| exposed_x_age_per_10y         | A2 study FE: exposure effect modification (study-level age, % female, follow-up) | 1.989 |        1.301 |         3.041 | 0.001  | 0.002  |
| exposed_x_female_per_10pct    | A2 study FE: exposure effect modification (study-level age, % female, follow-up) | 0.996 |        0.886 |         1.12  | 0.943  | 0.943  |
| exposed_x_followup_years      | A2 study FE: exposure effect modification (study-level age, % female, follow-up) | 1.021 |        0.945 |         1.103 | 0.595  | 0.687  |

Table: Marginal associations (cluster-robust SE by study; ecological, setting-confounded).

| Term                       |      OR |   95% CI low |   95% CI high | P      | q      |
|:---------------------------|--------:|-------------:|--------------:|:-------|:-------|
| therapy_dpp4i              |   1.055 |        0.712 |         1.564 | 0.789  | 0.789  |
| therapy_glp1ra_class       |   1.086 |        0.733 |         1.61  | 0.681  | 0.766  |
| therapy_other_antidiabetic |   0.628 |        0.342 |         1.154 | 0.134  | 0.201  |
| therapy_other_antiobesity  |   0.123 |        0.096 |         0.156 | <0.001 | <0.001 |
| therapy_sema_or_tirz       |   1.158 |        0.63  |         2.127 | 0.637  | 0.764  |
| therapy_semaglutide        |   1.215 |        0.954 |         1.547 | 0.114  | 0.187  |
| therapy_sglt2i             |   0.614 |        0.424 |         0.89  | 0.010  | 0.023  |
| indication_iih             |  86.796 |        1.014 |      7429.09  | 0.049  | 0.089  |
| indication_obesity         |   0.63  |        0.402 |         0.985 | 0.043  | 0.086  |
| region_nordic              |   1.951 |        1.52  |         2.506 | <0.001 | <0.001 |
| region_uk                  |   0.213 |        0.182 |         0.25  | <0.001 | <0.001 |
| source_type_claims         |   0.54  |        0.18  |         1.627 | 0.274  | 0.379  |
| source_type_single_center  | 326.045 |      217.289 |       489.235 | <0.001 | <0.001 |
| source_type_va_ehr         |   0.551 |        0.09  |         3.385 | 0.520  | 0.669  |
| outcome_def_ion            |   0.494 |        0.415 |         0.587 | <0.001 | <0.001 |
| age_per_10y                |   1.538 |        0.098 |        24.122 | 0.759  | 0.789  |
| female_per_10pct           |   0.499 |        0.301 |         0.828 | 0.007  | 0.018  |
| followup_years             |   0.756 |        0.697 |         0.82  | <0.001 | <0.001 |

Table: SHAP permutation tests (B=500).

| Feature        |   Mean |SHAP| |   P (global null) |     q |   P (within-study null) |
|:---------------|--------------:|------------------:|------:|------------------------:|
| followup_years |         0.031 |             0.186 | 0.209 |                   0.996 |
| glp1ra_exposed |         0.118 |             0.002 | 0.004 |                   0.002 |
| indication     |         0.015 |             0.008 | 0.014 |                   1     |
| mean_age       |         0.057 |             0.026 | 0.033 |                   0.255 |
| outcome_def    |         0.011 |             0.228 | 0.228 |                   0.03  |
| pct_female     |         0.112 |             0.002 | 0.004 |                   0.146 |
| region         |         0.263 |             0.002 | 0.004 |                   0.369 |
| source_type    |         0.143 |             0.002 | 0.004 |                   0.916 |
| therapy        |         0.141 |             0.01  | 0.015 |                   0.012 |

Table: Random-effects meta-analysis of published estimates.

| Analysis                                 |   k |   Pooled ratio |   95% CI low |   95% CI high |      I2 | P      | q      |
|:-----------------------------------------|----:|---------------:|-------------:|--------------:|--------:|:-------|:-------|
| all studies (most precise estimate each) |  25 |          1.341 |        1.161 |         1.549 |  86.553 | <0.001 | <0.001 |
| indication = t2d                         |  20 |          1.363 |        1.163 |         1.598 |  87.828 | <0.001 | <0.001 |
| indication = t2d_or_obesity              |   2 |          1.859 |        0.764 |         4.525 |  86.938 | 0.172  | 0.229  |
| subgroup difference by indication        |   2 |        nan     |      nan     |       nan     | nan     | 0.501  | 0.501  |
| drug = any_glp1ra                        |  15 |          1.302 |        1.128 |         1.504 |  81.066 | <0.001 | <0.001 |
| drug = semaglutide                       |   8 |          1.569 |        1.007 |         2.444 |  91.924 | 0.047  | 0.075  |
| subgroup difference by drug              |   2 |        nan     |      nan     |       nan     | nan     | 0.434  | 0.496  |
| excluding IIH cohort                     |  24 |          1.38  |        1.196 |         1.592 |  86.354 | <0.001 | <0.001 |

Table: Demographic contrasts.

| Contrast                                                 |   observed |   expected | test                                           |     P |     q |
|:---------------------------------------------------------|-----------:|-----------:|:-----------------------------------------------|------:|------:|
| case reports: male share vs exposed-cohort male share    |      0.688 |       0.49 | exact binomial                                 | 0.137 | 0.274 |
| case reports: age vs exposed-cohort mean age             |     47     |      60.15 | Wilcoxon signed-rank vs cohort mean            | 0.003 | 0.016 |
| case reports: obesity indication share vs 50%            |      0.786 |       0.5  | exact binomial                                 | 0.057 | 0.172 |
| meta-analysis 42612051: age <65 vs >=65                  |      2.187 |       1    | z-test of ratio of ratios (from published CIs) | 0.272 | 0.407 |
| meta-analysis 42612051: female- vs male-dominant cohorts |      0.951 |       1    | z-test of ratio of ratios (from published CIs) | 0.919 | 0.919 |
| meta-analysis 42612051: diabetes vs obesity indication   |      1.095 |       1    | z-test of ratio of ratios (from published CIs) | 0.85  | 0.919 |

Table: Tests of ML experiments.

| Experiment                                               |   estimate | test                                | P      | q      |
|:---------------------------------------------------------|-----------:|:------------------------------------|:-------|:-------|
| logreg: AUROC > 0.5 (10/90 hold-out)                     |      0.627 | Hanley-McNeil one-sided             | <0.001 | <0.001 |
| rf: AUROC > 0.5 (10/90 hold-out)                         |      0.642 | Hanley-McNeil one-sided             | <0.001 | <0.001 |
| xgb: AUROC > 0.5 (10/90 hold-out)                        |      0.657 | Hanley-McNeil one-sided             | <0.001 | <0.001 |
| xgb vs logreg: AUROC difference (10 x 10% folds)         |      0.041 | Nadeau-Bengio corrected resampled t | 0.218  | 0.277  |
| xgb vs rf: AUROC difference (10 x 10% folds)             |      0.011 | Nadeau-Bengio corrected resampled t | 0.745  | 0.745  |
| rf vs logreg: AUROC difference (10 x 10% folds)          |      0.03  | Nadeau-Bengio corrected resampled t | 0.314  | 0.339  |
| logreg: leave-one-study-out AUROC > 0.5                  |      0.51  | Wilcoxon over 11 held-out studies   | 0.289  | 0.337  |
| rf: leave-one-study-out AUROC > 0.5                      |      0.532 | Wilcoxon over 11 held-out studies   | 0.087  | 0.136  |
| xgb: leave-one-study-out AUROC > 0.5                     |      0.535 | Wilcoxon over 11 held-out studies   | 0.021  | 0.037  |
| logreg: AUROC trend with training fraction (10-90%)      |      0.304 | Spearman (90 runs)                  | 0.004  | 0.008  |
| rf: AUROC trend with training fraction (10-90%)          |      0.266 | Spearman (90 runs)                  | 0.011  | 0.023  |
| xgb: AUROC trend with training fraction (10-90%)         |      0.389 | Spearman (90 runs)                  | <0.001 | <0.001 |
| Stage 2 hybrid best vs PubMed-only (XGB, paired)         |      0.001 | paired z over 27 split/seed pairs   | 0.167  | 0.233  |
| Stage 2 synthetic-only best vs PubMed-only (XGB, paired) |     -0.01  | paired z over 27 split/seed pairs   | <0.001 | <0.001 |

![Within-study odds ratios.](paper/figures/fig_forest_within_study.png){width=100%}

![Most precise published estimate per study (log scale).](results/stage1b/fig_forest.png){width=80%}

### D. Threshold metrics (hold-out; P values against chance)

Table: Sensitivity, specificity, accuracy, PPV and NPV with P values; NIR, no-information rate; NB, net benefit at the threshold probability.

| Model    | Threshold   |   AUROC |   Sens | P      |   Spec |    P  |    Acc |   P vs NIR |      PPV | P vs prev   |    NPV | P      |    LR+ |       NB |   NB treat-all |
|:---------|:------------|--------:|-------:|:-------|-------:|------:|-------:|-----------:|---------:|:------------|-------:|:-------|-------:|---------:|---------------:|
| logreg   | Youden      |  0.6275 | 0.808  | <0.001 | 0.3958 | 0.377 | 0.396  |          1 | 0.000587 | <0.001      | 0.9998 | <0.001 | 1.3373 | 9.97e-05 |       1.7e-05  |
| logreg   | Top 10%     |  0.6275 | 0.2167 | <0.001 | 0.8697 | 0.424 | 0.8694 |          1 | 0.00073  | <0.001      | 0.9996 | <0.001 | 1.6631 | 3.73e-05 |      -4.75e-06 |
| logreg   | Top 20%     |  0.6275 | 0.3769 | <0.001 | 0.764  | 0.402 | 0.7638 |          1 | 0.000701 | <0.001      | 0.9996 | <0.001 | 1.5969 | 6.19e-05 |       3.39e-07 |
| rf       | Youden      |  0.6421 | 0.719  | <0.001 | 0.5107 | 0.365 | 0.5108 |          1 | 0.000645 | <0.001      | 0.9998 | <0.001 | 1.4694 | 9.86e-05 |      -4.56e-06 |
| rf       | Top 10%     |  0.6421 | 0.2051 | <0.001 | 0.9309 | 0.344 | 0.9305 |          1 | 0.0013   | <0.001      | 0.9996 | <0.001 | 2.9667 | 5.04e-05 |      -0.000134 |
| rf       | Top 20%     |  0.6421 | 0.3444 | <0.001 | 0.7934 | 0.399 | 0.7932 |          1 | 0.000731 | <0.001      | 0.9996 | <0.001 | 1.6674 | 4.17e-05 |      -9.08e-05 |
| xgb      | Youden      |  0.6569 | 0.719  | <0.001 | 0.5107 | 0.365 | 0.5108 |          1 | 0.000645 | <0.001      | 0.9998 | <0.001 | 1.4694 | 0.000108 |       1.51e-05 |
| xgb      | Top 10%     |  0.6569 | 0.3235 | <0.001 | 0.85   | 0.358 | 0.8498 |          1 | 0.000946 | <0.001      | 0.9997 | <0.001 | 2.1572 | 4.31e-05 |      -0.00022  |
| xgb      | Top 20%     |  0.6569 | 0.3754 | <0.001 | 0.7935 | 0.377 | 0.7933 |          1 | 0.000797 | <0.001      | 0.9997 | <0.001 | 1.8177 | 3.53e-05 |      -0.000188 |
| ensemble | Youden      |  0.6568 | 0.719  | <0.001 | 0.5107 | 0.365 | 0.5108 |          1 | 0.000645 | <0.001      | 0.9998 | <0.001 | 1.4694 | 9.73e-05 |      -7.17e-06 |
| ensemble | Top 10%     |  0.6568 | 0.3289 | <0.001 | 0.8449 | 0.359 | 0.8446 |          1 | 0.00093  | <0.001      | 0.9997 | <0.001 | 2.1204 | 5.77e-05 |      -0.00012  |
| ensemble | Top 20%     |  0.6568 | 0.3808 | <0.001 | 0.7883 | 0.378 | 0.7881 |          1 | 0.000789 | <0.001      | 0.9997 | <0.001 | 1.799  | 5.49e-05 |      -9.12e-05 |

Table: Threshold metrics across ten folds (mean).

| model    | threshold   |   auroc_mean |   sensitivity_mean |   specificity_mean |   accuracy_mean |   ppv_mean |   npv_mean |   lr_pos_mean |   flagged_frac_mean |
|:---------|:------------|-------------:|-------------------:|-------------------:|----------------:|-----------:|-----------:|--------------:|--------------------:|
| ensemble | T1_youden   |      0.65561 |            0.57248 |            0.62969 |         0.62967 |    0.00069 |    0.99971 |       1.57919 |             0.3704  |
| ensemble | T2_top10pct |      0.65561 |            0.31414 |            0.85511 |         0.85487 |    0.00101 |    0.99965 |       2.31932 |             0.14497 |
| ensemble | T3_top20pct |      0.65561 |            0.42501 |            0.76448 |         0.76433 |    0.0008  |    0.99967 |       1.82865 |             0.2356  |
| logreg   | T1_youden   |      0.61586 |            0.55674 |            0.61235 |         0.61233 |    0.00064 |    0.99969 |       1.47306 |             0.38772 |
| logreg   | T2_top10pct |      0.61586 |            0.25847 |            0.8481  |         0.84784 |    0.00075 |    0.99962 |       1.72242 |             0.15195 |
| logreg   | T3_top20pct |      0.61586 |            0.36911 |            0.76226 |         0.76209 |    0.00068 |    0.99964 |       1.55374 |             0.2378  |
| rf       | T1_youden   |      0.64577 |            0.59429 |            0.60213 |         0.60213 |    0.00066 |    0.99971 |       1.52031 |             0.39796 |
| rf       | T2_top10pct |      0.64577 |            0.3105  |            0.8459  |         0.84567 |    0.00096 |    0.99965 |       2.20604 |             0.15417 |
| rf       | T3_top20pct |      0.64577 |            0.40869 |            0.76917 |         0.76901 |    0.0008  |    0.99967 |       1.83686 |             0.23091 |
| xgb      | T1_youden   |      0.65646 |            0.59262 |            0.61114 |         0.61113 |    0.00068 |    0.99971 |       1.55053 |             0.38895 |
| xgb      | T2_top10pct |      0.65646 |            0.3522  |            0.8204  |         0.8202  |    0.00087 |    0.99966 |       2.00968 |             0.17967 |
| xgb      | T3_top20pct |      0.65646 |            0.43888 |            0.75359 |         0.75345 |    0.00078 |    0.99968 |       1.79597 |             0.24649 |

### E. Case-level phenotyping and study-level meta-regression

Among 18 NAION case patients (median age 47; 69% men; 15 semaglutide, 2 tirzepatide, 1 liraglutide; median onset 24 weeks), k-means, Gaussian mixture and Ward clustering found no stable multi-patient phenotype: the best solution (k=2, silhouette 0.41) isolated a single tirzepatide case. A random-forest meta-regression of 148 published estimates from 25 studies did not predict effect size in held-out studies (grouped cross-validated R² -0.06).

![Case-report phenotypes (PCA).](results/stage1b/fig_case_phenotypes_pca.png){width=60%}

![SHAP for the study-level meta-regression.](results/stage1b/fig_meta_shap.png){width=80%}

### F. Graph neural networks

The GraphSAGE GNN reached an AUROC of 0.649 ± 0.014 across the 10 folds, not different from XGBoost on identical splits (difference -0.0077; P=0.779), with O/E 1.01. In leave-one-study-out validation its AUROC was 0.531 (P=0.034 vs 0.5). Integrated-gradient attributions ranked therapy, data source and region highest, mirroring SHAP.

Nested tuning over 366 GNN configurations selected a SAGE network of depth 6 with k=20 neighbours, uniform edge weighting and same-study weight 5.0; on the untouched folds it differed from the default network by 0.0006 AUROC (P=0.948).

![GNN depth, edge weighting, study-link weight and neighbourhood size (untouched folds).](results/stage6c/fig_gnn_depth_weights.png){width=100%}

### G. Ensembles (including the GNN)

Among 11 member combinations × 3 combiners, the best ensemble (cross_family(pubmed+hybrid+synthetic), mean) reached AUROC 0.657, a difference of 0.0010 from the best single model (q=1.000).

### H. Synthetic digital twins (factorial grid with local refinement)

In the XGBoost grid (5,310 fits), no hybrid cell improved AUROC over PubMed-only training (0 of 162 cells with a confidence interval above zero); the best hybrid changed AUROC by 0.0009 (95% CI, -0.0004 to 0.0022), synthetic-only training with twins fitted to real data lost 0.010, and twins built only from published priors reached AUROC values near 0.55. Adding twins beyond about 9 to 27 times the real training size reduced performance.

![Stage 2 XGBoost grid: effect of twin volume and synthetic share (ΔAUROC vs PubMed-only, 95% CI).](results/stage2/fig_volume_share_response.png){width=100%}

![Stage 2 heatmap: synthetic share × real data held out (9× volume).](results/stage2/fig_heatmap_share_heldout_V9.png){width=100%}

Across the full factorial grid (45,360 configurations), 4452 hybrid or synthetic configurations improved discrimination over PubMed-only training after Benjamini-Hochberg correction.

![Full factorial: learning curves by family and model.](results/stage3/fig_learning_curves_families.png){width=100%}

![Full factorial: main effects of generator set, prior source, volume and share.](results/stage3/fig_main_effects.png){width=100%}

### I. Model summary

Table: Model performance by family.

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

### J. Data modalities and multimodal feasibility

Table: Data modalities available for modelling and their use.

| Modality                                   | Available    | Amount                                 | Use / feasibility                                          |
|:-------------------------------------------|:-------------|:---------------------------------------|:-----------------------------------------------------------|
| Structured tabular (categorical + numeric) | Yes          | 3.27 million reconstructed patients    | All supervised, graph, CNN-1D and twin models              |
| Free text (abstracts, full texts)          | Yes          | 169 records; 60 full texts             | Extraction only; too few labelled patients for text models |
| Images (fundus, OCT, visual fields, MRI)   | Figures only | about 6 patients with pre/post figures | Not trainable; 2D/3D CNN + Grad-CAM pipeline prepared      |
| 3D volumes (OCT cubes)                     | No           | 0                                      | Not available                                              |
| Video / time series                        | No           | 0                                      | Not available                                              |

![Background-risk model trained on non-GLP patients: internal vs external (GLP-1RA users) AUROC and O/E.](results/stage7/fig_stage7_external.png){width=100%}

![Fine-tuning experiments: AUROC in held-out GLP-1RA users by update method and fine-tuning data source.](results/stage8/fig_finetune.png){width=100%}

### K2. Reporting checklist

Table: TRIPOD+AI items.

| TRIPOD+AI domain       | Where addressed                                                                                                    |
|:-----------------------|:-------------------------------------------------------------------------------------------------------------------|
| Title/abstract         | Identifies prediction-model development with ML on published aggregate evidence; structured abstract               |
| Background/objectives  | Introduction: intended use = research risk stratification; future use = antiplatelet selection (not yet supported) |
| Data source            | Methods: PubMed search, extraction schema, pseudo-IPD reconstruction                                               |
| Participants           | Reconstructed cohorts table: 11 non-overlapping cohorts; eligibility = exposed/comparator counts for both arms                         |
| Outcome                | NAION (or ischemic optic neuropathy where only ION reported; flagged as a feature)                                 |
| Predictors             | Therapy, exposure, indication, region, data source, follow-up, study mean age, % female                            |
| Sample size            | 3,271,499 reconstructed patients, 1,425 events                                                                     |
| Missing data           | Study-level medians with missingness indicators                                                                    |
| Analysis/models        | LR, RF, XGBoost, GNN (GraphSAGE/GAT), ensembles; synthetic twins                                                   |
| Validation             | 10/90 hold-out, 10 disjoint 10% folds, training-fraction sweep, leave-one-study-out, nested GNN tuning             |
| Performance measures   | AUROC, PR-AUC, Brier, calibration slope, O/E, sensitivity, specificity, accuracy, PPV, NPV, net benefit            |
| Explainability         | SHAP with permutation tests; integrated gradients for GNN                                                          |
| Fairness               | Sex and age examined at study level only; patient-level fairness not assessable                                    |
| Limitations            | Ecological features; no patient-level covariates; no imaging; not prospectively validated                          |
| Code/data availability | Extraction files, code and results deposited with the preprint                                                     |

### K. Threshold metrics: advantages and limitations of the best model

(train 10% / test 90%), mean ± SD, unless stated.

| Metric | Best model (XGBoost) | vs chance | Reads as |
|---|---|---|---|
| AUROC | 0.657 (hold-out; 95% CI ≈0.64–0.67); 0.657 ± 0.011 over folds | p < 1e-70 vs 0.5 | Advantage: signal is real and reproducible. Limitation: modest; leave-one-study-out falls to 0.535 (p = 0.02 vs 0.5), so most signal is setting, not patient |
| Sensitivity | 0.59 ± 0.10 (Youden); 0.35 (top 10%); 0.44 (top 20%) | all p < 1e-40 vs flag rate | Advantage: flags a 1.5–2× larger share of events than random. Limitation: misses 40–65% of events, so it cannot serve as a rule-out screen |
| Specificity | 0.61 (Youden); 0.82 (top 10%); 0.75 (top 20%) | p ≈ 0.36–0.42 vs 1 − flag rate | Limitation: no better than chance at the same flag rate. With 99.96% non-events, specificity mostly reflects how many people are flagged |
| Accuracy | 0.61–0.85 | below the 0.9996 no-information rate (p = 1) | Limitation by construction: with 0.044% prevalence, calling everyone "no NAION" is 99.96% accurate. Report it but don't use it for model choice |
| PPV | 0.07% (Youden) – 0.09% (top 10%); prevalence 0.044% | p < 1e-28 vs prevalence | Advantage: about 2× enrichment (LR+ 1.5–2.2). Limitation: absolute PPV < 0.1%, so about 1 in 1,000–1,500 flagged patients develops NAION. Any intervention given to all flagged patients exposes about 1,000 patients per event |
| NPV | 99.97% | p < 1e-12 vs 1 − prevalence | Mostly reflects the low prevalence (baseline 99.956%). Advantage only for reassurance messaging; not evidence of safety |
| Net benefit | > treat-all and > treat-none at the trained thresholds | — | Advantage only at threshold probabilities ≈ 0.03–0.1%, i.e. for near-harmless actions (dilated exam / OCT of the disc, counselling, slower dose escalation). Not for antiplatelet therapy, whose bleeding risk implies a far higher treatment threshold |

## Model-vs-model
- k-fold corrected resampled t-test: no metric differs between XGBoost, RF, LR or the mean ensemble (all q ≥ 0.99).
- McNemar on the hold-out gives p ≈ 0. That reflects millions of patients and the differing flag rates at each threshold, not better ranking. Interpret only alongside the matched-flag-rate comparison.
- Predictions take few distinct values, because the reconstructed data have about 48 covariate patterns. So percentile thresholds are coarse: the realised flag rate for "top 10%" was 14–18%. This is a key limitation; patient-level features would fix it.

## Implication for the intended use (antiplatelet selection)
- At PPV < 0.1%, an antiplatelet given to everyone flagged would treat about 1,000 people per NAION event that occurs, before any treatment effect is considered. Antiplatelet benefit for NAION is unproven.
- The score in its current form supports research enrichment (choosing whom to examine or enrol), not treatment selection. Prospective, patient-level validation plus a risk-stratified trial are required.


## Discussion

Published evidence on GLP-1RAs and NAION contains a consistent exposure signal: within the same studies, exposed patients had roughly 1.6 times the odds of NAION, and the estimate was higher for semaglutide and for obesity indications. Machine learning on the same evidence, however, discriminated only modestly, and most of what it learned was where a patient was treated rather than who the patient was. Discrimination collapsed toward chance in studies the model had not seen, and the strongest attributions went to data source and region. More flexible learners — graph neural networks, ensembles and synthetic digital twins — did not change this picture.

The threshold metrics show what this means in practice. Sensitivity and PPV beat chance, so the score enriches for cases roughly twofold, which is an advantage for selecting patients for closer surveillance or for enrolment in research. Absolute PPV, however, stayed below 0.1%: about one in 1,100 flagged patients would develop NAION. Specificity added nothing beyond the flag rate, accuracy was uninformative because the outcome is rare, and the high NPV mainly reflected the low prevalence. Net benefit was positive only at threshold probabilities suited to low-risk actions such as a dilated examination or optical coherence tomography before treatment, counselling, or slower dose escalation, consistent with recent consensus recommendations [@p42132708; @p41058255].

The intended clinical question is whether a validated score can identify patients who should start antiplatelet therapy. These data cannot answer it. Risk prediction is not treatment-effect prediction; none of the extracted cohorts or case reports recorded antiplatelet exposure, NAION is not primarily thromboembolic, and the evidence for aspirin after NAION is limited. At the observed PPV, treating every flagged patient would expose roughly a thousand people to bleeding risk per event that occurs, before any benefit is assumed. Answering the question requires prospective validation with patient-level predictors (disc morphology, sleep apnoea, nocturnal hypotension, dose escalation) followed by a risk-stratified trial with bleeding as a co-primary safety outcome.

Synthetic twins helped only weak learners. In the full factorial experiment, thousands of hybrid configurations improved logistic regression, consistent with regularization, but the best hybrid XGBoost gained only 0.0015 AUROC; twins built from published priors alone produced little more than chance discrimination, and fine-tuning a background model worked only with real GLP-1RA data. Twins cannot create patient-level information that the source data do not contain.

The limiting factor is the feature set, not the learner. In simulation, adding cup-to-disc ratio from fundus photographs or OCT, with published effect sizes, raised AUROC from about 0.65 to 0.77 to 0.82, and plausible additional imaging reached about 0.84; even then positive predictive value stayed below 1%. A prospective cohort of GLP-1RA initiators with baseline optic-disc imaging, which no published study has reported, is therefore the most informative next step. The entire analysis used about 0.25 kWh of local compute, so the environmental cost of the modelling itself is negligible relative to the data collection it motivates.

We describe this work as a foundation machine learning model in the sense of a base layer: it fixes the outcome definition, the reconstructed evidence base, the validation protocol and a benchmark (AUROC about 0.66 within settings, about 0.53 across settings) against which every later module must be judged. It is not a large pretrained model. The intended next layers are encoders for optic-disc photography and OCT and separately pretrained components such as an oculomotor world model, a learned generative model of eye-movement dynamics adapted from Ha and Schmidhuber [@ha2018], trained on synthetic eye-movement data and sharing no variables with the present data, combined in a multimodal neuro-ophthalmic foundation model. No transfer between components was performed or evaluated here; it requires patients with linked risk factors, imaging and oculomotor recordings.

This study has limitations. Predictors were arm- or study-level, so associations with age, sex and setting are ecological. Reconstructed patients share covariate patterns, which makes thresholds coarse. Outcome definitions differed (one cohort reported ischemic optic neuropathy), TriNetX cells below 10 were masked, several studies were available only as abstracts, and overlapping registries were handled by selection rather than by individual linkage. No imaging was available, so multimodal and computer-vision models could not be trained. All results are retrospective and in silico.

In conclusion, ML on published evidence confirms a modest, reproducible association between GLP-1RA exposure and NAION and yields a research-grade enrichment score, not a treatment-selection tool. Patient-level prospective data are the necessary next step.

**Use of AI tools.** Literature retrieval, data extraction, analysis code, figures and the first draft of this manuscript were produced with an AI assistant (Claude, Anthropic) working under the direction of the authors. Every extracted value carries a source PMID and evidence quote. The authors are responsible for verifying all content before submission. AI tools are not authors.

## Data and code availability

Code, trained model weights, extraction files (every value with PMID and evidence quotation), reconstructed count-weighted patient-level data and synthetic digital-twin datasets are available at https://github.com/aeyemovment/naion-glp1-risk (release v1.0.0). A research-only web calculator is at https://naion-glp1-calculator.vercel.app. Copyright (c) 2026 NeuroAgent AI, Inc. All materials are released for non-commercial research use only, with attribution (citation of this article): code and model weights under a research-only, non-commercial licence included in the repository, data under CC BY-NC 4.0. Commercial use, and clinical, diagnostic or treatment-selection use, are not permitted without prior written permission (kemarearlgreen@neuroagentai.org).

## References
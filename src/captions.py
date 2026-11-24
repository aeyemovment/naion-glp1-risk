"""Full figure legends and table captions (what is shown, how to read it, keys, abbreviations) for the preprint / IEEE manuscript.
Keys: figure image file name, or the opening words of the existing table caption. research_prototype=true.
"""
import re

NOTE = " Remaining abbreviations and variable names: Nomenclature appendix."

FIG = {
    "fig1_study_flow.png":
        "Study flow and validation design. Boxes show the PubMed search, screening of PubMed Central full texts, structured extraction by "
        "publication type, reconstruction of count-weighted pseudo-individual patient data from published 2x2 counts, and the validation "
        "schemes (10% train / 90% test hold-out, ten disjoint 10% training folds, training fractions of 10% to 90%, leave-one-study-out) "
        "applied to all models.",
    "fig_holdout_roc_pr_cal.png":
        "Discrimination and calibration of PubMed-only models on the 90% hold-out (trained on 10%). (A) Receiver operating characteristic "
        "curves; AUROC in the legend; dashed diagonal = chance. (B) Precision-recall curves; precision is bounded by the NAION prevalence "
        "(0.044%), so curves fall close to zero. (C) Calibration by decile of predicted risk: predicted versus observed NAION risk per 1,000 "
        "patients; dashed line = perfect calibration. Blue, logistic regression (L2); orange, random forest; green, XGBoost.",
    "fig_kfold.png":
        "Performance across ten disjoint 10% training folds (each tested on the remaining 90%). (A) AUROC and (B) PR-AUC; each dot is one fold "
        "and the horizontal bar is the mean. Blue, logistic regression (L2); orange, random forest; green, XGBoost. PR-AUC values are small "
        "because the event rate is 0.044%.",
    "fig_train_fraction_curves.png":
        "Training-fraction sweep: models trained on 10% to 90% of patients and tested on the remainder (10 repeats per fraction; points = "
        "mean, bars = SD). (A) AUROC on the held-out remainder. (B) Calibration-in-the-large as observed/expected (O/E) events; dashed line "
        "= 1 (perfect). (C) Stability of explanations: Kendall's W agreement of SHAP feature rankings across repeats (1 = identical "
        "rankings). Blue, logistic regression (L2); orange, random forest; green, XGBoost.",
    "fig_shap_beeswarm_xgb.png":
        "SHAP summary (beeswarm) for XGBoost on the 90% hold-out. Each dot is one covariate pattern; the x-axis is its SHAP value (contribution "
        "to the predicted log-odds of NAION; right = higher risk); colour is the feature value (red = high or 1, blue = low or 0). Features "
        "are ordered by mean absolute SHAP value; one-hot features are named variable_level (e.g., therapy_sglt2i = SGLT2-inhibitor "
        "comparator arm; source_type_single_center = single-center data source).",
    "fig_shap_kfold_importance.png":
        "Feature-group importance across ten folds: mean absolute SHAP value (log-odds scale; bars = mean, whiskers = SD over folds), with "
        "one-hot levels summed into their parent variable (therapy = drug arm; region; source_type = data source; outcome_def = outcome "
        "definition). Blue, logistic regression (L2); green, XGBoost; random-forest SHAP values are on the probability scale "
        "(about 0.0001) and therefore not shown.",
    "fig_forest_within_study.png":
        "Within-study odds ratios for NAION (count-weighted logistic regression with study fixed effects, so each estimate compares patients "
        "within the same study). Points = odds ratio, lines = 95% CI, log scale; dashed line = no association. Blue, q < 0.05; grey, not "
        "significant; q = Benjamini-Hochberg adjusted P value. Values at right: OR (95% CI) and q.",
    "fig_forest.png":
        "Most precise published ratio estimate per study (smallest standard error), as reported by the authors (hazard, odds, risk or incidence "
        "rate ratios; not directly interchangeable). Points = estimate (size proportional to precision), lines = 95% CI, log scale; dashed "
        "line = 1. Labels: first author, year (population). T2D, type 2 diabetes; IIH, idiopathic intracranial hypertension.",
    "fig_case_phenotypes_pca.png":
        "Case-report phenotypes: NAION case patients from case reports and series projected onto the first two principal components (PC1, PC2; "
        "percentage of variance explained in parentheses) of standardized clinical features. Colours = k-means clusters at the best "
        "silhouette solution (k = 2); one cluster contains a single tirzepatide-treated patient, so no stable multi-patient phenotype was found.",
    "fig_meta_shap.png":
        "SHAP summary for the random-forest meta-regression of 148 published log ratio estimates from 25 studies. Each dot is one published "
        "estimate; the x-axis is its SHAP value on the log-ratio scale; colour is the feature value (red = 1 or high, blue = 0 or low). "
        "Features describe the publication: dataset_ (data source, e.g., danish = Danish registries, trinetx = TriNetX network, medicare = "
        "Medicare claims), design_ (cohort or target-trial emulation), comparator_cls_ (comparator class), exposure_drug_ (exposure drug), "
        "followup (years) and followup_missing (not reported).",
    "fig_gnn_depth_weights.png":
        "Graph neural network tuning, evaluated on untouched folds 5 to 9 (AUROC). (A) Network depth (1 to 6 message-passing layers) by edge "
        "weighting: uniform = all k-nearest-neighbour edges weight 1; gauss0.5, gauss1, gauss2 = Gaussian kernel of feature distance with "
        "bandwidth 0.5, 1 or 2; dashed = graph attention network (GAT) with learned weights. (B) Weight of same-study edges as a multiple of "
        "a k-nearest-neighbour edge. (C) Number of neighbours per node (k). Points = mean; bars = SD across folds. All differences are "
        "smaller than 0.002 AUROC.",
    "fig_volume_share_response.png":
        "Synthetic-data grid (Stage 2, XGBoost): change in AUROC versus PubMed-only training on the same split (mean and 95% CI over grid cells "
        "and seeds). Left, by synthetic volume (multiples of the real training size); right, by synthetic share of the training set. "
        "Blue, marginal generator; orange, Gaussian-copula generator; green, structural (logistic) generator. Dashed line = no change; "
        "values below zero mean synthetic data reduced discrimination.",
    "fig_heatmap_share_heldout_V9.png":
        "Synthetic-data grid at 9x volume: change in AUROC versus PubMed-only training (colour; red = worse, blue = better) by synthetic share "
        "of the training set (columns) and percentage of real PubMed data held out for testing (rows), for each generator. Training sets made "
        "entirely of synthetic data (100% share) lose the most.",
    "fig_learning_curves_families.png":
        "Full factorial experiment: AUROC on held-out real PubMed data versus the real training fraction, for each model. Black, PubMed-only "
        "training; orange, best hybrid (real plus synthetic) configuration; blue, best synthetic-only configuration; both selected as the "
        "best of the grid and therefore optimistic. Purple diamond at 0% real data: synthetic-only training from published priors alone.",
    "fig_main_effects.png":
        "Full factorial experiment: main effect of each design factor on AUROC relative to PubMed-only training (mean change and 95% CI, "
        "averaged over all other factors), by model (blue, logistic regression; orange, random forest; green, XGBoost; purple, mean "
        "ensemble). Generator set: Marg = marginal, Cop = Gaussian copula, Struct = structural (logistic), joined by + when combined. Prior "
        "source: real = fitted to the real training split; prior = published estimates only; blend = 50/50. Volume = synthetic records as "
        "a multiple of the real training size. Synthetic share = percentage of the training set that is synthetic. Dashed line = no change.",
    "fig_stage7_external.png":
        "Background-risk models trained only on non-users and applied to GLP-1 receptor agonist users. (A) AUROC per fold: blue, internal "
        "validation in held-out non-users; orange, external validation in GLP-1RA users; dashed line = chance. (B) Observed/expected (O/E) "
        "NAION events in GLP-1RA users (mean and 95% CI); values above the dashed line at 1 mean the non-user model under-predicts risk in "
        "users. LR, logistic regression; RF, random forest; XGB, XGBoost; GNN, graph neural network; Ens., mean ensemble.",
    "fig_finetune.png":
        "Fine-tuning a non-user model for GLP-1RA users: AUROC in held-out GLP-1RA users versus the percentage of GLP-1RA users used for "
        "fine-tuning, by fine-tuning data source (real, synthetic or hybrid). Lines = update methods (legend below the panels). Recalibration "
        "(intercept only, or intercept and slope) is a monotone transformation of the risk score, so it cannot change AUROC; both recalibration "
        "lines therefore lie on the dashed line (pretrained non-user model without update). GLP-1RA users only = model trained on GLP-1RA users "
        "without pretraining.",
    "fig_subgroups.png":
        "Detection within GLP-1RA users (orange) and within non-users (blue): (left) AUROC and (right) PPV at the threshold flagging the top "
        "10% of training risk (proportion; 0.001 = 0.1%), mean ± SD over ten folds; dashed line = AUROC 0.5. LR, logistic regression; RF, "
        "random forest; XGB, XGBoost; GNN, graph neural network; CNN, one-dimensional convolutional network; Ens., mean ensemble; Poisson, "
        "Poisson regression with person-time offset; IForest, isolation forest; AE, autoencoder (both anomaly detectors trained on non-events).",
    "fig_incidence_strata.png":
        "NAION incidence per 100,000 person-years (log scale; points = pooled rate, bars = exact Poisson 95% CI) in GLP-1RA users (orange) and "
        "comparators (blue), by (A) study mean age band, (B) indication (IIH = idiopathic intracranial hypertension) "
        "and (C) drug of the exposed arm (GLP-1RA class = studies pooling several GLP-1RAs). Grey band = published "
        "background incidence in adults aged 50 years or older.",
    "fig_feature_combinations.png":
        "Feature-group contributions across all 511 combinations of nine feature groups (XGBoost). Left: grey = stand-alone gain (AUROC - 0.5 "
        "of a model using only that group); blue = Shapley value (average gain in AUROC when the group is added, over all combinations). "
        "Right: pairwise Shapley interaction indices (AUROC); red = redundancy (the pair adds less together than separately), blue = synergy. "
        "Groups: exposure = GLP-1RA exposure; drug = therapy arm; indication; age = study mean age (with partial ABCD2 age item); sex = share "
        "of women; duration = follow-up; hypertension and diabetes = partial ABCD2 items; setting = region, data source and outcome definition.",
    "fig_imaging_whatif.png":
        "Simulated effect of adding ocular imaging and obstructive sleep apnoea (OSA) to the published-feature XGBoost model (AUROC; values at "
        "bar ends). Grey = base model; blue = with imaging features; orange = with OSA (non-imaging comparator). CDR = cup-to-disc ratio from "
        "fundus photographs or optical coherence tomography (OCT); drusen = optic disc drusen; OCTA = OCT angiography (peripapillary "
        "perfusion); serial = on-treatment change on repeat imaging. CDR and OSA effects use published odds ratios; drusen, OCTA and serial "
        "effects are hypothetical, and black bars show the range across low and high assumed effect sizes. All imaging values are synthetic.",
}

TAB = {
    "Sensitivity analysis restricted to cohorts reporting NAION risk-factor prevalence":
        "Sensitivity analysis restricted to cohorts reporting NAION risk-factor prevalence at study level (hypertension; hypertension and obstructive "
        "sleep apnea; 1:1 or 1:2 propensity-score matching). OR = within-study odds ratio for GLP-1RA exposure (count-weighted logistic regression with "
        "study fixed effects) with 95% CI; MH OR = Mantel-Haenszel odds ratio; I2 = between-study heterogeneity; XGBoost AUROC = mean over ten "
        "disjoint 10% training folds.",
    "NAION incidence per 100,000 person-years":
        "NAION incidence per 100,000 person-years (rate) in GLP-1RA users and comparators by stratum. Events = NAION events; IRR = incidence "
        "rate ratio (GLP-1RA / comparator) with exact conditional binomial 95% CI; MH IRR = Mantel-Haenszel IRR pooled within studies; q = "
        "Benjamini-Hochberg adjusted P value. Rates are per 100,000 person-years; Study mean age, "
        "Women in study and Follow-up are study-level bands; T2D, type 2 diabetes; IIH, idiopathic intracranial hypertension.",
    "Model performance within GLP-1RA users":
        "Model performance within GLP-1RA users and within non-users (mean over ten folds; threshold = top 10% of training risk). PR-AUC = area "
        "under the precision-recall curve; O/E = observed/expected events; Sens, sensitivity; Spec, specificity; Acc, accuracy; PPV and NPV = "
        "positive and negative predictive values (proportions). LR, logistic regression; RF, random forest; XGB, XGBoost; GNN, graph neural "
        "network; CNN, one-dimensional convolutional network; Ens., mean ensemble; Poisson, Poisson regression with person-time offset; IForest, "
        "isolation forest; AE, autoencoder; --- = not applicable (anomaly scores are not calibrated probabilities).",
    "AUROC with and without partial ABCD2":
        "AUROC with and without partial ABCD2 features (study-level age, hypertension and diabetes items) in every experiment, on identical "
        "splits. Difference = with minus without; q = Benjamini-Hochberg adjusted P value of the paired test. Experiment names refer to the "
        "Stages in Methods. Model or configuration: the model, ensemble (members joined by +; cross-family = PubMed-only, hybrid and "
        "synthetic-only members), or, for Stage 8, the update method followed by the pretraining data, fine-tuning data and share of GLP-1RA "
        "patients used for fine-tuning. LR, logistic regression; RF, random forest; XGB, XGBoost; GNN, graph neural network; CNN, "
        "one-dimensional convolutional network; Ens., mean ensemble; AE, autoencoder; IForest, isolation forest.",
    "Best feature-group combinations by number of groups":
        "Best feature-group combinations for each number of groups (two best per size; mean over folds; threshold = top 10% of training "
        "risk). SD = standard deviation of AUROC over folds; q (AUROC>0.5) and q (PPV>prev) = Benjamini-Hochberg adjusted P values for AUROC "
        "above chance and PPV above prevalence, over all 511 combinations and models; Sens, Spec, Acc, PPV, NPV as proportions. Feature groups are "
        "joined by +; LR, logistic regression; RF, random forest; XGB, XGBoost; GNN, graph neural network; CNN, one-dimensional convolutional network. Full results "
        "for every combination and model are in Supplementary Workbook S1.",
    "Odds-ratio reliability of each feature":
        "Reliability of each feature's odds ratio (OR) across all combinations containing it (count-weighted logistic regression with "
        "cluster-robust standard errors by study). Combinations = number of feature combinations containing the feature; Median/Min/Max OR across them; OR>1 = share of "
        "combinations with OR above 1; q<0.05 = share significant after Benjamini-Hochberg correction. Study-level covariates are ecological "
        "and imprecise with 11 clusters.",
    "Data sources by type and modality":
        "Data sources by type and modality. Structured tabular data (categorical and numeric) were the only modality usable for modelling; "
        "imaging existed only as figures in case reports and no video was located. Records = publications or extracted rows; Patients = "
        "reconstructed or reported patients; Used in = analyses that used the source.",
    "Reconstructed cohorts.":
        "Reconstructed cohorts (one primary, non-overlapping contrast per study). Setting = data source; Population: T2D, type 2 diabetes; IIH, "
        "idiopathic intracranial hypertension; events/N = NAION events and patients in each arm.",
    "Training-fraction sweep (mean over 10 repeats).":
        "Training-fraction sweep (mean over 10 repeats). Training fraction = share of patients used for training; AUROC and AUROC SD = mean and SD "
        "of AUROC on the remainder; PR-AUC = area under the precision-recall curve; Brier = Brier score; Cal. slope = calibration slope (1 = ideal); "
        "O/E and O/E SD = observed/expected events (1 = ideal). LR, L2 logistic regression; RF, random forest; XGB, XGBoost.",
    "Within-study (study fixed effects) associations.":
        "Within-study associations with NAION (count-weighted logistic regression with study fixed effects). Specification = the "
        "regression model in which the term was estimated. Terms: exposed_ = GLP-1RA "
        "exposure in the named drug or indication subgroup; exposed_vs_ = versus the named comparator; exposed_x_ = interaction with the "
        "named study-level covariate (per 10 years of mean age, per 10 percentage points of women, per year of follow-up). OR = odds ratio; "
        "q = Benjamini-Hochberg adjusted P value.",
    "Marginal associations":
        "Marginal associations with NAION across studies (count-weighted logistic regression with cluster-robust standard errors by study). "
        "These comparisons are ecological and confounded by study setting. Feature names: therapy_, indication_, region_, source_type_, "
        "outcome_def_ = one-hot levels (reference levels omitted); OR = odds ratio; q = Benjamini-Hochberg adjusted P value.",
    "Optional calculator factors":
        "Optional calculator factors: published odds ratios and forward-selection results (calculator version 2; simulated gains, not "
        "validated). Reference level = median prevalence across the training cohorts (cup-to-disc ratio: share of control eyes in the source "
        "study); OR = odds ratio (sleep apnoea: pooled risk ratio used as OR; cup-to-disc ratio relative to a ratio above 0.40); AUROC gain "
        "alone = gain over the core ensemble when the factor is added by itself; marginal AUROC gain = gain at its forward-selection step; "
        "factors were kept if the marginal gain was at least 0.005 with Benjamini-Hochberg q < 0.05.",
    "SHAP permutation tests":
        "SHAP permutation tests (B = 500 permutations). Mean abs. SHAP = mean absolute SHAP value (XGBoost, log-odds); P (global null) = outcome "
        "permuted across all patients; P (within-study null) = outcome permuted within studies, which tests whether the feature carries "
        "information beyond study membership; q = Benjamini-Hochberg adjusted P value.",
    "Random-effects meta-analysis of published estimates.":
        "Random-effects (DerSimonian-Laird) meta-analysis of published ratio estimates. k = number of estimates; I2 = percentage of "
        "variance due to between-study heterogeneity; q = Benjamini-Hochberg adjusted P value.",
    "Demographic contrasts.":
        "Demographic contrasts: observed versus expected values (e.g., share of men among case reports versus exposed cohorts) with the "
        "stated test; q = Benjamini-Hochberg adjusted P value.",
    "Tests of ML experiments.":
        "Statistical tests of the machine-learning experiments. Estimate = the tested quantity (AUROC, difference or rank correlation); "
        "Hanley-McNeil = AUROC standard error; Nadeau-Bengio corrected = resampled t test corrected for overlapping training sets; Wilcoxon "
        "and Spearman as stated; q = Benjamini-Hochberg adjusted P value.",
    "Sensitivity, specificity, accuracy, PPV and NPV":
        "Threshold metrics on the 90% hold-out with P values against chance. Thresholds: Youden = maximizes sensitivity + specificity - 1; "
        "Top 10% / Top 20% = flags the top 10% or 20% of training risk. Sens, sensitivity; Spec, specificity; Acc, accuracy; P (Sens) and P (Spec) = "
        "versus chance; P vs NIR = accuracy versus the no-information rate; P (NPV) = NPV versus 1 - prevalence; PPV versus prevalence (P vs prev); NPV, negative predictive value; LR+, positive likelihood "
        "ratio; NB, net benefit at the threshold probability; NB treat-all, net benefit of treating everyone.",
    "Threshold metrics across ten folds (mean).":
        "Threshold metrics across ten folds (mean). Thresholds: Youden = maximizes Youden's J; Top 10% and Top 20% = flag the top 10% "
        "or 20% of training risk. All values are means over folds; Sens, sensitivity; Spec, specificity; Acc, accuracy; LR+, positive likelihood ratio; "
        "Flagged = share of patients flagged. LR, logistic regression; RF, random forest; XGB, XGBoost; Ensemble = mean of members.",
    "Model performance by family.":
        "Model performance by family. AUROC 10/90 = trained on 10%, tested on 90%; AUROC 10x10% folds = mean ± SD over ten disjoint folds; "
        "AUROC LOSO = leave-one-study-out; O/E = observed/expected events; Sens/Spec and PPV/NPV at the top-10% threshold; hybrid = real plus "
        "synthetic training data; synthetic only = no real training data; priors = published estimates used to generate synthetic data.",
    "Data modalities available for modelling and their use.":
        "Data modalities available for modelling and their use. CNN, convolutional neural network; Grad-CAM, gradient-weighted class "
        "activation mapping; OCT, optical coherence tomography.",
    "TRIPOD+AI items.":
        "TRIPOD+AI reporting items (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis, AI "
        "extension) and where each is addressed.",
}


def expand(md):
    def fig(m):
        cap, path = m.group(1), m.group(2)
        new = FIG.get(path.split("/")[-1])
        return f"![{(new or cap.rstrip()) + NOTE}]({path})"
    md = re.sub(r"^!\[([^\]]*)\]\(([^)]*)\)", fig, md, flags=re.M)

    def tab(m):
        cap, rest = m.group(1), m.group(2) or ""
        if cap.startswith(("Abbreviations.", "Variable and label names")): return m.group(0)
        new = next((v for k, v in TAB.items() if cap.startswith(k)), cap.rstrip())
        return "Table: " + new + NOTE + rest
    return re.sub(r"^Table: (.*?)(\s*\{#[^}]*\}|\\label\{[^}]*\})?$", tab, md, flags=re.M)

"""Nomenclature for the manuscripts: abbreviations and the variable / label names that appear in tables and figures.
check(md) lists any abbreviation or snake_case label used in a manuscript that is not defined here.
research_prototype=true; compliance_ref=Addendum_5_DT9.
"""
import re

ABBR = {
    "ABCD²": "Age, Blood pressure, Clinical features, Duration of symptoms and Diabetes score for stroke risk after transient ischemic attack",
    "ABCD³-I": "ABCD² extended with dual transient ischemic attack and Imaging findings",
    "Acc": "accuracy", "AE": "autoencoder (anomaly detector)", "CDR": "cup-to-disc ratio", "Ens.": "ensemble (mean of member models)",
    "IForest": "isolation forest (anomaly detector)", "OSA": "obstructive sleep apnea", "PC": "principal component", "PY": "person-years",
    "Sens": "sensitivity", "Spec": "specificity", "B": "number of permutations", "k": "number of estimates (meta-analysis) or neighbours per node (graph)",
    "AI": "artificial intelligence", "AION": "anterior ischemic optic neuropathy",
    "AUROC": "area under the receiver operating characteristic curve", "BP": "blood pressure",
    "CI": "confidence interval", "CNN": "convolutional neural network (CNN-1D, one-dimensional)",
    "CO2e": "carbon dioxide equivalent", "CV": "cross-validation", "DL": "deep learning",
    "DPP-4i": "dipeptidyl peptidase-4 inhibitor", "EBGM": "empirical Bayes geometric mean (pharmacovigilance disproportionality)",
    "EHR": "electronic health record", "FAERS": "US Food and Drug Administration Adverse Event Reporting System",
    "FE": "fixed effects", "GAT": "graph attention network", "GB": "gigabyte",
    "GCIPL": "ganglion cell-inner plexiform layer", "GIP": "glucose-dependent insulinotropic polypeptide",
    "GLP-1": "glucagon-like peptide-1", "GLP-1RA": "glucagon-like peptide-1 receptor agonist",
    "GNN": "graph neural network", "Grad-CAM": "gradient-weighted class activation mapping",
    "GraphSAGE": "graph sample-and-aggregate network", "I²": "percentage of between-study variance due to heterogeneity",
    "IC025": "lower 95% credibility bound of the information component (pharmacovigilance)",
    "IIH": "idiopathic intracranial hypertension", "ION": "ischemic optic neuropathy",
    "IONDT": "Ischemic Optic Neuropathy Decompression Trial", "IPD": "individual patient data",
    "IRR": "incidence rate ratio", "L2": "ridge (squared-coefficient) penalty", "LOSO": "leave-one-study-out validation",
    "LR": "logistic regression", "LR+": "positive likelihood ratio", "MH": "Mantel-Haenszel", "ML": "machine learning",
    "MRI": "magnetic resonance imaging", "NAION": "nonarteritic anterior ischemic optic neuropathy", "NB": "net benefit",
    "NIR": "no-information rate", "NPV": "negative predictive value", "O/E": "observed-to-expected event ratio",
    "OCT": "optical coherence tomography", "OCTA": "optical coherence tomography angiography", "OR": "odds ratio",
    "PCA": "principal component analysis", "PMC": "PubMed Central", "PMID": "PubMed identifier",
    "PPV": "positive predictive value", "PR-AUC": "area under the precision-recall curve",
    "PRR": "proportional reporting ratio", "PUE": "power usage effectiveness", "q": "Benjamini-Hochberg adjusted P value",
    "QPI-1007": "intravitreal small interfering RNA against caspase-2 (investigational)", "RCT": "randomized controlled trial",
    "RF": "random forest", "RNFL": "retinal nerve fiber layer", "ROC": "receiver operating characteristic",
    "ROR": "reporting odds ratio", "SD": "standard deviation", "SE": "standard error",
    "SGLT2i": "sodium-glucose cotransporter-2 inhibitor", "SHAP": "Shapley additive explanations",
    "SUSTAIN-6": "Trial to Evaluate Cardiovascular and Other Long-term Outcomes with Semaglutide in Subjects with Type 2 Diabetes",
    "SVM": "support vector machine", "MLP": "multilayer perceptron", "LOF": "local outlier factor", "T2D": "type 2 diabetes", "TIA": "transient ischemic attack",
    "TRIPOD+AI": "Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis, AI extension",
    "VA": "US Department of Veterans Affairs", "XGB": "XGBoost (extreme gradient boosting)",
}
# also accepted without a dictionary entry (plain English, units, places, section/file labels, degrees, chemical/brand codes)
ALLOW_CAPS = {"PC1", "PC2", "RNA", "DPP", "US", "USA", "UK", "DO", "FANA", "IV", "K2", "S1", "M3", "A2", "CC", "BY-NC", "CIs", "CNNs", "GLP-1RAs", "GLP", "SGLT2",
              "DPP-4", "CNN-1D", "SAGE", "CAM", "DT", "I2", "ABCD", "ABCD2", "ABCD3-I", "MD", "TRIPOD"}

# variable / label names (exact names, or prefixes ending in "*")
VARS = {
    # model inputs (one-hot encoded categorical features and numeric features)
    "therapy_*": "drug arm of the patient (one-hot): semaglutide; sema_or_tirz = semaglutide or tirzepatide; glp1ra_class = any or unspecified GLP-1RA; comparators sglt2i, dpp4i, no_glp1ra (no GLP-1RA), other_antidiabetic, other_antiobesity",
    "indication_*": "study population: t2d = type 2 diabetes, obesity = obesity without diabetes, iih = idiopathic intracranial hypertension",
    "region_*": "study region: us = United States, nordic = Nordic countries, uk = United Kingdom",
    "source_type_*": "data source: claims = insurance claims, ehr_network = multi-institution EHR network, ehr_primary_care = primary-care EHR, national_registry = national registry, single_center = single-center records, va_ehr = Veterans Affairs EHR",
    "outcome_def_*": "outcome definition: naion = NAION, ion = ischemic optic neuropathy (when NAION was not reported separately)",
    "source_type": "data source (see source_type_…)", "outcome_def": "outcome definition (see outcome_def_…)",
    "case_control": "case-control design", "target_trial": "target-trial emulation design", "variable_level": "naming pattern of one-hot features: variable name, underscore, level",
    "glp1ra_exposed": "1 if the patient was in a GLP-1RA arm, 0 if in a comparator arm", "followup_years": "mean study follow-up (years)",
    "mean_age": "mean age of the study arm (years)", "pct_female": "percentage of women in the study arm",
    "*_missing": "1 if the preceding variable was not reported and was replaced by the median across studies",
    # values that appear without the feature prefix
    "semaglutide": "semaglutide", "sema_or_tirz": "semaglutide or tirzepatide", "glp1ra_class": "any or unspecified GLP-1RA",
    "no_glp1ra": "no GLP-1RA (comparator)", "sglt2i": "SGLT2 inhibitor", "dpp4i": "DPP-4 inhibitor",
    "other_antidiabetic": "other glucose-lowering drug", "other_antiobesity": "other anti-obesity drug", "ehr_network": "multi-institution EHR network",
    "ehr_primary_care": "primary-care EHR", "national_registry": "national registry", "single_center": "single-center records",
    "va_ehr": "Veterans Affairs EHR", "claims": "insurance claims", "any_glp1ra": "any GLP-1RA", "t2d_or_obesity": "type 2 diabetes or obesity",
    # association terms in P-value tables
    "exposed_vs_*": "GLP-1RA exposure versus the named comparator (within-study contrast)",
    "exposed_x_*": "interaction of GLP-1RA exposure with the named study-level covariate",
    "exposed_*": "GLP-1RA exposure within the named drug or indication subgroup", "age_per_10y": "per 10-year higher mean age",
    "female_per_10pct": "per 10-percentage-point higher share of women", "age_band": "stratum of study mean age",
    "female_band": "stratum of the share of women", "fu_band": "stratum of follow-up duration",
    # meta-regression features (study-level published estimates)
    "dataset_*": "data source of a published estimate (e.g. danish = Danish registries, trinetx = TriNetX network, medicare = Medicare claims)",
    "design_*": "study design of a published estimate (cohort, target_trial = target-trial emulation, case_control, other)",
    "comparator_cls_*": "comparator class of a published estimate (sglt2, unexposed, other_antidiabetic, ...)",
    "exposure_drug_*": "exposure drug of a published estimate (semaglutide, any_glp1ra, tirzepatide, ...)",
    "followup": "reported follow-up of a published estimate (years)", "followup_missing": "1 if follow-up was not reported",
    "population": "study population of a published estimate", "adjusted_f": "1 if the estimate was covariate-adjusted",
    "abcd_*": "partial ABCD² component derived from study-level data: A = age, B = blood pressure (hypertension prevalence), D = diabetes",
    # models and model variants
    "logreg": "L2-penalized logistic regression", "rf": "random forest", "xgb": "XGBoost", "gnn": "graph neural network",
    "cnn1d": "one-dimensional convolutional neural network", "ensemble": "ensemble (mean of member predictions unless stated)",
    "cross_family": "ensemble combining PubMed-only, synthetic and hybrid models", "focal_mlp": "multilayer perceptron trained with focal loss",
    "poisson_offset": "Poisson regression with log person-time offset", "logreg_kingzeng": "logistic regression with King-Zeng rare-event correction",
    "xgb_standard": "XGBoost, unweighted", "xgb_balanced_raw": "XGBoost with class-balanced weights, uncorrected",
    "xgb_balanced_priorcorrected": "XGBoost with class-balanced weights, prior-corrected probabilities",
    "xgb_incidence_anchored": "XGBoost with predictions anchored to published incidence", "xgb_continue": "XGBoost fine-tuned by continued boosting",
    "isolation_forest": "isolation forest (anomaly detection)", "one_class_svm": "one-class support vector machine (anomaly detection)",
    "lof_novelty": "local outlier factor in novelty mode (anomaly detection)", "gnn_joint": "graph neural network trained jointly on both groups",
    "pooled_refit": "model refitted on pooled source and target data", "pretrain_only": "model trained on non-users only, applied unchanged",
    "recal_intercept": "intercept-only recalibration", "recal_logistic": "logistic (intercept and slope) recalibration",
    "glp_only": "model trained on GLP-1RA users only", "internal_nonGLP": "internal validation in non-users",
    "external_GLP": "external validation in GLP-1RA users",
    # metrics and table columns
    "*_mean": "mean across folds or repeats", "*_sd": "standard deviation across folds or repeats", "auroc": "AUROC",
    "pr_auc": "PR-AUC", "cal_slope": "calibration slope", "o_e": "observed-to-expected event ratio", "lr_pos": "positive likelihood ratio",
    "flagged_frac": "fraction of patients flagged as high risk", "train_frac": "fraction of patients used for training",
    "T1_youden": "threshold maximizing Youden's J (sensitivity + specificity - 1)", "T2_top10pct": "threshold flagging the top 10% of training risk",
    "T3_top20pct": "threshold flagging the top 20% of training risk",
    "brier": "Brier score (mean squared error of predicted risk)", "uniform": "graph edges all weighted 1",
    "gauss*": "graph edges weighted by a Gaussian kernel of feature distance with the stated bandwidth",
    "threshold": "decision threshold (see T1_youden, T2_top10pct, T3_top20pct)",
    # data-release flags
    "synthetic_only": "1 for computationally generated synthetic records", "research_prototype": "research-prototype flag (not for clinical use)",
}


def defined(tok):
    for k in VARS:
        if k == tok: return True
        if k.endswith("*") and tok.startswith(k[:-1]): return True
        if k.startswith("*") and tok.endswith(k[1:]): return True
    return False


def check(md):
    """Return (undefined snake_case labels, undefined abbreviations) used in the manuscript body (image paths ignored)."""
    body = re.sub(r"\]\([^)]*\)", "]()", md.split("## References")[0])
    body = re.sub(r"\[@[^\]]*\]", "", body)
    snake = sorted({t for t in re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b", body) if not defined(t)})
    keys = {k.replace("²", "2").replace("³", "3") for k in ABBR} | set(ABBR) | ALLOW_CAPS
    caps = sorted({t for t in re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?\b", body) if t not in keys})
    return snake, caps


def markdown():
    ab = "\n".join(f"| {k} | {v} |" for k, v in sorted(ABBR.items(), key=lambda kv: kv[0].lower()))
    va = "\n".join(f"| {k.replace('*', '…')} | {v} |" for k, v in VARS.items())
    return ("### 0. Nomenclature: abbreviations and variable names\n\n"
            "Abbreviations are also defined at first use in the text. Variable and label names are those printed in tables and figures; "
            "\"…\" stands for any suffix or prefix (for example, therapy_… covers therapy_semaglutide and therapy_sglt2i).\n\n"
            "Table: Abbreviations.\n\n| Abbreviation | Definition |\n|:--|:--|\n" + ab + "\n\n"
            "Table: Variable and label names used in tables and figures.\n\n| Name | Definition |\n|:--|:--|\n" + va)


CAPTION_NOTE = " Abbreviations and variable names are defined in the Nomenclature appendix."


def annotate_captions(md):
    """Append a pointer to the Nomenclature to every table and figure caption (except the Nomenclature tables themselves)."""
    md = re.sub(r"^(Table: (?!Abbreviations\.|Variable and label names)[^\n]*?)(\{#[^}]*\})?$",
                lambda m: m.group(1).rstrip() + CAPTION_NOTE + (" " + m.group(2) if m.group(2) else ""), md, flags=re.M)
    md = re.sub(r"^!\[([^\]]*)\]\(", lambda m: "![" + m.group(1).rstrip() + CAPTION_NOTE + "](", md, flags=re.M)
    return md

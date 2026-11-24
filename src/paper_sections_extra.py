"""Extra manuscript sections (Introduction from the literature agent's evidence file; extended Methods; Stage 7-9 results).
Imported by make_papers.py. research_prototype=true. NOT for clinical use."""
import os, json, re
import pandas as pd

J = lambda p: json.load(open(p)) if os.path.exists(p) else None
def f(x, d=3):
    try: return f"{float(x):.{d}f}"
    except Exception: return "NA"
def pq(p):
    try: p = float(p)
    except Exception: return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
PENDING = lambda what: f"*[Pending — {what} is still running; filled automatically on the next build.]*"


def intro_full():
    t = open("data/intro/intro_outline.md").read()
    body = t.split("# Introduction", 1)[1].strip()
    extra = ("Beyond conventional supervised learning, we examined whether graph neural networks that share information across similar patient "
             "profiles, ensembles, convolutional networks and synthetic digital twins improve on simpler models; whether a model of background "
             "NAION risk learned without any GLP-1RA exposure transports to GLP-1RA users; and whether fine-tuning such a model with real, synthetic "
             "or hybrid GLP-1RA data improves risk stratification. Every comparison was tested statistically with correction for multiple testing.")
    return "## Introduction\n\n" + body + "\n\n" + extra


def intro_nejm():
    return "\n\n".join(["## Introduction",
        "Nonarteritic anterior ischemic optic neuropathy (NAION) is the most common acute optic nerve disease of adults older than 50 years [@p9186104], "
        "with an annual incidence of roughly 2 to 10 per 100,000 in that age group [@p8032479; @p9186104]. Vision loss is usually permanent, and the fellow eye "
        "is affected in about 15 to 21% of patients over several years [@p12208242; @p38358828]. The leading anatomic risk factor is a small, crowded optic disc "
        "[@p6514298; @p3684223; @p38211825], and NAION is attributed to infarction of the laminar and retrolaminar optic nerve head, with nocturnal hypotension, "
        "impaired autoregulation and a compartment syndrome in a crowded disc among proposed contributors [@p19596112]. No treatment is proven [@p25763979; @p35270411]; "
        "decompression surgery was ineffective and possibly harmful [@p7844872], and evidence that aspirin protects the fellow eye is limited to conflicting "
        "observational studies [@p9427177; @p9186127; @p12208242].",
        "Glucagon-like peptide-1 receptor agonists (GLP-1RAs) and the dual GIP/GLP-1 receptor agonist tirzepatide produce large reductions in glycated hemoglobin "
        "and body weight [@p33567185; @p35658024] and reduce cardiovascular events [@p27633186; @p37952131]. Since 2024, cohort studies have reported hazard ratios "
        "of about 1.8 to 2.4 for NAION with semaglutide [@p38958939; @p39696569; @p41678180; @p40146102], whereas other cohorts and meta-analyses found no "
        "consistent increase [@p39491755; @p39976940; @p41692115]. Absolute risks are low [@p41678180; @p42132708], and proposed mechanisms — rapid glycemic lowering, "
        "blood-pressure and volume changes, and intracranial-pressure effects — remain hypotheses [@p40055951; @p41857480; @p42866022; @p42547311]. Consensus "
        "statements recommend shared decision-making and call for identifying which patients are susceptible [@p42132708; @p41058255; @p41022666].",
        "No individual risk tool exists, and patient-level data remain siloed. We assessed whether machine learning applied to reconstructed published evidence "
        "can stratify NAION risk among GLP-1RA users, which features matter, and whether graph neural networks, ensembles, convolutional networks, synthetic digital "
        "twins or transfer from non-users improve on standard models."])


def methods_extended():
    return "\n\n".join([
        "### Detailed derivation of every dataset",
        "**Search corpus.** A single PubMed query combined drug terms (semaglutide, tirzepatide, liraglutide, dulaglutide, exenatide, GLP-1, glucagon-like peptide) "
        "with outcome terms (NAION, ischemic/ischaemic optic neuropathy, optic neuropathy). Metadata, abstracts and PubMed Central open-access full texts, including "
        "embedded tables, were retrieved programmatically; publication types were recorded.",
        "**Extraction datasets.** Four structured datasets were produced by three independent extraction agents working from the same rules: (1) a case-level table "
        "with one row per published patient (age, sex, drug, formulation, dose, recent escalation, time to onset, indication, comorbidities, disc morphology, drusen, "
        "dehydration, laterality, visual acuity converted to logMAR, imaging available, treatment and recovery); (2) a cohort-level table with one row per study, contrast "
        "and subgroup (design, data source, population, exposure, comparator, arm sizes and events, person-years, effect measure with CI, adjustment, follow-up and "
        "population characteristics); (3) a pharmacovigilance table (database, drug, formulation, subgroup, case counts, ROR, PRR, IC025, EBGM); and (4) a pooled-trial "
        "and meta-analysis table (pooled estimates by subgroup and risk factors named by consensus statements). Values not reported were left missing; every row "
        "carries the PMID, evidence level and a supporting quotation of at most 200 characters, and possible duplicate cohorts were flagged.",
        "**Reconstructed patient-level dataset (pseudo-IPD).** From the cohort table we kept contrasts with event counts in both arms and selected one primary, "
        "non-overlapping contrast per study, preferring crude full-cohort counts and excluding overlapping national-registry analyses and a case-control design. "
        "Each arm produced an event row and a non-event row weighted by their counts. Arm-level variables were exposure and therapy; study-level variables were "
        "indication, region, data source, outcome definition, follow-up, mean age and percentage female, with medians imputed and missingness flagged. Because "
        "uniform random sampling of patients is equivalent to binomial (single split) or multinomial (k-way partition) thinning of counts, every split is an exact "
        "patient-level split.",
        "**Non-GLP and GLP sub-datasets.** For transfer experiments the reconstructed data were divided into comparator-arm patients (no GLP-1RA) and exposed-arm "
        "patients. Background-risk models used only features defined in both populations (exposure and therapy excluded).",
        "**Synthetic datasets (digital twins).** Three generators produced twin profiles: P1 sampled each feature independently from training marginals with risk equal "
        "to the therapy-stratum rate; P2 sampled joint covariate patterns with Gaussian-copula jitter on numeric features and drew each twin's risk from the Beta "
        "posterior of its source pattern; P3 used P2 features with risk from a logistic model restricted to SHAP-stable features. Generators were fitted to the real "
        "training split only (source = real), to published priors only (source = prior: a non-user baseline of about 1.35 per 10,000 person-years derived from a pooled "
        "incidence of 85 per 100,000 GLP-1RA users and a pooled ratio of 2.52 [@p42060879], with drug-specific ratios of 1.93 and 1.70 [@p42612051; @p41498755]), "
        "or to a 50/50 blend. Twin counts were distributed over a bank of 5,000 profiles by multinomial sampling and outcomes drawn binomially.",
        "**Released datasets.** The search records, the four extraction tables, the reconstructed counts, an exact 1% patient-level expansion, the Introduction "
        "evidence table and eight synthetic sets of 200,000 twins each (three generators × two prior sources, plus non-GLP and GLP populations) are released with a "
        "data dictionary and checksums.",
        "### Detailed description of every experiment",
        "**Stage 1 — PubMed-only supervised learning.** Logistic regression (weights rescaled to the number of events so that the L2 penalty is meaningful), random forest "
        "and XGBoost were evaluated with a 10%/90% split, ten disjoint 10% training folds and leave-one-study-out validation; SHAP stability was summarized across folds. "
        "**Stage 1b** clustered case-report patients (k-means, Gaussian mixture, Ward; bootstrap co-assignment) and fitted a random-forest meta-regression of published "
        "log ratios with grouped cross-validation. **Stage 1c** repeated Stage 1 at training fractions of 10% to 90% (10 repeats each) and measured SHAP-rank agreement "
        "with Kendall's W.",
        "**Stage 2 — 3×6×9 twin lattice.** XGBoost was trained on mixtures defined by three generators, six volumes (1, 3, 6, 9, 27 and 100 times the real training size) "
        "and synthetic shares of 10% to 100%, crossed with real held-out fractions of 10% to 100% and three seeds (5,310 fits). The training set of size V×R contained "
        "(1−s)×V×R real patients (sampled without replacement when available, otherwise bootstrapped) and s×V×R twins. Refinement recursed around the best cell with "
        "halving steps while AUROC improved by ≥0.005 (maximum depth 3).",
        "**Stage 3 — full factorial.** Four models (logistic regression, random forest, XGBoost and their mean) trained on identical data per cell; real training fractions "
        "0% to 90%; seven generator sets (every non-empty combination of P1, P2, P3); three prior sources; six volumes; ten shares; ten repeats. Differences from "
        "PubMed-only training on the same split were tested per configuration (one-sided paired t) with Benjamini-Hochberg correction; robust configurations were "
        "ranked across all training fractions under a calibration constraint (|O/E−1|<0.10, slope 0.9 to 1.1).",
        "**Stage 4 — ensembles.** Eleven member subsets of logistic regression, random forest, XGBoost and the GNN, three combiners (mean, inner-CV AUROC weights, logistic "
        "stacking on inner out-of-fold logits) and a cross-family ensemble, under the same split paradigms.",
        "**Stage 5 — inference.** Count-weighted logistic models (marginal with cluster-robust errors; within-study with study fixed effects, including exposure by drug, "
        "indication, comparator and study-level age, sex and follow-up); SHAP permutation tests (500 refits, global and within-study nulls); random-effects meta-analysis "
        "with subgroup tests; demographic contrasts (exact binomial, Wilcoxon, ratio-of-ratios z tests); and threshold metrics with tests against chance, McNemar and "
        "corrected resampled t comparisons, and net benefit.",
        "**Stage 6 — graph neural networks.** GraphSAGE on a pattern graph (k-nearest neighbours plus same-study links), with integrated-gradient attributions, under all "
        "split paradigms; **Stage 6c** tuned depth (1 to 6), k, edge weighting and same-study weight (360 configurations) and learned attention weights (graph attention "
        "network, depths 1 to 6), selecting on folds 0 to 4, confirming on folds 5 to 9 and testing the top configurations by leave-one-study-out validation.",
        "**Stage 7 — background-risk model with external validation in GLP-1RA users.** Models trained only on non-GLP patients (internal splits as above) were applied to "
        "all GLP-1RA-exposed patients to discriminate NAION from no NAION; observed-to-expected ratios with exact Poisson intervals quantify how much a background model "
        "under-predicts GLP-1RA users. A GLP-aware model trained on both arms served as reference.",
        "**Stage 8 — fine-tuning.** Within each of five repeats, non-GLP patients were split 80/20 (pretraining pool, forgetting test) and GLP patients 50/10/40 "
        "(fine-tuning pool, validation, test). Pretraining sources (none, real, synthetic, hybrid) were crossed with fine-tuning sources (real, synthetic, hybrid) and "
        "GLP fractions (0%, 5%, 10%, 25%, 50%). Update methods were recalibration in the large, logistic recalibration, continued XGBoost boosting from the pretrained "
        "margin (rounds selected on validation), GNN joint training with GLP up-weighting (selected on validation), pooled refitting and GLP-only training. Three "
        "additional split schemes tested transport in each direction and the standard mixed split.",
        "**Stage 10 — incidence, rare-event and anomaly models.** Incidence rates per 100,000 person-years used reported person-years where available and otherwise N × "
        "follow-up; rates were stratified by drug, indication, comparator, region, data source, study age band, percentage female and follow-up, with exact conditional "
        "confidence intervals for incidence rate ratios, Mantel-Haenszel ratios stratified by study and Poisson models with study fixed effects, exposure-by-stratum terms and a "
        "log person-time offset (Wald tests of heterogeneity). Rare-event learners were a Poisson rate model with offset, class-weighted XGBoost with prior correction, King-Zeng "
        "corrected logistic regression, a focal-loss network and incidence-anchored XGBoost recalibrated to observed incidence within exposure-by-age strata. Anomaly detectors "
        "(Isolation Forest, one-class SVM, Local Outlier Factor and an autoencoder) were trained on non-event patients of the training split only.",
        "**Stage 11 — detection within GLP-1RA users and within non-users.** Every model (logistic regression, random forest, XGBoost, GNN, 1D-CNN, mean ensemble, "
        "Poisson rate model, Isolation Forest and autoencoder) was trained on mixed data in each of the ten folds and evaluated separately within the exposed and unexposed "
        "parts of the test set (AUROC, PR-AUC, O/E, calibration slope and threshold metrics at the training top-10% threshold); group differences were tested with corrected "
        "paired t tests and Benjamini-Hochberg correction.",
        "**Stage 12 — validated vascular score and a proposed NAION-adapted score.** The ABCD2 score and its newest extension ABCD3-I were derived where possible "
        "(age 60 years or older as the expected share from study mean age with an assumed SD of 12 years; blood pressure from hypertension prevalence; diabetes from the study "
        "population) and used only as features. A points score was fitted to the data and, separately, built from published patient-level ratios and our within-study "
        "GLP-1RA estimates, with points equal to log ratio divided by log 1.28 (hypertension = 1 point), recalibrated on training data and validated with the same splits. "
        "**Stage 12b** repeated every experiment with and without the ABCD features using identical splits and seeds, so that all differences are paired.",
        "**Stage 14 — feature combinations and the compounding hypothesis.** Because a calculator should accept whatever inputs are available, models were trained on "
        "every non-empty combination of nine feature groups (exposure, drug, indication, age, sex, duration, hypertension, diabetes and setting; 511 combinations) with "
        "logistic regression, random forest and XGBoost on all ten folds and with the GNN, 1D-CNN and mean ensemble on three folds. Exact Shapley values and pairwise "
        "Shapley interaction indices of AUROC quantified each group's contribution across all coalitions; the hypothesis that individually negligible groups contribute "
        "in combination was tested on fold-level values with corrected t tests and Benjamini-Hochberg correction. A single XGBoost trained with random feature-group "
        "dropout was compared with a bank of dedicated sub-models for each input pattern.",
        "**Stage 9 — convolutional networks and multimodal feasibility.** A one-dimensional CNN treated the encoded feature vector as a sequence (two convolution layers, "
        "global max pooling and a skip connection) under the same paradigms. A two- and three-dimensional imaging pipeline (fundus and OCT volume encoders, late fusion "
        "with tabular features and Grad-CAM) was implemented and verified on random tensors, but could not be trained because no analysable images exist in the "
        "published corpus.",
    ])


def results_extra():
    S7, S8, S9 = J("results/stage7/summary.json"), J("results/stage8/summary.json"), J("results/stage9/summary.json")
    out = []
    if S7:
        k = pd.DataFrame(S7["kfold"]); ext = k[k.population == "external_GLP"].set_index("model"); inn = k[k.population == "internal_nonGLP"].set_index("model")
        best = ext.auroc_mean.idxmax()
        out.append(f"**Background-risk model transported to GLP-1RA users.** Trained on {S7['n_train_population']:,} non-GLP patients ({S7['events_train_population']:,} events), "
                   f"background models discriminated within non-users (AUROC {f(inn.auroc_mean.min())} to {f(inn.auroc_mean.max())}) but much less among "
                   f"{S7['n_external_GLP']:,} GLP-1RA users ({S7['events_external_GLP']:,} events): AUROC {f(ext.auroc_mean.min())} to {f(ext.auroc_mean.max())} "
                   f"(best, {best}). Observed events among GLP-1RA users exceeded background predictions (O/E {f(ext.o_e_mean.min(),2)} to {f(ext.o_e_mean.max(),2)}), "
                   "an independent estimate of excess risk consistent with the within-study odds ratio. Traditional and setting features therefore explain little of "
                   "which GLP-1RA users develop NAION.")
    else: out.append(PENDING("the non-GLP to GLP external validation (Stage 7)"))
    if S8:
        t = S8["top10"][0]; sig = S8["significant_improvements_vs_pretrained"]
        out.append(f"**Fine-tuning.** The best update ({t['method'].replace('_',' ')}; pretraining {t['pretrain_source']}, fine-tuning {t['finetune_source']}, "
                   f"{int(round(100*t['glp_frac']))}% of GLP-1RA patients) reached AUROC {f(t['glp_auroc'])} in held-out GLP-1RA users with O/E {f(t['glp_o_e'],2)}. "
                   f"{len(sig)} configurations improved on the pretrained background model after Benjamini-Hochberg correction. Recalibration corrected "
                   "under-prediction without changing discrimination, as expected; synthetic fine-tuning data did not outperform real data.")
    else: out.append(PENDING("the fine-tuning experiments (Stage 8)"))
    if S9:
        out.append(f"**Convolutional network.** The 1D-CNN reached AUROC {f(S9['kfold']['auroc_mean'])} ± {f(S9['kfold']['auroc_sd'])} across ten folds, not different "
                   f"from XGBoost (difference {f(S9['kfold']['diff_vs_xgb'],4)}; P={pq(S9['kfold']['p_vs_xgb_corrected_t'])}), and {f(S9['loso']['auroc_mean'])} in "
                   "leave-one-study-out validation. Convolutions add nothing when features lack spatial or sequential structure. Two- and three-dimensional CNNs and "
                   "multimodal fusion were not trainable: the corpus contains one usable modality (structured tabular data derived from text), imaging only as "
                   "published figures for about six patients, and no video.")
    else: out.append(PENDING("the CNN experiment (Stage 9)"))
    return "\n\n".join(out)


def multimodal_table():
    rows = [("Structured tabular (categorical + numeric)", "Yes", "3.27 million reconstructed patients", "All supervised, graph, CNN-1D and twin models"),
            ("Free text (abstracts, full texts)", "Yes", "169 records; 60 full texts", "Extraction only; too few labelled patients for text models"),
            ("Images (fundus, OCT, visual fields, MRI)", "Figures only", "about 6 patients with pre/post figures", "Not trainable; 2D/3D CNN + Grad-CAM pipeline prepared"),
            ("3D volumes (OCT cubes)", "No", "0", "Not available"),
            ("Video / time series", "No", "0", "Not available")]
    return pd.DataFrame(rows, columns=["Modality", "Available", "Amount", "Use / feasibility"]).to_markdown(index=False)


def results_incidence():
    S = J("results/stage10/summary.json")
    if not S: return PENDING("the incidence, rare-event and anomaly-detection analysis (Stage 10)")
    st = pd.DataFrame(S["incidence_stratified"]); o = st[st.stratifier == "overall"].iloc[0]
    pois = S["poisson_effect_modification"]; mods = pd.DataFrame(S["models"]).set_index("model")
    def row(v, l): r = st[(st.stratifier == v) & (st.level == l)]; return r.iloc[0] if len(r) else None
    t2d, iih = row("indication", "t2d"), row("indication", "iih")
    anom = mods[mods.family == "anomaly"]
    return (f"**Incidence.** NAION occurred at {f(o.glp_rate_per_100k_py,1)} per 100,000 person-years among GLP-1RA users and {f(o.nonglp_rate_per_100k_py,1)} among "
            f"comparators (incidence rate ratio {f(o.crude_IRR,2)}; 95% CI, {f(o.IRR_lo,2)} to {f(o.IRR_hi,2)}; Mantel-Haenszel ratio stratified by study {f(o.MH_IRR_by_study,2)}), "
            f"a rate difference of {f(o.rate_diff_per_100k_py,1)} per 100,000 person-years. Both rates lie within or above the published background range for adults aged 50 "
            f"years or older [@p8032479; @p9186104]. The ratio was {f(t2d.crude_IRR,2)} in type 2 diabetes and {f(iih.crude_IRR,2)} in idiopathic intracranial hypertension, and "
            f"Poisson models with study fixed effects showed heterogeneity of the exposure effect by study age band (P={pq(pois['age_band']['p_heterogeneity'])}), indication "
            f"(P={pq(pois['indication']['p_heterogeneity'])}), drug (P={pq(pois['drug']['p_heterogeneity'])}), comparator (P={pq(pois['comparator']['p_heterogeneity'])}) and "
            f"follow-up (P={pq(pois['fu_band']['p_heterogeneity'])}) but not by the percentage of women (P={pq(pois['female_band']['p_heterogeneity'])}). Because age is a study-level "
            "attribute and person-years were reported by only two studies, these strata describe settings rather than individual patients.\n\n"
            f"**Rare-event and anomaly-detection models.** A Poisson rate model with a person-time offset (AUROC {f(mods.loc['poisson_offset','auroc_mean'])}), King-Zeng corrected "
            f"logistic regression ({f(mods.loc['logreg_kingzeng','auroc_mean'])}) and incidence-anchored XGBoost ({f(mods.loc['xgb_incidence_anchored','auroc_mean'])}) matched standard "
            f"XGBoost ({f(mods.loc['xgb_standard','auroc_mean'])}); class weighting required prior correction to restore calibration, and a focal-loss network was miscalibrated. "
            f"Anomaly detectors trained only on patients without NAION performed near chance (AUROC {f(anom.auroc_mean.min())} to {f(anom.auroc_mean.max())}), because patients "
            "with and without NAION from the same study arm share identical features in reconstructed aggregate data; anomaly methods need individual-level covariates.")


STRAT_LABELS = {"overall": "Overall", "drug": "Drug", "indication": "Indication", "comparator": "Comparator", "region": "Region", "source_type": "Data source",
                "age_band": "Study mean age", "female_band": "Women in study", "fu_band": "Follow-up"}
LEVEL_LABELS = {"all": "All studies", "glp1ra_class": "GLP-1RA class", "sema_or_tirz": "Semaglutide or tirzepatide", "semaglutide": "Semaglutide", "iih": "IIH",
                "t2d": "T2D", "dpp4i": "DPP-4i", "no_glp1ra": "No GLP-1RA", "other_antidiabetic": "Other antidiabetic", "sglt2i": "SGLT2i", "nordic": "Nordic",
                "uk": "UK", "us": "US", "claims": "Claims", "ehr_network": "EHR network", "ehr_primary_care": "Primary-care EHR", "national_registry": "National registry",
                "single_center": "Single center", "va_ehr": "VA EHR", "55-59": "55-59 y", "<55": "<55 y", ">=60": "60 y or older", "unknown": "Not reported",
                "40-54%": "40-54%", "<40%": "<40%", ">=55%": "55% or more", "<=1 y": "1 y or less", "1-2.5 y": "1-2.5 y", ">2.5 y": ">2.5 y"}


def table_incidence():
    p = "results/stage10/incidence_stratified.csv"
    if not os.path.exists(p): return ""
    d = pd.read_csv(p)
    d = d[["stratifier", "level", "studies", "glp_events", "glp_rate_per_100k_py", "nonglp_events", "nonglp_rate_per_100k_py", "crude_IRR", "IRR_lo", "IRR_hi", "MH_IRR_by_study", "q_bh"]]
    for c in ["glp_rate_per_100k_py", "nonglp_rate_per_100k_py"]: d[c] = d[c].map(lambda v: f(v, 1))
    for c in ["crude_IRR", "IRR_lo", "IRR_hi", "MH_IRR_by_study"]: d[c] = d[c].map(lambda v: f(v, 2))
    d["q_bh"] = d.q_bh.map(pq); d["glp_events"] = d.glp_events.astype(int); d["nonglp_events"] = d.nonglp_events.astype(int)
    order = {"overall": 0, "age_band": 1, "female_band": 2, "fu_band": 3, "indication": 4, "drug": 5, "comparator": 6, "region": 7, "source_type": 8}
    lvl_order = {"<55": 0, "55-59": 1, ">=60": 2, "<40%": 0, "40-54%": 1, ">=55%": 2, "<=1 y": 0, "1-2.5 y": 1, ">2.5 y": 2, "unknown": 9}
    d = d.assign(_o=d.stratifier.map(order).fillna(99), _l=d.level.map(lvl_order).fillna(5)).sort_values(["_o", "_l", "level"]).drop(columns=["_o", "_l"])
    d["stratifier"] = d.stratifier.map(STRAT_LABELS).fillna(d.stratifier); d["level"] = d.level.map(LEVEL_LABELS).fillna(d.level)
    return "Table: NAION incidence per 100,000 person-years in GLP-1RA users and comparators, by stratum (IRR, incidence rate ratio; MH, Mantel-Haenszel by study; q, Benjamini-Hochberg).\n\n" + d.rename(columns={
        "stratifier": "Stratifier", "level": "Level", "studies": "Studies", "glp_events": "GLP events", "glp_rate_per_100k_py": "GLP rate", "nonglp_events": "Non-GLP events",
        "nonglp_rate_per_100k_py": "Non-GLP rate", "crude_IRR": "IRR", "IRR_lo": "95% CI low", "IRR_hi": "95% CI high", "MH_IRR_by_study": "MH IRR", "q_bh": "q"}).to_markdown(index=False)


def results_subgroups():
    S = J("results/stage11/summary.json")
    if not S: return PENDING("the GLP-1RA vs non-user subgroup evaluation (Stage 11)")
    s = pd.DataFrame(S["summary"]); t = pd.DataFrame(S["tests"])
    a = s.pivot_table(index="model", columns="group", values="auroc_mean")
    sig = t[(t.metric == "auroc") & (t.q_bh < 0.05)]
    best_g = a["GLP-1RA users"].idxmax()
    return (f"**Performance within GLP-1RA users and within non-users.** Trained on mixed data, every model discriminated better among non-users (AUROC "
            f"{f(a['non-users'].min())} to {f(a['non-users'].max())}) than among GLP-1RA users ({f(a['GLP-1RA users'].min())} to {f(a['GLP-1RA users'].max())}; best, "
            f"{best_g}). The within-group difference in AUROC was significant after correction for {len(sig)} of {t[t.metric=='auroc'].shape[0]} models. Together with the "
            "under-prediction of a background model in GLP-1RA users, this indicates that the features available in published data capture background NAION risk but "
            "little of what determines which GLP-1RA users develop NAION.")


def table_subgroups():
    p = "results/stage11/subgroup_summary.csv"
    if not os.path.exists(p): return ""
    d = pd.read_csv(p); d = d[d.group != "all"][["model", "group", "auroc_mean", "pr_auc_mean", "o_e_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean", "ppv_mean", "npv_mean"]]
    for c in d.columns[2:]: d[c] = d[c].map(lambda v: f(v, 4) if pd.notna(v) else "—")
    MN = {"logreg": "LR", "rf": "RF", "xgb": "XGB", "gnn": "GNN", "cnn1d": "CNN", "ensemble": "Ens.", "poisson_offset": "Poisson", "isolation_forest": "IForest", "autoencoder": "AE"}
    d["model"] = pd.Categorical(d.model.map(MN).fillna(d.model), categories=list(MN.values()), ordered=True); d = d.sort_values(["model", "group"])
    return "Table: Model performance within GLP-1RA users and within non-users (mean over ten folds; threshold = top 10% of training risk).\n\n" + d.rename(columns={
        "model": "Model", "group": "Group", "auroc_mean": "AUROC", "pr_auc_mean": "PR-AUC", "o_e_mean": "O/E", "sensitivity_mean": "Sens", "specificity_mean": "Spec",
        "accuracy_mean": "Acc", "ppv_mean": "PPV", "npv_mean": "NPV"}).to_markdown(index=False)


def results_abcd():
    S, A = J("results/stage12/summary.json"), J("results/stage13/summary.json")
    if not S: return PENDING("ABCD2 derivation and the proposed NAION-adapted score (Stage 12)")
    k = pd.DataFrame(S["kfold"]).set_index("model")
    txt = (f"**Validated vascular score as a feature.** Only three ABCD2/ABCD3-I items could be derived: age 60 years or older (7 of 11 studies, from study mean age), "
           f"blood pressure (5 of 11, using hypertension prevalence as a proxy) and diabetes (11 of 11); clinical features, symptom duration, dual events and imaging are "
           f"specific to transient ischemic attack. Adding the partial items to XGBoost changed AUROC by {f(S['abcd_added_to_xgb']['diff'],4)} "
           f"(P={pq(S['abcd_added_to_xgb']['p'])}). The partial score alone, in studies with all items, reached AUROC "
           f"{f(k.loc['abcd2_partial_alone_complete_studies','auroc_mean']) if 'abcd2_partial_alone_complete_studies' in k.index else 'NA'}.\n\n"
           f"**Proposed NAION-adapted points score.** Fitting point weights to the reconstructed data produced implausible protective signs for hypertension, diabetes and "
           f"GLP-1RA exposure, because these items vary only between studies; we therefore report this fit only as a demonstration of ecological bias. A literature-informed "
           f"version assigned points from published patient-level ratios (male sex, hypertension, hyperlipidemia, diabetes, coronary heart disease and sleep apnea [@p34671609]) "
           f"and from our within-study GLP-1RA estimates (indication-specific), with age carried from the ABCD2 convention: "
           + ", ".join(f"{k_.replace('_',' ')} {v} point{'s' if v != 1 else ''}" for k_, v in S["literature_score_points"].items()) +
           f". On setting-level data it ranked cohorts poorly (AUROC {f(k.loc['proposed_literature_score','auroc_mean'])}; difference from XGBoost {f(S['literature_score_vs_xgb']['diff'],3)}), "
           "as expected when settings with more vascular risk factors recorded fewer events; the score requires prospective, patient-level calibration, including disc morphology.\n\n"
           "**Limitations of the partial score.** " + "; ".join(S["partial_score_limitations"]) + ".")
    cdr = S.get("cdr_evidence")
    if cdr:
        txt += ("\n\n**Cup-to-disc ratio.** No source reported a pre-treatment (pre-GLP-1RA) cup-to-disc ratio. In the case reports, disc status was recorded at or after "
                "NAION diagnosis, usually in the unaffected fellow eye. The largest source, a 16-center case-control study of 1,812 unaffected fellow eyes, found a fellow-eye "
                "ratio of 0.20 or less in 54% of patients with NAION versus 9% of controls (odds ratio 46.3; 95% CI, 22.9 to 93.4; ratio 0.21 to 0.40: 6.8; 3.5 to 13.2) "
                "and no difference between semaglutide users, other GLP-1RA users and unexposed patients [@p42556433]. We therefore entered the cup-to-disc ratio as the "
                "highest-weighted item of the proposed score but labelled it a post-exposure fellow-eye measurement: disc anatomy is structurally stable and the fellow eye is "
                "the accepted proxy, yet this is weaker evidence than a baseline measured before GLP-1RA initiation, and it could not be validated in the cohort data. "
                "The same limitation applies to NAION without GLP-1RA exposure: the classic and recent disc-at-risk studies measured the cup-to-disc ratio after the event, in "
                "fellow eyes or in affected eyes after edema resolved [@p6514298; @p3684223; @p38211825; @p42556433], and none had a measurement taken before NAION. Because "
                "both groups share this post-event design, the comparison between exposed and unexposed NAION is balanced, but no source supports a truly prospective (pre-event) "
                "estimate of disc-related risk. Baseline optic disc photography or OCT before GLP-1RA initiation in a prospective cohort would close this gap.")
    if A:
        txt += (f"\n\n**With versus without ABCD features across all experiments.** Over {A['n_comparisons']} paired comparisons using identical splits and seeds, adding the "
                f"partial ABCD items changed AUROC by {f(A['auroc_mean_diff'],4)} on average; {len(A['significant_q05'])} comparisons differed after Benjamini-Hochberg correction.")
    else: txt += "\n\n" + PENDING("the with/without-ABCD ablation across every experiment (Stage 12b)")
    return txt


def table_abcd_ablation():
    p = "results/stage13/abcd_ablation.csv"
    if not os.path.exists(p): return ""
    d = pd.read_csv(p); d = d[d.metric == "auroc"][["experiment", "model", "without_abcd", "with_abcd", "diff", "q_bh"]]
    for c in ["without_abcd", "with_abcd"]: d[c] = d[c].map(lambda v: f(v, 4))
    d["diff"] = d["diff"].map(lambda v: f(v, 4)); d["q_bh"] = d.q_bh.map(pq)
    return "Table: AUROC with and without partial ABCD2 features in every experiment (paired, identical splits; q, Benjamini-Hochberg).\n\n" + d.rename(columns={
        "experiment": "Experiment", "model": "Model", "without_abcd": "Without ABCD", "with_abcd": "With ABCD", "diff": "Difference", "q_bh": "q"}).to_markdown(index=False)


def results_combinations():
    S = J("results/stage14/summary.json")
    if not S: return PENDING("the feature-combination and compounding-hypothesis experiment (Stage 14)")
    sh = pd.DataFrame(S["shapley"]); x = sh[sh.model == "xgb"].sort_values("shapley_value", ascending=False)
    sup = pd.DataFrame(S["H_compound_supported"]); syn = pd.DataFrame(S["synergistic_pairs_q05"])
    top = ", ".join(f"{r.group} ({f(r.shapley_value,4)})" for r in x.head(4).itertuples())
    sup_txt = ("no feature group met the prespecified criteria" if sup.empty else
               "; ".join(f"{r.group} in {r.model} (stand-alone gain {f(r.standalone_gain,4)}, Shapley value {f(r.shapley_value,4)}, q={pq(r.q_shapley)})" for r in sup.itertuples()))
    syn_txt = ("no pair showed significant positive interaction" if syn.empty else
               f"{len(syn)} pairs showed significant positive interaction, led by " + ", ".join(f"{r.pair} in {r.model} ({f(r.interaction,4)})" for r in syn.sort_values('interaction', ascending=False).head(3).itertuples()))
    return (f"**Feature combinations and the compounding hypothesis.** Across all {S['n_subsets']} combinations of nine feature groups, the largest Shapley contributions to "
            f"XGBoost AUROC came from {top}. For the hypothesis that individually negligible features contribute in combination, {sup_txt}; {syn_txt}. "
            f"For an optional-input calculator, a single XGBoost trained with random feature-group dropout differed from a bank of dedicated sub-models by "
            f"{f(S['dropout_vs_bank_mean_diff'],4)} AUROC on average across input patterns.")


def tables_combinations():
    p, q = "results/stage14/combination_performance.csv", "results/stage14/odds_ratio_stability.csv"
    if not os.path.exists(p): return ""
    d = pd.read_csv(p); d = d[d.model != "xgb_dropout_single"]
    best = d.sort_values("auroc_mean", ascending=False).groupby("k").head(2).sort_values(["k", "auroc_mean"], ascending=[True, False])
    cols = {"subset": "Feature groups", "model": "Model", "auroc_mean": "AUROC", "auroc_std": "SD", "q_auroc_vs_0.5": "q (AUROC>0.5)",
            "T2_top10pct_sensitivity_mean": "Sens", "T2_top10pct_specificity_mean": "Spec", "T2_top10pct_accuracy_mean": "Acc",
            "T2_top10pct_ppv_mean": "PPV", "T2_top10pct_npv_mean": "NPV", "T2_top10pct_q_ppv_vs_prevalence": "q (PPV>prev)"}
    t = best[[c for c in cols if c in best.columns]].copy()
    for c in t.columns:
        if c.startswith("q_") or "_q_" in c: t[c] = t[c].map(pq)
        elif t[c].dtype.kind == "f": t[c] = t[c].map(lambda v: f(v, 4) if abs(v) >= 0.01 else f"{v:.5f}")
    out = ("Table: Best feature-group combinations by number of groups (two best per size; mean over folds; threshold = training top 10%; q, Benjamini-Hochberg over all "
           "combinations and models; full results for all 511 combinations and every model in Supplementary Workbook S1).\n\n" + t.rename(columns=cols).to_markdown(index=False))
    if os.path.exists(q):
        s = pd.read_csv(q).sort_values("share_q05", ascending=False)
        for c in ["median_OR", "min_OR", "max_OR"]: s[c] = s[c].map(lambda v: ">10,000" if v >= 1e4 else ("<0.01" if v < 0.005 else f(v, 2)))
        for c in ["share_OR_gt1", "share_q05"]: s[c] = (100 * s[c]).map(lambda v: f"{v:.0f}%")
        out += ("\n\nTable: Odds-ratio reliability of each feature across all combinations that contain it (count-weighted logistic regression with cluster-robust standard "
                "errors by study; study-level covariates are ecological and imprecise with 11 clusters).\n\n" + s.rename(columns={"base_feature": "Feature", "n_models": "Models",
                "median_OR": "Median OR", "min_OR": "Min OR", "max_OR": "Max OR", "share_OR_gt1": "OR>1", "share_q05": "q<0.05"}).to_markdown(index=False))
    return out


def results_compute_imaging(short=False):
    C, I = J("results/stage15/summary.json"), J("results/stage16/summary.json")
    out = []
    if C:
        lt = C["largest_stages"][0]
        out.append(f"**Compute, energy and token use.** All model training and inference ran locally on one laptop (Apple M3 Pro, 11 cores, 18 GB). Across every experiment "
                   f"the analyses used {f(C['compute_hours_total'],1)} wall-clock hours and {f(C['core_hours_total'],1)} core-hours ({f(100*C['share_runtime_logged'],0)}% "
                   f"of run time measured from logs, the rest estimated), consuming an estimated {f(C['energy_kwh']['central'],2)} kWh (range {f(C['energy_kwh']['low'],2)} to "
                   f"{f(C['energy_kwh']['high'],2)}) and emitting {f(1000*C['co2e_kg']['central'],0)} g CO2e at the US average grid intensity (range "
                   f"{f(1000*C['co2e_kg']['low'],0)} to {f(1000*C['co2e_kg']['high'],0)} g), about {f(C['equivalents_central']['km_driven_passenger_car_(0.17 kg/km)'],1)} km "
                   f"of car travel. The largest single consumer was the {lt['stage'].split('·')[-1].strip()} ({f(lt['seconds']/3600,1)} h). Inference was negligible: "
                   + ", ".join(f"{r['model']} {r['seconds_per_1000']*1000:.2f} ms" for r in C["inference"]) + " per 1,000 patients. The AI assistant used for retrieval, "
                   f"extraction and drafting consumed at least {C['llm_tokens_total_known']:,} tokens in the four metered extraction and literature agents; tokens of the main "
                   "session were not metered, and assistant inference energy, which occurs in remote data centres, is not included in these totals.")
    else: out.append(PENDING("the compute and carbon accounting (Stage 15)"))
    if I:
        s = pd.DataFrame(I["summary"]); c = s[s.effect_level == "central"].set_index("scenario")
        g = lambda k, col="auroc": c.loc[k, col]
        out.append(f"**What-if: adding ocular imaging (simulation).** Imaging values were simulated for the reconstructed patients conditional on their observed outcome. "
                   f"Adding cup-to-disc ratio from fundus photographs, with published odds ratios [@p42556433] and 25% photographic misclassification, raised AUROC from "
                   f"{f(g('base (published features only)'))} to {f(g('+ fundus-photo CDR'))}; OCT-measured ratio reached {f(g('+ OCT CDR'))}. Adding hypothetical effects for optic "
                   f"disc drusen, OCT angiography perfusion and serial on-treatment imaging reached {f(g('+ all imaging incl. serial'))} (range across hypothetical effect sizes "
                   f"{f(s[(s.scenario=='+ all imaging incl. serial')].auroc.min())} to {f(s[(s.scenario=='+ all imaging incl. serial')].auroc.max())}), and {f(g('+ all imaging + OSA'))} "
                   f"with sleep apnoea added; sleep apnoea alone reached {f(g('+ OSA only (non-imaging)'))}. Positive predictive value at the top-10% threshold rose from "
                   f"{f(100*g('base (published features only)','ppv_top10'),3)}% to {f(100*g('+ all imaging + OSA','ppv_top10'),3)}%, still below 1% because NAION is rare. "
                   f"A prospective cohort would need about {int(g('+ all imaging incl. serial','prospective_events_for_auroc_ci_pm0.03'))} NAION events, roughly "
                   f"{int(g('+ all imaging incl. serial','approx_patients_to_observe_those_events')):,} GLP-1RA initiators at the observed incidence, to estimate AUROC within "
                   "±0.03; a case-control design enriched for NAION would need far fewer imaged patients. These figures are simulations that assume conditional independence "
                   "of features and, for drusen, OCT angiography and serial imaging, effect sizes that have not been measured.")
    else: out.append(PENDING("the imaging what-if simulation (Stage 16)"))
    return "\n\n".join(out)


def methods_compute_imaging():
    return ("**Stage 15 — compute, energy and token use.** Energy was estimated with the Green Algorithms method (runtime × [cores × per-core power + memory × 0.3725 "
            "W/GB] × PUE) for an Apple M3 Pro laptop (PUE 1.0, package power 25 to 45 W), with run times taken from each stage's log where available, and converted to CO2e "
            "at 0.10 to 0.70 (central 0.367) kgCO2e/kWh. Inference time was measured per 1,000 predictions. Tokens consumed by the AI assistant were recorded where metered.\n\n"
            "**Stage 16 — what-if: multimodal ocular imaging.** For each reconstructed patient, imaging features were drawn conditional on the observed outcome "
            "(P[feature | NAION] ∝ prior × odds ratio; P[feature | no NAION] = prior), independently across features: cup-to-disc ratio categories with the published "
            "control distribution and odds ratios (fundus photography with 25% and OCT with 5% adjacent-category misclassification), and hypothetical optic disc drusen "
            "(prevalence 2%, odds ratio 2 to 8), OCT angiography hypoperfusion (20%, 1.5 to 3) and serial on-treatment imaging change (5%, 2 to 10), with sleep apnoea "
            "(27%, 3.28) as a non-imaging comparator. XGBoost was trained with and without these features on five patient-level 10%/90% folds for each of three "
            "generation seeds, and the number of events needed to estimate AUROC within ±0.03 in a prospective cohort was derived from the Hanley-McNeil variance.")

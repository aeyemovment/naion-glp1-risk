"""Stage 12 - validated vascular score (ABCD2 / ABCD3-I) as derived features, and a proposed NAION-adapted points score.

ABCD2 items (Johnston 2007; ABCD3-I adds dual TIA and imaging): Age >=60 (1), Blood pressure >=140/90 (1), Clinical features
(weakness 2 / speech 1), Duration (>=60 min 2 / 10-59 min 1), Diabetes (1). Only A, B and D can be derived from published
NAION cohorts; C, D(uration), dual-event and imaging items are TIA-specific and not derivable. Derived per study arm:
  A = P(age >= 60) from study mean age assuming SD 12 y (available 7/11 studies; else NA + flag)
  B = proportion with hypertension, proxy for BP >= 140/90 (available 5/11; else NA + flag)
  D = 1 for type 2 diabetes cohorts, 0 otherwise (11/11)
  ABCD2_partial = expected A + B + D points (0-3), computed where A and B are both available; reliability flag per study.
Experiments (same paradigms: 10/90, 10 disjoint 10% folds, LOSO):
  (1) ABCD2_partial alone as a risk score (studies with all items)
  (2) XGBoost with vs without ABCD features (paired corrected t), SHAP of ABCD features
  (3) Proposed NAION-adapted points score (working name "NAION-GLP score", proposed, not validated): logistic regression on
      clinician-enterable items - age band, hypertension, diabetes, GLP-1RA exposure, semaglutide, obesity indication -
      coefficients scaled to integer points; points-to-risk table; AUROC, calibration, threshold metrics with P values.
      Crowded disc / small cup-to-disc ratio cannot be estimated from cohort data and is listed as an item for prospective addition.
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
from common import *
from splits import holdout_10_90, kfold_10pct
from scipy import stats
from joblib import Parallel, delayed
from build_ipd import PRIMARY
import os
OUT = "results/stage12"; os.makedirs(OUT, exist_ok=True)
src = open("src/stage5b_classification.py").read(); exec(src[src.index("def w_quantile"):src.index("# ---- (i) hold-out 10/90")])

co = pd.read_json("data/extracted/cohort_level.json")
study_age = co.groupby(co.source_pmid.astype(str)).mean_age_exposed.max()
study_htn = co.groupby(co.source_pmid.astype(str)).pct_hypertension.max()
d = ipd.copy()
d["abcd_A"] = d.study.map(study_age).apply(lambda m: float(stats.norm.sf(60, m, 12)) if pd.notna(m) else np.nan)
# obesity arm of 38958939 has no separate hypertension figure -> NA
htn = d.study.map(study_htn) / 100; htn[(d.study == "38958939") & (d.indication == "obesity")] = np.nan
d["abcd_B"] = htn
d["abcd_D"] = (d.indication == "t2d").astype(float)
d["abcd_complete"] = d.abcd_A.notna() & d.abcd_B.notna()
d["abcd2_partial"] = d.abcd_A + d.abcd_B + d.abcd_D
d[["study", "glp1ra_exposed", "indication", "abcd_A", "abcd_B", "abcd_D", "abcd2_partial", "abcd_complete"]].drop_duplicates().to_csv(f"{OUT}/abcd_items_by_arm.csv", index=False)
ABCD_COLS = ["abcd_A", "abcd_B", "abcd_D", "abcd2_partial"]
XA = X_all.join(d[ABCD_COLS].fillna(d[ABCD_COLS].median())).join(d[["abcd_A", "abcd_B"]].isna().astype(float).add_suffix("_missing"))


def xgb_on(cols, tr, te):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist",
                      base_score=float(np.average(tr.y, weights=tr.n)), verbosity=0, n_jobs=1, random_state=0).fit(XA.loc[tr.index, cols], tr.y, sample_weight=tr.n)
    return m, m.predict_proba(XA.loc[te.index, cols])[:, 1]


def proposed_design(df):
    """Clinician-enterable items of the proposed NAION-adapted score."""
    age = d.loc[df.index, "abcd_A"]
    return pd.DataFrame({"age_ge60_expected": age.fillna(age.median() if age.notna().any() else 0.5),
                         "hypertension": d.loc[df.index, "abcd_B"].fillna(d.abcd_B.median()),
                         "diabetes": d.loc[df.index, "abcd_D"], "glp1ra": df.glp1ra_exposed.astype(float),
                         "semaglutide": ((df.glp1ra_exposed == 1) & df.therapy.isin(["semaglutide", "sema_or_tirz"])).astype(float),
                         "obesity_indication": (df.indication == "obesity").astype(float)}, index=df.index)


def fit_points(tr):
    Z = proposed_design(tr); y, w = tr.y.to_numpy(), tr.n.to_numpy().astype(float)
    m = LogisticRegression(C=1e4, max_iter=5000).fit(Z, y, sample_weight=w * (np.sum(w * y) / np.sum(w)))
    beta = pd.Series(m.coef_[0], index=Z.columns); unit = beta.abs()[beta.abs() > 0.05].min() if (beta.abs() > 0.05).any() else 1.0
    pts = (beta / unit).round().astype(int)
    # recalibrate points -> risk with a 1-variable logistic model on training data
    s_tr = (Z * pts).sum(1)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(s_tr.to_frame(), y, sample_weight=w)
    return m, beta, pts, cal


def score_predict(fitd, df):
    _, _, pts, cal = fitd; s = (proposed_design(df) * pts).sum(1); return cal.predict_proba(s.to_frame())[:, 1], s


# ---------------- literature-informed proposed score (weights from published patient-level RRs + our within-study GLP ORs)
UNIT = np.log(1.28)   # hypertension RR (Liu 2021, PMID 34671609) = 1 point
LIT = {  # item: (ratio, source)
    "male_sex": (1.67, "PMID 34671609"), "hypertension": (1.28, "PMID 34671609"), "hyperlipidemia": (1.43, "PMID 34671609"),
    "diabetes": (1.53, "PMID 34671609"), "coronary_heart_disease": (1.68, "PMID 34671609"), "sleep_apnea": (3.28, "PMID 34671609"),
    "age_ge60": (None, "ABCD2 convention, 1 point (no verified age ratio)"),
    # Disc-at-Risk Study (PMID 42556433): fellow-eye CDR measured AT/AFTER NAION (post-exposure for GLP-1RA users), case-control ORs vs CDR > 0.40.
    # No pre-GLP-1RA (baseline) CDR exists in any source; weights are therefore labelled post-exposure / weaker evidence.
    "cdr_le_0.20_post_exposure_fellow_eye": (46.3, "PMID 42556433 (fellow eye, post-exposure; case-control OR vs CDR > 0.40)"),
    "cdr_0.21_0.40_post_exposure_fellow_eye": (6.8, "PMID 42556433 (fellow eye, post-exposure; case-control OR vs CDR > 0.40)")}
CDR_POP = {"cdr_le_0.20_post_exposure_fellow_eye": 0.09, "cdr_0.21_0.40_post_exposure_fellow_eye": 0.56 - 0.09}   # control-eye distribution, PMID 42556433
S5 = pd.read_csv("results/stage5/pvalues_all.csv"); A2 = S5[S5.family == "A2_within_study_exposure"].drop_duplicates("term").set_index("term")
GLP = {"glp1ra_t2d": (float(A2.loc["exposed_t2d", "OR"]), "this study, within-study OR"), "glp1ra_obesity": (float(A2.loc["exposed_obesity", "OR"]), "this study, within-study OR")}
LIT_PTS = {k: (1 if v[0] is None else int(round(np.log(v[0]) / UNIT))) for k, v in {**LIT, **GLP}.items()}
cov = co.groupby(co.source_pmid.astype(str)).agg(female=("pct_female_exposed", "max"), hld=("pct_hyperlipidemia", "max"), cvd=("pct_cvd", "max"), osa=("pct_osa", "max"))


def lit_items(df):
    s = df.study; med = lambda c: cov[c].median()
    return pd.DataFrame({"male_sex": 1 - s.map(cov.female).fillna(med("female")) / 100, "hypertension": d.loc[df.index, "abcd_B"].fillna(d.abcd_B.median()),
                         "hyperlipidemia": s.map(cov.hld).fillna(med("hld")) / 100, "diabetes": d.loc[df.index, "abcd_D"],
                         "coronary_heart_disease": s.map(cov.cvd).fillna(med("cvd")) / 100, "sleep_apnea": s.map(cov.osa).fillna(med("osa")) / 100,
                         "age_ge60": d.loc[df.index, "abcd_A"].fillna(d.abcd_A.median()),
                         "glp1ra_t2d": ((df.glp1ra_exposed == 1) & (df.indication == "t2d")).astype(float),
                         "glp1ra_obesity": ((df.glp1ra_exposed == 1) & (df.indication == "obesity")).astype(float),
                         # CDR not reported by any cohort: population prevalence assumed identical in every arm (adds no discrimination here)
                         **{k: np.full(len(df), v) for k, v in CDR_POP.items()}}, index=df.index)


def lit_score(df): return (lit_items(df) * pd.Series(LIT_PTS)).sum(1)


def evaluate(tag, tr, te):
    _, y, w = XY(te); out = []
    _, p0 = xgb_on(COLS, tr, te); out.append({"tag": tag, "model": "xgb_base", **metrics(y, p0, w)})
    _, p1 = xgb_on(COLS + ABCD_COLS + ["abcd_A_missing", "abcd_B_missing"], tr, te); out.append({"tag": tag, "model": "xgb_plus_abcd", **metrics(y, p1, w)})
    comp = d.loc[te.index, "abcd_complete"].to_numpy()
    if comp.any() and (w[comp] * y[comp]).sum() > 0:
        out.append({"tag": tag, "model": "abcd2_partial_alone_complete_studies", **metrics(y[comp], np.clip(d.loc[te.index, "abcd2_partial"].to_numpy()[comp] / 3 * 1e-3, 1e-7, 1), w[comp])})
        out[-1]["o_e"] = np.nan; out[-1]["cal_slope"] = np.nan    # a points score is not a probability
    fitd = fit_points(tr); p2, _ = score_predict(fitd, te); r = {"tag": tag, "model": "proposed_points_score", **metrics(y, p2, w)}
    p2tr, _ = score_predict(fitd, tr); _, ytr, wtr = XY(tr)
    for tn, t in thresholds(p2tr, ytr, wtr).items():
        c = conf(p2, y, w, t); r.update({f"{tn}_{k}": c[k] for k in ["sensitivity", "p_sens_vs_chance", "specificity", "accuracy", "ppv", "p_ppv_vs_prevalence", "npv", "lr_pos", "net_benefit"]})
    r["points"] = json.dumps(fitd[2].to_dict()); r["betas"] = json.dumps(fitd[1].round(4).to_dict()); out.append(r)
    # literature-informed score: expected points per arm, recalibrated (intercept + slope) on training data only
    st, se = lit_score(tr), lit_score(te); ytr, wtr = tr.y.to_numpy(), tr.n.to_numpy().astype(float)
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(st.to_frame(), ytr, sample_weight=wtr)
    p3 = cal.predict_proba(se.to_frame())[:, 1]; p3tr = cal.predict_proba(st.to_frame())[:, 1]
    r3 = {"tag": tag, "model": "proposed_literature_score", **metrics(y, p3, w)}
    for tn, t in thresholds(p3tr, ytr, wtr).items():
        c = conf(p3, y, w, t); r3.update({f"{tn}_{k}": c[k] for k in ["sensitivity", "p_sens_vs_chance", "specificity", "accuracy", "ppv", "p_ppv_vs_prevalence", "npv", "lr_pos", "net_benefit"]})
    out.append(r3)
    return out


if __name__ == "__main__":
    rows = []
    tr, te = holdout_10_90(ipd, seed=2026); rows += evaluate("holdout_10_90", tr, te)
    rows += [r for ch in Parallel(n_jobs=4)(delayed(evaluate)(f"fold{f}", tr, te) for f, tr, te in kfold_10pct(ipd, k=10, seed=7)) for r in ch]
    rows += [r for ch in Parallel(n_jobs=4)(delayed(evaluate)(f"loso_{s}", ipd[ipd.study != s], ipd[ipd.study == s]) for s in sorted(ipd.study.unique())) for r in ch]
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/abcd_all.csv", index=False)
    kf = df[df.tag.str.startswith("fold")]
    def paired(a, b):
        x = kf[kf.model == a].set_index("tag").auroc; z = kf[kf.model == b].set_index("tag").auroc.reindex(x.index); dd = (x - z).dropna(); v = dd.var(ddof=1)
        return dd.mean(), (2 * stats.t.sf(abs(dd.mean() / np.sqrt((1 / len(dd) + 9) * v)), len(dd) - 1) if v > 0 else 1.0)
    d_abcd, p_abcd = paired("xgb_plus_abcd", "xgb_base"); d_pts, p_pts = paired("proposed_points_score", "xgb_base"); d_lit, p_lit = paired("proposed_literature_score", "xgb_base")
    # final proposed score fitted on ALL data (for the points table), with risk by total points
    fitd = fit_points(ipd); pts = fitd[2]; _, s_all = score_predict(fitd, ipd)
    tab = pd.DataFrame({"points": s_all, "y": ipd.y, "n": ipd.n}).groupby("points").apply(lambda g: pd.Series({"patients": g.n.sum(), "events": (g.n * g.y).sum()}))
    tab["observed_risk_per_100k"] = 1e5 * tab.events / tab.patients
    tab["predicted_risk_per_100k"] = 1e5 * fitd[3].predict_proba(tab.index.to_frame(name=0).rename(columns={0: "points"}).set_axis([0], axis=1) if False else pd.DataFrame({0: tab.index.astype(float)}))[:, 1]
    tab.reset_index().to_csv(f"{OUT}/points_to_risk.csv", index=False)
    # SHAP of ABCD features in the augmented XGB (hold-out)
    m, _ = xgb_on(COLS + ABCD_COLS + ["abcd_A_missing", "abcd_B_missing"], tr, te)
    sv = shap.TreeExplainer(m).shap_values(XA.loc[te.index, COLS + ABCD_COLS + ["abcd_A_missing", "abcd_B_missing"]])
    shp = pd.Series(np.average(np.abs(sv), axis=0, weights=te.n), index=COLS + ABCD_COLS + ["abcd_A_missing", "abcd_B_missing"]).sort_values(ascending=False)
    summ = {"derivability": {"A_age_ge60": "7/11 studies (expected share from mean age, SD 12 assumed)", "B_bp_ge140_90": "5/11 studies (% hypertension proxy)",
                             "D_diabetes": "11/11 (population)", "C_clinical, D_duration, dual_TIA, imaging": "not derivable / not applicable to NAION"},
            "kfold": kf.groupby("model")[["auroc", "o_e", "cal_slope", "brier"]].agg(["mean", "std"]).round(5).pipe(lambda x: x.set_axis(["_".join(c) for c in x.columns], axis=1)).reset_index().to_dict("records"),
            "abcd_added_to_xgb": {"diff": d_abcd, "p": p_abcd}, "points_score_vs_xgb": {"diff": d_pts, "p": p_pts},
            "loso_auroc": df[df.tag.str.startswith("loso")].groupby("model").auroc.mean().round(4).to_dict(),
            "data_fitted_score_points": pts.to_dict(), "data_fitted_score_betas": fitd[1].round(4).to_dict(),
            "data_fitted_score_note": "coefficients carry implausible signs because items are study-level (ecological fallacy); shown as a negative example",
            "literature_score_points": LIT_PTS, "literature_score_sources": {k: v[1] for k, v in {**LIT, **GLP}.items()},
            "literature_score_ratios": {k: v[0] for k, v in {**LIT, **GLP}.items()}, "literature_score_vs_xgb": {"diff": d_lit, "p": p_lit},
            "literature_score_threshold_kfold": kf[kf.model == "proposed_literature_score"][[c for c in kf.columns if c.startswith("T2_top10pct_")]].mean().round(5).to_dict(),
            "cdr_evidence": {"pre_glp1ra_baseline_cdr_available": False, "pre_naion_cdr_available_in_unexposed_naion": False,
                             "unexposed_naion_note": "classic (PMIDs 6514298, 3684223) and recent (38211825, 42556433) studies all measured CDR after the event (fellow eye, or affected eye after edema); limitation is symmetric between exposed and unexposed NAION",
                             "source": "Disc-at-Risk Study, PMID 42556433: unaffected fellow eyes of 2,169 NAION patients (117 semaglutide, 83 other GLP-1RA, 1,612 unexposed) vs 465 controls",
                             "timing": "measured at/after NAION diagnosis (post-exposure for GLP-1RA users); disc anatomy assumed stable but this is weaker than a pre-treatment baseline",
                             "odds_ratios": {"CDR<=0.20": "46.3 (22.9-93.4)", "CDR 0.21-0.40": "6.8 (3.5-13.2)", "reference": "CDR>0.40"},
                             "glp1ra_difference": "no significant difference in fellow-eye CDR between semaglutide, other GLP-1RA and unexposed NAION",
                             "use_here": "literature-weighted item in the proposed score; cannot be validated in cohort data (not reported)"},
            "partial_score_limitations": ["only A, B and D of ABCD2/ABCD3-I are derivable; C, duration, dual-TIA and imaging items are TIA-specific",
                                          "A and B are study-level proxies (age from study mean with assumed SD 12; hypertension diagnosis for BP >= 140/90)",
                                          "A available for 7/11 and B for 5/11 studies; remaining values imputed and flagged",
                                          "both arms of a study share the same A and B, so the score cannot separate exposed from unexposed patients within a study",
                                          "any apparent importance reflects between-setting differences, not individual patient risk"],
            "proposed_score_threshold_kfold": kf[kf.model == "proposed_points_score"][[c for c in kf.columns if c.startswith("T2_top10pct_")]].mean().round(5).to_dict(),
            "shap_with_abcd": shp.round(5).head(12).to_dict(),
            "note": "Proposed points score is a research prototype derived from study-level data; crowded disc/cup-to-disc ratio to be added prospectively.",
            "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ["kfold", "abcd_added_to_xgb", "points_score_vs_xgb", "literature_score_vs_xgb", "loso_auroc", "literature_score_points", "data_fitted_score_betas"]}, indent=1, default=lambda v: round(float(v), 4)))
    print(tab.round(2).to_string())

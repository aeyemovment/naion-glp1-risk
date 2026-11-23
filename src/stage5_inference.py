"""Stage 5 - statistical inference (p-values) for risk factors, SHAP-flagged features,
demographics and ML experiments. Benjamini-Hochberg FDR within each family of tests.

A. Pseudo-IPD (patient-level, published counts): weighted logistic GLM
   A1 marginal model, cluster-robust SEs by study (11 clusters)
   A2 within-study exposure effect: study fixed effects + GLP-1RA exposure (+ exposure x drug / indication)
B. SHAP permutation test: XGBoost refit on outcome labels permuted within study (B=500);
   p = P(null mean|SHAP| >= observed) per feature
C. Random-effects meta-analysis (DerSimonian-Laird) of published ratio estimates + subgroup tests
D. Demographics: case reports vs exposed cohorts (sex, age); pharmacovigilance sex signal; meta-analytic age/sex subgroups
E. ML experiments: AUROC vs 0.5 (Hanley-McNeil), model-vs-model (Nadeau-Bengio corrected resampled t-test
   on the 10 x 10% folds), training-fraction trend (Spearman / linear on repeats)

research_prototype=true; synthetic_only=false; compliance_ref=Addendum_5_DT9. NOT for clinical use.
Associations from published aggregate data are ecological at the study level and are not causal.
"""
import json, os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from xgboost import XGBClassifier
import shap
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
OUT = "results/stage5"; os.makedirs(OUT, exist_ok=True)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def fmt(df, fam):
    df = df.copy(); df["family"] = fam; df["q_bh"] = bh(df.p.fillna(1)); return df


ipd = pd.read_csv("data/processed/pseudo_ipd.csv", dtype={"study": str})
tables = []

# ---------------- A1 marginal weighted logistic, cluster-robust by study
X = pd.get_dummies(ipd[["therapy", "indication", "region", "source_type", "outcome_def"]], drop_first=False, dtype=float)
ref_drop = ["therapy_no_glp1ra", "indication_t2d", "region_us", "source_type_ehr_network", "outcome_def_naion"]
X = X.drop(columns=[c for c in ref_drop if c in X])
X["age_per_10y"] = ipd.mean_age / 10; X["female_per_10pct"] = ipd.pct_female / 10; X["followup_years"] = ipd.followup_years
X = sm.add_constant(X)
# drop columns perfectly collinear with others (study-level design) via QR rank check
keep = []
for c in X.columns:
    if np.linalg.matrix_rank(X[keep + [c]].to_numpy()) > len(keep): keep.append(c)
X = X[keep]
groups = ipd.study.astype("category").cat.codes
m1 = sm.GLM(ipd.y, X, family=sm.families.Binomial(), freq_weights=ipd.n).fit(cov_type="cluster", cov_kwds={"groups": groups})
a1 = pd.DataFrame({"term": m1.params.index, "OR": np.exp(m1.params), "ci_lo": np.exp(m1.conf_int()[0]), "ci_hi": np.exp(m1.conf_int()[1]), "p": m1.pvalues})
a1 = a1[a1.term != "const"]; a1["model"] = "A1 marginal logistic (cluster-robust SE, 11 studies)"
tables.append(fmt(a1, "A1_marginal_risk_factors"))
dropped = sorted(set(pd.get_dummies(ipd[["therapy", "indication", "region", "source_type", "outcome_def"]]).columns) - set(X.columns) - set(ref_drop))

# ---------------- A2 within-study exposure effect (study fixed effects)
FE = pd.get_dummies(ipd.study, prefix="study", drop_first=True, dtype=float)
a2rows = []
def fe_fit(extra, label):
    Xf = sm.add_constant(pd.concat([FE, extra], axis=1))
    m = sm.GLM(ipd.y, Xf, family=sm.families.Binomial(), freq_weights=ipd.n).fit()
    for t in extra.columns:
        a2rows.append({"term": t, "OR": np.exp(m.params[t]), "ci_lo": np.exp(m.conf_int().loc[t, 0]), "ci_hi": np.exp(m.conf_int().loc[t, 1]), "p": m.pvalues[t], "model": label})
fe_fit(ipd[["glp1ra_exposed"]].astype(float), "A2 study fixed effects: any GLP-1RA vs comparator")
drug = pd.DataFrame({f"exposed_{d}": ((ipd.glp1ra_exposed == 1) & (ipd.therapy == d)).astype(float) for d in ["semaglutide", "glp1ra_class", "sema_or_tirz"]})
fe_fit(drug, "A2 study FE: exposure by drug")
ind = pd.DataFrame({f"exposed_{i}": ((ipd.glp1ra_exposed == 1) & (ipd.indication == i)).astype(float) for i in ["t2d", "obesity", "iih"]})
fe_fit(ind, "A2 study FE: exposure by indication")
cmp_cls = ipd.groupby("study").apply(lambda g: g.loc[g.glp1ra_exposed == 0, "therapy"].iloc[0])
cmp = pd.DataFrame({f"exposed_vs_{c}": ((ipd.glp1ra_exposed == 1) & ipd.study.map(cmp_cls).eq(c)).astype(float) for c in sorted(cmp_cls.unique())})
fe_fit(cmp, "A2 study FE: exposure by comparator class")
# effect modification by study-level demographics (exposure x age, exposure x female)
em = pd.DataFrame({"glp1ra_exposed": ipd.glp1ra_exposed.astype(float),
                   "exposed_x_age_per_10y": ipd.glp1ra_exposed * (ipd.mean_age - ipd.mean_age.mean()) / 10,
                   "exposed_x_female_per_10pct": ipd.glp1ra_exposed * (ipd.pct_female - ipd.pct_female.mean()) / 10,
                   "exposed_x_followup_years": ipd.glp1ra_exposed * (ipd.followup_years - ipd.followup_years.mean())})
fe_fit(em, "A2 study FE: exposure effect modification (study-level age, % female, follow-up)")
tables.append(fmt(pd.DataFrame(a2rows), "A2_within_study_exposure"))

# ---------------- B. SHAP permutation test
from build_ipd import FEATURES_CAT, FEATURES_NUM
NUMS = FEATURES_NUM + ["followup_years_missing", "mean_age_missing", "pct_female_missing"]
XS = pd.get_dummies(ipd[FEATURES_CAT], dtype=float).join(ipd[NUMS].astype(float))
GROUP = {c: next((f for f in FEATURES_CAT if c.startswith(f + "_")), c.replace("_missing", "")) for c in XS.columns}


def shap_imp(yv, wv, seed=0):
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist",
                      base_score=float(np.average(yv, weights=wv)), verbosity=0, n_jobs=1, random_state=seed)
    m.fit(XS, yv, sample_weight=wv)
    sv = shap.TreeExplainer(m).shap_values(XS)
    return pd.Series(np.average(np.abs(sv), axis=0, weights=wv), index=XS.columns).groupby(GROUP).sum()


# aggregate rows -> per (pattern) counts; permutation of outcomes within study = hypergeometric reshuffle of events across the study's arms
obs = shap_imp(ipd.y.to_numpy(), ipd.n.to_numpy().astype(float))
arms = ipd.groupby(["study", "glp1ra_exposed"]).apply(lambda g: pd.Series({"N": g.n.sum(), "E": g.loc[g.y == 1, "n"].sum()})).reset_index()


def perm(b):
    rng = np.random.default_rng(10_000 + b)
    w = ipd.n.to_numpy().astype(np.int64).copy()
    for s, g in arms.groupby("study"):
        E = int(g.E.sum()); Ns = g.N.to_numpy().astype(np.int64)
        e_new = rng.multivariate_hypergeometric(Ns, E)          # events re-dealt to arms at random within study
        for (_, r), e in zip(g.iterrows(), e_new):
            m1_ = (ipd.study == s) & (ipd.glp1ra_exposed == r.glp1ra_exposed)
            w[(m1_ & (ipd.y == 1)).to_numpy()] = e; w[(m1_ & (ipd.y == 0)).to_numpy()] = r.N - e
    return shap_imp(ipd.y.to_numpy(), w.astype(float), seed=b)


def perm_global(b):
    """Null for study-level features: events re-dealt across ALL arms (breaks study-level associations)."""
    rng = np.random.default_rng(20_000 + b)
    E = int(arms.E.sum()); e_new = rng.multivariate_hypergeometric(arms.N.to_numpy().astype(np.int64), E)
    w = ipd.n.to_numpy().astype(np.int64).copy()
    for (_, r), e in zip(arms.iterrows(), e_new):
        m1_ = (ipd.study == r.study) & (ipd.glp1ra_exposed == r.glp1ra_exposed)
        w[(m1_ & (ipd.y == 1)).to_numpy()] = e; w[(m1_ & (ipd.y == 0)).to_numpy()] = r.N - e
    return shap_imp(ipd.y.to_numpy(), w.astype(float), seed=b)


B = 500
null_w = pd.DataFrame(Parallel(n_jobs=4)(delayed(perm)(b) for b in range(B)))
null_g = pd.DataFrame(Parallel(n_jobs=4)(delayed(perm_global)(b) for b in range(B)))
brows = []
for f in obs.index:
    brows.append({"term": f, "obs_mean_abs_shap": obs[f],
                  "null_within_study_mean": null_w[f].mean(), "p_within_study": (1 + (null_w[f] >= obs[f]).sum()) / (B + 1),
                  "null_global_mean": null_g[f].mean(), "p": (1 + (null_g[f] >= obs[f]).sum()) / (B + 1),
                  "model": f"B XGBoost SHAP permutation test (B={B}); p = global null, p_within_study = exposure-only null"})
bt = fmt(pd.DataFrame(brows), "B_shap_permutation"); bt["q_bh_within_study"] = bh(bt.p_within_study)
tables.append(bt)

# ---------------- C. random-effects meta-analysis + subgroup tests
co = pd.read_json("data/extracted/cohort_level.json")
mm = co[co.effect_measure.fillna("").str.contains(r"HR|OR|RR|IRR", regex=True) & co.effect_estimate.gt(0) & co.ci_low.gt(0) & co.ci_high.gt(0)].copy()
mm["y"] = np.log(mm.effect_estimate); mm["se"] = (np.log(mm.ci_high) - np.log(mm.ci_low)) / 3.92
mm = mm[mm.se > 0]
prim = mm.loc[mm.groupby("source_pmid").se.idxmin()]           # one (most precise) estimate per study


def dl(y, se):
    w = 1 / se ** 2; fe = np.sum(w * y) / w.sum(); Q = np.sum(w * (y - fe) ** 2); k = len(y)
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); wr = 1 / (se ** 2 + tau2)
    mu = np.sum(wr * y) / wr.sum(); s = np.sqrt(1 / wr.sum()); z = mu / s
    return {"k": k, "pooled_ratio": np.exp(mu), "ci_lo": np.exp(mu - 1.96 * s), "ci_hi": np.exp(mu + 1.96 * s),
            "p": 2 * stats.norm.sf(abs(z)), "I2": max(0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0, "tau2": tau2,
            "Q_p": stats.chi2.sf(Q, k - 1), "mu": mu, "s": s}


crows = [{"term": "all studies (most precise estimate each)", **dl(prim.y.to_numpy(), prim.se.to_numpy())}]
for col, lab in [("population", "indication"), ("exposure_drug", "drug")]:
    subs = {}
    for g, d in prim.groupby(col):
        if len(d) >= 2: r = dl(d.y.to_numpy(), d.se.to_numpy()); subs[g] = r; crows.append({"term": f"{lab} = {g}", **r})
    if len(subs) >= 2:
        mus = np.array([v["mu"] for v in subs.values()]); ss = np.array([v["s"] for v in subs.values()])
        w = 1 / ss ** 2; Qb = np.sum(w * (mus - np.sum(w * mus) / w.sum()) ** 2)
        crows.append({"term": f"subgroup difference by {lab}", "p": stats.chi2.sf(Qb, len(subs) - 1), "k": len(subs)})
# leave-out of IIH (protective, different population)
d = prim[prim.population != "iih"]; crows.append({"term": "excluding IIH cohort", **dl(d.y.to_numpy(), d.se.to_numpy())})
ct = pd.DataFrame(crows).drop(columns=["mu", "s"], errors="ignore"); ct["model"] = "C DerSimonian-Laird random effects"
tables.append(fmt(ct, "C_meta_analysis"))

# ---------------- D. demographics
cases = pd.read_json("data/extracted/case_level.json"); cases = cases[cases.diagnosis == "naion"]
expo = co.dropna(subset=["pct_female_exposed"])
pf_pop = np.average(expo.pct_female_exposed, weights=expo.n_exposed.fillna(expo.n_exposed.median())) / 100
males = int((cases.sex == "M").sum()); ns = int(cases.sex.notna().sum())
drows = [{"term": "case reports: male share vs exposed-cohort male share", "observed": males / ns, "expected": 1 - pf_pop,
          "p": stats.binomtest(males, ns, 1 - pf_pop).pvalue, "n": ns, "test": "exact binomial"}]
ages = cases.age.dropna(); age_pop = np.average(co.mean_age_exposed.dropna(), weights=co.loc[co.mean_age_exposed.notna(), "n_exposed"].fillna(1))
drows.append({"term": "case reports: age vs exposed-cohort mean age", "observed": ages.median(), "expected": age_pop,
              "p": stats.wilcoxon(ages - age_pop).pvalue, "n": len(ages), "test": "Wilcoxon signed-rank vs cohort mean"})
ob = cases.indication.isin(["obesity", "both"]).sum(); ni = cases.indication.notna().sum()
drows.append({"term": "case reports: obesity indication share vs 50%", "observed": ob / ni, "expected": 0.5, "p": stats.binomtest(int(ob), int(ni), 0.5).pvalue, "n": int(ni), "test": "exact binomial"})
rm = pd.read_json("data/extracted/rct_meta.json")
def z_from_ci(est, lo, hi): se = (np.log(hi) - np.log(lo)) / 3.92; return np.log(est), se
def diff_p(a, b):
    ya, sa = z_from_ci(*a); yb, sb = z_from_ci(*b); z = (ya - yb) / np.sqrt(sa ** 2 + sb ** 2); return 2 * stats.norm.sf(abs(z)), np.exp(ya - yb)
for lab, s1, s2 in [("meta-analysis 42612051: age <65 vs >=65", "age <65", "age >=65"),
                    ("meta-analysis 42612051: female- vs male-dominant cohorts", "sex: female-dominant studies", "sex: male-dominant studies"),
                    ("meta-analysis 42612051: diabetes vs obesity indication", "indication: diabetes", "indication: overweight/obesity")]:
    r1 = rm[(rm.source_pmid == 42612051) & (rm.subgroup_label == s1)].iloc[0]; r2 = rm[(rm.source_pmid == 42612051) & (rm.subgroup_label == s2)].iloc[0]
    p, rr = diff_p((r1.pooled_estimate, r1.ci_low, r1.ci_high), (r2.pooled_estimate, r2.ci_low, r2.ci_high))
    drows.append({"term": lab, "observed": rr, "expected": 1.0, "p": p, "test": "z-test of ratio of ratios (from published CIs)"})
pv = pd.read_json("data/extracted/pharmacovigilance.json")
male = pv[pv.subgroup_label.fillna("").str.contains("male sex")]
if len(male) and pd.notna(male.iloc[0].get("ROR_ci_low")):
    r = male.iloc[0]; y, se = z_from_ci(r.ROR, r.ROR_ci_low, r.ROR_ci_high)
    drows.append({"term": "FAERS 41807083: semaglutide NAION ROR in men", "observed": r.ROR, "expected": 1.0, "p": 2 * stats.norm.sf(abs(y / se)), "test": "z from published CI"})
tables.append(fmt(pd.DataFrame(drows), "D_demographics"))

# ---------------- E. ML experiments
s1 = json.load(open("results/stage1/stage1_results.json"))
erows = []
for mdl, r in s1["holdout_10_90"].items():
    A, n1, n0 = r["auroc"], r["events_test"], r["n_test"] - r["events_test"]
    q1, q2 = A / (2 - A), 2 * A ** 2 / (1 + A)
    se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A ** 2) + (n0 - 1) * (q2 - A ** 2)) / (n1 * n0))
    erows.append({"term": f"{mdl}: AUROC > 0.5 (10/90 hold-out)", "estimate": A, "p": stats.norm.sf((A - 0.5) / se), "test": "Hanley-McNeil one-sided"})
kf = {m: pd.DataFrame(v).auroc.to_numpy() for m, v in s1["kfold_10pct"].items()}
for a, b in [("xgb", "logreg"), ("xgb", "rf"), ("rf", "logreg")]:
    d = kf[a] - kf[b]; k = len(d); ratio = 0.9 / 0.1          # n_test / n_train
    t = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1))
    erows.append({"term": f"{a} vs {b}: AUROC difference (10 x 10% folds)", "estimate": d.mean(), "p": 2 * stats.t.sf(abs(t), k - 1), "test": "Nadeau-Bengio corrected resampled t"})
for m, v in s1["loso"].items():
    a = pd.DataFrame(v).auroc
    erows.append({"term": f"{m}: leave-one-study-out AUROC > 0.5", "estimate": a.mean(), "p": stats.wilcoxon(a - 0.5, alternative="greater").pvalue, "test": "Wilcoxon over 11 held-out studies"})
if os.path.exists("results/stage1c/train_fraction_sweep.csv"):
    sw = pd.read_csv("results/stage1c/train_fraction_sweep.csv")
    for m, g in sw.groupby("model"):
        rho, p = stats.spearmanr(g.train_frac, g.auroc)
        erows.append({"term": f"{m}: AUROC trend with training fraction (10-90%)", "estimate": rho, "p": p, "test": "Spearman (90 runs)"})
if os.path.exists("results/stage2/stage2_summary.json"):
    s2 = json.load(open("results/stage2/stage2_summary.json"))
    for fam, r in [("hybrid best", s2["selected_top3_for_app"][2]), ("synthetic-only best", s2["selected_top3_for_app"][1])]:
        se = (r["d_auroc_hi"] - r["d_auroc_lo"]) / 3.92
        erows.append({"term": f"Stage 2 {fam} vs PubMed-only (XGB, paired)", "estimate": r["d_auroc"], "p": 2 * stats.norm.sf(abs(r["d_auroc"] / se)) if se > 0 else np.nan, "test": "paired z over 27 split/seed pairs"})
tables.append(fmt(pd.DataFrame(erows), "E_ml_experiments"))

allt = pd.concat(tables, ignore_index=True)
for c in ["research_prototype", "synthetic_only"]: allt[c] = c == "research_prototype"
allt.to_csv(f"{OUT}/pvalues_all.csv", index=False)
json.dump({"collinear_terms_dropped_from_A1": dropped, "note": "Associations are ecological (study-level covariates) and not causal. Within-study exposure (A2) is the least confounded estimate."},
          open(f"{OUT}/notes.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
for fam, g in allt.groupby("family", sort=False):
    cols = [c for c in ["term", "OR", "ci_lo", "ci_hi", "pooled_ratio", "obs_mean_abs_shap", "p_within_study", "observed", "expected", "estimate", "k", "I2", "p", "q_bh"] if c in g and g[c].notna().any()]
    print(f"\n== {fam}"); print(g[cols].round(4).to_string(index=False))
print("\ndropped (collinear) from A1:", dropped)

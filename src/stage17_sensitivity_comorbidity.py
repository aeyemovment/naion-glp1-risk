"""Stage 17 - sensitivity analysis restricted to cohorts that reported vascular/NAION risk-factor prevalence.

Question: does the GLP-1RA-NAION association hold in the better-characterized, risk-factor-matched cohorts?
Subsets: all 11 cohorts; cohorts reporting hypertension prevalence; reporting hypertension + obstructive sleep apnea;
1:1 or 1:2 propensity-score-matched cohorts.
Per subset: within-study OR (count-weighted logistic GLM with study fixed effects, as Stage 5 A2), Mantel-Haenszel OR,
between-study heterogeneity (Cochran Q, I2) of per-study ORs, and XGBoost AUROC over 10 disjoint 10% training folds.
Comorbidity prevalences are study-level (one value per cohort, arms balanced by matching) so they cannot be modelled
per patient; this analysis tests robustness of the exposure effect only.
research_prototype=true; synthetic_only=false; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, json, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import sys; sys.path.insert(0, "src")
from common import ipd, fit, predict
from splits import kfold_10pct
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
OUT = "results/stage17"; os.makedirs(OUT, exist_ok=True)
ipd = ipd.copy(); ipd["study"] = ipd.study.astype(str)

c = pd.DataFrame(json.load(open("data/extracted/cohort_level.json"))); c["pm"] = c.source_pmid.astype(str)
c = c[c.pm.isin(ipd.study.unique())]
rep = c.groupby("pm")[["pct_hypertension", "pct_osa", "pct_hyperlipidemia", "pct_cvd"]].agg(lambda s: s.notna().any())
match = c.groupby("pm").matching.agg(lambda s: next((x for x in s if isinstance(x, str)), ""))
adj = c.groupby("pm").adjusted.agg(lambda s: bool(s.fillna(False).any()))
prev = c.groupby("pm")[["pct_hypertension", "pct_osa"]].max()
S = sorted(ipd.study.unique())
SUBSETS = {
    "All cohorts": S,
    "Reported hypertension prevalence": [s for s in S if rep.loc[s, "pct_hypertension"]],
    "Reported hypertension and sleep apnea prevalence": [s for s in S if rep.loc[s, "pct_hypertension"] and rep.loc[s, "pct_osa"]],
    "1:1 or 1:2 propensity-score matched": [s for s in S if "PSM" in match.get(s, "")],
}

def within_or(d):
    FE = pd.get_dummies(d.study, prefix="s", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([FE, d[["glp1ra_exposed"]].astype(float)], axis=1))
    m = sm.GLM(d.y, X, family=sm.families.Binomial(), freq_weights=d.n).fit()
    ci = m.conf_int().loc["glp1ra_exposed"]
    return np.exp(m.params["glp1ra_exposed"]), np.exp(ci[0]), np.exp(ci[1]), m.pvalues["glp1ra_exposed"]

def per_study(d):
    rows = []
    for s, g in d.groupby("study"):
        a = g[(g.glp1ra_exposed == 1) & (g.y == 1)].n.sum(); b = g[(g.glp1ra_exposed == 1) & (g.y == 0)].n.sum()
        cc = g[(g.glp1ra_exposed == 0) & (g.y == 1)].n.sum(); dd = g[(g.glp1ra_exposed == 0) & (g.y == 0)].n.sum()
        rows.append((s, a, b, cc, dd))
    return pd.DataFrame(rows, columns=["study", "a", "b", "c", "d"])

def mh_and_het(t):
    N = t[["a", "b", "c", "d"]].sum(axis=1)
    mh = (t.a * t.d / N).sum() / (t.b * t.c / N).sum()
    tt = t[["a", "b", "c", "d"]] + 0.5
    lor = np.log(tt.a * tt.d / (tt.b * tt.c)); v = 1 / tt.a + 1 / tt.b + 1 / tt.c + 1 / tt.d; w = 1 / v
    pooled = (w * lor).sum() / w.sum(); Q = (w * (lor - pooled) ** 2).sum(); k = len(t)
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    return mh, Q, stats.chi2.sf(Q, k - 1) if k > 1 else np.nan, I2

def kfold_auc(d):
    out = []
    for f, tr, te in kfold_10pct(d, k=10, seed=11):
        m = fit("xgb", tr); p = predict("xgb", m, te)
        out.append(roc_auc_score(te.y, p, sample_weight=te.n))
    return np.mean(out), np.std(out)

rows = []
for name, studies in SUBSETS.items():
    d = ipd[ipd.study.isin(studies)]
    OR, lo, hi, p = within_or(d); mh, Q, pQ, I2 = mh_and_het(per_study(d)); auc, sd = kfold_auc(d)
    hp = prev.loc[[s for s in studies if s in prev.index], "pct_hypertension"].dropna()
    op = prev.loc[[s for s in studies if s in prev.index], "pct_osa"].dropna()
    rows.append({"subset": name, "cohorts": len(studies), "patients": int(d.n.sum()), "events": int(d[d.y == 1].n.sum()),
                 "OR": OR, "ci_lo": lo, "ci_hi": hi, "p": p, "MH_OR": mh, "Q_p": pQ, "I2": I2, "xgb_auroc": auc, "xgb_auroc_sd": sd,
                 "htn_range": f"{hp.min():.0f}-{hp.max():.0f}%" if len(hp) else "", "osa_range": f"{op.min():.0f}-{op.max():.0f}%" if len(op) else "",
                 "studies": ";".join(studies)})
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/sensitivity_comorbidity.csv", index=False)
json.dump({"design": __doc__, "results": R.round(6).to_dict("records"), "research_prototype": True, "synthetic_only": False,
           "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); print(R.drop(columns="studies").round(4).to_string(index=False))

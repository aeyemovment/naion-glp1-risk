"""Stage 18 - which optional patient factors, and how many, add risk prediction to the core calculator model.

Core model: mean of XGBoost and L2 logistic-regression probabilities (the paper's mean ensemble), trained on each of the
ten disjoint 10% folds and tested on the remaining 90% (same splits as Stage 1).
Candidates: every patient-level factor with a verified published ratio - male sex, hypertension, hyperlipidaemia, coronary
heart disease, obstructive sleep apnea (Liu 2021, PMID 34671609) and cup-to-disc ratio (PMID 42556433). Diabetes is not a
candidate because the indication input already fixes it.
Because the published cohorts report none of these per patient, each factor is assigned to the test patients conditional on
their actual outcome (P(x | no NAION) = training-cohort reference prevalence; P(x | NAION) from the published OR), independently
across factors given the outcome - the Stage 16 design. Assignment is EXACT (every factor combination carried as a weighted
row), so no random seed is involved. Score = logit(core) + sum ln(OR_j) (x_j - ref_j), the calculator formula.
Selection: greedy forward selection by AUROC gain; a factor is kept if its marginal gain is >= 0.005 AUROC and the paired
gain over the ten folds is significant after Benjamini-Hochberg correction. synthetic_only=true (factor values are
simulated); research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, json, itertools, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import sys; sys.path.insert(0, "src")
from common import ipd, fit, predict, XY
from splits import kfold_10pct
OUT = "results/stage18"; os.makedirs(OUT, exist_ok=True)

co = pd.read_json("data/extracted/cohort_level.json")
ref = lambda c: float(co.groupby(co.source_pmid.astype(str))[c].max().median()) / 100
CAND = {  # name: (levels [(OR, P(level | no NAION))], label)
    "male_sex": ([(1.67, 1 - ref("pct_female_exposed")), (1.0, ref("pct_female_exposed"))], "Male sex"),
    "hypertension": ([(1.28, ref("pct_hypertension")), (1.0, 1 - ref("pct_hypertension"))], "Hypertension"),
    "hyperlipidaemia": ([(1.43, ref("pct_hyperlipidemia")), (1.0, 1 - ref("pct_hyperlipidemia"))], "Hyperlipidaemia"),
    "coronary_heart_disease": ([(1.68, ref("pct_cvd")), (1.0, 1 - ref("pct_cvd"))], "Coronary heart disease"),
    "sleep_apnoea": ([(3.28, ref("pct_osa")), (1.0, 1 - ref("pct_osa"))], "Obstructive sleep apnea"),
    "cup_to_disc": ([(46.3, 0.09), (6.8, 0.47), (1.0, 0.44)], "Cup-to-disc ratio (3 levels)"),
}


def levels(name):
    lv, _ = CAND[name]; p0 = np.array([q for _, q in lv]); orr = np.array([o for o, _ in lv])
    p1 = p0 * orr / (p0 * orr).sum()                          # P(level | NAION), rare-disease approximation
    beta = np.log(orr) - (p0 * np.log(orr)).sum()               # centred contribution, as in the calculator
    return p0, p1, beta


def auroc(core_logit, y, w, factors):
    if not factors: return roc_auc_score(y, core_logit, sample_weight=w)
    L = [levels(f) for f in factors]
    S, Y, W = [], [], []
    for combo in itertools.product(*[range(len(l[0])) for l in L]):
        add = sum(l[2][k] for l, k in zip(L, combo))
        pr0 = np.prod([l[0][k] for l, k in zip(L, combo)]); pr1 = np.prod([l[1][k] for l, k in zip(L, combo)])
        S.append(core_logit + add); Y.append(y); W.append(w * np.where(y == 1, pr1, pr0))
    return roc_auc_score(np.concatenate(Y), np.concatenate(S), sample_weight=np.concatenate(W))


folds = []
for f, tr, te in kfold_10pct(ipd):
    p = (predict("xgb", fit("xgb", tr), te) + predict("logreg", fit("logreg", tr), te)) / 2
    _, y, w = XY(te); folds.append((np.log(p / (1 - p)), y, w))
    print("fold", f, "core AUROC", round(roc_auc_score(y, p, sample_weight=w), 4), flush=True)

A = lambda fs: np.array([auroc(c, y, w, fs) for c, y, w in folds])
core = A([])
alone = {k: A([k]) - core for k in CAND}
chosen, path, cur = [], [], core
remaining = list(CAND)
while remaining:
    gains = {k: A(chosen + [k]) for k in remaining}
    best = max(gains, key=lambda k: gains[k].mean()); d = gains[best] - cur
    t = stats.ttest_1samp(d, 0) if d.std() > 0 else None
    path.append({"step": len(chosen) + 1, "factor": best, "label": CAND[best][1], "auroc_mean": gains[best].mean(), "auroc_sd": gains[best].std(ddof=1),
                 "marginal_gain": d.mean(), "marginal_gain_sd": d.std(ddof=1), "p": float(t.pvalue) if t else np.nan})
    chosen.append(best); remaining.remove(best); cur = gains[best]
P = pd.DataFrame(path)
p_ = P.p.fillna(1).to_numpy(); o = np.argsort(p_); n = len(p_); q = np.empty(n); q[o] = np.minimum.accumulate((p_[o] * n / np.arange(1, n + 1))[::-1])[::-1]
P["q_bh"] = np.clip(q, 0, 1)
P["kept"] = (P.marginal_gain >= 0.005) & (P.q_bh < 0.05)
# keep the prefix of the forward path that meets the rule (a later factor cannot be kept after one that failed)
P["kept"] = P.kept.cumprod().astype(bool)
P["alone_gain"] = P.factor.map(lambda k: alone[k].mean())
P.to_csv(f"{OUT}/forward_selection.csv", index=False)
summ = {"core_auroc_mean": float(core.mean()), "core_auroc_sd": float(core.std(ddof=1)), "selected": P[P.kept].factor.tolist(),
        "n_selected": int(P.kept.sum()), "auroc_all_candidates": float(P.auroc_mean.iloc[-1]),
        "auroc_selected": float(P[P.kept].auroc_mean.iloc[-1]) if P.kept.any() else float(core.mean()),
        "alone_gain": {k: float(v.mean()) for k, v in alone.items()}, "rule": "marginal gain >= 0.005 AUROC and BH q < 0.05 (paired over 10 folds)",
        "design": __doc__.split("\n\n")[0], "synthetic_only": True, "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200); print(P.round(4).to_string(index=False)); print(json.dumps({k: summ[k] for k in ["core_auroc_mean", "selected", "auroc_selected", "auroc_all_candidates"]}, indent=1))

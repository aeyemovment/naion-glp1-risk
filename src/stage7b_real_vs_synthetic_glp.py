"""Stage 7b - background (non-GLP) NAION model validated on real vs synthetic GLP-1RA users, in isolation and together.

Training: comparator (non-GLP) arms only, the same ten disjoint 10% folds as Stage 7 (kfold_10pct(nonglp, k=10, seed=7));
          features exclude exposure and drug (Stage 7 BG_FEATS). Models: L2 logistic regression, random forest, XGBoost and
          their mean ensemble (the GNN is omitted here: its graph edges need study identity, which twins do not have).
Test sets, scored by the SAME fitted models in every fold:
  A  real GLP-1RA users only (all 1,287,543 reconstructed exposed patients; never used in training)  = Stage 7 external
  B  synthetic GLP-1RA users only: Gaussian-copula twins fit to the real GLP-1RA arms (same size; new twins each fold)
  C  real + synthetic GLP-1RA users together
  D  all real patients: held-out non-GLP (90% of comparators) + real GLP-1RA users
  E  everyone: D + synthetic GLP-1RA users
  (I) internal reference: held-out non-GLP only
Metrics: AUROC (weighted), O/E. Synthetic twins are fit to the real GLP data, so B and C are not independent validation.
synthetic_only=true for twins; research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys; sys.path.insert(0, "src")
import stage2_twins_3x6x9 as S2
from common import ipd, X_all, COLS
from splits import kfold_10pct
OUT = "results/stage7b"; os.makedirs(OUT, exist_ok=True)
BG = [c for c in COLS if not (c.startswith("therapy_") or c == "glp1ra_exposed")]
nonglp, glp = ipd[ipd.glp1ra_exposed == 0], ipd[ipd.glp1ra_exposed == 1]


def fit(name, X, y, w):
    if name == "logreg":
        mu, sd = X.mean(), X.std().replace(0, 1)
        m = LogisticRegression(C=1.0, max_iter=5000).fit((X - mu) / sd, y, sample_weight=w * (np.sum(w * y) / np.sum(w)))
        return lambda Z: m.predict_proba((Z - mu) / sd)[:, 1]
    if name == "rf":
        m = RandomForestClassifier(n_estimators=300, max_depth=4, max_features="sqrt", bootstrap=False, min_weight_fraction_leaf=0.01, random_state=0, n_jobs=4).fit(X, y, sample_weight=w)
    else:
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist", base_score=float(np.average(y, weights=w)),
                          verbosity=0, n_jobs=4, random_state=0).fit(X, y, sample_weight=w)
    return lambda Z: m.predict_proba(Z)[:, 1]


def real(df): return X_all.loc[df.index, BG], df.y.to_numpy(), df.n.to_numpy().astype(float)
def synth(df): return S2.encode(df.reset_index(drop=True))[BG], df.y.to_numpy(), df.n.to_numpy().astype(float)
def cat(*parts): return pd.concat([p[0] for p in parts], ignore_index=True), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])


rows = []
for f, tr, te in kfold_10pct(nonglp, k=10, seed=7):
    rng = np.random.default_rng(7000 + f)
    prof = S2.generate("P2_copula", glp[S2.FEAT + ["y", "n"]], rng, None).sample(1000, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    tw = S2.twins_aggregated(prof, int(glp.n.sum()), rng)
    sets = {"I_internal_nonGLP": real(te), "A_real_GLP": real(glp), "B_synthetic_GLP": synth(tw)}
    sets["C_real_plus_synthetic_GLP"] = cat(sets["A_real_GLP"], sets["B_synthetic_GLP"])
    sets["D_all_real"] = cat(sets["I_internal_nonGLP"], sets["A_real_GLP"])
    sets["E_all_real_plus_synthetic_GLP"] = cat(sets["D_all_real"], sets["B_synthetic_GLP"])
    Xtr, ytr, wtr = real(tr)
    F = {m: fit(m, Xtr, ytr, wtr) for m in ["logreg", "rf", "xgb"]}
    for s, (X, y, w) in sets.items():
        P = {m: F[m](X) for m in F}; P["ensemble"] = np.mean([P[m] for m in F], 0)
        for m, p in P.items():
            rows.append({"fold": f, "test_set": s, "model": m, "auroc": roc_auc_score(y, p, sample_weight=w), "o_e": (w * y).sum() / (w * p).sum(),
                         "n": w.sum(), "events": (w * y).sum()})
    print("fold", f, flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/folds.csv", index=False)
S = D.groupby(["test_set", "model"]).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), o_e=("o_e", "mean"), o_e_sd=("o_e", "std"),
                                          n=("n", "mean"), events=("events", "mean")).reset_index()
S.to_csv(f"{OUT}/summary.csv", index=False)
json.dump({"design": __doc__.split("\n\n")[0], "summary": S.round(5).to_dict("records"), "synthetic_only_twins": True, "research_prototype": True,
           "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200); print(S.round(4).to_string(index=False))

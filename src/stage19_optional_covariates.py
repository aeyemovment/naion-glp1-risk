"""Stage 19 - optional-input model trained on the REAL (sparsely reported) cohort covariates, instead of literature odds ratios.

Question: if the calculator lets users enter any covariate that some cohorts reported, and a model trained on the actual data
for that availability pattern is used, how much does each covariate add - measured on held-out data, not simulated?
Covariates (study-level prevalences or means from data/extracted/cohort_level.json; reported by k of 11 cohorts):
  hypertension (5), sleep apnoea (4), hyperlipidaemia (6), cardiovascular disease (6), HbA1c (3), BMI (2).
Core features: the Stage 1 columns (drug, comparator, indication, age, sex share, follow-up, setting + missing flags).
Model bank: for every subset of the six covariates (64 patterns) a dedicated model is trained on core + that subset
(XGBoost with native missing handling; L2 logistic regression with median imputation + missing flags; mean ensemble).
Validation, paired against the core model on identical splits:
  (1) ten disjoint 10% training folds (patients from the same studies) and (2) leave-one-study-out (unseen setting).
Selection without optimism: forward selection on folds 0-4, keep if gain >= 0.005 AUROC and BH q < 0.05; confirm on folds 5-9 and LOSO.
Caveat: covariates are study-level, so a patient's own yes/no can only be mapped onto them ecologically.
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, json, itertools, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import sys; sys.path.insert(0, "src")
from common import ipd, X_all, COLS
from splits import kfold_10pct
OUT = "results/stage19"; os.makedirs(OUT, exist_ok=True)

co = pd.read_json("data/extracted/cohort_level.json"); co["pm"] = co.source_pmid.astype(str)
COV = {"hypertension": "pct_hypertension", "sleep_apnoea": "pct_osa", "hyperlipidaemia": "pct_hyperlipidemia",
       "cardiovascular_disease": "pct_cvd", "hba1c": "mean_hba1c", "bmi": "mean_bmi"}
study_cov = co.groupby("pm")[list(COV.values())].max()
EXTRA = pd.DataFrame({g: ipd.study.astype(str).map(study_cov[c]).to_numpy() for g, c in COV.items()}, index=ipd.index)
avail = {g: int(study_cov.loc[study_cov.index.isin(ipd.study.astype(str).unique()), c].notna().sum()) for g, c in COV.items()}
print("reported by (of 11):", avail)


def design(groups, df, impute=None):
    X = X_all.loc[df.index, COLS].copy()
    for g in groups:
        v = EXTRA.loc[df.index, g]
        X[g] = v if impute is None else v.fillna(impute[g]); X[g + "_missing"] = v.isna().astype(float)
    return X


def fit_predict(groups, tr, te):
    ytr, wtr = tr.y.to_numpy(), tr.n.to_numpy().astype(float)
    Xtr, Xte = design(groups, tr), design(groups, te)
    xg = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist",
                       base_score=float(np.average(ytr, weights=wtr)), verbosity=0, n_jobs=4, random_state=0).fit(Xtr, ytr, sample_weight=wtr)
    med = {g: float(np.nanmedian(EXTRA.loc[tr.index, g])) if EXTRA.loc[tr.index, g].notna().any() else 0.0 for g in groups}
    Ltr, Lte = design(groups, tr, med), design(groups, te, med)
    mu, sd = Ltr.mean(), Ltr.std().replace(0, 1)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit((Ltr - mu) / sd, ytr, sample_weight=wtr * (np.sum(wtr * ytr) / np.sum(wtr)))
    p = {"xgb": xg.predict_proba(Xte)[:, 1], "logreg": lr.predict_proba((Lte - mu) / sd)[:, 1]}
    p["ensemble"] = (p["xgb"] + p["logreg"]) / 2
    return p


def auc(df, p):
    y, w = df.y.to_numpy(), df.n.to_numpy().astype(float)
    return roc_auc_score(y, p, sample_weight=w) if len(np.unique(y)) > 1 else np.nan


SUBSETS = [tuple(s) for k in range(len(COV) + 1) for s in itertools.combinations(COV, k)]
folds = list(kfold_10pct(ipd))
rows = []
for s in SUBSETS:
    for f, tr, te in folds:
        for m, p in fit_predict(s, tr, te).items(): rows.append({"subset": "+".join(s) or "core", "k": len(s), "scheme": "fold", "split": f, "model": m, "auroc": auc(te, p)})
    for st in sorted(ipd.study.unique()):
        tr, te = ipd[ipd.study != st], ipd[ipd.study == st]
        for m, p in fit_predict(s, tr, te).items(): rows.append({"subset": "+".join(s) or "core", "k": len(s), "scheme": "loso", "split": st, "model": m, "auroc": auc(te, p)})
    print(s or "core", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/all_subsets.csv", index=False)

# paired gain vs core, per subset/model/scheme (LOSO: within-study AUROC is undefined where a study has one covariate pattern; nan dropped)
core = R[R.subset == "core"].set_index(["scheme", "split", "model"]).auroc
R["gain"] = R.auroc.to_numpy() - core.reindex(pd.MultiIndex.from_frame(R[["scheme", "split", "model"]])).to_numpy()


def nb_t(d, n_tr=0.1, n_te=0.9):  # Nadeau-Bengio corrected resampled t for the overlapping-fold paradigm
    d = d.dropna(); J = len(d)
    if J < 2 or d.std() == 0: return np.nan
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * d.var(ddof=1)); return float(2 * stats.t.sf(abs(t), J - 1))


S = (R.groupby(["subset", "k", "scheme", "model"]).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), gain=("gain", "mean"))
     .reset_index())
S["p_fold"] = [nb_t(R[(R.subset == r.subset) & (R.model == r.model) & (R.scheme == "fold")].gain) if r.scheme == "fold" else np.nan for r in S.itertuples()]
S.to_csv(f"{OUT}/subset_summary.csv", index=False)

# forward selection on folds 0-4 (ensemble), confirmation on folds 5-9 and LOSO
E = R[(R.model == "ensemble") & (R.scheme == "fold")]
sel_folds, conf_folds = E[E.split < 5], E[E.split >= 5]
m_ = lambda D, s: D[D.subset == ("+".join(s) or "core")].set_index("split").auroc
chosen, path = (), []
for _ in range(len(COV)):
    cand = [g for g in COV if g not in chosen]
    if not cand: break
    def key(g): return tuple(sorted(chosen + (g,), key=list(COV).index))
    best = max(cand, key=lambda g: m_(sel_folds, key(g)).mean())
    d = m_(sel_folds, key(best)) - m_(sel_folds, chosen)
    p = nb_t(d)
    path.append({"step": len(chosen) + 1, "added": best, "reported_by": avail[best], "auroc_sel": m_(sel_folds, key(best)).mean(), "gain_sel": d.mean(), "p_sel": p})
    chosen = key(best)
P = pd.DataFrame(path)
pp = P.p_sel.fillna(1).to_numpy(); o = np.argsort(pp); n = len(pp); q = np.empty(n); q[o] = np.minimum.accumulate((pp[o] * n / np.arange(1, n + 1))[::-1])[::-1]
P["q_sel"] = np.clip(q, 0, 1); P["kept"] = ((P.gain_sel >= 0.005) & (P.q_sel < 0.05)).cumprod().astype(bool)
kept = tuple(sorted(P[P.kept].added, key=list(COV).index))
conf = {"kept": list(kept)}
for scheme, D in [("folds_5_9", conf_folds), ("loso", R[(R.model == "ensemble") & (R.scheme == "loso")])]:
    a, b = m_(D, kept), m_(D, ())
    conf[scheme] = {"auroc_kept": float(a.mean()), "auroc_core": float(b.mean()), "gain": float((a - b).mean())}
allk = tuple(COV)
for scheme, D in [("folds_all", E), ("loso", R[(R.model == "ensemble") & (R.scheme == "loso")])]:
    conf[f"all_six_{scheme}"] = {"auroc_all": float(m_(D, allk).mean()), "auroc_core": float(m_(D, ()).mean()), "gain": float((m_(D, allk) - m_(D, ())).mean())}
P.to_csv(f"{OUT}/forward_selection.csv", index=False)
single = S[(S.k == 1) & (S.model == "ensemble")].pivot(index="subset", columns="scheme", values="gain").rename(columns={"fold": "gain_folds", "loso": "gain_loso"})
single["reported_by"] = single.index.map(avail); single = single.sort_values("gain_folds", ascending=False)
single.to_csv(f"{OUT}/single_covariate_gain.csv")
json.dump({"design": __doc__.split("\n\n")[0], "reported_by": avail, "single_covariate_gain": single.round(5).reset_index().to_dict("records"),
           "forward_selection": P.round(5).to_dict("records"), "confirmation": conf, "research_prototype": True, "compliance_ref": "Addendum_5_DT9"},
          open(f"{OUT}/summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(single.round(4).to_string()); print(P.round(4).to_string(index=False)); print(json.dumps(conf, indent=1, default=float))

"""Shared model/metric/SHAP code (from Stage 1). Stage 1 - PubMed-only supervised models on count-aggregated pseudo-IPD.

Validation:
  A. single 10% train / 90% held-out test
  B. 10 folds: each fold trains on a DIFFERENT disjoint random 10% of patients, tests on the other 90%
  C. leave-one-study-out (LOSO): train on all other studies, test on unseen study
Models: L2 logistic regression, random forest, XGBoost. Weighted metrics + SHAP.

research_prototype=true; synthetic_only=false; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from splits import holdout_10_90, kfold_10pct
from build_ipd import FEATURES_CAT, FEATURES_NUM

warnings.filterwarnings("ignore")
OUT = "results/stage1"; import os; os.makedirs(OUT, exist_ok=True)
C = {"logreg": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a"}
NAMES = {"logreg": "Logistic regression (L2)", "rf": "Random forest", "xgb": "XGBoost"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "grid.color": "#e4e3df"})

ipd = pd.read_csv("data/processed/pseudo_ipd.csv", dtype={"study": str})
NUMS = FEATURES_NUM + ["followup_years_missing", "mean_age_missing", "pct_female_missing"]
X_all = pd.get_dummies(ipd[FEATURES_CAT], dtype=float).join(ipd[NUMS].astype(float))
COLS = list(X_all.columns)
GROUP = {c: next((f for f in FEATURES_CAT if c.startswith(f + "_")), c) for c in COLS}
GROUP.update({c: c.replace("_missing", "") for c in COLS if c.endswith("_missing")})
MU, SD = X_all.mean(), X_all.std().replace(0, 1)


def XY(df):
    X = X_all.loc[df.index]
    return X, df.y.to_numpy(), df.n.to_numpy().astype(float)


def make(name):
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=5000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=4, max_features="sqrt", bootstrap=False,
                                      min_weight_fraction_leaf=0.01, random_state=0, n_jobs=-1)
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=1.0,
                         min_child_weight=0.5, reg_lambda=1.0, tree_method="hist", random_state=0, verbosity=0)


def fit(name, tr):
    X, y, w = XY(tr)
    m = make(name)
    Xs = (X - MU) / SD if name == "logreg" else X
    if name == "xgb":
        m.set_params(base_score=float(np.average(y, weights=w)))
    if name == "logreg":  # rescale weights to sum to #events so the L2 penalty is not swamped by millions of rows
        w = w * (np.sum(w * y) / np.sum(w))
    m.fit(Xs, y, sample_weight=w)
    return m


def predict(name, m, df):
    X, _, _ = XY(df)
    return m.predict_proba((X - MU) / SD if name == "logreg" else X)[:, 1]


def cal_slope(y, p, w):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y, sample_weight=w)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def metrics(y, p, w):
    s, i = cal_slope(y, p, w)
    return {"auroc": roc_auc_score(y, p, sample_weight=w), "pr_auc": average_precision_score(y, p, sample_weight=w),
            "prevalence": float(np.average(y, weights=w)), "brier": brier_score_loss(y, p, sample_weight=w),
            "cal_slope": s, "cal_intercept": i, "o_e": float(np.sum(w * y) / np.sum(w * p)),
            "n_test": int(w.sum()), "events_test": int((w * y).sum())}


def shap_importance(name, m, df):
    """Weighted mean |SHAP| (log-odds scale) per original feature, using test-set patient counts."""
    X, _, w = XY(df)
    if name == "logreg":
        Xs = (X - MU) / SD
        mean_w = np.average(Xs, axis=0, weights=w)
        sv = (Xs.to_numpy() - mean_w) * m.coef_[0]
    else:
        sv = shap.TreeExplainer(m).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        if sv.ndim == 3: sv = sv[:, :, 1]
    imp = pd.Series(np.average(np.abs(sv), axis=0, weights=w), index=COLS)
    return imp.groupby(GROUP).sum(), sv



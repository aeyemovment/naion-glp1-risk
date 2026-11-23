"""Stage 6 - GNN with the same paradigms as Stage 1/1c/5b:
  A. 10% train / 90% hold-out           B. 10 disjoint 10% folds (test = other 90%)
  C. leave-one-study-out (inductive: held-out study nodes have no training labels)
  D. training fraction 10..90% x 10 repeats
  E. threshold metrics (Youden / top 10% / top 20%) with p-values, paired comparison vs XGBoost (same splits)
  F. explainability: integrated gradients (graph Grad-CAM analogue), stability across folds
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
from common import *
from splits import holdout_10_90, kfold_10pct
from gnn import fit_predict_gnn, integrated_gradients
from scipy import stats
from joblib import Parallel, delayed
import os, importlib.util
spec = importlib.util.spec_from_file_location("s5b", "src/stage5b_classification.py")
OUT6 = "results/stage6"; os.makedirs(OUT6, exist_ok=True)

# reuse threshold/confusion helpers without re-running Stage 5b's main body
src = open("src/stage5b_classification.py").read()
helpers = src[src.index("def w_quantile"):src.index("# ---- (i) hold-out 10/90")]
exec(helpers)


def gnn_eval(tr, te, seed=0, attrib=False):
    Xtr, ytr, wtr = XY(tr); Xte, yte, wte = XY(te)
    p_te, g = fit_predict_gnn(Xtr, ytr, wtr, Xte, tr.study.to_numpy(), te.study.to_numpy(), seed=seed, return_model=True)
    p_tr = fit_predict_gnn(Xtr, ytr, wtr, Xtr, tr.study.to_numpy(), tr.study.to_numpy(), seed=seed)
    out = {"p_te": p_te, "p_tr": p_tr, "y": yte, "w": wte, "ytr": ytr, "wtr": wtr}
    if attrib: out["ig"] = integrated_gradients(g, wte).groupby(GROUP).sum()
    return out


def xgb_eval(tr, te):
    m = fit("xgb", tr); return predict("xgb", m, te)


res = {"meta": {"model": "GraphSAGE-mean 2-layer, h=32, kNN k=10 + same-study edges, count-weighted binomial loss, 300 epochs",
                "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}}

# A. hold-out
tr, te = holdout_10_90(ipd, seed=2026)
r = gnn_eval(tr, te, attrib=True)
res["holdout_10_90"] = metrics(r["y"], r["p_te"], r["w"])
res["holdout_xgb_same_split"] = metrics(r["y"], xgb_eval(tr, te), r["w"])
th = thresholds(r["p_tr"], r["ytr"], r["wtr"])
res["holdout_thresholds"] = {k: conf(r["p_te"], r["y"], r["w"], t) for k, t in th.items()}

# B. 10 x 10% folds (paired with XGB on identical splits)
def fold_job(f, tr, te):
    r = gnn_eval(tr, te, seed=f, attrib=True); px = xgb_eval(tr, te)
    th = thresholds(r["p_tr"], r["ytr"], r["wtr"])
    return {"fold": f, "gnn": metrics(r["y"], r["p_te"], r["w"]), "xgb": metrics(r["y"], px, r["w"]),
            "thr": {k: conf(r["p_te"], r["y"], r["w"], t) for k, t in th.items()}, "ig": r["ig"].to_dict()}
folds = Parallel(n_jobs=4)(delayed(fold_job)(f, tr, te) for f, tr, te in kfold_10pct(ipd, k=10, seed=7))
kf = pd.DataFrame([{"fold": x["fold"], **{f"gnn_{k}": v for k, v in x["gnn"].items()}, **{f"xgb_{k}": v for k, v in x["xgb"].items()}} for x in folds])
kf.to_csv(f"{OUT6}/gnn_kfold.csv", index=False)
d = kf.gnn_auroc - kf.xgb_auroc; t = d.mean() / np.sqrt((1 / 10 + 9) * d.var(ddof=1))
res["kfold"] = {"gnn_auroc_mean": kf.gnn_auroc.mean(), "gnn_auroc_sd": kf.gnn_auroc.std(), "xgb_auroc_mean": kf.xgb_auroc.mean(),
                "gnn_o_e_mean": kf.gnn_o_e.mean(), "gnn_cal_slope_mean": kf.gnn_cal_slope.mean(), "gnn_brier_mean": kf.gnn_brier.mean(),
                "diff_vs_xgb": d.mean(), "p_vs_xgb_corrected_t": 2 * stats.t.sf(abs(t), 9)}
thr = pd.DataFrame([{"fold": x["fold"], "threshold": k, **v} for x in folds for k, v in x["thr"].items()])
thr.to_csv(f"{OUT6}/gnn_threshold_kfold.csv", index=False)
_t = thr.groupby("threshold")[["sensitivity", "specificity", "accuracy", "ppv", "npv", "lr_pos", "net_benefit"]].agg(["mean", "std"]).round(5)
_t.columns = ["_".join(c) for c in _t.columns]; res["threshold_kfold_summary"] = _t.to_dict("index")
IG = pd.DataFrame([x["ig"] for x in folds]); IG.to_csv(f"{OUT6}/gnn_ig_kfold.csv", index=False)
res["ig_importance"] = IG.mean().sort_values(ascending=False).round(5).to_dict()
res["ig_top5_frequency"] = (IG.rank(axis=1, ascending=False) <= 5).mean().sort_values(ascending=False).to_dict()

# C. leave-one-study-out (inductive)
def loso_job(s):
    tr, te = ipd[ipd.study != s], ipd[ipd.study == s]
    r = gnn_eval(tr, te); px = xgb_eval(tr, te)
    return {"study": s, "gnn": metrics(r["y"], r["p_te"], r["w"]), "xgb": metrics(r["y"], px, r["w"])}
lo = Parallel(n_jobs=4)(delayed(loso_job)(s) for s in sorted(ipd.study.unique()))
lo = pd.DataFrame([{"study": x["study"], **{f"gnn_{k}": v for k, v in x["gnn"].items()}, **{f"xgb_{k}": v for k, v in x["xgb"].items()}} for x in lo])
lo.to_csv(f"{OUT6}/gnn_loso.csv", index=False)
res["loso"] = {"gnn_auroc_mean": lo.gnn_auroc.mean(), "xgb_auroc_mean": lo.xgb_auroc.mean(),
               "gnn_median_o_e": lo.gnn_o_e.median(), "xgb_median_o_e": lo.xgb_o_e.median(),
               "p_gnn_vs_0.5": stats.wilcoxon(lo.gnn_auroc - 0.5, alternative="greater").pvalue,
               "p_gnn_vs_xgb": stats.wilcoxon(lo.gnn_auroc - lo.xgb_auroc).pvalue if (lo.gnn_auroc != lo.xgb_auroc).any() else 1.0,
               "gnn_abs_log_oe_median": float(np.median(np.abs(np.log(lo.gnn_o_e.clip(1e-6))))),
               "xgb_abs_log_oe_median": float(np.median(np.abs(np.log(lo.xgb_o_e.clip(1e-6)))))}

# D. training-fraction sweep
def frac_job(f, rep):
    tr, te = holdout_10_90(ipd, train_frac=f, seed=1000 * rep + int(f * 100))
    r = gnn_eval(tr, te, seed=rep); return {"train_frac": f, "repeat": rep, **metrics(r["y"], r["p_te"], r["w"])}
sw = pd.DataFrame(Parallel(n_jobs=4)(delayed(frac_job)(f, rep) for f in [i / 10 for i in range(1, 10)] for rep in range(10)))
sw.to_csv(f"{OUT6}/gnn_train_fraction_sweep.csv", index=False)
_f = sw.groupby("train_frac")[["auroc", "o_e", "brier"]].agg(["mean", "std"]).round(5); _f.columns = ["_".join(c) for c in _f.columns]
res["train_fraction"] = {str(k): v for k, v in _f.to_dict("index").items()}
rho, p = stats.spearmanr(sw.train_frac, sw.auroc); res["train_fraction_trend"] = {"spearman_rho": rho, "p": p}

json.dump(res, open(f"{OUT6}/stage6_gnn_results.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["holdout_10_90", "holdout_xgb_same_split", "kfold", "loso", "train_fraction_trend", "ig_importance"]}, indent=1, default=lambda x: round(float(x), 4)))
print(pd.DataFrame(res["threshold_kfold_summary"]).T.to_string())

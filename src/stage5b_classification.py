"""Stage 5b - threshold metrics with p-values: AUROC, sensitivity, specificity, accuracy, PPV, NPV.

Thresholds are fixed on the TRAINING split only (no leakage):
  T1 Youden-optimal on training predictions
  T2 top 10% predicted risk (90th percentile of training predictions)
  T3 top 20% predicted risk
Evaluated on (i) the 10/90 hold-out and (ii) the 10 disjoint 10% folds (test = other 90%).
Tests:
  vs chance - sensitivity > flagged fraction (one-sided binomial), specificity > 1 - flagged fraction,
              PPV > prevalence (binomial), NPV > 1 - prevalence (binomial), accuracy vs no-information rate,
              AUROC > 0.5 (Hanley-McNeil)
  model vs model - McNemar (exact) on discordant events (sensitivity) / non-events (specificity) on the hold-out;
              Nadeau-Bengio corrected resampled t-test across the 10 folds for every metric
  decision curve - net benefit vs treat-all / treat-none at the threshold probability
Patient-level counts are exact (count-weighted pseudo-IPD). research_prototype=true. NOT for clinical use.
"""
from common import *
from splits import holdout_10_90, kfold_10pct
from scipy import stats
import os
OUT5 = "results/stage5"; os.makedirs(OUT5, exist_ok=True)
MODELS = ["logreg", "rf", "xgb", "ensemble"]


def fit_predict(tr, te):
    P_tr, P_te = {}, {}
    for nm in ["logreg", "rf", "xgb"]:
        m = fit(nm, tr); P_tr[nm] = predict(nm, m, tr); P_te[nm] = predict(nm, m, te)
    P_tr["ensemble"] = np.mean([P_tr[k] for k in ["logreg", "rf", "xgb"]], 0)
    P_te["ensemble"] = np.mean([P_te[k] for k in ["logreg", "rf", "xgb"]], 0)
    return P_tr, P_te


def w_quantile(p, w, q):
    o = np.argsort(p); cw = np.cumsum(w[o]) / w.sum(); return p[o][np.searchsorted(cw, q)]


def thresholds(p, y, w):
    o = np.unique(p); best, bt = -1, o[0]
    for t in o:
        f = p >= t; sens = (w * y * f).sum() / (w * y).sum(); spec = (w * (1 - y) * ~f).sum() / (w * (1 - y)).sum()
        if sens + spec - 1 > best: best, bt = sens + spec - 1, t
    return {"T1_youden": bt, "T2_top10pct": w_quantile(p, w, 0.90), "T3_top20pct": w_quantile(p, w, 0.80)}


def conf(p, y, w, t):
    f = p >= t
    TP = (w * y * f).sum(); FN = (w * y * ~f).sum(); FP = (w * (1 - y) * f).sum(); TN = (w * (1 - y) * ~f).sum()
    N = TP + FN + FP + TN; prev = (TP + FN) / N; flag = (TP + FP) / N
    r = {"TP": TP, "FN": FN, "FP": FP, "TN": TN, "prevalence": prev, "flagged_frac": flag,
         "sensitivity": TP / (TP + FN), "specificity": TN / (TN + FP), "accuracy": (TP + TN) / N,
         "ppv": TP / (TP + FP) if TP + FP else np.nan, "npv": TN / (TN + FN) if TN + FN else np.nan}
    r["lr_pos"] = r["sensitivity"] / (1 - r["specificity"]) if r["specificity"] < 1 else np.nan
    r["net_benefit"] = TP / N - FP / N * (t / (1 - t)); r["net_benefit_treat_all"] = prev - (1 - prev) * (t / (1 - t))
    I = lambda k, n: (int(round(k)), int(round(n)))
    r["p_sens_vs_chance"] = stats.binomtest(*I(TP, TP + FN), flag, alternative="greater").pvalue
    r["p_spec_vs_chance"] = stats.binomtest(*I(TN, TN + FP), 1 - flag, alternative="greater").pvalue
    r["p_ppv_vs_prevalence"] = stats.binomtest(*I(TP, TP + FP), prev, alternative="greater").pvalue if TP + FP else np.nan
    r["p_npv_vs_1minusprev"] = stats.binomtest(*I(TN, TN + FN), 1 - prev, alternative="greater").pvalue if TN + FN else np.nan
    nir = max(prev, 1 - prev); r["no_information_rate"] = nir
    r["p_acc_vs_nir"] = stats.binomtest(*I(TP + TN, N), nir, alternative="greater").pvalue
    for k, (num, den) in {"sensitivity": (TP, TP + FN), "specificity": (TN, TN + FP), "ppv": (TP, TP + FP), "npv": (TN, TN + FN), "accuracy": (TP + TN, N)}.items():
        if den > 0:
            ci = stats.binomtest(*I(num, den)).proportion_ci(method="wilson"); r[k + "_lo"], r[k + "_hi"] = ci.low, ci.high
    return r


def auc_p(A, n1, n0):
    q1, q2 = A / (2 - A), 2 * A ** 2 / (1 + A)
    se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A ** 2) + (n0 - 1) * (q2 - A ** 2)) / (n1 * n0))
    return se, stats.norm.sf((A - 0.5) / se)


# ---- (i) hold-out 10/90
tr, te = holdout_10_90(ipd, seed=2026)
_, ytr, wtr = XY(tr); _, yte, wte = XY(te)
P_tr, P_te = fit_predict(tr, te)
hold = []
for m in MODELS:
    A = roc_auc_score(yte, P_te[m], sample_weight=wte); n1 = (wte * yte).sum(); n0 = (wte * (1 - yte)).sum(); se, pa = auc_p(A, n1, n0)
    for tn, t in thresholds(P_tr[m], ytr, wtr).items():
        hold.append({"model": m, "threshold": tn, "t_value": t, "auroc": A, "auroc_lo": A - 1.96 * se, "auroc_hi": A + 1.96 * se, "p_auroc_vs_0.5": pa, **conf(P_te[m], yte, wte, t)})
hold = pd.DataFrame(hold); hold.to_csv(f"{OUT5}/classification_holdout.csv", index=False)

# McNemar between models (hold-out), per threshold, for sensitivity and specificity
mc = []
TH = {m: thresholds(P_tr[m], ytr, wtr) for m in MODELS}
for tn in ["T1_youden", "T2_top10pct", "T3_top20pct"]:
    for a, b in [("xgb", "logreg"), ("xgb", "rf"), ("xgb", "ensemble"), ("ensemble", "logreg"), ("rf", "logreg")]:
        fa, fb = P_te[a] >= TH[a][tn], P_te[b] >= TH[b][tn]
        for lab, mask in [("sensitivity", yte == 1), ("specificity", yte == 0)]:
            good_a = fa if lab == "sensitivity" else ~fa; good_b = fb if lab == "sensitivity" else ~fb
            b01 = wte[mask & good_a & ~good_b].sum(); b10 = wte[mask & ~good_a & good_b].sum()
            n = int(round(b01 + b10))
            p = stats.binomtest(int(round(b01)), n, 0.5).pvalue if n > 0 else 1.0
            mc.append({"threshold": tn, "comparison": f"{a} vs {b}", "metric": lab, "a_better_n": b01, "b_better_n": b10, "p_mcnemar_exact": p})
mc = pd.DataFrame(mc); mc["q_bh"] = np.clip(np.minimum.accumulate((mc.p_mcnemar_exact.sort_values()[::-1] * len(mc) / np.arange(len(mc), 0, -1)))[::-1].reindex(mc.index), 0, 1)
mc.to_csv(f"{OUT5}/mcnemar_holdout.csv", index=False)

# ---- (ii) 10 folds (train 10%, test 90%)
rows = []
for f, tr, te in kfold_10pct(ipd, k=10, seed=7):
    _, ytr, wtr = XY(tr); _, yte, wte = XY(te)
    P_tr, P_te = fit_predict(tr, te)
    for m in MODELS:
        A = roc_auc_score(yte, P_te[m], sample_weight=wte)
        for tn, t in thresholds(P_tr[m], ytr, wtr).items():
            c = conf(P_te[m], yte, wte, t); rows.append({"fold": f, "model": m, "threshold": tn, "auroc": A,
                **{k: c[k] for k in ["sensitivity", "specificity", "accuracy", "ppv", "npv", "lr_pos", "net_benefit", "net_benefit_treat_all", "flagged_frac", "prevalence"]}})
kf = pd.DataFrame(rows); kf.to_csv(f"{OUT5}/classification_kfold.csv", index=False)
METRICS = ["auroc", "sensitivity", "specificity", "accuracy", "ppv", "npv", "net_benefit"]
summ = kf.groupby(["model", "threshold"])[METRICS + ["lr_pos", "flagged_frac"]].agg(["mean", "std"])
summ.columns = ["_".join(c) for c in summ.columns]; summ.reset_index().to_csv(f"{OUT5}/classification_kfold_summary.csv", index=False)
cmp = []
for tn in ["T1_youden", "T2_top10pct", "T3_top20pct"]:
    for a, b in [("xgb", "logreg"), ("xgb", "rf"), ("xgb", "ensemble"), ("ensemble", "logreg"), ("rf", "logreg")]:
        for met in METRICS:
            A_ = kf[(kf.model == a) & (kf.threshold == tn)].sort_values("fold")[met].to_numpy()
            B_ = kf[(kf.model == b) & (kf.threshold == tn)].sort_values("fold")[met].to_numpy()
            d = A_ - B_; k = len(d); v = d.var(ddof=1)
            t = d.mean() / np.sqrt((1 / k + 9.0) * v) if v > 0 else 0.0
            cmp.append({"threshold": tn, "comparison": f"{a} vs {b}", "metric": met, "mean_diff": d.mean(), "p_corrected_t": 2 * stats.t.sf(abs(t), k - 1) if v > 0 else 1.0})
cmp = pd.DataFrame(cmp)
o = np.argsort(cmp.p_corrected_t.to_numpy()); n = len(cmp); q = np.empty(n)
q[o] = np.minimum.accumulate((cmp.p_corrected_t.to_numpy()[o] * n / np.arange(1, n + 1))[::-1])[::-1]; cmp["q_bh"] = np.clip(q, 0, 1)
cmp.to_csv(f"{OUT5}/model_comparison_kfold.csv", index=False)

pd.set_option("display.width", 250)
show = ["model", "threshold", "auroc", "p_auroc_vs_0.5", "sensitivity", "p_sens_vs_chance", "specificity", "p_spec_vs_chance", "accuracy", "no_information_rate", "p_acc_vs_nir", "ppv", "prevalence", "p_ppv_vs_prevalence", "npv", "p_npv_vs_1minusprev", "lr_pos", "net_benefit", "net_benefit_treat_all"]
print(hold[show].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print(summ.round(4).reset_index().to_string(index=False))
print(cmp[cmp.q_bh < 0.05].to_string(index=False))
print("min q in k-fold comparisons:", cmp.q_bh.min().round(4))
print(mc.sort_values("p_mcnemar_exact").head(12).to_string(index=False))

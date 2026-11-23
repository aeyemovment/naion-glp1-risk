"""Stage 11 - how well does each tool detect NAION risk WITHIN GLP-1RA users and WITHIN non-users?

Models trained on mixed data (each of the 10 disjoint 10% folds, test = other 90%) and evaluated separately in the
GLP-exposed and non-GLP subsets of the test set: AUROC, PR-AUC, O/E, calibration slope, and sensitivity / specificity /
accuracy / PPV / NPV at the top-10% threshold fixed on training data. Group difference per model: Nadeau-Bengio
corrected paired t across folds, Benjamini-Hochberg across models x metrics.
Models: logreg, rf, xgb, gnn (default GraphSAGE), cnn1d, mean ensemble (LR+RF+XGB+GNN), poisson_offset (rate model),
isolation_forest and autoencoder (anomaly detectors trained on non-events).
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
from common import *
from splits import kfold_10pct
from gnn import fit_predict_gnn_cfg
from scipy import stats
from joblib import Parallel, delayed
import importlib.util, os
OUT = "results/stage11"; os.makedirs(OUT, exist_ok=True)
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
S9 = _load("s9", "src/stage9_cnn.py"); S10 = _load("s10", "src/stage10_incidence_anomaly.py")
src = open("src/stage5b_classification.py").read(); exec(src[src.index("def w_quantile"):src.index("# ---- (i) hold-out 10/90")])


def all_preds(tr, te, seed):
    P = {}; Ptr = {}
    for nm in ["logreg", "rf", "xgb"]:
        m = fit(nm, tr); P[nm] = predict(nm, m, te); Ptr[nm] = predict(nm, m, tr)
    Xtr, ytr, wtr = XY(tr)
    both = pd.concat([te, tr]); pg = fit_predict_gnn_cfg(Xtr, ytr, wtr, XY(both)[0], tr.study.to_numpy(), both.study.to_numpy(), seed=seed)
    P["gnn"], Ptr["gnn"] = pg[:len(te)], pg[len(te):]
    c = S9.fit_cnn(tr, seed); P["cnn1d"], Ptr["cnn1d"] = S9.pred(c, te), S9.pred(c, tr)
    P["ensemble"] = np.mean([P[k] for k in ["logreg", "rf", "xgb", "gnn"]], 0); Ptr["ensemble"] = np.mean([Ptr[k] for k in ["logreg", "rf", "xgb", "gnn"]], 0)
    r = S10.rare_models(tr, te); rt = S10.rare_models(tr, tr); P["poisson_offset"], Ptr["poisson_offset"] = r["poisson_offset"], rt["poisson_offset"]
    a = S10.anomaly_models(tr, te, seed); at = S10.anomaly_models(tr, tr, seed)
    for k in ["isolation_forest", "autoencoder"]: P[k], Ptr[k] = a[k], at[k]
    return P, Ptr


def fold_job(f, tr, te):
    P, Ptr = all_preds(tr, te, f); _, ytr, wtr = XY(tr); rows = []
    for m in P:
        t = thresholds(Ptr[m], ytr, wtr)["T2_top10pct"]
        for grp, mask in [("GLP-1RA users", te.glp1ra_exposed.to_numpy() == 1), ("non-users", te.glp1ra_exposed.to_numpy() == 0), ("all", np.ones(len(te), bool))]:
            sub = te[mask]; _, y, w = XY(sub); p = P[m][mask]
            r = {"fold": f, "model": m, "group": grp, "auroc": roc_auc_score(y, p, sample_weight=w), "pr_auc": average_precision_score(y, p, sample_weight=w),
                 "prevalence": float(np.average(y, weights=w))}
            if m not in ("isolation_forest", "autoencoder"): r.update({k: v for k, v in metrics(y, p, w).items() if k in ("o_e", "cal_slope", "brier")})
            c = conf(p, y, w, t); r.update({k: c[k] for k in ["sensitivity", "specificity", "accuracy", "ppv", "npv", "lr_pos", "flagged_frac"]})
            rows.append(r)
    return rows


if __name__ == "__main__":
    rows = [r for ch in Parallel(n_jobs=1)(delayed(fold_job)(f, tr, te) for f, tr, te in kfold_10pct(ipd, k=10, seed=7)) for r in ch]
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/subgroup_folds.csv", index=False)
    M = ["auroc", "pr_auc", "o_e", "cal_slope", "sensitivity", "specificity", "accuracy", "ppv", "npv"]
    summ = df.groupby(["model", "group"])[M + ["prevalence"]].agg(["mean", "std"]); summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index(); summ.to_csv(f"{OUT}/subgroup_summary.csv", index=False)
    tests = []
    for m in df.model.unique():
        a = df[(df.model == m) & (df.group == "GLP-1RA users")].sort_values("fold"); b = df[(df.model == m) & (df.group == "non-users")].sort_values("fold")
        for k in M:
            if k not in a or a[k].isna().all(): continue
            d = a[k].to_numpy() - b[k].to_numpy(); v = d.var(ddof=1)
            t = d.mean() / np.sqrt((1 / 10 + 9) * v) if v > 0 else 0.0
            tests.append({"model": m, "metric": k, "GLP_mean": a[k].mean(), "nonGLP_mean": b[k].mean(), "diff": d.mean(), "p": 2 * stats.t.sf(abs(t), 9) if v > 0 else 1.0})
    tests = pd.DataFrame(tests); o = np.argsort(tests.p.to_numpy()); n = len(tests); q = np.empty(n)
    q[o] = np.minimum.accumulate((tests.p.to_numpy()[o] * n / np.arange(1, n + 1))[::-1])[::-1]; tests["q_bh"] = np.clip(q, 0, 1)
    tests.to_csv(f"{OUT}/subgroup_tests.csv", index=False)
    # figure: AUROC by group per model
    order = ["logreg", "rf", "xgb", "gnn", "cnn1d", "ensemble", "poisson_offset", "isolation_forest", "autoencoder"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for j, (k, lab) in enumerate([("auroc", "AUROC"), ("ppv", "PPV at top-10% threshold")]):
        for g, c, off in [("non-users", "#2a78d6", -0.15), ("GLP-1RA users", "#eb6834", 0.15)]:
            s_ = summ[summ.group == g].set_index("model").reindex(order)
            ax[j].errorbar(np.arange(len(order)) + off, s_[f"{k}_mean"], yerr=s_[f"{k}_std"], fmt="o", color=c, capsize=2, ms=5, label=g)
        ax[j].set_xticks(range(len(order)), ["LR", "RF", "XGB", "GNN", "CNN", "Ens.", "Poisson", "IForest", "AE"], fontsize=7)
        ax[j].set_ylabel(lab); ax[j].spines[["top", "right"]].set_visible(False); ax[j].grid(axis="y", color="#e4e3df")
    ax[0].axhline(0.5, color="#a3a29d", ls="--", lw=1); ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_subgroups.png", dpi=300)
    json.dump({"summary": summ.round(5).to_dict("records"), "tests": tests.round(5).to_dict("records"),
               "significant_differences": tests[tests.q_bh < 0.05].round(5).to_dict("records"), "research_prototype": True, "compliance_ref": "Addendum_5_DT9"},
              open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(summ[["model", "group", "auroc_mean", "pr_auc_mean", "o_e_mean", "sensitivity_mean", "specificity_mean", "ppv_mean", "npv_mean"]].round(4).to_string(index=False))
    print(tests[tests.metric == "auroc"].round(4).to_string(index=False))

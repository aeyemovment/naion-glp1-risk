"""Stage 13 (12b) - with vs without partial ABCD2 features, across every experiment. Identical seeds/splits -> paired comparisons.
Reads results/ (without) and variant_abcd/results/ (with). Writes results/stage13/. research_prototype=true. NOT for clinical use."""
import os, json, numpy as np, pandas as pd
from scipy import stats
R = os.path.expanduser("~/naion-glp1-risk"); os.chdir(R); OUT = "results/stage13"; os.makedirs(OUT, exist_ok=True)
W, V = "results", "variant_abcd/results"
def rd(p): return pd.read_csv(p) if os.path.exists(p) else None
def corrected_t(d, k_ratio=9.0):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 2 or d.std(ddof=1) == 0: return np.nan
    return 2 * stats.t.sf(abs(d.mean() / np.sqrt((1 / len(d) + k_ratio) * d.var(ddof=1))), len(d) - 1)
rows = []
def add(exp, model, a, b, metric="auroc", paired=True, ratio=9.0):
    a = pd.Series(a, dtype=float); b = pd.Series(b, dtype=float)
    d = (b - a) if paired else None
    rows.append({"experiment": exp, "model": model, "metric": metric, "without_abcd": a.mean(), "with_abcd": b.mean(), "diff": (b.mean() - a.mean()),
                 "p": corrected_t(d, ratio) if paired else stats.ttest_ind(b, a, equal_var=False).pvalue, "n_pairs": len(a)})
# Stage 1 folds (logreg/rf/xgb)
for st in ["stage1/stage1_results.json"]:
    if os.path.exists(f"{W}/{st}") and os.path.exists(f"{V}/{st}"):
        a, b = json.load(open(f"{W}/{st}")), json.load(open(f"{V}/{st}"))
        for m in ["logreg", "rf", "xgb"]:
            for met in ["auroc", "o_e", "brier"]:
                add("Stage 1: 10 x 10% folds", m, [x[met] for x in a["kfold_10pct"][m]], [x[met] for x in b["kfold_10pct"][m]], met)
            add("Stage 1: leave-one-study-out", m, [x["auroc"] for x in a["loso"][m]], [x["auroc"] for x in b["loso"][m]], "auroc", ratio=1 / 10)
# Stage 1c fraction sweep
a, b = rd(f"{W}/stage1c/train_fraction_sweep.csv"), rd(f"{V}/stage1c/train_fraction_sweep.csv")
if a is not None and b is not None:
    for m in a.model.unique():
        x = a[a.model == m].sort_values(["train_frac", "repeat"]); y = b[b.model == m].sort_values(["train_frac", "repeat"])
        add("Stage 1c: training fractions 10-90%", m, x.auroc.values, y.auroc.values, ratio=1.0)
# Stage 5b thresholds (kfold)
a, b = rd(f"{W}/stage5/classification_kfold.csv"), rd(f"{V}/stage5/classification_kfold.csv")
if a is not None and b is not None:
    for (m, t), g in a.groupby(["model", "threshold"]):
        h = b[(b.model == m) & (b.threshold == t)].sort_values("fold"); g = g.sort_values("fold")
        for met in ["sensitivity", "specificity", "ppv", "npv", "accuracy"]:
            add(f"Stage 5b: threshold {t}", m, g[met].values, h[met].values, met)
# Stage 6 GNN, Stage 9 CNN, Stage 6 LOSO
for exp, f_, col in [("Stage 6: GNN folds", "stage6/gnn_kfold.csv", "gnn_auroc"), ("Stage 6: GNN LOSO", "stage6/gnn_loso.csv", "gnn_auroc")]:
    a, b = rd(f"{W}/{f_}"), rd(f"{V}/{f_}")
    if a is not None and b is not None: add(exp, "gnn", a[col].values, b[col].values, ratio=9.0 if "folds" in exp else 0.1)
a, b = rd(f"{W}/stage9/cnn_all.csv"), rd(f"{V}/stage9/cnn_all.csv")
if a is not None and b is not None:
    for pre, lab, r_ in [("fold", "folds", 9.0), ("loso", "LOSO", 0.1)]:
        x = a[a.tag.str.startswith(pre)].sort_values("tag"); y = b[b.tag.str.startswith(pre)].sort_values("tag"); add(f"Stage 9: CNN {lab}", "cnn1d", x.auroc.values, y.auroc.values, ratio=r_)
# Stage 10 rare-event / anomaly, Stage 11 subgroups, Stage 7 external, Stage 4 ensembles
a, b = rd(f"{W}/stage10/rare_event_anomaly_all.csv"), rd(f"{V}/stage10/rare_event_anomaly_all.csv")
if a is not None and b is not None:
    for m in a.model.unique():
        x = a[(a.model == m) & a.tag.str.startswith("fold")].sort_values("tag"); y = b[(b.model == m) & b.tag.str.startswith("fold")].sort_values("tag")
        add("Stage 10: rare-event / anomaly folds", m, x.auroc.values, y.auroc.values)
a, b = rd(f"{W}/stage11/subgroup_folds.csv"), rd(f"{V}/stage11/subgroup_folds.csv")
if a is not None and b is not None:
    for (m, g_), x in a.groupby(["model", "group"]):
        y = b[(b.model == m) & (b.group == g_)].sort_values("fold"); add(f"Stage 11: within {g_}", m, x.sort_values("fold").auroc.values, y.auroc.values)
a, b = rd(f"{W}/stage7/stage7_all.csv"), rd(f"{V}/stage7/stage7_all.csv")
if a is not None and b is not None:
    for (m, pop), x in a[a.tag.str.startswith("fold")].groupby(["model", "population"]):
        y = b[b.tag.str.startswith("fold") & (b.model == m) & (b.population == pop)].sort_values("tag")
        for met in ["auroc", "o_e"]: add(f"Stage 7: non-GLP model, {pop}", m, x.sort_values("tag")[met].values, y[met].values, met)
a, b = rd(f"{W}/stage4/ensembles_all.csv"), rd(f"{V}/stage4/ensembles_all.csv")
if a is not None and b is not None:
    for (e, mt), x in a[a.tag.str.startswith("fold")].groupby(["ensemble", "method"]):
        y = b[b.tag.str.startswith("fold") & (b.ensemble == e) & (b.method == mt)].sort_values("tag")
        if len(y) == len(x): add("Stage 4: ensembles folds", f"{e} ({mt})", x.sort_values("tag").auroc.values, y.auroc.values)
# Stage 8 fine-tuning (by config, paired over repeats)
a, b = rd(f"{W}/stage8/finetune_all.csv"), rd(f"{V}/stage8/finetune_all.csv")
if a is not None and b is not None:
    k = ["pretrain_source", "finetune_source", "glp_frac", "method"]
    for key, x in a.groupby(k):
        y = b.set_index(k).loc[key] if key in b.set_index(k).index else None
        if y is not None and len(y) == len(x): add("Stage 8: fine-tuning", " / ".join(map(str, key)), x.sort_values("rep").glp_auroc.values, y.sort_values("rep").glp_auroc.values, ratio=40 / 50)
# Stage 2/3 twins (by configuration, paired over seeds)
for exp, f_, keys in [("Stage 2: twin lattice (XGB)", "stage2/lattice_results.csv", ["h", "parent", "V", "s"]),
                      ("Stage 3: full factorial", None, ["model", "train_frac", "genset", "source", "V", "s"])]:
    if f_:
        a, b = rd(f"{W}/{f_}"), rd(f"{V}/{f_}")
    else:
        import glob
        fa, fb = sorted(glob.glob(f"{W}/stage3/factorial_tf*.csv")), sorted(glob.glob(f"{V}/stage3/factorial_tf*.csv"))
        a = pd.concat([pd.read_csv(x) for x in fa]) if fa else None; b = pd.concat([pd.read_csv(x) for x in fb]) if fb else None
    if a is not None and b is not None:
        if "depth" in a: a = a[a.depth.isna()]
        if "depth" in b: b = b[b.depth.isna()]
        m = a.merge(b, on=keys + ["seed"], suffixes=("_wo", "_w"))
        if len(m):
            d = m.groupby(keys).apply(lambda g: pd.Series({"wo": g.auroc_wo.mean(), "w": g.auroc_w.mean(), "p": corrected_t(g.auroc_w - g.auroc_wo, 1.0)})).reset_index()
            rows.append({"experiment": exp, "model": f"all {len(d)} configurations", "metric": "auroc", "without_abcd": d.wo.mean(), "with_abcd": d.w.mean(),
                         "diff": (d.w - d.wo).mean(), "p": np.nan, "n_pairs": len(m), "configs_better_p05": int(((d.w > d.wo) & (d.p < 0.05)).sum()), "configs_worse_p05": int(((d.w < d.wo) & (d.p < 0.05)).sum())})
df = pd.DataFrame(rows)
if len(df):
    p = df.p.fillna(1).to_numpy(); o = np.argsort(p); n = len(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; df["q_bh"] = np.clip(q, 0, 1)
df.to_csv(f"{OUT}/abcd_ablation.csv", index=False)
auc = df[df.metric == "auroc"] if len(df) else df
json.dump({"n_comparisons": int(len(df)), "auroc_mean_diff": float(auc["diff"].mean()) if len(auc) else None,
           "significant_q05": df[(df.q_bh < 0.05)].round(5).to_dict("records") if len(df) else [],
           "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(df.round(4).to_string(index=False) if len(df) else "no paired results yet")

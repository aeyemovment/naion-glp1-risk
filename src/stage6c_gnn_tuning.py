"""Stage 6c - GNN edge-weight and depth optimisation (nested: tune on folds 0-4, confirm on folds 5-9, LOSO for top configs).
Grid (SAGE): depth L 1..6 x k {5,10,20} x edge weighting {uniform, gauss0.5, gauss1, gauss2} x same-study weight {0,.5,1,2,5} = 360
Learned weights (GAT attention): depth 1..6, k=10, same-study weight 1.
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use."""
from common import *
from splits import kfold_10pct
from gnn import fit_predict_gnn_cfg
from scipy import stats
from joblib import Parallel, delayed
import os, itertools
OUT = "results/stage6c"; os.makedirs(OUT, exist_ok=True)
GRID = [dict(arch="sage", L=L, k=k, edge_mode=e, study_w=s) for L in range(1, 7) for k in (5, 10, 20)
        for e in ("uniform", "gauss0.5", "gauss1", "gauss2") for s in (0, 0.5, 1, 2, 5)]
GRID += [dict(arch="gat", L=L, k=10, edge_mode="uniform", study_w=1) for L in range(1, 7)]
FOLDS = list(kfold_10pct(ipd, k=10, seed=7))
cid = lambda c: f"{c['arch']}|L{c['L']}|k{c['k']}|{c['edge_mode']}|sw{c['study_w']}"

def job(ci, c, f):
    _, tr, te = FOLDS[f]; Xtr, ytr, wtr = XY(tr); Xte, yte, wte = XY(te)
    p = fit_predict_gnn_cfg(Xtr, ytr, wtr, Xte, tr.study.to_numpy(), te.study.to_numpy(), seed=f, **c)
    return {"config": cid(c), **c, "fold": f, **metrics(yte, p, wte)}

path = f"{OUT}/grid_folds.csv"
if not os.path.exists(path):
    rows = Parallel(n_jobs=4, batch_size=4)(delayed(job)(i, c, f) for i, c in enumerate(GRID) for f in range(10))
    pd.DataFrame(rows).to_csv(path, index=False)
g = pd.read_csv(path)
tune = g[g.fold < 5].groupby("config").agg(auroc=("auroc", "mean"), o_e=("o_e", "mean"), brier=("brier", "mean")).reset_index()
conf_ = g[g.fold >= 5].groupby("config").agg(auroc_confirm=("auroc", "mean"), sd_confirm=("auroc", "std")).reset_index()
tab = tune.merge(conf_, on="config").merge(g.drop_duplicates("config")[["config", "arch", "L", "k", "edge_mode", "study_w"]], on="config")
tab["calibrated"] = tab.o_e.sub(1).abs() < 0.1
tab = tab.sort_values("auroc", ascending=False); tab.to_csv(f"{OUT}/grid_ranking.csv", index=False)
best = tab[tab.calibrated].iloc[0]
default = tab[tab.config == "sage|L2|k10|uniform|sw1.0"].iloc[0] if (tab.config == "sage|L2|k10|uniform|sw1.0").any() else tab[tab.config.str.startswith("sage|L2|k10|uniform|sw1")].iloc[0]
# paired test on the 5 untouched folds: tuned best vs default (L2, k10, uniform, sw1)
a = g[(g.config == best.config) & (g.fold >= 5)].sort_values("fold").auroc.to_numpy(); b = g[(g.config == default.config) & (g.fold >= 5)].sort_values("fold").auroc.to_numpy()
d = a - b; t = d.mean() / np.sqrt((1 / 5 + 9) * d.var(ddof=1)) if d.var(ddof=1) > 0 else 0
# main effects on confirm folds
eff = {f: g[g.fold >= 5].groupby(f).auroc.agg(["mean", "std", "size"]).reset_index().to_dict("records") for f in ["L", "k", "edge_mode", "study_w", "arch"]}
# LOSO for top-5 tuned configs + default + best GAT
top = list(tab[tab.calibrated].config.head(5)) + [default.config] + [tab[tab.arch == "gat"].config.iloc[0]]
def loso(cfg, s):
    c = tab[tab.config == cfg].iloc[0]; kw = dict(arch=c.arch, L=int(c.L), k=int(c.k), edge_mode=c.edge_mode, study_w=float(c.study_w))
    tr, te = ipd[ipd.study != s], ipd[ipd.study == s]; Xtr, ytr, wtr = XY(tr); Xte, yte, wte = XY(te)
    p = fit_predict_gnn_cfg(Xtr, ytr, wtr, Xte, tr.study.to_numpy(), te.study.to_numpy(), **kw)
    return {"config": cfg, "study": s, **metrics(yte, p, wte)}
lo = pd.DataFrame(Parallel(n_jobs=4)(delayed(loso)(c, s) for c in dict.fromkeys(top) for s in sorted(ipd.study.unique())))
lo.to_csv(f"{OUT}/loso_top.csv", index=False)
los = lo.groupby("config").agg(loso_auroc=("auroc", "mean"), loso_med_abs_log_oe=("o_e", lambda x: float(np.median(np.abs(np.log(x.clip(1e-6))))))).reset_index()
summ = {"n_configs": len(GRID), "best_tuned": best.to_dict(), "default": default.to_dict(),
        "best_vs_default_untouched_folds": {"diff": float(d.mean()), "p_corrected_t": float(2 * stats.t.sf(abs(t), 4)) if t else 1.0},
        "main_effects_confirm_folds": eff, "loso_top": los.to_dict("records"),
        "gat_by_depth": tab[tab.arch == "gat"].sort_values("L")[["L", "auroc", "auroc_confirm", "o_e"]].to_dict("records")}
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
# figures: depth curve per edge mode; study-weight curve
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
cc = {"uniform": "#2a78d6", "gauss0.5": "#eb6834", "gauss1": "#1baf7a", "gauss2": "#eda100"}
cf = g[(g.fold >= 5)]
for e_, col in cc.items():
    q = cf[(cf.arch == "sage") & (cf.edge_mode == e_)].groupby("L").auroc.mean(); ax[0].plot(q.index, q.values, "o-", color=col, lw=2, ms=4, label=e_)
q = cf[cf.arch == "gat"].groupby("L").auroc.mean(); ax[0].plot(q.index, q.values, "s--", color="#6250d6", lw=2, ms=4, label="GAT (learned)")
ax[0].set(xlabel="GNN depth (layers)", ylabel="AUROC (untouched folds 5-9)", title="A  Depth × edge weighting"); ax[0].legend(frameon=False, fontsize=7)
q = cf[cf.arch == "sage"].groupby("study_w").auroc.agg(["mean", "std", "size"]); ax[1].errorbar(q.index, q["mean"], yerr=1.96 * q["std"] / np.sqrt(q["size"]), fmt="o-", color="#2a78d6", lw=2, capsize=2)
ax[1].set(xlabel="Same-study edge weight (× kNN edge)", ylabel="AUROC", title="B  Study-link weight")
q = cf[cf.arch == "sage"].groupby("k").auroc.agg(["mean", "std", "size"]); ax[2].errorbar(q.index, q["mean"], yerr=1.96 * q["std"] / np.sqrt(q["size"]), fmt="o-", color="#1baf7a", lw=2, capsize=2)
ax[2].set(xlabel="Neighbours per node (k)", ylabel="AUROC", title="C  Neighbourhood size")
for a_ in ax: a_.grid(axis="y", color="#e4e3df"); a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_gnn_depth_weights.png", dpi=300)
print(json.dumps({k: summ[k] for k in ["best_tuned", "default", "best_vs_default_untouched_folds", "loso_top", "gat_by_depth"]}, indent=1, default=lambda x: round(float(x), 4) if isinstance(x, (float, np.floating)) else str(x)))

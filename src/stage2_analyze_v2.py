"""Stage 2 v2 analysis: 3 models x training fractions 0-90% x 10 repeats.
Paired comparison (same test split, same model) of PubMed-only vs synthetic-only vs hybrid,
learning curves per family, heatmaps, SHAP agreement, robust top-3 selection.
research_prototype=true; twins synthetic_only=true; compliance_ref=Addendum_5_DT9. NOT for clinical use."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
OUT = "results/stage2"
full = pd.read_csv(f"{OUT}/lattice_results_v2.csv")
base = full[full.depth.isna()].copy()
ref = base[(base.parent == "real_only") & (base.V == 1)].set_index(["h", "seed", "model"])
d = base[base.h < 1.0].copy()
key = list(zip(d.h, d.seed, d.model))
d["d_auroc"] = d.auroc.to_numpy() - ref.loc[key, "auroc"].to_numpy()
d["d_brier"] = d.brier.to_numpy() - ref.loc[key, "brier"].to_numpy()
d["family"] = np.where(d.parent == "real_only", "pubmed_only", np.where(d.s == 1.0, "synthetic_only", "hybrid"))
MOD = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost"}
CM = {"logreg": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a"}

def ci(x):
    x = np.asarray(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); return m, m - 1.96 * se, m + 1.96 * se

cfg = d.groupby(["model", "family", "parent", "V", "s"]).agg(n=("auroc", "size"), auroc=("auroc", "mean"),
      d_auroc=("d_auroc", "mean"), d_sd=("d_auroc", "std"), o_e=("o_e", "mean"), cal_slope=("cal_slope", "mean"),
      brier=("brier", "mean"), shap_rho=("shap_spearman_vs_real", "mean")).reset_index()
cfg["d_lo"] = cfg.d_auroc - 1.96 * cfg.d_sd / np.sqrt(cfg.n); cfg["d_hi"] = cfg.d_auroc + 1.96 * cfg.d_sd / np.sqrt(cfg.n)
cfg["calibrated"] = (cfg.o_e.sub(1).abs() < 0.10) & cfg.cal_slope.between(0.9, 1.1)
cfg.sort_values(["d_lo", "auroc"], ascending=False).to_csv(f"{OUT}/v2_config_ranking.csv", index=False)

# robust top configuration per family (across all models): rank by mean AUROC across training fractions, require calibration
top = {}
for fam in ["pubmed_only", "synthetic_only", "hybrid"]:
    c = cfg[(cfg.family == fam) & cfg.calibrated]
    if fam == "pubmed_only": c = c[c.V == 1]
    top[fam] = c.sort_values(["auroc", "d_lo"], ascending=False).head(3)
top3 = pd.concat([t.head(1) for t in top.values()]); top3.to_csv(f"{OUT}/v2_top3_models.csv", index=False)

# per training fraction: PubMed-only vs best-hybrid vs best-synthetic-only (best = best mean over repeats; optimistic, flagged)
lc = []
for (mdl, tf), g in d.groupby(["model", "train_frac"]):
    po = g[(g.family == "pubmed_only") & (g.V == 1)].auroc
    for fam in ["hybrid", "synthetic_only"]:
        gg = g[g.family == fam].groupby(["parent", "V", "s"]).agg(a=("auroc", "mean"), sd=("auroc", "std"), dd=("d_auroc", "mean"), dsd=("d_auroc", "std"), n=("auroc", "size"))
        b = gg.a.idxmax(); r = gg.loc[b]
        lc.append({"model": mdl, "train_frac": tf, "family": fam, "best_cfg": "|".join(map(str, b)), "auroc": r.a, "sd": r.sd,
                   "d_auroc": r.dd, "d_lo": r.dd - 1.96 * r.dsd / np.sqrt(r.n), "d_hi": r.dd + 1.96 * r.dsd / np.sqrt(r.n)})
    lc.append({"model": mdl, "train_frac": tf, "family": "pubmed_only", "best_cfg": "real_only|1|0.0", "auroc": po.mean(), "sd": po.std(), "d_auroc": 0, "d_lo": 0, "d_hi": 0})
lc = pd.DataFrame(lc); lc.to_csv(f"{OUT}/v2_learning_curves_by_family.csv", index=False)
ext = full[(full.h == 1.0) & full.depth.isna()].groupby(["model", "parent", "V"]).agg(auroc=("auroc", "mean"), sd=("auroc", "std"), o_e=("o_e", "mean")).reset_index()
ext.to_csv(f"{OUT}/v2_synthetic_only_external_priors.csv", index=False)

# Figure 1: learning curves per model, 3 families (+ external-prior synthetic-only at 0%)
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
FC = {"pubmed_only": "#0b0b0b", "hybrid": "#eb6834", "synthetic_only": "#2a78d6"}
FL = {"pubmed_only": "PubMed-only", "hybrid": "Best hybrid", "synthetic_only": "Best synthetic-only (twins fit to real train)"}
for i, mdl in enumerate(["logreg", "rf", "xgb"]):
    for fam in ["pubmed_only", "hybrid", "synthetic_only"]:
        q = lc[(lc.model == mdl) & (lc.family == fam)].sort_values("train_frac")
        ax[i].errorbar(q.train_frac * 100, q.auroc, yerr=1.96 * q.sd / np.sqrt(10), color=FC[fam], marker="o", ms=4, lw=2, capsize=2, label=FL[fam])
    e = ext[ext.model == mdl].auroc.max()
    ax[i].scatter([0], [e], marker="D", s=40, color="#6250d6", zorder=4, label="Synthetic-only, external priors (0% real)")
    ax[i].set_title(MOD[mdl]); ax[i].set_xlabel("Real PubMed training fraction (%)"); ax[i].grid(axis="y", color="#e4e3df")
ax[0].set_ylabel("AUROC on held-out real PubMed"); ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/v2_fig_learning_curves_families.png", dpi=300); plt.close(fig)

# Figure 2: heatmaps training fraction x synthetic share (XGB, V=9) per parent
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for i, p in enumerate(["P1_marginal", "P2_copula", "P3_structural"]):
    pv = d[(d.parent == p) & (d.V == 9) & (d.model == "xgb")].pivot_table(index="train_frac", columns="s", values="d_auroc")
    im = ax[i].imshow(pv.values, cmap="RdBu", vmin=-0.12, vmax=0.12, aspect="auto", origin="lower")
    ax[i].set_xticks(range(pv.shape[1]), [f"{int(round(s*100))}" for s in pv.columns], fontsize=7)
    ax[i].set_yticks(range(pv.shape[0]), [f"{int(round(t*100))}" for t in pv.index], fontsize=7)
    ax[i].set_title(p.replace("_", " ")); ax[i].set_xlabel("Synthetic share (%)")
ax[0].set_ylabel("Real training fraction (%)")
cb = fig.colorbar(im, ax=ax, fraction=0.02); cb.set_label("ΔAUROC vs PubMed-only (XGBoost, 9× volume)")
fig.savefig(f"{OUT}/v2_fig_heatmap_trainfrac_share.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# Figure 3: volume & share response by model (hybrid)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for mdl in ["logreg", "rf", "xgb"]:
    v = d[(d.model == mdl) & (d.family == "hybrid")].groupby("V").d_auroc.apply(ci)
    ax[0].errorbar(range(len(v)), [x[0] for x in v], yerr=[[x[0]-x[1] for x in v], [x[2]-x[0] for x in v]], color=CM[mdl], marker="o", ms=5, lw=2, capsize=2, label=MOD[mdl])
    s = d[(d.model == mdl) & (d.family != "pubmed_only")].groupby("s").d_auroc.apply(ci)
    ax[1].errorbar([x * 100 for x in s.index], [x[0] for x in s], yerr=[[x[0]-x[1] for x in s], [x[2]-x[0] for x in s]], color=CM[mdl], marker="o", ms=5, lw=2, capsize=2)
ax[0].set_xticks(range(6), ["1×", "3×", "6×", "9×", "27×", "100×"]); ax[0].set_xlabel("Twin volume (× real training size)")
ax[1].set_xlabel("Synthetic share (%)")
for a in ax: a.axhline(0, color="#a3a29d", ls="--", lw=1); a.set_ylabel("ΔAUROC vs PubMed-only"); a.grid(axis="y", color="#e4e3df")
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/v2_fig_volume_share_by_model.png", dpi=300); plt.close(fig)

summ = {"n_fits": int(len(full)), "top3_for_app": top3.round(5).to_dict("records"),
        "hybrid_cells_ci_above_zero": cfg[(cfg.family == "hybrid") & (cfg.d_lo > 0)][["model", "parent", "V", "s", "d_auroc", "d_lo"]].round(5).to_dict("records"),
        "n_hybrid_cells": int((cfg.family == "hybrid").sum()),
        "external_priors_best": ext.loc[ext.groupby("model").auroc.idxmax()].round(4).to_dict("records"),
        "recursion": json.load(open(f"{OUT}/lattice_meta_v2.json"))["recursion"]}
json.dump(summ, open(f"{OUT}/v2_summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(top3[["model", "family", "parent", "V", "s", "auroc", "d_auroc", "d_lo", "d_hi", "o_e", "cal_slope", "shap_rho"]].round(4).to_string(index=False))
print("hybrid cells with CI>0:", len(summ["hybrid_cells_ci_above_zero"]), "/", summ["n_hybrid_cells"])
print(pd.DataFrame(summ["hybrid_cells_ci_above_zero"]).head(15).to_string())
print(lc.pivot_table(index=["model", "train_frac"], columns="family", values="auroc").round(4).to_string())
print(pd.DataFrame(summ["external_priors_best"]).to_string())
print("recursion gains>0.005:", sum(r["gain"] >= 0.005 for r in summ["recursion"]), "/", len(summ["recursion"]))

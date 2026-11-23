"""Stage 2 analysis: paired (same test split) comparison of PubMed-only vs synthetic-only vs hybrid,
robust (across-h) top configurations, figures.  research_prototype=true. NOT for clinical use."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
OUT = "results/stage2"
d = pd.read_csv(f"{OUT}/lattice_results.csv")
d = d[d.depth.isna()] if "depth" in d else d
ref = d[(d.parent == "real_only") & (d.V == 1)].set_index(["h", "seed"])
d = d[d.h < 1.0].copy()
d["d_auroc"] = d.apply(lambda r: r.auroc - ref.loc[(r.h, r.seed), "auroc"], axis=1)
d["d_brier"] = d.apply(lambda r: r.brier - ref.loc[(r.h, r.seed), "brier"], axis=1)
d["family"] = np.where(d.parent == "real_only", "pubmed_only", np.where(d.s == 1.0, "synthetic_only", "hybrid"))

def ci(x):
    x = np.asarray(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); return m, m - 1.96 * se, m + 1.96 * se

cfg = d.groupby(["family", "parent", "V", "s"]).agg(n=("auroc", "size"), auroc=("auroc", "mean"),
      d_auroc=("d_auroc", "mean"), d_auroc_sd=("d_auroc", "std"), o_e=("o_e", "mean"), cal_slope=("cal_slope", "mean"),
      brier=("brier", "mean"), shap_rho=("shap_spearman_vs_real", "mean")).reset_index()
cfg["d_auroc_lo"] = cfg.d_auroc - 1.96 * cfg.d_auroc_sd / np.sqrt(cfg.n)
cfg["d_auroc_hi"] = cfg.d_auroc + 1.96 * cfg.d_auroc_sd / np.sqrt(cfg.n)
# robust selection: rank by lower CI bound of paired delta (penalises winner's-curse picks), require |O/E-1|<0.1
cfg["calibrated"] = (cfg.o_e.sub(1).abs() < 0.10) & cfg.cal_slope.between(0.9, 1.1)
best = {}
for fam in ["pubmed_only", "synthetic_only", "hybrid"]:
    c = cfg[(cfg.family == fam) & cfg.calibrated]
    best[fam] = c.sort_values(["d_auroc_lo", "auroc"], ascending=False).head(3)
top3 = pd.concat([best["pubmed_only"].head(1), best["synthetic_only"].head(1), best["hybrid"].head(1)])
cfg.sort_values("d_auroc_lo", ascending=False).to_csv(f"{OUT}/config_ranking.csv", index=False)
top3.to_csv(f"{OUT}/top3_models.csv", index=False)
# external-prior synthetic-only (h=1.0) separately
full = pd.read_csv(f"{OUT}/lattice_results.csv"); ext = full[(full.h == 1.0) & full.depth.isna()]
ext_s = ext.groupby(["parent", "V"]).agg(auroc=("auroc", "mean"), sd=("auroc", "std"), o_e=("o_e", "mean")).reset_index()
ext_s.to_csv(f"{OUT}/synthetic_only_external_priors.csv", index=False)

fam = d.groupby(["family", "h", "seed"]).d_auroc.max().reset_index()   # NB best-of-cell (optimistic) - reported as such
summ = {"robust_top": {k: v.round(5).to_dict("records") for k, v in best.items()},
        "selected_top3_for_app": top3.round(5).to_dict("records"),
        "external_priors_synthetic_only": ext_s.round(4).to_dict("records"),
        "hybrid_any_cell_ci_excludes_zero": int(((cfg.family == "hybrid") & (cfg.d_auroc_lo > 0)).sum()),
        "n_hybrid_cells": int((cfg.family == "hybrid").sum()),
        "volume_effect": d[d.family == "hybrid"].groupby("V").d_auroc.apply(lambda x: list(np.round(ci(x), 4))).to_dict(),
        "share_effect": d[d.family != "pubmed_only"].groupby("s").d_auroc.apply(lambda x: list(np.round(ci(x), 4))).to_dict()}
json.dump(summ, open(f"{OUT}/stage2_summary.json", "w"), indent=1, default=float)

# Fig: heatmaps share x heldout (delta AUROC vs PubMed-only) at V=9 per parent
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for i, p in enumerate(["P1_marginal", "P2_copula", "P3_structural"]):
    pv = d[(d.parent == p) & (d.V == 9)].pivot_table(index="h", columns="s", values="d_auroc")
    im = ax[i].imshow(pv.values, cmap="RdBu", vmin=-0.12, vmax=0.12, aspect="auto", origin="lower")
    ax[i].set_xticks(range(pv.shape[1]), [f"{int(s*100)}" for s in pv.columns], fontsize=7)
    ax[i].set_yticks(range(pv.shape[0]), [f"{int(h*100)}" for h in pv.index], fontsize=7)
    ax[i].set_title(p.replace("_", " "), fontsize=9); ax[i].set_xlabel("Synthetic share of training set (%)")
ax[0].set_ylabel("Real PubMed held out for testing (%)")
cb = fig.colorbar(im, ax=ax, fraction=0.02); cb.set_label("ΔAUROC vs PubMed-only")
fig.savefig(f"{OUT}/fig_heatmap_share_heldout_V9.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# Fig: volume and share dose-response (mean ΔAUROC ± 95% CI)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
C = {"P1_marginal": "#2a78d6", "P2_copula": "#eb6834", "P3_structural": "#1baf7a"}
for p, c in C.items():
    v = d[(d.parent == p) & (d.family == "hybrid")].groupby("V").d_auroc.apply(ci)
    ax[0].errorbar(range(len(v)), [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], color=c, marker="o", ms=5, lw=2, capsize=2, label=p.replace("_", " "))
    s = d[(d.parent == p) & (d.family != "pubmed_only")].groupby("s").d_auroc.apply(ci)
    ax[1].errorbar([x * 100 for x in s.index], [x[0] for x in s], yerr=[[x[0] - x[1] for x in s], [x[2] - x[0] for x in s]], color=c, marker="o", ms=5, lw=2, capsize=2)
ax[0].set_xticks(range(6), ["1×", "3×", "6×", "9×", "27×", "100×"]); ax[0].set_xlabel("Twin volume level (× real training size)")
ax[1].set_xlabel("Synthetic share (%)")
for a in ax: a.axhline(0, color="#a3a29d", ls="--", lw=1); a.set_ylabel("ΔAUROC vs PubMed-only"); a.grid(axis="y", color="#e4e3df")
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_volume_share_response.png", dpi=300); plt.close(fig)
print(json.dumps({k: summ[k] for k in ["selected_top3_for_app", "hybrid_any_cell_ci_excludes_zero", "n_hybrid_cells", "volume_effect", "external_priors_synthetic_only"]}, indent=1, default=float))

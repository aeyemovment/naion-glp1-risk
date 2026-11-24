"""Re-plot publication figures from saved results (no model re-fitting). Fixes label overlap, clipped text,
raw variable names and tick formatting found in the medRxiv v1 review. research_prototype=true. NOT for clinical use.
Run from the repository root: python src/replot_figures.py"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREEN, PURPLE, GOLD, RED, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e34948", "#a3a29d", "#0b0b0b"
C = {"logreg": BLUE, "rf": ORANGE, "xgb": GREEN}
NAMES = {"logreg": "Logistic regression (L2)", "rf": "Random forest", "xgb": "XGBoost"}
import sys; sys.path.insert(0, "src")
from feature_labels import nice
plain = FuncFormatter(lambda v, _: f"{v:g}")


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---- imaging what-if: labels clear of the range bars; full y labels -----------------------------------------
def imaging():
    s = pd.read_csv("results/stage16/imaging_whatif_summary.csv")
    c = s[s.effect_level == "central"]
    lo = s[s.effect_level == "low"].set_index("scenario").auroc; hi = s[s.effect_level == "high"].set_index("scenario").auroc
    lab = {sc: sc.replace("imaging alone (OCT CDR + drusen + OCTA + serial)", "imaging alone (OCT CDR + drusen\n+ OCTA + serial)") for sc in c.scenario}
    fig, ax = plt.subplots(figsize=(7.5, 3.9)); y_ = np.arange(len(c))[::-1]
    ax.barh(y_, c.auroc, color=[GREY if t.startswith("base") else (ORANGE if "OSA only" in t else BLUE) for t in c.scenario], height=0.6)
    for yy, t, a in zip(y_, c.scenario, c.auroc):
        right = a
        if t in lo.index:
            ax.plot([lo[t], hi[t]], [yy, yy], color=INK, lw=1.5); ax.plot([lo[t]] * 2, [yy - 0.15, yy + 0.15], color=INK, lw=1.5); ax.plot([hi[t]] * 2, [yy - 0.15, yy + 0.15], color=INK, lw=1.5)
            right = max(a, hi[t])
        ax.text(right + 0.006, yy, f"{a:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y_, [lab[t] for t in c.scenario], fontsize=7.5); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC (simulated; black bars = range across hypothetical effect sizes)"); clean(ax)
    fig.tight_layout(); fig.savefig("results/stage16/fig_imaging_whatif.png", dpi=300); plt.close(fig)


# ---- incidence strata: ordered, readable levels; plain log ticks --------------------------------------------
def incidence():
    d = pd.read_csv("results/stage10/incidence_stratified.csv")
    ORDER = {"age_band": ["<55", "55-59", ">=60", "unknown"], "indication": ["t2d", "iih"], "drug": ["semaglutide", "sema_or_tirz", "glp1ra_class"]}
    LAB = {"<55": "<55 y", "55-59": "55-59 y", ">=60": "≥60 y", "unknown": "Not\nreported", "t2d": "Type 2\ndiabetes", "iih": "IIH",
           "semaglutide": "Semaglutide", "sema_or_tirz": "Semaglutide or\ntirzepatide", "glp1ra_class": "GLP-1RA class\n(mixed)"}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for i, var in enumerate(["age_band", "indication", "drug"]):
        s_ = d[d.stratifier == var].set_index("level").reindex(ORDER[var]).reset_index(); x = np.arange(len(s_))
        ax[i].axhspan(2.3, 10.2, color="#e4e3df", alpha=0.6, zorder=0, label="Published background, age ≥50")
        ax[i].errorbar(x - 0.12, s_.nonglp_rate_per_100k_py, yerr=[s_.nonglp_rate_per_100k_py - s_.nonglp_lo, s_.nonglp_hi - s_.nonglp_rate_per_100k_py], fmt="o", color=BLUE, capsize=3, label="No GLP-1RA")
        ax[i].errorbar(x + 0.12, s_.glp_rate_per_100k_py, yerr=[s_.glp_rate_per_100k_py - s_.glp_lo, s_.glp_hi - s_.glp_rate_per_100k_py], fmt="o", color=ORANGE, capsize=3, label="GLP-1RA")
        ax[i].set_xticks(x, [LAB[v] for v in s_.level], fontsize=7.5); ax[i].set_xlim(-0.6, len(s_) - 0.4)
        ax[i].set_yscale("log"); ax[i].set_ylim(1.5, 200)
        ax[i].yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200])); ax[i].yaxis.set_major_formatter(plain); ax[i].yaxis.set_minor_locator(NullLocator())
        ax[i].set_title({"age_band": "A  Study mean age", "indication": "B  Indication", "drug": "C  Drug (exposed arm)"}[var]); clean(ax[i])
    ax[0].set_ylabel("NAION per 100,000 person-years (log)")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("results/stage10/fig_incidence_strata.png", dpi=300); plt.close(fig)


# ---- Stage 1 k-fold dots: y-axis labels ---------------------------------------------------------------------
def kfold():
    r = json.load(open("results/stage1/stage1_results.json")); M = ["logreg", "rf", "xgb"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, k in enumerate(["auroc", "pr_auc"]):
        for j, nm in enumerate(M):
            v = pd.DataFrame(r["kfold_10pct"][nm])[k]
            ax[i].scatter(np.full(len(v), j) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=C[nm], zorder=3)
            ax[i].hlines(v.mean(), j - 0.25, j + 0.25, color=INK, lw=2)
        ax[i].set_xticks(range(3), [NAMES[n] for n in M], fontsize=8); ax[i].grid(axis="y", color="#e4e3df"); clean(ax[i])
        ax[i].set_ylabel(["AUROC (90% test set)", "PR-AUC (90% test set)"][i])
        ax[i].set_title(["A  AUROC per fold (train 10%, test 90%)", "B  PR-AUC per fold"][i])
    fig.tight_layout(); fig.savefig("results/stage1/fig_kfold.png", dpi=300); plt.close(fig)


# ---- SHAP fold importance: only models whose bars are visible ------------------------------------------------
def shap_importance():
    st = {nm: pd.read_csv(f"results/stage1/shap_kfold_{nm}.csv", index_col=0) for nm in ["logreg", "xgb"]}
    top = st["xgb"].index[:10][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for j, nm in enumerate(["logreg", "xgb"]):
        s = st[nm].reindex(top)
        ax.barh(np.arange(len(top)) + (j - 0.5) * 0.36, s.mean_abs_shap, height=0.34, xerr=s.sd, color=C[nm], error_kw={"lw": 1, "ecolor": "#52514e"}, label=NAMES[nm])
    ax.set_yticks(np.arange(len(top)), [nice(t) for t in top]); ax.set_xlabel("Mean absolute SHAP value (log-odds), mean ± SD over 10 folds")
    ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.grid(axis="x", color="#e4e3df"); ax.set_axisbelow(True); clean(ax)
    fig.tight_layout(); fig.savefig("results/stage1/fig_shap_kfold_importance.png", dpi=300); plt.close(fig)


def beeswarm():
    """XGBoost SHAP beeswarm on the 90% hold-out (same sample as stage1_models.py), with readable feature labels."""
    import shap
    from common import ipd, X_all, fit
    from splits import holdout_10_90
    tr, te = holdout_10_90(ipd, seed=2026); m = fit("xgb", tr); rng = np.random.default_rng(0)
    idx = np.concatenate([np.repeat(te.index[te.y == 1], te.n[te.y == 1]),
                          rng.choice(te.index[te.y == 0], 6000, p=(te.n[te.y == 0] / te.n[te.y == 0].sum()).to_numpy())])
    sv = shap.TreeExplainer(m)(X_all.loc[idx]); sv.feature_names = [nice(c) for c in X_all.columns]
    plt.figure(); shap.plots.beeswarm(sv, max_display=12, show=False, plot_size=(7, 4.2))
    plt.xlabel("SHAP value (impact on model output, log-odds)"); plt.tight_layout()
    plt.savefig("results/stage1/fig_shap_beeswarm_xgb.png", dpi=300); plt.close("all")


# ---- published-estimate forest: full x label, plain ticks -----------------------------------------------------
def forest():
    fp = pd.read_csv("results/stage1b/forest_table.csv"); refs = json.load(open("data/raw/refs.json"))
    POP = {"t2d": "T2D", "t2d_or_obesity": "T2D or obesity", "iih": "IIH", "liver_tx": "liver transplant", "obesity": "obesity", "other": "mixed"}
    def lab(r):
        fa = r.first_author if isinstance(r.first_author, str) and r.first_author.strip() else ""
        if not fa:
            au = refs.get(str(int(r.source_pmid)), {}).get("authors", []); fa = au[0].split()[0] if au else f"PMID {int(r.source_pmid)}"
        return f"{fa} {int(r.year) if pd.notna(r.year) else ''} ({POP.get(r.population, r.population)})"
    fp = fp.assign(se=(np.log(fp.ci_high) - np.log(fp.ci_low)) / 3.92).sort_values("effect_estimate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.8, 0.28 * len(fp) + 0.9)); yy = np.arange(len(fp))
    ax.hlines(yy, fp.ci_low, fp.ci_high, color="#52514e", lw=1.2)
    ax.scatter(fp.effect_estimate, yy, s=10 + 40 * (1 / fp.se) / (1 / fp.se).max(), color=BLUE, zorder=3)
    ax.axvline(1, color=GREY, ls="--", lw=1); ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.25, 0.5, 1, 2, 4, 10])); ax.xaxis.set_major_formatter(plain); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(yy, [lab(r) for r in fp.itertuples()], fontsize=7); ax.set_ylim(-0.7, len(fp) - 0.3)
    ax.set_xlabel("Published ratio estimate (log scale)\nmost precise estimate per study"); clean(ax)
    fig.tight_layout(); fig.savefig("results/stage1b/fig_forest.png", dpi=300); plt.close(fig)


# ---- GNN tuning: integer k ticks, readable edge-weight names --------------------------------------------------
def gnn():
    g = pd.read_csv("results/stage6c/grid_folds.csv"); cf = g[g.fold >= 5]
    cc = {"uniform": (BLUE, "uniform"), "gauss0.5": (ORANGE, "Gaussian, bw 0.5"), "gauss1": (GREEN, "Gaussian, bw 1"), "gauss2": (GOLD, "Gaussian, bw 2")}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for e_, (col, nm) in cc.items():
        q = cf[(cf.arch == "sage") & (cf.edge_mode == e_)].groupby("L").auroc.mean(); ax[0].plot(q.index, q.values, "o-", color=col, lw=2, ms=4, label=nm)
    q = cf[cf.arch == "gat"].groupby("L").auroc.mean(); ax[0].plot(q.index, q.values, "s--", color=PURPLE, lw=2, ms=4, label="GAT (learned)")
    ax[0].set(xlabel="GNN depth (layers)", ylabel="AUROC (untouched folds 5-9)", title="A  Depth × edge weighting"); ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    q = cf[cf.arch == "sage"].groupby("study_w").auroc.agg(["mean", "std", "size"]); ax[1].errorbar(q.index, q["mean"], yerr=1.96 * q["std"] / np.sqrt(q["size"]), fmt="o-", color=BLUE, lw=2, capsize=2)
    ax[1].set(xlabel="Same-study edge weight (× kNN edge)", ylabel="AUROC", title="B  Study-link weight"); ax[1].set_xticks(sorted(q.index), [f"{v:g}" for v in sorted(q.index)])
    q = cf[cf.arch == "sage"].groupby("k").auroc.agg(["mean", "std", "size"]); ax[2].errorbar(q.index, q["mean"], yerr=1.96 * q["std"] / np.sqrt(q["size"]), fmt="o-", color=GREEN, lw=2, capsize=2)
    ax[2].set(xlabel="Neighbours per node (k)", ylabel="AUROC", title="C  Neighbourhood size"); ax[2].set_xticks(sorted(q.index), [f"{int(v)}" for v in sorted(q.index)])
    for a_ in ax: a_.grid(axis="y", color="#e4e3df"); clean(a_); a_.ticklabel_format(axis="y", useOffset=False)
    fig.tight_layout(); fig.savefig("results/stage6c/fig_gnn_depth_weights.png", dpi=300); plt.close(fig)


# ---- Stage 7: legend clear of the chance line; clear population names -----------------------------------------
def stage7():
    df = pd.read_csv("results/stage7/stage7_all.csv"); kf = df[df.tag.str.startswith("fold")]
    MODELS = ["logreg", "rf", "xgb", "gnn", "ensemble"]; CC = {"internal_nonGLP": BLUE, "external_GLP": ORANGE}
    LBL = {"internal_nonGLP": "Non-users (internal)", "external_GLP": "GLP-1RA users (external)"}
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for j, popn in enumerate(["internal_nonGLP", "external_GLP"]):
        for i, m in enumerate(MODELS):
            v = kf[(kf.model == m) & (kf.population == popn)].auroc
            ax[0].scatter(np.full(len(v), i) + (j - 0.5) * 0.3 + np.linspace(-0.05, 0.05, len(v)), v, s=16, color=CC[popn], label=LBL[popn] if i == 0 else None)
    ax[0].axhline(0.5, color=GREY, ls="--", lw=1); ax[0].set_xticks(range(5), ["LR", "RF", "XGB", "GNN", "Ens."]); ax[0].set_ylabel("AUROC")
    ax[0].set_title("A  Discrimination, 10 folds"); ax[0].set_ylim(0.47, 0.72); ax[0].legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    oe = kf[kf.population == "external_GLP"].groupby("model").agg(o=("o_e", "mean"), lo=("oe_ci_lo", "mean"), hi=("oe_ci_hi", "mean")).reindex(MODELS)
    ax[1].errorbar(range(5), oe.o, yerr=[oe.o - oe.lo, oe.hi - oe.o], fmt="o", color=ORANGE, capsize=3, ms=6)
    ax[1].axhline(1, color=GREY, ls="--", lw=1); ax[1].set_xticks(range(5), ["LR", "RF", "XGB", "GNN", "Ens."]); ax[1].set_ylabel("Observed / expected in GLP-1RA users")
    ax[1].set_title("B  Under-prediction by background model")
    for a_ in ax: clean(a_); a_.grid(axis="y", color="#e4e3df")
    fig.tight_layout(); fig.savefig("results/stage7/fig_stage7_external.png", dpi=300); plt.close(fig)


# ---- Stage 8: coincident recalibration lines made visible; readable method names ------------------------------
def stage8():
    g = pd.read_csv("results/stage8/finetune_summary.csv")
    CM = {"recal_intercept": (BLUE, "Recalibration, intercept", "o-", 6), "recal_logistic": (GREEN, "Recalibration, intercept + slope", "s:", 4),
          "xgb_continue": (ORANGE, "XGBoost continued boosting", "o-", 4), "gnn_joint": (PURPLE, "GNN joint training", "o-", 4),
          "pooled_refit": (GOLD, "Pooled refit", "o-", 4), "glp_only": (RED, "GLP-1RA users only", "o-", 4)}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for i, ft in enumerate(["real", "synthetic", "hybrid"]):
        for m, (c, nm, sty, ms) in CM.items():
            q_ = g[(g.pretrain_source.isin(["real", "none"] if m == "glp_only" else ["real"])) & (g.finetune_source == ft) & (g.method == m)].groupby("glp_frac").glp_auroc.mean()
            ax[i].plot(q_.index * 100, q_.values, sty, color=c, lw=2, ms=ms, label=nm, zorder=3 if m.startswith("recal") else 2)
        base = g[(g.pretrain_source == "real") & (g.method == "pretrain_only")].glp_auroc.mean()
        ax[i].axhline(base, color=INK, ls="--", lw=1, label="Pretrained on non-users, no update")
        ax[i].set_title(f"Fine-tuning data: {ft}"); ax[i].set_xlabel("GLP-1RA patients used for fine-tuning (%)"); ax[i].grid(axis="y", color="#e4e3df"); clean(ax[i])
    ax[0].set_ylabel("AUROC in held-out GLP-1RA users")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig("results/stage8/fig_finetune.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    for fn in (imaging, incidence, kfold, shap_importance, beeswarm, forest, gnn, stage7, stage8):
        fn(); print("ok", fn.__name__)

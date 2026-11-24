"""Stage 3 analysis - full factorial. Paired (same split, same model) ΔAUROC vs PubMed-only,
BH-FDR corrected; factor main effects; family learning curves; robust top-3 for the app.
research_prototype=true; twins synthetic_only=true; compliance_ref=Addendum_5_DT9. NOT for clinical use."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import ttest_1samp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
OUT = "results/stage3"
full = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/factorial_tf*.csv"))], ignore_index=True)
if "depth" not in full: full["depth"] = np.nan
base = full[full.depth.isna()].copy()
MOD = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost", "ensemble": "Ensemble (mean)"}
CM = {"logreg": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a", "ensemble": "#6250d6"}

ref = base[(base.genset == "none") & (base.V == 1)].set_index(["train_frac", "seed", "model"])
d = base[(base.train_frac > 0)].copy()
k = list(zip(d.train_frac, d.seed, d.model))
d["d_auroc"] = d.auroc.to_numpy() - ref.loc[k, "auroc"].to_numpy()
d["family"] = np.where(d.genset == "none", "pubmed_only", np.where(d.s == 1.0, "synthetic_only", "hybrid"))

# ---- per-(model, train_frac, config) paired test, BH-FDR
grp = d[d.family != "pubmed_only"].groupby(["model", "train_frac", "family", "genset", "source", "V", "s"])
st = grp.d_auroc.agg(["mean", "std", "size"]).reset_index()
st["p"] = grp.d_auroc.apply(lambda x: ttest_1samp(x, 0, alternative="greater").pvalue if x.std() > 0 else 1.0).to_numpy()
st = st.sort_values("p"); m_ = len(st); st["q_bh"] = np.minimum.accumulate((st.p * m_ / np.arange(1, m_ + 1))[::-1])[::-1].clip(upper=1)
aux = grp.agg(auroc=("auroc", "mean"), o_e=("o_e", "mean"), cal_slope=("cal_slope", "mean"), rho=("shap_rho_vs_pubmed", "mean")).reset_index()
st = st.merge(aux, on=["model", "train_frac", "family", "genset", "source", "V", "s"])
st.to_csv(f"{OUT}/paired_tests_all_configs.csv", index=False)
sig = st[(st.q_bh < 0.05) & (st["mean"] > 0)]

# ---- robust config ranking across ALL training fractions (consistency), calibrated
rob = d.groupby(["model", "family", "genset", "source", "V", "s"]).agg(auroc=("auroc", "mean"), d=("d_auroc", "mean"),
      d_sd=("d_auroc", "std"), n=("d_auroc", "size"), o_e=("o_e", "mean"), cal=("cal_slope", "mean"), rho=("shap_rho_vs_pubmed", "mean"),
      frac_pos=("d_auroc", lambda x: (x > 0).mean())).reset_index()
rob["d_lo"] = rob.d - 1.96 * rob.d_sd / np.sqrt(rob.n)
rob["calibrated"] = (rob.o_e.sub(1).abs() < 0.10) & rob.cal.between(0.9, 1.1)
rob.sort_values("auroc", ascending=False).to_csv(f"{OUT}/robust_config_ranking.csv", index=False)
top = {}
for fam in ["pubmed_only", "hybrid", "synthetic_only"]:
    c = rob[(rob.family == fam) & rob.calibrated]
    if fam == "pubmed_only": c = c[c.V == 1]
    top[fam] = c.sort_values(["auroc", "d_lo"], ascending=False).head(5)
top3 = pd.concat([t.head(1) for t in top.values()]); top3.to_csv(f"{OUT}/top3_for_app.csv", index=False)

# ---- synthetic-only with NO real data (0% real, published priors)
z = base[base.train_frac == 0].groupby(["model", "genset", "V"]).agg(auroc=("auroc", "mean"), sd=("auroc", "std"), o_e=("o_e", "mean")).reset_index()
z.to_csv(f"{OUT}/synthetic_only_zero_real.csv", index=False)

# ---- factor main effects on ΔAUROC (hybrid + synthetic), per model
eff = {}
for f in ["genset", "source", "V", "s"]:
    e = d[d.family != "pubmed_only"].groupby(["model", f]).d_auroc.agg(["mean", "std", "size"]).reset_index()
    e["lo"] = e["mean"] - 1.96 * e["std"] / np.sqrt(e["size"]); e["hi"] = e["mean"] + 1.96 * e["std"] / np.sqrt(e["size"])
    eff[f] = e; e.to_csv(f"{OUT}/main_effect_{f}.csv", index=False)

# ---- figures
lc = []
for (mdl, tf), g in d.groupby(["model", "train_frac"]):
    lc.append({"model": mdl, "tf": tf, "fam": "PubMed-only", "auroc": g[(g.family == "pubmed_only") & (g.V == 1)].auroc.mean() if len(g[(g.family == "pubmed_only") & (g.V == 1)]) else ref.xs(mdl, level="model").loc[tf].auroc.mean()})
    for fam, lab in [("hybrid", "Best hybrid"), ("synthetic_only", "Best synthetic-only")]:
        lc.append({"model": mdl, "tf": tf, "fam": lab, "auroc": g[g.family == fam].groupby(["genset", "source", "V", "s"]).auroc.mean().max()})
lc = pd.DataFrame(lc)
po = ref.reset_index().groupby(["model", "train_frac"]).auroc.mean()
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
for i, mdl in enumerate(MOD):
    ax[i].plot(po[mdl].index * 100, po[mdl].values, color="#0b0b0b", marker="o", ms=4, lw=2, label="PubMed-only")
    for lab, c in [("Best hybrid", "#eb6834"), ("Best synthetic-only", "#2a78d6")]:
        q = lc[(lc.model == mdl) & (lc.fam == lab)].sort_values("tf"); ax[i].plot(q.tf * 100, q.auroc, color=c, marker="o", ms=4, lw=2, label=lab + " (best-of-grid, optimistic)")
    zz = z[z.model == mdl].auroc.max(); ax[i].scatter([0], [zz], marker="D", s=40, color="#6250d6", zorder=4, label="Synthetic-only, 0% real (published priors)")
    ax[i].set_title(MOD[mdl]); ax[i].set_xlabel("Real training fraction (%)"); ax[i].grid(axis="y", color="#e4e3df")
ax[0].set_ylabel("AUROC, held-out real PubMed"); ax[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_learning_curves_families.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
for i, f in enumerate(["genset", "source", "V", "s"]):
    e = eff[f]; levels = list(dict.fromkeys(e[f]))
    for j, mdl in enumerate(MOD):
        q = e[e.model == mdl].set_index(f).reindex(levels); x = np.arange(len(levels)) + (j - 1.5) * 0.15
        ax[i].errorbar(x, q["mean"], yerr=[q["mean"] - q.lo, q.hi - q["mean"]], fmt="o", ms=4, color=CM[mdl], capsize=2, label=MOD[mdl])
    GL = lambda g: "+".join({"P1": "Marg", "P2": "Cop", "P3": "Struct"}.get(x, x) for x in str(g).split("+"))
    ax[i].set_xticks(range(len(levels)), [GL(l) if f == "genset" else (str(l) if f != "s" else f"{int(l*100)}") for l in levels], fontsize=7, rotation=45 if f == "genset" else 0)
    ax[i].axhline(0, color="#a3a29d", ls="--", lw=1); ax[i].set_title({"genset": "Generator set", "source": "Prior source", "V": "Volume (×)", "s": "Synthetic share (%)"}[f])
    ax[i].grid(axis="y", color="#e4e3df")
ax[0].set_ylabel("Mean ΔAUROC vs PubMed-only")
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/fig_main_effects.png", dpi=300); plt.close(fig)

summ = {"n_rows": int(len(full)), "n_configs_tested": int(m_), "n_significant_bh05": int(len(sig)),
        "significant_configs": sig.head(30).round(5).to_dict("records"),
        "top3_for_app": top3.round(5).to_dict("records"), "top5_per_family": {k: v.round(5).to_dict("records") for k, v in top.items()},
        "zero_real_best": z.loc[z.groupby("model").auroc.idxmax()].round(4).to_dict("records"),
        "recursion": [r for f in sorted(glob.glob(f"{OUT}/*_recursion.json")) for r in json.load(open(f))]}
json.dump(summ, open(f"{OUT}/stage3_summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 230)
print("rows", len(full), "configs tested", m_, "significant after BH:", len(sig))
print(sig.head(20)[["model", "train_frac", "family", "genset", "source", "V", "s", "mean", "q_bh", "auroc", "o_e"]].round(4).to_string(index=False))
print(top3[["model", "family", "genset", "source", "V", "s", "auroc", "d", "d_lo", "frac_pos", "o_e", "cal", "rho"]].round(4).to_string(index=False))
for f in ["genset", "source", "V"]: print(eff[f].pivot(index=f, columns="model", values="mean").round(4).to_string())
print(pd.DataFrame(summ["zero_real_best"]).to_string())
print("recursion gains >= .005:", sum(r["gain"] >= 0.005 for r in summ["recursion"]), "/", len(summ["recursion"]))

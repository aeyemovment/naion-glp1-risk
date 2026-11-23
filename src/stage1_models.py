"""Stage 1 runner - see common.py for models/metrics. research_prototype=true. NOT for clinical use."""
from common import *  # shared models, metrics, SHAP, data
res = {"meta": {"research_prototype": True, "synthetic_only": False, "compliance_ref": "Addendum_5_DT9",
                "n_patients": int(ipd.n.sum()), "n_events": int(ipd.loc[ipd.y == 1, "n"].sum()),
                "n_studies": int(ipd.study.nunique())}}
MODELS = ["logreg", "rf", "xgb"]

# ---- A. 10/90 holdout
tr, te = holdout_10_90(ipd, seed=2026)
res["holdout_10_90"] = {}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for nm in MODELS:
    m = fit(nm, tr); p = predict(nm, m, te); _, y, w = XY(te)
    res["holdout_10_90"][nm] = metrics(y, p, w)
    fpr, tpr, _ = roc_curve(y, p, sample_weight=w); ax[0].plot(fpr, tpr, lw=2, color=C[nm], label=f"{NAMES[nm]} ({res['holdout_10_90'][nm]['auroc']:.3f})")
    pr, rc, _ = precision_recall_curve(y, p, sample_weight=w); ax[1].plot(rc, pr, lw=2, color=C[nm])
    # calibration by decile of predicted risk
    q = pd.DataFrame({"p": p, "y": y, "w": w}).sort_values("p")
    q["bin"] = np.minimum((q.w.cumsum() / q.w.sum() * 10).astype(int), 9)
    g = q.groupby("bin").apply(lambda d: pd.Series({"pred": np.average(d.p, weights=d.w), "obs": np.average(d.y, weights=d.w)}))
    ax[2].plot(g.pred * 1000, g.obs * 1000, "o-", ms=5, lw=2, color=C[nm])
ax[0].plot([0, 1], [0, 1], ls="--", color="#a3a29d", lw=1); ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title="A  ROC (90% held-out)")
ax[0].legend(frameon=False, fontsize=7, loc="lower right")
ax[1].axhline(res["holdout_10_90"]["xgb"]["prevalence"], ls="--", color="#a3a29d", lw=1)
ax[1].set(xlabel="Recall", ylabel="Precision", title="B  Precision-recall")
lim = ax[2].get_xlim()[1]; ax[2].plot([0, lim], [0, lim], ls="--", color="#a3a29d", lw=1)
ax[2].set(xlabel="Predicted risk per 1,000", ylabel="Observed risk per 1,000", title="C  Calibration (deciles)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_holdout_roc_pr_cal.png", dpi=300); plt.close(fig)

# ---- B. 10 folds, each a different 10% train
res["kfold_10pct"] = {nm: [] for nm in MODELS}
shap_folds = {nm: [] for nm in MODELS}
for f, tr, te in kfold_10pct(ipd, k=10, seed=7):
    for nm in MODELS:
        m = fit(nm, tr); p = predict(nm, m, te); _, y, w = XY(te)
        res["kfold_10pct"][nm].append({"fold": f, **metrics(y, p, w)})
        imp, _ = shap_importance(nm, m, te); shap_folds[nm].append(imp.rename(f))
summary = {}
for nm in MODELS:
    d = pd.DataFrame(res["kfold_10pct"][nm])
    summary[nm] = {k: {"mean": d[k].mean(), "sd": d[k].std(), "min": d[k].min(), "max": d[k].max()}
                   for k in ["auroc", "pr_auc", "brier", "cal_slope", "o_e"]}
res["kfold_summary"] = summary

# SHAP stability across folds
stab = {}
for nm in MODELS:
    S = pd.concat(shap_folds[nm], axis=1)
    ranks = S.rank(ascending=False)
    stab[nm] = pd.DataFrame({"mean_abs_shap": S.mean(1), "sd": S.std(1),
                             "top5_in_folds": (ranks <= 5).sum(1)}).sort_values("mean_abs_shap", ascending=False)
    stab[nm].to_csv(f"{OUT}/shap_kfold_{nm}.csv")
res["shap_stability"] = {nm: stab[nm].round(5).reset_index(names="feature").to_dict("records") for nm in MODELS}
consensus = pd.concat([stab[nm].mean_abs_shap / stab[nm].mean_abs_shap.sum() for nm in MODELS], axis=1).mean(1).sort_values(ascending=False)
stable = [f for f in consensus.index if all(stab[nm].loc[f, "top5_in_folds"] >= 8 for nm in ["xgb", "rf"])]
res["selected_features"] = {"consensus_normalized_importance": consensus.round(4).to_dict(), "stable_top5_ge8of10_folds": stable}

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
pos = np.arange(len(MODELS))
for i, k in enumerate(["auroc", "pr_auc"]):
    data = [pd.DataFrame(res["kfold_10pct"][nm])[k] for nm in MODELS]
    for j, nm in enumerate(MODELS):
        ax[i].scatter(np.full(10, j) + np.linspace(-0.12, 0.12, 10), data[j], s=22, color=C[nm], zorder=3)
        ax[i].hlines(data[j].mean(), j - 0.25, j + 0.25, color="#0b0b0b", lw=2)
    ax[i].set_xticks(pos, [NAMES[n] for n in MODELS], fontsize=8)
    ax[i].set_title(["A  AUROC per fold (train 10%, test 90%)", "B  PR-AUC per fold"][i]); ax[i].grid(axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_kfold.png", dpi=300); plt.close(fig)

top = stab["xgb"].index[:10][::-1]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for j, nm in enumerate(MODELS):
    s = stab[nm].reindex(top)
    ax.barh(np.arange(len(top)) + (j - 1) * 0.26, s.mean_abs_shap, height=0.24, xerr=s.sd, color=C[nm],
            error_kw={"lw": 1, "ecolor": "#52514e"}, label=NAMES[nm])
ax.set_yticks(np.arange(len(top)), top); ax.set_xlabel("Mean |SHAP| (log-odds), mean ± SD over 10 folds")
ax.legend(frameon=False, fontsize=7); ax.grid(axis="x")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_shap_kfold_importance.png", dpi=300); plt.close(fig)

# SHAP beeswarm for XGB on 90% holdout test, 6000 patients sampled proportional to counts (all events kept)
tr, te = holdout_10_90(ipd, seed=2026)
m = fit("xgb", tr)
rng = np.random.default_rng(0)
idx = np.concatenate([np.repeat(te.index[te.y == 1], te.n[te.y == 1]),
                      rng.choice(te.index[te.y == 0], 6000, p=(te.n[te.y == 0] / te.n[te.y == 0].sum()).to_numpy())])
Xb = X_all.loc[idx]
ex = shap.TreeExplainer(m); sv = ex(Xb)
plt.figure(); shap.plots.beeswarm(sv, max_display=12, show=False, plot_size=(7, 4.2))
plt.tight_layout(); plt.savefig(f"{OUT}/fig_shap_beeswarm_xgb.png", dpi=300); plt.close("all")

# ---- C. leave-one-study-out
res["loso"] = {nm: [] for nm in MODELS}
for s in sorted(ipd.study.unique()):
    tr, te = ipd[ipd.study != s], ipd[ipd.study == s]
    for nm in MODELS:
        m = fit(nm, tr); p = predict(nm, m, te); _, y, w = XY(te)
        r = metrics(y, p, w); r["study"] = s; res["loso"][nm].append(r)
res["loso_summary"] = {nm: {k: float(pd.DataFrame(res["loso"][nm])[k].median()) for k in ["auroc", "o_e", "brier"]} for nm in MODELS}

json.dump(res, open(f"{OUT}/stage1_results.json", "w"), indent=1, default=float)
print(json.dumps({"holdout": res["holdout_10_90"], "kfold": summary, "loso_median": res["loso_summary"],
                  "selected": res["selected_features"]}, indent=1, default=lambda x: round(float(x), 4)))

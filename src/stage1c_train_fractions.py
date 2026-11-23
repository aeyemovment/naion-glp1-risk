"""Stage 1c - training-fraction sweep (learning curve), PubMed-only.

Train fraction f in {10, 20, ..., 90}% of patients; test on the remaining (1-f).
R=10 repeated random patient-level splits per f (Monte-Carlo CV) for LR / RF / XGB.
Also: for each f, SHAP importance per repeat -> rank stability (Kendall W) and top-5 frequency.

research_prototype=true; synthetic_only=false; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
from common import *
from splits import holdout_10_90
from joblib import Parallel, delayed

OUT2 = "results/stage1c"; import os; os.makedirs(OUT2, exist_ok=True)
FRACS = [i / 10 for i in range(1, 10)]
REPEATS = 10
MODELS = ["logreg", "rf", "xgb"]


def one(f, rep):
    tr, te = holdout_10_90(ipd, train_frac=f, seed=1000 * rep + int(f * 100))
    out = []
    for nm in MODELS:
        m = fit(nm, tr); p = predict(nm, m, te); _, y, w = XY(te)
        imp, _ = shap_importance(nm, m, te)
        out.append({"train_frac": f, "repeat": rep, "model": nm, "n_train": int(tr.n.sum()),
                    "events_train": int(tr.loc[tr.y == 1, "n"].sum()), **metrics(y, p, w),
                    **{f"shap_{k}": v for k, v in imp.items()}})
    return out


rows = [r for chunk in Parallel(n_jobs=-1)(delayed(one)(f, r) for f in FRACS for r in range(REPEATS)) for r in chunk]
df = pd.DataFrame(rows); df.to_csv(f"{OUT2}/train_fraction_sweep.csv", index=False)

agg = df.groupby(["model", "train_frac"]).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), pr_auc=("pr_auc", "mean"),
      brier=("brier", "mean"), cal_slope=("cal_slope", "mean"), o_e=("o_e", "mean"), o_e_sd=("o_e", "std")).reset_index()
agg.to_csv(f"{OUT2}/train_fraction_summary.csv", index=False)

# SHAP rank stability per fraction (Kendall's W across repeats) + top-5 frequency
from scipy.stats import rankdata
SH = [c for c in df.columns if c.startswith("shap_")]
stab = []
for (nm, f), g in df.groupby(["model", "train_frac"]):
    M = g[SH].to_numpy(); R = np.vstack([rankdata(-r) for r in M]); m_, n_ = R.shape
    S = ((R.sum(0) - R.sum(0).mean()) ** 2).sum(); W = 12 * S / (m_ ** 2 * (n_ ** 3 - n_))
    top5 = pd.Series((R <= 5).mean(0), index=[c[5:] for c in SH])
    stab.append({"model": nm, "train_frac": f, "kendall_W": W, **{f"top5_{k}": v for k, v in top5.items()}})
stab = pd.DataFrame(stab); stab.to_csv(f"{OUT2}/shap_stability_by_fraction.csv", index=False)

# Figure: learning curves (AUROC, O/E) + SHAP Kendall W
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for nm in MODELS:
    a = agg[agg.model == nm]; x = a.train_frac * 100
    ax[0].errorbar(x, a.auroc, yerr=1.96 * a.auroc_sd / np.sqrt(REPEATS), color=C[nm], marker="o", ms=5, lw=2, capsize=2, label=NAMES[nm])
    ax[1].errorbar(x, a.o_e, yerr=1.96 * a.o_e_sd / np.sqrt(REPEATS), color=C[nm], marker="o", ms=5, lw=2, capsize=2)
    s = stab[stab.model == nm]; ax[2].plot(s.train_frac * 100, s.kendall_W, color=C[nm], marker="o", ms=5, lw=2)
ax[0].set(xlabel="Training fraction (%)", ylabel="AUROC (held-out remainder)", title="A  Discrimination")
ax[1].axhline(1, color="#a3a29d", ls="--", lw=1); ax[1].set(xlabel="Training fraction (%)", ylabel="Observed / expected", title="B  Calibration-in-the-large")
ax[2].set(xlabel="Training fraction (%)", ylabel="Kendall's W (SHAP ranks)", title="C  Explanation stability", ylim=(0, 1.02))
ax[0].legend(frameon=False, fontsize=7)
for a_ in ax: a_.grid(axis="y", color="#e4e3df")
fig.tight_layout(); fig.savefig(f"{OUT2}/fig_train_fraction_curves.png", dpi=300); plt.close(fig)

pd.set_option("display.width", 200)
print(agg.round(4).to_string(index=False))
print(stab[["model", "train_frac", "kendall_W"]].round(3).pivot(index="train_frac", columns="model", values="kendall_W").to_string())

"""Stage 7 - background-risk model trained on NON-GLP-1 patients only; external validation on ALL GLP-1RA-exposed patients.

Question: can a model of traditional/background NAION risk (learned without any GLP-1 exposure) separate GLP-1 users who
develop NAION from GLP-1 users who do not, and how far does it under-predict their risk (O/E = excess risk)?

Training population : comparator arms (glp1ra_exposed == 0) of the reconstructed cohorts
External population : exposed arms (glp1ra_exposed == 1), all of them, never seen in training
Features            : indication, region, data source, outcome definition, follow-up, study mean age, % female (+ missing flags);
                      exposure and drug are excluded by design (they are constant/unknown in the training population)
Models              : logreg, rf, xgb, gnn (Stage 6c tuned config), mean ensemble
Paradigms (internal, on non-GLP): 10/90 hold-out; 10 disjoint 10% folds; training fraction 10..90% x 10 repeats; LOSO
External: every fitted model scored on all GLP-exposed patients: AUROC (Hanley-McNeil vs 0.5), PR-AUC, O/E with exact Poisson CI,
          calibration slope, threshold metrics (Youden / top 10% / top 20% fixed on non-GLP training) with P values.
Reference: same models trained on BOTH arms (GLP-aware, exposure + therapy included) evaluated on the same GLP-exposed test rows.
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
from common import *
from splits import holdout_10_90, kfold_10pct
from gnn import fit_predict_gnn_cfg
from scipy import stats
from joblib import Parallel, delayed
import os
OUT7 = "results/stage7"; os.makedirs(OUT7, exist_ok=True)
src = open("src/stage5b_classification.py").read(); exec(src[src.index("def w_quantile"):src.index("# ---- (i) hold-out 10/90")])

BG_FEATS = [c for c in COLS if not (c.startswith("therapy_") or c == "glp1ra_exposed")]
GNN_CFG = dict(arch="sage", L=2, k=10, edge_mode="uniform", study_w=1.0)
if os.path.exists("results/stage6c/summary.json"):
    b = json.load(open("results/stage6c/summary.json"))["best_tuned"]
    GNN_CFG = dict(arch=b["arch"], L=int(b["L"]), k=int(b["k"]), edge_mode=b["edge_mode"], study_w=float(b["study_w"]))
nonglp = ipd[ipd.glp1ra_exposed == 0]; glp = ipd[ipd.glp1ra_exposed == 1]


def XYb(df, cols):
    X = X_all.loc[df.index, cols]; return X, df.y.to_numpy(), df.n.to_numpy().astype(float)


def fit_bg(name, tr, cols):
    X, y, w = XYb(tr, cols)
    if name == "logreg":
        mu, sd = X.mean(), X.std().replace(0, 1)
        m = LogisticRegression(C=1.0, max_iter=5000).fit((X - mu) / sd, y, sample_weight=w * (np.sum(w * y) / np.sum(w)))
        return lambda df: m.predict_proba((XYb(df, cols)[0] - mu) / sd)[:, 1], m
    if name == "rf":
        m = RandomForestClassifier(n_estimators=300, max_depth=4, max_features="sqrt", bootstrap=False, min_weight_fraction_leaf=0.01, random_state=0, n_jobs=1).fit(X, y, sample_weight=w)
    else:
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist", base_score=float(np.average(y, weights=w)), verbosity=0, n_jobs=1, random_state=0).fit(X, y, sample_weight=w)
    return lambda df: m.predict_proba(XYb(df, cols)[0])[:, 1], m


def predict_all(tr, tests, cols):
    """Fit all base models on tr; return {model: [p for each test df]} incl. GNN and mean ensemble."""
    out = {}
    for nm in ["logreg", "rf", "xgb"]:
        f_, _ = fit_bg(nm, tr, cols); out[nm] = [f_(t) for t in tests]
    Xtr, ytr, wtr = XYb(tr, cols)
    out["gnn"] = [fit_predict_gnn_cfg(Xtr, ytr, wtr, XYb(t, cols)[0], tr.study.to_numpy(), t.study.to_numpy(), **GNN_CFG) for t in tests]
    out["ensemble"] = [np.mean([out[m][i] for m in ["logreg", "rf", "xgb", "gnn"]], 0) for i in range(len(tests))]
    return out


def ext_metrics(p, df, p_tr=None, tr=None):
    y, w = df.y.to_numpy(), df.n.to_numpy().astype(float)
    r = metrics(y, p, w)
    O, E = (w * y).sum(), (w * p).sum()
    r["oe_ci_lo"], r["oe_ci_hi"] = stats.chi2.ppf(0.025, 2 * O) / 2 / E, stats.chi2.ppf(0.975, 2 * (O + 1)) / 2 / E
    n1, n0 = (w * y).sum(), (w * (1 - y)).sum(); A = r["auroc"]; q1, q2 = A / (2 - A), 2 * A ** 2 / (1 + A)
    se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A ** 2) + (n0 - 1) * (q2 - A ** 2)) / (n1 * n0)); r["auroc_lo"], r["auroc_hi"], r["p_auroc_vs_0.5"] = A - 1.96 * se, A + 1.96 * se, stats.norm.sf((A - .5) / se)
    if p_tr is not None:
        ytr, wtr = tr.y.to_numpy(), tr.n.to_numpy().astype(float)
        for tn, t in thresholds(p_tr, ytr, wtr).items():
            c = conf(p, y, w, t); r.update({f"{tn}_{k}": c[k] for k in ["sensitivity", "p_sens_vs_chance", "specificity", "p_spec_vs_chance", "accuracy", "p_acc_vs_nir", "ppv", "p_ppv_vs_prevalence", "npv", "p_npv_vs_1minusprev", "lr_pos", "net_benefit"]})
    return r


rows = []
def run_split(tag, tr_bg, te_bg, tr_full=None, te_glp=None):
    """tr_bg/te_bg: non-GLP internal split; external = all GLP-exposed (or te_glp if given)."""
    ext = glp if te_glp is None else te_glp
    P = predict_all(tr_bg, [te_bg, ext, tr_bg], BG_FEATS)
    out = []
    for m, (p_int, p_ext, p_tr) in P.items():
        out.append({"tag": tag, "model": m, "population": "internal_nonGLP", **ext_metrics(p_int, te_bg)})
        out.append({"tag": tag, "model": m, "population": "external_GLP", **ext_metrics(p_ext, ext, p_tr, tr_bg)})
    if tr_full is not None:     # reference: GLP-aware model trained on both arms, scored on the same GLP test rows
        Pr = predict_all(tr_full, [ext], COLS)
        for m, (p_ext,) in Pr.items():
            out.append({"tag": tag, "model": m, "population": "external_GLP_reference_GLPaware", **ext_metrics(p_ext, ext)})
    return out


if __name__ == "__main__":
    tr, te = holdout_10_90(nonglp, seed=2026)
    trf, tef = holdout_10_90(ipd, seed=2026)
    rows += run_split("holdout_10_90", tr, te, tr_full=trf, te_glp=tef[tef.glp1ra_exposed == 1])
    rows += run_split("holdout_10_90_allGLP", tr, te)
    folds = list(kfold_10pct(nonglp, k=10, seed=7))
    rows += [r for ch in Parallel(n_jobs=4)(delayed(run_split)(f"fold{f}", tr, te) for f, tr, te in folds) for r in ch]
    def fr(f, rep):
        tr, te = holdout_10_90(nonglp, train_frac=f, seed=1000 * rep + int(f * 100)); return run_split(f"frac{f:.1f}_rep{rep}", tr, te)
    rows += [r for ch in Parallel(n_jobs=4)(delayed(fr)(f, rep) for f in [i / 10 for i in range(1, 10)] for rep in range(10)) for r in ch]
    def lo(s):  # train on non-GLP of other studies; internal test = non-GLP of study s; external = GLP of study s
        return run_split(f"loso_{s}", nonglp[nonglp.study != s], nonglp[nonglp.study == s], te_glp=glp[glp.study == s])
    rows += [r for ch in Parallel(n_jobs=4)(delayed(lo)(s) for s in sorted(ipd.study.unique())) for r in ch]
    df = pd.DataFrame(rows); df.to_csv(f"{OUT7}/stage7_all.csv", index=False)

    # SHAP of the background model (XGB, hold-out) - what traditional/background features drive risk
    _, m = fit_bg("xgb", tr, BG_FEATS); Xg = X_all.loc[glp.index, BG_FEATS]
    sv = shap.TreeExplainer(m).shap_values(Xg)
    shap_bg = pd.Series(np.average(np.abs(sv), axis=0, weights=glp.n), index=BG_FEATS).groupby(GROUP).sum().sort_values(ascending=False)

    kf = df[df.tag.str.startswith("fold")]
    summ = {"design": __doc__.split("\n\n")[0], "n_train_population": int(nonglp.n.sum()), "events_train_population": int(nonglp[nonglp.y == 1].n.sum()),
            "n_external_GLP": int(glp.n.sum()), "events_external_GLP": int(glp[glp.y == 1].n.sum()), "gnn_config": GNN_CFG,
            "holdout": df[df.tag == "holdout_10_90_allGLP"].round(5).to_dict("records"),
            "holdout_reference_GLPaware": df[(df.tag == "holdout_10_90") & (df.population == "external_GLP_reference_GLPaware")].round(5).to_dict("records"),
            "kfold": kf.groupby(["model", "population"])[["auroc", "o_e", "cal_slope", "brier"]].agg(["mean", "std"]).round(5).pipe(lambda d: d.set_axis(["_".join(c) for c in d.columns], axis=1)).reset_index().to_dict("records"),
            "loso_external_auroc": df[df.tag.str.startswith("loso") & (df.population == "external_GLP")].groupby("model").auroc.mean().round(4).to_dict(),
            "train_fraction_external": df[df.tag.str.startswith("frac") & (df.population == "external_GLP")].assign(tf=lambda d: d.tag.str.slice(4, 7)).groupby(["model", "tf"]).auroc.mean().round(4).unstack().to_dict(),
            "shap_background_model_on_GLP": shap_bg.round(5).to_dict(), "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}
    json.dump(summ, open(f"{OUT7}/summary.json", "w"), indent=1, default=float)

    # figure: internal vs external AUROC by model (10 folds) + O/E in GLP users
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    MODELS = ["logreg", "rf", "xgb", "gnn", "ensemble"]; CC = {"internal_nonGLP": "#2a78d6", "external_GLP": "#eb6834"}
    for j, popn in enumerate(["internal_nonGLP", "external_GLP"]):
        for i, m in enumerate(MODELS):
            v = kf[(kf.model == m) & (kf.population == popn)].auroc
            ax[0].scatter(np.full(len(v), i) + (j - 0.5) * 0.3 + np.linspace(-0.05, 0.05, len(v)), v, s=16, color=CC[popn], label=("non-GLP internal" if popn.startswith("internal") else "GLP external") if i == 0 else None)
    ax[0].axhline(0.5, color="#a3a29d", ls="--", lw=1); ax[0].set_xticks(range(5), ["LR", "RF", "XGB", "GNN", "Ens."]); ax[0].set_ylabel("AUROC"); ax[0].set_title("A  Discrimination, 10 folds"); ax[0].legend(frameon=False, fontsize=7)
    oe = kf[kf.population == "external_GLP"].groupby("model").agg(o=("o_e", "mean"), lo=("oe_ci_lo", "mean"), hi=("oe_ci_hi", "mean")).reindex(MODELS)
    ax[1].errorbar(range(5), oe.o, yerr=[oe.o - oe.lo, oe.hi - oe.o], fmt="o", color="#eb6834", capsize=3, ms=6)
    ax[1].axhline(1, color="#a3a29d", ls="--", lw=1); ax[1].set_xticks(range(5), ["LR", "RF", "XGB", "GNN", "Ens."]); ax[1].set_ylabel("Observed / expected in GLP-1RA users"); ax[1].set_title("B  Under-prediction by background model")
    for a_ in ax: a_.spines[["top", "right"]].set_visible(False); a_.grid(axis="y", color="#e4e3df")
    fig.tight_layout(); fig.savefig(f"{OUT7}/fig_stage7_external.png", dpi=300)
    print(json.dumps({k: summ[k] for k in ["n_train_population", "events_train_population", "n_external_GLP", "events_external_GLP", "kfold", "loso_external_auroc", "shap_background_model_on_GLP"]}, indent=1, default=float))

"""Stage 14 - feature-combination experiment for an optional-input calculator, and the compounding hypothesis.

Hypothesis (H-compound): feature groups with negligible stand-alone predictive value can contribute when combined with others
(positive interaction / super-additivity).

Feature groups (9): exposure | drug | indication | age (study mean age + ABCD-A) | sex (% female) | duration (follow-up) |
                    hypertension (ABCD-B) | diabetes (ABCD-D) | setting (region, data source, outcome definition)
Design: every non-empty subset (511) x models. LR, RF, XGB on all 10 disjoint 10% folds; GNN (GraphSAGE), 1D-CNN and the
mean ensemble (LR+RF+XGB+GNN) on folds 0-2. Value v(S) = AUROC on the 90% test set (v(empty) = 0.5).
Analysis per model: exact Shapley value of each group (AUROC contribution averaged over all coalitions), Shapley interaction
index for every pair, stand-alone gain v({i}) - 0.5, fold-level paired tests (Nadeau-Bengio corrected t) with BH-FDR.
H-compound is supported for group i if its stand-alone gain is negligible (< 0.005 or not significant) while its Shapley value
or an interaction with another group is significantly > 0.
Calculator: (a) bank of dedicated sub-models, one per input pattern (best model per subset of clinician-enterable groups);
(b) one XGBoost trained with random feature-group dropout (p=0.3, masked as NaN) evaluated under every missing pattern.
Uses the ABCD-augmented reconstructed data (variant_abcd). research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
import os, json, itertools, math, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import torch
from splits import kfold_10pct
from gnn import fit_predict_gnn_cfg
warnings.filterwarnings("ignore"); torch.set_num_threads(1)
OUT = "results/stage14"; os.makedirs(OUT, exist_ok=True)
ipd = pd.read_csv("variant_abcd/data/processed/pseudo_ipd.csv", dtype={"study": str})
D = pd.get_dummies(ipd[["therapy", "indication", "region", "source_type", "outcome_def"]], dtype=float)
NUM = ipd[["glp1ra_exposed", "followup_years", "followup_years_missing", "mean_age", "mean_age_missing", "pct_female", "pct_female_missing",
           "abcd_A", "abcd_A_missing", "abcd_B", "abcd_B_missing", "abcd_D"]].astype(float)
X = D.join(NUM)
GROUPS = {"exposure": ["glp1ra_exposed"], "drug": [c for c in D if c.startswith("therapy_")], "indication": [c for c in D if c.startswith("indication_")],
          "age": ["mean_age", "mean_age_missing", "abcd_A", "abcd_A_missing"], "sex": ["pct_female", "pct_female_missing"],
          "duration": ["followup_years", "followup_years_missing"], "hypertension": ["abcd_B", "abcd_B_missing"], "diabetes": ["abcd_D"],
          "setting": [c for c in D if c.startswith(("region_", "source_type_", "outcome_def_"))]}
G = list(GROUPS); N = len(G)
CLIN = [g for g in G if g != "setting"]
SUBSETS = [frozenset(c) for r in range(1, N + 1) for c in itertools.combinations(G, r)]
FOLDS = list(kfold_10pct(ipd, k=10, seed=7))


def cols_of(S): return [c for g in G if g in S for c in GROUPS[g]]


def fit_pred(model, S, tr, te):
    cols = cols_of(S); Xtr, Xte = X.loc[tr.index, cols], X.loc[te.index, cols]; y, w = tr.y.to_numpy(), tr.n.to_numpy().astype(float)
    if model == "logreg":
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        m = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, y, sample_weight=w * (np.sum(w * y) / np.sum(w))); return m.predict_proba((Xte - mu) / sd)[:, 1]
    if model == "rf":
        return RandomForestClassifier(n_estimators=150, max_depth=4, max_features="sqrt", bootstrap=False, min_weight_fraction_leaf=0.01, random_state=0, n_jobs=1).fit(Xtr, y, sample_weight=w).predict_proba(Xte)[:, 1]
    if model == "xgb":
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, min_child_weight=0.5, tree_method="hist", base_score=float(np.average(y, weights=w)), verbosity=0, n_jobs=1, random_state=0).fit(Xtr, y, sample_weight=w).predict_proba(Xte)[:, 1]
    if model == "gnn":
        return fit_predict_gnn_cfg(Xtr, y, w, Xte, tr.study.to_numpy(), te.study.to_numpy(), epochs=200)
    if model == "cnn1d":
        import importlib.util
        torch.manual_seed(0); mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        x = torch.tensor(((Xtr - mu) / sd).to_numpy(np.float32)); yt = torch.tensor(y, dtype=torch.float32); wt = torch.tensor(w, dtype=torch.float32)
        d = x.shape[1]; net = torch.nn.ModuleDict({"c": torch.nn.Conv1d(1, 8, 3, padding=1), "f": torch.nn.Linear(8 + d, 1)})
        with torch.no_grad(): net["f"].bias.fill_(float(np.log(max(np.average(y, weights=w), 1e-6))))
        fwd = lambda z: net["f"](torch.cat([torch.relu(net["c"](z.unsqueeze(1))).amax(-1), z], 1)).squeeze(-1)
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4); sc = 1.0 / max(float((w * y).sum()), 1.0)
        for _ in range(200):
            opt.zero_grad(); z = fwd(x); loss = -(wt * (yt * torch.nn.functional.logsigmoid(z) + (1 - yt) * torch.nn.functional.logsigmoid(-z))).sum() * sc; loss.backward(); opt.step()
        with torch.no_grad(): return torch.sigmoid(fwd(torch.tensor(((Xte - mu) / sd).to_numpy(np.float32)))).numpy()


def auc(p, te): return roc_auc_score(te.y, p, sample_weight=te.n)


_src5b = open("src/stage5b_classification.py").read(); exec(_src5b[_src5b.index("def w_quantile"):_src5b.index("# ---- (i) hold-out 10/90")])
from sklearn.metrics import average_precision_score


def full_metrics(p_te, te, p_tr, tr):
    y, w = te.y.to_numpy(), te.n.to_numpy().astype(float); A = roc_auc_score(y, p_te, sample_weight=w)
    n1, n0 = (w * y).sum(), (w * (1 - y)).sum(); q1, q2 = A / (2 - A), 2 * A ** 2 / (1 + A)
    se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A ** 2) + (n0 - 1) * (q2 - A ** 2)) / (n1 * n0))
    r = {"auroc": A, "auroc_lo": A - 1.96 * se, "auroc_hi": A + 1.96 * se, "p_auroc_vs_0.5": stats.norm.sf((A - 0.5) / se),
         "pr_auc": average_precision_score(y, p_te, sample_weight=w), "o_e": float(n1 / max((w * p_te).sum(), 1e-12))}
    for tn, t in thresholds(p_tr, tr.y.to_numpy(), tr.n.to_numpy().astype(float)).items():
        if tn == "T3_top20pct": continue
        c = conf(p_te, y, w, t)
        r.update({f"{tn}_{k}": c[k] for k in ["sensitivity", "p_sens_vs_chance", "specificity", "p_spec_vs_chance", "accuracy", "p_acc_vs_nir",
                                              "ppv", "p_ppv_vs_prevalence", "npv", "p_npv_vs_1minusprev", "lr_pos", "net_benefit", "flagged_frac"]})
    return r


def job(fi, S, models):
    _, tr, te = FOLDS[fi]; out = []; P = {}; Ptr = {}
    both = pd.concat([te, tr])
    for m in models:
        pb = fit_pred(m, S, tr, both); P[m], Ptr[m] = pb[:len(te)], pb[len(te):]
        out.append({"fold": fi, "subset": "|".join(sorted(S)), "k": len(S), "model": m, **full_metrics(P[m], te, Ptr[m], tr)})
    if all(m in P for m in ["logreg", "rf", "xgb", "gnn"]):
        pe = np.mean([P[m] for m in ["logreg", "rf", "xgb", "gnn"]], 0); pt = np.mean([Ptr[m] for m in ["logreg", "rf", "xgb", "gnn"]], 0)
        out.append({"fold": fi, "subset": "|".join(sorted(S)), "k": len(S), "model": "ensemble", **full_metrics(pe, te, pt, tr)})
    return out


def odds_ratios(S):
    """Count-weighted logistic regression for one combination on all data; cluster-robust SE by study (11 clusters)."""
    import statsmodels.api as sm
    cols = [c for c in cols_of(S) if not c.endswith("_missing")]
    Xs = X[cols].copy()
    # drop one reference level per categorical group and any collinear column
    for g, pref in [("drug", "therapy_"), ("indication", "indication_"), ("setting", "region_"), ("setting", "source_type_"), ("setting", "outcome_def_")]:
        cs = [c for c in Xs if c.startswith(pref)]
        if cs: Xs = Xs.drop(columns=[max(cs, key=lambda c: (X.loc[:, c] * ipd.n).sum())])
    Xs = sm.add_constant(Xs, has_constant="add"); keep = []
    for c in Xs.columns:
        if np.linalg.matrix_rank(Xs[keep + [c]].to_numpy()) > len(keep): keep.append(c)
    Xs = Xs[keep]
    sc = {c: (10 if c in ("mean_age", "pct_female") else (0.1 if c in ("abcd_A", "abcd_B") else 1)) for c in Xs.columns}
    Xs = Xs / pd.Series(sc)
    try:
        m = sm.GLM(ipd.y, Xs, family=sm.families.Binomial(), freq_weights=ipd.n).fit(cov_type="cluster", cov_kwds={"groups": ipd.study.astype("category").cat.codes})
    except Exception as e:
        return [{"subset": "|".join(sorted(S)), "feature": "FIT_FAILED", "note": str(e)[:80]}]
    ci = m.conf_int()
    return [{"subset": "|".join(sorted(S)), "k": len(S), "feature": c + (" (per 10 units)" if sc[c] == 10 else (" (per 10 percentage points)" if sc[c] == 0.1 else "")), "OR": float(np.exp(m.params[c])),
             "ci_lo": float(np.exp(ci.loc[c, 0])), "ci_hi": float(np.exp(ci.loc[c, 1])), "p": float(m.pvalues[c])} for c in Xs.columns if c != "const"]


def shapley(v):
    """v: dict frozenset->value (v[empty]=0.5). Returns Shapley values and pairwise Shapley interaction indices."""
    phi = {}
    for i in G:
        others = [g for g in G if g != i]; s = 0.0
        for r in range(0, N):
            wgt = math.factorial(r) * math.factorial(N - r - 1) / math.factorial(N)
            for c in itertools.combinations(others, r):
                S = frozenset(c); s += wgt * (v[S | {i}] - v[S])
        phi[i] = s
    inter = {}
    for i, j in itertools.combinations(G, 2):
        others = [g for g in G if g not in (i, j)]; s = 0.0
        for r in range(0, N - 1):
            wgt = math.factorial(r) * math.factorial(N - r - 2) / math.factorial(N - 1)
            for c in itertools.combinations(others, r):
                S = frozenset(c); s += wgt * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
        inter[(i, j)] = s
    return phi, inter


def dropout_model(fi, p_drop=0.3, reps=20):
    """One XGB trained with random group masking (NaN), evaluated under every input pattern.
    Masks are drawn per study ARM (shared by its event and non-event rows) so missingness cannot leak the outcome."""
    _, tr, te = FOLDS[fi]; rng = np.random.default_rng(fi)
    arm = (tr.study.astype(str) + "_" + tr.glp1ra_exposed.astype(str)).to_numpy(); arms = np.unique(arm)
    parts = []
    for _ in range(reps):
        Xa = X.loc[tr.index].copy()
        for g in G:
            masked_arms = arms[rng.random(len(arms)) < p_drop]
            Xa.loc[np.isin(arm, masked_arms), GROUPS[g]] = np.nan
        parts.append(Xa)
    Xtr = pd.concat(parts); y = np.tile(tr.y.to_numpy(), reps); w = np.tile(tr.n.to_numpy().astype(float), reps)
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist", base_score=float(np.average(y, weights=w)), verbosity=0, n_jobs=1, random_state=0).fit(Xtr, y, sample_weight=w)
    out = []
    for S in SUBSETS:
        Xt = X.loc[te.index].copy()
        for g in G:
            if g not in S: Xt[GROUPS[g]] = np.nan
        Xtr_m = X.loc[tr.index].copy()
        for g in G:
            if g not in S: Xtr_m[GROUPS[g]] = np.nan
        out.append({"fold": fi, "subset": "|".join(sorted(S)), "k": len(S), "model": "xgb_dropout_single", **full_metrics(m.predict_proba(Xt)[:, 1], te, m.predict_proba(Xtr_m)[:, 1], tr)})
    return out


if __name__ == "__main__":
    path = f"{OUT}/subset_folds.csv"
    if not os.path.exists(path):
        jobs = [(fi, S, ["logreg", "rf", "xgb"] + (["gnn", "cnn1d"] if fi < 3 else [])) for fi in range(10) for S in SUBSETS]
        rows = [r for ch in Parallel(n_jobs=3, batch_size=8)(delayed(job)(*j) for j in jobs) for r in ch]
        rows += [r for ch in Parallel(n_jobs=3)(delayed(dropout_model)(fi) for fi in range(10)) for r in ch]
        pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.read_csv(path)
    key = lambda s: frozenset(s.split("|"))
    # ---- full performance per combination x model (mean over folds, SD = stability) with fold-combined P values (Fisher)
    mcols = [c for c in df.columns if c not in ("fold", "subset", "k", "model") and not c.startswith("p_")]
    agg = df.groupby(["subset", "k", "model"])[mcols].agg(["mean", "std"]); agg.columns = ["_".join(c) for c in agg.columns]
    pcols = [c for c in df.columns if c.startswith("p_") or "_p_" in c]
    fisher = df.groupby(["subset", "k", "model"])[pcols].agg(lambda x: stats.combine_pvalues(np.clip(x.dropna(), 1e-300, 1), method="fisher").pvalue if x.notna().any() else np.nan)
    perf = agg.join(fisher).reset_index()
    for c in pcols:   # BH-FDR within each P-value family across all combinations x models
        p = perf[c].fillna(1).to_numpy(); o = np.argsort(p); n = len(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        perf[c.replace("p_", "q_", 1) if c.startswith("p_") else c.replace("_p_", "_q_")] = np.clip(q, 0, 1)
    perf.to_csv(f"{OUT}/combination_performance.csv", index=False)
    # ---- odds ratios per combination (count-weighted logistic, cluster-robust by study)
    orp = f"{OUT}/combination_odds_ratios.csv"
    if not os.path.exists(orp):
        ors = pd.DataFrame([r for ch in Parallel(n_jobs=3)(delayed(odds_ratios)(S) for S in SUBSETS) for r in ch])
        p = ors.p.fillna(1).to_numpy(); o = np.argsort(p); n = len(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; ors["q_bh"] = np.clip(q, 0, 1)
        ors.to_csv(orp, index=False)
    ors = pd.read_csv(orp)
    # how consistent is each feature's OR across all combinations that contain it (direction + significance)
    ors["base_feature"] = ors.feature.str.replace(" (per 10 units)", "", regex=False).str.replace(" (per 10 percentage points)", "", regex=False)
    stab = ors[ors.feature != "FIT_FAILED"].groupby("base_feature").agg(n_models=("OR", "size"), median_OR=("OR", "median"), min_OR=("OR", "min"), max_OR=("OR", "max"),
              share_OR_gt1=("OR", lambda x: float((x > 1).mean())), share_q05=("q_bh", lambda x: float((x < 0.05).mean()))).reset_index()
    stab.to_csv(f"{OUT}/odds_ratio_stability.csv", index=False)
    try:
        with pd.ExcelWriter(f"{OUT}/Supplementary_feature_combinations.xlsx") as xw:
            pd.DataFrame({"README": ["Research prototype; not for clinical use. compliance_ref=Addendum_5_DT9.",
                                     "performance: every feature-group combination x model; mean and SD over folds; P values combined across folds (Fisher) and BH-corrected (q_).",
                                     "odds_ratios: count-weighted logistic regression per combination; cluster-robust SE by study; study-level covariates are ecological.",
                                     "or_stability: distribution of each feature's OR across all combinations containing it.",
                                     "shapley / interactions: AUROC contributions of feature groups across all coalitions (fold-level tests)."]}).to_excel(xw, "README", index=False)
            perf.to_excel(xw, "performance", index=False); ors.to_excel(xw, "odds_ratios", index=False); stab.to_excel(xw, "or_stability", index=False)
    except Exception as e:
        print("excel write skipped:", e)
    res, inter_rows, shap_rows = {}, [], []
    for m, g in df.groupby("model"):
        folds = sorted(g.fold.unique()); per_fold = []
        for fi in folds:
            v = {frozenset(): 0.5}; v.update({key(r.subset): r.auroc for r in g[g.fold == fi].itertuples()})
            if len(v) < 2 ** N: continue
            phi, it = shapley(v); per_fold.append((fi, phi, it, v))
        if not per_fold: continue
        ratio = 9.0; kf = len(per_fold)
        def tt(vals):
            vals = np.asarray(vals); sd = vals.std(ddof=1)
            return (2 * stats.t.sf(abs(vals.mean() / np.sqrt((1 / kf + ratio) * vals.var(ddof=1))), kf - 1)) if kf > 1 and sd > 0 else np.nan
        for i in G:
            ph = [pf[1][i] for pf in per_fold]; alone = [pf[3][frozenset([i])] - 0.5 for pf in per_fold]
            shap_rows.append({"model": m, "group": i, "standalone_gain": np.mean(alone), "p_standalone": tt(alone), "shapley_value": np.mean(ph), "p_shapley": tt(ph), "folds": kf})
        for (i, j) in itertools.combinations(G, 2):
            vals = [pf[2][(i, j)] for pf in per_fold]
            inter_rows.append({"model": m, "pair": f"{i} x {j}", "interaction": np.mean(vals), "p": tt(vals), "folds": kf})
    sh = pd.DataFrame(shap_rows); it = pd.DataFrame(inter_rows)
    for d_, pc in [(sh, "p_shapley"), (sh, "p_standalone"), (it, "p")]:
        p = d_[pc].fillna(1).to_numpy(); o = np.argsort(p); n = len(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        d_["q_" + pc.replace("p_", "") if pc != "p" else "q"] = np.clip(q, 0, 1)
    sh["H_compound_supported"] = ((sh.standalone_gain < 0.005) | (sh.q_standalone >= 0.05)) & (sh.shapley_value > 0) & (sh.q_shapley < 0.05)
    syn = it[(it.interaction > 0) & (it.q < 0.05)]
    sh.to_csv(f"{OUT}/shapley_groups.csv", index=False); it.to_csv(f"{OUT}/shapley_interactions.csv", index=False)
    try:
        with pd.ExcelWriter(f"{OUT}/Supplementary_feature_combinations.xlsx", mode="a", if_sheet_exists="replace") as xw:
            sh.to_excel(xw, "shapley", index=False); it.to_excel(xw, "interactions", index=False)
    except Exception as e:
        print("excel append skipped:", e)
    # calculator: best model per clinician-enterable input pattern (dedicated bank) vs single dropout model
    mean = df.groupby(["subset", "model"]).auroc.mean().reset_index()
    clin = mean[mean.subset.apply(lambda s: "setting" not in s.split("|"))]
    bank = clin[clin.model != "xgb_dropout_single"].sort_values("auroc", ascending=False).groupby("subset").head(1).rename(columns={"model": "best_model", "auroc": "best_auroc"})
    single = clin[clin.model == "xgb_dropout_single"][["subset", "auroc"]].rename(columns={"auroc": "dropout_single_auroc"})
    cal = bank.merge(single, on="subset", how="left"); cal["k"] = cal.subset.str.count(r"\|") + 1
    cal.sort_values(["k", "best_auroc"], ascending=[True, False]).to_csv(f"{OUT}/calculator_model_bank.csv", index=False)
    # figure: Shapley value vs stand-alone gain per group (XGB) + interaction heatmap
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
    x_ = sh[sh.model == "xgb"].set_index("group").reindex(G)
    ax[0].barh(np.arange(N) + 0.18, x_.standalone_gain, height=0.34, color="#a3a29d", label="Stand-alone gain (AUROC − 0.5)")
    ax[0].barh(np.arange(N) - 0.18, x_.shapley_value, height=0.34, color="#2a78d6", label="Shapley value (all combinations)")
    ax[0].set_yticks(range(N), G); ax[0].axvline(0, color="#52514e", lw=0.8); ax[0].legend(frameon=False, fontsize=7); ax[0].set_xlabel("AUROC contribution (XGBoost)")
    M = np.zeros((N, N))
    for r in it[it.model == "xgb"].itertuples():
        a, b = r.pair.split(" x "); M[G.index(a), G.index(b)] = M[G.index(b), G.index(a)] = r.interaction
    lim = max(abs(M).max(), 1e-4); im = ax[1].imshow(M, cmap="RdBu", vmin=-lim, vmax=lim)
    ax[1].set_xticks(range(N), G, rotation=60, fontsize=7); ax[1].set_yticks(range(N), G, fontsize=7); fig.colorbar(im, ax=ax[1], fraction=0.04).set_label("Shapley interaction (AUROC)")
    for a_ in ax[:1]: a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_feature_combinations.png", dpi=300)
    json.dump({"n_subsets": len(SUBSETS), "groups": GROUPS, "shapley": sh.round(5).to_dict("records"),
               "synergistic_pairs_q05": syn.round(5).to_dict("records"), "H_compound_supported": sh[sh.H_compound_supported].round(5).to_dict("records"),
               "calculator_top_by_k": cal.sort_values(["k", "best_auroc"], ascending=[True, False]).groupby("k").head(3).round(4).to_dict("records"),
               "dropout_vs_bank_mean_diff": float((cal.dropout_single_auroc - cal.best_auroc).mean()),
               "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(sh.round(4).to_string(index=False)); print(syn.round(4).to_string(index=False)); print(cal.sort_values("best_auroc", ascending=False).head(15).round(4).to_string(index=False))

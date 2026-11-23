"""Stage 10 - incidence-based analysis, rare-event modelling and anomaly detection.

A. Incidence. Per study arm: cumulative incidence (events/N) and incidence rate per 100,000 person-years
   (reported person-years where available, else N x follow-up; follow-up median-imputed and flagged).
   Stratified by exposure, drug, indication, comparator, region, data source, study age band, % female band, follow-up band.
   GLP vs non-GLP: incidence rate ratio (IRR, exact conditional binomial CI) and rate difference per stratum; Mantel-Haenszel
   IRR stratified by study; Poisson GLM  events ~ exposure x stratum + study, offset log(person-years)  (effect modification).
   Context: published background incidence in adults >=50 y (about 2.3-10 per 100,000 per year; Introduction refs).
B. Rare-event supervised modelling (same 10 x 10% folds + LOSO):
   poisson_offset (rate model, risk = 1 - exp(-rate x follow-up)), xgb_balanced (class-weighted) + prior correction,
   logreg_kingzeng (balanced fit + King-Zeng intercept correction), focal-loss MLP, incidence_anchored XGB
   (recalibrated so predicted cumulative incidence matches observed training incidence within exposure x age band).
C. Anomaly detection (NAION = rare anomaly), trained on NON-EVENT patients of the training split only:
   Isolation Forest, one-class SVM, Local Outlier Factor (novelty), deep autoencoder (reconstruction error).
   Scored on held-out patients; AUROC / PR-AUC (Hanley-McNeil P vs 0.5), paired comparison vs XGBoost.
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
from common import *
from splits import holdout_10_90, kfold_10pct
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import statsmodels.api as sm, torch, os
from build_ipd import PRIMARY
torch.set_num_threads(1)
OUT = "results/stage10"; os.makedirs(OUT, exist_ok=True)

# ---------------- person-years per arm
co = pd.read_json("data/extracted/cohort_level.json")
py_rep = {}
for idx in PRIMARY:
    r = co.loc[idx]; py_rep[(str(r.source_pmid), 1)] = r.person_years_exposed; py_rep[(str(r.source_pmid), 0)] = r.person_years_comparator
arm = ipd.groupby(["study", "glp1ra_exposed"]).agg(N=("n", "sum"), therapy=("therapy", "first"), indication=("indication", "first"), region=("region", "first"),
       source_type=("source_type", "first"), followup=("followup_years", "first"), fu_missing=("followup_years_missing", "first"), age=("mean_age", "first"),
       age_missing=("mean_age_missing", "first"), female=("pct_female", "first"), female_missing=("pct_female_missing", "first")).reset_index()
arm["E"] = arm.apply(lambda a: ipd[(ipd.study == a.study) & (ipd.glp1ra_exposed == a.glp1ra_exposed) & (ipd.y == 1)].n.sum(), axis=1)
arm["PY"] = arm.apply(lambda a: py_rep.get((a.study, a.glp1ra_exposed)) if pd.notna(py_rep.get((a.study, a.glp1ra_exposed))) else a.N * a.followup, axis=1)
arm["py_source"] = arm.apply(lambda a: "reported" if pd.notna(py_rep.get((a.study, a.glp1ra_exposed))) else ("N x imputed follow-up" if a.fu_missing else "N x follow-up"), axis=1)
arm["age_band"] = np.where(arm.age_missing == 1, "unknown", pd.cut(arm.age, [0, 55, 60, 120], right=False, labels=["<55", "55-59", ">=60"]).astype(str))
arm["female_band"] = np.where(arm.female_missing == 1, "unknown", pd.cut(arm.female, [0, 40, 55, 101], right=False, labels=["<40%", "40-54%", ">=55%"]).astype(str))
arm["fu_band"] = pd.cut(arm.followup, [0, 1.01, 2.51, 99], labels=["<=1 y", "1-2.5 y", ">2.5 y"]).astype(str)
cmp_of = arm[arm.glp1ra_exposed == 0].set_index("study").therapy
arm["comparator"] = arm.study.map(cmp_of)
arm["drug"] = arm.study.map(arm[arm.glp1ra_exposed == 1].set_index("study").therapy)


def rate_ci(E, PY):
    lo = stats.chi2.ppf(0.025, 2 * E) / 2 if E > 0 else 0.0; hi = stats.chi2.ppf(0.975, 2 * (E + 1)) / 2
    return 1e5 * E / PY, 1e5 * lo / PY, 1e5 * hi / PY


def irr_ci(E1, PY1, E0, PY0):
    """Exact conditional (binomial) CI for the incidence rate ratio."""
    n = E1 + E0; p = PY1 / (PY1 + PY0)
    if n == 0 or E0 == 0: return np.nan, np.nan, np.nan, np.nan
    ci = stats.binomtest(int(E1), int(n), p).proportion_ci(method="exact")
    f = lambda q: q / (1 - q) * PY0 / PY1
    return (E1 / PY1) / (E0 / PY0), f(ci.low), f(ci.high) if ci.high < 1 else np.inf, stats.binomtest(int(E1), int(n), p).pvalue


rows = []
for s, g in arm.groupby("study"):
    e1, e0 = g[g.glp1ra_exposed == 1].iloc[0], g[g.glp1ra_exposed == 0].iloc[0]
    for e in (e1, e0):
        ir, lo, hi = rate_ci(e.E, e.PY)
        rows.append({"level": "arm", "study": s, "exposed": e.glp1ra_exposed, "therapy": e.therapy, "events": e.E, "N": e.N, "person_years": e.PY, "py_source": e.py_source,
                     "cum_incidence_per_100k": 1e5 * e.E / e.N, "rate_per_100k_py": ir, "rate_lo": lo, "rate_hi": hi, "age_band": e.age_band, "indication": e.indication})
inc_arm = pd.DataFrame(rows); inc_arm.to_csv(f"{OUT}/incidence_by_arm.csv", index=False)

strat_rows = []
for var in ["overall", "drug", "indication", "comparator", "region", "source_type", "age_band", "female_band", "fu_band"]:
    groups = [("all", arm)] if var == "overall" else list(arm.groupby(var))
    for lvl, g in groups:
        E1, P1 = g[g.glp1ra_exposed == 1].E.sum(), g[g.glp1ra_exposed == 1].PY.sum(); E0, P0 = g[g.glp1ra_exposed == 0].E.sum(), g[g.glp1ra_exposed == 0].PY.sum()
        r1, r1l, r1h = rate_ci(E1, P1); r0, r0l, r0h = rate_ci(E0, P0); irr, il, ih, p = irr_ci(E1, P1, E0, P0)
        # Mantel-Haenszel IRR stratified by study (removes between-study confounding)
        num = sum(a.E * b.PY / (a.PY + b.PY) for _, sg in g.groupby("study") for a, b in [(sg[sg.glp1ra_exposed == 1].iloc[0], sg[sg.glp1ra_exposed == 0].iloc[0])])
        den = sum(b.E * a.PY / (a.PY + b.PY) for _, sg in g.groupby("study") for a, b in [(sg[sg.glp1ra_exposed == 1].iloc[0], sg[sg.glp1ra_exposed == 0].iloc[0])])
        strat_rows.append({"stratifier": var, "level": lvl, "studies": g.study.nunique(), "glp_events": E1, "glp_py": P1, "glp_rate_per_100k_py": r1, "glp_lo": r1l, "glp_hi": r1h,
                           "nonglp_events": E0, "nonglp_py": P0, "nonglp_rate_per_100k_py": r0, "nonglp_lo": r0l, "nonglp_hi": r0h,
                           "crude_IRR": irr, "IRR_lo": il, "IRR_hi": ih, "p": p, "MH_IRR_by_study": num / den if den > 0 else np.nan, "rate_diff_per_100k_py": r1 - r0})
strat = pd.DataFrame(strat_rows)
o = np.argsort(strat.p.fillna(1).to_numpy()); n = len(strat); q = np.empty(n)
q[o] = np.minimum.accumulate((strat.p.fillna(1).to_numpy()[o] * n / np.arange(1, n + 1))[::-1])[::-1]; strat["q_bh"] = np.clip(q, 0, 1)
strat.to_csv(f"{OUT}/incidence_stratified.csv", index=False)

# Poisson GLM with study fixed effects + exposure x stratum interactions (effect modification), offset log PY
pois = {}
for var in ["age_band", "indication", "drug", "comparator", "female_band", "fu_band"]:
    d = arm.copy(); d["exp_x"] = d.glp1ra_exposed.astype(str) + "_" + d[var].astype(str)
    X = pd.get_dummies(d.study, prefix="s", drop_first=True, dtype=float)
    levels = sorted(d[var].unique())
    for lv in levels: X[f"exposed:{lv}"] = ((d.glp1ra_exposed == 1) & (d[var] == lv)).astype(float)
    X = sm.add_constant(X)
    m = sm.GLM(d.E, X, family=sm.families.Poisson(), offset=np.log(d.PY)).fit()
    terms = [t for t in X.columns if t.startswith("exposed:")]
    res = [{"stratum": t.split(":", 1)[1], "IRR": float(np.exp(m.params[t])), "lo": float(np.exp(m.conf_int().loc[t, 0])), "hi": float(np.exp(m.conf_int().loc[t, 1])), "p": float(m.pvalues[t])} for t in terms]
    # Wald test of homogeneity across strata
    if len(terms) > 1:
        R = np.zeros((len(terms) - 1, len(X.columns)))
        for i in range(len(terms) - 1): R[i, list(X.columns).index(terms[i])] = 1; R[i, list(X.columns).index(terms[-1])] = -1
        het = float(m.wald_test(R, scalar=True).pvalue)
    else: het = np.nan
    pois[var] = {"by_stratum": res, "p_heterogeneity": het}
json.dump(pois, open(f"{OUT}/poisson_effect_modification.json", "w"), indent=1, default=float)

# Figure: incidence rates by age band and indication, GLP vs non-GLP
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for i, var in enumerate(["age_band", "indication", "drug"]):
    s_ = strat[strat.stratifier == var].reset_index(drop=True); x = np.arange(len(s_))
    ax[i].errorbar(x - 0.12, s_.nonglp_rate_per_100k_py, yerr=[s_.nonglp_rate_per_100k_py - s_.nonglp_lo, s_.nonglp_hi - s_.nonglp_rate_per_100k_py], fmt="o", color="#2a78d6", capsize=3, label="No GLP-1RA")
    ax[i].errorbar(x + 0.12, s_.glp_rate_per_100k_py, yerr=[s_.glp_rate_per_100k_py - s_.glp_lo, s_.glp_hi - s_.glp_rate_per_100k_py], fmt="o", color="#eb6834", capsize=3, label="GLP-1RA")
    ax[i].axhspan(2.3, 10.2, color="#e4e3df", alpha=0.6, zorder=0, label="Published background, age >=50")
    ax[i].set_xticks(x, s_.level, fontsize=7, rotation=20); ax[i].set_yscale("log"); ax[i].set_title({"age_band": "A  Study mean age", "indication": "B  Indication", "drug": "C  Drug (exposed arm)"}[var])
    ax[i].spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("NAION per 100,000 person-years (log)"); ax[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_incidence_strata.png", dpi=300); plt.close(fig)

# ---------------- B. rare-event supervised models
PYcol = ipd.apply(lambda r: arm.set_index(["study", "glp1ra_exposed"]).loc[(r.study, r.glp1ra_exposed), "PY"] / arm.set_index(["study", "glp1ra_exposed"]).loc[(r.study, r.glp1ra_exposed), "N"], axis=1)  # PY per patient


def rare_models(tr, te):
    X, y, w = XY(tr); Xt, yt, wt = XY(te); out = {}
    # Poisson rate model with offset (per-patient person-time)
    Xs = sm.add_constant(((X - MU) / SD).astype(float), has_constant="add"); Xts = sm.add_constant(((Xt - MU) / SD).astype(float), has_constant="add")
    keep = [c for c in Xs.columns if Xs[c].std() > 0 or c == "const"]
    pm = sm.GLM(y, Xs[keep], family=sm.families.Poisson(), offset=np.log(PYcol.loc[tr.index].to_numpy()), freq_weights=w).fit_regularized(alpha=1e-4, L1_wt=0.0)
    rate = np.exp(Xts[keep].to_numpy() @ pm.params.to_numpy()); out["poisson_offset"] = 1 - np.exp(-rate * PYcol.loc[te.index].to_numpy())
    # class-weighted XGB with prior correction
    tau = np.average(y, weights=w); wb = np.where(y == 1, w * (1 - tau) / tau, w)
    xb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist", verbosity=0, n_jobs=1, random_state=0).fit(X, y, sample_weight=wb)
    pb = xb.predict_proba(Xt)[:, 1]; ybar = 0.5
    out["xgb_balanced_raw"] = pb
    out["xgb_balanced_priorcorrected"] = 1 / (1 + (1 / pb - 1) * (ybar / (1 - ybar)) * ((1 - tau) / tau))
    # King-Zeng rare-events logistic: balanced fit then intercept correction
    lr = LogisticRegression(C=1.0, max_iter=5000).fit((X - MU) / SD, y, sample_weight=wb / wb.mean())
    z = lr.decision_function((Xt - MU) / SD) - np.log(((1 - tau) / tau) * (ybar / (1 - ybar)))
    out["logreg_kingzeng"] = 1 / (1 + np.exp(-z))
    # incidence-anchored XGB: standard XGB recalibrated to observed training incidence within exposure x age band
    xs = fit("xgb", tr); p_tr = predict("xgb", xs, tr); p_te = predict("xgb", xs, te)
    band = lambda df: df.glp1ra_exposed.astype(str) + "_" + np.where(df.mean_age_missing == 1, "u", pd.cut(df.mean_age, [0, 55, 60, 120], right=False, labels=["a", "b", "c"]).astype(str))
    btr, bte = band(tr), band(te); adj = {}
    for b_ in btr.unique():
        mtr = (btr == b_).to_numpy(); adj[b_] = (w[mtr] * y[mtr]).sum() / max((w[mtr] * p_tr[mtr]).sum(), 1e-12)
    out["xgb_incidence_anchored"] = np.clip(p_te * bte.map(adj).fillna(1.0).to_numpy(), 0, 1)
    # focal-loss MLP (gamma=2) - deep rare-event learner
    torch.manual_seed(0); xt_ = torch.tensor(((X - MU) / SD).to_numpy(np.float32)); yt_ = torch.tensor(y, dtype=torch.float32); wt_ = torch.tensor(w / w.mean(), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xt_.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    with torch.no_grad(): net[-1].bias.fill_(float(np.log(tau / (1 - tau))))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4); alpha = 0.25
    for _ in range(300):
        opt.zero_grad(); pz = torch.sigmoid(net(xt_).squeeze(-1)).clamp(1e-7, 1 - 1e-7)
        loss = -(wt_ * (alpha * yt_ * (1 - pz) ** 2 * torch.log(pz) + (1 - alpha) * (1 - yt_) * pz ** 2 * torch.log(1 - pz))).sum() / wt_.sum()
        loss.backward(); opt.step()
    with torch.no_grad(): out["focal_mlp"] = torch.sigmoid(net(torch.tensor(((Xt - MU) / SD).to_numpy(np.float32))).squeeze(-1)).numpy()
    out["xgb_standard"] = p_te
    return out


# ---------------- C. anomaly detection (trained on non-events only)
def expand_sample(df, n_max, rng):
    w = df.n.to_numpy().astype(float); k = rng.multinomial(min(n_max, int(w.sum())), w / w.sum())
    return X_all.loc[np.repeat(df.index, k)]


class AE(torch.nn.Module):
    def __init__(self, d): super().__init__(); self.e = torch.nn.Sequential(torch.nn.Linear(d, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)); self.d = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, d))
    def forward(self, x): return self.d(self.e(x))


def anomaly_models(tr, te, seed=0):
    rng = np.random.default_rng(seed); neg = tr[tr.y == 0]
    Xn = ((expand_sample(neg, 20000, rng) - MU) / SD).to_numpy(np.float32)
    Xt = ((X_all.loc[te.index] - MU) / SD).to_numpy(np.float32); out = {}
    out["isolation_forest"] = -IsolationForest(n_estimators=300, random_state=seed).fit(Xn).score_samples(Xt)
    out["one_class_svm"] = -OneClassSVM(nu=0.01, gamma="scale").fit(Xn[:5000]).score_samples(Xt)
    out["lof_novelty"] = -LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Xn[:10000]).score_samples(Xt)
    torch.manual_seed(seed); ae = AE(Xn.shape[1]); opt = torch.optim.Adam(ae.parameters(), lr=0.005); xn = torch.tensor(Xn)
    for _ in range(200):
        opt.zero_grad(); loss = ((ae(xn) - xn) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad(): out["autoencoder"] = ((ae(torch.tensor(Xt)) - torch.tensor(Xt)) ** 2).mean(1).numpy()
    return out


def hm_p(A, n1, n0):
    q1, q2 = A / (2 - A), 2 * A ** 2 / (1 + A); se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A ** 2) + (n0 - 1) * (q2 - A ** 2)) / (n1 * n0))
    return stats.norm.sf((A - 0.5) / se)


def evaluate(tag, tr, te, seed=0):
    _, y, w = XY(te); rows = []
    for fam, preds in [("rare_event", rare_models(tr, te)), ("anomaly", anomaly_models(tr, te, seed))]:
        for m, p in preds.items():
            r = {"tag": tag, "family": fam, "model": m, "auroc": roc_auc_score(y, p, sample_weight=w), "pr_auc": average_precision_score(y, p, sample_weight=w)}
            r["p_auroc_vs_0.5"] = hm_p(r["auroc"], (w * y).sum(), (w * (1 - y)).sum())
            if fam == "rare_event": r.update({k: v for k, v in metrics(y, p, w).items() if k in ("brier", "cal_slope", "o_e")})
            rows.append(r)
    return rows


if __name__ == "__main__":
    allrows = []
    tr, te = holdout_10_90(ipd, seed=2026); allrows += evaluate("holdout_10_90", tr, te)
    allrows += [r for ch in Parallel(n_jobs=1)(delayed(evaluate)(f"fold{f}", tr, te, f) for f, tr, te in kfold_10pct(ipd, k=10, seed=7)) for r in ch]
    allrows += [r for ch in Parallel(n_jobs=1)(delayed(evaluate)(f"loso_{s}", ipd[ipd.study != s], ipd[ipd.study == s]) for s in sorted(ipd.study.unique())) for r in ch]
    df = pd.DataFrame(allrows); df.to_csv(f"{OUT}/rare_event_anomaly_all.csv", index=False)
    kf = df[df.tag.str.startswith("fold")]
    ref = kf[kf.model == "xgb_standard"].set_index("tag").auroc
    summ_m = []
    for (fam, m), g in kf.groupby(["family", "model"]):
        d = g.set_index("tag").auroc - ref.reindex(g.tag).to_numpy(); v = d.var(ddof=1)
        t = d.mean() / np.sqrt((1 / 10 + 9) * v) if v > 0 else 0
        summ_m.append({"family": fam, "model": m, "auroc_mean": g.auroc.mean(), "auroc_sd": g.auroc.std(), "pr_auc_mean": g.pr_auc.mean(), "o_e_mean": g.get("o_e", pd.Series(dtype=float)).mean(),
                       "cal_slope_mean": g.get("cal_slope", pd.Series(dtype=float)).mean(), "diff_vs_xgb": d.mean(), "p_vs_xgb": 2 * stats.t.sf(abs(t), 9) if v > 0 else 1.0,
                       "loso_auroc": df[df.tag.str.startswith("loso") & (df.model == m)].auroc.mean()})
    sm_ = pd.DataFrame(summ_m).sort_values("auroc_mean", ascending=False); sm_.to_csv(f"{OUT}/rare_event_anomaly_summary.csv", index=False)
    json.dump({"incidence_overall": strat[strat.stratifier == "overall"].to_dict("records"), "incidence_stratified": strat.round(4).to_dict("records"),
               "poisson_effect_modification": pois, "models": sm_.round(5).to_dict("records"),
               "py_note": "person-years reported for 2 studies; otherwise N x follow-up (median-imputed where missing)",
               "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(strat[["stratifier", "level", "studies", "glp_rate_per_100k_py", "nonglp_rate_per_100k_py", "crude_IRR", "IRR_lo", "IRR_hi", "MH_IRR_by_study", "q_bh"]].round(3).to_string(index=False))
    print({k: v["p_heterogeneity"] for k, v in pois.items()})
    print(sm_.round(4).to_string(index=False))

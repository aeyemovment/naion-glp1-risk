"""Stage 2 - digital-twin augmentation lattice, 3x6x9 -> infinity (DESIGN_synthetic_3x6x9.md).

3 parents (generators) x 6 volume levels x 9 synthetic shares (+0%/100% anchors),
crossed with real held-out fraction h in {10..100%}. Test set is ALWAYS held-out real
PubMed pseudo-IPD; twins are never scored. Generators are fit only on each cell's real
training split (or on external priors when h = 100%).

Twin rows: synthetic_only=true. research_prototype=true. compliance_ref=Addendum_5_DT9.
NOT for clinical / diagnostic / regulatory use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
import json, os, sys, time, warnings, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
from build_ipd import FEATURES_CAT

warnings.filterwarnings("ignore")
OUT = "results/stage2"; os.makedirs(OUT, exist_ok=True)
S1 = json.load(open("results/stage1/stage1_results.json"))
STABLE = S1["selected_features"]["stable_top5_ge8of10_folds"]           # SHAP-stable across 10 folds
NUMC = ["glp1ra_exposed", "followup_years", "mean_age", "pct_female"]
FLAGS = ["followup_years_missing", "mean_age_missing", "pct_female_missing"]
ipd = pd.read_csv("data/processed/pseudo_ipd.csv", dtype={"study": str})
from build_ipd import ABCD_COLS
if os.environ.get("ABCD") == "1": NUMC = NUMC + ABCD_COLS
FEAT = FEATURES_CAT + NUMC + FLAGS
DUMMY_COLS = list(pd.get_dummies(ipd[FEATURES_CAT], dtype=float).columns)
PROFILE_BANK = 5000                       # digital-patient profiles per generator draw
VOLUMES = [1, 3, 6, 9, 27, 100]           # 6 levels
SHARES = [0.0] + [i / 10 for i in range(1, 10)] + [1.0]   # 9 variants + anchors
HELDOUT = [i / 10 for i in range(1, 11)]
PARENTS = ["P1_marginal", "P2_copula", "P3_structural"]
# External priors (used only when h=100%): published pooled estimates, see rct_meta.json
# Non-user baseline: pooled NAION incidence in GLP-1RA users 85/100k (PMID 42060879) / pooled RR 2.52 (same paper)
# = ~34/100k cumulative over ~2.5 y typical follow-up -> ~1.35e-4 per person-year (approximation, documented in paper)
PRIOR_BASE_RATE_PER_YEAR = 85e-5 / 2.52 / 2.5
PRIOR_RR = {"semaglutide": 1.93, "glp1ra_class": 1.70, "sema_or_tirz": 1.93}  # PMIDs 42612051, 41498755


def encode(df):
    X = pd.get_dummies(df[FEATURES_CAT], dtype=float).reindex(columns=DUMMY_COLS, fill_value=0.0)
    return X.join(df[NUMC + FLAGS].astype(float).reset_index(drop=True).set_index(X.index))


def wmetrics(y, p, w):
    lp = np.log(np.clip(p, 1e-9, 1) / np.clip(1 - p, 1e-9, 1))
    cs = LogisticRegression(C=1e6, max_iter=500).fit(lp.reshape(-1, 1), y, sample_weight=w).coef_[0][0]
    return {"auroc": roc_auc_score(y, p, sample_weight=w), "pr_auc": average_precision_score(y, p, sample_weight=w),
            "brier": brier_score_loss(y, p, sample_weight=w), "cal_slope": float(cs), "o_e": float((w * y).sum() / (w * p).sum())}


def split_real(h, seed):
    rng = np.random.default_rng(seed)
    te_n = rng.binomial(ipd.n.to_numpy(), h)
    test = ipd.assign(n=te_n); train = ipd.assign(n=ipd.n - te_n)
    return train[train.n > 0], test[test.n > 0]


def arm_table(train):
    """Collapse to one row per covariate pattern with N and events."""
    key = FEAT
    g = train.groupby(key, dropna=False).apply(lambda d: pd.Series({"N": d.n.sum(), "E": d.loc[d.y == 1, "n"].sum()})).reset_index()
    return g[g.N > 0]


def generate(parent, train, rng, catalogue):
    """Return a bank of PROFILE_BANK twin profiles with risk p (aggregated later)."""
    if train is None:                                  # h = 100%: external priors only
        base = catalogue.sample(PROFILE_BANK, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        if parent == "P1_marginal":
            for c in FEAT: base[c] = rng.permutation(base[c].to_numpy())
        if parent != "P1_marginal":
            for c, sd in [("mean_age", 2.0), ("pct_female", 3.0), ("followup_years", 0.25)]:
                base[c] = np.clip(base[c] + rng.normal(0, sd, len(base)), 0, None)
        rr = base.therapy.map(PRIOR_RR).fillna(1.0).where(base.glp1ra_exposed == 1, 1.0)
        base["p"] = np.clip(PRIOR_BASE_RATE_PER_YEAR * base.followup_years.clip(0.25) * rr, 1e-6, 0.5)
        return base
    A = arm_table(train)
    w = (A.N / A.N.sum()).to_numpy()
    prev = A.E.sum() / A.N.sum()
    if parent == "P1_marginal":
        # independent marginals; risk = therapy-stratum rate (each feature sampled independently)
        prof = pd.DataFrame({c: rng.choice(A[c].to_numpy(), PROFILE_BANK, p=w) for c in FEAT})
        rate = A.groupby("therapy").apply(lambda d: (d.E.sum() + prev) / (d.N.sum() + 1))
        prof["p"] = prof.therapy.map(rate).fillna(prev).to_numpy()
        return prof
    # P2 / P3: sample joint patterns (keeps correlations) + Gaussian-copula jitter on numerics
    idx = rng.choice(len(A), PROFILE_BANK, p=w)
    prof = A.iloc[idx][FEAT].reset_index(drop=True)
    z = rng.multivariate_normal(np.zeros(3), np.array([[1, .3, .2], [.3, 1, .1], [.2, .1, 1]]), PROFILE_BANK)
    for j, (c, sd) in enumerate([("mean_age", 2.0), ("pct_female", 3.0), ("followup_years", 0.25)]):
        prof[c] = np.clip(prof[c].to_numpy() + sd * z[:, j], 0, None)
    if parent == "P2_copula":
        # Beta-posterior draw of the source pattern's risk (pattern-level uncertainty)
        a = A.E.to_numpy()[idx] + prev * 10; b = A.N.to_numpy()[idx] - A.E.to_numpy()[idx] + (1 - prev) * 10
        prof["p"] = rng.beta(a, b)
        return prof
    # P3 structural: logistic model on SHAP-stable features + exposure, fit to real training counts
    cols = [c for c in DUMMY_COLS if any(c.startswith(f + "_") for f in STABLE if f in FEATURES_CAT)]
    numc = [f for f in STABLE if f in NUMC] + ["glp1ra_exposed"]
    def mat(df): return pd.get_dummies(df[FEATURES_CAT], dtype=float).reindex(columns=cols, fill_value=0.0).to_numpy(), df[numc].astype(float).to_numpy()
    Xc, Xn = mat(A); mu, sd = Xn.mean(0), Xn.std(0) + 1e-9
    Xtr = np.hstack([Xc, (Xn - mu) / sd]); Xtr = np.vstack([Xtr, Xtr]); y = np.r_[np.ones(len(A)), np.zeros(len(A))]
    ww = np.r_[A.E, A.N - A.E].astype(float); ww *= A.E.sum() / ww.sum()
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, y, sample_weight=ww)
    Pc, Pn = mat(prof); prof["p"] = lr.predict_proba(np.hstack([Pc, (Pn - mu) / sd]))[:, 1]
    return prof


def twins_aggregated(prof, n_twins, rng):
    """Distribute n_twins over the profile bank (multinomial) and draw outcomes (binomial)."""
    cnt = rng.multinomial(int(n_twins), np.full(len(prof), 1 / len(prof)))
    ev = rng.binomial(cnt, prof.p.to_numpy())
    t = prof[FEAT]
    out = pd.concat([t.assign(y=1, n=ev), t.assign(y=0, n=cnt - ev)], ignore_index=True)
    return out[out.n > 0]


def real_draw(train, n_real, rng):
    c = train.n.to_numpy().astype(np.int64); R = c.sum()
    if n_real <= 0 or R == 0: return train.iloc[:0]
    k = rng.multivariate_hypergeometric(c, int(n_real)) if n_real <= R else rng.multinomial(int(n_real), c / R)
    d = train.assign(n=k); return d[d.n > 0][FEAT + ["y", "n"]]


def fit_eval(train_df, test, ref_imp=None, model="xgb"):
    X = encode(train_df.reset_index(drop=True)); y = train_df.y.to_numpy(); w = train_df.n.to_numpy().astype(float)
    Xt = encode(test.reset_index(drop=True)); yt = test.y.to_numpy(); wt = test.n.to_numpy().astype(float)
    if model == "logreg":
        mu, sd = X.mean(), X.std().replace(0, 1)
        m = LogisticRegression(C=1.0, max_iter=5000).fit((X - mu) / sd, y, sample_weight=w * (np.sum(w * y) / np.sum(w)))
        Xts = (Xt - mu) / sd; p = m.predict_proba(Xts)[:, 1]
        sv = (Xts.to_numpy() - np.average(Xts, axis=0, weights=wt)) * m.coef_[0]
    else:
        if model == "rf":
            m = RandomForestClassifier(n_estimators=100, max_depth=4, max_features="sqrt", bootstrap=False,
                                       min_weight_fraction_leaf=0.01, random_state=0, n_jobs=1)
        else:
            m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, min_child_weight=0.5, tree_method="hist",
                              base_score=float(np.clip(np.average(y, weights=w), 1e-6, 0.5)), verbosity=0, n_jobs=1, random_state=0)
        m.fit(X, y, sample_weight=w); p = m.predict_proba(Xt)[:, 1]
        sv = shap.TreeExplainer(m).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        if sv.ndim == 3: sv = sv[:, :, 1]
    r = wmetrics(yt, p, wt)
    imp = pd.Series(np.average(np.abs(sv), axis=0, weights=wt), index=Xt.columns)
    grp = imp.groupby(lambda c: next((f for f in FEATURES_CAT if c.startswith(f + "_")), c.replace("_missing", ""))).sum()
    r["shap"] = grp.round(6).to_dict()
    if ref_imp is not None:
        common = grp.index.intersection(ref_imp.index)
        r["shap_spearman_vs_real"] = float(spearmanr(grp[common], ref_imp[common]).correlation)
    return r


def run_cell(h, parent, V, s, seed, model="xgb", ref=None):
    rng = np.random.default_rng(zlib.crc32(repr((h, parent, V, s, seed)).encode()))
    train, test = split_real(h, seed)
    R = int(train.n.sum()) if h < 1.0 else int(0.1 * ipd.n.sum())   # reference size when no real train
    catalogue = ipd.drop_duplicates(subset=FEAT)[FEAT].reset_index(drop=True)  # covariate design catalogue (no outcomes)
    if h == 1.0 and s < 1.0: return None
    T = V * R
    n_syn, n_real = int(round(s * T)), int(round((1 - s) * T))
    parts = []
    if n_real > 0: parts.append(real_draw(train, n_real, rng))
    if n_syn > 0:
        prof = generate(parent, None if h == 1.0 else train, rng, catalogue)
        parts.append(twins_aggregated(prof, n_syn, rng))
    tr = pd.concat(parts, ignore_index=True)
    if tr.loc[tr.y == 1, "n"].sum() == 0: return None
    r = fit_eval(tr, test, None if ref is None else pd.Series(ref), model)
    return {"model": model, "train_frac": round(1 - h, 2), "h": h, "parent": parent, "V": V, "s": s, "seed": seed, "n_real": n_real, "n_syn": n_syn,
            "n_test": int(test.n.sum()), **r}


def lattice(cells, n_jobs=-1):
    out = Parallel(n_jobs=n_jobs, verbose=0, batch_size=8)(delayed(run_cell)(*c[:5], model=c[5], ref=REF_IMP.get((c[0], c[4], c[5]))) for c in cells)
    return [o for o in out if o is not None]


REF_IMP = {}

if __name__ == "__main__":
    SEEDS = list(range(101, 111))          # 10 repeats per training fraction (matches Stage 1c)
    MODELS = ["logreg", "rf", "xgb"]
    TAG = os.environ.get("TAG", "v2")
    t0 = time.time()
    refs = lattice([(h, "real_only", 1, 0.0, sd, mdl) for h in HELDOUT[:-1] for sd in SEEDS for mdl in MODELS])
    for r in refs: REF_IMP[(r["h"], r["seed"], r["model"])] = r["shap"]
    print(f"refs {len(refs)} in {time.time()-t0:.0f}s", flush=True)
    cells = [(h, p, V, s, sd, mdl) for h in HELDOUT for p in PARENTS for V in VOLUMES for s in SHARES if s > 0 for sd in SEEDS for mdl in MODELS]
    cells += [(h, "real_only", V, 0.0, sd, mdl) for h in HELDOUT[:-1] for V in VOLUMES if V > 1 for sd in SEEDS for mdl in MODELS]
    res = refs + lattice(cells)
    print(f"base lattice: {len(res)} fits in {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(res)
    rec_log = []
    for mdl in MODELS:
        for h in HELDOUT:
            cur = df[(df.h == h) & (df.model == mdl) & (df.parent != "real_only")].groupby(["parent", "V", "s"]).auroc.mean()
            if cur.empty: continue
            best_key, best = cur.idxmax(), cur.max(); step_s, depth = 0.05, 0
            while depth < 3:
                p, V, s_ = best_key
                Vs = sorted({max(1, round(V / 1.5)), V, round(V * 1.5)})
                Ss = sorted({min(1.0, max(0.01, round(s_ + d, 4))) for d in (-step_s, 0, step_s)})
                child = [(h, p, v, ss, sd, mdl) for v in Vs for ss in Ss for sd in SEEDS if (h < 1.0 or ss == 1.0)]
                cr = lattice(child)
                for r in cr: r["depth"] = depth + 1
                res += cr
                c = pd.DataFrame(cr).groupby(["parent", "V", "s"]).auroc.mean(); gain = c.max() - best
                rec_log.append({"model": mdl, "train_frac": round(1 - h, 2), "depth": depth + 1, "parent_cell": list(best_key),
                                "best_child": list(c.idxmax()), "best_auroc": float(max(best, c.max())), "gain": float(gain)})
                if gain < 0.005: break
                best_key, best = c.idxmax(), c.max(); step_s /= 2; depth += 1
    pd.DataFrame(res).drop(columns=["shap"]).to_csv(f"{OUT}/lattice_results_{TAG}.csv", index=False)
    pd.DataFrame([{**{k: r[k] for k in ("model", "train_frac", "parent", "V", "s", "seed")}, **{f"shap_{k}": v for k, v in r["shap"].items()}} for r in res]).to_csv(f"{OUT}/lattice_shap_{TAG}.csv", index=False)
    json.dump({"meta": {"lattice": "3x6x9->inf", "parents": PARENTS, "volumes": VOLUMES, "shares": SHARES, "train_fracs": [round(1 - h, 2) for h in HELDOUT],
                        "seeds": SEEDS, "models": MODELS, "profile_bank": PROFILE_BANK, "stable_features": STABLE,
                        "synthetic_only_twins": True, "research_prototype": True, "compliance_ref": "Addendum_5_DT9",
                        "policy_adapted_from": "CUSP-DT-3X6X9-ANTIBH-MEIOSIS-001"}, "recursion": rec_log},
              open(f"{OUT}/lattice_meta_{TAG}.json", "w"), default=float)
    print(f"total fits {len(res)} in {time.time()-t0:.0f}s")

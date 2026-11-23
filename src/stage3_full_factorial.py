"""Stage 3 - FULL-FACTORIAL twin experiment (supersedes Stage 2 v2).

Factors (all crossed):
  models        : logreg, rf, xgb, ensemble(mean of the three)       - all fit on the SAME training set per cell
  real train %  : 0, 10, 20, ..., 90                                  - test = remaining real PubMed pseudo-IPD
  generator set : P1, P2, P3, P1+P2, P1+P3, P2+P3, P1+P2+P3            - 2^3-1 combinations of the 3x6x9 parents
  prior source  : real (fit on real train), prior (published external priors only), blend (50/50)
  volume V      : 1, 3, 6, 9, 27, 100  (x real training size)
  synthetic share s : 10..100%  (0% = PubMed-only reference)
  repeats       : 10 random patient-level splits per training fraction
At 0% real, only source=prior is possible (no real data to fit a generator).
-> infinity: recursive refinement around the best cell per (model, train %), depth <= 3, stop if gain < 0.005.
Resumable: one CSV per training fraction in results/stage3/.

Twin rows synthetic_only=true; research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, sys, json, time, zlib, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap
import stage2_twins_3x6x9 as S2
from build_ipd import FEATURES_CAT

warnings.filterwarnings("ignore")
OUT = "results/stage3"; os.makedirs(OUT, exist_ok=True)
MODELS = ["logreg", "rf", "xgb", "ensemble"]
GENSETS = ["P1", "P2", "P3", "P1+P2", "P1+P3", "P2+P3", "P1+P2+P3"]
PMAP = {"P1": "P1_marginal", "P2": "P2_copula", "P3": "P3_structural"}
SOURCES = ["real", "prior", "blend"]
VOLUMES, SHARES = S2.VOLUMES, [i / 10 for i in range(1, 11)]
TRAIN_FRACS = [round(i / 10, 1) for i in range(0, 10)]
SEEDS = list(range(201, 211))
CATALOGUE = S2.ipd.drop_duplicates(subset=S2.FEAT)[S2.FEAT].reset_index(drop=True)


def group_imp(sv, cols, wt):
    imp = pd.Series(np.average(np.abs(sv), axis=0, weights=wt), index=cols)
    return imp.groupby(lambda c: next((f for f in FEATURES_CAT if c.startswith(f + "_")), c.replace("_missing", ""))).sum()


def fit_all(train_df, test):
    """Fit logreg/rf/xgb on identical data; ensemble = mean probability. Returns {model: (p, grouped |SHAP|)}."""
    X = S2.encode(train_df.reset_index(drop=True)); y = train_df.y.to_numpy(); w = train_df.n.to_numpy().astype(float)
    Xt = S2.encode(test.reset_index(drop=True)); wt = test.n.to_numpy().astype(float)
    out = {}
    mu, sd = X.mean(), X.std().replace(0, 1)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit((X - mu) / sd, y, sample_weight=w * (np.sum(w * y) / np.sum(w)))
    Xts = (Xt - mu) / sd
    out["logreg"] = (lr.predict_proba(Xts)[:, 1], group_imp((Xts.to_numpy() - np.average(Xts, axis=0, weights=wt)) * lr.coef_[0], Xt.columns, wt))
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, max_features="sqrt", bootstrap=False,
                                min_weight_fraction_leaf=0.01, random_state=0, n_jobs=1).fit(X, y, sample_weight=w)
    sv = shap.TreeExplainer(rf).shap_values(Xt); sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    out["rf"] = (rf.predict_proba(Xt)[:, 1], group_imp(sv, Xt.columns, wt))
    xg = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, min_child_weight=0.5, tree_method="hist",
                       base_score=float(np.clip(np.average(y, weights=w), 1e-6, 0.5)), verbosity=0, n_jobs=1, random_state=0).fit(X, y, sample_weight=w)
    out["xgb"] = (xg.predict_proba(Xt)[:, 1], group_imp(shap.TreeExplainer(xg).shap_values(Xt), Xt.columns, wt))
    norm = [out[m][1] / out[m][1].sum() for m in ["logreg", "rf", "xgb"]]          # scale-free ensemble importance
    out["ensemble"] = (np.mean([out[m][0] for m in ["logreg", "rf", "xgb"]], axis=0), sum(norm) / 3)
    return out


def make_twins(genset, source, train, n_syn, rng):
    parents = [PMAP[g] for g in genset.split("+")]
    banks = []
    for p in parents:
        if source == "real":
            banks.append(S2.generate(p, train, rng, CATALOGUE))
        elif source == "prior":
            banks.append(S2.generate(p, None, rng, CATALOGUE))
        else:
            a = S2.generate(p, train, rng, CATALOGUE); b = S2.generate(p, None, rng, CATALOGUE)
            banks.append(pd.concat([a.sample(len(a) // 2, random_state=int(rng.integers(1e9))),
                                    b.sample(len(b) // 2, random_state=int(rng.integers(1e9)))], ignore_index=True))
    prof = pd.concat(banks, ignore_index=True)
    return S2.twins_aggregated(prof, n_syn, rng)


def run_cell(tf, genset, source, V, s, seed, refs=None):
    h = round(1 - tf, 1)
    rng = np.random.default_rng(zlib.crc32(repr((tf, genset, source, V, s, seed)).encode()))
    train, test = S2.split_real(h, seed)
    R = int(train.n.sum()) if tf > 0 else int(0.1 * S2.ipd.n.sum())
    T = V * R; n_syn, n_real = int(round(s * T)), int(round((1 - s) * T))
    parts = []
    if n_real > 0: parts.append(S2.real_draw(train, n_real, rng))
    if n_syn > 0: parts.append(make_twins(genset, source, None if tf == 0 else train, n_syn, rng))
    tr = pd.concat(parts, ignore_index=True)
    if tr.loc[tr.y == 1, "n"].sum() == 0: return []
    fits = fit_all(tr, test)
    yt, wt = test.y.to_numpy(), test.n.to_numpy().astype(float)
    rows = []
    for m, (p, imp) in fits.items():
        r = S2.wmetrics(yt, p, wt)
        rho = np.nan
        if refs is not None and (seed, m) in refs:
            ref = pd.Series(refs[(seed, m)]); c = imp.index.intersection(ref.index); rho = spearmanr(imp[c], ref[c]).correlation
        rows.append({"model": m, "train_frac": tf, "genset": genset, "source": source, "V": V, "s": s, "seed": seed,
                     "n_real": n_real, "n_syn": n_syn, "n_test": int(wt.sum()), **r, "shap_rho_vs_pubmed": rho,
                     **{f"shap_{k}": v for k, v in imp.items()}})
    return rows


def cells_for(tf):
    if tf == 0:
        return [(tf, g, "prior", V, 1.0, sd) for g in GENSETS for V in VOLUMES for sd in SEEDS]
    c = [(tf, "none", "none", V, 0.0, sd) for V in VOLUMES for sd in SEEDS]           # PubMed-only (V>1 = bootstrap upweight)
    c += [(tf, g, src, V, s, sd) for g in GENSETS for src in SOURCES for V in VOLUMES for s in SHARES for sd in SEEDS]
    return c


def run(cells, refs, n_jobs=-1):
    out = Parallel(n_jobs=n_jobs, batch_size=16)(delayed(run_cell)(*c, refs=refs) for c in cells)
    return [r for chunk in out for r in chunk]


if __name__ == "__main__":
    t0 = time.time()
    for tf in TRAIN_FRACS:
        path = f"{OUT}/factorial_tf{int(tf*100):02d}.csv"
        if os.path.exists(path): print("skip", path, flush=True); continue
        refs = {}
        if tf > 0:
            base = run([(tf, "none", "none", 1, 0.0, sd) for sd in SEEDS], None)
            refs = {(r["seed"], r["model"]): {k[5:]: v for k, v in r.items() if k.startswith("shap_") and k != "shap_rho_vs_pubmed"} for r in base}
        else:   # 0% real: compare SHAP against PubMed-only trained on 90% (largest real reference)
            base = []
        rows = base + run([c for c in cells_for(tf) if not (c[1] == "none" and c[3] == 1)], refs)
        # recursion (-> infinity) per model around best (genset, source, V, s)
        df = pd.DataFrame(rows); rec = []
        for m in MODELS:
            cur = df[(df.model == m) & (df.genset != "none")].groupby(["genset", "source", "V", "s"]).auroc.mean()
            if cur.empty: continue
            bk, best, step, depth = cur.idxmax(), cur.max(), 0.05, 0
            while depth < 3:
                g, src, V, s_ = bk
                Vs = sorted({max(1, round(V / 1.5)), V, round(V * 1.5)})
                Ss = sorted({min(1.0, max(0.01, round(s_ + d, 4))) for d in (-step, 0, step)}) if tf > 0 else [1.0]
                child = run([(tf, g, src, v, ss, sd) for v in Vs for ss in Ss for sd in SEEDS], refs)
                child = [r for r in child if r["model"] == m]
                for r in child: r["depth"] = depth + 1
                rows += child
                c = pd.DataFrame(child).groupby(["genset", "source", "V", "s"]).auroc.mean(); gain = c.max() - best
                rec.append({"model": m, "train_frac": tf, "depth": depth + 1, "from": list(map(str, bk)), "best_child": list(map(str, c.idxmax())),
                            "best_auroc": float(max(best, c.max())), "gain": float(gain)})
                if gain < 0.005: break
                bk, best, step, depth = c.idxmax(), c.max(), step / 2, depth + 1
        pd.DataFrame(rows).to_csv(path, index=False)
        json.dump(rec, open(path.replace(".csv", "_recursion.json"), "w"), default=float)
        print(f"tf={tf:.1f}: {len(rows)} rows, elapsed {time.time()-t0:.0f}s", flush=True)
    json.dump({"factors": {"models": MODELS, "train_fracs": TRAIN_FRACS, "gensets": GENSETS, "sources": SOURCES, "volumes": VOLUMES,
                           "shares": SHARES, "seeds": SEEDS}, "lattice": "3x6x9->inf (full factorial extension)",
               "policy_adapted_from": "CUSP-DT-3X6X9-ANTIBH-MEIOSIS-001", "synthetic_only_twins": True, "research_prototype": True,
               "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/factorial_meta.json", "w"))
    print("DONE", time.time() - t0)

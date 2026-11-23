"""Stage 4 - ensembles, now including the GNN.

Base learners: logreg, rf, xgb, gnn (GNN config = best calibrated config from Stage 6c, else default GraphSAGE L2/k10).
Combinations: every subset of {logreg, rf, xgb, gnn} with >= 2 members (11 subsets)
Combiners:
  mean     - simple average of probabilities
  weighted - weights proportional to (inner-CV AUROC - 0.5), inner CV inside the training split only
  stacked  - logistic meta-learner on logit(p) of members, fit on inner out-of-fold predictions (no test leakage)
Cross-family ensemble: mean of PubMed-only XGB + best hybrid + best synthetic-only (configs from Stage 3 top-3), each
  member refit on the same real training split (twins generated from that split only).
Paradigms (same as other stages): 10/90 hold-out; 10 disjoint 10% folds; training fraction 10..90% x 5 repeats; LOSO.
Threshold metrics + p-values via Stage 5b helpers; paired corrected t vs best single model.
research_prototype=true; twins synthetic_only=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
from common import *
from splits import holdout_10_90, kfold_10pct
from gnn import fit_predict_gnn_cfg
from scipy import stats
from joblib import Parallel, delayed
import os, itertools
OUT4 = "results/stage4"; os.makedirs(OUT4, exist_ok=True)
src = open("src/stage5b_classification.py").read(); exec(src[src.index("def w_quantile"):src.index("# ---- (i) hold-out 10/90")])

BASE = ["logreg", "rf", "xgb", "gnn"]
SUBSETS = [c for r in range(2, 5) for c in itertools.combinations(BASE, r)]
GNN_CFG = dict(arch="sage", L=2, k=10, edge_mode="uniform", study_w=1.0)
if os.path.exists("results/stage6c/summary.json"):
    b = json.load(open("results/stage6c/summary.json"))["best_tuned"]
    GNN_CFG = dict(arch=b["arch"], L=int(b["L"]), k=int(b["k"]), edge_mode=b["edge_mode"], study_w=float(b["study_w"]))


def base_preds(tr, te):
    """Return {model: (p_on_te)} for the 4 base learners trained on tr."""
    out = {}
    for nm in ["logreg", "rf", "xgb"]:
        m = fit(nm, tr); out[nm] = predict(nm, m, te)
    Xtr, ytr, wtr = XY(tr); Xte, _, _ = XY(te)
    out["gnn"] = fit_predict_gnn_cfg(Xtr, ytr, wtr, Xte, tr.study.to_numpy(), te.study.to_numpy(), **GNN_CFG)
    return out


def inner_oof(tr, k=3, seed=0):
    """Out-of-fold member predictions inside the training split (patient-level thinning)."""
    oof_parts = []
    for f, itr, ite in kfold_10pct(tr, k=k, seed=seed):
        # kfold_10pct gives 1/k train; we want (k-1)/k train -> swap roles
        p = base_preds(ite, itr); _, y, w = XY(itr)
        oof_parts.append(pd.DataFrame({**p, "y": y, "w": w}))
    return pd.concat(oof_parts, ignore_index=True)


def logit(p): p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))


def combine(P_te, oof, subset):
    res = {"mean": np.mean([P_te[m] for m in subset], 0)}
    aucs = np.array([max(roc_auc_score(oof.y, oof[m], sample_weight=oof.w) - 0.5, 1e-3) for m in subset])
    wts = aucs / aucs.sum(); res["weighted"] = np.sum([w * P_te[m] for w, m in zip(wts, subset)], 0)
    Z = np.column_stack([logit(oof[m].to_numpy()) for m in subset]); w = oof.w.to_numpy() * (np.sum(oof.w * oof.y) / oof.w.sum())
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(Z, oof.y, sample_weight=w)
    res["stacked"] = meta.predict_proba(np.column_stack([logit(P_te[m]) for m in subset]))[:, 1]
    return res, dict(zip(subset, wts.round(4)))


def cross_family(tr, te):
    """Mean of PubMed-only XGB + best hybrid + best synthetic-only (Stage 3 top-3), same split."""
    path = "results/stage3/top3_for_app.csv"
    if not os.path.exists(path): return None
    import stage3_full_factorial as S3
    top = pd.read_csv(path); preds = []
    for _, r in top.iterrows():
        mdl = r.model if r.model in ("logreg", "rf", "xgb") else "xgb"
        if r.family == "pubmed_only":
            preds.append(predict(mdl, fit(mdl, tr), te)); continue
        rng = np.random.default_rng(0); R = int(tr.n.sum()); T = int(r.V) * R
        n_syn, n_real = int(round(r.s * T)), int(round((1 - r.s) * T))
        parts = ([S3.S2.real_draw(tr[S3.S2.FEAT + ["y", "n"]], n_real, rng)] if n_real > 0 else []) + \
                [S3.make_twins(r.genset, r.source, tr, n_syn, rng)]
        mix = pd.concat(parts, ignore_index=True)
        f = S3.fit_all(mix, te[S3.S2.FEAT + ["y", "n"]])
        preds.append(f[r.model][0] if r.model in f else f["xgb"][0])
    return np.mean(preds, 0)


def evaluate(tr, te, tag, seed=0, with_cross=True):
    P_te = base_preds(tr, te); P_tr = base_preds(tr, tr); oof = inner_oof(tr, k=3, seed=seed)
    _, y, w = XY(te); _, ytr, wtr = XY(tr); rows = []
    for m in BASE:
        rows.append({"tag": tag, "ensemble": m, "method": "single", **metrics(y, P_te[m], w)})
    for sub in SUBSETS:
        comb, wts = combine(P_te, oof, sub)
        comb_tr, _ = combine(P_tr, oof, sub)
        for meth, p in comb.items():
            r = {"tag": tag, "ensemble": "+".join(sub), "method": meth, "weights": json.dumps({k: float(v) for k, v in wts.items()}), **metrics(y, p, w)}
            t = thresholds(comb_tr[meth], ytr, wtr)
            for tn, tv in t.items():
                c = conf(p, y, w, tv); r.update({f"{tn}_{k}": c[k] for k in ["sensitivity", "specificity", "accuracy", "ppv", "npv", "net_benefit"]})
            rows.append(r)
    if with_cross:
        pc = cross_family(tr, te)
        if pc is not None: rows.append({"tag": tag, "ensemble": "cross_family(pubmed+hybrid+synthetic)", "method": "mean", **metrics(y, pc, w)})
    return rows


if __name__ == "__main__":
    rows = []
    tr, te = holdout_10_90(ipd, seed=2026); rows += evaluate(tr, te, "holdout_10_90")
    folds = list(kfold_10pct(ipd, k=10, seed=7))
    rows += [r for chunk in Parallel(n_jobs=4)(delayed(evaluate)(tr, te, f"fold{f}", f) for f, tr, te in folds) for r in chunk]
    def fr(f, rep):
        tr, te = holdout_10_90(ipd, train_frac=f, seed=1000 * rep + int(f * 100)); return evaluate(tr, te, f"frac{f:.1f}_rep{rep}", rep, with_cross=False)
    rows += [r for chunk in Parallel(n_jobs=4)(delayed(fr)(f, rep) for f in [i / 10 for i in range(1, 10)] for rep in range(5)) for r in chunk]
    def lo(s): return evaluate(ipd[ipd.study != s], ipd[ipd.study == s], f"loso_{s}", with_cross=False)
    rows += [r for chunk in Parallel(n_jobs=4)(delayed(lo)(s) for s in sorted(ipd.study.unique())) for r in chunk]
    df = pd.DataFrame(rows); df.to_csv(f"{OUT4}/ensembles_all.csv", index=False)
    kf = df[df.tag.str.startswith("fold")]
    summ = kf.groupby(["ensemble", "method"]).agg(auroc=("auroc", "mean"), sd=("auroc", "std"), o_e=("o_e", "mean"), brier=("brier", "mean")).reset_index().sort_values("auroc", ascending=False)
    best_single = summ[summ.method == "single"].iloc[0]
    def paired(e, m):
        a = kf[(kf.ensemble == e) & (kf.method == m)].sort_values("tag").auroc.to_numpy()
        b = kf[(kf.ensemble == best_single.ensemble) & (kf.method == "single")].sort_values("tag").auroc.to_numpy()
        d = a - b; v = d.var(ddof=1); return (d.mean(), 2 * stats.t.sf(abs(d.mean() / np.sqrt((1 / 10 + 9) * v)), 9) if v > 0 else 1.0)
    summ[["diff_vs_best_single", "p_corrected_t"]] = summ.apply(lambda r: pd.Series(paired(r.ensemble, r.method)), axis=1)
    o = np.argsort(summ.p_corrected_t.to_numpy()); n = len(summ); q = np.empty(n)
    q[o] = np.minimum.accumulate((summ.p_corrected_t.to_numpy()[o] * n / np.arange(1, n + 1))[::-1])[::-1]; summ["q_bh"] = np.clip(q, 0, 1)
    summ.to_csv(f"{OUT4}/ensemble_ranking_kfold.csv", index=False)
    lo_s = df[df.tag.str.startswith("loso")].groupby(["ensemble", "method"]).auroc.mean().rename("loso_auroc").reset_index()
    json.dump({"gnn_config": GNN_CFG, "best_single": best_single.to_dict(), "top10": summ.head(10).to_dict("records"),
               "gnn_containing_best": summ[summ.ensemble.str.contains("gnn")].head(5).to_dict("records"),
               "loso_top": lo_s.sort_values("loso_auroc", ascending=False).head(10).to_dict("records"),
               "research_prototype": True, "compliance_ref": "Addendum_5_DT9"},
              open(f"{OUT4}/stage4_summary.json", "w"), indent=1, default=float)
    print(summ.head(15).round(4).to_string(index=False))

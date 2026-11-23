"""Stage 8 - fine-tuning / model-updating experiments across non-GLP and GLP NAION patient profiles.

Per repeat (5 repeats), patient-level splits:
  non-GLP: 80% pretraining pool | 20% non-GLP test (forgetting check)
  GLP    : 50% fine-tuning pool | 10% validation (selects update strength) | 40% GLP test (primary)
Factors (fully crossed):
  pretrain_source  : none | real | synthetic | hybrid      (non-GLP pool; synthetic = copula twins fit to the pool, 1x;
                                                            hybrid = pool + equal number of twins)
  finetune_source  : real | synthetic | hybrid              (GLP pool, same definitions)
  glp_frac         : 0, 5, 10, 25, 50 % of all GLP patients drawn from the fine-tuning pool
  method           : pretrain_only (no update) | recal_intercept | recal_logistic | xgb_continue (extra boosting rounds
                     on GLP data with pretrained margin, rounds chosen on validation) | gnn_joint (GNN trained on pretrain +
                     GLP data with GLP up-weighting alpha chosen on validation) | pooled_refit (XGB from scratch on both) |
                     glp_only (XGB on GLP fine-tune data only)
Extra split schemes (real data, XGB): A) train non-GLP -> test GLP; B) train GLP -> test non-GLP (reverse transport);
  C) train mixed 10% -> test mixed 90% (standard)
Metrics: AUROC, O/E, calibration slope, Brier on GLP test (NAION vs no NAION among GLP users) and on non-GLP test.
Twins synthetic_only=true; research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
import os, json, zlib, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import xgboost as xgb
import stage2_twins_3x6x9 as S2
from gnn import fit_predict_gnn_cfg
warnings.filterwarnings("ignore")
OUT = "results/stage8"; os.makedirs(OUT, exist_ok=True)
ipd = S2.ipd.copy()
FEAT = S2.FEAT
# Stage 6c found no significant gain from deeper/tuned GNNs (diff +0.0006, P=0.95), so the cheaper default is used here
GNN_CFG = dict(arch="sage", L=2, k=10, edge_mode="uniform", study_w=1.0)


def thin(df, frac, rng):
    k = rng.binomial(df.n.to_numpy().astype(np.int64), frac); a = df.assign(n=k); b = df.assign(n=df.n - k)
    return a[a.n > 0], b[b.n > 0]


def twins_like(df, rng, n=None):
    if df is None or len(df) == 0 or df.loc[df.y == 1, "n"].sum() == 0: return df.iloc[:0] if df is not None else None
    prof = S2.generate("P2_copula", df[FEAT + ["y", "n"]], rng, None).sample(1000, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    t = S2.twins_aggregated(prof, int(n if n is not None else df.n.sum()), rng); t["study"] = "synthetic"; return t


def source(df, kind, rng):
    if df is None or len(df) == 0: return df
    if kind == "real": return df
    tw = twins_like(df, rng)
    return tw if kind == "synthetic" else pd.concat([df, tw], ignore_index=True)


def enc(df): return S2.encode(df.reset_index(drop=True))
def yw(df): return df.y.to_numpy(), df.n.to_numpy().astype(float)
def logit(p): p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
def sig(z): return 1 / (1 + np.exp(-z))


def met(p, df):
    y, w = yw(df)
    if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0: return {}
    from sklearn.metrics import roc_auc_score, brier_score_loss
    cs = LogisticRegression(C=1e6, max_iter=500).fit(logit(p).reshape(-1, 1), y, sample_weight=w).coef_[0][0]
    return {"auroc": roc_auc_score(y, p, sample_weight=w), "o_e": float((w * y).sum() / (w * p).sum()), "cal_slope": float(cs), "brier": brier_score_loss(y, p, sample_weight=w)}


def xgb_fit(df, base_margin=None, rounds=300, booster=None):
    X = enc(df); y, w = yw(df)
    d = xgb.DMatrix(X, label=y, weight=w, base_margin=base_margin)
    params = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.05, "min_child_weight": 0.5, "tree_method": "hist", "nthread": 1, "verbosity": 0}
    if booster is None and base_margin is None: params["base_score"] = float(np.clip(np.average(y, weights=w), 1e-6, 0.5))
    return xgb.train(params, d, num_boost_round=rounds, xgb_model=booster)


def xgb_margin(bst, df, base_margin=None):
    return bst.predict(xgb.DMatrix(enc(df), base_margin=base_margin), output_margin=True)


def run(rep, pre_src, ft_src, glp_frac):
    rng = np.random.default_rng(zlib.crc32(repr((rep, pre_src, ft_src, glp_frac)).encode()))
    r0 = np.random.default_rng(500 + rep)                       # split rng shared across factor levels within a repeat
    ng = ipd[ipd.glp1ra_exposed == 0]; g = ipd[ipd.glp1ra_exposed == 1]
    ng_pool, ng_test = thin(ng, 0.8, r0)
    g_pool, g_rest = thin(g, 0.5, r0); g_val, g_test = thin(g_rest, 0.2, r0)
    g_ft = thin(g_pool, min(1.0, glp_frac / 0.5), rng)[0] if glp_frac > 0 else None
    PRE = None if pre_src == "none" else source(ng_pool, pre_src, rng)
    FT = None if g_ft is None else source(g_ft, ft_src, rng)
    out = []
    def rec(method, p_g, p_ng):
        out.append({"rep": rep, "pretrain_source": pre_src, "finetune_source": ft_src if glp_frac > 0 else "none", "glp_frac": glp_frac, "method": method,
                    **{f"glp_{k}": v for k, v in met(p_g, g_test).items()}, **{f"nonglp_{k}": v for k, v in met(p_ng, ng_test).items()}})
    if PRE is not None:
        bst = xgb_fit(PRE)
        mg_test, mg_ng = xgb_margin(bst, g_test), xgb_margin(bst, ng_test)
        rec("pretrain_only", sig(mg_test), sig(mg_ng))
        if FT is not None and FT.loc[FT.y == 1, "n"].sum() > 0:
            mft = xgb_margin(bst, FT); y, w = yw(FT)
            # recalibration in the large (intercept) and logistic recalibration (intercept + slope)
            a = LogisticRegression(C=1e6, max_iter=500, fit_intercept=True).fit(np.zeros((len(y), 1)), y, sample_weight=w, ) if False else None
            from scipy.optimize import minimize_scalar, minimize
            nll = lambda c: -np.sum(w * (y * np.log(sig(mft + c) + 1e-12) + (1 - y) * np.log(1 - sig(mft + c) + 1e-12)))
            c = minimize_scalar(nll, bounds=(-5, 5), method="bounded").x
            rec("recal_intercept", sig(mg_test + c), sig(mg_ng + c))
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(mft.reshape(-1, 1), y, sample_weight=w)
            rec("recal_logistic", lr.predict_proba(mg_test.reshape(-1, 1))[:, 1], lr.predict_proba(mg_ng.reshape(-1, 1))[:, 1])
            # continued boosting on GLP data from the pretrained margin; rounds chosen on GLP validation
            best, best_r = -1, 0
            for R in (0, 25, 50, 100, 200):
                if R == 0: pv = sig(xgb_margin(bst, g_val))
                else:
                    b2 = xgb_fit(FT, base_margin=mft, rounds=R); pv = sig(xgb_margin(b2, g_val, base_margin=xgb_margin(bst, g_val)))
                a_ = met(pv, g_val).get("auroc", -1)
                if a_ > best: best, best_r = a_, R
            if best_r == 0: rec("xgb_continue", sig(mg_test), sig(mg_ng))
            else:
                b2 = xgb_fit(FT, base_margin=mft, rounds=best_r)
                rec("xgb_continue", sig(xgb_margin(b2, g_test, base_margin=mg_test)), sig(xgb_margin(b2, ng_test, base_margin=mg_ng)))
            out[-1]["chosen_rounds"] = best_r
    if FT is not None and FT.loc[FT.y == 1, "n"].sum() > 0:
        both = pd.concat([d for d in [PRE, FT] if d is not None], ignore_index=True)
        bp = xgb_fit(both); rec("pooled_refit", sig(xgb_margin(bp, g_test)), sig(xgb_margin(bp, ng_test)))
        bo = xgb_fit(FT); rec("glp_only", sig(xgb_margin(bo, g_test)), sig(xgb_margin(bo, ng_test)))
        # GNN joint training with GLP up-weighting alpha chosen on validation
        if PRE is not None:
            best, best_a = -1, 1.0
            for alpha in (1.0, 10.0):
                J = pd.concat([PRE, FT.assign(n=FT.n * alpha)], ignore_index=True); Xj = enc(J); yj, wj = yw(J)
                pv = fit_predict_gnn_cfg(Xj, yj, wj, enc(g_val), J.study.fillna("synthetic").to_numpy(), g_val.study.to_numpy(), epochs=150, **GNN_CFG)
                a_ = met(pv, g_val).get("auroc", -1)
                if a_ > best: best, best_a = a_, alpha
            J = pd.concat([PRE, FT.assign(n=FT.n * best_a)], ignore_index=True); Xj = enc(J); yj, wj = yw(J)
            tests = pd.concat([g_test, ng_test], ignore_index=True)
            p = fit_predict_gnn_cfg(Xj, yj, wj, enc(tests), J.study.fillna("synthetic").to_numpy(), tests.study.to_numpy(), epochs=150, **GNN_CFG)
            rec("gnn_joint", p[:len(g_test)], p[len(g_test):]); out[-1]["chosen_alpha"] = best_a
    return out


def schemes(rep):
    """Extra real-data split schemes with XGB."""
    r0 = np.random.default_rng(900 + rep); ng = ipd[ipd.glp1ra_exposed == 0]; g = ipd[ipd.glp1ra_exposed == 1]; out = []
    bA = xgb_fit(ng); out.append({"rep": rep, "scheme": "A_train_nonGLP_test_GLP", **met(sig(xgb_margin(bA, g)), g)})
    bB = xgb_fit(g); out.append({"rep": rep, "scheme": "B_train_GLP_test_nonGLP", **met(sig(xgb_margin(bB, ng)), ng)})
    tr, te = thin(ipd, 0.1, r0); bC = xgb_fit(tr)
    out.append({"rep": rep, "scheme": "C_train_mixed10_test_mixed90", **met(sig(xgb_margin(bC, te)), te)})
    out.append({"rep": rep, "scheme": "C_..._GLP_subset_of_test", **met(sig(xgb_margin(bC, te[te.glp1ra_exposed == 1])), te[te.glp1ra_exposed == 1])})
    out.append({"rep": rep, "scheme": "C_..._nonGLP_subset_of_test", **met(sig(xgb_margin(bC, te[te.glp1ra_exposed == 0])), te[te.glp1ra_exposed == 0])})
    return out


if __name__ == "__main__":
    REPS = range(5)
    cells = [(rep, pre, ft, fr) for rep in REPS for pre in ["none", "real", "synthetic", "hybrid"] for fr in [0, 0.05, 0.10, 0.25, 0.50]
             for ft in (["real", "synthetic", "hybrid"] if fr > 0 else ["real"]) if not (pre == "none" and fr == 0)]
    rows = [r for ch in Parallel(n_jobs=4, batch_size=2)(delayed(run)(*c) for c in cells) for r in ch]
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/finetune_all.csv", index=False)
    sc = pd.DataFrame([r for ch in Parallel(n_jobs=4)(delayed(schemes)(rep) for rep in REPS) for r in ch]); sc.to_csv(f"{OUT}/split_schemes.csv", index=False)
    g = df.groupby(["pretrain_source", "finetune_source", "glp_frac", "method"]).agg(glp_auroc=("glp_auroc", "mean"), glp_auroc_sd=("glp_auroc", "std"),
         glp_o_e=("glp_o_e", "mean"), glp_cal=("glp_cal_slope", "mean"), nonglp_auroc=("nonglp_auroc", "mean"), nonglp_o_e=("nonglp_o_e", "mean"), n=("rep", "size")).reset_index()
    ref = df[(df.method == "pretrain_only") & (df.pretrain_source == "real")].groupby("rep").glp_auroc.mean()
    def paired(sub):
        d = sub.set_index("rep").glp_auroc - ref.reindex(sub.rep).to_numpy()
        d = d.dropna(); return pd.Series({"d_vs_pretrain_real": d.mean(), "p": stats.ttest_1samp(d, 0).pvalue if len(d) > 1 and d.std() > 0 else np.nan})
    pt = df.groupby(["pretrain_source", "finetune_source", "glp_frac", "method"]).apply(paired).reset_index()
    g = g.merge(pt, on=["pretrain_source", "finetune_source", "glp_frac", "method"])
    p = g.p.fillna(1).to_numpy(); o = np.argsort(p); n = len(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]; g["q_bh"] = np.clip(q, 0, 1)
    g.sort_values("glp_auroc", ascending=False).to_csv(f"{OUT}/finetune_summary.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    CM = {"recal_intercept": "#2a78d6", "recal_logistic": "#1baf7a", "xgb_continue": "#eb6834", "gnn_joint": "#6250d6", "pooled_refit": "#eda100", "glp_only": "#e34948"}
    for i, ft in enumerate(["real", "synthetic", "hybrid"]):
        for m, c in CM.items():
            q_ = g[(g.pretrain_source.isin(["real", "none"] if m == "glp_only" else ["real"])) & (g.finetune_source == ft) & (g.method == m)].groupby("glp_frac").glp_auroc.mean()
            ax[i].plot(q_.index * 100, q_.values, "o-", color=c, lw=2, ms=4, label=m.replace("_", " "))
        base = g[(g.pretrain_source == "real") & (g.method == "pretrain_only")].glp_auroc.mean(); ax[i].axhline(base, color="#0b0b0b", ls="--", lw=1, label="pretrained on non-GLP, no update")
        ax[i].set_title(f"Fine-tune data: {ft}"); ax[i].set_xlabel("GLP-1RA patients used for fine-tuning (%)"); ax[i].grid(axis="y", color="#e4e3df"); ax[i].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("AUROC in held-out GLP-1RA users"); ax[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_finetune.png", dpi=300)
    json.dump({"top10": g.sort_values("glp_auroc", ascending=False).head(10).to_dict("records"),
               "schemes": sc.groupby("scheme")[["auroc", "o_e", "cal_slope"]].mean().round(4).reset_index().to_dict("records"),
               "significant_improvements_vs_pretrained": g[(g.q_bh < 0.05) & (g.d_vs_pretrain_real > 0)].to_dict("records"),
               "gnn_config": GNN_CFG, "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print(g.sort_values("glp_auroc", ascending=False).head(15).round(4).to_string(index=False)); print(sc.groupby("scheme")[["auroc", "o_e"]].mean().round(4))

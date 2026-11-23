"""Stage 9 - convolutional neural network (1D-CNN) on the tabular feature vector, same paradigms as Stages 1/6:
10/90 hold-out, 10 disjoint 10% folds (paired vs XGBoost), training fraction 10..90% x 5 repeats, LOSO, threshold metrics.
The encoded features are treated as a 1-channel sequence (Conv1d k=3 x2, global max-pool, dense). This is a deliberate,
honest comparator: with no image, signal or volume data there is no spatial structure for 2D/3D convolutions to exploit
(see src/imaging_cnn_scaffold.py for the 2D/3D OCT/fundus pipeline with Grad-CAM, prepared for future patient-level imaging).
Attribution: integrated gradients on input features. research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os as _os_env; _os_env.environ.setdefault("OMP_NUM_THREADS", "1"); _os_env.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch/xgboost OpenMP clashes in workers
from common import *
from splits import holdout_10_90, kfold_10pct
from scipy import stats
from joblib import Parallel, delayed
import torch, os
torch.set_num_threads(1)
OUT9 = "results/stage9"; os.makedirs(OUT9, exist_ok=True)
src = open("src/stage5b_classification.py").read(); exec(src[src.index("def w_quantile"):src.index("# ---- (i) hold-out 10/90")])


class CNN1D(torch.nn.Module):
    def __init__(self, d, ch=16, p_init=1e-3):
        super().__init__()
        self.c1 = torch.nn.Conv1d(1, ch, 3, padding=1); self.c2 = torch.nn.Conv1d(ch, ch, 3, padding=1)
        self.fc = torch.nn.Linear(ch + d, 1)        # global-pooled conv features + skip connection of raw features
        with torch.no_grad(): self.fc.bias.fill_(float(np.log(p_init / (1 - p_init))))
    def forward(self, x):
        h = torch.relu(self.c2(torch.relu(self.c1(x.unsqueeze(1))))).amax(-1)
        return self.fc(torch.cat([h, x], 1)).squeeze(-1)


def fit_cnn(tr, seed=0, epochs=300):
    torch.manual_seed(seed)
    X, y, w = XY(tr); Z = ((X - MU) / SD).to_numpy(np.float32)
    x = torch.tensor(Z); yt = torch.tensor(y, dtype=torch.float32); wt = torch.tensor(w, dtype=torch.float32)
    m = CNN1D(Z.shape[1], 16, max(float((w * y).sum() / w.sum()), 1e-6)); opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
    sc = 1.0 / max(float((w * y).sum()), 1.0)
    for _ in range(epochs):
        opt.zero_grad(); z = m(x)
        loss = -(wt * (yt * torch.nn.functional.logsigmoid(z) + (1 - yt) * torch.nn.functional.logsigmoid(-z))).sum() * sc
        loss.backward(); opt.step()
    m.eval(); return m


def pred(m, df):
    X, _, _ = XY(df)
    with torch.no_grad(): return torch.sigmoid(m(torch.tensor(((X - MU) / SD).to_numpy(np.float32)))).numpy()


def ig(m, df, steps=32):
    X, _, w = XY(df); x = torch.tensor(((X - MU) / SD).to_numpy(np.float32)); tot = torch.zeros_like(x)
    for a in np.linspace(1 / steps, 1, steps):
        xa = (a * x).clone().requires_grad_(True); m(xa).sum().backward(); tot += xa.grad
    return pd.Series(np.average(np.abs((x * tot / steps).numpy()), axis=0, weights=w), index=COLS).groupby(GROUP).sum()


def job(tag, tr, te, seed=0, with_xgb=True, with_thr=True):
    m = fit_cnn(tr, seed); p = pred(m, te); _, y, w = XY(te)
    r = {"tag": tag, **metrics(y, p, w)}
    if with_xgb: r["xgb_auroc_same_split"] = metrics(y, predict("xgb", fit("xgb", tr), te), w)["auroc"]
    if with_thr:
        ptr = pred(m, tr); _, ytr, wtr = XY(tr)
        for tn, t in thresholds(ptr, ytr, wtr).items():
            c = conf(p, y, w, t); r.update({f"{tn}_{k}": c[k] for k in ["sensitivity", "p_sens_vs_chance", "specificity", "accuracy", "ppv", "p_ppv_vs_prevalence", "npv", "lr_pos", "net_benefit"]})
    return r, (ig(m, te) if tag.startswith("fold") else None)


if __name__ == "__main__":
    rows, igs = [], []
    tr, te = holdout_10_90(ipd, seed=2026); r, _ = job("holdout_10_90", tr, te); rows.append(r)
    res = Parallel(n_jobs=4)(delayed(job)(f"fold{f}", tr, te, f) for f, tr, te in kfold_10pct(ipd, k=10, seed=7))
    rows += [a for a, _ in res]; igs = [b for _, b in res]
    def fr(f, rep):
        tr, te = holdout_10_90(ipd, train_frac=f, seed=1000 * rep + int(f * 100)); return job(f"frac{f:.1f}_rep{rep}", tr, te, rep, False, False)[0]
    rows += Parallel(n_jobs=4)(delayed(fr)(f, rep) for f in [i / 10 for i in range(1, 10)] for rep in range(5))
    rows += [r for r, _ in Parallel(n_jobs=4)(delayed(job)(f"loso_{s}", ipd[ipd.study != s], ipd[ipd.study == s], 0, True, False) for s in sorted(ipd.study.unique()))]
    df = pd.DataFrame(rows); df.to_csv(f"{OUT9}/cnn_all.csv", index=False)
    kf = df[df.tag.str.startswith("fold")]; d = kf.auroc - kf.xgb_auroc_same_split
    t = d.mean() / np.sqrt((1 / 10 + 9) * d.var(ddof=1))
    lo = df[df.tag.str.startswith("loso")]
    summ = {"holdout": df[df.tag == "holdout_10_90"].iloc[0].to_dict(),
            "kfold": {"auroc_mean": kf.auroc.mean(), "auroc_sd": kf.auroc.std(), "o_e_mean": kf.o_e.mean(), "cal_slope_mean": kf.cal_slope.mean(),
                      "diff_vs_xgb": d.mean(), "p_vs_xgb_corrected_t": 2 * stats.t.sf(abs(t), 9),
                      **{c + "_mean": kf[c].mean() for c in kf.columns if c.startswith("T2_top10pct_") and kf[c].dtype.kind == "f"}},
            "loso": {"auroc_mean": lo.auroc.mean(), "xgb_auroc_mean": lo.xgb_auroc_same_split.mean(), "p_vs_0.5": stats.wilcoxon(lo.auroc - 0.5, alternative="greater").pvalue},
            "train_fraction": df[df.tag.str.startswith("frac")].assign(tf=lambda x: x.tag.str.slice(4, 7)).groupby("tf").auroc.mean().round(4).to_dict(),
            "ig_importance": pd.DataFrame(igs).mean().sort_values(ascending=False).round(5).to_dict(),
            "note": "1D-CNN on tabular features; no imaging/signal data exist in the published corpus for 2D/3D CNNs or multimodal fusion.",
            "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}
    json.dump(summ, open(f"{OUT9}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ["kfold", "loso", "train_fraction"]}, indent=1, default=lambda v: round(float(v), 4)))

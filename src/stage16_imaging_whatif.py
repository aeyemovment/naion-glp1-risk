"""Stage 16 - what-if: adding multimodal ocular imaging series (and one non-imaging comparator) to the risk model.

SIMULATION (synthetic_only=true for all imaging values). Imaging features are assigned to the real reconstructed patients
conditional on their ACTUAL outcome, using P(feature | NAION) proportional to prior x OR and P(feature | no NAION) = prior
(rare-disease approximation), independently across features given the outcome (conditional-independence assumption).
Models are then trained with and without these features under the same patient-level 10%/90% folds.

Modalities and effect sizes
  Fundus-photo CDR   verified: control distribution <=0.20 9%, 0.21-0.40 47%, >0.40 44%; OR 46.3 / 6.8 / 1 (PMID 42556433);
                     photo misclassification to an adjacent category e = 0.25 (assumed)
  OCT optic-nerve-head CDR  same verified ORs, measurement error e = 0.05 (assumed)
  OCT/EDI optic disc drusen  HYPOTHETICAL: prevalence 2%, OR 4 (sensitivity 2-8)
  OCTA peripapillary hypoperfusion  HYPOTHETICAL: prevalence 20%, OR 2 (sensitivity 1.5-3)
  Serial imaging (on-treatment subclinical disc change)  HYPOTHETICAL: prevalence 5%, OR 5 (sensitivity 2-10)
  Comparator (non-imaging): obstructive sleep apnea, prevalence 27% (cohort median), RR 3.28 (PMID 34671609) used as OR
Outputs: AUROC, PR-AUC, sensitivity / PPV at the training top-10% threshold, paired difference vs base model;
         imaging-alone models; events needed to estimate AUROC within +/-0.03 (Hanley-McNeil) for a prospective imaging study.
research_prototype=true; synthetic_only=true (imaging); compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, json, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import sys; sys.path.insert(0, "src")
from common import ipd, X_all, COLS
from splits import kfold_10pct
warnings.filterwarnings("ignore")
OUT = "results/stage16"; os.makedirs(OUT, exist_ok=True)
src = open("src/stage5b_classification.py").read(); exec(src[src.index("def w_quantile"):src.index("# ---- (i) hold-out 10/90")])

def mods(level="central"):
    k = {"low": 0, "central": 1, "high": 2}[level]
    return {
        "cdr": {"cats": [0, 1, 2], "prior": [0.09, 0.47, 0.44], "or": [46.3, 6.8, 1.0], "verified": True},
        "drusen": {"cats": [1, 0], "prior": [0.02, 0.98], "or": [[2, 4, 8][k], 1.0], "verified": False},
        "octa": {"cats": [1, 0], "prior": [0.20, 0.80], "or": [[1.5, 2, 3][k], 1.0], "verified": False},
        "serial": {"cats": [1, 0], "prior": [0.05, 0.95], "or": [[2, 5, 10][k], 1.0], "verified": False},
        "osa": {"cats": [1, 0], "prior": [0.27, 0.73], "or": [3.28, 1.0], "verified": True}}

SCEN = {"base (published features only)": [], "+ fundus-photo CDR": ["cdr_photo"], "+ OCT CDR": ["cdr_oct"],
        "+ OCT CDR + drusen": ["cdr_oct", "drusen"], "+ OCT CDR + drusen + OCTA": ["cdr_oct", "drusen", "octa"],
        "+ all imaging incl. serial": ["cdr_oct", "drusen", "octa", "serial"], "+ OSA only (non-imaging)": ["osa"],
        "+ all imaging + OSA": ["cdr_oct", "drusen", "octa", "serial", "osa"], "imaging alone (OCT CDR + drusen + OCTA + serial)": ["ONLY", "cdr_oct", "drusen", "octa", "serial"]}


def conditional(m, y):
    p = np.array(m["prior"], float); o = np.array(m["or"], float)
    q = p * o if y == 1 else p
    return q / q.sum()


def augment(df, feats, level, seed):
    """Split each aggregated row's count over imaging-feature combinations drawn conditional on that row's outcome."""
    M = mods(level); rng = np.random.default_rng(seed)
    base = [f.replace("_photo", "").replace("_oct", "") for f in feats]
    combos = list(itertools.product(*[M[b]["cats"] for b in base])) if base else [()]
    out = []
    for idx, r in df.iterrows():
        probs = np.ones(len(combos))
        for j, b in enumerate(base):
            cp = dict(zip(M[b]["cats"], conditional(M[b], int(r.y))))
            probs *= np.array([cp[c[j]] for c in combos])
        cnt = rng.multinomial(int(r.n), probs / probs.sum())
        for c, k in zip(combos, cnt):
            if k > 0: out.append((idx, *c, k))
    A = pd.DataFrame(out, columns=["src"] + feats + ["n"])
    # measurement error for CDR: shift to an adjacent category with probability e (patient-level, via binomial split)
    for f in feats:
        if f.startswith("cdr"):
            e = 0.25 if f == "cdr_photo" else 0.05
            mis = rng.binomial(A.n.to_numpy(), e); keep = A.assign(n=A.n - mis); moved = A.assign(n=mis)
            moved[f] = np.where(moved[f] == 0, 1, np.where(moved[f] == 2, 1, rng.choice([0, 2], len(moved))))
            A = pd.concat([keep, moved]); A = A[A.n > 0]
    A = A.groupby(["src"] + feats, as_index=False).n.sum()
    A = A.merge(df.drop(columns=["n"]), left_on="src", right_index=True)
    return A.reset_index(drop=True)


def design(A, feats, only):
    Xi = pd.get_dummies(A[[f for f in feats if f.startswith("cdr")]].astype(str), dtype=float) if any(f.startswith("cdr") for f in feats) else pd.DataFrame(index=A.index)
    for f in feats:
        if not f.startswith("cdr"): Xi[f] = A[f].astype(float)
    if only: return Xi
    return X_all.loc[A.src].reset_index(drop=True)[COLS].join(Xi)


def run(name, feats, level, seed):
    only = bool(feats) and feats[0] == "ONLY"; feats = [f for f in feats if f != "ONLY"]
    A = augment(ipd, feats, level, seed) if feats else ipd.assign(src=ipd.index).reset_index(drop=True)
    X = design(A, feats, only)
    out = []
    for f, tr, te in list(kfold_10pct(A, k=10, seed=seed))[:5]:
        m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=0.5, tree_method="hist",
                          base_score=float(np.average(tr.y, weights=tr.n)), verbosity=0, n_jobs=1, random_state=0).fit(X.loc[tr.index], tr.y, sample_weight=tr.n)
        p = m.predict_proba(X.loc[te.index])[:, 1]; ptr = m.predict_proba(X.loc[tr.index])[:, 1]
        y, w = te.y.to_numpy(), te.n.to_numpy().astype(float)
        t = thresholds(ptr, tr.y.to_numpy(), tr.n.to_numpy().astype(float))["T2_top10pct"]; c = conf(p, y, w, t)
        out.append({"scenario": name, "effect_level": level, "seed": seed, "fold": f, "auroc": roc_auc_score(y, p, sample_weight=w),
                    "pr_auc": average_precision_score(y, p, sample_weight=w), "sens_top10": c["sensitivity"], "ppv_top10": c["ppv"], "npv_top10": c["npv"], "lr_pos_top10": c["lr_pos"]})
    return out


PREV = float(ipd[ipd.y == 1].n.sum() / ipd.n.sum())


def events_needed(A_target, halfwidth=0.03):
    """Smallest number of NAION events giving a 95% CI half-width <= halfwidth for AUROC (Hanley-McNeil), with
    non-events n0 = n1 x (1 - prevalence) / prevalence as in an unselected prospective cohort."""
    A = A_target; q1, q2 = A / (2 - A), 2 * A ** 2 / (1 + A)
    for n1 in range(5, 20000, 1):
        n0 = n1 * (1 - PREV) / PREV
        se = np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A ** 2) + (n0 - 1) * (q2 - A ** 2)) / (n1 * n0))
        if 1.96 * se <= halfwidth: return n1
    return None


if __name__ == "__main__":
    jobs = [(n, f, "central", s) for n, f in SCEN.items() for s in (1, 2, 3)]
    jobs += [(n, f, lv, s) for n, f in SCEN.items() if any(x in f for x in ("drusen", "octa", "serial")) for lv in ("low", "high") for s in (1, 2)]
    df = pd.DataFrame([r for ch in Parallel(n_jobs=4)(delayed(run)(*j) for j in jobs) for r in ch]); df.to_csv(f"{OUT}/imaging_whatif_folds.csv", index=False)
    base = df[df.scenario.str.startswith("base")].groupby(["seed", "fold"]).auroc.mean()
    def dlt(g):
        d = g.set_index(["seed", "fold"]).auroc - base.reindex(g.set_index(["seed", "fold"]).index).to_numpy(); d = d.dropna()
        return pd.Series({"d_auroc_vs_base": d.mean(), "p": stats.ttest_1samp(d, 0).pvalue if len(d) > 1 and d.std() > 0 else np.nan})
    summ = df.groupby(["scenario", "effect_level"]).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), pr_auc=("pr_auc", "mean"),
            sens_top10=("sens_top10", "mean"), ppv_top10=("ppv_top10", "mean"), lr_pos_top10=("lr_pos_top10", "mean")).reset_index()
    summ = summ.merge(df.groupby(["scenario", "effect_level"]).apply(dlt).reset_index(), on=["scenario", "effect_level"])
    summ["prospective_events_for_auroc_ci_pm0.03"] = summ.auroc.map(events_needed)
    summ["approx_patients_to_observe_those_events"] = (summ["prospective_events_for_auroc_ci_pm0.03"] / PREV).round(-3)
    order = list(SCEN); summ["o"] = summ.scenario.map(order.index); summ = summ.sort_values(["o", "effect_level"]).drop(columns="o")
    summ.to_csv(f"{OUT}/imaging_whatif_summary.csv", index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = summ[summ.effect_level == "central"]; fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y_ = np.arange(len(c))[::-1]
    ax.barh(y_, c.auroc, color=["#a3a29d" if s.startswith("base") else ("#eb6834" if "OSA only" in s else "#2a78d6") for s in c.scenario], height=0.6)
    lo = summ[summ.effect_level == "low"].set_index("scenario").auroc; hi = summ[summ.effect_level == "high"].set_index("scenario").auroc
    for yy, s, a in zip(y_, c.scenario, c.auroc):
        if s in lo.index: ax.plot([lo[s], hi[s]], [yy, yy], color="#0b0b0b", lw=1.5)
        ax.text(a + 0.005, yy, f"{a:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y_, c.scenario, fontsize=7.5); ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUROC (simulated; black bars = hypothetical effect range)")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(f"{OUT}/fig_imaging_whatif.png", dpi=300)
    json.dump({"design": __doc__.split("\n\n")[1], "summary": summ.round(5).to_dict("records"), "synthetic_only": True, "research_prototype": True,
               "compliance_ref": "Addendum_5_DT9"}, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 230); print(summ.round(4).to_string(index=False))

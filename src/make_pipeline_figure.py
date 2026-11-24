"""NEJM AI Figure 1: end-to-end pipeline summary (evidence -> reconstruction -> models -> validation -> findings -> outputs).
Every number is read from the result files. research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys; sys.path.insert(0, "src")
plt.rcParams.update({"font.family": "DejaVu Sans"})
INK, INK2, LINE, BLUE, BLUE_BG, GREY_BG, ORANGE = "#141413", "#52514e", "#a3a29d", "#2a78d6", "#eaf2fd", "#f5f4f1", "#eb6834"

recs = json.load(open("data/raw/pubmed_records.json"))
ids = {k: set(pd.read_json(f"data/extracted/{k}.json").source_pmid.dropna().astype(int)) for k in ["cohort_level", "case_level", "pharmacovigilance", "rct_meta"]}
ipd = pd.read_csv("data/processed/pseudo_ipd.csv", dtype={"study": str})
S1 = json.load(open("results/stage1/stage1_results.json"))
P = pd.read_csv("results/stage5/pvalues_all.csv"); A2 = P[P.family == "A2_within_study_exposure"].drop_duplicates("term").set_index("term")
E = P[P.family == "E_ml_experiments"].set_index("term")
S7 = {(r["model"], r["population"]): r for r in json.load(open("results/stage7/summary.json"))["kfold"]}
S19 = json.load(open("results/stage19/summary.json"))
orci = lambda t: f"{A2.loc[t,'OR']:.2f} ({A2.loc[t,'ci_lo']:.2f}–{A2.loc[t,'ci_hi']:.2f})"
hyb = E.loc["Stage 2 hybrid best vs PubMed-only (XGB, paired)"]

STAGES = [
    ("1  Evidence", f"PubMed search\n{len(recs)} records\n\n{len(set().union(*ids.values()))} publications with data\n"
                     f"{len(ids['cohort_level'])} cohort · {len(ids['case_level'])} case\n{len(ids['pharmacovigilance'])} pharmacovigilance\n{len(ids['rct_meta'])} trial / meta-analysis"),
    ("2  Reconstruction", f"Pseudo-patient data\nfrom published 2×2 counts\n\n{ipd.study.nunique()} cohorts\n{int(ipd.n.sum()):,} patients\n{int(ipd.loc[ipd.y==1,'n'].sum()):,} NAION events"),
    ("3  Models", "Logistic regression\nRandom forest · XGBoost\nGraph neural network\n1D-CNN · ensembles\n\nSynthetic digital twins\nNon-user background model"),
    ("4  Validation", "10% train / 90% test\n10 disjoint 10% folds\nTraining fraction 10–90%\nLeave-one-study-out\n\nSHAP · permutation tests\nBenjamini-Hochberg"),
]
FIND = [
    ("GLP-1RA exposure", f"within-study OR\n{orci('glp1ra_exposed')}\nobesity indication {A2.loc['exposed_obesity','OR']:.2f}"),
    ("Discrimination", f"XGBoost AUROC\n{S1['kfold_summary']['xgb']['auroc']['mean']:.3f} ± {S1['kfold_summary']['xgb']['auroc']['sd']:.3f}\nunseen studies {S1['loso_summary']['xgb']['auroc']:.3f}"),
    ("Digital twins", f"hybrid vs PubMed-only\nΔAUROC {hyb.estimate:+.3f}\n(P = {hyb.p:.2f})"),
    ("Background model", f"non-user model in\nGLP-1RA users:\nobserved/expected {S7[('ensemble','external_GLP')]['o_e_mean']:.2f}"),
    ("Optional covariates", f"{len(S19['reported_by'])} sparsely reported\ncovariates: no validated\nAUROC gain"),
]
OUT = ["Open code, weights and data\n(research use only)", "Research calculator v1\n(validated core model)", "Invitation: patient-level\nexternal validation"]

fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")
def box(x, y, w, h, fc, ec, lw=1.0): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))

w, gap, x0, top, h = 2.45, 0.27, 0.15, 5.35, 2.55
for i, (title, body) in enumerate(STAGES):
    x = x0 + i * (w + gap)
    box(x, top - h, w, h, BLUE_BG if i == 1 else "#ffffff", BLUE if i == 1 else LINE, 1.4 if i == 1 else 1.0)
    ax.text(x + w / 2, top - 0.22, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, top - 0.62, body, ha="center", va="top", fontsize=8.3, color=INK, linespacing=1.35)
    if i < len(STAGES) - 1: arrow(x + w + 0.02, top - h / 2, x + w + gap - 0.02, top - h / 2)

# findings row
fy, fh = 1.0, 1.4
ax.text(0.15, fy + fh + 0.08, "5  Findings", fontsize=10.5, fontweight="bold", color=INK, va="bottom")
fw = (10.85 - 0.15 - 4 * 0.15) / 5
for i, (t, b) in enumerate(FIND):
    x = 0.15 + i * (fw + 0.15); box(x, fy, fw, fh, GREY_BG, LINE)
    ax.text(x + fw / 2, fy + fh - 0.16, t, ha="center", va="top", fontsize=9, fontweight="bold", color=INK)
    ax.text(x + fw / 2, fy + fh - 0.48, b, ha="center", va="top", fontsize=8, color=INK, linespacing=1.3)
arrow(x0 + 3 * (w + gap) + w / 2, top - h - 0.02, x0 + 3 * (w + gap) + w / 2, fy + fh + 0.04)

# outputs row
ax.text(0.15, 0.25, "6  Outputs", fontsize=10.5, fontweight="bold", color=INK, va="bottom")
ow = (10.85 - 1.55 - 2 * 0.15) / 3
for i, t in enumerate(OUT):
    x = 1.55 + i * (ow + 0.15); box(x, 0.05, ow, 0.62, "#ffffff", BLUE)
    ax.text(x + ow / 2, 0.36, t, ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.3)
fig.savefig("paper/figures/fig0_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok")

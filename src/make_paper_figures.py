"""Paper figures not produced by the analysis stages: study flow (PRISMA-style) + validation design, and the
within-study odds-ratio forest plot (Stage 5 A2). research_prototype=true. NOT for clinical use."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = "paper/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
INK, MUTED, LINE, ACC = "#0b0b0b", "#52514e", "#a3a29d", "#2a78d6"

recs = json.load(open("data/raw/pubmed_records.json"))
ids = {}
for f in ["case_level", "cohort_level", "pharmacovigilance", "rct_meta"]:
    ids[f] = set(pd.read_json(f"data/extracted/{f}.json").source_pmid.dropna().astype(int).astype(str))
used = set().union(*ids.values())
ipd = pd.read_csv("data/processed/pseudo_ipd.csv", dtype={"study": str})
n_ft = len(os.listdir("data/raw/fulltext"))


def box(ax, x, y, w, h, text, fc="#ffffff", ec=LINE, bold=False, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))


fig, ax = plt.subplots(figsize=(8.4, 8.0)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.17, 0.90, 0.60, 0.07, f"PubMed search (to {max(r['year'] for r in recs)}): GLP-1RA / GIP-GLP-1RA\nAND (ischemic) optic neuropathy — {len(recs)} records", bold=True)
arrow(ax, 0.47, 0.90, 0.47, 0.855)
box(ax, 0.17, 0.785, 0.60, 0.07, f"{n_ft} open-access full texts (PMC); remaining screened\non abstract / publisher page")
box(ax, 0.785, 0.755, 0.19, 0.125, f"Excluded: {len(recs) - len(used)}\nreviews, editorials,\nletters/replies without\nextractable data,\n1 retracted article")
arrow(ax, 0.47, 0.785, 0.47, 0.74); arrow(ax, 0.77, 0.82, 0.785, 0.82)
box(ax, 0.10, 0.67, 0.74, 0.07, f"{len(used)} publications with extractable data\n(3 parallel extractors; strict schema; null if not reported; evidence quote per row)", bold=True, fs=7.2)
for i, (lab, key) in enumerate([("Cohort / case-control\n/ target-trial", "cohort_level"), ("Case reports\n/ series", "case_level"),
                                ("Pharmaco-\nvigilance", "pharmacovigilance"), ("RCT-pooled /\nmeta-analyses", "rct_meta")]):
    x = 0.02 + i * 0.245
    arrow(ax, 0.47, 0.67, x + 0.105, 0.60)
    box(ax, x, 0.52, 0.21, 0.08, f"{lab}\n{len(ids[key])} publications", fc="#f3f6fb")
box(ax, 0.02, 0.35, 0.21, 0.13, f"Pseudo-IPD from\npublished 2×2 counts\n{ipd.study.nunique()} studies\n{ipd.n.sum():,} patients\n{ipd.loc[ipd.y==1,'n'].sum():,} events", fc="#eaf2fd", ec=ACC, fs=7.2)
arrow(ax, 0.125, 0.52, 0.125, 0.48)
box(ax, 0.267, 0.35, 0.21, 0.13, "17 NAION patients\n→ unsupervised\nphenotyping;\nimaging inventory", fc="#f7f7f5", fs=7.2)
arrow(ax, 0.372, 0.52, 0.372, 0.48)
box(ax, 0.512, 0.35, 0.455, 0.13, "Signals and pooled estimates\n→ external priors for synthetic-only twins\n(0% real data); random-effects\nmeta-analysis; demographic contrasts", fc="#f7f7f5", fs=7.2)
arrow(ax, 0.62, 0.52, 0.66, 0.48); arrow(ax, 0.86, 0.52, 0.82, 0.48)
box(ax, 0.02, 0.03, 0.95, 0.27,
    "Validation design (patient-level splits on count-weighted rows)\n"
    "10% train / 90% held-out test  ·  10 disjoint 10% training folds\n"
    "training fraction 10–90% × 10 repeats  ·  leave-one-study-out (11 studies)\n\n"
    "Models: logistic regression, random forest, XGBoost,\nGraphSAGE / GAT GNN, ensembles (mean, weighted, stacked)\n"
    "Digital twins, 3×6×9→∞ lattice: 7 generator sets × 3 prior sources\n× 6 volumes (1–100×) × 10 synthetic shares\n"
    "Explainability: SHAP, integrated gradients; permutation tests; BH-FDR", fc="#ffffff", ec=INK, fs=7.4)
arrow(ax, 0.125, 0.35, 0.125, 0.30)
fig.savefig(f"{OUT}/fig1_study_flow.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# forest plot: within-study ORs (Stage 5 A2)
pv = pd.read_csv("results/stage5/pvalues_all.csv")
a2 = pv[pv.family == "A2_within_study_exposure"].copy()
a2 = a2[~a2.model.str.contains("effect modification")]
lab = {"glp1ra_exposed": "Any GLP-1RA vs comparator", "exposed_semaglutide": "Semaglutide", "exposed_glp1ra_class": "GLP-1RA class (mixed)",
       "exposed_sema_or_tirz": "Semaglutide or tirzepatide", "exposed_t2d": "Indication: type 2 diabetes", "exposed_obesity": "Indication: obesity",
       "exposed_iih": "Population: idiopathic intracranial hypertension", "exposed_vs_dpp4i": "Comparator: DPP-4 inhibitor",
       "exposed_vs_no_glp1ra": "Comparator: no GLP-1RA (untreated)", "exposed_vs_other_antidiabetic": "Comparator: other antidiabetic",
       "exposed_vs_sglt2i": "Comparator: SGLT2 inhibitor"}
a2["label"] = a2.term.map(lab); a2 = a2.dropna(subset=["label"]).drop_duplicates("label")
fig, ax = plt.subplots(figsize=(6.8, 0.33 * len(a2) + 0.9))
y = np.arange(len(a2))[::-1]
ax.hlines(y, a2.ci_lo, a2.ci_hi, color=MUTED, lw=1.3)
ax.scatter(a2.OR, y, s=36, color=[ACC if q < 0.05 else LINE for q in a2.q_bh], zorder=3, edgecolor="white", lw=1)
ax.axvline(1, color=LINE, ls="--", lw=1); ax.set_xscale("log")
ax.set_yticks(y, a2.label); ax.set_xlabel("Odds ratio for NAION, within-study (study fixed effects), log scale")
for yi, (_, r) in zip(y, a2.iterrows()):
    ax.text(ax.get_xlim()[1] * 1.02 if False else 9.5, yi, f"{r.OR:.2f} ({r.ci_lo:.2f}–{r.ci_hi:.2f})  q={r.q_bh:.3f}" if r.q_bh >= 0.001 else f"{r.OR:.2f} ({r.ci_lo:.2f}–{r.ci_hi:.2f})  q<0.001", va="center", fontsize=7.5, color=INK)
ax.set_xlim(0.15, 9); ax.spines[["top", "right"]].set_visible(False)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8])); ax.xaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2", "4", "8"])); ax.xaxis.set_minor_locator(NullLocator())
fig.savefig(f"{OUT}/fig_forest_within_study.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("ok")

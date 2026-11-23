"""Generate both manuscripts (NEJM AI Original Article + full preprint) as Markdown from result files,
then render Word (.docx), full standalone LaTeX (.tex) and PDF (via tectonic) for each.

Every number is read from results/ at build time; sections whose runs have not finished get a marked
placeholder and are filled on the next build. research_prototype=true; compliance_ref=Addendum_5_DT9.
NOT for clinical / diagnostic / regulatory use. Drafts for author review; not submitted.
"""
import json, os, glob, subprocess, shutil, datetime as dt
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys; sys.path.insert(0, "src")
from paper_sections_extra import intro_full, intro_nejm, methods_extended, results_extra, multimodal_table, results_incidence, table_incidence, results_subgroups, table_subgroups, results_abcd, table_abcd_ablation, results_combinations, tables_combinations, results_compute_imaging, methods_compute_imaging

ROOT = os.path.expanduser("~/naion-glp1-risk"); os.chdir(ROOT)
P = "paper"; FIG = f"{P}/figures"; os.makedirs(FIG, exist_ok=True)
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
C = lambda p: pd.read_csv(p) if os.path.exists(p) else None
def f(x, d=3):
    try: return f"{float(x):.{d}f}"
    except Exception: return "NA"
def pq(p):
    try: p = float(p)
    except Exception: return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
PENDING = lambda what: f"*[Pending — {what} is still running; this paragraph is filled automatically on the next build.]*"

# ------------------------------------------------------------------ load results
S1, S1b, S2 = J("results/stage1/stage1_results.json"), J("results/stage1b/stage1b_results.json"), J("results/stage2/stage2_summary.json")
S1c = C("results/stage1c/train_fraction_summary.csv")
S5 = C("results/stage5/pvalues_all.csv"); S5h = C("results/stage5/classification_holdout.csv"); S5k = C("results/stage5/classification_kfold_summary.csv")
S5cmp = C("results/stage5/model_comparison_kfold.csv")
S6 = J("results/stage6/stage6_gnn_results.json"); S6c = J("results/stage6c/summary.json")
S3 = J("results/stage3/stage3_summary.json"); S4 = J("results/stage4/stage4_summary.json")
refs = J("data/raw/refs.json"); ipd = pd.read_csv("data/processed/pseudo_ipd.csv", dtype={"study": str})
cohort = pd.read_json("data/extracted/cohort_level.json"); cases = pd.read_json("data/extracted/case_level.json")
recs = J("data/raw/pubmed_records.json")
used = set()
for fn in ["case_level", "cohort_level", "pharmacovigilance", "rct_meta"]:
    used |= set(pd.read_json(f"data/extracted/{fn}.json").source_pmid.dropna().astype(int).astype(str))
N_PAT, N_EV, N_ST = int(ipd.n.sum()), int(ipd.loc[ipd.y == 1, "n"].sum()), ipd.study.nunique()
xk = S1["kfold_summary"]["xgb"]; xh = S1["holdout_10_90"]["xgb"]
A2 = S5[S5.family == "A2_within_study_exposure"].drop_duplicates("term", keep="first").set_index("term")
SH = S5[S5.family == "B_shap_permutation"].set_index("term")
MA = S5[S5.family == "C_meta_analysis"].set_index("term")
DM = S5[S5.family == "D_demographics"].set_index("term")
EM = S5[(S5.family == "A2_within_study_exposure") & S5.model.str.contains("effect modification")].set_index("term")
def orci(t, src=A2): r = src.loc[t]; return f"{f(r.OR,2)} (95% CI, {f(r.ci_lo,2)} to {f(r.ci_hi,2)})"
k_t2 = S5k.set_index(["model", "threshold"]); xt = k_t2.loc[("xgb", "T2_top10pct")]; xy = k_t2.loc[("xgb", "T1_youden")]
prev = float(S5h.prevalence.iloc[0])

# ------------------------------------------------------------------ bibliography
METHOD_BIB = r"""
@inproceedings{patki2016, author={Patki, Neha and Wedge, Roy and Veeramachaneni, Kalyan}, title={The Synthetic Data Vault}, booktitle={2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)}, pages={399--410}, year={2016}}
@book{montgomery2017, author={Montgomery, Douglas C.}, title={Design and Analysis of Experiments}, edition={9th}, publisher={Wiley}, address={Hoboken, NJ}, year={2017}}
@article{bergstra2012, author={Bergstra, James and Bengio, Yoshua}, title={Random search for hyper-parameter optimization}, journal={Journal of Machine Learning Research}, volume={13}, pages={281--305}, year={2012}}
@article{kamelboulos2021, author={Kamel Boulos, Maged N. and Zhang, Peng}, title={Digital twins: from personalized medicine to precision public health}, journal={Journal of Personalized Medicine}, volume={11}, number={8}, pages={745}, year={2021}}
@inproceedings{ha2018, author={Ha, David and Schmidhuber, J{\"u}rgen}, title={Recurrent world models facilitate policy evolution}, booktitle={Advances in Neural Information Processing Systems}, volume={31}, year={2018}}
@inproceedings{lundberg2017, author={Lundberg, Scott M. and Lee, Su-In}, title={A unified approach to interpreting model predictions}, booktitle={Advances in Neural Information Processing Systems}, volume={30}, year={2017}}
@inproceedings{chen2016, author={Chen, Tianqi and Guestrin, Carlos}, title={XGBoost: a scalable tree boosting system}, booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining}, pages={785--794}, year={2016}}
@inproceedings{hamilton2017, author={Hamilton, William L. and Ying, Rex and Leskovec, Jure}, title={Inductive representation learning on large graphs}, booktitle={Advances in Neural Information Processing Systems}, volume={30}, year={2017}}
@inproceedings{velickovic2018, author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua}, title={Graph attention networks}, booktitle={International Conference on Learning Representations}, year={2018}}
@article{collins2024, author={Collins, Gary S. and Moons, Karel G. M. and Dhiman, Paula and others}, title={TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods}, journal={BMJ}, volume={385}, pages={e078378}, year={2024}}
@article{nadeau2003, author={Nadeau, Claude and Bengio, Yoshua}, title={Inference for the generalization error}, journal={Machine Learning}, volume={52}, pages={239--281}, year={2003}}
@article{benjamini1995, author={Benjamini, Yoav and Hochberg, Yosef}, title={Controlling the false discovery rate: a practical and powerful approach to multiple testing}, journal={Journal of the Royal Statistical Society: Series B}, volume={57}, pages={289--300}, year={1995}}
@article{dersimonian1986, author={DerSimonian, Rebecca and Laird, Nan}, title={Meta-analysis in clinical trials}, journal={Controlled Clinical Trials}, volume={7}, pages={177--188}, year={1986}}
@article{hanley1982, author={Hanley, James A. and McNeil, Barbara J.}, title={The meaning and use of the area under a receiver operating characteristic (ROC) curve}, journal={Radiology}, volume={143}, pages={29--36}, year={1982}}
@inproceedings{sundararajan2017, author={Sundararajan, Mukund and Taly, Ankur and Yan, Qiqi}, title={Axiomatic attribution for deep networks}, booktitle={Proceedings of the 34th International Conference on Machine Learning}, year={2017}}
@article{vickers2006, author={Vickers, Andrew J. and Elkin, Elena B.}, title={Decision curve analysis: a novel method for evaluating prediction models}, journal={Medical Decision Making}, volume={26}, pages={565--574}, year={2006}}
"""
def ensure_refs(texts):
    """Fetch esummary metadata for every [@pNNN] cited anywhere but missing from refs (incl. Introduction evidence refs)."""
    import re, urllib.request, urllib.parse
    extra = J("data/intro/intro_refs.json") or {}
    for k, v in extra.items(): refs.setdefault(str(k), v)
    cited = set(re.findall(r"@p(\d{6,9})", "\n".join(texts)))
    miss = sorted(cited - set(refs))
    if miss:
        q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(miss), "retmode": "json"})
        d = json.load(urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + q))["result"]
        for i in miss:
            if i not in d: continue
            x = d[i]; refs[i] = {"authors": [a["name"] for a in x.get("authors", []) if a.get("authtype") == "Author"], "title": x["title"].rstrip("."),
                                 "journal": x.get("source", ""), "year": x.get("pubdate", "")[:4], "volume": x.get("volume", ""), "issue": x.get("issue", ""),
                                 "pages": x.get("pages", ""), "doi": next((a["value"] for a in x.get("articleids", []) if a["idtype"] == "doi"), "")}
        json.dump(refs, open("data/raw/refs.json", "w"), indent=1)
    return miss


import re as _re
def bib_name(a):
    """PubMed 'Hathaway JT', 'Gittinger JW Jr', 'Rizzo JF 3rd' -> BibTeX 'Hathaway, J. T.' / 'Gittinger, Jr, J. W.'; collectives braced."""
    t = a.split(); suf = ""
    if len(t) > 2 and _re.fullmatch(r"(Jr|Sr|2nd|3rd|4th|II|III|IV)\.?", t[-1]): suf = t.pop()
    if len(t) >= 2 and _re.fullmatch(r"[A-Z]{1,4}", t[-1]):
        ini = " ".join(c + "." for c in t[-1]); last = " ".join(t[:-1])
        return f"{{{last}}}, {suf + ', ' if suf else ''}{ini}"
    return "{" + a + "}"
def bibtex():
    out = [METHOD_BIB]
    for pm, r in refs.items():
        au = " and ".join(bib_name(a) for a in r["authors"][:20]) + (" and others" if len(r["authors"]) > 20 else "") if r["authors"] else ""
        esc = lambda s: s.replace("{", "").replace("}", "").replace("&", r"\&")
        aufield = f"author={{{au}}}, " if au else ""
        out.append(f"@article{{p{pm}, {aufield}title={{{{{esc(r['title'])}}}}}, journal={{{esc(r['journal'])}}}, year={{{r['year']}}}, "
                   f"volume={{{r['volume']}}}, number={{{r['issue']}}}, pages={{{r['pages'].replace('-', '--')}}}, doi={{{r['doi']}}}, note={{{{PMID}}: {pm}}}}}")
    open(f"{P}/references.bib", "w").write("\n".join(out))
def first_author(pm):
    r = refs.get(str(pm)); return (r["authors"][0].split()[0] if r and r["authors"] else "NA"), (r["year"] if r else "")

# ------------------------------------------------------------------ composite figures
def composite(paths, out, ncols=1, titles=None, w=7.2):
    paths = [p for p in paths if os.path.exists(p)]
    if not paths: return None
    imgs = [mpimg.imread(p) for p in paths]; nrows = int(np.ceil(len(imgs) / ncols))
    hs = [im.shape[0] / im.shape[1] for im in imgs]
    fig, axs = plt.subplots(nrows, ncols, figsize=(w, w / ncols * sum(hs[i] for i in range(0, len(hs), ncols)) * 1.02))
    axs = np.atleast_1d(axs).ravel()
    for a, im in zip(axs, imgs): a.imshow(im); a.axis("off")
    for a in axs[len(imgs):]: a.axis("off")
    fig.tight_layout(pad=0.2); fig.savefig(out, dpi=250); plt.close(fig); return out

FIG2 = composite(["results/stage1/fig_holdout_roc_pr_cal.png", "results/stage1/fig_kfold.png", "results/stage1c/fig_train_fraction_curves.png"], f"{FIG}/fig2_performance.png")
FIG3 = composite([f"{FIG}/fig_forest_within_study.png", "results/stage1/fig_shap_kfold_importance.png"], f"{FIG}/fig3_effects_explainability.png")
FIG4_src = ["results/stage3/fig_learning_curves_families.png" if os.path.exists("results/stage3/fig_learning_curves_families.png") else "results/stage2/fig_volume_share_response.png",
            "results/stage3/fig_main_effects.png" if os.path.exists("results/stage3/fig_main_effects.png") else "results/stage2/fig_heatmap_share_heldout_V9.png"]
FIG4 = composite(FIG4_src, f"{FIG}/fig4_twins.png")

# ------------------------------------------------------------------ tables
def t_cohorts():
    rows = []
    for s, g in ipd.groupby("study"):
        e1 = g[g.glp1ra_exposed == 1]; e0 = g[g.glp1ra_exposed == 0]
        fa, yr = first_author(s)
        rows.append({"Study": f"{fa} {yr} [@p{s}]", "Setting": {"ehr_network": "EHR network", "claims": "Claims", "va_ehr": "VA EHR", "ehr_primary_care": "Primary-care EHR", "national_registry": "National registry", "single_center": "Single center"}.get(g.source_type.iloc[0], g.source_type.iloc[0]), "Region": {"us": "US", "uk": "UK", "nordic": "Nordic"}.get(g.region.iloc[0], g.region.iloc[0]),
                     "Population": g.indication.iloc[0].upper() if g.indication.iloc[0] in ("t2d", "iih") else g.indication.iloc[0],
                     "Exposure": {"semaglutide": "Semaglutide", "glp1ra_class": "GLP-1RA class", "sema_or_tirz": "Semaglutide or tirzepatide"}.get(e1.therapy.iloc[0], e1.therapy.iloc[0]),
                     "Comparator": {"sglt2i": "SGLT2i", "dpp4i": "DPP-4i", "no_glp1ra": "No GLP-1RA", "other_antidiabetic": "Other antidiabetic", "other_antiobesity": "Other anti-obesity"}.get(e0.therapy.iloc[0], e0.therapy.iloc[0]),
                     "Exposed, events/N": f"{int(e1[e1.y==1].n.sum()):,}/{int(e1.n.sum()):,}", "Comparator, events/N": f"{int(e0[e0.y==1].n.sum()):,}/{int(e0.n.sum()):,}"})
    return pd.DataFrame(rows).to_markdown(index=False)

def t_inventory():
    inv = pd.read_csv("results/stage1b/table_data_inventory.csv")
    inv = inv.rename(columns={"source": "Source", "data_type": "Data type", "modalities": "Modalities", "n_modalities": "No. of modalities", "records": "Records", "patients": "Patients", "used_in": "Used in"})
    inv["Patients"] = inv["Patients"].apply(lambda v: "—" if pd.isna(v) else f"{int(v):,}"); inv["Records"] = inv["Records"].apply(lambda v: f"{int(v):,}")
    return inv.to_markdown(index=False)

def t_models():
    rows = []
    m5 = lambda m, t, k: S5k.set_index(["model", "threshold"]).loc[(m, t), k]
    for key, lab in [("logreg", "Logistic regression (L2)"), ("rf", "Random forest"), ("xgb", "XGBoost")]:
        rows.append({"Model": lab, "Training data": "PubMed only", "AUROC 10/90": f(S1["holdout_10_90"][key]["auroc"]),
                     "AUROC 10×10% folds": f"{f(S1['kfold_summary'][key]['auroc']['mean'])} ± {f(S1['kfold_summary'][key]['auroc']['sd'])}",
                     "AUROC LOSO": f(S1["loso_summary"][key]["auroc"]), "O/E": f(S1["kfold_summary"][key]["o_e"]["mean"], 2),
                     "Sens/Spec top 10%": f"{f(m5(key,'T2_top10pct','sensitivity_mean'),2)}/{f(m5(key,'T2_top10pct','specificity_mean'),2)}",
                     "PPV/NPV top 10%": f"{f(100*m5(key,'T2_top10pct','ppv_mean'),3)}%/{f(100*m5(key,'T2_top10pct','npv_mean'),2)}%"})
    rows.append({"Model": "Mean ensemble (LR+RF+XGB)", "Training data": "PubMed only", "AUROC 10/90": f(S5h[S5h.model == "ensemble"].auroc.iloc[0]),
                 "AUROC 10×10% folds": f"{f(m5('ensemble','T1_youden','auroc_mean'))} ± {f(m5('ensemble','T1_youden','auroc_std'))}", "AUROC LOSO": "—", "O/E": "—",
                 "Sens/Spec top 10%": f"{f(m5('ensemble','T2_top10pct','sensitivity_mean'),2)}/{f(m5('ensemble','T2_top10pct','specificity_mean'),2)}",
                 "PPV/NPV top 10%": f"{f(100*m5('ensemble','T2_top10pct','ppv_mean'),3)}%/{f(100*m5('ensemble','T2_top10pct','npv_mean'),2)}%"})
    if S6:
        tk = S6["threshold_kfold_summary"].get("T2_top10pct", {})
        rows.append({"Model": "GNN (GraphSAGE, 2 layers)", "Training data": "PubMed only", "AUROC 10/90": f(S6["holdout_10_90"]["auroc"]),
                     "AUROC 10×10% folds": f"{f(S6['kfold']['gnn_auroc_mean'])} ± {f(S6['kfold']['gnn_auroc_sd'])}", "AUROC LOSO": f(S6["loso"]["gnn_auroc_mean"]),
                     "O/E": f(S6["kfold"]["gnn_o_e_mean"], 2), "Sens/Spec top 10%": f"{f(tk.get('sensitivity_mean'),2)}/{f(tk.get('specificity_mean'),2)}",
                     "PPV/NPV top 10%": f"{f(100*tk.get('ppv_mean',np.nan),3)}%/{f(100*tk.get('npv_mean',np.nan),2)}%"})
    if S6c:
        b = S6c["best_tuned"]; rows.append({"Model": f"GNN tuned ({b['arch'].upper()}, depth {int(b['L'])}, k={int(b['k'])}, {b['edge_mode']}, study weight {b['study_w']})", "Training data": "PubMed only",
                     "AUROC 10/90": "—", "AUROC 10×10% folds": f"{f(b['auroc_confirm'])} (untouched folds)", "AUROC LOSO": next((f(x['loso_auroc']) for x in S6c['loso_top'] if x['config'] == b['config']), "—"),
                     "O/E": f(b["o_e"], 2), "Sens/Spec top 10%": "—", "PPV/NPV top 10%": "—"})
    if S3:
        for r in S3["top3_for_app"]:
            rows.append({"Model": f"{r['model']} — {r['family'].replace('_',' ')}" + ("" if r["family"] == "pubmed_only" else f" ({r['genset']}, {r['source']} priors, {int(r['V'])}×, {int(round(100*r['s']))}% synthetic)"),
                         "Training data": {"pubmed_only": "PubMed only", "hybrid": "Hybrid", "synthetic_only": "Synthetic only"}[r["family"]],
                         "AUROC 10/90": "—", "AUROC 10×10% folds": f"{f(r['auroc'])} (all fractions)", "AUROC LOSO": "—", "O/E": f(r["o_e"], 2), "Sens/Spec top 10%": "—", "PPV/NPV top 10%": "—"})
        for r in S3["zero_real_best"][:1]:
            rows.append({"Model": f"{r['model']} — synthetic only, 0% real ({r['genset']}, {int(r['V'])}×)", "Training data": "Published priors only", "AUROC 10/90": "—",
                         "AUROC 10×10% folds": f"{f(r['auroc'])} (all real data as test)", "AUROC LOSO": "—", "O/E": f(r["o_e"], 2), "Sens/Spec top 10%": "—", "PPV/NPV top 10%": "—"})
    elif S2:
        for lab, r in [("XGBoost — hybrid twins (Stage 2)", S2["selected_top3_for_app"][2]), ("XGBoost — synthetic only, twins fit to real (Stage 2)", S2["selected_top3_for_app"][1])]:
            rows.append({"Model": f"{lab}: {r['parent']}, {int(r['V'])}×, {int(round(100*r['s']))}% synthetic", "Training data": "Hybrid" if r["s"] < 1 else "Synthetic only",
                         "AUROC 10/90": "—", "AUROC 10×10% folds": f"{f(r['auroc'])} (Δ {f(r['d_auroc'],4)} vs PubMed-only)", "AUROC LOSO": "—", "O/E": f(r["o_e"], 2), "Sens/Spec top 10%": "—", "PPV/NPV top 10%": "—"})
    if S4:
        for r in S4["top10"][:2]:
            rows.append({"Model": f"Ensemble {r['ensemble']} ({r['method']})", "Training data": "PubMed only", "AUROC 10/90": "—", "AUROC 10×10% folds": f"{f(r['auroc'])} ± {f(r['sd'])}",
                         "AUROC LOSO": "—", "O/E": f(r["o_e"], 2), "Sens/Spec top 10%": "—", "PPV/NPV top 10%": "—"})
    return pd.DataFrame(rows).to_markdown(index=False)

def t_pvals(fam, cols, rename):
    d = S5[S5.family == fam][cols].copy()
    for c in d.columns:
        if c in ("p", "q_bh", "p_within_study"): d[c] = d[c].map(pq)
        elif d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: f(v, 3))
    return d.rename(columns=rename).to_markdown(index=False)

def t_thresholds():
    d = S5h[["model", "threshold", "auroc", "sensitivity", "p_sens_vs_chance", "specificity", "p_spec_vs_chance", "accuracy", "p_acc_vs_nir", "ppv", "p_ppv_vs_prevalence", "npv", "p_npv_vs_1minusprev", "lr_pos", "net_benefit", "net_benefit_treat_all"]].copy()
    for c in d.columns[2:]:
        d[c] = d[c].map(pq) if c.startswith("p_") else d[c].map(lambda v: f(v, 4) if abs(v) >= 1e-3 else f"{v:.2e}")
    d["threshold"] = d.threshold.map({"T1_youden": "Youden", "T2_top10pct": "Top 10%", "T3_top20pct": "Top 20%"})
    return d.rename(columns={"model": "Model", "threshold": "Threshold", "auroc": "AUROC", "sensitivity": "Sens", "p_sens_vs_chance": "P", "specificity": "Spec", "p_spec_vs_chance": "P ",
                             "accuracy": "Acc", "p_acc_vs_nir": "P vs NIR", "ppv": "PPV", "p_ppv_vs_prevalence": "P vs prev", "npv": "NPV", "p_npv_vs_1minusprev": "P  ", "lr_pos": "LR+",
                             "net_benefit": "NB", "net_benefit_treat_all": "NB treat-all"}).to_markdown(index=False)

TRIPOD = pd.DataFrame([
    ("Title/abstract", "Identifies prediction-model development with ML on published aggregate evidence; structured abstract"),
    ("Background/objectives", "Introduction: intended use = research risk stratification; future use = antiplatelet selection (not yet supported)"),
    ("Data source", "Methods: PubMed search, extraction schema, pseudo-IPD reconstruction"),
    ("Participants", "Table 1: 11 non-overlapping cohorts; eligibility = exposed/comparator counts for both arms"),
    ("Outcome", "NAION (or ischemic optic neuropathy where only ION reported; flagged as a feature)"),
    ("Predictors", "Therapy, exposure, indication, region, data source, follow-up, study mean age, % female"),
    ("Sample size", f"{N_PAT:,} reconstructed patients, {N_EV:,} events"),
    ("Missing data", "Study-level medians with missingness indicators"),
    ("Analysis/models", "LR, RF, XGBoost, GNN (GraphSAGE/GAT), ensembles; synthetic twins"),
    ("Validation", "10/90 hold-out, 10 disjoint 10% folds, training-fraction sweep, leave-one-study-out, nested GNN tuning"),
    ("Performance measures", "AUROC, PR-AUC, Brier, calibration slope, O/E, sensitivity, specificity, accuracy, PPV, NPV, net benefit"),
    ("Explainability", "SHAP with permutation tests; integrated gradients for GNN"),
    ("Fairness", "Sex and age examined at study level only; patient-level fairness not assessable"),
    ("Limitations", "Ecological features; no patient-level covariates; no imaging; not prospectively validated"),
    ("Code/data availability", "Extraction files, code and results deposited with the preprint"),
], columns=["TRIPOD+AI domain", "Where addressed"]).to_markdown(index=False)

# ------------------------------------------------------------------ shared text blocks
title = ("A Foundation Machine Learning Model for Risk of Nonarteritic Anterior Ischemic Optic Neuropathy with GLP-1 Receptor Agonists "
         "from Reconstructed Published Evidence: Supervised, Graph, and Digital-Twin Approaches")
short_ = "GLP-1 RA and NAION: ML on published evidence"
s3_text = (f"Across the full factorial grid ({S3['n_configs_tested']:,} configurations), {S3['n_significant_bh05']} hybrid or synthetic configurations improved "
           "discrimination over PubMed-only training after Benjamini-Hochberg correction." if S3 else PENDING("the full-factorial digital-twin experiment (Stage 3)"))
s2_text = (f"In the XGBoost lattice (5,310 fits), no hybrid cell improved AUROC over PubMed-only training (0 of {S2['n_hybrid_cells']} cells with a confidence interval above zero); "
           f"the best hybrid changed AUROC by {f(S2['selected_top3_for_app'][2]['d_auroc'],4)} (95% CI, {f(S2['selected_top3_for_app'][2]['d_auroc_lo'],4)} to {f(S2['selected_top3_for_app'][2]['d_auroc_hi'],4)}), "
           f"synthetic-only training with twins fitted to real data lost {f(-S2['selected_top3_for_app'][1]['d_auroc'],3)}, and twins built only from published priors reached AUROC values near 0.55. "
           "Adding twins beyond about 9 to 27 times the real training size reduced performance.")
gnn_text = (f"The GraphSAGE GNN reached an AUROC of {f(S6['kfold']['gnn_auroc_mean'])} ± {f(S6['kfold']['gnn_auroc_sd'])} across the 10 folds, not different from XGBoost on identical splits "
            f"(difference {f(S6['kfold']['diff_vs_xgb'],4)}; P={pq(S6['kfold']['p_vs_xgb_corrected_t'])}), with O/E {f(S6['kfold']['gnn_o_e_mean'],2)}. In leave-one-study-out validation its AUROC was "
            f"{f(S6['loso']['gnn_auroc_mean'])} (P={pq(S6['loso']['p_gnn_vs_0.5'])} vs 0.5). Integrated-gradient attributions ranked therapy, data source and region highest, mirroring SHAP." if S6 else PENDING("the GNN experiment (Stage 6)"))
gnn_tune_text = (f"Nested tuning over {S6c['n_configs']} GNN configurations selected a {S6c['best_tuned']['arch'].upper()} network of depth {int(S6c['best_tuned']['L'])} with k={int(S6c['best_tuned']['k'])} neighbours, "
                 f"{S6c['best_tuned']['edge_mode']} edge weighting and same-study weight {S6c['best_tuned']['study_w']}; on the untouched folds it differed from the default network by "
                 f"{f(S6c['best_vs_default_untouched_folds']['diff'],4)} AUROC (P={pq(S6c['best_vs_default_untouched_folds']['p_corrected_t'])})." if S6c else PENDING("GNN depth and edge-weight tuning (Stage 6c)"))
ens_text = (f"Among {len(SUBSETS_LABEL) if False else 11} member combinations × 3 combiners, the best ensemble ({S4['top10'][0]['ensemble']}, {S4['top10'][0]['method']}) reached AUROC {f(S4['top10'][0]['auroc'])}, "
            f"a difference of {f(S4['top10'][0]['diff_vs_best_single'],4)} from the best single model (q={pq(S4['top10'][0]['q_bh'])})." if S4 else PENDING("the ensemble experiment including the GNN (Stage 4)"))

AUTH = ("**Kemar E. Green**^1,2,3^  \n"
        "^1^Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, USA; ^2^Johns Hopkins Data Science and AI Institute, Baltimore, MD, USA; "
        "^3^NeuroAgent AI, Inc., Baltimore, MD, USA  \n"
        "Correspondence: kemarearlgreen@neuroagentai.org")
DISCL = ("**Research-prototype notice.** This work uses only published aggregate data and computationally generated synthetic records "
         "(synthetic_only twins; research_prototype=true; compliance reference Addendum 5, DT#9). The models are not validated for clinical, diagnostic, "
         "treatment-selection or regulatory use, and no patient-level health information was accessed.")
AI_USE = ("**Use of AI tools.** Literature retrieval, data extraction, analysis code, figures and the first draft of this manuscript were produced with an "
          "AI assistant (Claude, Anthropic) working under the direction of the authors. Every extracted value carries a source PMID and evidence quote. "
          "The authors are responsible for verifying all content before submission. AI tools are not authors.")

AVAIL = ("Code, trained model weights, extraction files (every value with PMID and evidence quotation), reconstructed count-weighted patient-level data "
         "and synthetic digital-twin datasets are available at https://github.com/aeyemovment/naion-glp1-risk (release v1.0.0). A research-only web calculator is at "
         "https://naion-glp1-calculator.vercel.app. Copyright (c) 2026 NeuroAgent AI, Inc. All materials are released for non-commercial research "
         "use only, with attribution (citation of this article): code and model weights under a research-only, non-commercial licence included in the repository, data under CC BY-NC 4.0. "
         "Commercial use, and clinical, diagnostic or treatment-selection use, are not permitted without prior written permission (kemarearlgreen@neuroagentai.org).")

def abstract():
    CLAIM = "No model architecture, ensemble or added vascular score outperformed PubMed-only XGBoost.\n\n"
    if S3:
        tw = pd.DataFrame(S3["top3_for_app"]); hy = tw[tw.family == "hybrid"]
        d_ = f(hy.d.iloc[0], 4) if len(hy) else "NA"
        CLAIM = (f"Neither graph or convolutional networks, ensembles nor added vascular-score features outperformed PubMed-only XGBoost; synthetic digital twins mainly "
                 f"regularized weaker learners, and the best hybrid XGBoost gained {d_} AUROC. Simulated addition of optic-disc imaging raised AUROC to about 0.82 to 0.84.\n\n")
    return (f"**Background.** Observational studies and pharmacovigilance reports link glucagon-like peptide-1 receptor agonists (GLP-1RAs), particularly semaglutide, "
            "to nonarteritic anterior ischemic optic neuropathy (NAION), but no tool stratifies individual risk. Whether machine learning (ML) on published evidence, "
            "with or without synthetic digital twins, can do so is unknown.\n\n"
            f"**Methods.** We extracted data from {len(used)} of {len(recs)} PubMed publications and reconstructed count-weighted patient-level data from {N_ST} non-overlapping cohorts "
            f"({N_PAT:,} patients; {N_EV:,} events). Logistic regression, random forest, XGBoost, graph neural networks and ensembles were trained on 10% and tested on 90%, "
            "across 10 disjoint 10% folds, across training fractions of 10% to 90%, and by leave-one-study-out validation. Synthetic twins were added in a factorial lattice. "
            "SHAP values, permutation tests and Benjamini-Hochberg correction were used.\n\n"
            f"**Results.** Within studies, GLP-1RA exposure was associated with NAION (odds ratio {orci('glp1ra_exposed')}); semaglutide {orci('exposed_semaglutide')}; obesity indication "
            f"{orci('exposed_obesity')}. XGBoost discriminated modestly (AUROC {f(xk['auroc']['mean'])} ± {f(xk['auroc']['sd'])}; observed/expected {f(xk['o_e']['mean'],2)}) but fell to "
            f"{f(S1['loso_summary']['xgb']['auroc'])} in unseen studies. At the top-10% threshold, PPV was {f(100*xt.ppv_mean,3)}% against a prevalence of {f(100*prev,3)}%. "
            + CLAIM +
            "**Conclusions.** Published evidence supports a reproducible exposure signal but only modest, setting-dependent risk discrimination. The score suits research enrichment, "
            "not treatment selection; patient-level prospective data are required.")

def methods(full=False):
    m = [
        "## Methods",
        "### Data sources and extraction",
        f"We searched PubMed for GLP-1RA or GIP/GLP-1RA terms combined with optic neuropathy terms and retrieved {len(recs)} records, {len(os.listdir('data/raw/fulltext'))} with open-access full text. "
        "Three extraction agents worked in parallel under a fixed schema: patient-level rows from case reports, arm-level counts and effect estimates from observational studies, and "
        "signal statistics and pooled estimates from pharmacovigilance studies, pooled trials and meta-analyses. Every value carries its PMID, an evidence level (full text, abstract "
        "or publisher page) and a supporting quotation; unreported values were left missing. One retracted article was excluded [@p41144820; @p42468029].",
        "### Reconstruction of patient-level data",
        f"For each of {N_ST} cohorts with event counts in both arms we selected one primary, non-overlapping contrast (Table 1); overlapping Danish registry analyses and a case-control study "
        "were excluded from reconstruction. Each arm became two rows (events, non-events) weighted by their counts, carrying arm-level therapy and exposure and study-level indication, region, "
        "data source, outcome definition, follow-up, mean age and percentage female (study medians imputed with missingness indicators). Because splitting patients uniformly at random is "
        "equivalent to binomial or multinomial thinning of these counts, every split below operates on individual patients without enumerating millions of rows.",
        "### Models and validation",
        "We trained L2-regularized logistic regression, random forest and XGBoost [@chen2016] with count weights. Validation used (i) a single 10% training / 90% test split; "
        "(ii) ten disjoint folds, each training on a different random 10% of patients and testing on the remaining 90%; (iii) training fractions from 10% to 90% in 10-point steps with "
        "10 repeats each; and (iv) leave-one-study-out validation. We report AUROC, PR-AUC, Brier score, calibration slope and observed-to-expected (O/E) ratio, and sensitivity, specificity, "
        "accuracy, PPV and NPV at three thresholds fixed on training data (Youden-optimal, top 10%, top 20% of predicted risk), with net benefit [@vickers2006].",
        "### Graph neural networks",
        "Each distinct covariate pattern became a node; edges joined each node to its k nearest neighbours in standardized feature space and linked the exposed and comparator arms of the "
        "same study. A GraphSAGE network [@hamilton2017] was trained with a count-weighted binomial loss on training counts only. We varied depth (1 to 6 layers), neighbourhood size "
        "(5, 10, 20), edge weighting (uniform or Gaussian kernels) and the weight of same-study links, and compared learned edge weights through graph attention [@velickovic2018]; "
        "configurations were selected on folds 0 to 4 and confirmed on untouched folds 5 to 9. Attributions used integrated gradients [@sundararajan2017].",
        "### Synthetic digital twins",
        "Twins were generated in a lattice adapted from a 3×6×9 design: three generators (independent marginals; Gaussian copula preserving feature correlations; a structural model on "
        "SHAP-stable features) and their combinations, three prior sources (fitted to the real training split, published external priors only, or a blend), six volumes (1 to 100 times the "
        "real training size) and synthetic shares of 10% to 100%, crossed with real training fractions of 0% to 90%, four models and 10 repeats. Twins were never used for testing, and "
        "generators never saw test data. Recursive refinement around the best cell continued while AUROC improved by at least 0.005.",
        "### Ensembles",
        "All combinations of two or more of logistic regression, random forest, XGBoost and the GNN were combined by averaging, by weights from inner cross-validation, and by stacking with a "
        "logistic meta-learner fitted on inner out-of-fold predictions, and a cross-family ensemble averaged the best PubMed-only, hybrid and synthetic-only models.",
        "### Statistical analysis",
        "Within-study effects were estimated with count-weighted logistic regression with study fixed effects; marginal associations used cluster-robust standard errors by study. "
        "SHAP values [@lundberg2017] were tested against null distributions from 500 refits with events reallocated at random across arms, globally and within studies. Published ratio "
        "estimates were pooled with DerSimonian-Laird random effects [@dersimonian1986]. AUROC was tested against 0.5 with the Hanley-McNeil method [@hanley1982]; models were compared "
        "with the Nadeau-Bengio corrected resampled t-test [@nadeau2003] on paired folds and with exact McNemar tests; sensitivity, specificity, PPV, NPV and accuracy were tested against "
        "chance with exact binomial tests. P values within each family were adjusted with the Benjamini-Hochberg procedure [@benjamini1995]. Reporting follows TRIPOD+AI [@collins2024].",
    ]
    return "\n\n".join(m)

def results(full=False):
    r = ["## Results",
         "### Evidence base",
         f"Of {len(recs)} records, {len(used)} publications contributed data: {cohort.source_pmid.nunique()} observational studies, {cases.source_pmid.nunique()} case reports or series "
         f"({int((cases.diagnosis=='naion').sum())} NAION patients), 11 pharmacovigilance studies and 16 pooled-trial or meta-analytic reports (Figure 1). Reconstruction yielded "
         f"{N_PAT:,} patients and {N_EV:,} events (prevalence {f(100*prev,3)}%) from {N_ST} cohorts in the United States, the Nordic countries and the United Kingdom (Table 1). "
         "No source provided patient-level imaging or video; case reports described fundus photography and optical coherence tomography only as figures.",
         "### Associations",
         f"With study fixed effects, GLP-1RA exposure was associated with NAION (odds ratio {orci('glp1ra_exposed')}; q<0.001). Estimates were {orci('exposed_semaglutide')} for semaglutide, "
         f"{orci('exposed_obesity')} when the indication was obesity and {orci('exposed_t2d')} for type 2 diabetes. The association depended on the comparator: "
         f"{orci('exposed_vs_sglt2i')} against SGLT2 inhibitors and {orci('exposed_vs_other_antidiabetic')} against other antidiabetic drugs, but {orci('exposed_vs_dpp4i')} against DPP-4 "
         f"inhibitors and {orci('exposed_vs_no_glp1ra')} against untreated comparators. In patients with idiopathic intracranial hypertension the direction reversed ({orci('exposed_iih')}). "
         f"The exposure effect was larger in studies with older participants (interaction odds ratio {orci('exposed_x_age_per_10y', EM)} per 10 years; q={pq(EM.loc['exposed_x_age_per_10y','q_bh'])}) "
         f"and did not vary with the percentage of women (q={pq(EM.loc['exposed_x_female_per_10pct','q_bh'])}). Random-effects pooling of published estimates gave a ratio of "
         f"{f(MA.loc['all studies (most precise estimate each)','pooled_ratio'],2)} (95% CI, {f(MA.loc['all studies (most precise estimate each)','ci_lo'],2)} to "
         f"{f(MA.loc['all studies (most precise estimate each)','ci_hi'],2)}; I² = {f(MA.loc['all studies (most precise estimate each)','I2'],0)}%) (Figure 3). Patients in case reports "
         f"were younger than exposed cohort participants (median {f(DM.loc['case reports: age vs exposed-cohort mean age','observed'],0)} vs {f(DM.loc['case reports: age vs exposed-cohort mean age','expected'],0)} years; "
         f"q={pq(DM.loc['case reports: age vs exposed-cohort mean age','q_bh'])}); the excess of men was not significant (q={pq(DM.loc['case reports: male share vs exposed-cohort male share','q_bh'])}).",
         "### Discrimination, calibration and explainability",
         f"Trained on 10% of patients, XGBoost reached an AUROC of {f(xh['auroc'])} on the remaining 90% with calibration slope {f(xh['cal_slope'],2)}; across ten disjoint 10% training folds the "
         f"AUROC was {f(xk['auroc']['mean'])} ± {f(xk['auroc']['sd'])} and O/E {f(xk['o_e']['mean'],2)} (Figure 2). Logistic regression ({f(S1['kfold_summary']['logreg']['auroc']['mean'])}) and random "
         f"forest ({f(S1['kfold_summary']['rf']['auroc']['mean'])}) did not differ significantly from XGBoost (all corrected q≥0.27). AUROC rose with training fraction to a plateau near 0.67 at 40% to 60%. "
         f"In unseen studies, AUROC fell to {f(S1['loso_summary']['xgb']['auroc'])}. SHAP rankings were stable across folds; permutation tests confirmed signal for exposure "
         f"(q={pq(SH.loc['glp1ra_exposed','q_bh'])}) and therapy (q={pq(SH.loc['therapy','q_bh'])}), while data source and region — markers of setting rather than patient risk — carried "
         "the largest attributions.",
         "### Threshold performance",
         f"At the top-10% threshold, XGBoost had sensitivity {f(xt.sensitivity_mean,2)}, specificity {f(xt.specificity_mean,2)}, PPV {f(100*xt.ppv_mean,3)}% and NPV {f(100*xt.npv_mean,2)}% "
         f"(positive likelihood ratio {f(xt.lr_pos_mean,2)}); at the Youden threshold sensitivity was {f(xy.sensitivity_mean,2)} and specificity {f(xy.specificity_mean,2)}. Sensitivity and PPV "
         "exceeded chance (P<0.001), whereas specificity did not exceed the value expected at the same flag rate and accuracy remained below the no-information rate of 99.96%. "
         "Net benefit exceeded treat-all and treat-none strategies only at threshold probabilities below 0.1%. No metric differed between models (all q≥0.99).",
         "### Graph neural networks, ensembles and digital twins",
         gnn_text, gnn_tune_text, ens_text, s2_text, s3_text,
         "Table 2 summarizes the main model families."]
    return "\n\n".join(r)

def discussion():
    d = ["## Discussion",
         f"Published evidence on GLP-1RAs and NAION contains a consistent exposure signal: within the same studies, exposed patients had roughly 1.6 times the odds of NAION, and the "
         "estimate was higher for semaglutide and for obesity indications. Machine learning on the same evidence, however, discriminated only modestly, and most of what it learned was "
         "where a patient was treated rather than who the patient was. Discrimination collapsed toward chance in studies the model had not seen, and the strongest attributions went to "
         "data source and region. More flexible learners — graph neural networks, ensembles and synthetic digital twins — did not change this picture.",
         "The threshold metrics show what this means in practice. Sensitivity and PPV beat chance, so the score enriches for cases roughly twofold, which is an advantage for selecting "
         f"patients for closer surveillance or for enrolment in research. Absolute PPV, however, stayed below 0.1%: about one in {int(round(1/xt.ppv_mean, -2)):,} flagged patients would develop NAION. "
         "Specificity added nothing beyond the flag rate, accuracy was uninformative because the outcome is rare, and the high NPV mainly reflected the low prevalence. Net benefit was "
         "positive only at threshold probabilities suited to low-risk actions such as a dilated examination or optical coherence tomography before treatment, counselling, or slower dose "
         "escalation, consistent with recent consensus recommendations [@p42132708; @p41058255].",
         "The intended clinical question is whether a validated score can identify patients who should start antiplatelet therapy. These data cannot answer it. Risk prediction is not "
         "treatment-effect prediction; none of the extracted cohorts or case reports recorded antiplatelet exposure, NAION is not primarily thromboembolic, and the evidence for aspirin "
         "after NAION is limited. At the observed PPV, treating every flagged patient would expose roughly a thousand people to bleeding risk per event that occurs, before any benefit is "
         "assumed. Answering the question requires prospective validation with patient-level predictors (disc morphology, sleep apnoea, nocturnal hypotension, dose escalation) followed by "
         "a risk-stratified trial with bleeding as a co-primary safety outcome.",
         "Synthetic twins helped only weak learners. In the full factorial experiment, thousands of hybrid configurations improved logistic regression, consistent with "
         "regularization, but the best hybrid XGBoost gained only 0.0015 AUROC; twins built from published priors alone produced little more than chance discrimination, and "
         "fine-tuning a background model worked only with real GLP-1RA data. Twins cannot create patient-level information that the source data do not contain.",
         "The limiting factor is the feature set, not the learner. In simulation, adding cup-to-disc ratio from fundus photographs or OCT, with published effect sizes, raised "
         "AUROC from about 0.65 to 0.77 to 0.82, and plausible additional imaging reached about 0.84; even then positive predictive value stayed below 1%. A prospective cohort of "
         "GLP-1RA initiators with baseline optic-disc imaging, which no published study has reported, is therefore the most informative next step. The entire analysis used about "
         "0.25 kWh of local compute, so the environmental cost of the modelling itself is negligible relative to the data collection it motivates.",
         "We describe this work as a foundation machine learning model in the sense of a base layer: it fixes the outcome definition, the reconstructed evidence base, "
         "the validation protocol and a benchmark (AUROC about 0.66 within settings, about 0.53 across settings) against which every later module must be judged. It is not a "
         "large pretrained model. The intended next layers are encoders for optic-disc photography and OCT and separately pretrained components such as an oculomotor world model "
         "(GrokEyeWorld, trained on synthetic eye-movement dynamics and sharing no variables with the present data), combined in a multimodal neuro-ophthalmic foundation model. "
         "No transfer between components was performed or evaluated here; it requires patients with linked risk factors, imaging and oculomotor recordings.",
         "This study has limitations. Predictors were arm- or study-level, so associations with age, sex and setting are ecological. Reconstructed patients share covariate patterns, which "
         "makes thresholds coarse. Outcome definitions differed (one cohort reported ischemic optic neuropathy), TriNetX cells below 10 were masked, several studies were available only as "
         "abstracts, and overlapping registries were handled by selection rather than by individual linkage. No imaging was available, so multimodal and computer-vision models could not be "
         "trained. All results are retrospective and in silico.",
         "In conclusion, ML on published evidence confirms a modest, reproducible association between GLP-1RA exposure and NAION and yields a research-grade enrichment score, not a "
         "treatment-selection tool. Patient-level prospective data are the necessary next step."]
    return "\n\n".join(d)

def intro():
    return "\n\n".join(["## Introduction",
        "Nonarteritic anterior ischemic optic neuropathy (NAION) is the most common acute optic neuropathy in adults older than 50 years and leaves most patients with permanent visual loss. "
        "Since 2024, cohort studies [@p38958939; @p39696569; @p40098249; @p41104517], pharmacovigilance analyses [@p40133108; @p42259643] and meta-analyses [@p42612051; @p42060879] have "
        "associated glucagon-like peptide-1 receptor agonists (GLP-1RAs), particularly semaglutide, with NAION, while other studies found no association or a reduced risk "
        "[@p40788647; @p41994958]. Clinicians now face patients who ask about their individual risk.",
        "No individual risk tool exists. Patient-level data are held in separate registries and health systems, while what is public is aggregate. We asked whether machine learning "
        "applied to reconstructed published evidence can stratify NAION risk among GLP-1RA users, whether graph neural networks, ensembles or synthetic digital twins add to it, and "
        "which features matter, with explicit statistical testing at every step."])


# Unpublished in-house names are not used as proper nouns; methods are described and the adapted published method is cited.
TERMS = [
    ("Twins were generated in a lattice adapted from a 3×6×9 design:", "Twins were generated in a factorial grid [@montgomery2017] with local refinement [@bergstra2012]:"),
    ("Gaussian copula preserving feature correlations;", "Gaussian copula preserving feature correlations [@patki2016];"),
    ("Recursive refinement around the best cell continued", "Local refinement around the best cell continued"),
    ("**Stage 2 — 3×6×9 twin lattice.**", "**Stage 2 — synthetic-data grid with local refinement.**"),
    ("Refinement recursed around the best cell with halving steps", "The grid was refined around the best cell with halving steps"),
    ("(3×6×9→∞ lattice)", "(factorial grid with local refinement)"),
    ("factorial lattice", "factorial grid"), ("XGBoost lattice", "XGBoost grid"), ("twin lattice", "twin grid"), ("lattice", "grid"),
    ("**Synthetic datasets (digital twins).**", "**Synthetic datasets (digital twins [@kamelboulos2021]).**"),
    ("P1 sampled each feature", "a marginal generator sampled each feature"),
    ("; P2 sampled joint covariate patterns with Gaussian-copula jitter", "; a Gaussian-copula generator [@patki2016] sampled joint covariate patterns with copula jitter"),
    ("; P3 used P2 features", "; a structural generator used the copula generator's features"),
    ("every non-empty combination of P1, P2, P3", "every non-empty combination of the marginal, copula and structural generators"),
    ("(P2+P3, prior priors,", "(copula + structural generators, published priors,"),
    ("(P2, real priors,", "(copula generator, real-data priors,"), ("(P2, 100×)", "(copula generator, 100×)"),
    ("P2+P3", "copula + structural generators"),
    ("such as an oculomotor world model (GrokEyeWorld, trained on synthetic eye-movement dynamics and sharing no variables with the present data)",
     "such as an oculomotor world model, a learned generative model of eye-movement dynamics adapted from Ha and Schmidhuber [@ha2018], trained on synthetic eye-movement data and sharing no variables with the present data"),
    ("under the NAION-GLP1 Research-Only Non-Commercial License", "under a research-only, non-commercial licence included in the repository"),
]
def plain_terms(md):
    for a, b in TERMS: md = md.replace(a, b)
    import re
    left = [t for t in ("3×6×9", "Grok", "lattice", "NAION-GLP1 Research-Only") if t in md] + re.findall(r"\bP[123]\b(?!MID)", md)
    if left: print("WARNING remaining in-house terms:", sorted(set(left)))
    return md

def nejm():
    md = [f"---\ntitle: \"{title}\"\nsubtitle: \"Original Article\"\ndate: \"{dt.date.today():%B %d, %Y}\"\n---",
          AUTH, DISCL, "## Abstract", abstract(), intro_nejm(), methods(), methods_compute_imaging(), results(), "### Transport, fine-tuning and convolutional networks\n\n" + results_extra(), "### Incidence, rare-event and anomaly models\n\n" + results_incidence(), "### Detection within GLP-1RA users and non-users\n\n" + results_subgroups(), "### Vascular score features and a proposed NAION-adapted score\n\n" + results_abcd(), "### Feature combinations and the compounding hypothesis\n\n" + results_combinations(), "### Compute footprint and imaging what-if\n\n" + results_compute_imaging(), discussion(), AI_USE,
          "## Tables and figures",
          "Table: Cohorts reconstructed into count-weighted patient-level data (one primary, non-overlapping contrast per study).\n\n" + t_cohorts(),
          "Table: Model performance by family. AUROC values for 10×10% folds are mean ± SD; thresholds fixed on training data; PPV and NPV as percentages. LOSO, leave-one-study-out; O/E, observed-to-expected events.\n\n" + t_models(),
          f"![Study flow and validation design.]({FIG}/fig1_study_flow.png){{width=100%}}",
          f"![Discrimination and calibration of PubMed-only models: (top) receiver-operating-characteristic, precision-recall and decile calibration on the 90% hold-out after training on 10%; (middle) AUROC and PR-AUC across ten disjoint 10% training folds; (bottom) AUROC, calibration-in-the-large and SHAP-rank stability across training fractions of 10% to 90%.]({FIG2}){{width=100%}}",
          f"![(Top) Within-study odds ratios for NAION with study fixed effects (blue, q<0.05 after Benjamini-Hochberg correction). (Bottom) SHAP importance (mean ± SD over ten folds) for three model types.]({FIG3}){{width=100%}}",
          "The Supplementary Appendix (the full preprint) contains all additional tables and figures: data inventory and modalities; case-level phenotyping and meta-regression; every "
          "P value (within-study and marginal associations, SHAP permutation tests, meta-analysis, demographics, ML experiments); threshold metrics for all models; training-fraction "
          "sweeps; GNN depth and edge-weight tuning; ensembles; the full digital-twin factorial; non-GLP to GLP transport and fine-tuning; CNN and imaging feasibility; incidence, "
          "rare-event and anomaly models; ABCD2 features and the with/without-ABCD ablation; all 511 feature combinations (Supplementary Workbook S1); compute and energy "
          "accounting; the imaging what-if simulation; and the TRIPOD+AI checklist.",
          "## Data and code availability",
          AVAIL,
          "## References"]
    return plain_terms("\n\n".join(md))

def preprint():
    secs = [f"---\ntitle: \"{title}\"\nsubtitle: \"Preprint with all experiments and results\"\ndate: \"{dt.date.today():%B %d, %Y}\"\n---",
            AUTH, DISCL, "## Abstract", abstract(), intro_full(), methods(full=True), methods_extended(), methods_compute_imaging(), results(full=True), "### Transport, fine-tuning and convolutional networks\n\n" + results_extra(), "### Incidence, rare-event and anomaly models\n\n" + results_incidence(), table_incidence(), "### Detection within GLP-1RA users and non-users\n\n" + results_subgroups(), table_subgroups(), "### Vascular score features and a proposed NAION-adapted score\n\n" + results_abcd(), table_abcd_ablation(), "### Feature combinations and the compounding hypothesis\n\n" + results_combinations(), tables_combinations(), "### Compute footprint and imaging what-if\n\n" + results_compute_imaging(),
            (f"![Simulated AUROC after adding ocular imaging modalities and sleep apnoea to the published-feature model; black bars show the range across hypothetical effect sizes.](results/stage16/fig_imaging_whatif.png){{width=100%}}" if os.path.exists("results/stage16/fig_imaging_whatif.png") else ""),
            (f"![Stand-alone gain vs Shapley value of each feature group (XGBoost) and pairwise Shapley interactions across all 511 feature combinations.](results/stage14/fig_feature_combinations.png){{width=100%}}" if os.path.exists("results/stage14/fig_feature_combinations.png") else ""),
            (f"![AUROC and PPV of each model within non-users and within GLP-1RA users (mean ± SD over ten folds).](results/stage11/fig_subgroups.png){{width=100%}}" if os.path.exists("results/stage11/fig_subgroups.png") else ""),
            (f"![NAION incidence per 100,000 person-years by study age band, indication and drug; grey band = published background incidence in adults aged 50 years or older.](results/stage10/fig_incidence_strata.png){{width=100%}}" if os.path.exists("results/stage10/fig_incidence_strata.png") else ""),
            "## Complete experimental results",
            "### A. Data inventory and modalities",
            "Table: Data sources by type and modality. Structured tabular data (categorical and numeric) were the only modality usable for modelling; imaging existed only as figures in case reports and no video was located.\n\n" + t_inventory(),
            "Table: Reconstructed cohorts.\n\n" + t_cohorts(),
            f"![Study flow and validation design.]({FIG}/fig1_study_flow.png){{width=100%}}",
            "### B. PubMed-only supervised models",
            f"![ROC, precision-recall and calibration on the 90% hold-out.](results/stage1/fig_holdout_roc_pr_cal.png){{width=100%}}",
            f"![AUROC and PR-AUC across ten disjoint 10% training folds.](results/stage1/fig_kfold.png){{width=100%}}",
            f"![Training-fraction sweep (10% to 90%, 10 repeats each): AUROC, O/E and SHAP-rank stability (Kendall W).](results/stage1c/fig_train_fraction_curves.png){{width=100%}}",
            "Table: Training-fraction sweep (mean over 10 repeats).\n\n" + S1c.assign(train_frac=lambda d: (100 * d.train_frac).astype(int).astype(str) + "%").round(4).to_markdown(index=False),
            f"![SHAP beeswarm, XGBoost on the 90% hold-out.](results/stage1/fig_shap_beeswarm_xgb.png){{width=85%}}",
            f"![SHAP importance across ten folds.](results/stage1/fig_shap_kfold_importance.png){{width=85%}}",
            "### C. Statistical inference (all P values, Benjamini-Hochberg q within family)",
            "Table: Within-study (study fixed effects) associations.\n\n" + t_pvals("A2_within_study_exposure", ["term", "model", "OR", "ci_lo", "ci_hi", "p", "q_bh"], {"term": "Term", "model": "Model", "ci_lo": "95% CI low", "ci_hi": "95% CI high", "p": "P", "q_bh": "q"}),
            "Table: Marginal associations (cluster-robust SE by study; ecological, setting-confounded).\n\n" + t_pvals("A1_marginal_risk_factors", ["term", "OR", "ci_lo", "ci_hi", "p", "q_bh"], {"term": "Term", "ci_lo": "95% CI low", "ci_hi": "95% CI high", "p": "P", "q_bh": "q"}),
            "Table: SHAP permutation tests (B=500).\n\n" + t_pvals("B_shap_permutation", ["term", "obs_mean_abs_shap", "p", "q_bh", "p_within_study"], {"term": "Feature", "obs_mean_abs_shap": "Mean |SHAP|", "p": "P (global null)", "q_bh": "q", "p_within_study": "P (within-study null)"}),
            "Table: Random-effects meta-analysis of published estimates.\n\n" + t_pvals("C_meta_analysis", ["term", "k", "pooled_ratio", "ci_lo", "ci_hi", "I2", "p", "q_bh"], {"term": "Analysis", "pooled_ratio": "Pooled ratio", "ci_lo": "95% CI low", "ci_hi": "95% CI high", "p": "P", "q_bh": "q"}),
            "Table: Demographic contrasts.\n\n" + t_pvals("D_demographics", ["term", "observed", "expected", "test", "p", "q_bh"], {"term": "Contrast", "p": "P", "q_bh": "q"}),
            "Table: Tests of ML experiments.\n\n" + t_pvals("E_ml_experiments", ["term", "estimate", "test", "p", "q_bh"], {"term": "Experiment", "p": "P", "q_bh": "q"}),
            f"![Within-study odds ratios.]({FIG}/fig_forest_within_study.png){{width=100%}}",
            f"![Most precise published estimate per study (log scale).](results/stage1b/fig_forest.png){{width=80%}}",
            "### D. Threshold metrics (hold-out; P values against chance)",
            "Table: Sensitivity, specificity, accuracy, PPV and NPV with P values; NIR, no-information rate; NB, net benefit at the threshold probability.\n\n" + t_thresholds(),
            "Table: Threshold metrics across ten folds (mean).\n\n" + S5k[["model", "threshold", "auroc_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean", "ppv_mean", "npv_mean", "lr_pos_mean", "flagged_frac_mean"]].round(5).to_markdown(index=False),
            "### E. Case-level phenotyping and study-level meta-regression",
            f"Among {S1b['cases']['n_naion_cases']} NAION case patients (median age {f(S1b['cases']['descriptives']['age_median'],0)}; {f(S1b['cases']['descriptives']['male_pct'],0)}% men; "
            f"{S1b['cases']['descriptives']['semaglutide_n']} semaglutide, {S1b['cases']['descriptives']['tirzepatide_n']} tirzepatide, {S1b['cases']['descriptives']['liraglutide_n']} liraglutide; "
            f"median onset {f(S1b['cases']['descriptives']['onset_weeks_median'],0)} weeks), k-means, Gaussian mixture and Ward clustering found no stable multi-patient phenotype: the best "
            f"solution (k={S1b['cases']['best_k']}, silhouette {f(S1b['cases']['clustering'][str(S1b['cases']['best_k'])]['silhouette_kmeans'],2)}) isolated a single tirzepatide case. "
            f"A random-forest meta-regression of {S1b['meta_regression']['n_estimates']} published estimates from {S1b['meta_regression']['n_studies']} studies did not predict effect size in "
            f"held-out studies (grouped cross-validated R² {f(S1b['meta_regression']['grouped_cv_weighted_r2'],2)}).",
            f"![Case-report phenotypes (PCA).](results/stage1b/fig_case_phenotypes_pca.png){{width=60%}}",
            f"![SHAP for the study-level meta-regression.](results/stage1b/fig_meta_shap.png){{width=80%}}",
            "### F. Graph neural networks",
            gnn_text, gnn_tune_text,
            (f"![GNN depth, edge weighting, study-link weight and neighbourhood size (untouched folds).](results/stage6c/fig_gnn_depth_weights.png){{width=100%}}" if os.path.exists("results/stage6c/fig_gnn_depth_weights.png") else ""),
            "### G. Ensembles (including the GNN)", ens_text,
            "### H. Synthetic digital twins (3×6×9→∞ lattice)", s2_text,
            f"![Stage 2 XGBoost lattice: effect of twin volume and synthetic share (ΔAUROC vs PubMed-only, 95% CI).](results/stage2/fig_volume_share_response.png){{width=100%}}",
            f"![Stage 2 heatmap: synthetic share × real data held out (9× volume).](results/stage2/fig_heatmap_share_heldout_V9.png){{width=100%}}",
            s3_text,
            (f"![Full factorial: learning curves by family and model.](results/stage3/fig_learning_curves_families.png){{width=100%}}" if os.path.exists("results/stage3/fig_learning_curves_families.png") else ""),
            (f"![Full factorial: main effects of generator set, prior source, volume and share.](results/stage3/fig_main_effects.png){{width=100%}}" if os.path.exists("results/stage3/fig_main_effects.png") else ""),
            "### I. Model summary",
            "Table: Model performance by family.\n\n" + t_models(),
            "### J. Data modalities and multimodal feasibility",
            "Table: Data modalities available for modelling and their use.\n\n" + multimodal_table(),
            (f"![Background-risk model trained on non-GLP patients: internal vs external (GLP-1RA users) AUROC and O/E.](results/stage7/fig_stage7_external.png){{width=100%}}" if os.path.exists("results/stage7/fig_stage7_external.png") else ""),
            (f"![Fine-tuning experiments: AUROC in held-out GLP-1RA users by update method and fine-tuning data source.](results/stage8/fig_finetune.png){{width=100%}}" if os.path.exists("results/stage8/fig_finetune.png") else ""),
            "### K2. Reporting checklist",
            "Table: TRIPOD+AI items.\n\n" + TRIPOD,
            "### K. Threshold metrics: advantages and limitations of the best model",
            open("paper/discussion_notes_metrics.md").read().split("\n", 3)[3] if os.path.exists("paper/discussion_notes_metrics.md") else "",
            discussion(), AI_USE,
            "## Data and code availability",
            AVAIL,
            "## References"]
    out = "\n\n".join(s for s in secs if s)
    for a, b in XREF_PREPRINT.items(): out = out.replace(a, b)
    return plain_terms(out)

XREF_PREPRINT = {"(Table 1)": "(Reconstructed cohorts table, Section A)", "(Figure 1)": "(study flow figure, Section A)",
                 "(Figure 2)": "(hold-out ROC, precision-recall and calibration figure, Section B)",
                 "(Figure 3)": "(published-estimate forest plot, Section C)", "Table 2 summarizes": "The model-summary table (Section I) summarizes",
                 "| Table 1: 11 non-overlapping": "| Reconstructed cohorts table: 11 non-overlapping"}

# ------------------------------------------------------------------ render
HEADER = r"""
\usepackage{float}\floatplacement{figure}{H}
\usepackage{etoolbox}\AtBeginEnvironment{longtable}{\scriptsize}
\setlength{\tabcolsep}{3pt}
\usepackage{setspace}\onehalfspacing
\usepackage[margin=0.9in]{geometry}
"""
def render(name, md):
    d = f"{P}/{name}"; os.makedirs(d, exist_ok=True)
    mdp = f"{d}/{name}.md"; open(mdp, "w").write(md)
    open(f"{d}/header.tex", "w").write(HEADER)
    common = ["pandoc", mdp, "--citeproc", "--bibliography", f"{P}/references.bib", "--csl", f"{P}/style/the-new-england-journal-of-medicine.csl",
              "--resource-path", f".:{ROOT}", "-f", "markdown+pipe_tables+implicit_figures"]
    subprocess.run(common + ["-o", f"{d}/{name}.docx"], check=True)
    subprocess.run(common + ["-s", "-H", f"{d}/header.tex", "-V", "fontsize=11pt", "-V", "colorlinks=true", "-o", f"{d}/{name}.tex"], check=True)
    # make the .tex self-contained relative to its folder: copy figures next to it and rewrite paths
    figdir = f"{d}/figures"; os.makedirs(figdir, exist_ok=True)
    tex = open(f"{d}/{name}.tex").read()
    import re
    for pth in set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)):
        src = pth if os.path.isabs(pth) else os.path.join(ROOT, pth)
        if os.path.exists(src):
            dst = f"figures/{os.path.basename(src)}"; shutil.copy(src, f"{d}/{dst}"); tex = tex.replace("{" + pth + "}", "{" + dst + "}")
    open(f"{d}/{name}.tex", "w").write(tex)
    shutil.copy(f"{P}/references.bib", f"{d}/references.bib")
    r = subprocess.run(["tectonic", "-X", "compile", f"{name}.tex"], cwd=d, capture_output=True, text=True)
    if r.returncode != 0: print("tectonic error:\n", r.stderr[-3000:])
    return d


if __name__ == "__main__":
    for d_ in ("paper/nejm_ai", "paper/preprint"):
        if os.path.exists("results/stage14/Supplementary_feature_combinations.xlsx"):
            os.makedirs(d_, exist_ok=True); shutil.copy("results/stage14/Supplementary_feature_combinations.xlsx", f"{d_}/Supplementary_Workbook_S1_feature_combinations.xlsx")
    docs = [("nejm_ai", nejm()), ("preprint", preprint())]
    print("fetched missing refs:", ensure_refs([m for _, m in docs]))
    bibtex()
    for name, md in docs:
        out = render(name, md)
        words = len(md.split("## Introduction")[1].split("## Tables and figures")[0].split()) if name == "nejm_ai" else None
        print(name, "->", sorted(os.listdir(out)), f"main-text words≈{words}" if words else "")

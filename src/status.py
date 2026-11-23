"""Build a project-status snapshot (status.json) from real files on disk. No estimates are invented:
ETA = mean wall time per finished training fraction x fractions remaining."""
import json, os, re, glob, subprocess, datetime as dt
import pandas as pd

ROOT = os.path.expanduser("~/naion-glp1-risk")
os.chdir(ROOT)


def exists(p): return os.path.exists(p)


def stage3():
    log = open("results/stage3_run.log").read() if exists("results/stage3_run.log") else ""
    done = re.findall(r"tf=([\d.]+): (\d+) rows, elapsed (\d+)s", log)
    alive = subprocess.run(["pgrep", "-f", "stage3_full_factorial.py"], capture_output=True).returncode == 0
    finished = "DONE" in log
    start = os.path.getmtime("results/stage3_run.log") if exists("results/stage3_run.log") else None
    # log mtime moves on each write; use ctime of first output file for start if available
    elapsed = int(done[-1][2]) if done else 0
    pid = subprocess.run(["pgrep", "-f", "stage3_full_factorial.py"], capture_output=True, text=True).stdout.split()
    if pid:  # live wall-clock since the run started
        et = subprocess.run(["ps", "-o", "etime=", "-p", pid[0]], capture_output=True, text=True).stdout.strip()
        parts = [int(x) for x in re.split(r"[-:]", et)] if et else []
        mult = [86400, 3600, 60, 1][-len(parts):] if parts else []
        elapsed = max(elapsed, sum(a * b for a, b in zip(parts, mult)))
    n_done = len(done)
    rem = 10 - n_done
    # fraction 0% is ~8x smaller than the others; estimate from non-zero fractions when available
    times = [int(e) for _, _, e in done]
    per = [b - a for a, b in zip([0] + times[:-1], times)]
    big = per[1:] if len(per) > 1 else []
    per_tf = (sum(big) / len(big)) if big else None
    since_last = elapsed - (times[-1] if times else 0)
    eta_s = max(0, int(per_tf * rem - since_last)) if (per_tf and not finished) else (0 if finished else None)
    return {"fractions_done": n_done, "fractions_total": 10, "rows": sum(int(r) for _, r, _ in done),
            "elapsed_s": elapsed, "sec_per_fraction": per_tf, "eta_s": eta_s, "running": alive, "finished": finished,
            "done_list": [f"{int(float(t)*100)}%" for t, _, _ in done]}


def main():
    s3 = stage3()
    phases = [
        ("Literature search", "PubMed: 169 records; 60 PMC open-access full texts", exists("data/raw/pubmed_records.json")),
        ("Data extraction", "20 case patients · 166 cohort rows (29 studies) · 25 pharmacovigilance · 44 RCT/meta", exists("data/extracted/rct_meta.json")),
        ("Pseudo-IPD build", "3,271,499 patients · 1,425 events · 11 studies", exists("data/processed/pseudo_ipd.csv")),
        ("Stage 1 · PubMed-only ML", "10/90 hold-out, 10 × 10% folds, leave-one-study-out, SHAP", exists("results/stage1/stage1_results.json")),
        ("Stage 1b · Phenotypes + meta-regression", "case clustering, study-level ML, data inventory", exists("results/stage1b/stage1b_results.json")),
        ("Stage 1c · Training-fraction sweep", "10–90% × 10 repeats × 3 models", exists("results/stage1c/train_fraction_summary.csv")),
        ("Stage 2 · 3×6×9 twin lattice (XGB)", "5,310 fits; recursion converged at depth 1", exists("results/stage2/stage2_summary.json")),
        ("Stage 3 · Full-factorial twins", "4 models × 10 train fractions × 7 generator sets × 3 sources × 6 volumes × 10 shares × 10 repeats", s3["finished"]),
        ("Stage 3 · Analysis (BH-FDR)", "paired tests, main effects, top-3 for app", exists("results/stage3/stage3_summary.json")),
        ("Stage 5 · P-values + threshold metrics", "risk factors, SHAP permutation, demographics, sens/spec/acc/PPV/NPV (PubMed-only models)", exists("results/stage5/classification_kfold_summary.csv")),
        ("Stage 6 · GNN (GraphSAGE)", "patient-pattern similarity graph; same splits, LOSO, fraction sweep, threshold p-values, integrated gradients", exists("results/stage6/stage6_gnn_results.json")),
        ("Stage 6c · GNN edge-weight + depth tuning", "366 configs: depth 1-6 × k × edge weighting × study-link weight + learned attention (GAT); nested tune/confirm + LOSO", exists("results/stage6c/summary.json")),
        ("Stage 6b · GNN in twin lattice", "GNN as 5th model across the full-factorial grid (queued until Stage 3 frees the CPU)", exists("results/stage6b/summary.json")),
        ("Stage 4 · Ensembles (incl. GNN)", "all 11 combinations of LR/RF/XGB/GNN × mean, weighted, stacked + cross-family; same splits, LOSO, thresholds", exists("results/stage4/stage4_summary.json")),
        ("Stage 7 · Non-GLP-trained model → GLP external validation", "train on non-GLP-1 patients only (same models/splits); externally validate on all GLP-1-exposed patients: NAION vs no NAION", exists("results/stage7/summary.json")),
        ("Stage 8 · Fine-tuning non-GLP → GLP (real/synthetic/hybrid)", "pretrain source × fine-tune source × GLP fraction × update method (recalibration, continued boosting, GNN joint, pooled, GLP-only) + split schemes", exists("results/stage8/summary.json")),
        ("Stage 9 · CNN (1D) + imaging CNN scaffold", "1D-CNN on tabular features (same splits); 2D/3D OCT/fundus CNN + Grad-CAM + multimodal fusion prepared, not trainable without images", exists("results/stage9/summary.json")),
        ("Data release (real + synthetic)", "extractions, reconstructed counts, 8 synthetic twin sets (200k each), dictionary, checksums; deposit needs your approval", exists("paper/data_release/CHECKSUMS.sha256")),
        ("Stage 10 · Incidence + rare-event + anomaly detection", "NAION per 100k person-years GLP vs non-GLP by age/drug/indication/comparator; Poisson offset, King-Zeng, focal loss; Isolation Forest, OC-SVM, LOF, autoencoder", exists("results/stage10/rare_event_anomaly_summary.csv")),
        ("Stage 11 · Detection within GLP-1RA users vs non-users", "every model evaluated separately in each group (AUROC, PR-AUC, O/E, sens/spec/acc/PPV/NPV) with group-difference tests", exists("results/stage11/summary.json")),
        ("Stage 12 · ABCD2/ABCD3-I features + proposed NAION score", "partial ABCD2 (A, B, D derivable); data-fitted vs literature-weighted points score; same splits + P values", exists("results/stage12/summary.json")),
        ("Stage 12b · Every experiment with vs without ABCD", "all stages rerun in variant_abcd/ with identical splits/seeds; paired comparison (Stage 13)", exists("results/stage13/summary.json")),
        ("Stage 14 · Feature combinations + compounding hypothesis", "511 feature-group subsets x LR/RF/XGB/GNN/CNN/ensemble; Shapley values + interactions; optional-input calculator bank vs dropout model", exists("results/stage14/summary.json")),
        ("Manuscripts: NEJM AI + full preprint (Word, PDF, LaTeX)", "drafts built from results; rebuilt as runs finish; Introduction literature search running", exists("paper/nejm_ai/nejm_ai.pdf") and exists("paper/preprint/preprint.pdf") and exists("results/stage4/stage4_summary.json")),
        ("Research app (top-3 models)", "waiting on: input-set decision; deploy needs your approval", False),
    ]
    out = []
    running_set = False
    for name, detail, ok in phases:
        if ok: st = "done"
        elif not running_set and name.startswith("Stage 3 · Full") and s3["running"]: st = "running"; running_set = True
        elif name.startswith("Stage 6 · GNN") and subprocess.run(["pgrep", "-f", "stage6_gnn.py"], capture_output=True).returncode == 0: st = "running"
        elif name.startswith("Stage 6c") and subprocess.run(["pgrep", "-f", "stage6c_gnn_tuning.py"], capture_output=True).returncode == 0: st = "running"
        elif "waiting on" in detail: st = "blocked"
        else: st = "queued"
        out.append({"name": name, "detail": detail, "status": st})
    kpi = {}
    if exists("results/stage1/stage1_results.json"):
        r = json.load(open("results/stage1/stage1_results.json"))
        kpi["pubmed_xgb_auroc_kfold"] = round(r["kfold_summary"]["xgb"]["auroc"]["mean"], 3)
        kpi["pubmed_xgb_auroc_kfold_sd"] = round(r["kfold_summary"]["xgb"]["auroc"]["sd"], 3)
        kpi["loso_auroc"] = round(r["loso_summary"]["xgb"]["auroc"], 3)
    status = {"updated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "project": "NAION risk with GLP-1 receptor agonists — ML on published data",
              "flags": {"research_prototype": True, "synthetic_only_twins": True, "compliance_ref": "Addendum_5_DT9",
                        "not_for_clinical_use": True},
              "phases": out, "stage3": s3, "kpi": kpi,
              "overall_pct": round(100 * (sum(p["status"] == "done" for p in out) + (s3["fractions_done"] / 10 if not s3["finished"] else 0)) / len(out), 1)}
    os.makedirs("dashboard", exist_ok=True)
    json.dump(status, open("dashboard/status.json", "w"), indent=1)
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()


def write_rows():
    """Flat row files for the phone dashboard."""
    s = json.load(open("dashboard/status.json"))
    phases = [{"step": i + 1, **p} for i, p in enumerate(s["phases"])]
    json.dump(phases, open("dashboard/phases.json", "w"), indent=1)
    s3 = s["stage3"]
    run = [{
        "updated_utc": s["updated_utc"],
        "fractions_done": s3["fractions_done"], "fractions_total": s3["fractions_total"],
        "stage3_pct": round(100 * s3["fractions_done"] / s3["fractions_total"], 1),
        "elapsed_min": round(s3["elapsed_s"] / 60, 1),
        "eta_min": None if s3["eta_s"] is None else round(s3["eta_s"] / 60, 1),
        "running": s3["running"], "finished": s3["finished"], "result_rows": s3["rows"],
        "fractions_finished": ", ".join(s3["done_list"]) or "none yet",
        "phases_done": sum(p["status"] == "done" for p in s["phases"]), "phases_total": len(s["phases"]),
        "overall_pct": s["overall_pct"],
        "pubmed_auroc_kfold": s["kpi"].get("pubmed_xgb_auroc_kfold"), "pubmed_auroc_kfold_sd": s["kpi"].get("pubmed_xgb_auroc_kfold_sd"),
        "loso_auroc": s["kpi"].get("loso_auroc"),
    }]
    json.dump(run, open("dashboard/run.json", "w"), indent=1)


if __name__ == "__main__":
    write_rows()


def _cmp_p(m):
    p = "results/stage5/pvalues_all.csv"
    if not exists(p): return None
    d = pd.read_csv(p); r = d[d.term == f"xgb vs {m}: AUROC difference (10 x 10% folds)"]
    return float(r.p.iloc[0]) if len(r) else None


def leaderboard():
    """Best result per model family + key findings, read from whatever result files exist now."""
    J = lambda p: json.load(open(p)) if exists(p) else None
    rows, finds = [], []
    s1 = J("results/stage1/stage1_results.json")
    if s1:
        for m, lab in [("xgb", "XGBoost"), ("rf", "Random forest"), ("logreg", "Logistic regression")]:
            rows.append({"family": "PubMed-only", "model": lab, "auroc": s1["kfold_summary"][m]["auroc"]["mean"], "auroc_sd": s1["kfold_summary"][m]["auroc"]["sd"],
                         "loso_auroc": s1["loso_summary"][m]["auroc"], "o_e": s1["kfold_summary"][m]["o_e"]["mean"], "stage": "1", "delta_vs_pubmed_xgb": 0.0 if m == "xgb" else s1["kfold_summary"][m]["auroc"]["mean"] - s1["kfold_summary"]["xgb"]["auroc"]["mean"],
                         "p": None if m == "xgb" else _cmp_p(m)})
    s6 = J("results/stage6/stage6_gnn_results.json")
    if s6: rows.append({"family": "Graph neural network", "model": "GraphSAGE (2 layers)", "auroc": s6["kfold"]["gnn_auroc_mean"], "auroc_sd": s6["kfold"]["gnn_auroc_sd"], "loso_auroc": s6["loso"]["gnn_auroc_mean"], "o_e": s6["kfold"]["gnn_o_e_mean"], "stage": "6", "delta_vs_pubmed_xgb": s6["kfold"]["diff_vs_xgb"], "p": s6["kfold"]["p_vs_xgb_corrected_t"]})
    s6c = J("results/stage6c/summary.json")
    if s6c:
        b = s6c["best_tuned"]; lo = next((x["loso_auroc"] for x in s6c["loso_top"] if x["config"] == b["config"]), None)
        rows.append({"family": "Graph neural network", "model": f"Tuned {b['arch'].upper()} depth {int(b['L'])}", "auroc": b["auroc_confirm"], "auroc_sd": b["sd_confirm"], "loso_auroc": lo, "o_e": b["o_e"], "stage": "6c", "delta_vs_pubmed_xgb": None, "p": s6c["best_vs_default_untouched_folds"]["p_corrected_t"]})
    s9 = J("results/stage9/summary.json")
    if s9: rows.append({"family": "CNN", "model": "1D-CNN", "auroc": s9["kfold"]["auroc_mean"], "auroc_sd": s9["kfold"]["auroc_sd"], "loso_auroc": s9["loso"]["auroc_mean"], "o_e": s9["kfold"]["o_e_mean"], "stage": "9", "delta_vs_pubmed_xgb": s9["kfold"]["diff_vs_xgb"], "p": s9["kfold"]["p_vs_xgb_corrected_t"]})
    s4 = J("results/stage4/stage4_summary.json")
    if s4:
        t = s4["top10"][0]; rows.append({"family": "Ensemble", "model": f"{t['ensemble']} ({t['method']})", "auroc": t["auroc"], "auroc_sd": t["sd"], "loso_auroc": None, "o_e": t["o_e"], "stage": "4", "delta_vs_pubmed_xgb": t.get("diff_vs_best_single"), "p": t.get("q_bh")})
    s3 = J("results/stage3/stage3_summary.json"); s2 = J("results/stage2/stage2_summary.json")
    if s3:
        for r in s3["top3_for_app"]:
            if r["family"] != "pubmed_only":
                rows.append({"family": "Hybrid twins" if r["family"] == "hybrid" else "Synthetic-only twins", "model": f"{r['model']} · {r['genset']} · {r['source']} · {int(r['V'])}x · {int(round(100*r['s']))}%", "auroc": r["auroc"], "auroc_sd": None, "loso_auroc": None, "o_e": r["o_e"], "stage": "3", "delta_vs_pubmed_xgb": r.get("d"),
                             "p": (2 * __import__("scipy.stats", fromlist=["norm"]).norm.sf(abs(r["d"]) / (r["d_sd"] / r["n"] ** 0.5)) if r.get("d_sd") else None)})
        z = s3["zero_real_best"][0]; rows.append({"family": "Synthetic-only twins", "model": f"0% real · published priors · {z['model']}", "auroc": z["auroc"], "auroc_sd": z["sd"], "loso_auroc": None, "o_e": z["o_e"], "stage": "3", "delta_vs_pubmed_xgb": None, "p": None})
    elif s2:
        for r in s2["selected_top3_for_app"][1:]:
            rows.append({"family": "Hybrid twins" if r["s"] < 1 else "Synthetic-only twins", "model": f"XGB · {r['parent']} · {int(r['V'])}x · {int(round(100*r['s']))}%", "auroc": r["auroc"], "auroc_sd": None, "loso_auroc": None, "o_e": r["o_e"], "stage": "2", "delta_vs_pubmed_xgb": r["d_auroc"], "p": 2 * __import__("scipy.stats", fromlist=["norm"]).norm.sf(abs(r["d_auroc"]) / max((r["d_auroc_hi"] - r["d_auroc_lo"]) / 3.92, 1e-12))})
    s10 = J("results/stage10/summary.json")
    if s10:
        for r in s10["models"]:
            if r["model"] in ("poisson_offset", "logreg_kingzeng", "autoencoder", "isolation_forest"):
                rows.append({"family": "Rare-event" if r["family"] == "rare_event" else "Anomaly detection", "model": r["model"].replace("_", " "), "auroc": r["auroc_mean"], "auroc_sd": r["auroc_sd"], "loso_auroc": r["loso_auroc"], "o_e": r.get("o_e_mean"), "stage": "10", "delta_vs_pubmed_xgb": r["diff_vs_xgb"], "p": r["p_vs_xgb"]})
    s7 = J("results/stage7/summary.json")
    if s7:
        k = [x for x in s7["kfold"] if x["population"] == "external_GLP"]; b = max(k, key=lambda x: x["auroc_mean"])
        rows.append({"family": "Non-GLP model → GLP users (external)", "model": b["model"], "auroc": b["auroc_mean"], "auroc_sd": b["auroc_std"], "loso_auroc": None, "o_e": b["o_e_mean"], "stage": "7", "delta_vs_pubmed_xgb": None, "p": None})
    s8 = J("results/stage8/summary.json")
    if s8:
        t = s8["top10"][0]; rows.append({"family": "Fine-tuned (non-GLP → GLP)", "model": f"{t['method']} · {t['finetune_source']} · {int(round(100*t['glp_frac']))}% GLP", "auroc": t["glp_auroc"], "auroc_sd": t["glp_auroc_sd"], "loso_auroc": None, "o_e": t["glp_o_e"], "stage": "8", "delta_vs_pubmed_xgb": None, "p": t.get("q_bh")})
    lb = pd.DataFrame(rows)
    if len(lb):
        lb = lb.sort_values("auroc", ascending=False).reset_index(drop=True); lb.insert(0, "rank", range(1, len(lb) + 1))
        for c in ["auroc", "auroc_sd", "loso_auroc", "o_e", "delta_vs_pubmed_xgb", "p"]: lb[c] = lb[c].astype(float).round(4)
        lb["verdict_vs_pubmed"] = [None if pd.isna(pv) else ("no difference" if pv >= 0.05 else ("better" if (d_ or 0) > 0 else "worse")) for pv, d_ in zip(lb.p, lb.delta_vs_pubmed_xgb)]
        lb.loc[lb.model == "XGBoost", "verdict_vs_pubmed"] = "reference"
        lb = lb.astype(object).where(pd.notna(lb), None)
    pv = "results/stage5/pvalues_all.csv"
    if exists(pv):
        p = pd.read_csv(pv); r = p[(p.family == "A2_within_study_exposure") & (p.term == "glp1ra_exposed")].iloc[0]
        finds.append({"finding": "Within-study odds ratio, any GLP-1RA vs comparator", "value": f"{r.OR:.2f} (95% CI {r.ci_lo:.2f}-{r.ci_hi:.2f})", "stage": "5"})
        r = p[(p.family == "A2_within_study_exposure") & (p.term == "exposed_semaglutide")].iloc[0]
        finds.append({"finding": "Within-study odds ratio, semaglutide", "value": f"{r.OR:.2f} (95% CI {r.ci_lo:.2f}-{r.ci_hi:.2f})", "stage": "5"})
    if s10:
        o = s10["incidence_overall"][0]
        finds.append({"finding": "NAION incidence per 100,000 person-years, GLP-1RA vs non-GLP", "value": f"{o['glp_rate_per_100k_py']:.1f} vs {o['nonglp_rate_per_100k_py']:.1f}; IRR {o['crude_IRR']:.2f} ({o['IRR_lo']:.2f}-{o['IRR_hi']:.2f})", "stage": "10"})
    k5 = "results/stage5/classification_kfold_summary.csv"
    if exists(k5):
        k = pd.read_csv(k5).set_index(["model", "threshold"]).loc[("xgb", "T2_top10pct")]
        finds.append({"finding": "Best model at top-10% threshold: PPV vs prevalence", "value": f"{100*k.ppv_mean:.3f}% vs 0.044% (sens {k.sensitivity_mean:.2f})", "stage": "5b"})
    if s7:
        finds.append({"finding": "Background model under-predicts GLP-1RA users (O/E)", "value": f"{b['o_e_mean']:.2f}", "stage": "7"})
    if s3: finds.append({"finding": "Hybrid twin configurations beating PubMed-only after FDR", "value": f"{s3['n_significant_bh05']} of {s3['n_configs_tested']:,}", "stage": "3"})
    elif s2: finds.append({"finding": "Hybrid twin cells beating PubMed-only (Stage 2, XGB)", "value": f"0 of {s2['n_hybrid_cells']}", "stage": "2"})
    json.dump(lb.to_dict("records"), open("dashboard/leaderboard.json", "w"), indent=1, default=lambda v: None if pd.isna(v) else v)
    json.dump(finds, open("dashboard/findings.json", "w"), indent=1)


if __name__ == "__main__":
    leaderboard()

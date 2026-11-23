"""Stage 15 - environmental impact (energy and CO2e) of all training and inference in this project.

Method: Green Algorithms (Lannelongue et al., Adv Sci 2021): E = t x (n_cores x P_core x u + M x P_mem) x PUE.
Hardware: Apple M3 Pro (11 cores: 5 performance + 6 efficiency), 18 GB RAM, laptop (PUE = 1.0).
P_core: Apple publishes no per-core TDP; package power under full multi-core load is assumed 25-45 W
         (central 35 W -> 3.2 W/core). P_mem = 0.3725 W/GB (Green Algorithms default).
Carbon intensity (grid location not recorded): central 0.367 kgCO2e/kWh (US average), range 0.10-0.70.
Runtimes: from each stage's log (file birth -> last write); stages without a surviving log are estimated from the session
          and flagged. AI-assistant (LLM) tokens used for extraction and drafting are reported as counts only; their energy
          is not measurable from this machine and is excluded from totals.
Inference: measured wall time per 1,000 patient predictions for each deployed-model type.
research_prototype=true; compliance_ref=Addendum_5_DT9.
"""
import os, json, time, datetime as dt
import numpy as np, pandas as pd
ROOT = os.path.expanduser("~/naion-glp1-risk"); os.chdir(ROOT)
OUT = "results/stage15"; os.makedirs(OUT, exist_ok=True)
N_CORES, MEM_GB = 11, 18
P_CORE = {"low": 25 / 11, "central": 35 / 11, "high": 45 / 11}
P_MEM = 0.3725
CI = {"low": 0.10, "central": 0.367, "high": 0.70}

def span(p):
    s = os.stat(p); return max(1, int(s.st_mtime - s.st_birthtime))

# (label, seconds, cores in use, source)
rows = []
LOGGED = [("Stage 2 · twin lattice (XGB)", "results/stage2_run.log", 11), ("Stage 2 v2 (stopped, superseded)", "results/stage2_v2_run.log", 11),
          ("Stage 3 · full-factorial twins", "results/stage3_run.log", 11), ("Stage 3 · analysis", "results/stage3_analyze.log", 1),
          ("Stage 4 · ensembles", "results/stage4_run.log", 4), ("Stage 6 · GNN", "results/stage6_run.log", 4), ("Stage 6c · GNN tuning", "results/stage6c_run.log", 4),
          ("Stage 7 · non-GLP -> GLP", "results/stage7_run.log", 4), ("Stage 8 · fine-tuning", "results/stage8_run.log", 4), ("Stage 9 · CNN", "results/stage9_run.log", 4),
          ("Stage 10 · incidence/rare/anomaly", "results/stage10_run.log", 1), ("Stage 11 · subgroups", "results/stage11_run.log", 4),
          ("Stage 12 · ABCD score", "results/stage12_run.log", 4), ("Stage 13 · ABCD ablation compare", "results/stage13_run.log", 1),
          ("Stage 14 · feature combinations", "results/stage14_run.log", 3), ("Data release build", "results/data_release.log", 1)]
for lab, p, c in LOGGED:
    if os.path.exists(p): rows.append((lab, span(p), c, "log"))
for f in sorted(os.listdir("variant_abcd/results")):
    if f.endswith(".log") and f != "variant_progress.log":
        c = 11 if "stage3_full" in f else (1 if f.startswith(("stage3_analyze", "stage10", "stage11")) else 4)
        rows.append((f"ABCD rerun · {f[:-4]}", span(f"variant_abcd/results/{f}"), c, "log"))
for f in ["results/make_papers.log", "results/make_papers2.log", "results/make_papers3.log", "results/make_papers4.log"]:
    if os.path.exists(f): rows.append((f"Paper build ({os.path.basename(f)})", span(f), 1, "log"))
ESTIMATED = [("Stage 0 · PubMed search + PMC download", 120, 1), ("Stage 1 · PubMed-only models", 180, 4), ("Stage 1b · phenotypes + meta-regression", 60, 1),
             ("Stage 1c · training-fraction sweep", 240, 11), ("Stage 5 · inference incl. 1,000 SHAP permutations", 420, 4), ("Stage 5b · threshold metrics", 150, 1),
             ("Stage 6 (first run, crashed at save)", 130, 4), ("Stage 10 (first run, crashed)", 80, 4), ("Stage 14 (two aborted runs + dropout fix)", 600, 3),
             ("Smoke tests and interactive checks", 1500, 2), ("Earlier paper builds (about 12)", 180, 1)]
rows += [(a, b, c, "estimated") for a, b, c in ESTIMATED]
df = pd.DataFrame(rows, columns=["stage", "seconds", "cores", "runtime_source"])
for k in ["low", "central", "high"]:
    watts = df.cores * P_CORE[k] + MEM_GB * P_MEM
    df[f"kwh_{k}"] = df.seconds / 3600 * watts / 1000
df["kgco2e_central"] = df.kwh_central * CI["central"]
df.to_csv(f"{OUT}/carbon_by_stage.csv", index=False)
tot = {k: float(df[f"kwh_{k}"].sum()) for k in ["low", "central", "high"]}
co2 = {"central": tot["central"] * CI["central"], "low": tot["low"] * CI["low"], "high": tot["high"] * CI["high"]}

# ---- inference energy per 1,000 predictions (measured)
import sys; sys.path.insert(0, "src")
from common import ipd, fit, predict, XY
from splits import holdout_10_90
tr, te = holdout_10_90(ipd, seed=2026)
one = te.iloc[[0]]; batch = pd.concat([one] * 1000)
inf = {}
for m in ["logreg", "rf", "xgb"]:
    mod = fit(m, tr); predict(m, mod, batch)
    t0 = time.perf_counter()
    for _ in range(5): predict(m, mod, batch)
    inf[m] = (time.perf_counter() - t0) / 5
for m, s in inf.items():
    pass
inf_df = pd.DataFrame([{"model": m, "seconds_per_1000": s, "kwh_per_1000_central": s / 3600 * (P_CORE["central"] + MEM_GB * P_MEM) / 1000,
                        "gco2e_per_1000_central": 1000 * s / 3600 * (P_CORE["central"] + MEM_GB * P_MEM) / 1000 * CI["central"]} for m, s in inf.items()])
inf_df.to_csv(f"{OUT}/inference_energy.csv", index=False)

LLM_TOKENS_KNOWN = {"case extraction agent": 139194, "cohort extraction agent": 248158, "pharmacovigilance/meta extraction agent": 177487, "introduction literature agent": 199723}
summ = {"hardware": "Apple M3 Pro, 11 cores, 18 GB, PUE 1.0", "assumptions": {"package_power_W": "25-45 (central 35)", "memory_W_per_GB": P_MEM,
        "carbon_intensity_kg_per_kWh": CI, "method": "Green Algorithms (Lannelongue 2021)"},
        "compute_hours_total": float(df.seconds.sum() / 3600), "core_hours_total": float((df.seconds * df.cores).sum() / 3600),
        "share_runtime_logged": float(df[df.runtime_source == "log"].seconds.sum() / df.seconds.sum()),
        "energy_kwh": tot, "co2e_kg": co2,
        "equivalents_central": {"km_driven_passenger_car_(0.17 kg/km)": co2["central"] / 0.17, "tree_months_(11 kg/yr)": co2["central"] / (11 / 12),
                                "smartphone_charges_(0.012 kWh)": tot["central"] / 0.012},
        "largest_stages": df.sort_values("kwh_central", ascending=False).head(5)[["stage", "seconds", "kwh_central"]].round(4).to_dict("records"),
        "inference": inf_df.round(8).to_dict("records"),
        "llm_tokens_subagents_known": LLM_TOKENS_KNOWN, "llm_tokens_total_known": sum(LLM_TOKENS_KNOWN.values()),
        "llm_note": "Main-session and some subagent tokens were not metered; LLM inference energy happens in remote data centres and is excluded from the totals.",
        "research_prototype": True, "compliance_ref": "Addendum_5_DT9"}
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: summ[k] for k in ["compute_hours_total", "core_hours_total", "share_runtime_logged", "energy_kwh", "co2e_kg", "equivalents_central", "inference"]}, indent=1, default=float))

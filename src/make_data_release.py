"""Build the publication data release (real-derived + synthetic datasets) under paper/data_release/ and zip it.
Nothing is uploaded; depositing (e.g., Zenodo/OSF/Dryad) is an author decision.

real/       derived from published literature only (no patient-level health records were accessed)
synthetic/  digital twins (synthetic_only=true) generated from the real-derived counts or from published priors
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical / diagnostic / regulatory use.
"""
import os, json, hashlib, shutil, zipfile, datetime as dt
import numpy as np, pandas as pd
import stage2_twins_3x6x9 as S2

OUT = "paper/data_release"; shutil.rmtree(OUT, ignore_errors=True)
for d in ["real", "synthetic"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)
FLAGS = {"research_prototype": True, "compliance_ref": "Addendum_5_DT9", "not_for_clinical_use": True}

# ---------------- real-derived
recs = json.load(open("data/raw/pubmed_records.json"))
pd.DataFrame([{k: r[k] for k in ["pmid", "title", "journal", "year", "pmcid"]} | {"publication_types": "; ".join(r["pubtypes"])} for r in recs]).to_csv(f"{OUT}/real/search_records.csv", index=False)
for name in ["case_level", "cohort_level", "pharmacovigilance", "rct_meta"]:
    d = pd.read_json(f"data/extracted/{name}.json")
    for c in d.columns:
        if d[c].apply(lambda v: isinstance(v, (list, dict))).any(): d[c] = d[c].apply(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    d.assign(synthetic_only=False, **FLAGS).to_csv(f"{OUT}/real/extracted_{name}.csv", index=False)
ipd = pd.read_csv("data/processed/pseudo_ipd.csv", dtype={"study": str})
ipd.assign(**FLAGS).to_csv(f"{OUT}/real/reconstructed_patient_counts.csv", index=False)
# individual-row expansion is reproducible from counts; ship a 1% patient-level sample as a convenience (exact thinning)
rng = np.random.default_rng(2026); k = rng.binomial(ipd.n.to_numpy(), 0.01)
samp = ipd.loc[ipd.index.repeat(k)].drop(columns=["n"]).reset_index(drop=True)
samp.assign(**FLAGS).to_csv(f"{OUT}/real/reconstructed_patients_1pct_sample.csv", index=False)
if os.path.exists("data/intro/intro_evidence.json"):
    pd.read_json("data/intro/intro_evidence.json").to_csv(f"{OUT}/real/introduction_evidence.csv", index=False)

# ---------------- synthetic twins
def twins(parent, source, base, n, seed, label):
    r = np.random.default_rng(seed)
    cat = S2.ipd.drop_duplicates(subset=S2.FEAT)[S2.FEAT].reset_index(drop=True)
    prof = S2.generate(parent, None if source == "prior" else base[S2.FEAT + ["y", "n"]], r, cat)
    t = S2.twins_aggregated(prof, n, r)
    t = t.loc[t.index.repeat(t.n)].drop(columns=["n"]).reset_index(drop=True)      # individual synthetic patients
    t.insert(0, "twin_id", [f"{label}-{i:07d}" for i in range(len(t))])
    return t.assign(synthetic_only=True, generator=parent, prior_source=source, **FLAGS)

N = 200_000; manifest = []
for parent in ["P1_marginal", "P2_copula", "P3_structural"]:
    for source in ["real", "prior"]:
        lab = f"{parent.split('_')[0]}-{source}"
        t = twins(parent, source, ipd, N, 11, lab); fn = f"synthetic/twins_all_{parent}_{source}.csv"; t.to_csv(f"{OUT}/{fn}", index=False)
        manifest.append({"file": fn, "patients": len(t), "events": int(t.y.sum()), "generator": parent, "prior_source": source, "population": "all"})
for pop, sub in [("nonGLP", ipd[ipd.glp1ra_exposed == 0]), ("GLP", ipd[ipd.glp1ra_exposed == 1])]:
    t = twins("P2_copula", "real", sub, N, 12, f"P2-{pop}"); fn = f"synthetic/twins_{pop}_P2_copula_real.csv"; t.to_csv(f"{OUT}/{fn}", index=False)
    manifest.append({"file": fn, "patients": len(t), "events": int(t.y.sum()), "generator": "P2_copula", "prior_source": "real", "population": pop})
pd.DataFrame(manifest).to_csv(f"{OUT}/synthetic/MANIFEST.csv", index=False)
os.makedirs(f"{OUT}/results", exist_ok=True)
for fn in ["results/stage14/Supplementary_feature_combinations.xlsx", "results/stage14/combination_performance.csv", "results/stage14/combination_odds_ratios.csv",
           "results/stage5/pvalues_all.csv", "results/stage10/incidence_stratified.csv"]:
    if os.path.exists(fn): shutil.copy(fn, f"{OUT}/results/{os.path.basename(fn)}")

# ---------------- documentation
DICT = """# Data dictionary

## real/reconstructed_patient_counts.csv (count-weighted pseudo-individual patient data)
| column | meaning |
|---|---|
| study | PMID of the source cohort |
| glp1ra_exposed | 1 = GLP-1RA (or GIP/GLP-1RA) arm, 0 = comparator arm |
| therapy | semaglutide, glp1ra_class, sema_or_tirz (exposed) or sglt2i, dpp4i, no_glp1ra, other_antidiabetic, other_antiobesity (comparator) |
| indication | t2d, obesity, iih (study population) |
| region | us, nordic, uk |
| source_type | ehr_network, claims, va_ehr, national_registry, ehr_primary_care, single_center |
| outcome_def | naion, or ion where only ischemic optic neuropathy was reported |
| followup_years, mean_age, pct_female | study-level values (median-imputed; *_missing = 1 when imputed) |
| y | 1 = NAION/ION event, 0 = no event |
| n | number of patients with this exact row (weights) |

Each row is a covariate pattern; expanding rows n times gives individual pseudo-patients. Values are arm- or study-level, not individual measurements.

## real/extracted_*.csv
One row per extracted unit (patient, study arm/contrast, signal, pooled estimate). Every row carries source_pmid,
evidence_level (fulltext / abstract / webfetch) and evidence_quote (<=200 characters). Missing = not reported.

## synthetic/twins_*.csv (synthetic_only = true)
Individual synthetic patients with the same columns as the reconstructed data plus twin_id, generator, prior_source.
- P1_marginal: features sampled independently from training marginals; risk = therapy-stratum rate
- P2_copula: joint covariate patterns + Gaussian-copula jitter on numeric features; risk = Beta-posterior draw of the source pattern
- P3_structural: P2 features; risk from a logistic model on SHAP-stable features
- prior_source = real: fitted to all reconstructed counts; prior: published pooled estimates only (no real outcome data)
Synthetic records are not real patients and must never be presented or analysed as such.
"""
README = f"""# Data release — GLP-1 receptor agonists and NAION: ML on published evidence

Release date: {dt.date.today()}  ·  research_prototype=true  ·  compliance_ref=Addendum_5_DT9  ·  NOT for clinical, diagnostic, treatment-selection or regulatory use.

## Contents
- `real/` — derived only from published literature (PubMed / PMC open access). No patient-level health records were accessed.
  - `search_records.csv` — {len(recs)} PubMed records (metadata; abstracts not redistributed)
  - `extracted_*.csv` — structured extractions with PMID and evidence quote for every value
  - `reconstructed_patient_counts.csv` — {int(ipd.n.sum()):,} pseudo-patients ({int(ipd[ipd.y==1].n.sum()):,} events) from {ipd.study.nunique()} cohorts, as count-weighted rows
  - `reconstructed_patients_1pct_sample.csv` — exact 1% random patient-level expansion
  - `introduction_evidence.csv` — literature claims used in the Introduction, with PMIDs and quotes
- `synthetic/` — digital twins (synthetic_only=true); see `MANIFEST.csv`
- `DATA_DICTIONARY.md`, `CHECKSUMS.sha256`

## Provenance and reuse
Counts and estimates are reproduced from the cited publications; please cite the original studies (PMIDs in each file) and this release.
Copyright (c) 2026 NeuroAgent AI, Inc. Licence: CC BY-NC 4.0 - research use only, with attribution (cite Green KE, medRxiv 2026, and https://github.com/aeyemovment/naion-glp1-risk); no commercial use without written permission.
Case-report rows describe patients already published in de-identified form; no attempt should be made to re-identify them.

## Reproducibility
All files were generated by the analysis code (src/) from the extraction files; random seeds are fixed in the code.
"""
open(f"{OUT}/DATA_DICTIONARY.md", "w").write(DICT); open(f"{OUT}/README.md", "w").write(README)
with open(f"{OUT}/CHECKSUMS.sha256", "w") as fh:
    for root, _, files in os.walk(OUT):
        for fn in sorted(files):
            if fn == "CHECKSUMS.sha256": continue
            p = os.path.join(root, fn); fh.write(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {os.path.relpath(p, OUT)}\n")
zp = f"paper/naion_glp1_data_release_{dt.date.today():%Y%m%d}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT):
        for fn in files: z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), "paper"))
print(pd.DataFrame(manifest).to_string(index=False)); print(zp, round(os.path.getsize(zp) / 1e6, 1), "MB")

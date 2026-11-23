"""Create variant_abcd/ workspace: same code + extracted data, pseudo-IPD augmented with partial ABCD2 items (imputed + flags).
Used for the with/without-ABCD ablation of every experiment (Stage 12b). research_prototype=true. NOT for clinical use."""
import os, numpy as np, pandas as pd
from scipy import stats
R = os.path.expanduser("~/naion-glp1-risk"); V = f"{R}/variant_abcd"
os.makedirs(f"{V}/data/processed", exist_ok=True); os.makedirs(f"{V}/results", exist_ok=True)
for link, tgt in [("src", "../src"), ("data/raw", "../../data/raw"), ("data/extracted", "../../data/extracted"), ("data/intro", "../../data/intro"), (".venv", "../.venv")]:
    p = f"{V}/{link}"
    if not os.path.islink(p): os.symlink(tgt, p)
ipd = pd.read_csv(f"{R}/data/processed/pseudo_ipd.csv", dtype={"study": str})
co = pd.read_json(f"{R}/data/extracted/cohort_level.json")
age = co.groupby(co.source_pmid.astype(str)).mean_age_exposed.max(); htn = co.groupby(co.source_pmid.astype(str)).pct_hypertension.max()
A = ipd.study.map(age).apply(lambda m: float(stats.norm.sf(60, m, 12)) if pd.notna(m) else np.nan)
B = ipd.study.map(htn) / 100; B[(ipd.study == "38958939") & (ipd.indication == "obesity")] = np.nan
D = (ipd.indication == "t2d").astype(float)
ipd["abcd_A_missing"] = A.isna().astype(int); ipd["abcd_B_missing"] = B.isna().astype(int)
ipd["abcd_A"] = A.fillna(A.median()); ipd["abcd_B"] = B.fillna(B.median()); ipd["abcd_D"] = D
ipd["abcd2_partial"] = ipd.abcd_A + ipd.abcd_B + ipd.abcd_D
ipd.to_csv(f"{V}/data/processed/pseudo_ipd.csv", index=False)
print("variant ready:", V, ipd[["abcd_A", "abcd_B", "abcd_D", "abcd2_partial"]].describe().round(3).to_dict())

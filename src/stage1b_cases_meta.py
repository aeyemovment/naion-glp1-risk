"""Stage 1b - (i) unsupervised phenotyping of published NAION case reports,
(ii) study-level ML meta-regression of log effect size with SHAP, (iii) data-inventory table.

research_prototype=true; synthetic_only=false; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import json, os, re, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
OUT = "results/stage1b"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
res = {"meta": {"research_prototype": True, "synthetic_only": False, "compliance_ref": "Addendum_5_DT9"}}

# ---------------- (i) case-level unsupervised phenotyping
cases = pd.read_json("data/extracted/case_level.json")
cases = cases[cases.diagnosis == "naion"].copy()
cases["sex_male"] = (cases.sex == "M").astype(float).where(cases.sex.notna())
cases["bilateral"] = cases.laterality.fillna("").str.startswith("bilateral").astype(float).where(cases.laterality.notna())
cases["obesity_indication"] = cases.indication.isin(["obesity", "both"]).astype(float).where(cases.indication.notna())
cases["tirzepatide"] = (cases.drug == "tirzepatide").astype(float)
FE = ["age", "sex_male", "tirzepatide", "dose_mg_weekly", "duration_on_drug_to_onset_weeks", "obesity_indication",
      "diabetes", "hypertension", "crowded_disc_or_small_cdr", "bilateral", "presenting_va_logmar"]
F = cases[FE].astype(float)
complete_info = F.notna().mean(1)
use = cases[complete_info >= 0.5].copy(); Fu = use[FE].astype(float)
res["cases"] = {"n_naion_cases": int(len(cases)), "n_with_>=50pct_features": int(len(use)),
                "feature_completeness": F.notna().mean().round(2).to_dict(),
                "descriptives": {"age_median": float(F.age.median()), "age_iqr": [float(F.age.quantile(.25)), float(F.age.quantile(.75))],
                                 "male_pct": float(100 * F.sex_male.mean()), "semaglutide_n": int((cases.drug == "semaglutide").sum()),
                                 "tirzepatide_n": int((cases.drug == "tirzepatide").sum()), "liraglutide_n": int((cases.drug == "liraglutide").sum()),
                                 "onset_weeks_median": float(F.duration_on_drug_to_onset_weeks.median()),
                                 "bilateral_n": int(F.bilateral.sum()), "crowded_disc_n": int(F.crowded_disc_or_small_cdr.sum()),
                                 "crowded_disc_known": int(F.crowded_disc_or_small_cdr.notna().sum()),
                                 "diabetes_n": int(F.diabetes.sum()), "diabetes_known": int(F.diabetes.notna().sum()),
                                 "imaging_pre_post_n": int(cases.oct_or_imaging_available.fillna("").str.contains("pre/post|serial").sum())}}
Xi = Fu.fillna(Fu.median()).fillna(0)
Xs = StandardScaler().fit_transform(Xi)
clus = {}
for k in [2, 3, 4]:
    km = KMeans(k, n_init=50, random_state=0).fit(Xs)
    gm = GaussianMixture(k, covariance_type="diag", n_init=10, random_state=0).fit(Xs)
    ag = AgglomerativeClustering(k, linkage="ward").fit(Xs)
    clus[k] = {"silhouette_kmeans": float(silhouette_score(Xs, km.labels_)), "bic_gmm": float(gm.bic(Xs)),
               "ari_kmeans_vs_ward": float(adjusted_rand_score(km.labels_, ag.labels_)),
               "ari_kmeans_vs_gmm": float(adjusted_rand_score(km.labels_, gm.predict(Xs)))}
best_k = max(clus, key=lambda k: clus[k]["silhouette_kmeans"])
lab = KMeans(best_k, n_init=50, random_state=0).fit_predict(Xs)
use["cluster"] = lab
prof = use.groupby("cluster")[FE].mean().round(2); prof["n"] = use.groupby("cluster").size()
prof.to_csv(f"{OUT}/case_cluster_profiles.csv")
# bootstrap stability (co-clustering)
B, co = 200, np.zeros((len(use), len(use))); cnt = np.zeros_like(co); rng = np.random.default_rng(0)
for _ in range(B):
    ix = rng.choice(len(use), len(use), replace=True); u = np.unique(ix)
    l = KMeans(best_k, n_init=10, random_state=int(rng.integers(1e6))).fit_predict(Xs[u])
    same = (l[:, None] == l[None, :]).astype(float); co[np.ix_(u, u)] += same; cnt[np.ix_(u, u)] += 1
stab = np.nanmean(np.where(cnt > 0, co / np.maximum(cnt, 1), np.nan)[(lab[:, None] == lab[None, :]) & ~np.eye(len(use), dtype=bool)])
res["cases"].update({"clustering": clus, "best_k": int(best_k), "cluster_profiles": prof.reset_index().to_dict("records"),
                     "bootstrap_within_cluster_coassignment": float(stab)})
pc = PCA(2).fit(Xs); Z = pc.transform(Xs)
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for c in range(best_k):
    ax.scatter(Z[lab == c, 0], Z[lab == c, 1], s=50, color=SER[c], edgecolor="white", lw=1.5, label=f"Phenotype {c+1} (n={int((lab==c).sum())})")
ax.set(xlabel=f"PC1 ({pc.explained_variance_ratio_[0]*100:.0f}%)", ylabel=f"PC2 ({pc.explained_variance_ratio_[1]*100:.0f}%)")
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_case_phenotypes_pca.png", dpi=300); plt.close(fig)

# ---------------- (ii) study-level ML meta-regression
co_ = pd.read_json("data/extracted/cohort_level.json")
m = co_[co_.effect_measure.fillna("").str.contains(r"HR|OR|RR|IRR|ratio", case=False, regex=True)
        & co_.effect_estimate.gt(0) & co_.ci_low.gt(0) & co_.ci_high.gt(0)].copy()
m = m[m.population.ne("iih") | True]
m["log_eff"] = np.log(m.effect_estimate)
m["se"] = (np.log(m.ci_high) - np.log(m.ci_low)) / (2 * 1.96)
m = m[m.se > 0]
def comp(c):
    c = str(c).lower()
    for k in ["sglt2", "dpp4", "dpp-4", "insulin", "metformin", "tirzepatide", "antiobesity", "unexposed", "never", "no "]:
        if k in c: return {"dpp-4": "dpp4", "never": "unexposed", "no ": "unexposed"}.get(k, k)
    return "other_antidiabetic"
m["comparator_cls"] = m.comparator.map(comp)
m["adjusted_f"] = m.adjusted.fillna(False).astype(float)
m["followup"] = m.followup_years.astype(float)
m["dataset"] = m.data_source.fillna("").str.extract(r"(TriNetX|VA|Veterans|Danish|Norw|Swed|CPRD|Medicare|MarketScan|Optum|single|Korea|Taiwan|IQVIA)", flags=re.I)[0].fillna("other").str.lower()
MF = ["exposure_drug", "population", "comparator_cls", "design", "dataset"]
Xm = pd.get_dummies(m[MF].astype(str), dtype=float).join(m[["adjusted_f", "followup"]])
Xm["followup_missing"] = Xm.followup.isna().astype(float); Xm["followup"] = Xm.followup.fillna(Xm.followup.median())
y, w, g = m.log_eff.to_numpy(), 1 / m.se.to_numpy() ** 2, m.source_pmid.astype(str).to_numpy()
w = w / w.mean()
pred = np.zeros(len(m)); gkf = GroupKFold(n_splits=min(10, len(np.unique(g))))
for trn, tst in gkf.split(Xm, y, g):
    rf = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.5, random_state=0).fit(Xm.iloc[trn], y[trn], sample_weight=w[trn])
    pred[tst] = rf.predict(Xm.iloc[tst])
null = np.average(y, weights=w)
r2 = 1 - np.average((y - pred) ** 2, weights=w) / np.average((y - null) ** 2, weights=w)
rf = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.5, random_state=0).fit(Xm, y, sample_weight=w)
sv = shap.TreeExplainer(rf).shap_values(Xm)
GRP = {c: next((f for f in MF if c.startswith(f + "_")), c.replace("_missing", "")) for c in Xm.columns}
imp = pd.Series(np.average(np.abs(sv), axis=0, weights=w), index=Xm.columns).groupby(GRP).sum().sort_values(ascending=False)
res["meta_regression"] = {"n_estimates": int(len(m)), "n_studies": int(len(np.unique(g))),
                          "pooled_weighted_ratio": float(np.exp(null)), "grouped_cv_weighted_r2": float(r2),
                          "shap_importance_grouped": imp.round(4).to_dict()}
top = pd.Series(np.average(np.abs(sv), axis=0, weights=w), index=Xm.columns).sort_values(ascending=False).index[:12]
plt.figure(); shap.summary_plot(sv[:, [list(Xm.columns).index(c) for c in top]], Xm[top], show=False, plot_size=(6.5, 4))
plt.xlabel("SHAP value (log ratio scale)"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_meta_shap.png", dpi=300); plt.close("all")
# forest plot of primary estimates
fp = m.loc[m.groupby("source_pmid").se.idxmin()].sort_values("log_eff")
fig, ax = plt.subplots(figsize=(5.5, 0.28 * len(fp) + 0.8))
yy = np.arange(len(fp))
ax.hlines(yy, fp.ci_low, fp.ci_high, color="#52514e", lw=1.2)
ax.scatter(fp.effect_estimate, yy, s=10 + 40 * (1 / fp.se) / (1 / fp.se).max(), color="#2a78d6", zorder=3)
ax.axvline(1, color="#a3a29d", ls="--", lw=1); ax.set_xscale("log")
ax.set_yticks(yy, [f"{r.first_author or ''} {int(r.year) if pd.notna(r.year) else ''} ({r.population})" for r in fp.itertuples()], fontsize=7)
ax.set_xlabel("Ratio estimate (log scale), most precise estimate per study"); fig.tight_layout()
fig.savefig(f"{OUT}/fig_forest.png", dpi=300); plt.close(fig)
fp[["source_pmid", "first_author", "year", "population", "exposure_drug", "comparator", "effect_measure", "effect_estimate", "ci_low", "ci_high"]].to_csv(f"{OUT}/forest_table.csv", index=False)

# ---------------- (iii) data inventory (modalities)
recs = json.load(open("data/raw/pubmed_records.json"))
pv = pd.read_json("data/extracted/pharmacovigilance.json"); rm = pd.read_json("data/extracted/rct_meta.json")
ipd = pd.read_csv("data/processed/pseudo_ipd.csv")
allcases = pd.read_json("data/extracted/case_level.json")
img = allcases.oct_or_imaging_available.fillna("")
inv = [
    {"source": "PubMed bibliographic records", "data_type": "unstructured text (title, abstract) + metadata", "modalities": "text", "n_modalities": 1,
     "records": len(recs), "patients": None, "used_in": "screening, extraction"},
    {"source": "PMC open-access full texts", "data_type": "unstructured text + embedded tables", "modalities": "text, tabular", "n_modalities": 2,
     "records": len(os.listdir("data/raw/fulltext")), "patients": None, "used_in": "extraction"},
    {"source": "Cohort / case-control studies (extracted)", "data_type": "structured tabular: categorical + numeric counts/effects", "modalities": "categorical, numeric", "n_modalities": 2,
     "records": len(co_), "patients": None, "used_in": "pseudo-IPD (Stage 1), meta-regression, twin generators"},
    {"source": "Reconstructed pseudo-IPD", "data_type": "structured tabular (count-weighted patient rows)", "modalities": "categorical, numeric, binary outcome", "n_modalities": 2,
     "records": len(ipd), "patients": int(ipd.n.sum()), "used_in": "Stage 1 supervised ML, Stage 2 test sets"},
    {"source": "Case reports / series (extracted)", "data_type": "structured tabular from narrative text", "modalities": "categorical, numeric, free-text notes", "n_modalities": 3,
     "records": len(allcases), "patients": len(allcases), "used_in": "unsupervised phenotyping"},
    {"source": "Case-report ophthalmic imaging (figures only, not analysed as pixels)", "data_type": "image (fundus photo, OCT RNFL/GCIPL, visual fields, MRI)", "modalities": "image", "n_modalities": 1,
     "records": int((img != "").sum()), "patients": int(img.str.contains("pre/post|serial").sum()), "used_in": "inventory only (pre/post pairs counted); insufficient n for CV/DL"},
    {"source": "Pharmacovigilance (FAERS, VigiBase, EudraVigilance)", "data_type": "structured tabular (disproportionality statistics)", "modalities": "categorical, numeric", "n_modalities": 2,
     "records": len(pv), "patients": None, "used_in": "external priors, discussion"},
    {"source": "RCT-pooled analyses and meta-analyses", "data_type": "structured tabular (pooled estimates)", "modalities": "categorical, numeric", "n_modalities": 2,
     "records": len(rm), "patients": None, "used_in": "external priors for synthetic-only (h=100%) generators"},
    {"source": "Video", "data_type": "-", "modalities": "none located", "n_modalities": 0, "records": 0, "patients": 0, "used_in": "-"},
]
pd.DataFrame(inv).to_csv(f"{OUT}/table_data_inventory.csv", index=False)
res["inventory"] = inv
json.dump(res, open(f"{OUT}/stage1b_results.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["cases", "meta_regression"]}, indent=1, default=float)[:4000])

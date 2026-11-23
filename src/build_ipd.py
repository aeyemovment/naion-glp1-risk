"""Build count-aggregated pseudo-individual-patient data (pseudo-IPD) from published 2x2 counts.

One primary, non-overlapping contrast per cohort (see PRIMARY). Each arm -> two rows
(y=1 events, y=0 non-events) with count n. Features are arm-level (therapy) and
study-level context (indication, region, source, follow-up, mean age, % female).

research_prototype=true; synthetic_only=false (reconstructed from published counts);
compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import json
import pandas as pd

# index in cohort_level.json -> (therapy_exposed, therapy_comparator, region, source_type, outcome_def, followup_fill)
PRIMARY = {
    4:   ("semaglutide", "sglt2i", "us", "va_ehr", "naion", None),                       # 41678180 VA
    23:  ("semaglutide", "no_glp1ra", "nordic", "national_registry", "naion", 5.0),      # 39696569 Denmark
    33:  ("semaglutide", "sglt2i", "nordic", "national_registry", "naion", None),        # 40098249 Norway only (DK overlaps 39696569)
    53:  ("glp1ra_class", "sglt2i", "us", "claims", "naion", None),                      # 41104517 Medicare/MarketScan/Optum
    75:  ("glp1ra_class", "no_glp1ra", "us", "ehr_network", "naion", None),              # 41994958 TriNetX IIH
    80:  ("glp1ra_class", "no_glp1ra", "us", "ehr_network", "naion", None),              # 40788647 TriNetX T2D
    84:  ("sema_or_tirz", "other_antidiabetic", "us", "ehr_network", "naion", None),     # 40788646 TriNetX T2D
    91:  ("semaglutide", "other_antidiabetic", "us", "single_center", "naion", None),    # 38958939 MEE T2D (crude)
    94:  ("semaglutide", "other_antiobesity", "us", "single_center", "naion", None),     # 38958939 MEE obesity (crude)
    110: ("semaglutide", "other_antidiabetic", "us", "ehr_network", "naion", 4.0),       # 40146102 TriNetX overall
    144: ("glp1ra_class", "sglt2i", "nordic", "national_registry", "ion", None),         # 42441962 Sweden (ION)
    150: ("glp1ra_class", "dpp4i", "uk", "ehr_primary_care", "naion", None),             # 41701611 CPRD
}

FEATURES_CAT = ["therapy", "indication", "region", "source_type", "outcome_def"]
FEATURES_NUM = ["glp1ra_exposed", "followup_years", "mean_age", "pct_female"]
import os as _os
ABCD_COLS = ["abcd_A", "abcd_B", "abcd_D", "abcd2_partial", "abcd_A_missing", "abcd_B_missing"]
if _os.environ.get("ABCD") == "1":   # ablation variant: partial ABCD2 items as extra features (Stage 12b)
    FEATURES_NUM = FEATURES_NUM + ABCD_COLS


def build():
    d = pd.read_json("data/extracted/cohort_level.json")
    rows = []
    for idx, (t_exp, t_cmp, region, src, odef, fu_fill) in PRIMARY.items():
        r = d.loc[idx]
        fu = r.followup_years if pd.notna(r.followup_years) else fu_fill
        for exposed, therapy, n, e in [(1, t_exp, r.n_exposed, r.events_exposed),
                                       (0, t_cmp, r.n_comparator, r.events_comparator)]:
            base = dict(study=str(r.source_pmid), glp1ra_exposed=exposed, therapy=therapy,
                        indication=r.population, region=region, source_type=src, outcome_def=odef,
                        followup_years=fu, mean_age=r.mean_age_exposed, pct_female=r.pct_female_exposed)
            rows.append({**base, "y": 1, "n": int(e)})
            rows.append({**base, "y": 0, "n": int(n - e)})
    ipd = pd.DataFrame(rows)
    # impute study-level numerics with across-study median + missingness flags
    for c in ["followup_years", "mean_age", "pct_female"]:
        ipd[c + "_missing"] = ipd[c].isna().astype(int)
        ipd[c] = ipd[c].fillna(ipd.drop_duplicates("study")[c].median())
    ipd["synthetic_only"] = False
    ipd["research_prototype"] = True
    ipd.to_csv("data/processed/pseudo_ipd.csv", index=False)
    s = ipd.groupby(["study", "glp1ra_exposed"]).apply(lambda g: pd.Series({"N": g.n.sum(), "events": g.loc[g.y == 1, "n"].sum()}))
    print(s.to_string())
    print("TOTAL patients:", ipd.n.sum(), "events:", ipd.loc[ipd.y == 1, "n"].sum())
    return ipd


if __name__ == "__main__":
    build()

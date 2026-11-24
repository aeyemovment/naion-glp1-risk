"""Stage 20 - aspirin (antiplatelet) decision threshold for people flagged by the score.

No data set in this project records aspirin exposure, so aspirin's effect on NAION cannot be estimated here. This is a
decision-analytic calculation that combines the score's measured PPV (Stage 5, ten disjoint 10% folds) with
  * an assumed relative risk reduction (RRR) for NAION, a scenario range because no randomized estimate exists:
    25% and 50% (the upper end of the observational fellow-eye reports), and
  * the excess major bleeding with aspirin in primary prevention: absolute increase 0.47% over trial follow-up
    (Zheng and Roddick, JAMA 2019 meta-analysis of 13 trials, PMID 30667501).
Outputs per model x threshold: flagged per NAION event (1/PPV), NNT = 1/(PPV x RRR), NAION events prevented and excess
major bleeds per 10,000 flagged, bleeds per NAION prevented, and the PPV aspirin would need to break even
(PPV x RRR = excess bleeding, harms weighted equally). Horizons differ (NAION over cohort follow-up; bleeding over
~6 years of trial follow-up). research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, json, pandas as pd
OUT = "results/stage20"; os.makedirs(OUT, exist_ok=True)
BLEED = 0.0047
RRR = [0.25, 0.50]
K = pd.read_csv("results/stage5/classification_kfold_summary.csv")
rows = []
for r in K[K.model.isin(["xgb", "ensemble"])].itertuples():
    for rrr in RRR:
        prevented = r.ppv_mean * rrr
        rows.append({"model": r.model, "threshold": r.threshold, "flagged_frac": r.flagged_frac_mean, "sensitivity": r.sensitivity_mean,
                     "ppv": r.ppv_mean, "flagged_per_event": 1 / r.ppv_mean, "assumed_rrr": rrr, "nnt": 1 / prevented,
                     "prevented_per_10k": 1e4 * prevented, "bleeds_per_10k": 1e4 * BLEED, "bleeds_per_naion_prevented": BLEED / prevented,
                     "ppv_needed_break_even": BLEED / rrr, "ppv_shortfall_x": (BLEED / rrr) / r.ppv_mean})
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/aspirin_threshold.csv", index=False)
json.dump({"design": __doc__.split("\n\n")[0], "excess_major_bleeding": BLEED, "bleeding_source_pmid": 30667501, "assumed_rrr": RRR,
           "rows": D.round(6).to_dict("records"), "research_prototype": True, "compliance_ref": "Addendum_5_DT9"},
          open(f"{OUT}/summary.json", "w"), indent=1)
pd.set_option("display.width", 220); print(D.round(4).to_string(index=False))

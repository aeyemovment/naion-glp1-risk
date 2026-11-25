"""Readable labels for model feature names in figures. research_prototype=true. NOT for clinical use."""
import re

GROUP = {"therapy": "Therapy", "region": "Region", "source_type": "Data source", "pct_female": "Female (%)", "followup_years": "Follow-up (years)",
         "followup": "Follow-up (years)", "mean_age": "Mean age", "glp1ra_exposed": "GLP-1RA exposure", "indication": "Indication",
         "outcome_def": "Outcome definition", "exposure_drug": "Drug", "population": "Population", "comparator_cls": "Comparator",
         "design": "Design", "dataset": "Data source", "adjusted_f": "Adjusted estimate"}
LEVEL = {"dpp4i": "DPP-4 inhibitor", "dpp4": "DPP-4 inhibitor", "glp1ra_class": "GLP-1RA class", "no_glp1ra": "no GLP-1RA",
         "other_antidiabetic": "other antidiabetic", "other_antiobesity": "other anti-obesity", "sema_or_tirz": "semaglutide or tirzepatide",
         "sglt2i": "SGLT2 inhibitor", "sglt2": "SGLT2 inhibitor", "iih": "IIH", "obesity": "obesity", "t2d": "T2D", "nordic": "Nordic", "uk": "UK",
         "us": "US", "claims": "claims", "ehr_network": "EHR network", "ehr_primary_care": "primary-care EHR", "national_registry": "national registry",
         "single_center": "single center", "va_ehr": "VA EHR", "ion": "ION", "naion": "NAION", "va": "VA", "trinetx": "TriNetX", "danish": "Danish",
         "norw": "Norwegian", "swed": "Swedish", "cprd": "CPRD", "medicare": "Medicare", "marketscan": "MarketScan", "optum": "Optum",
         "korea": "Korea", "taiwan": "Taiwan", "iqvia": "IQVIA", "single": "single center", "veterans": "VA", "other": "other",
         "semaglutide": "semaglutide", "tirzepatide": "tirzepatide", "any_glp1ra": "any GLP-1RA", "t2d_or_obesity": "T2D or obesity",
         "unexposed": "unexposed", "insulin": "insulin", "metformin": "metformin", "antiobesity": "anti-obesity"}


def nice(col):
    c = str(col)
    if c in GROUP: return GROUP[c]
    miss = c.endswith("_missing")
    if miss: c = c[: -len("_missing")]
    for g in sorted(GROUP, key=len, reverse=True):
        if c == g: return f"{GROUP[g]} missing" if miss else GROUP[g]
        if c.startswith(g + "_"):
            lv = c[len(g) + 1:]
            lab = LEVEL.get(lv.lower(), re.sub(r"[_]+", " ", lv))
            return f"{GROUP[g]}: {lab}" + (" (missing)" if miss else "")
    return re.sub(r"_", " ", c).capitalize() + (" missing" if miss else "")

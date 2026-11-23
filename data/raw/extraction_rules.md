# Extraction rules (all agents)
- Source of truth: data/raw/pubmed_records.json (abstract/metadata keyed by pmid) and data/raw/fulltext/<pmid>.txt (PMC full text incl. TABLES, when present).
- You MAY use WebFetch on open pages (pubmed.ncbi.nlm.nih.gov, pmc.ncbi.nlm.nih.gov, europepmc.org, publisher OA pages) to get missing full text for your assigned PMIDs. Do not use paywalled bypasses.
- NEVER invent values. Missing -> null. Every row has "source_pmid" and "evidence_level": "fulltext" | "abstract" | "webfetch".
- Put a short verbatim-ish "evidence_quote" (<=200 chars) supporting the key numbers.
- Skip PMID 41144820 (RETRACTED) and 42468029 (retraction notice).
- Doses: record exactly as written plus normalized "dose_mg_weekly" (semaglutide SC weekly mg; oral daily mg*7 recorded separately as "oral_daily_mg"; liraglutide daily mg; tirzepatide weekly mg). Drug names normalized lowercase generic.
- Flag duplicate cohorts (same database/authors reported in letters/replies) via "duplicate_of_pmid".
- Output: valid JSON list written with the Write tool to the path given. Then reply with a <=150 word summary: rows written, PMIDs with no usable data, caveats.
- All outputs are research prototype; not for clinical use.

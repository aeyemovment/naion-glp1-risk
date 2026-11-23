# NAION risk with GLP-1 receptor agonists — machine learning on published evidence

> **Research prototype. Not for clinical, diagnostic, treatment-selection or regulatory use.**
> `research_prototype=true` · `compliance_ref=Addendum_5_DT9` · synthetic digital twins carry `synthetic_only=true`.

Code, model weights, derived data, synthetic data and manuscript for:

**Green K.** *Machine Learning on Reconstructed Published Evidence to Stratify Risk of Nonarteritic Anterior Ischemic Optic
Neuropathy with GLP-1 Receptor Agonists: Supervised, Graph, and Digital-Twin Approaches.* medRxiv preprint (not peer reviewed), 2026.

## Key results
- Within-study OR for NAION with GLP-1RA exposure 1.59 (95% CI 1.42–1.77); semaglutide 1.73; obesity indication 2.70.
- PubMed-only XGBoost AUROC 0.656 ± 0.011 (ten disjoint 10% training folds); ≈0.53 in unseen studies.
- No GNN, CNN, ensemble, digital-twin or vascular-score variant meaningfully beat PubMed-only XGBoost.

## Repository layout
| Path | Contents |
|---|---|
| `src/` | full pipeline: PubMed/PMC retrieval, extraction post-processing, pseudo-IPD reconstruction, Stages 1–16, paper builders |
| `models/` | final PubMed-only weights trained on all data: XGBoost (`.json`/`.ubj`), logistic regression (`.json`), random forest (`.joblib`), `MODEL_CARD.md` |
| `app/` | static provider calculator (client-side XGBoost; nothing leaves the browser) |
| `data/extracted/` | structured extractions — every value has source PMID + evidence quote |
| `data/processed/pseudo_ipd.csv` | count-weighted reconstructed patient-level data (11 cohorts) |
| `data/raw/` | PMIDs, reference metadata, extraction rules (full texts/abstracts not redistributed) |
| `paper/ieee/` | IEEE (IEEEtran) LaTeX source, figures, bib and PDF; `paper/preprint`, `paper/nejm_ai` PDFs + LaTeX |
| `paper/data_release/` | release README, data dictionary, checksums, real-derived CSVs, result tables |
| `results/` | per-stage summaries, tables and figures |

Synthetic digital-twin datasets (≈240 MB uncompressed) are attached to the GitHub Release as `naion_glp1_data_release_*.zip`.

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # plus torch-geometric for Stage 6
.venv/bin/python src/build_ipd.py && .venv/bin/python src/stage1_models.py   # ... stages 1b-16, see src/ docstrings
.venv/bin/python src/export_models.py   # final weights + app/model.js
.venv/bin/python src/make_papers.py && .venv/bin/python src/make_ieee.py   # needs pandoc + tectonic
```

## Use of AI tools
The analysis was carried out with Claude (Claude Opus 5.5, Anthropic) via Claude Code as an agentic research tool under the
direction of the author, who reviewed all results and takes responsibility for the content. AI tools are not authors.

## Licence
Code: MIT (`LICENSE`). Derived and synthetic data: CC BY 4.0 (`LICENSE-DATA.md`). Please also cite the original studies (PMIDs in each file).

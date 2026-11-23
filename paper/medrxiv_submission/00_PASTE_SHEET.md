# medRxiv upload — paste sheet

Submit at https://submit.medrxiv.org (log in with your existing account; New Submission).

## Files
- Main manuscript (PDF): ~/Downloads/Toward_a_Foundation_Model_for_NAION_Risk_Stratification_in_GLP_1_Receptor_Agonist_Users__Machine_Learning__Graph_Neural_Networks_and_Synthetic_Dig__1_.pdf
- Supplementary: 02_Supplementary_Workbook_S1.xlsx and 03_TRIPOD_AI_checklist.docx (this folder)

## Article type / category / licence
- Article type: New Results
- Subject area: Health Informatics (alternative: Ophthalmology)
- Licence: CC BY-NC 4.0

## Title
Toward a Foundation Model for NAION Risk Stratification in GLP-1 Receptor Agonist Users: Machine Learning, Graph Neural Networks and Synthetic Digital Twins Built from PubMed-Indexed Evidence

## Author (one author)
Kemar Earl Green, DO, FANA — corresponding author, kemarearlgreen@neuroagentai.org
1. Johns Hopkins Data Science and AI Institute, Baltimore, MD, USA
2. NeuroAgentAI, Inc., Baltimore, MD, USA

## Abstract (plain text)
Background. Observational studies and pharmacovigilance reports link glucagon-like peptide-1 receptor agonists (GLP-1RAs), particularly semaglutide, to nonarteritic anterior ischemic optic neuropathy (NAION), but no tool stratifies individual risk. Whether machine learning (ML) on published evidence, with or without synthetic digital twins, can do so is unknown. 
Methods. We extracted data from 73 of 169 PubMed publications and reconstructed count-weighted patient-level data from 11 non-overlapping cohorts (3,271,499 patients; 1,425 events). Logistic regression, random forest, XGBoost, graph neural networks and ensembles were trained on 10% and tested on 90%, across 10 disjoint 10% folds, across training fractions of 10% to 90%, and by leave-one-study-out validation. Synthetic twins were added in a factorial grid. SHAP values, permutation tests and Benjamini-Hochberg correction were used. 
Results. Within studies, GLP-1RA exposure was associated with NAION (odds ratio 1.59 (95% CI, 1.42 to 1.77)); semaglutide 1.73 (95% CI, 1.44 to 2.08); obesity indication 2.70 (95% CI, 1.64 to 4.44). XGBoost discriminated modestly (AUROC 0.656 ± 0.011; observed/expected 1.00) but fell to 0.534 in unseen studies. At the top-10% threshold, PPV was 0.087% against a prevalence of 0.044%. Neither graph or convolutional networks, ensembles nor added vascular-score features outperformed PubMed-only XGBoost; synthetic digital twins mainly regularized weaker learners, and the best hybrid XGBoost gained 0.0015 AUROC. Simulated addition of optic-disc imaging raised AUROC to about 0.82 to 0.84. 
Conclusions. Published evidence supports a reproducible exposure signal but only modest, setting-dependent risk discrimination. The score suits research enrichment, not treatment selection; patient-level prospective data are required.

## Keywords
Nonarteritic anterior ischemic optic neuropathy; GLP-1 receptor agonists; semaglutide; machine learning; synthetic data; digital twins; graph neural networks; risk prediction; explainable AI

## Ethics
This study used only data reported in published, peer-reviewed articles and publicly available pharmacovigilance summaries, together with computationally generated synthetic records. No individual-level health information was accessed and no human participants were recruited; institutional review board approval and informed consent were therefore not required.
Form answers: ethics/IRB approval not required (published aggregate data and synthetic records only; no human participants).

## Clinical trial
Not applicable (no clinical trial).

## Funding
This work received no specific funding.

## Competing interests
K. E. Green is the founder of NeuroAgent AI, Inc. and HazyEyes AI. The author declares no other competing interests.

## Data availability
Code, trained model weights, extraction files (every value with PMID and evidence quotation), reconstructed count-weighted patient-level data and synthetic digital-twin datasets are available at https://github.com/aeyemovment/naion-glp1-risk (release v1.0.0). A research-only web calculator is at https://naion-glp1-calculator.vercel.app. Copyright (c) 2026 NeuroAgent AI, Inc. All materials are released for non-commercial research use only, with attribution (citation of this article): code and model weights under a research-only, non-commercial licence included in the repository, data under CC BY-NC 4.0. Commercial use, and clinical, diagnostic or treatment-selection use, are not permitted without prior written permission (kemarearlgreen@neuroagentai.org).

## AI disclosure
This work was carried out with Claude Opus 5.5 (claude-opus-5-5; Anthropic), used through the Claude Code agentic command-line tool (November 21-23, 2026) as an agentic research tool under the direction of the author. The AI system ran the PubMed/PMC searches and extracted structured data (each value stored with its PMID and a verbatim evidence quotation); wrote and executed all analysis code (data reconstruction, model training and validation, statistics, synthetic digital-twin experiments), which generated all tables and figures; and drafted the Introduction, Methods, Results and Discussion. The author conceived the study, set the research questions, chose the validation scheme, synthetic-data design and every additional experiment, reviewed the results and manuscript, and takes full responsibility for the content. Claude is not an author because an AI system cannot take responsibility for the work (ICMJE, IEEE and medRxiv policy).

## Reporting guideline
TRIPOD+AI (checklist uploaded as a supplementary file).

## Declarations you must confirm yourself
- You are the sole author and consent to posting (a posted medRxiv preprint cannot be removed).
- The statements above are accurate.

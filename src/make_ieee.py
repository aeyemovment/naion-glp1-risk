"""IEEE (IEEEtran journal, two-column) LaTeX + PDF of the full preprint, for medRxiv.
Body text is identical to the preprint built by make_papers.py; only layout, citation style (IEEEtran.bst, numbered)
and front/back matter differ. Large result tables and secondary figures go to one-column appendices.
research_prototype=true; compliance_ref=Addendum_5_DT9. NOT for clinical use.
"""
import os, re, shutil, subprocess, sys
sys.path.insert(0, "src")
import make_papers as MP

AUTHOR_NAME = "Kemar Green"
AFFIL = "NeuroAgent AI"                                   # TODO author to confirm affiliation / address
EMAIL = "kemarearlgreen@neuroagentai.org"                 # TODO author to confirm corresponding e-mail
ORCID = ""                                                # optional
KEYWORDS = ("Nonarteritic anterior ischemic optic neuropathy, GLP-1 receptor agonists, semaglutide, machine learning, "
            "synthetic data, digital twins, graph neural networks, risk prediction, explainable AI")
ACK = ("\\section*{Acknowledgment}\n"
       "Use of AI tools: this study was carried out with Claude (model Claude Opus 5.5, Anthropic, accessed through the Claude Code agentic "
       "command-line tool) acting as an agentic research tool under the direction of the author. The AI system performed PubMed and PubMed Central "
       "retrieval, structured data extraction (every extracted value carries its source PMID and a verbatim evidence quotation), analysis code, "
       "model training and evaluation, figures and the first draft of the manuscript. The author designed the study, directed every experiment, "
       "reviewed the results and the manuscript, and takes full responsibility for the content. In line with medRxiv and ICMJE policy, AI tools "
       "are not listed as authors. Approximately 0.25 kWh of local compute and at least 764{,}000 metered language-model tokens were used (see Methods).\n\n"
       "Funding: none declared.\\,\\textsuperscript{\\dag}\\quad Competing interests: the author is the founder of NeuroAgent AI and HazyEyes AI; no relationship "
       "with manufacturers of GLP-1 receptor agonists.\\,\\textsuperscript{\\dag}\\quad Ethics: only published aggregate data and synthetic records were used; "
       "no human participants were involved and IRB approval was not required.\\,\\textsuperscript{\\dag}\n\n"
       "\\textsuperscript{\\dag}\\,To be confirmed by the author before posting.\n\n"
       "Data and code availability: extraction files (with PMID and evidence quotation per value), reconstructed count-weighted patient-level data, "
       "synthetic digital-twin datasets, model weights, analysis code and all result files are available at \\url{https://github.com/aeyemovment/naion-glp1-risk} (release v1.0.0). A research-only web calculator running the PubMed-only XGBoost model in the browser is at \\url{https://naion-glp1-calculator.vercel.app}.\n")

def build():
    md = MP.preprint()
    md = re.sub(r"\A---.*?---\s*", "", md, flags=re.S)
    for s in (MP.AUTH, MP.DISCL, MP.AI_USE): md = md.replace(s, "")
    abstract_md = md.split("## Abstract", 1)[1].split("## Introduction", 1)[0].strip()
    abstract_md = re.sub(r"\*\*(Background|Methods|Results|Conclusions)\.\*\*", r"*\1:*", abstract_md).replace("\n\n", " ")
    body = "## Introduction" + md.split("## Introduction", 1)[1]
    body = body.replace("## References", "")
    body = re.sub(r"## Data and code availability\n\n.*?(?=\n## |\Z)", "", body, flags=re.S)
    disc = MP.discussion()
    main, appx = body.split("## Complete experimental results", 1)
    appx = appx.replace(disc, "")
    appx = re.sub(r"^## ", "### ", appx, flags=re.M)                     # notes inside section K -> subsections
    appx = re.sub(r"^### [A-Z]\d?\. ", "## ", appx, flags=re.M)          # A. ... K. -> Appendix sections
    raw = lambda t: f"\n\n```{{=latex}}\n{t}\n```\n\n"
    md_ieee = (main + "\n\n" + disc + raw(ACK.replace("\\section*{Acknowledgment}", "")).replace("```{=latex}\n", "```{=latex}\n\\section*{Acknowledgment}\n", 1)
               + raw("\\clearpage\\onecolumn\\appendices") + appx + raw("\\clearpage\\twocolumn"))
    md_ieee = md_ieee.replace(raw("\\clearpage\\onecolumn\\appendices"), raw("\\clearpage\\onecolumn\n\\appendices"))
    # move the acknowledgment after the appendices (IEEE order: appendices, acknowledgment, references)
    ack_block = raw("\\section*{Acknowledgment}\n" + ACK.split("\n", 1)[1])
    md_ieee = md_ieee.replace(ack_block, "") if ack_block in md_ieee else md_ieee
    md_ieee = re.sub(r"```\{=latex\}\n\\section\*\{Acknowledgment\}.*?```", "", md_ieee, flags=re.S)
    md_ieee = md_ieee.replace(raw("\\clearpage\\twocolumn"), raw("\\clearpage\\twocolumn\n" + ACK))

    d = "paper/ieee"; os.makedirs(f"{d}/figures", exist_ok=True)
    open(f"{d}/ieee.md", "w").write(md_ieee)
    tolatex = lambda s: subprocess.run(["pandoc", "-f", "markdown", "-t", "latex"], input=s, capture_output=True, text=True, check=True).stdout.strip()
    thanks = (f"{AUTHOR_NAME} is with {AFFIL} (e-mail: {EMAIL}){'; ORCID ' + ORCID if ORCID else ''}. "
              "Research prototype: uses only published aggregate data and computationally generated synthetic records (synthetic\\_only twins; "
              "compliance reference Addendum 5, DT\\#9); not validated for clinical, diagnostic, treatment-selection or regulatory use. "
              "This preprint has not been peer reviewed.")
    meta = {"title": MP.title, "author": f"{AUTHOR_NAME}\\thanks{{{thanks}}}", "shorttitle": "Green: GLP-1 receptor agonists and NAION --- ML on published evidence",
            "abstract": tolatex(abstract_md), "keywords": KEYWORDS}
    args = ["pandoc", f"{d}/ieee.md", "-f", "markdown+pipe_tables+implicit_figures+raw_attribute", "-t", "latex", "-s", "--natbib",
            "--shift-heading-level-by=-1", "--template", "paper/style/ieee_template.tex", "--lua-filter", "paper/style/ieee_floats.lua",
            "--resource-path", f".:{MP.ROOT}", "-o", f"{d}/naion_glp1_ieee.tex"]
    for k, v in meta.items(): args += ["-V", f"{k}={v}"]
    subprocess.run(args, check=True)
    t = open(f"{d}/naion_glp1_ieee.tex").read()
    t = re.sub(r"\\cite[pt]\{", r"\\cite{", t)
    for pth in set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)):
        src = pth if os.path.isabs(pth) else os.path.join(MP.ROOT, pth)
        if os.path.exists(src):
            dst = f"figures/{os.path.basename(src)}"; shutil.copy(src, f"{d}/{dst}"); t = t.replace("{" + pth + "}", "{" + dst + "}")
    open(f"{d}/naion_glp1_ieee.tex", "w").write(t)
    MP.bibtex(); shutil.copy(f"{MP.P}/references.bib", f"{d}/references.bib")
    r = subprocess.run(["tectonic", "-X", "compile", "naion_glp1_ieee.tex"], cwd=d, capture_output=True, text=True)
    print(r.stderr[-4000:] if r.returncode else "compiled")
    miss = sorted(set(re.findall(r"Missing character: There is no (\S+)", r.stderr)))
    print("missing glyphs:", miss)
    return d

if __name__ == "__main__":
    build()

"""Download PMC full text (body text) for records with PMCID; save as plain text."""
import json, time, urllib.request, urllib.parse, os
from lxml import etree
E="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
os.makedirs("data/raw/fulltext",exist_ok=True)
recs=json.load(open("data/raw/pubmed_records.json"))
ok=0
for r in recs:
    if not r["pmcid"]: continue
    out=f"data/raw/fulltext/{r['pmid']}.txt"
    if os.path.exists(out): ok+=1; continue
    try:
        x=etree.fromstring(urllib.request.urlopen(E+"efetch.fcgi?"+urllib.parse.urlencode({"db":"pmc","id":r["pmcid"].replace("PMC",""),"retmode":"xml"}),timeout=60).read())
        body=x.find(".//body")
        tabs=x.findall(".//table-wrap")
        txt=("".join(body.itertext()) if body is not None else "")
        txt+="\n\nTABLES:\n"+"\n---\n".join(" | ".join(t.itertext()) for t in tabs)
        if len(txt)>500: open(out,"w").write(txt); ok+=1
    except Exception as e: print(r["pmid"],e)
    time.sleep(0.4)
print("fulltexts:",ok)

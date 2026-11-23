"""Fetch PubMed records (abstract + metadata) and PMC OA full text where available."""
import json, time, urllib.request, urllib.parse
from lxml import etree
E="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ids=open("data/raw/pmids.txt").read().split()
recs=[]
for i in range(0,len(ids),100):
    q=urllib.parse.urlencode({"db":"pubmed","id":",".join(ids[i:i+100]),"retmode":"xml"})
    x=etree.fromstring(urllib.request.urlopen(E+"efetch.fcgi?"+q).read())
    for a in x.findall(".//PubmedArticle"):
        g=lambda p:" ".join("".join(e.itertext()) for e in a.findall(p))
        pmc=[e.text for e in a.findall(".//ArticleId") if e.get("IdType")=="pmc"]
        recs.append({"pmid":g(".//MedlineCitation/PMID"),"title":g(".//ArticleTitle"),
          "journal":g(".//Journal/Title"),"year":g(".//PubDate/Year") or g(".//PubDate/MedlineDate")[:4],
          "pubtypes":[e.text for e in a.findall(".//PublicationType")],
          "abstract":g(".//Abstract/AbstractText"),"pmcid":pmc[0] if pmc else None})
    time.sleep(0.4)
json.dump(recs,open("data/raw/pubmed_records.json","w"),indent=1)
print(len(recs),"records;",sum(1 for r in recs if r["pmcid"]),"with PMCID")

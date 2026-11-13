import pandas as pd, re, docx
R=r"results\dose_response\\"
rc=pd.read_csv(R+"responsiveness_corrected.csv")
print(rc.columns.tolist())
print(rc.comparison.unique() if 'comparison' in rc else '')
bs=pd.read_csv(R+"behaviour_summary.csv"); print(bs.columns.tolist())
m=pd.read_csv(R+"models.csv"); print(len(m), m.columns.tolist()[:30])
d=docx.Document(r"paper\paper_draft.docx")
ps=[p.text for p in d.paragraphs]
ab=[p for p in ps if p.startswith("Abstract")][0]
print("abstract words", len(ab.split())-1)
i=ps.index([p for p in ps if p.startswith("Highlights")][0])
for h in ps[i+1:i+6]: print(len(h), h)

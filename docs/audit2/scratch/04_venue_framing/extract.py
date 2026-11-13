import docx,sys
d=docx.Document(r"paper\paper_draft.docx")
out=[]
from docx.oxml.ns import qn
body=d.element.body
for el in body.iterchildren():
    if el.tag==qn('w:p'):
        p=docx.text.paragraph.Paragraph(el,d)
        st=p.style.name if p.style is not None else ''
        out.append(f"[{st}] {p.text}")
    elif el.tag==qn('w:tbl'):
        t=docx.table.Table(el,d)
        out.append("<TABLE>")
        for r in t.rows:
            out.append(" | ".join(c.text.replace("\n"," / ") for c in r.cells))
        out.append("</TABLE>")
open(r"docs\audit2\scratch\04_venue_framing\paper_text.txt","w",encoding="utf-8").write("\n".join(out))
print(len(out))

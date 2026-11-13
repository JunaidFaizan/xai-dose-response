t=open(r"docs\audit2\scratch\04_venue_framing\paper_text.txt",encoding="utf-8").read()
lines=t.split("\n")
i=[k for k,l in enumerate(lines) if "1. Introduction" in l][0]
j=[k for k,l in enumerate(lines) if l.startswith("[Heading 1] CRediT")][0]
a=[k for k,l in enumerate(lines) if "Appendix A." in l and l.startswith("[Heading 1]")][0]
r=[k for k,l in enumerate(lines) if l.startswith("[Heading 1] References")][0]
w=lambda L: sum(len(x.split()) for x in L)
intab=False; body=[];tab=0
for l in lines[i:j]:
    if l=="<TABLE>": intab=True; continue
    if l=="</TABLE>": intab=False; continue
    if intab: tab+=len(l.split())
    else: body.append(l)
print("main prose words",w(body),"main table words",tab,"appendix words",w(lines[a:r]),"refs",r, len(lines)-r)

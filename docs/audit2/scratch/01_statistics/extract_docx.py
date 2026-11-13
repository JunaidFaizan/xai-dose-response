import docx, pathlib
d = docx.Document("paper/paper_draft.docx")
out = []
for block in d.element.body.iterchildren():
    tag = block.tag.split('}')[1]
    if tag == 'p':
        t = ''.join(n.text or '' for n in block.iter() if n.tag.endswith('}t'))
        if t.strip(): out.append(t)
    elif tag == 'tbl':
        t = docx.table.Table(block, d)
        out.append('[TABLE]')
        for r in t.rows:
            out.append(' | '.join(c.text for c in r.cells))
        out.append('[/TABLE]')
pathlib.Path("docs/audit2/scratch/01_statistics/paper_text.txt").write_text('\n'.join(out), encoding='utf8')
print(len(out))

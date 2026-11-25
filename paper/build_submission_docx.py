"""Writes the Word files of the journal submission that are not the manuscript (T6.1 conformance):
docs/submission/highlights.docx (separate editable file, bullets from paper/highlights_submission.txt),
docs/submission/declaration_of_competing_interests.docx (convenience draft; the final file must come from the
journal's declarations tool) and docs/submission/author_biography_TEMPLATE.docx. Run after build_paper.py --submission:
    .venv/Scripts/python.exe paper/build_submission_docx.py
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt

PAPER = Path(__file__).resolve().parent
OUT = PAPER.parent / "docs" / "submission"
TITLE = ("A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut "
         "Reliance in CNN Image Classifiers")


def new_document():
    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"
    d.styles["Normal"].font.size = Pt(12)
    return d


def main():
    highlights = (PAPER / "highlights_submission.txt").read_text(encoding="utf8").splitlines()
    assert 3 <= len(highlights) <= 5 and all(len(h) <= 85 for h in highlights)
    d = new_document()
    d.add_heading("Highlights", level=1)
    d.add_paragraph(TITLE)
    for h in highlights:
        d.add_paragraph(h, style="List Bullet")
    d.save(OUT / "highlights.docx")

    d = new_document()
    d.add_heading("Declaration of interests", level=1)
    d.add_paragraph(TITLE)
    d.add_paragraph("Author: Md Junaid Faizan Shaikh")
    d.add_paragraph("Convenience draft. The file that is uploaded at submission must be generated with the journal's "
                    "declarations tool (Elsevier); copy the author's choice below into it.").runs[0].italic = True
    d.add_paragraph("\u2612 The author declares that he has no known competing financial interests or personal "
                    "relationships that could have appeared to influence the work reported in this paper.")
    d.add_paragraph("\u2610 The author declares the following financial interests/personal relationships which may be "
                    "considered as potential competing interests:")
    d.save(OUT / "declaration_of_competing_interests.docx")

    d = new_document()
    d.add_heading("Author biography (Vitae)", level=1)
    d.add_paragraph("Template. Elsevier asks for a biography of at most 100 words for each author, and a "
                    "passport-type photograph of the author as a separate file (figure) placed next to it. "
                    "Keep the text in the third person.").runs[0].italic = True
    d.add_paragraph("Md Junaid Faizan Shaikh is an M.Tech student in the Department of Computer Science and "
                    "Engineering at Khaja Bandanawaz University. <degree year, research interests, previous positions "
                    "or publications, at most 100 words in total>")
    d.add_paragraph("Photograph: <attach passport-type photograph as a separate file>")
    d.save(OUT / "author_biography_TEMPLATE.docx")
    print("wrote highlights.docx, declaration_of_competing_interests.docx, author_biography_TEMPLATE.docx")


if __name__ == "__main__":
    main()

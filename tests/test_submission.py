"""Format checks of the journal-submission build (T6.1/T6.2).

Builds paper/paper_submission.docx and paper/supplementary_submission.docx with
`paper/build_paper.py --submission` (about half a minute) and checks them. The build itself also asserts
the title, the forbidden words, the cross-references and the declared markers; these tests repeat the
important ones from the outside and check the files in docs/submission.
"""

import hashlib
import re
import subprocess
import sys
import unittest
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SUBMISSION_DOCS = ROOT / "docs" / "submission"
TITLE = ("A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut "
         "Reliance in CNN Image Classifiers")
MARKER = re.compile(r"\[\[[^\]]+\]\]")


def text_of(path):
    d = Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for r in t.rows:
            parts.extend(c.text for c in r.cells)
    return "\n".join(parts)


def references_of(path):
    return [p.text for p in Document(path).paragraphs if re.match(r"\[\d+\] ", p.text)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SubmissionBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run = subprocess.run([sys.executable, str(PAPER / "build_paper.py"), "--submission"], cwd=ROOT,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout[-1500:] + run.stderr[-1500:]
        cls.main = text_of(PAPER / "paper_submission.docx")
        cls.supp = text_of(PAPER / "supplementary_submission.docx")

    def test_title_exact_and_no_internal_references(self):
        for text in (self.main, self.supp):
            self.assertIn(TITLE, text)
            for pattern in (r"audit", r"reviewer", r"(?<!original )(?<!help )\bdrafts?\b", r"docs/", r"earlier draft"):
                self.assertIsNone(re.search(pattern, text, flags=re.I), pattern)

    def test_only_declared_markers_and_all_listed_in_deferred(self):
        found = set(MARKER.findall(self.main + self.supp))
        source = (PAPER / "build_paper.py").read_text(encoding="utf8")
        declared = set(re.findall(r'"(\[\[[^\]]+\]\])"', source[source.index("DECLARED_MARKERS"):]))
        self.assertTrue(found <= declared, found - declared)
        deferred = (SUBMISSION_DOCS / "DEFERRED.md").read_text(encoding="utf8")
        for marker in declared:
            self.assertIn(marker, deferred, marker)
        # T5.1 replaced every SD3/SD4 marker; only author and Zenodo-release markers remain
        self.assertFalse([m for m in found if m.startswith("[[SD")], found)
        self.assertTrue(all(m.startswith("[[AUTHOR:") or m == "[[ZENODO_RELEASE_DOI]]" for m in found), found)

    def test_final_fails_while_markers_remain_and_writes_nothing(self):
        before = (sha(PAPER / "paper_submission.docx"), sha(PAPER / "supplementary_submission.docx"))
        run = subprocess.run([sys.executable, str(PAPER / "build_paper.py"), "--final"], cwd=ROOT,
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("unfinished markers", run.stderr + run.stdout)
        after = (sha(PAPER / "paper_submission.docx"), sha(PAPER / "supplementary_submission.docx"))
        self.assertEqual(before, after)

    def test_tables_moved_to_supplement_and_cross_references_resolve(self):
        main_tables = [int(n) for n in re.findall(r"^Table (\d+)\. ", self.main, flags=re.M)]
        self.assertEqual(main_tables, list(range(1, len(main_tables) + 1)))
        supp_tables = [int(n) for n in re.findall(r"^Table S(\d+)\. ", self.supp, flags=re.M)]
        self.assertEqual(supp_tables, [1, 2, 3, 4, 5, 6, 7])
        plain = re.sub(r"Supplementary Table S\d+", "", self.main)
        for n in {int(x) for x in re.findall(r"\bTable (\d+)\b", plain)}:
            self.assertLessEqual(n, len(main_tables))
        for n in {int(x) for x in re.findall(r"Supplementary Table S(\d+)", self.main)}:
            self.assertIn(n, supp_tables)
        # the abridged pre-registration table has about ten rows
        d = Document(PAPER / "paper_submission.docx")
        prereg = next(t for t in d.tables if t.rows[0].cells[0].text == "Pred.")
        self.assertTrue(10 <= len(prereg.rows) - 1 <= 12)
        self.assertNotIn("Appendix B", self.main)
        self.assertNotIn("Appendix C", self.main)

    def test_reference_list_extends_the_draft_list(self):
        """Every draft reference is still listed, in the same relative order. The submission adds the datasets,
        software and code release the guide asks for, and completes three entries (MedMNIST, Pet, Captum)."""
        sub = [re.sub(r"^\[\d+\] ", "", r) for r in references_of(PAPER / "paper_submission.docx")]
        draft = [re.sub(r"^\[\d+\] ", "", r) for r in references_of(PAPER / "paper_draft.docx")]
        completed = ("Yang J, et al. (2023) MedMNIST", "Parkhi OM", "Kokhlikyan N")
        position = -1
        for entry in draft:
            key = entry[:30] if entry.startswith(completed) else entry
            hits = [i for i, s in enumerate(sub) if s.startswith(key)]
            self.assertTrue(hits, entry[:60])
            self.assertGreater(hits[0], position, entry[:60])
            position = hits[0]
        for needle in ("LeCun Y", "Krizhevsky A", "Paszke A", "TorchVision maintainers", "road_evaluation",
                       "Data and code for"):
            self.assertTrue(any(needle in s for s in sub), needle)

    def test_guide_formatting(self):
        """Elsevier guide: horizontal table rules only, no shading, British spelling, keywords, figure files."""
        d = Document(PAPER / "paper_submission.docx")
        xml = d.element.xml
        for tag in ("<w:shd", "<w:insideV", "<w:left ", "<w:right ", "TableGrid"):
            self.assertNotIn(tag, xml, tag)
        keywords = self.main.split("Keywords ", 1)[1].split("\n", 1)[0].split(" \u00b7 ")
        self.assertTrue(1 <= len(keywords) <= 7, keywords)
        self.assertFalse([k for k in keywords if re.search(r"\b(and|of)\b", k)], keywords)
        self.assertIn("Perturbation metrics", keywords)
        body = self.main.split("\nReferences\n")[0]
        self.assertIsNone(re.search(r"behavior|grayscale|normaliz|analyz|\bcenter|labeled", body))
        self.assertIn("Declaration of generative AI and AI-assisted technologies in the manuscript preparation "
                      "process", self.main)
        self.assertRegex(self.main, r"(?m)^Table A\.1\. ")
        self.assertRegex(self.main, r"(?m)^Fig\. A\.1\. ")
        head = self.main[:1500]
        self.assertIn("Rauza-i Buzurg, Kalaburagi, Karnataka 585104, India", head)
        self.assertEqual(self.main.count("junaid.faizan.codes@gmail.com"), 1)
        self.assertIn("junaid.faizan.codes@gmail.com", head)
        self.assertNotIn("Supervision", self.main)
        self.assertIn("Sole author.", self.main)
        positions = [self.main.rfind("\n" + h + "\n") for h in ("Acknowledgements", "Declaration of generative AI and "
                     "AI-assisted technologies in the manuscript preparation process")]
        self.assertTrue(0 < positions[0] < positions[1] < self.main.rfind("\nReferences\n"), positions)
        self.assertIn("The author thanks Dr. Raafiya Gulmaher, Assistant Professor, Department of Computer Science "
                      "and Engineering, Khaja Bandanawaz University, for serving as the guide of the author's M.Tech "
                      "project.", self.main)

    def test_figure_files(self):
        from PIL import Image
        folder = PAPER / "figures_submission"
        labels = [str(n) for n in range(1, 8)] + ["A1"]
        for label in labels:
            png, pdf = folder / f"Figure_{label}.png", folder / f"Figure_{label}.pdf"
            self.assertTrue(png.exists() and pdf.exists(), label)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"), label)
        for label in ("1", "2", "4", "5", "6"):  # regenerated from the CSVs: vector PDF and >= 2244 px
            self.assertGreaterEqual(Image.open(folder / f"Figure_{label}.png").width, 2244, label)
    def test_preregistration_dois_in_main_text(self):
        for doi in ("10.5281/zenodo.24285532", "10.5281/zenodo.24333540", "10.5281/zenodo.24416017",
                    "10.5281/zenodo.24495658"):
            self.assertIn(doi, self.main)
        self.assertIn("[[ZENODO_RELEASE_DOI]]", self.main)

    def test_highlights(self):
        generated = (PAPER / "highlights_submission.txt").read_text(encoding="utf8").strip().split("\n")
        self.assertTrue(3 <= len(generated) <= 5)
        self.assertTrue(all(len(h) <= 85 for h in generated))
        doc = (SUBMISSION_DOCS / "highlights.md").read_text(encoding="utf8")
        listed = [l[2:] for l in doc.split("\n") if l.startswith("- ")]
        self.assertEqual(listed, generated)

    def test_abstract_within_250_words(self):
        abstract = self.main.split("Abstract ", 1)[1].split("\nKeywords", 1)[0]
        self.assertLessEqual(len(abstract.replace("\u2013", " ").split()), 250)


class SubmissionDocs(unittest.TestCase):
    def test_files_exist_and_carry_the_official_title(self):
        for name in ("cover_letter.md", "highlights.md", "declarations.md", "suggested_reviewers.md",
                     "submission_checklist.md", "DEFERRED.md"):
            text = (SUBMISSION_DOCS / name).read_text(encoding="utf8")
            self.assertIn(TITLE, text, name)

    def test_no_earlier_title_in_project_documents(self):
        old = re.compile(r"Do XAI Metrics Notice|Seminar Guide: Do Explanation Metrics Catch|"
                         r"A Ground-Truth Evaluation of XAI Metrics for CNN Image Classifiers")
        for path in [ROOT / "README.md", ROOT / "SYNOPSIS.md", ROOT / "docs" / "SEMINAR_GUIDE.md",
                     ROOT / "docs" / "ROADMAP_Q1.md", *SUBMISSION_DOCS.glob("*.md")]:
            if path.exists():
                self.assertIsNone(old.search(path.read_text(encoding="utf8")), path.name)

    def test_cover_letter_names_all_zenodo_versions(self):
        text = (SUBMISSION_DOCS / "cover_letter.md").read_text(encoding="utf8")
        for doi in ("10.5281/zenodo.24285532", "10.5281/zenodo.24333540", "10.5281/zenodo.24416017",
                    "10.5281/zenodo.24495658"):
            self.assertIn(doi, text)


if __name__ == "__main__":
    unittest.main()

# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Submission checklist

Tick items when done. Items with a row number are in `DEFERRED.md`.

**Before submission (author)**

- [ ] Browser check of the Neurocomputing guide for authors: fee route, abstract limit, highlights, length, reference style (D12)
- [x] Kaggle S3 and S4 finished, SD3 and SD4 integrated in the paper (D01 to D10; T5.1, 2026-11-25)
- [x] Zenodo version 4 (SD4 completion note) uploaded (D05; 10.5281/zenodo.24495658)
- [ ] Zenodo release of code, results and checkpoints, DOI in the paper (D13)
- [ ] E-mail, funding, competing interests, acknowledgements, AI-use statement filled in the paper and in `docs/submission/*.md` (D14 to D18)
- [ ] Suggested reviewers: names, affiliations, e-mails, conflicts checked (D27)
- [ ] GitHub repository public, no `data/`, `.pt`, zips or `.agents/` in it (D19)

**Build (agent)**

- [ ] `.venv\Scripts\python.exe paper\build_paper.py --submission` passes its checks
- [ ] `.venv\Scripts\python.exe paper\build_paper.py --final` passes (no marker left in the paper or in `docs/submission/*.md`)
- [ ] `.venv\Scripts\python.exe -m unittest discover -s tests -t .` passes
- [ ] Title is exactly "A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers" in every file
- [ ] `docs/submission/highlights.md` equals `paper/highlights_submission.txt`

**Files to upload**

- [ ] `paper/paper_submission.docx` (manuscript, with references and declarations)
- [ ] `paper/supplementary_submission.docx` (supplementary material)
- [ ] Highlights file (3 to 5 bullets, each at most 85 characters), from `docs/submission/highlights.md`
- [ ] Cover letter from `docs/submission/cover_letter.md`
- [ ] Figures: `paper/figures/*_submission.png` and the others used in the paper, at 200 dpi (check the journal's resolution rule, D12)
- [ ] Declaration of competing interests form if the journal asks for one (D16)

**Final read-through (author, T6.5)**

- [ ] Every number checked against the CSVs after SD3 and SD4 (D25)
- [ ] Word or PDF page count within the journal limit (D22, D23)

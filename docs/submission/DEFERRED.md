# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Deferred and skipped items for the journal submission (T6.1 / T6.2)

Created with the M7 submission build (2026-11-23). This file lists everything that the
offline submission build did not do or could not do yet, where each item plugs in, who does it,
and what blocks it. Update the status column when an item is done. `build_paper.py --final`
fails while any `[[marker]]` below is still in `paper/` or in `docs/submission/*.md`
(this file and `submission_checklist.md` are not scanned).

Build commands (project root, `.venv\Scripts\python.exe`):

- `paper\build_paper.py` builds the unchanged guide-review draft, `paper/paper_draft.docx`.
- `paper\build_paper.py --submission` builds `paper/paper_submission.docx`, the figure files `paper/figures_submission/Figure_N.pdf` and `.png`,
  `paper/supplementary_submission.docx` and `paper/highlights_submission.txt`. It checks the
  title, the forbidden words, the cross-references and that only the declared markers remain.
- `paper\build_paper.py --final` runs the same build and fails if any marker remains. It
  writes no file when it fails.

## 1. Markers

Every marker the submission build may contain. The list is `DECLARED_MARKERS` in
`paper/build_paper.py`; a marker that is not declared there fails the build. The eight SD3/SD4
markers were replaced in T5.1 (0b8b723) and are no longer declared; they are kept in the table,
without brackets, for the record.

| Marker | Where | Replaced by | Row |
|---|---|---|---|
| `SD3_METHODS` (replaced in 0b8b723) | main text, Section 3.9 | final SD3 method details (timing-gate outcome, image counts actually used) | D03 |
| `SD4_METHODS` (replaced in 0b8b723) | main text, Section 3.9 | final SD4 doses, pilot outcome, mark description | D05 |
| `SD3_RESULTS` (replaced in 0b8b723) | main text, Section 5.11 | SD3 results, tables, verdicts on P10 and P11 | D01, D02 |
| `SD4_RESULTS` (replaced in 0b8b723) | main text, Section 5.12 | SD4 results, tables, verdicts on P12 and P13 | D04 |
| `SD3_POSITIONING` (replaced in 0b8b723) | note of Table 1 | one sentence on exact ROAD results in the positioning | D09 |
| `SD3_SD4_LIMITATIONS` (replaced in 0b8b723) | Section 7, last bullet | limitations updated with what SD3 and SD4 showed (delete the "pending" bullet) | D07 |
| `SD3_SD4_CONCLUSION` (replaced in 0b8b723) | end of Section 8 | one or two sentences if a pre-registered prediction is confirmed or refuted | D08 |
| `SD3_SD4_SUPPLEMENT` (replaced in 0b8b723) | end of the supplementary document | ROAD and SD4 supplementary tables | D10 |
| `[[AUTHOR: e-mail]]` | front matter | corresponding author's e-mail | D14 |
| `[[AUTHOR: postal address]]` | title page | full postal address of the affiliation, with the country | D32 |
| `[[AUTHOR: competing interests]]` | Declaration of competing interest | the author's own statement | D16 |
| `[[AUTHOR: funding statement]]` | Funding | the author's own statement | D15 |
| `[[AUTHOR: acknowledgements]]` | Acknowledgements | acknowledgements, or delete the section | D18 |
| `[[AUTHOR: confirm AI-use statement per Elsevier policy]]` | AI declaration | delete the marker after confirming tool names, purpose and oversight | D17 |
| `[[ZENODO_RELEASE_DOI]]` | Data availability | DOI of the Zenodo release of code, results and checkpoints | D13 |

Markers that only appear in `docs/submission/*.md` (cover letter, declarations, reviewers) are
listed in the rows D14 to D19 and D29. They are also scanned by `--final`.

## 2. Ledger

Status: `todo`, `blocked (reason)`, `done (commit or date)`.

| ID | What is missing | Why | Plugs in at | Who | Blocked by | Status |
|---|---|---|---|---|---|---|
| D01 | SD3 results section and tables: ROAD MoRF and LeRF preference, over-detection, switch and ρ for all comparisons, per setting; verdicts on P10 and P11 | SD3 scores on the GPU models do not exist yet. The laptop ROAD scores for 206 models exist in `per_model/*_road.csv` but are not analysed | `[[SD3_RESULTS]]`, Section 5.11, new tables | Agent (T5.1) | Kaggle S4 (T4.4); ROAD analysis code in `analyze` | done (0b8b723; data b43fbf9): Section 5.11, Table 14, Supplementary Table S6; P10 holds, P11 holds |
| D02 | P10 paired image-bootstrap analysis (difference Deletion (black) − ROAD in mean preference(R over O) at the lowest-dose models, 2000 replicates, seed 0, same images) | The analysis is specified in `docs/PREDICTIONS_sd3_sd4.md` but not coded | `SD3_RESULTS` (replaced in 0b8b723) | Agent (T5.1) | D01 data; code not written | done (05055ab): `road_p10` in `xai/dose_analysis.py`, `road_p10.csv`, 8 unit tests in `tests/test_road_predictions.py` |
| D03 | Final SD3 method text: whether the Pet timing gate reduced Pet to 25 images, the exact image counts used, the licence line for the ported ROAD operator | The timing gate is settled (note A3-C1, 2026-11-24): ROAD on 50 images took 2.51 min (`pet_resnet18`) and 2.28 min (`pet_resnet18_pretrained`) on a Kaggle T4 and 5.4-5.6 min on the laptop CPU, far below the 15-minute gate, so Pet-sized settings keep 50 images (`ROAD_PET_IMAGES` = 50). Still missing: the final image counts per scored setting after S4 and the licence line | `[[SD3_METHODS]]`, Section 3.9 | Agent after S4 (T5.1) | T4.4 (final counts) | done (0b8b723): Section 3.9 gives the operator, the MIT licence line, the image counts (100 or 50, all 293 models) and the timing-gate outcome |
| D04 | SD4 results section: P12 (ρ > 0.8 for at least three metrics) and P13 (black and blur over-detection > 0.5), behaviour, SD4 against SD1 | SD4 models do not exist yet | `[[SD4_RESULTS]]`, Section 5.12, new tables | Agent (T5.1) | Kaggle S3 and S4 (T4.2, T4.4) | done (0b8b723): Section 5.12, Table 15, Supplementary Table S7; P12 holds, P13 refuted |
| D05 | SD4 pilot completion note (final doses by the A3 dose rule), Zenodo version 4 DOI, and the final SD4 method text | Pilot done (Kaggle S3, 2026-11-24) and the note A3-C1 is written (`docs/PREDICTIONS_amendments.md`): dose-0 pilot r_c 0.84, +0.25 pilot r_c 97.02, rule "below 60", stopping rule not triggered, final doses -0.5, -0.2, 0, +0.1, +0.25, +0.4 (18 models, seeds 42, 1, 2). Zenodo version 4 is published (10.5281/zenodo.24495658, 2026-11-24 10:07 UTC) and recorded in the note. Still missing: add version 4 to the Pre-registration paragraph in Section 4 of the paper, and the final SD4 method text | `[[SD4_METHODS]]`; add version 4 to the Pre-registration paragraph in Section 4 | Agent (T5.1) | Kaggle S4 | done (0b8b723): Zenodo version 4 in the Pre-registration paragraph; SD4 method text in Section 3.9 |
| D06 | Abstract, highlights and cover letter changes | The abstract must change only if a pre-registered SD3 or SD4 prediction is confirmed or refuted (roadmap T5.1). The abstract is at the 250-word mark, so an addition needs a cut | Abstract in `build_paper.py`; `docs/submission/highlights.md` is regenerated from `paper/highlights_submission.txt` | Agent (T5.1) | D01, D04 | done (0b8b723): abstract adds the A3 outcome (250 words; cut: the edge note, the label range, two words); highlights 1, 2 and 5 changed; `highlights.md` regenerated |
| D07 | Limitations update (replace the "pending" bullet) | Depends on SD3 and SD4 outcomes | `[[SD3_SD4_LIMITATIONS]]`, Section 7 | Agent (T5.1) | D01, D04 | done (0b8b723): two limitation bullets replace the pending one |
| D08 | Conclusion update | Same | `[[SD3_SD4_CONCLUSION]]`, Section 8 | Agent (T5.1) | D01, D04 | done (0b8b723): one conclusion sentence (three of four held, P13 refuted) |
| D09 | ROAD in the positioning (Table 1) and in Section 2.3 | Exact ROAD is described as a pre-registered extension in Section 3.9, but Table 1 has no exact-ROAD result | `SD3_POSITIONING` (replaced in 0b8b723) | Agent (T5.1) | D01 | done (0b8b723): one sentence in the note of Table 1; Section 2.3 unchanged (it already describes the ROAD mechanism) |
| D10 | Supplementary tables for ROAD (all comparisons) and SD4 | No data | `[[SD3_SD4_SUPPLEMENT]]` in `supplementary_submission.docx` | Agent (T5.1) | D01, D04 | done (0b8b723): Tables S6 and S7 |
| D11 | Multiplicity caveat for the new tables: say which Benjamini–Hochberg and Holm families the ROAD and SD4 tests belong to | The families in Section 3.7 were fixed for the existing analyses; new tests need their own stated families | Section 3.7 and the notes of the new tables | Agent (T5.1) | D01, D04 | done (0b8b723): Section 3.7 (SD4 in the families of the all-model analysis; ROAD families; P10-P13 rules unadjusted) and the note of Table S6 |
| D12 | Neurocomputing guide-for-authors check | Done 2026-11-25 from the guide text read by the parent. Findings: abstract at most 250 words; highlights 3 to 5 bullets of at most 85 characters including spaces, as a separate editable file with "highlights" in its name; no page limit stated; no graphical abstract required; reference style is free at submission; research-data deposit (Option C) is mandatory, or a statement why not; single-anonymized review; vitae required (at most 100 words and a passport-type photograph); competing-interest declaration as a Word file from the journal's tool; a personal e-mail address is not prohibited; open access and its APC are chosen only at the publishing agreement after acceptance | Whole paper | Author (T6.4) | – | done (2026-11-25; conformance work in the T6.1 conformance commit) |
| D13 | Zenodo release of code, per-model results, analysis outputs and checkpoints, and its DOI | T6.3 comes after T5.1 | `[[ZENODO_RELEASE_DOI]]`, Data availability; `declarations.md`, `cover_letter.md` | Author (T6.3) | D01, D04 | todo (unblocked: T5.1 done) |
| D14 | Corresponding author's e-mail | Unknown to the agent | `[[AUTHOR: e-mail]]`, front matter; `cover_letter.md` | Author | none | todo |
| D15 | Funding statement | Unknown. The draft's text ("did not receive any specific grant") was not carried over because it is a claim only the author can make | `[[AUTHOR: funding statement]]` | Author | none | todo |
| D16 | Competing-interests statement | Same | `[[AUTHOR: competing interests]]`; `cover_letter.md` | Author | none | todo |
| D17 | AI-use statement: name of each tool, what it was used for and the author's oversight (Elsevier asks for all three) | The draft says only "AI assistants". Policy: https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals (read 2026-11-23). The section is placed immediately before the references, with the title Elsevier suggests | `[[AUTHOR: confirm AI-use statement per Elsevier policy]]` | Author | none | todo |
| D18 | Guide's role and acknowledgements. The draft listed the guide for "Supervision" in CRediT; the submission is sole-author as the roadmap states | Whether the guide is acknowledged, or is a co-author, is the author's decision | `[[AUTHOR: acknowledgements]]` | Author | none | todo |
| D19 | Make the GitHub repository public (`https://github.com/JunaidFaizan/xai-dose-response`) and check it for `data/`, `.pt`, zips and `.agents/` | T6.5 | `declarations.md` code availability | Author (T6.5) | D13 | todo |
| D20 | N15 items not done: (a) leading the abstract with the CIFAR ResNet-18 inpainting and insertion switches (8.6–8.7%, bracketed) was only "consider"; (b) panel titles in Figs. 3 and 7 are the raw setting identifiers (`pet_resnet18`, `cifar10`); the caption explains them; (c) in the reordered Fig. 3 the legend sits over the third panel (it was in the first panel of the original figure) | (b) and (c) need the analysis figures to be regenerated from the CSVs, which `dose_response.py` and `xai/` rules forbid changing here | Figs. 3 and 7, abstract | Agent or author | decision | todo |
| D21 | Reference list: the style question is resolved (free at submission, D12). The N13 content fixes (LATEC track, venues and page numbers, full author lists for four entries, "no authors" entries) are still not done; the T6.1 conformance step only added the dataset and software references (D31) | M7 left the draft list unchanged; the conformance step extends it | References | Agent (optional) | – | done for the style question (2026-11-25); N13 content fixes todo (optional) |
| D22 | Exact page count of the main text and the supplement | The guide states no page limit (D12). No Word or LibreOffice on the build machine; the count remains the author's check in Word | `paper_submission.docx` | Author | Word | todo (no limit stated) |
| D23 | Optional length cuts and sparing use of tables (Elsevier: avoid duplicating results). Candidates: Tables 7, 8, 10 and 13 of the main text (behaviour, labels, confound, P9 switches) | The guide states no page limit (D12); nothing was deleted | main-text tables | Agent or author | decision | todo (optional) |
| D24 | `elsarticle` LaTeX version | Optional; out of scope in the roadmap | – | Agent | decision | todo (optional) |
| D25 | Final number check of every number in the paper against the CSVs after T5.1, then `build_paper.py --final` | The build asserts cover most numbers, but the SD3 and SD4 text does not exist yet | whole paper | Agent (T5.1), author (T6.5) | D01, D04 | agent part done (0b8b723: guard asserts for every new SD3/SD4 number and verdict); the author's final number check (T6.5) is open |
| D26 | Cover letter: sentences on SD3 and SD4 results; reviewer and editor details; article type | See `cover_letter.md` markers | `cover_letter.md` | Author and agent | D01, D04 | SD3/SD4 sentences and Zenodo version 4 done (0b8b723); article type (Regular article) and the factual journal-fit sentence done in T6.1 conformance; open: date, e-mail and editor details (author) |
| D27 | Suggested reviewers: institutional e-mail addresses and affiliations, and a check for conflicts of interest | Not known to the agent; none was invented | `suggested_reviewers.md` | Author | none | todo |
| D28 | The pilot table text in Section 4 describes the SD2 pilot as "one of the SD2 models, not scored yet" (from `pilots.csv`) although it is scored and pooled | Found while building; wording comes from the `PILOTS` text in `build_paper.py` and also appears in the draft | Section 4, Amendment A2 paragraph | Agent | decision | done (0b8b723): the branch is removed; `pilots.csv` holds only the SD1 and SD4 pilots (asserted) |
| D29 | Markers in `docs/submission/*.md` | See the files | `cover_letter.md`, `declarations.md`, `suggested_reviewers.md` | Author | none | todo |
| D30 | Rebuild and re-check after any change: `--submission`, then `--final`, then regenerate `docs/submission/highlights.md` from `paper/highlights_submission.txt` if the abstract changed | Highlights are generated from the same numbers as the abstract | – | Agent | – | standing; last run 0b8b723: --submission passed, highlights regenerated, --final fails only on author and Zenodo-release markers |
| D31 | References for datasets and software, verified online on 2026-11-25: LeCun et al. 1998 (Proc. IEEE 86(11):2278-2324, doi 10.1109/5.726791; MNIST page seen in an archive snapshot, the live page timed out), Krizhevsky 2009 (technical report; cifar.html), MedMNIST v2 (Sci. Data 10:41, doi 10.1038/s41597-022-01721-8; the full author list could not be read, so the entry keeps "Yang J, et al."), Parkhi et al. 2012 (CVPR pp. 3498-3505; licence CC BY-SA 4.0 on the dataset page), Paszke et al. 2019 (arXiv 1912.01703; NeurIPS 2019 pages not verified, so omitted), TorchVision (its CITATION.cff: "TorchVision maintainers and contributors", 2016), Captum (repository github.com/pytorch/captum, now under meta-pytorch), ROAD code (github.com/tleemann/road_evaluation, MIT, revision e6b9987 as in `xai/road.py`), own code release (DOI marker) | Elsevier asks for author, title, repository or venue, version, year, identifier | References, Section 4 and Data availability | Agent | – | done (T6.1 conformance); TODO: the MNIST URL and the MedMNIST author list if the author wants full author lists |
| D32 | Full postal address of the affiliation, including the country | Not known to the agent | `[[AUTHOR: postal address]]`, title page; `cover_letter.md`; `declarations.md` | Author | none | todo |
| D33 | Figure resolution gaps: Fig. 3 (2210 px wide), Fig. 7 (2195 px) and Fig. A1 (1430 px) are bitmaps from the analysis plots (`dose_response.py`, `analyze_results.py`), below or near the 2244 px full-page minimum, not vector, and their colours were not changed (not checked for colour-blind safety). Figures 1, 2, 4, 5, 6 are regenerated from the CSVs in `build_paper.py`: vector PDF and 2960 px PNG, colour-blind-safe palette | Analysis code and results must not change here | `paper/figures_submission/` | Author or agent | decision to touch `dose_response.py` plotting | todo |
| D34 | Competing-interest declaration as a Word file generated by the journal's declarations tool | The journal requires the tool's file; `docs/submission/declaration_of_competing_interests.docx` is only a draft | submission form | Author | author's statement (D16) | todo |
| D35 | Author biography (at most 100 words) and passport-type photograph as a separate file | Facts and photograph only the author has; template in `docs/submission/author_biography_TEMPLATE.docx` (markers inside the file) | Vitae | Author | none | todo |
| D36 | Zenodo release upload: package built by `paper/make_release_package.py` into `release/zenodo/` (git-ignored); fields in `docs/submission/zenodo_release_metadata.md`; then replace `[[ZENODO_RELEASE_DOI]]` in the paper, `CITATION.cff`, `RELEASE_README.md`, `declarations.md` and `cover_letter.md` | The author uploads (roadmap T6.3) | see D13 | Author | the last commit before upload | todo |

## 3. Choices made in M7 and in the guide-conformance step (2026-11-25)

- **Tables** have horizontal rules only and no shading in the submission build; the draft keeps its grid and shading.
- **Appendix A** is numbered Table A.1 and Fig. A.1 in the submission, so the main text has 15 tables and 7 figures.
- **Keyword** "Deletion and insertion" became "Perturbation metrics" in the submission (the draft keeps the old list).
- **Declarations** are in the order Acknowledgements, AI declaration, CRediT, competing interest, Funding, Data availability.
- **Colours** of the five regenerated figures changed to a colour-blind-safe palette and the layout to a 3 x 3 grid so that the text is readable at page width; data and panels are unchanged.
- **Draft docx** was not rebuilt in the commit: rebuilding it now changes numbers (the committed draft is older than the T5.1 results); the draft code path is unchanged (text compared before and after the change).

## 3a. Choices made in M7 that the author may want to reverse

- **Tables moved to the supplement** (the audit's table numbers were from an older draft; the
  current ones are): the full pre-registered Pet table (Table S1, abridged to 11 rows in the main
  text), the size-matched responsiveness table (S2), the paired metric-versus-metric table (S3),
  the legacy analysis (S4) and the original-model-set check (S5). The audit's "Tables 7, 8, 12, 13"
  were mapped by content: its Table 7 is now the size-matched table, Table 8 the pairs table,
  Tables 12 and 13 the legacy and model-set tables.
- **Abridged Table 6 rule:** every P1 and P4 row, the P2 contrast and the Deletion (black) rows of
  P3 (the rows of the one failed prediction). 11 rows.
- **Draft build is unchanged.** All N15 wording fixes, the figure reorder and the removal of
  internal references apply to the submission only. The draft text stays byte-identical, as asked.
- **Declarations:** the draft statements "no competing interests" and "no grant" were not carried
  into the submission. They are author claims and became markers.
- **Pre-registration paragraph** (Section 4) lists Zenodo versions 1 to 4 with their DOIs and dates (version 4 added in T5.1)
  from `docs/PREDICTIONS_amendments.md`.

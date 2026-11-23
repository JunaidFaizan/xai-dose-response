# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Deferred and skipped items for the journal submission (T6.1 / T6.2)

Created with the M7 submission build (2026-11-23). This file lists everything that the
offline submission build did not do or could not do yet, where each item plugs in, who does it,
and what blocks it. Update the status column when an item is done. `build_paper.py --final`
fails while any `[[marker]]` below is still in `paper/` or in `docs/submission/*.md`
(this file and `submission_checklist.md` are not scanned).

Build commands (project root, `.venv\Scripts\python.exe`):

- `paper\build_paper.py` builds the unchanged guide-review draft, `paper/paper_draft.docx`.
- `paper\build_paper.py --submission` builds `paper/paper_submission.docx`,
  `paper/supplementary_submission.docx` and `paper/highlights_submission.txt`. It checks the
  title, the forbidden words, the cross-references and that only the declared markers remain.
- `paper\build_paper.py --final` runs the same build and fails if any marker remains. It
  writes no file when it fails.

## 1. Markers

Every marker the submission build may contain. The list is `DECLARED_MARKERS` in
`paper/build_paper.py`; a marker that is not declared there fails the build.

| Marker | Where | Replaced by | Row |
|---|---|---|---|
| `[[SD3_METHODS]]` | main text, Section 3.9 | final SD3 method details (timing-gate outcome, image counts actually used) | D03 |
| `[[SD4_METHODS]]` | main text, Section 3.9 | final SD4 doses, pilot outcome, mark description | D05 |
| `[[SD3_RESULTS]]` | main text, Section 5.11 | SD3 results, tables, verdicts on P10 and P11 | D01, D02 |
| `[[SD4_RESULTS]]` | main text, Section 5.12 | SD4 results, tables, verdicts on P12 and P13 | D04 |
| `[[SD3_POSITIONING]]` | note of Table 1 | one sentence on exact ROAD results in the positioning | D09 |
| `[[SD3_SD4_LIMITATIONS]]` | Section 7, last bullet | limitations updated with what SD3 and SD4 showed (delete the "pending" bullet) | D07 |
| `[[SD3_SD4_CONCLUSION]]` | end of Section 8 | one or two sentences if a pre-registered prediction is confirmed or refuted | D08 |
| `[[SD3_SD4_SUPPLEMENT]]` | end of the supplementary document | ROAD and SD4 supplementary tables | D10 |
| `[[AUTHOR: e-mail]]` | front matter | corresponding author's e-mail | D14 |
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
| D01 | SD3 results section and tables: ROAD MoRF and LeRF preference, over-detection, switch and ρ for all comparisons, per setting; verdicts on P10 and P11 | SD3 scores on the GPU models do not exist yet. The laptop ROAD scores for 206 models exist in `per_model/*_road.csv` but are not analysed | `[[SD3_RESULTS]]`, Section 5.11, new tables | Agent (T5.1) | Kaggle S4 (T4.4); ROAD analysis code in `analyze` | blocked (T4.4) |
| D02 | P10 paired image-bootstrap analysis (difference Deletion (black) − ROAD in mean preference(R over O) at the lowest-dose models, 2000 replicates, seed 0, same images) | The analysis is specified in `docs/PREDICTIONS_sd3_sd4.md` but not coded | `[[SD3_RESULTS]]` | Agent (T5.1) | D01 data; code not written | todo |
| D03 | Final SD3 method text: whether the Pet timing gate reduced Pet to 25 images, the exact image counts used, the licence line for the ported ROAD operator | The timing test (T3.2) and the Kaggle timing check (T4.2) are not done | `[[SD3_METHODS]]`, Section 3.9 | Agent after author's timing | T3.2, T4.2 | todo |
| D04 | SD4 results section: P12 (ρ > 0.8 for at least three metrics) and P13 (black and blur over-detection > 0.5), behaviour, SD4 against SD1 | SD4 models do not exist yet | `[[SD4_RESULTS]]`, Section 5.12, new tables | Agent (T5.1) | Kaggle S3 and S4 (T4.2, T4.4) | blocked (T4.4) |
| D05 | SD4 pilot completion note (final doses by the A3 dose rule), Zenodo version 4 DOI, and the final SD4 method text | The pilot has not run | `[[SD4_METHODS]]`; add version 4 to the Pre-registration paragraph in Section 4 | Agent writes the note (T4.3), author uploads to Zenodo | Kaggle S3 (T4.2), Zenodo v4 | blocked (T4.2) |
| D06 | Abstract, highlights and cover letter changes | The abstract must change only if a pre-registered SD3 or SD4 prediction is confirmed or refuted (roadmap T5.1). The abstract is at the 250-word mark, so an addition needs a cut | Abstract in `build_paper.py`; `docs/submission/highlights.md` is regenerated from `paper/highlights_submission.txt` | Agent (T5.1) | D01, D04 | blocked (D01, D04) |
| D07 | Limitations update (replace the "pending" bullet) | Depends on SD3 and SD4 outcomes | `[[SD3_SD4_LIMITATIONS]]`, Section 7 | Agent (T5.1) | D01, D04 | blocked |
| D08 | Conclusion update | Same | `[[SD3_SD4_CONCLUSION]]`, Section 8 | Agent (T5.1) | D01, D04 | blocked |
| D09 | ROAD in the positioning (Table 1) and in Section 2.3 | Exact ROAD is described as a pre-registered extension in Section 3.9, but Table 1 has no exact-ROAD result | `[[SD3_POSITIONING]]` | Agent (T5.1) | D01 | blocked |
| D10 | Supplementary tables for ROAD (all comparisons) and SD4 | No data | `[[SD3_SD4_SUPPLEMENT]]` in `supplementary_submission.docx` | Agent (T5.1) | D01, D04 | blocked |
| D11 | Multiplicity caveat for the new tables: say which Benjamini–Hochberg and Holm families the ROAD and SD4 tests belong to | The families in Section 3.7 were fixed for the existing analyses; new tests need their own stated families | Section 3.7 and the notes of the new tables | Agent (T5.1) | D01, D04 | todo |
| D12 | Neurocomputing guide-for-authors check in a browser: no-fee (subscription) route, abstract word limit, highlights rule, article length, reference style and author-list format, graphical abstract, whether a separate title page is required | The guide is behind a login redirect for the tool used (`sciencedirect.com` redirects to `id.elsevier.com`). The abstract is 250 words and each highlight is at most 85 characters, which follow the usual Elsevier rules but are not confirmed for this journal | Whole paper | Author (T6.4) | browser | todo |
| D13 | Zenodo release of code, per-model results, analysis outputs and checkpoints, and its DOI | T6.3 comes after T5.1 | `[[ZENODO_RELEASE_DOI]]`, Data availability; `declarations.md`, `cover_letter.md` | Author (T6.3) | D01, D04 | blocked (T5.1) |
| D14 | Corresponding author's e-mail | Unknown to the agent | `[[AUTHOR: e-mail]]`, front matter; `cover_letter.md` | Author | none | todo |
| D15 | Funding statement | Unknown. The draft's text ("did not receive any specific grant") was not carried over because it is a claim only the author can make | `[[AUTHOR: funding statement]]` | Author | none | todo |
| D16 | Competing-interests statement | Same | `[[AUTHOR: competing interests]]`; `cover_letter.md` | Author | none | todo |
| D17 | AI-use statement: name of each tool, what it was used for and the author's oversight (Elsevier asks for all three) | The draft says only "AI assistants". Policy: https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals (read 2026-11-23). The section is placed immediately before the references, with the title Elsevier suggests | `[[AUTHOR: confirm AI-use statement per Elsevier policy]]` | Author | none | todo |
| D18 | Guide's role and acknowledgements. The draft listed the guide for "Supervision" in CRediT; the submission is sole-author as the roadmap states | Whether the guide is acknowledged, or is a co-author, is the author's decision | `[[AUTHOR: acknowledgements]]` | Author | none | todo |
| D19 | Make the GitHub repository public (`https://github.com/JunaidFaizan/xai-dose-response`) and check it for `data/`, `.pt`, zips and `.agents/` | T6.5 | `declarations.md` code availability | Author (T6.5) | D13 | todo |
| D20 | N15 items not done: (a) leading the abstract with the CIFAR ResNet-18 inpainting and insertion switches (8.6–8.7%, bracketed) was only "consider"; (b) panel titles in Figs. 3 and 7 are the raw setting identifiers (`pet_resnet18`, `cifar10`); the caption explains them; (c) in the reordered Fig. 3 the legend sits over the third panel (it was in the first panel of the original figure) | (b) and (c) need the analysis figures to be regenerated from the CSVs, which `dose_response.py` and `xai/` rules forbid changing here | Figs. 3 and 7, abstract | Agent or author | decision | todo |
| D21 | Reference list: N13 fixes (LATEC track, venues and page numbers, full author lists for four entries, Neurocomputing author-list style), and the "no authors" entries | M7 left the list identical to the draft (checked by a test). It needs D12 for the style | References | Agent after D12 | D12 | todo |
| D22 | Exact page count of the main text and the supplement | No Word or LibreOffice on the build machine. Estimate in the final report of T6.1 only | `paper_submission.docx` | Author | Word | todo |
| D23 | Possible length cuts if the guide for authors sets a page limit (the main text is still long, 14 tables and 8 figures) | The limit is unknown (D12) | main-text Tables 7, 8, 10 and 13 (behaviour, labels, confound, P9 switches) are the next candidates | Agent | D12 | todo |
| D24 | `elsarticle` LaTeX version | Optional; out of scope in the roadmap | – | Agent | decision | todo (optional) |
| D25 | Final number check of every number in the paper against the CSVs after T5.1, then `build_paper.py --final` | The build asserts cover most numbers, but the SD3 and SD4 text does not exist yet | whole paper | Agent (T5.1), author (T6.5) | D01, D04 | blocked |
| D26 | Cover letter: sentences on SD3 and SD4 results; reviewer and editor details; article type | See `cover_letter.md` markers | `cover_letter.md` | Author and agent | D01, D04 | todo |
| D27 | Suggested reviewers: institutional e-mail addresses and affiliations, and a check for conflicts of interest | Not known to the agent; none was invented | `suggested_reviewers.md` | Author | none | todo |
| D28 | The pilot table text in Section 4 describes the SD2 pilot as "one of the SD2 models, not scored yet" (from `pilots.csv`) although it is scored and pooled | Found while building; wording comes from the `PILOTS` text in `build_paper.py` and also appears in the draft | Section 4, Amendment A2 paragraph | Agent | decision | todo |
| D29 | Markers in `docs/submission/*.md` | See the files | `cover_letter.md`, `declarations.md`, `suggested_reviewers.md` | Author | none | todo |
| D30 | Rebuild and re-check after any change: `--submission`, then `--final`, then regenerate `docs/submission/highlights.md` from `paper/highlights_submission.txt` if the abstract changed | Highlights are generated from the same numbers as the abstract | – | Agent | – | standing |

## 3. Choices made in M7 that the author may want to reverse

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
- **Pre-registration paragraph** (Section 4) lists Zenodo versions 1 to 3 with their DOIs and dates
  from `docs/PREDICTIONS_amendments.md`.

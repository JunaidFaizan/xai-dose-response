# Audit 2, reviewer 04: Venue and framing

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Lens: venue strategy and framing, judged against what Neurocomputing and Pattern Recognition reviewers expect. This is a read-only audit at commit `80cd09f`. No existing file was changed. Scratch files (docx text extraction, check scripts, SCImago widget images, an over-detection sensitivity table) are in `docs/audit2/scratch/04_venue_framing/`.

What I read: `README.md`, `SYNOPSIS.md`, `docs/PAPER_OUTLINE.md`, `docs/PREDICTIONS_pet.md`, `docs/PREDICTIONS_amendments.md`, `docs/audit/04_venue_framing.md` (my first-audit counterpart), `docs/audit/06_must_do_resolution.md`, and the full text of `paper/paper_draft.docx`, which I extracted with python-docx (`scratch/04_venue_framing/paper_text.txt`). I checked the docx numbers that matter for framing against `responsiveness_corrected.csv`, `behaviour_summary.csv` and `preference.csv` (`check2.py`, `check4.py`).

---

## 1. Score

**Q1 readiness now: 6.5 / 10 for Neurocomputing** (first audit: 5 / 10). **For Pattern Recognition: 5.5 / 10** (first audit: 4 / 10).

Why it went up:
- The paper now targets a Q1 journal. It contains the Pet study and the pre-registration, and it has one thesis that the abstract, introduction, discussion and conclusion all state the same way.
- Natural images come first.
- The novelty claim is narrowed, and BAM, SMERF, OSCAR and Bastings et al. are cited and set apart from this work (Table 1).
- The Elsevier declarations, highlights and keywords are present.
- Highlights are 61–81 characters. The abstract is 236 words by whitespace count; `06` says 244, and both are under 250.

Why it is not higher:
1. **Scope is unchanged.** All networks are trained from scratch, all shortcuts are synthetic, and only one metric family is tested. The paper itself lists all three as limitations (Section 7, bullets 1–3). A Pattern Recognition reviewer will treat them as the main objection.
2. **The manuscript reads as a revision log.** It refers to "earlier drafts", "the audit" and internal file paths, and it has a "Legacy analysis" appendix covering drafts no reviewer has seen (Section 4, N1).
3. **The headline over-detection numbers depend on the threshold.** The Pet figures rest on near-ties, and the CIFAR-10 figures rest on a Gaussian object map that the paper itself calls secondary evidence. The abstract qualifies neither (N2, N3).
4. **The paper is long and table-heavy for a journal.** The main text has about 6,500 words of prose plus about 4,500 words inside the main-text tables, 13 tables in total and 8 figures (N5).

---

## 2. First-audit weaknesses: status

| First-audit item | Status | Evidence |
|---|---|---|
| **B1** Target is Q2 (SN Computer Science) | **Resolved** | Docx front matter, README L17, SYNOPSIS L33, PAPER_OUTLINE L11: Neurocomputing, Q1. Grep finds no "SN Computer Science" in the docs or builder. The quartile is re-confirmed in Section 5. |
| **B2** Paper lacks Pet and pre-registration; `N_MODELS == 174` | **Resolved** | Docx Section 4: 234 models (Pet 18, CIFAR ResNet-18 18, CIFAR CNN 12, MNIST 102, PneumoniaMNIST 84). Section 5.3 and Table 5 give P1–P4 with commits a962b6c → 20694f2 → d823737 → 1e29918. Pet is in every table. |
| **M1** No single thesis | **Resolved** | "the metrics respond to reliance but are not calibrated to it, and their verdicts depend on the size and form of the explanations being compared" appears in the introduction, Discussion and Conclusion, and is echoed in README L16 and SYNOPSIS L32. The four contributions map onto the thesis. |
| **M2** OSCAR and Bastings not cited; novelty too broad | **Resolved** | Section 2.1 and Table 1 cite BAM [19], SMERF [20], Bastings [21] and OSCAR [22]. The intro says "In our searches we did not find…". Table 1 ends with "new as a combination; its individual pieces are not". This is the honest positioning a reviewer wants. |
| **M3** Scope: small images, synthetic shortcuts, no pretrained model, weak Pet models | **Not resolved** | Section 4: "All networks are trained from scratch". Limitations bullets 2–3. Table 2: Pet accuracy without the shortcut is 82.4 → 65.2%. SD1 and SD4 have not been run. |
| **M4** Weakest settings first | **Resolved** | The abstract, Section 5.2 and Table 3 lead with Pet, then CIFAR-10. MNIST and PneumoniaMNIST are called "mechanism settings" (Section 5.5). |
| m1 Title or keyword discoverability | Resolved (no title change) | The keywords include "Faithfulness metrics" and "Meta-evaluation". The title matches the project rule in the docx, README, SYNOPSIS, PAPER_OUTLINE, PREDICTIONS_* and VIVA_PREP. See N6 for one file that does not. |
| m2 7-panel strip figure | Partly resolved | Fig. 1 is now described as four metrics × seven settings, natural images first. Figs 3 and 7 still say "alphabetical setting order" (N4). I did not inspect the rendered image sizes. |
| m3 Sanity-check study dilutes the message | Resolved | It moved to Appendix A. |
| m4 Cover letter | Not done | No cover letter exists. Placeholders remain: [email], [Guide Name], the data-availability link. |
| R6 Zenodo artifact and checklist | Not done | The Data availability statement is a placeholder. SD6 is listed as not done in `06` §4. |

---

## 3. What a Neurocomputing or Pattern Recognition reviewer will now say

This section is my judgement as a reviewer for these journals. It is not a measured fact.

- **Neurocomputing.** The scope fits: an empirical study of neural-network evaluation, with XAI evaluation papers common in the journal. The statistics (two-level bootstrap, BH and Holm, effect sizes, isotonic switch with identification status) are now above the journal's typical level. The pre-registration and the honest reporting of P3's failure, the A1 results that weaken earlier claims, and Appendix C are real strengths. The likely main objections are the scope items (from-scratch CNNs, a white-square shortcut, deletion and insertion only) and "how practical is this?". With SD1 or SD4 done, I would expect major revision rather than rejection.
- **Pattern Recognition.** This journal places more weight on generality and on realistic vision settings. With 144 of the 234 models at 28×28, a synthetic patch, a CIFAR-10 "object map" that is a Gaussian prior, and only one metric family, desk-reject or reject risk is high until at least SD1 and SD4 are done. Length is a second problem. A search-result snippet of the Pattern Recognition Guide for Authors (the page itself would not open) says submissions are single-column, double-spaced and 20–35 pages. My estimate puts the current draft at about 40–45 double-spaced pages: about 23 pages of prose, about 10 pages of main-text tables, 7 main-text figures and 4 pages of references. **UNVERIFIED** limit.

---

## 4. New problems introduced by the fixes

Each item was checked against the extracted docx text or the CSVs. The numbers below are reproduced in `scratch/04_venue_framing/check2.py` and `check4.py`, and in `overdetection_band.csv`.

**N1. The paper refers to its own revision history and to internal audit files (major for submission).**
- Section 2, first paragraph: "The literature audit behind this section … docs/audit/02_literature_novelty.md".
- Table 1 caption: "after the audit's positioning table, docs/audit/02…".
- Section 3.6: "(after the audit's reviewer 05)" and "A check label (reviewer 03)".
- Section 5.1: "Earlier drafts reported 'about 15%, not 0' here".
- Section 5.5: "earlier drafts' statement that … 'clearly before 50%' was not supported".
- Section 5.6: "'rises to 0.93–0.99 for every metric' in earlier drafts was too general".
- Limitations: "chosen after the audit".
- Appendix B, "Legacy analysis", reports values from unpublished drafts.

A journal reviewer has never seen these drafts or audits, so the corrections read as rebuttals to criticism the reviewer never made. This is right for the guide-review draft, but the submission build must remove it. The honest part should stay: the analysis choices were made after the original-set results had been inspected. State it once, as a plain limitation without the word "audit". Move Appendix B to the repository. The front-matter note citing `docs/audit/00_combined_audit.md` must also go.

**N2. The Pet over-detection headline depends on the threshold.**
- The abstract says "black and blur deletion still prefer the shortcut map for 75–92% of Pet models". This matches the CSV (`fpr` 0.750 and 0.917 for Pet).
- The decision rule is preference > 0.5, and the Pet per-image AUC gaps at the lowest dose are near zero (median −0.009 and 0.0002). The paper admits this in Section 5.2 ("close to ties"), but not in the abstract or the highlights.
- My sensitivity check (`check4.py`, read-only over `preference.csv`) requires preference > 0.6, which is about the 95% band for a fair coin over 100 images.
  - Pet black and blur over-detection falls from 0.75 / 0.92 to 0.50 / 0.58.
  - Pet inpainting falls from 0.58 to 0.17.
  - CIFAR-10 black and blur stay at 0.93–1.00.
- So the Pet over-detection finding survives, but at about half the stated size. A careful reviewer will run exactly this check.

**N3. The CIFAR-10 headline rests on the oracle that the paper calls secondary.**
- The abstract says "100% of CIFAR-10 models" over-detect, and that "on CIFAR-10 black deletion prefers a random ordering over a centred object map 81% of the time".
- Section 3.4 says the CIFAR object map is a centred Gaussian prior, "Pet is therefore the primary natural-image evidence".
- The 81% is the small CNN only. ResNet-18 is 72% (`pref_low` 0.815 vs 0.723, `random vs full object`).
- Table 4 shows that on CIFAR-10 S–O ≈ R–O ≈ P–O (for example 0.81 / 0.81 / 0.81 for the CNN with black deletion). Any small map beats the Gaussian prior, so the "over-detection" there partly measures a bad oracle.
- Fix: in the abstract, write "against a centred Gaussian object prior" and give 72–81% (both networks).

**N4. Small internal inconsistencies.**
- Section 5.2 says preference rises "in 27 of 28 cells (ρ = 0.27–0.99)". The 0.27 is the non-significant exception. The abstract correctly gives 0.49–0.99 for the 27 significant cells, and the CSV confirms a minimum of 0.488 among BH-significant cells.
- Figs 3 and 7 are "rearranged … (alphabetical setting order)". Every table and Fig. 1 use the natural-first order.
- `06` §1 MD6 says the abstract is 244 words, while my whitespace count is 236. This is harmless.

**N5. The paper is too long and table-heavy.**
- Table 5 (pre-registration) has 27 rows, and the verdict string "Fails for Deletion (black); holds for the other three metrics" repeats on 16 of them.
- Tables 3 and 7 each have 28 rows × 8 columns, and Table 8 has 18 rows.
- A journal version should keep Table 3 in full, shrink Table 5 to one row per prediction × metric with a single verdict column (about 10 rows), and move Tables 7, 8, 12 and 13 to supplementary material.

**N6. One document does not use the project title.** `docs/SEMINAR_GUIDE.md` lines 1–4 read "Seminar Guide: Do Explanation Metrics Catch Shortcuts?" and "A Ground-Truth Evaluation of XAI Metrics for CNN Image Classifiers". The project rule requires the exact title in every document. This was not in the first audit's list. I did not change it because this audit is read-only; it should be updated.

**N7. The Pet switch points are stated without their caveats.**
- The abstract says "on Pet the metrics switch at 1.3–5.4% chance-corrected reliance, while the models follow the shortcut on half of the images only at 54.5%".
- The 1.3% (black) is flagged "(e)": it is identified only at the lowest-reliance edge. Blur has no switch at all.
- At these points the preferences start at 0.35–0.50 with near-zero gaps.
- The contrast with 54.5% is the most quotable sentence in the paper, and it is the weakest number. Add "near ties at low reliance" to the abstract, or lead with the CIFAR ResNet-18 inpainting and insertion switches (8.6–8.7%, both "identified" with bracketing models), which are better identified.

None of N1–N7 is a wrong number. All are framing and presentation problems that the Section 6 recommendations can fix without new training.

---

## 5. Venue verification (done again on this run)

Method:
- **Quartiles:** I downloaded each SCImago widget image (`journal_img.php?id=…`) into `scratch/04_venue_framing/sjr_<id>.png` and viewed it. Each image shows the journal name, best quartile, best category and SJR 2025.
- **Fee model:** I queried the OpenAlex source API by ISSN. It reports whether a journal is fully open access, whether it is in DOAJ, and its listed open-access APC. A journal that is not fully OA but lists an APC is hybrid, which means a subscription (no-fee) route normally exists. OpenAlex is a third party, not the journal's own page.
- **What failed:** the journals' own pages could not be opened. ScienceDirect guide and OA pages redirect to a login loop, the elsevier.com pages redirect to ScienceDirect, the SpringerLink page redirects to a login, and the SCImago journal pages and Sherpa Romeo return HTTP 403. Elsevier's Neurocomputing open-access-options page appears in search results with the snippet "No open access publication fee", but I could not open it, so it is **UNVERIFIED**.

| Journal | Quartile and best category (widget viewed) | SJR 2025 | OpenAlex: fully OA? / APC for OA | No-fee route, own page | Scope fit for this paper |
|---|---|---|---|---|---|
| Neurocomputing (Elsevier) | Q1, Artificial Intelligence | 1.47 | No / USD 2,670 | **UNVERIFIED**. Hybrid per OpenAlex, so a subscription route is very likely | Good |
| Pattern Recognition (Elsevier) | Q1, Artificial Intelligence | 2.15 | No / USD 2,590 | **UNVERIFIED**. Hybrid per OpenAlex | Very good topic fit; generality bar high |
| Machine Learning (Springer) | Q1, Artificial Intelligence | 1.03 | No / USD 3,290 | **UNVERIFIED**. Hybrid per OpenAlex | Good, if framed as measurement validity |
| Neural Networks (Elsevier) | Q1, Artificial Intelligence | 1.55 | No / USD 3,110 | **UNVERIFIED** | Good |
| Computer Vision and Image Understanding (Elsevier) | Q1, Computer Vision and Pattern Recognition | 0.82 | No / USD 2,090 | **UNVERIFIED** | Fair–good |
| Pattern Recognition Letters (Elsevier) | Q1, Computer Vision and Pattern Recognition | 0.79 | No / USD 2,500 | **UNVERIFIED** | Letter format only; would need heavy cutting |
| Data Mining and Knowledge Discovery (Springer) | Q1, Computer Networks and Communications | 1.2 | No / USD 3,190 | **UNVERIFIED** | Weak for image CNNs |
| IEEE TNNLS | Q1, Artificial Intelligence | 3.15 | No / USD 2,645 | **Verified** on the IEEE CIS page: the $110/page charge is not obligatory. A mandatory $200/page applies beyond 10 pages for a full paper, with a 15-page maximum. Free only if ≤ 10 pages | Good topic fit; bar very high, and length makes it impractical |

All quartiles match the first audit and the docx front matter (Neurocomputing Q1 AI, SJR 1.47). Every no-fee route except TNNLS's still needs a 10-minute check in a normal browser. That check is the only remaining blocker on the venue side that the agents cannot do.

### Updated shortlist (submit to one journal at a time)

1. **Neurocomputing.** This is still the first choice. The topic fits, the statistics are now strong, and no length limit was found (UNVERIFIED). Submit once R1–R3 are done. R4–R6 raise the odds further.
2. **Pattern Recognition.** Submit here first only if SD1 (pretrained backbone) and SD4 (realistic shortcut) are both done and the paper fits in about 35 double-spaced pages (snippet only, UNVERIFIED). Otherwise use it as the second choice after Neurocomputing.
3. **Machine Learning (Springer).** Use it if Neurocomputing rejects on scope rather than on substance. Frame the paper as "measurement validity of evaluation metrics", where the pre-registration counts most.
4. **Neural Networks**, then **CVIU**, as further fallbacks.
5. Avoid TNNLS because of the 10-page cap with mandatory over-length charges. Avoid PRL unless a short letter version is wanted. Avoid DAMI, ESWA and KBS as first choices because of scope.

---

## 6. Ranked recommendations

"Days" are working days for one student with AI help. "GPU h" are Kaggle T4 hours, at about 30 h per week with sessions of at most 12 h. GPU estimates are from the first audit and from `docs/audit/06` timings: Pet from scratch about 25 min per model and CIFAR ResNet-18 about 11 min per model (docx Section 4).

| Rank | Recommendation | Addresses | Cost | Value |
|---|---|---|---|---|
| **R1** | **Turn the guide draft into a submission manuscript.** Add a submission mode to `build_paper.py` that removes every "earlier drafts", "the audit" and "reviewer 03/05" reference and the `docs/…` paths in captions and text (N1). Move Appendix B, Tables 7, 8, 12 and 13, and the full Table 5 to a supplementary file. Shrink Table 5 to about 10 rows. Keep the "analysis choices made after inspecting the original set" limitation, reworded without "audit". | N1, N5 | 1.5 days, 0 GPU h | High. Without it a reviewer reads the paper as a patched draft. |
| **R2** | **Qualify the headline numbers.** Add a ±0.1 indifference-band (or per-model binomial) sensitivity for over-detection to Table 3 or a supplementary table: Pet black/blur 0.75/0.92 → 0.50/0.58 (`overdetection_band.csv`). In the abstract, say "near ties" for the Pet switches and "against a centred Gaussian prior" for CIFAR-10. Give 72–81% for both CIFAR networks. Fix the 5.2 range (0.49–0.99) and the Fig 3/7 order. | N2, N3, N4, N7 | 1 day, 0 GPU h (CPU over `preference.csv`, built through `build_paper.py`) | High. It pre-empts the most obvious statistical objection to the thesis. |
| **R3** | **SD1: ImageNet-pretrained ResNet-18 on Pet**, 6 doses × 3 seeds, standard stem, about 10 fine-tuning epochs. Pilot one model, set the doses, and record an amendment before training (as A1 did). Add one or two pre-registered predictions, for example that P1 holds and that black and blur over-detection persists. | M3, Limitation 2 | 3 days, about 5–6 GPU h | Highest-value GPU item. It answers "weak from-scratch models" and is the minimum for Pattern Recognition. |
| **R4** | **SD4: realistic shortcut on Pet**, such as a semi-transparent text or logo watermark at a class-coded position, on the pretrained backbone so it shares R3's pipeline. Use the same doses and seeds. | M3, Limitation 3 | 3 days, about 6–8 GPU h | High. It changes "white square" into a practical scenario and improves the paper's practical relevance. |
| **R5** | **SD3: exact ROAD (noisy linear imputation, MoRF and LeRF)**, scored on the existing 234 models. If time allows, add one non-ordered-removal metric such as faithfulness correlation on a subset. | Limitation 1 ("one metric family") | 2–3 days, about 2–4 GPU h (Pet and ResNet models; the rest on CPU) | Medium–high. "Only deletion and insertion" is the second most likely reviewer objection, and ROAD is the metric the inpainting variant is compared with. |
| **R6** | **Submission package.** Check the Neurocomputing no-fee route, abstract and highlight limits and reference style in a browser (10 min). Fill the [email], [Guide Name] and data-availability placeholders. Add a Zenodo DOI and the `validate_metric()` entry point (SD6). Write a cover letter naming the pre-registration commits and 3–5 suggested reviewers from the cited meta-evaluation papers. Prepare an elsarticle LaTeX version if the guide prefers. | m4, R6 of audit 1 | 1.5 days, 0 GPU h | Medium. Editors check these at desk review. |
| **R7** | **SD2: fill the reliance gaps** for Pet and CIFAR-10 ResNet-18 (about 39–96% on ResNet-18; `06` §6), with about 6 models each under a new dated amendment. | Section 5.4 behavioural switch and calibration-gap claims on natural images | 1 day, about 4–5 GPU h | Medium–low. Do it only after R3–R5, or if the switch-point contrast stays in the abstract. |
| **R8** | **Fix N6.** Put the exact project title in `docs/SEMINAR_GUIDE.md`. | Project rule | 0.1 day, 0 GPU h | Required by the rule; no score effect. |

Total: about 13 days and about 17–23 GPU h. This fits the 3-week window and one Kaggle week, with R7 as the optional extra. If GPU time runs short, do R3 before R4, and R4 before R5.

---

## 7. Projected score

| Work done | Neurocomputing | Pattern Recognition |
|---|---|---|
| Now | 6.5 | 5.5 |
| R1, R2, R6, R8 (writing only, about 4 days) | 7.0 | 6.0 |
| plus R3 and R4 (pretrained backbone and realistic shortcut) | 7.5 | 7.0 |
| plus R5 (ROAD) and R7 (gap filling) | 8.0 | 7.5 |

These projections are my judgement and assume the new settings broadly replicate the thesis. If the pretrained or watermark models do not over-detect, the thesis must be narrowed to "from-scratch CNNs and synthetic markers". That would still be publishable in Neurocomputing, but at about 7.

---

## 8. External URLs cited (all opened on this run)

- SCImago widget images, viewed as downloaded images:
  - https://www.scimagojr.com/journal_img.php?id=24807 (Neurocomputing)
  - https://www.scimagojr.com/journal_img.php?id=24823 (Pattern Recognition)
  - https://www.scimagojr.com/journal_img.php?id=24775 (Machine Learning)
  - https://www.scimagojr.com/journal_img.php?id=24804 (Neural Networks)
  - https://www.scimagojr.com/journal_img.php?id=24825 (Pattern Recognition Letters)
  - https://www.scimagojr.com/journal_img.php?id=24161 (CVIU)
  - https://www.scimagojr.com/journal_img.php?id=21100235616 (IEEE TNNLS)
  - https://www.scimagojr.com/journal_img.php?id=13579 (Data Mining and Knowledge Discovery)
- OpenAlex source API (full query strings in `scratch/04_venue_framing`):
  - https://api.openalex.org/sources/issn:0925-2312 (Neurocomputing)
  - https://api.openalex.org/sources/issn:0031-3203 (Pattern Recognition)
  - https://api.openalex.org/sources/issn:0885-6125 (Machine Learning)
  - https://api.openalex.org/sources/issn:0893-6080 (Neural Networks)
  - https://api.openalex.org/sources/issn:0167-8655 (Pattern Recognition Letters)
  - https://api.openalex.org/sources/issn:1077-3142 (CVIU)
  - https://api.openalex.org/sources/issn:2162-237X (TNNLS)
  - https://api.openalex.org/sources/issn:1384-5810 (DAMI)
- IEEE CIS, TNNLS page charges: https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-page-charges

Tried but could not open, so their content is UNVERIFIED and not cited as a source:
- the ScienceDirect guide-for-authors and open-access-options pages for Neurocomputing and Pattern Recognition (403 or login redirect);
- the elsevier.com Neurocomputing pages (they redirect to ScienceDirect);
- the SpringerLink "how to publish" page for Machine Learning (login redirect);
- the SCImago journal page for id 24807 (403);
- Sherpa Romeo for Neurocomputing (403).

Content from these sources was paraphrased for compliance with licensing restrictions.

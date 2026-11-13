# Audit 2, Reviewer 2: literature and novelty

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Date: 2026-11-13. State audited: `main` at 80cd09f. I read `paper/paper_draft.docx` (text extracted with python-docx), `paper/build_paper.py`, README, SYNOPSIS, `docs/PAPER_OUTLINE.md`, the prediction files, `docs/audit/02_literature_novelty.md` (my first-audit counterpart) and `docs/audit/06_must_do_resolution.md`. I checked each reference by opening its page or its arXiv API record. Where a page redirected to a login or a browser check, I relied on a search-result snippet and mark the entry "snippet". Paper summaries are paraphrased from abstracts. Content was rephrased for compliance with licensing restrictions.

Scratch files: `docs/audit2/scratch/02_literature/` holds `paper_text.txt` and `paper_tables.txt` (docx extraction), three arXiv API dumps, and `citation_check.md` (the full 54-row citation table).

## 1. Score

**Q1 readiness from this lens: 6.5/10 (first audit: 5/10).**

The related work is now accurate and well bounded, which is what held the first score down:
- BAM and SMERF are cited and correctly described.
- The confound literature (ROAD, ROAR pitfalls, Blücher, Back to the Baseline, Gomez, SaCo) is linked to specific findings in the Discussion.
- The novelty claim is narrowed to "in our searches we did not find".
- None of the 54 references is invented, and every one of the paper's statements about a cited work matches that work's abstract.

My own search on 2026-11-13 found no paper that varies measured reliance to meta-evaluate faithfulness metrics, so the core novelty still holds.

Three things keep the score below 7:
1. **Breadth.** The fixed title promises "Explanation Metrics", but every metric tested is one family (ordered pixel deletion or insertion with three fills). Every comparable benchmark tests more families (LATEC 20 metrics, M4, Quantus). This is the first-audit weakness M1, and it is unchanged.
2. **A positioning gap that touches the main thesis.** The headline finding is that the metrics over-detect a shortcut the model barely uses. Two cited papers already report the same kind of false positive for attribution *methods*: BAM [19] and Adebayo et al. 2022 [26]. The paper does not say this, so a reviewer from either line can call over-detection "known".
3. **Recent close work not cited.** Two 2026 papers are missing. AIM measures masking-operator bias and random-attribution bias. "How to Evaluate and Refine your CAM" (ICPR 2026) compares CAM evaluation metrics against synthetic ground truth. There are also several bibliographic updates (Section 4).

## 2. Comparison with my first-audit weaknesses

| First-audit item | Status | Evidence |
|---|---|---|
| **B1.** BAM, SMERF and the confound papers missing; novelty not bounded | **Resolved** | Docx Section 2.1–2.3 and Table 1 cite BAM [19], SMERF [20], OSCAR [22], Bastings [21], Lin [24], Blücher [13], ROAR pitfalls [41], Gomez [42], Gevaert [11], Quantus [32], Hesse 2024 [35], FunnyBirds [28], XAI-Units [29] and Miró-Nicolau 2026 [31]. Table 1 states: "The protocol is new as a combination; its individual pieces are not. BAM and SMERF already controlled what a model relies on". Section 1 uses "In our searches we did not find". |
| **M1.** One metric family; title says "Explanation Metrics" | **Not resolved** (now disclosed) | Limitations, first bullet: one family; faithfulness correlation and region perturbation not tested; the inpainting variant is not ROAD. The title is fixed by the project rules, so this can only be fixed with data (R3) or by narrowing the abstract and keywords. |
| **M2.** Pet excluded from the paper | **Resolved** | Section 5.3 and Table 5. Pet is first in every table. The docx says 234 models (Pet 18). |
| **M3.** No comparison with ground-truth-free meta-evaluation (MetaQuantus) | **Not resolved** | MetaQuantus is cited [9] but not run. No limitation mentions it. |
| **M4.** "Dose-response" used as a label, not a model | **Resolved** | Section 3.7: isotonic switch with a four-parameter logistic sensitivity check, identification and bracketing rules, two-level bootstrap. Tables 3 and 7 report the switch status. |
| **M5.** Only ResNet-18s trained from scratch on natural images | **Not resolved** (disclosed) | Limitations, second bullet. SD1 is planned but not done. |
| **m1.** Citation errors: 2401.10640 description, LATEC title, MetaQuantus venue, 2401.06465 vs 2405.02383 | **Resolved**, with two leftovers | All four are fixed in the docx ([9], [10], [34], [38], [39]). Leftovers: the LATEC entry lacks its track (it is the NeurIPS 2024 Datasets and Benchmarks Track), and [39] now has a published Springer version. |
| **m2.** Findings not linked to known mechanisms | **Mostly resolved** | Discussion, "Size and form", links findings to [12], [41], [13], [42] and [43]. "From Flexibility to Manipulation", which I recommended, was not added. |
| **m3.** The 50% ideal is an assumption; cite Bilodeau | **Resolved** | Section 3.6 derives the behavioural switch (Pet 54.5%, PneumoniaMNIST patch 36.9%). Bilodeau [44] is cited in 2.3 and in the Discussion. |
| **m4.** About 15% error floor on PneumoniaMNIST and Pet | **Resolved** | Section 3.3 chance correction. Section 5.1 gives the Pet lowest r_c as −0.1% (raw 15.5%). |
| **m5.** Shortcuts are all synthetic | **Not resolved** (disclosed) | Limitations, third bullet. SD4 is planned but not done. |

Of the 11 items: 6 are resolved, 1 is mostly resolved and 4 are not resolved (M1, M3, M5, m5). All four unresolved items need new experiments. All the writing-only items are done.

## 3. Novelty re-check (searches on 2026-11-13)

I ran queries on metric meta-evaluation with ground truth, graded shortcut or spurious strength, backdoor-based XAI evaluation, deletion/insertion validation and pre-registration in XAI evaluation. I found no paper that trains models at controlled, measured reliance levels to test faithfulness metrics. The paper's claim stands, worded as it is now.

New or closer work since the first audit, all opened:

- **"How to Evaluate and Refine your CAM"** (Domeniconi et al., accepted at ICPR 2026, arXiv May 2026). It builds a synthetic dataset with ground-truth attributions in order to compare CAM evaluation metrics, and proposes a composite metric. This is metric meta-evaluation with ground truth at a fixed reliance. It belongs in Section 2.2 and in the Table 1 row on meta-evaluation.
- **AIM** (arXiv May 2026). It argues that the masking operator confounds removal-based scores: zero masking produces out-of-distribution inputs, and interpolation-based masking leaves predictive information in place. It also measures random-attribution bias. Both points describe findings in this paper directly: blur deletion leaves 20–25% of the patch intensity, and black or blur deletion prefer a random ordering over the object map. AIM is independent support and should be cited in Section 2.3 and the Discussion.
- **BonaFide** (Gur-Arieh et al., May 2026; LLM chains of thought). It meta-evaluates faithfulness metrics against ground-truth labels and scores them by AUROC. This is a cross-domain precedent for the per-image AUROC design in Section 5.4. Optional.
- **OSCAR** [22] and **Miró-Nicolau et al. 2026** [31] are already cited. They remain the two groups closest to this design, so the scoop risk from the first audit still applies.

Pre-registration: I found no XAI-evaluation benchmark with pre-registered predictions, so the claim in Table 1 ("in our searches we did not find…") holds.

## 4. Reference list check (all 54 entries)

The full table, with what I opened for each entry, is in `scratch/02_literature/citation_check.md`. Summary:

- **Existence and identity.** All 54 references exist and their titles match. 50 were confirmed on a page I opened or on the arXiv API record. For 4 entries, the venue detail came only from search snippets because the publisher page redirected to a login: Gevaert in *Machine Learning*, DeGrave in *Nature Machine Intelligence*, the Fresh Look Springer chapter, and the Kermany page range.
- **Claims about cited work.** All 54 are accurate against the abstracts. Two are incomplete in a way that matters (N1). One table placement is wrong (N2).
- **Entries to correct or complete:**

| Ref | Problem | Correct detail (source opened) |
|---|---|---|
| [10] LATEC | Venue given as plain NeurIPS | NeurIPS 2024 **Datasets and Benchmarks Track** (proceedings page) |
| [11] Gevaert | Cited as arXiv 2022 | Published in *Machine Learning* (Springer) 2024, DOI 10.1007/s10994-024-06550-x (snippet; page redirected) |
| [14] Back to the Baseline | Cited by title only | Picard AM, Boissin T, Subhash V, Cadène R, Fel T (2025) (arXiv API) |
| [17] DeGrave | "Author manuscript, final venue to be confirmed" | *Nature Machine Intelligence* 2021, article s42256-021-00338-7. PMC7523163 is the medRxiv preprint version. Volume and pages still to confirm. |
| [25] BackX | Cited by title only | Yang P, Akhtar N, Jiang J, Mian A (arXiv API) |
| [30] AttributionLab | Cited by title only | Zhang Y, Li Y, Brown H, Rezaei M, Bischl B, Torr P, Khakzar A, Kawaguchi K (arXiv API) |
| [33] M4 | Cited by title only | Li X, Du M, Chen J, Chai Y, Lakkaraju H, Xiong H (authors' repository README; NeurIPS 2023 D&B page) |
| [39] Fresh look | Cited as arXiv | xAI 2024, Springer CCIS, DOI 10.1007/978-3-031-63787-2_21 (snippet). It overlaps heavily with [38]; consider citing one of them. |
| [42] Gomez | Cited as arXiv | ICPRAI 2022, LNCS, DOI 10.1007/978-3-031-09037-0_8 (HAL record) |
| [54] He | Cited as arXiv | CVPR 2016; He K, Zhang X, Ren S, Sun J |

All four title-only entries in `06_must_do_resolution.md` §4 can now be completed from the list above. Many entries also use "et al." Elsevier numbered styles normally list the authors in full, so check the Neurocomputing Guide for Authors. The full author lists for every arXiv entry are in `scratch/02_literature/arxiv_api*.txt`.

## 5. New problems introduced by the fixes

**N1. The over-detection thesis is not linked to the method-level false positives already reported.** (Major for positioning.)
- The abstract, Section 5.2 and the Conclusion present over-detection as the main finding.
- BAM's abstract reports that certain attribution methods produce false-positive explanations.
- Adebayo et al. 2022 report that feature attributions can indicate dependence on a spurious signal when the model does not rely on it.
- The paper cites both, but only as "controlled reliance" [19] and "ask whether explanations reveal spurious correlations" [26].

The fix strengthens the paper. State that these works found false positives at the **method** level, and that this study shows the **evaluation metric** over-detects too, with oracle maps, so no attribution method is involved. Say it in Section 2.1, Table 1 and the first Discussion paragraph.

**N2. Table 1 misplaces [31].** Table 1 lists Miró-Nicolau et al. 2026 under "Meta-evaluation of faithfulness metrics" as "synthetic intervention-based ground truth". The abstract says the framework evaluates nine XAI methods. Section 2.1 describes it correctly. Move it to the ground-truth-benchmark row, and put the ICPR 2026 CAM-metric paper in the meta-evaluation row.

**N3. The manuscript cites the internal audit and its reviewers.**
- Section 2 opens with "literature audit committed on 2026-11-13, docs/audit/02_literature_novelty.md".
- The Table 1 caption says "after the audit's positioning table, docs/audit/02_literature_novelty.md §2".
- Section 3.6 says "after the audit's reviewer 05" and "(reviewer 03)".
- Limitations say "chosen after the audit".

These references to repository files and internal reviewers do not belong in a journal manuscript and will confuse referees. Keep the disclosure content: the search window, and the fact that the analysis choices were made after the original-set results were inspected. Drop the file paths and reviewer numbers.

**N4. SYNOPSIS misattributes AttributionLab.**
- SYNOPSIS (Problem Statement) says the metrics were meta-evaluated "in hand-designed settings with a fixed, known explanation [19]", where [19] is AttributionLab.
- AttributionLab's abstract describes a faithfulness sanity check for attribution **methods**, not a meta-evaluation of metrics.
- The paper itself does not repeat the error. The correct citation for metric meta-evaluation against a known explanation is Miró-Nicolau et al. 2024, the decision-tree study (paper [34]), which is missing from the SYNOPSIS list.

**N5. "From Flexibility to Manipulation" is still missing.** My first audit recommended it (m2 and recommendation 7). `06_must_do_resolution.md` MD5 does not list it, and the docx does not cite it. It is the most direct support for reporting several fills and for the paired-test recommendation in Section 6.

**N6. Non-archival sources carry two mechanism claims.**
- [41] (ROAR pitfalls) is an ICML 2026 workshop paper, and the Discussion uses it as the mechanism for the "spatially blurry masks" bias.
- [14], [22], [31] and [46] are arXiv preprints.

This is acceptable. Label [41] as a workshop paper, and back the blurry-mask point with ROAD [12] and AIM (above) so it does not rest on one workshop paper.

I checked the numbers in the related-work and abstract text against `responsiveness_corrected.csv`; they agree:
- 27 of 28 cells significant, ρ = 0.49–0.99, exception ρ = 0.27;
- natural-image ρ 0.88–0.99;
- Pet over-detection 0.75–0.92 for black and blur, CIFAR-10 1.00;
- Pet switches 1.3–5.4%;
- Pet median gap −0.060 to 0.000.

The "17 methods, 20 metrics" figure (LATEC) and the "52,000 synthetic 128×128 images" figure (Miró-Nicolau 2024) match their abstracts.

## 6. Ranked recommendations

Costs are my estimates. GPU hours assume a Kaggle T4, and the existing checkpoints are re-scored without retraining wherever possible.

| Rank | Recommendation | Fixes | Cost |
|---|---|---|---|
| **R1** | **Positioning edit.** (a) N1: frame over-detection as the metric-level counterpart of the method-level false positives in BAM [19] and Adebayo 2022 [26]. (b) Add AIM and the ICPR 2026 CAM-metric paper to 2.2–2.3, Table 1 and the Discussion. Add "From Flexibility to Manipulation", F-Fidelity and Hase et al. to 2.3. Add Bhatt et al. as the source of Faithfulness Correlation in Limitations. (c) N2: fix the Table 1 row. (d) N3: remove the internal audit paths and reviewer numbers. (e) N4: fix the SYNOPSIS sentence. | N1–N5 | 1 day, 0 GPU h |
| **R2** | **Bibliography update.** Apply the 10 corrections in Section 4. Complete the author lists to the Neurocomputing style. Label [41] as a workshop paper. Confirm the DeGrave volume and pages and the GLBW entry (if added) in a normal browser. | §4, N6 | 0.5 day, 0 GPU h |
| **R3** | **A second metric family on the existing checkpoints** (addresses M1 and makes "Explanation Metrics" defensible). Implement exact ROAD (noisy linear imputation, MoRF and LeRF, which is SD3). Add Blücher's symmetric relevance gain (MIF and LIF) and Faithfulness Correlation (Bhatt et al.). Score the same oracle maps and run the same responsiveness, over-detection and AUROC analysis. MNIST, PneumoniaMNIST and the CIFAR CNN run on the laptop CPU; the CIFAR ResNet-18 and Pet run on Kaggle. Without this, the most likely literature objection is "one family of 4 near-identical variants". | M1 | 4 days, ~5–8 GPU h |
| **R4** | **SD1: pretrained ImageNet ResNet-18 on Pet**, at the 6 existing doses × 3 seeds, under a dated pre-registered amendment in the style of `PREDICTIONS_amendments.md`. Hesse et al. [35] and LATEC [10] show that backbone design changes attribution quality, so referees will ask. It also removes the "CIFAR stem at 128 px, trained from scratch" caveat. | M5 | 3 days, ~6–10 GPU h |
| **R5** | **SD4: realistic watermark shortcut on Pet** (a semi-transparent text or logo stamp, like the source tags behind the Clever-Hans finding in Lapuschkin et al. [16]), at 6 doses × 2–3 seeds. This answers "synthetic white square", tests the size confound with a shortcut of a different shape, and links to [16] and [17]. | m5 | 3 days, ~6–10 GPU h |
| **R6** | **MetaQuantus on a subset** (MNIST patch and Pet, 3 doses, the 4 current metrics plus R3's). Report whether its noise-resilience and adversarial-reactivity ranking agrees with the dose-response ranking. Either outcome is publishable. A paired paper already exists in the docx: Appendix A does the same thing for the sanity check. | M3 | 3 days, ~2–4 GPU h |
| **R7** | **SD2: fill the reliance gaps for the CIFAR ResNet-18 and Pet.** This has low value from my lens: it tightens the switch points but adds no literature position. Do it only if GPU time is left after R3–R5. | stats | 1.5 days, ~4–8 GPU h |

Suggested order within about 3 weeks:
1. R1 and R2 in days 1–2 (CPU).
2. R3 in week 1. The CPU settings run on the laptop while the Kaggle settings run.
3. R4 and R5 in week 2 (about 12–20 GPU h, inside one week's quota).
4. R6 in week 3, if time allows.

## 7. Projected score

- R1 + R2 only: **7/10**. The positioning becomes hard to attack, but breadth is unchanged.
- R1–R3: **7.5/10**. Two or three metric families make the title's claim defensible.
- R1–R5: **8/10**. A pretrained backbone and a realistic shortcut answer the two remaining standard referee requests from the literature.
- R6 adds novelty, not acceptance odds. I would not count it towards the score.

The planned GPU items rank as follows from this lens. SD3 has the highest value, as part of R3. SD1 and SD4 have medium-high value. SD2 has low value.

## 8. External URLs cited in this report and its scratch table

Every URL below was opened by me on 2026-11-13, except those marked "snippet".

- https://arxiv.org/abs/2609.30397
- https://arxiv.org/abs/2512.18888
- https://arxiv.org/abs/2512.11433
- https://arxiv.org/abs/2405.02344
- https://arxiv.org/abs/2502.09150
- https://arxiv.org/abs/2304.13836
- https://arxiv.org/abs/2404.01415
- https://arxiv.org/abs/2007.07797
- https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7523163/
- https://www.nature.com/articles/s42256-021-00338-7 (snippet; page redirected to login)
- https://hal.science/hal-03633006
- https://aclanthology.org/2022.emnlp-main.64/
- https://proceedings.mlr.press/v162/kim22h.html
- https://proceedings.mlr.press/v70/sundararajan17a.html
- https://jmlr.org/papers/v24/22-0142.html
- https://openaccess.thecvf.com/content/CVPR2023/html/Binder_Shortcomings_of_Top-Down_Randomization-Based_Sanity_Checks_for_Evaluations_of_Deep_CVPR_2023_paper.html
- https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html
- https://github.com/holyseven/M4_XAI_Benchmark
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb04401ba21869f23a77f3e4e1-Abstract-Datasets_and_Benchmarks_Track.html
- https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/
- https://link.springer.com/article/10.1007/s10994-024-06550-x (snippet; page redirected to login)
- https://link.springer.com/chapter/10.1007/978-3-031-63787-2_21 (snippet)
- https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5 (snippet)
- https://www.nature.com/articles/s41467-019-08987-4 (snippet)
- https://dl.acm.org/doi/10.1145/3447548.3467213 (snippet)
- https://arxiv.org/abs/2407.11910
- https://arxiv.org/abs/2310.16228
- https://arxiv.org/abs/2401.06654
- https://arxiv.org/abs/2202.12270
- https://arxiv.org/abs/2409.16756
- https://arxiv.org/abs/2202.00449
- https://arxiv.org/abs/2310.06514
- https://arxiv.org/abs/1907.09701
- https://arxiv.org/abs/2212.04629
- https://arxiv.org/abs/2212.11870
- https://arxiv.org/abs/2605.14641
- https://arxiv.org/abs/2605.16905
- https://arxiv.org/abs/2606.28391
- https://arxiv.org/abs/2601.11625
- https://arxiv.org/abs/2202.05189
- https://arxiv.org/abs/2605.25052
- https://arxiv.org/abs/2412.05592
- https://arxiv.org/abs/2410.02970
- https://arxiv.org/abs/2005.00631
- https://arxiv.org/abs/2106.00786
- https://export.arxiv.org/api/query (arXiv API; queried IDs are listed in `scratch/02_literature/arxiv_api*.txt`)
- https://proceedings.iclr.cc/paper_files/paper/2024/file/bddd4e769412635c62edd8e74a413e28-Paper-Conference.pdf (GLBW, ICLR 2024; snippet only, PDF not opened, OpenReview blocked by a browser check)
- https://openreview.net/forum?id=r1iCWpYhx4 (still UNVERIFIED: browser check; do not cite)

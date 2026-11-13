# Combined audit 2: Q1 readiness after the must-do fixes

Project: **A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers**

State audited: `main` at 80cd09f, 234 models. This report merges five independent read-only reviews in `docs/audit2/01`–`05`. The aggregation changed nothing outside `docs/audit2/` and committed nothing. The only extra check the aggregator made was a read-only look at `results/dose_response/models.csv` to settle a disagreement about SD2 dose placement (Section 7, item 1).

**Aggregate verdict: CHANGES_REQUESTED.** The must-do round fixed what it set out to fix, and no reviewer found a number in the paper that contradicts its CSV. The paper is still not ready to submit. Some of the new analyses (the behavioural switch, the behavioural labels, the over-detection rate) bring new problems of interpretation. The manuscript also still reads as a revision log, and the three external-validity gaps that need GPU time are all still open.

---

## 1. Reviewer status and scores

All five reports are present. Each gives a score now, a projected score and ranked recommendations with costs.

| # | Report | Lens | Score now | First audit | Projected (as stated by the reviewer) |
|---|---|---|---|---|---|
| 01 | `01_statistics.md` | Statistics and rigour | **7** | 5 | 8 after its R1–R3, R5–R7, R10 (≈ 5 days, ≈ 8 GPU h, includes SD2); 8.5 with the decorrelation experiment and more images per model |
| 02 | `02_literature.md` | Literature and novelty | **6.5** | 5 | 7 after positioning and bibliography (0 GPU); 7.5 with a second metric family (SD3+); 8 with SD1 and SD4 as well |
| 03 | `03_methodology.md` | Methodology and validity | **6.5** | 5 | 7 after its CSV-only fixes R1–R4; 7.5 with SD2 and SD1; 8 with SD5-style re-evaluation and SD4 |
| 04 | `04_venue_framing.md` | Venue and framing | **6.5** (Neurocomputing); 5.5 (Pattern Recognition) | 5 (PR 4) | 7.0 after writing-only fixes; 7.5 with SD1 and SD4; 8.0 with ROAD and SD2 as well. About 7 if the new settings do not replicate the thesis |
| 05 | `05_additions.md` | High-leverage additions | **6** | 5 | 7.5 after A0–A2, SD2, SD1, SD3, SD4 with working pilots; 7 if only one of SD1/SD4 reaches a usable reliance range |

### How the consensus is derived

- **Now.** The five scores are 7, 6.5, 6.5, 6.5 and 6. Median 6.5, mean 6.5, range 6–7. For reviewer 04 the Neurocomputing score is used, because Neurocomputing is the target. **Consensus now: 6.5 / 10.**
- **After the CPU-only must-do items (Section 5).** Reviewers 02, 03 and 04 each give 7. Reviewer 05's text-only gains (A0–A2, +0.45) give about 6.5. Reviewer 01 does not separate a CPU-only score; its 8 includes SD2. Median of the stated values: **7 / 10.**
- **After the full plan, including SD1–SD4.** 8, 8, 8, 8.0 and 7.5. Median **8 / 10**, mean 7.9. Reviewers 04 and 05 both warn that this assumes the pretrained and watermark pilots work and broadly replicate the thesis. If one of them fails, the realistic score is **7.5**.

### Against the first audit

| | Score |
|---|---|
| First audit (all five reviewers) | 5 / 10 |
| First audit's projection after MD1–MD8 | about 7 / 10 |
| Now, with MD1–MD8 done | **6.5 / 10** |
| Projected after this plan | **8 / 10** (7.5 if a pilot fails) |

The realised score is half a point below the first projection. The must-do items were done correctly, and every reviewer confirms that. The shortfall has two causes:

- The new reference quantities (behavioural switch, over-detection rate) are now in the abstract, and they carry problems of their own (Section 3, N1–N3).
- The manuscript still carries its own revision history: "earlier drafts", "the audit", "reviewer 03/05", internal file paths (N10).

Neither cause needs GPU time to fix.

## 2. What the reviewers agree is now strong

- **Numbers match their sources.** Reviewers 01, 02, 03 and 05 recomputed or cross-checked the headline numbers against the CSVs and found no contradiction. Reviewer 01 recomputed every headline count, ρ to within 0.0004, the over-detection and true-positive rates, the Pet pre-registration intervals, the tie rates and the confound table.
- **Statistics are implemented correctly where they were specified.** Reviewers 01 and 03 confirm each of the following:
  - chance-corrected reliance r_c;
  - balanced accuracy and FPR;
  - a two-level bootstrap paired across metrics;
  - BH within families and Holm over the headline tests (01 checked the Grad-CAM Holm step by hand);
  - isotonic switches with the ±10-point identification rule, bracketing and edge flags;
  - the exact Kendall null.

  Reviewer 01 accepts the correction in `06` of its first-audit tail probabilities.
- **The paper now has a Q1 target and one thesis.** It targets Neurocomputing (Q1, re-verified by widget). The single thesis is stated the same way in the introduction, Discussion and Conclusion, natural images come first, and the sanity-check study is in an appendix (04).
- **Related work is accurate and bounded.** All 54 references exist, and every statement about a cited work matches its abstract (02). Reviewer 02's search on 2026-11-13 found no paper that varies measured reliance to meta-evaluate faithfulness metrics, so the core novelty still holds.
- **Results that go against the thesis are reported.** These include the P3 failure, the A1 results that weaken earlier claims (Appendix C), the PneumoniaMNIST weak case and the Pet near-ties (04, 05).
- **The over-detection direction survives every robustness check the reviewers ran**, though at smaller size (N3). On natural images, every metric switch (1.3–17.7% r_c) lies well below the behavioural follow-rate crossing under every label. The paired bootstrap of (behavioural − metric) switch excludes 0 in all 12 natural-image cells (01).

## 3. First-audit weaknesses W-A to W-O: resolution status

Summary: **7 resolved** (W-E and W-F with new caveats), **4 partly resolved**, **4 not resolved**. All four unresolved items are disclosed in the Limitations. Three of them (W-J, W-K, W-L) need the planned GPU work. The fourth, W-G, needs a CPU experiment.

| ID | First-audit weakness | Status now | Evidence and what remains | Reviewers |
|---|---|---|---|---|
| W-A | Paper had no Pet or pre-registration; `N_MODELS == 174` assert | **Resolved** | 234 models (192 + 42); Pet first in every table; §5.3 with P1–P4 and commit hashes. Remaining: the manuscript now cites its own audit and drafts (N10) | 02, 03, 04 |
| W-B | Reliance floor equals the error rate | **Resolved** | r_c implemented correctly. The per-image follow rate agrees with r_c to within 1.7–4.4 points MAE, so the uniform-error assumption does no visible harm (03) | 01, 02, 03 |
| W-C | 50% ideal assumed; preference scored on aligned images, reliance on conflicting ones | **Partly resolved** | Per-image labels and a behavioural switch now exist. But the switch is not identified on CIFAR-10 (N1) and depends on the label (N2). The primary label measures sufficiency under conflict, while deletion measures necessity on aligned images, so the aligned-vs-conflicting concern is still open (03 N1) | 01, 03, 05 |
| W-D | Switch not identified in reliance gaps | **Partly resolved** | MNIST and PneumoniaMNIST gaps filled by A1; every MNIST-patch switch is bracketed. Still open: CIFAR CNN r_c 42.7→96.0, CIFAR ResNet-18 36.5→95.8, Pet 41.1→60.1. The behavioural switch sits in these gaps (N1) | 01, 03, 05 |
| W-E | No multiplicity control; seed-only bootstrap; no effect sizes | **Resolved, new caveats** | BH/Holm, two-level bootstrap and median AUC gaps are all present. New: percentile CIs for ρ are biased downward, and the bootstrap is nested although the design is crossed (N4) | 01, 03 |
| W-F | Unbalanced decision accuracy | **Resolved, new caveat** | Balanced accuracy, TPR and FPR are reported. New: the FPR depends on where the doses are placed and on the 0.5 threshold, and it has no interval (N3) | 01, 03, 05 |
| W-G | Reliance collinear with competence | **Not resolved (disclosed)** | `confound.csv` and Table 10 are correct but use raw reliance (N7). Collinear in 4 of 7 settings. No decorrelation experiment was run | 01, 03, 05 |
| W-H | Closest prior work missing; novelty too broad | **Resolved, new gaps** | BAM, SMERF, OSCAR, Bastings and the confound papers are cited; novelty is narrowed. New: over-detection is not linked to the method-level false positives in BAM and Adebayo 2022; two 2026 papers are missing (N11) | 02, 04, 05 |
| W-I | Q2 target, no single thesis, MNIST first | **Resolved** | Neurocomputing; one thesis; natural images first. Remaining: length, placeholders, unverified no-fee route (N12, N16) | 04 |
| W-J | One metric family; inpainting is not ROAD | **Not resolved (disclosed)** | Limitations, first bullet. SD3 not run. The fixed title says "Explanation Metrics", so this is the main literature objection (02) | 02, 03, 04, 05 |
| W-K | All models trained from scratch; CIFAR stem at 128 px | **Not resolved (disclosed)** | Pet 82% accuracy without the shortcut. SD1 not run | 02, 03, 04, 05 |
| W-L | Only shortcut on natural images is a white square | **Not resolved (disclosed)** | Every `cifar10*` and `pet_resnet18` row has `shortcut = patch`. SD4 not run | 02, 03, 04, 05 |
| W-M | "Object" oracle not an object on PneumoniaMNIST and CIFAR-10 | **Partly resolved** | Disclosed, and Pet is promoted to primary. But the CIFAR Gaussian prior still drives abstract numbers ("100% of CIFAR-10 models", "81%") (N3) | 03, 04 |
| W-N | AUC dominated by the random tail; blur leaves 20–25% of the patch | **Partly resolved** | Blur residual and inpaint end-point documented. The tail is not addressed (SD5 not run), and it is now what the low-reliance over-detection rests on (03 N2) | 03, 05 |
| W-O | "Inpainting most robust" by vote count | **Resolved** | Paired bootstrap in `metric_pairs.csv`; 18/126 pairs p_bh < 0.05; "most robust" removed | 01 |

## 4. New issues found (deduplicated)

Severity is the highest any reviewer assigned, unless noted. "Docx" is `paper/paper_draft.docx`, which is built by `paper/build_paper.py`, so every docx fix goes through the builder.

### Major

**N1. The behavioural switch fails the paper's own identification rule on CIFAR-10 and is not bracketed on Pet.** Raised by 01 N1, 03 N3 and 05 N1.
- *Where:*
  - Docx abstract ("54.5%"), §5.4 ("The assumed 50% is therefore about right for Pet, CIFAR-10 and MNIST patch") and Table 6.
  - README finding 3 and SYNOPSIS finding 3.
  - `behaviour_summary.csv`, where `observed_switch_rc_status` = "crossed" for every row.
  - `xai/dose_analysis.py`, where the rule is never applied to the behavioural switch.
- *Facts:*
  - CIFAR small CNN: switch 55.0, nearest model 12.3 points away.
  - CIFAR ResNet-18: switch 52.9, nearest model 16.4 points away. Its bootstrap interval of 45.0–57.3 hides a gap with no models between 36.5 and 95.8.
  - Pet: switch 54.5, with models at 41.1 and 60.1, so it is identified but not bracketed.
- *Fix:*
  - Apply the ±10-point and bracketing rule to the behavioural switch, and add a status column to Table 6.
  - Write CIFAR-10 as "> 43% (not identified)" and Pet as "54.5% (not bracketed)".
  - Delete "about right for … CIFAR-10".
  - Then run SD2.
- *Not affected:* the ordering "metric switch ≪ behavioural switch" on natural images. Every natural-image model with r_c ≤ 42.7 has a follow rate ≤ 0.43.

**N2. The behavioural results depend on the label, and the check labels are not reported.** Raised by 01 N2, 03 N1 and 05 N3.
- *Where:* docx §3.6, §5.4, Table 6 and the abstract. The check-label numbers appear only in `summary.md`.
- *Facts:*
  - The Pet behavioural switch is 54.5 under the primary (causal) label, 29.4 under the moved-only label and 49.6 under the dominance label. PneumoniaMNIST texture ranges from 34.1 to 62.3.
  - PneumoniaMNIST patch, pooled stratified AUROC: 0.46/0.50/0.65/0.41 under the primary label, against 0.90/0.92/0.94/0.91 under `shortcut_dominant`. The paper calls this "the weak case" (Holm p = 1.000).
  - The undetermined share is 33% on Pet and PneumoniaMNIST patch, up to 49% for one Pet model. It rises with reliance (Spearman 0.75–0.91).
  - On binary tasks, every label-1 image under the primary label is a redundancy image, where both cues are sufficient (03).
- *Why it is major:* the Limitations already concede that the causal label and the stratified AUROC were chosen after the original results were seen, and the commit order confirms it. A reviewer who opens `summary.md` will read the omission as selective reporting (03).
- *Fix:*
  - Add a twin table with all three labels: switch, pooled and within-model AUROC, undetermined share and its correlation with r_c.
  - Say which question each label answers: sufficiency under conflict versus necessity on aligned images.
  - Give the Pet switch as "29–55% depending on label", or "54.5% under the causal label".
  - Reword "PneumoniaMNIST patch is the weak case" as label-dependent.
  - Flag that `shortcut_dominant` is partly circular for black deletion on MNIST patch (03).

**N3. The over-detection headline overstates its size and partly measures the object oracle rather than the shortcut.** Raised by 01 N3, 03 N2/N5, 04 N2/N3 and 05 N2.
- *Where:* docx abstract, Highlight 2, §5.2, Tables 3 and 7; README finding 2; SYNOPSIS finding 2.
- *Facts:*
  - **Mechanism (03).** At the lowest dose, shortcut-vs-object (S–O) equals random-vs-object (R–O) within 0.07 in all 28 cells. Example: CIFAR CNN black, 0.815 / 0.815. Shortcut-vs-random (S–R) is 0.45–0.62. At low reliance, then, the metrics rate the shortcut pixels like random pixels, which is correct. The preference above 0.5 comes from the object map not beating random. On CIFAR-10 that map is a centred Gaussian. The "81%" figure is the small CNN only; ResNet-18 gives 72% (04).
  - **Threshold (03, 04).** Requiring a margin moves Pet black/blur FPR from 0.75 / 0.92 to 0.50 / 0.58. Both reviewers' margins (CI lower bound > 0.5, or preference > 0.6) give the same result. Pet inpaint falls to 0.17–0.25. CIFAR black/blur stays at 0.93–1.00.
  - **Design dependence (01).** At r_c ≤ 10, MNIST patch FPR is 0 for every metric, and Pet inpaint/insertion FPR is 0.17. The Pet inpaint/insertion over-detection comes from models at r_c 15–41 that follow the shortcut on 14–43% of images, so "barely use" does not fit those models.
  - **No interval (01).** Clopper–Pearson intervals: Pet black 0.75 [0.43, 0.95]. CIFAR TPR rests on 2–3 models (n_above = 2 and 3) (01, 05).
- *Reviewers agree* that the CIFAR black/blur over-detection is real at every threshold. The disagreement is about what it measures (Section 7, item 3).
- *Fix:*
  - Report excess shortcut preference, pref(S>O) − pref(R>O), per model, with its ρ against r_c. Put S–O, R–O and S–R side by side.
  - Report FPR with a margin, split at r_c ≤ 10 and 10–50, with intervals. Add `n_above`.
  - Abstract: "about half to three-quarters of Pet models" for black/blur, and "against a centred Gaussian prior, 72–81%" for CIFAR-10.
  - Highlight: "black and blur deletion over-detect shortcuts on natural images".
  - State the Pet R–O ≈ 0.5 finding directly: the metrics do not rank the real object mask above random pixels.

**N10. The manuscript cites its own audit, reviewers and earlier drafts.** Raised by 02 N3 and 04 N1.
- *Where:*
  - Docx §2, first paragraph ("docs/audit/02_literature_novelty.md").
  - Table 1 caption.
  - §3.6 ("after the audit's reviewer 05", "reviewer 03").
  - §5.1 ("Earlier drafts reported…"), §5.5 and §5.6.
  - Limitations ("chosen after the audit").
  - Appendix B "Legacy analysis".
  - Front-matter note citing `docs/audit/00_combined_audit.md`.
- *Fix:* add a submission mode to `build_paper.py` that removes these references. Keep one plain limitation saying that the analysis choices were made after the original-set results had been inspected. Move Appendix B to the repository.

**N11. Positioning gap on the main thesis.** Raised by 02 N1, N2 and N5.
- *Where:* docx §2.1, Table 1, first Discussion paragraph.
- *Facts:*
  - BAM [19] and Adebayo et al. 2022 [26] already report false positives at the attribution-*method* level. The paper cites both only for other points.
  - Missing: AIM (2026, masking-operator and random-attribution bias), "How to Evaluate and Refine your CAM" (ICPR 2026, CAM metrics against synthetic ground truth) and "From Flexibility to Manipulation". Reviewer 02 recommended the last one in the first audit, and it was not added.
  - Table 1 lists Miró-Nicolau 2026 [31] under metric meta-evaluation, although it evaluates XAI *methods*.
  - The blurry-mask mechanism rests on a single workshop paper [41].
- *Fix:* frame the finding as the metric-level counterpart of method-level false positives, shown with oracle maps so that no attribution method is involved. Add the three papers, move [31], and support [41] with ROAD and AIM.

### Moderate

**N4. The bootstrap intervals are wrong in two separate ways.** Raised by 01 N4 and 03 N4.
- *Where:* `xai/stats.py` (`two_level_picks`) and `xai/dose_analysis.py`; docx §3.7, and §5.3 for the P1-black sentence.
- *Too narrow on levels (03).* All models of a setting share the same 100 images, but images are resampled per model. A crossed (pigeonhole) resample widens the level intervals. PneumoniaMNIST patch black switch: width 13.5 → 61.9. Pet inpaint switch: 9.6 → 15.5. Low-dose preference: 20–50% wider. The ρ intervals barely change, and the MNIST-patch switch contrast survives (crossed upper bound 48.3).
- *Biased downward for ρ (01).* The median position of the estimate inside its CI is 0.73. The estimate lies above the upper bound in 20 of 196 cells, though in none of the 28 headline cells. Duplicated models create ties, and image noise attenuates the bootstrap correlations. p-values (Spearman t over models) and CIs disagree in 3 cells.
- *Fix:*
  - Use a crossed bootstrap for switch points and preferences.
  - Use BCa or bias-corrected intervals for ρ, with jitter to break the ties from duplicated models.
  - State that significance comes from the model-level test.
  - Re-word the P1-black "reaches below 0.8" sentence once the corrected CI is known.

**N5. "Earlier than behaviour" is claimed without a test.** Raised by 01 N7 and R6, and 03 N3.
- *Where:* docx §5.5.
- *Facts:* the paired bootstrap of (behavioural − metric) switch gives:
  - MNIST patch black [3.5, 28.9] and blur [4.0, 28.3]: earlier;
  - MNIST patch inpaint [0.1, 23.2]: borderline;
  - MNIST patch insertion [−2.7, 23.4]: not distinguishable;
  - PneumoniaMNIST patch inpaint [−9.6, 31.9]: not distinguishable.
- *Fix:* add this paired bootstrap to `behaviour_summary.csv`, and say "earlier" only where the CI excludes 0.

**N12. The paper is too long and table-heavy for a journal.** Raised by 04 N5.
- *Facts:* about 40–45 double-spaced pages. There are 13 tables. Table 5 has 27 rows, and one verdict string repeats on 16 of them. Pattern Recognition's 20–35-page limit comes from a search snippet and is unverified.
- *Fix:* shrink Table 5 to about 10 rows. Move Tables 7, 8, 12 and 13 to supplementary material.

**N8. Misclassified aligned images sit in the headline population.** Raised by 03 W10/N7.
- *Facts:* 20.7% of aligned images are misclassified in Pet models at r_c ≤ 20, 12.4% on PneumoniaMNIST and 12.8% on the CIFAR CNN. The paper does not mention it.
- *Fix:* add a robustness row on correctly classified images only, from `pred_aligned` and `scores_oracle.csv`. No new runs are needed.

**N9. The within-model AUROC has little power.** Raised by 03 N6, 01 R8 and 05 N5.
- *Facts:* the median minority-label class has 6–25 images per model, so each per-model AUROC has a standard error of about 0.1.
- *Fix:* state this next to "within a model only 0.43–0.80". Optionally, score 300 images per model on Pet and CIFAR ResNet-18.

### Minor

| ID | Issue | Where | Fix | Raised by |
|---|---|---|---|---|
| N6 | Calibration gap uses mismatched denominators (preference over 100 images, follow rate over determined images only). The natural-image range becomes 0.22–0.63 | Docx §5.4 | Compute on determined images | 01 N5 |
| N7 | Table 10 uses raw reliance, although "all primary analyses use r_c". On r_c, PneumoniaMNIST patch inpaint has partial ρ 0.24, p = 0.124. No multiplicity adjustment over 28 tests | Docx Table 10, §5.9 | Use `partial_rc_given_acc`, raw values in parentheses, BH within table | 01 N6 |
| N13 | Bibliography: LATEC track; Gevaert in *Machine Learning* 2024; DeGrave in *Nat. Mach. Intell.* 2021; Gomez at ICPRAI 2022; He at CVPR 2016; the Fresh-look Springer chapter; full authors for [14], [25], [30], [33]; [41] labelled as a workshop paper | Docx references | Apply the table in 02 §4. Check the Neurocomputing author-list style | 02 §4, N6 |
| N14 | SYNOPSIS misattributes metric meta-evaluation to AttributionLab | `SYNOPSIS.md`, Problem Statement | Cite Miró-Nicolau 2024 [34] instead | 02 N4 |
| N15 | Wording: §5.2 gives ρ 0.27–0.99 (should be 0.49–0.99 for the significant cells). Figs 3 and 7 use alphabetical order. "Within 1.0 points" should be "about 1 point". README supporting finding 2 heading still says "rewards the wrong methods". The one-sided AUROC choice is missing from Limitations. Abstract Pet switches lack caveats: 1.3% is an edge (e) value and blur has no switch | Docx §5.2, §5.5, Figs 3/7, abstract; `README.md` | Edit as listed. Consider leading with the CIFAR ResNet-18 inpaint/insertion switches (8.6–8.7%, bracketed) | 01 N7, 04 N4/N7 |
| N16 | Submission package incomplete: e-mail, guide name and data-availability placeholders; no cover letter; no Zenodo DOI; no-fee route unverified | Docx front matter and declarations | See M8 in Section 5 | 04, 05 |
| N17 | Pre-registration has only commit hashes, with no third-party timestamp. This matters for SD1–SD4 | `docs/PREDICTIONS_amendments.md` | OSF or Zenodo timestamp before any new run | 01 m2 |
| N18 | **Title rule.** `docs/SEMINAR_GUIDE.md` lines 1–4 use "Seminar Guide: Do Explanation Metrics Catch Shortcuts?" and "A Ground-Truth Evaluation of XAI Metrics for CNN Image Classifiers" | `docs/SEMINAR_GUIDE.md` L1–4 | Replace with the exact project title. Not changed here because this audit is read-only | 04 N6 |
| N19 | Hygiene: `size_matched` ties are not broken at random (W5), and there are no common random numbers across maps (W11) | `dose_response.py` | Fix inside any re-evaluation run | 03 |

## 5. Ranked remaining action plan

Costs are student working days with AI help. GPU h means Kaggle T4 hours. The ranges cover all reviewers' estimates.

### Must-do (before any submission; CPU only)

| # | Action | Fixes | Days | GPU h |
|---|---|---|---|---|
| M1 | **Behavioural switch: status, labels and tests.** Apply the identification and bracketing rule; add the three-label table with undetermined shares; add the paired bootstrap of (behavioural − metric) switch and a CI for the behavioural switch; reword §5.4, §5.5, the abstract, README and SYNOPSIS | N1, N2, N5 | 1.5 | 0 |
| M2 | **Over-detection reframe.** Excess shortcut preference and its ρ; S–O / R–O / S–R side by side; FPR with a margin and an r_c band split, with intervals; `n_above`; correct the abstract, highlight, README and SYNOPSIS | N3 | 1.5 | 0 |
| M3 | **Fix the intervals.** Crossed bootstrap for levels, BCa or bias-corrected intervals for ρ; state which test governs significance; re-run `dose_response.py analyze` (≈ 1 h CPU) | N4 | 1 | 0 |
| M4 | **Small analysis fixes from existing CSVs.** Calibration gap on determined images, Table 10 on r_c with BH, the correctly-classified robustness row, a power note on the within-model AUROC | N6, N7, N8, N9 | 0.5–1 | 0 |
| M5 | **Positioning and bibliography.** Link over-detection to BAM and Adebayo 2022; add AIM, the ICPR 2026 CAM paper and Flexibility-to-Manipulation; move [31]; apply the 10 reference corrections; fix SYNOPSIS | N11, N13, N14 | 1.5 | 0 |
| M6 | **Pre-register the GPU work before any of it runs.** Amendment A2 (SD2 doses, as corrected in Section 7 item 1), `PREDICTIONS_pretrained.md` with P5–P9 for SD1, SD3 and SD4 (05 lists candidates), the primary label for new models (03), and an OSF or Zenodo timestamp | N17; makes SD1/SD4 confirmatory | 0.5 | 0 |
| M7 | **Submission build.** A `build_paper.py` submission mode without audit, draft and reviewer references; Appendix B and Tables 7, 8, 12, 13 to supplementary; Table 5 cut to about 10 rows; wording fixes N15; title fix N18 | N10, N12, N15, N18 | 1.5–2 | 0 |
| M8 | **Submission package.** Browser check of the Neurocomputing no-fee route, abstract and highlight limits and reference style (10 min); fill the placeholders; Zenodo DOI for code, CSVs and checkpoints; cover letter naming the pre-registration commits | N16 | 1–1.5 | 0 |

Must-do total: about **9–10 days, 0 GPU h**. M6 must be committed before the first Kaggle session. M7 and M8 are done last, after the GPU results are in the paper.

### Should-do (Kaggle work, in this order)

| Order | Item | Why this position | Days | GPU h | Kaggle session |
|---|---|---|---|---|---|
| 1 | **SD2: fill the reliance gaps.** CIFAR ResNet-18: pilot one model at dose −0.05, then about 4 doses between −0.08 and −0.01 × 3 seeds (≈ 2.3 h). CIFAR small CNN: same doses × 2 seeds, on the laptop CPU (≈ 3 CPU h). Pet: dose −0.15 × 3 seeds (≈ 1.7 h), optionally −0.05 × 3 | Repairs a claim already in the abstract (N1, raised by three reviewers). Gives CIFAR TPR more than 2–3 models. Cheapest GPU item. If the CIFAR curve stays a step even at intermediate doses, report that as a finding | 1.5 | 4.5–8 | Session 1 |
| 2 | **SD1: ImageNet-pretrained ResNet-18 on Pet.** Standard stem, 128 px, about 10–15 epochs, 6 doses × 3 seeds. Pilot d = 0 and d = −0.5 in session 1. If reliance at d = 0 is below about 60%, add positive doses, recorded in A2 before the full run | Answers W-K, the most common referee request. It is the only item that turns the post-hoc headlines into confirmatory tests (03). It also gives SD4 its pipeline, and may partly decouple reliance from competence (05) | 3 | 3.5–11 (pilot settles it) | Pilot in session 1; full in session 2 |
| 3 | **SD4: realistic watermark on Pet, on the SD1 network.** Content-coded marks, one semi-transparent glyph per class in the same corner (opacity about 0.3–0.5, about 24×12 px); reliance measured by swapping the marks; oracle = glyph stroke mask. 2-model pilot first | Answers W-L. Content coding differs from the patch in shape, contrast and mechanism, where a position-coded mark would mostly repeat the patch (05). Run on the pretrained net so that SD1 isolates pretraining and SD4 isolates shortcut form. If the SD1 pilot fails, run SD4 from scratch instead (10–11 GPU h) and report SD1 as a negative pilot | 3–3.5 | 4–10 | Session 3 |
| 4 | **SD3: exact ROAD** (noisy linear imputation, MoRF and LeRF; official repo) on the oracle maps of the existing checkpoints. 28/32 px models on the laptop; CIFAR ResNet-18, Pet, and the SD1/SD4 models on Kaggle. Time one Pet model first, because the solve time is unverified | Answers W-J and tests the paper's leakage explanation. Uses no new training and most of it runs on the CPU, so it can be coded in parallel with sessions 1–3 and its Kaggle part comes last. Either outcome is publishable | 2–3 | 2–8 | Session 4 |
| 5 | **More images per model** (100 → 300) for Pet and CIFAR ResNet-18, evaluation only | Power for the within-model AUROC (N9) and the labels (N2) | 1 | 3–4 | Session 4, ride-along |
| 6 | **Decorrelation experiment on MNIST**, CPU only: vary shortcut salience or the shortcut-free share at a fixed dose; pre-register ρ > 0.8 and partial ρ > 0.5 | The only way to turn W-G from a limitation into evidence (01). Fits only if days remain | 2–3 | 0 (≈ 3 CPU h) | — |

Should-do GPU total: about **17–41 h**, central estimate about 22 h. That is within one to one and a half Kaggle weeks of the ~90 h available over 3 weeks. Days are the binding constraint, not GPU: must-do (9–10 days) plus SD1–SD4 (about 10 days) is about 19–20 days, against about 18 working days. Several CPU and GPU tasks run in parallel, and items 5 and 6 are first to drop.

### Optional

| # | Item | Source | Days | GPU h |
|---|---|---|---|---|
| O1 | SD5: matched-remainder maps, truncated first-k AUC, 100 or log-spaced steps on a subset (tests whether the Pet near-ties come from the 5% steps) | 03 R7, 05 SD5 | 1–3 | 1.5–6 |
| O2 | Faithfulness Correlation and Blücher's symmetric relevance gain as further metric families, alongside SD3 (watch for ties on binary maps) | 02 R3 | 1–2 | 1–3 |
| O3 | Metric verdicts on the five real attribution maps across reliance, to show the practical cost of over-detection | 05 N4 | 2 | ≈ 2 |
| O4 | MetaQuantus on a subset, compared with the dose-response ranking | 02 R6 | 3 | 2–4 |
| O5 | `validate_metric()` entry point and checklist; elsarticle LaTeX version | 04 R6, 05 SD6 | 1.5 | 0 |

### Three-week schedule

| Week | Laptop / CPU | Kaggle | GPU h |
|---|---|---|---|
| 1 | M6 first (days 1–2; commit and timestamp before any run). M1–M4 on the existing CSVs. SD2 small-CNN models on the laptop. Code SD1 and SD3 | Session 1: SD2 pilot, then CIFAR ResNet-18 and Pet gap models; SD1 pilot (d = 0, −0.5) | ≈ 5–9 |
| 2 | M5. Analyse SD2 and the SD1 pilot; finalise the SD1 doses in A2 if needed. Code SD4. ROAD on the 28/32 px models overnight | Session 2: SD1 full (18 models). Session 3: SD4 pilot, then full (18 models) | ≈ 8–20 |
| 3 | Re-run `analyze`; write the new §5 subsections and update the abstract and Limitations; M7 submission build; M8 package; final number check against the CSVs | Session 4: SD3 ROAD for CIFAR ResNet-18, Pet, SD1, SD4; 300-image re-evaluation if time allows | ≈ 5–12 |

**If time slips, drop in this order:** O1–O5, then item 6, then item 5, then the Kaggle part of SD3 (keep ROAD on the CPU settings and say so), then SD4. Do not drop M1–M8 or the M6 pre-registration. Without pre-registration, SD1 and SD4 lose most of their value as confirmatory tests (05).

## 6. Journal shortlist update

Quartiles were re-verified by reviewer 04 from SCImago widget images. The fee model comes from OpenAlex, which is a third-party source. The journals' own pages could not be opened. **The no-fee route is verified only for TNNLS**; for every other journal it is UNVERIFIED and needs a 10-minute check in a normal browser.

| Rank | Journal | Quartile, SJR 2025 | No-fee route | When to submit | Change from first audit |
|---|---|---|---|---|---|
| 1 | **Neurocomputing** (Elsevier) | Q1 AI, 1.47 | Hybrid per OpenAlex; UNVERIFIED | After M1–M8. Odds are better after SD1 and SD4. Reviewer 04 expects major revision rather than rejection once SD1 or SD4 is done | Unchanged |
| 2 | **Pattern Recognition** (Elsevier) | Q1 AI, 2.15 | Hybrid per OpenAlex; UNVERIFIED | Submit here first only if SD1 and SD4 are done and the paper fits about 35 double-spaced pages (snippet, UNVERIFIED). Current score from this lens: 5.5 | Moved up from 3 as a conditional option |
| 3 | **Machine Learning** (Springer) | Q1 AI, 1.03 | Hybrid per OpenAlex; UNVERIFIED | If Neurocomputing rejects on scope. Frame the paper as measurement validity, where the pre-registration counts most | Moved down from 2 |
| 4 | Neural Networks (Elsevier) | Q1 AI, 1.55 | UNVERIFIED | Fallback | Unchanged |
| 5 | CVIU (Elsevier) | Q1 CV&PR, 0.82 | UNVERIFIED | Fallback | Promoted from backup list |
| — | Avoid as first choice: IEEE TNNLS (verified mandatory $200/page beyond 10 pages, 15-page cap), Pattern Recognition Letters (letter format), DAMI, ESWA, KBS (scope) | | | | TNNLS dropped from the ranked list |

Submit to one journal at a time.

## 7. Where reviewers disagreed, and which view this report adopts

1. **SD2 value and dose placement.**
   - *Value:* reviewers 01, 03 and 05 rank SD2 high because it repairs N1. Reviewers 02 and 04 rank it low, because it adds no literature position or generality. **Adopted:** the *reporting* fix (M1) is must-do. SD2 itself goes first in the Kaggle order because it is the cheapest GPU item and supports a sentence already in the abstract, but it is should-do.
   - *Doses:* reviewers 01 and 03 propose CIFAR ResNet-18 doses −0.2, −0.15, −0.05 and Pet −0.25, −0.15, −0.05. Reviewer 05 proposes doses between −0.08 and −0.01 for CIFAR and −0.15 for Pet. The aggregator checked `models.csv`:
     - CIFAR ResNet-18 dose −0.1 gives r_c 31.8–36.5 and dose 0 gives 95.8–97.0, while −0.25 gives 16.1–17.5. The gap therefore lies between −0.1 and 0, and −0.2 or −0.15 would land below it.
     - Pet dose −0.2 gives r_c 22.9–41.1 and −0.1 gives 60.1–82.7. The gap lies between −0.2 and −0.1, so −0.15 is the right dose and −0.25 is not.

     **Reviewer 05's placement is adopted.**
2. **Which GPU item comes first.** Reviewer 01 says SD2. Reviewer 02 says SD3 (as part of a second metric family). Reviewers 03 and 04 say SD1. Reviewer 05 says SD2 by gain per day. **Adopted order: SD2 → SD1 → SD4 → SD3**, for these reasons:
   - SD2 is cheap and fixes a stated claim, and it shares session 1 with the SD1 pilot.
   - SD1 is the confirmatory test and a prerequisite for SD4 on the pretrained net.
   - SD3 needs no training, runs mostly on the CPU and can be coded in parallel, so its Kaggle part goes last without delaying anything.

   Reviewer 02's concern about breadth is real (the title says "Explanation Metrics"), so SD3 stays in should-do and is not dropped before SD4.
3. **What over-detection measures.** Reviewer 01 calls the CIFAR black/blur over-detection robust, since it holds even at r_c ≤ 10. Reviewers 03 and 04 say it mainly measures how the metric treats a centred Gaussian object map (S–O ≈ R–O). **Both are right, about different things:** the effect is robust to threshold and dose, and at low reliance it is driven by the object map. The report adopts 03's decomposition (excess shortcut preference) as the fix, because it states the finding at the strength the data support.
4. **Over-detection threshold.** Reviewer 03 uses the per-model CI lower bound > 0.5, reviewer 04 uses preference > 0.6, and reviewer 01 splits at r_c ≤ 10. The methods agree on Pet black/blur (0.50 / 0.58). **Adopted:** report the point FPR, one margin-based FPR (03's CI version, since it uses each model's own uncertainty) and 01's band split.
5. **Which behavioural label is primary.** Reviewer 01 reports the label sensitivity of the switch. Reviewer 03 argues that the primary label answers a different question from deletion and that `shortcut_dominant` reverses the PneumoniaMNIST verdict, while also flagging it as circular on MNIST patch. **Adopted:** neither label is ground truth for every metric. Report all three with what each measures (03), and name the primary label for new models in A2 before they are trained.
6. **Bootstrap problem.** Reviewer 01 finds downward bias in the ρ percentile CIs; reviewer 03 finds the nested scheme too narrow for level quantities. These are different defects and do not conflict. **Both fixes are adopted** (M3). Reviewer 03 states that its claim that a crossed bootstrap is "mildly conservative" is unverified background knowledge.
7. **Score now: 7 (01) against 6 (05).** Reviewer 01's lens is where the fixes landed most. Reviewer 05's lens counts the GPU additions, none of which are done. Both are consistent with their remits. **The median (6.5) is adopted**, which equals the mean.
8. **ROAD's priority.** Reviewer 02 ranks it highest and reviewer 03 lowest, since inpainting is already disclosed as not being ROAD. Reviewers 04 and 05 put it in the middle. **Adopted:** middle (4th of the Kaggle items), for the reasons in item 2.
9. **GPU estimates for SD1.** These range from 3.5 h (05, using a FLOP estimate for the standard stem) to 9–11 h (03). **Adopted:** the range; the pilot settles it.

## 8. Go / no-go

**No-go for submitting now.** Five things block it, and none needs GPU time:
- the manuscript cites its own audit and drafts (N10);
- two headline numbers in the abstract are stated beyond what the data support (N1 behavioural switch, N3 over-detection size);
- the check labels that reverse a reported verdict are not in the paper (N2);
- the switch intervals are too narrow (N4);
- the data-availability statement, the placeholders and the fee-route check are not done (N16).

A referee would find all of them.

**Recommendation: submit after the GPU work, at the end of week 3.**
- Do M1–M6 in week 1 and run SD2 and the SD1 pilot.
- Run SD1 and SD4 in week 2, and SD3 in week 3.
- Finish with M7–M8 and submit to Neurocomputing.

Expected readiness then is about 8 / 10 (7.5 if a pilot fails).

The minimum acceptable submission point is M1–M8 plus SD2 and SD1, at about 7.5. Submitting after the CPU-only fixes (about 7 / 10) is possible, but it is not recommended. GPU time is free and the user has said time is available, and both the literature and venue reviewers expect W-J, W-K and W-L to be the first objections a referee raises.

## 9. Sources

- Reviewer reports: `docs/audit2/01_statistics.md`, `02_literature.md`, `03_methodology.md`, `04_venue_framing.md`, `05_additions.md`. Their scratch scripts and outputs are in `docs/audit2/scratch/01_statistics/`, `02_literature/`, `03_methodology/`, `04_venue_framing/` and `05_additions/`.
- First audit: `docs/audit/00_combined_audit.md` and `docs/audit/06_must_do_resolution.md`.
- External URLs: each reviewer lists the pages it opened and marks unverified ones (01 §7, 02 §8, 03 §7, 04 §8, 05 §6). The aggregator opened no external pages.
- Aggregator check: `results/dose_response/models.csv`, read-only, for dose versus r_c on `cifar10_resnet18` and `pet_resnet18`.

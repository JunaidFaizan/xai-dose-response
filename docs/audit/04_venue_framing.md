# Audit 04: Venue strategy and framing

Project: *A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers*

Reviewer lens: venue strategy and framing (senior ML researcher). Read-only audit: no project file outside `docs/audit/` was changed.

**Web access note.** Web search and fetch worked, but bot protection or login redirects blocked the journal pages on ScienceDirect, SpringerLink, Scopus and the SCImago journal pages (HTTP 403, Cloudflare check, or a login redirect loop). Quartiles were read from SCImago's official public widget image (`journal_img.php?id=<Scopus source id>`). Each image shows the journal name, best quartile, best category and SJR 2025, and I opened all of them; the saved copies are in `docs/audit/scratch/04/*.png`. Publisher facts come from Wikipedia pages I opened and from one IEEE page. Where a fact comes only from a search-result snippet of a page I could not open, it is marked **UNVERIFIED**.

---

## 1. Summary verdict

**Q1 readiness: 5 / 10** (as a submission to a mid-tier Q1 journal such as Neurocomputing or Machine Learning). For Pattern Recognition or IEEE TNNLS the score is 4 / 10.

Justification:

- **Strong:** the experimental design. It has 192 models (`results/dose_response/models.csv`), reliance is measured behaviourally rather than assumed, the oracle controls are form-matched (placebo and random maps), seed-bootstrap intervals are reported, and the Pet extension was pre-registered before training (`docs/PREDICTIONS_pet.md`, commits `a962b6c` and `d823737`, both before `1e29918`). Few XAI meta-evaluation papers have a pre-registered test, and a journal reviewer will notice it.
- **Holding the score down (framing and venue):**
  1. The paper is aimed at a Q2 journal. `paper/build_paper.py` line 624 targets *SN Computer Science*, and SCImago rates it **Q2** (best category Computational Theory and Mathematics, SJR 2025 = 0.45).
  2. The paper draft predates the strongest evidence. `paper_draft.docx` was last written on 09-11-2026, before the Pet results. The abstract lists only "MNIST, PneumoniaMNIST and CIFAR-10". `build_paper.py` line 141 asserts `N_MODELS == 174`, but `models.csv` now has 192 rows, so the builder will stop at that assert if it is re-run unchanged. `README.md` still says Pet is "prepared but not trained yet".
  3. The contribution reads as a list of eight findings with no single thesis.
  4. Two closely related papers are not cited: OSCAR, which varies shortcut-label association strength, and Bastings et al., a shortcut-injection protocol for salience evaluation. The novelty sentence therefore needs sharper wording.
  5. Q1 CV/PR reviewers will see MNIST-scale images, synthetic shortcuts and no pretrained model as narrow scope (this is my reviewer-expectation judgement, not something I measured).

None of these needs new science. Points 1–4 are writing and positioning work for about 4–5 days. Point 5 can be reduced with one cheap Kaggle run (R5).

---

## 2. Strengths (as a reviewer at a Q1 journal would see them)

| # | Strength | Evidence |
|---|---|---|
| S1 | Tests the metrics, not the methods, against a controlled ground truth that varies in steps. Most prior work uses one fixed shortcut strength. | `docs/PAPER_OUTLINE.md` related-work table; `xai/data.py` signed dose |
| S2 | Pre-registered predictions, with an amendment recorded before any metric result was seen, and an honestly reported failure (P3 fails for black deletion: ρ = 0.75 vs random, 0.73 vs placebo). | `docs/PREDICTIONS_pet.md`, outcome table |
| S3 | Real segmentation masks on 128×128 photos (Pet). Reliance spans 17–99%. P1 holds: ρ = 0.85 / 0.96 / 0.95 / 0.93 for black / blur / inpaint / insertion. | `responsiveness.csv` rows 16–19 |
| S4 | Form-matched controls separate a genuine response from a form bias. On Pet, shortcut-vs-random rises from 0.45–0.53 to 0.76–0.99. Placebo-vs-random stays flat for deletion (ρ = 0.44, 0.20, −0.07). | `responsiveness.csv` rows 156–158, 184–187 |
| S5 | One finding replicates across architectures. On CIFAR-10, black/blur deletion prefer a random ordering over the object map 75–82% of the time (small CNN) and 56–72% (ResNet-18) at the lowest reliance. On Pet with real masks the effect shrinks to 0.49–0.53, which supports the Gaussian-object-map hypothesis. | README findings 6; `PREDICTIONS_pet.md` exploratory notes |
| S6 | Fully reproducible on free compute. Every number in the paper is generated from CSVs. | `paper/build_paper.py`, `kaggle/README.md` |

## 3. Weaknesses ranked by severity

### Blockers (fix before any Q1 submission)

**B1. The current target venue is Q2, and the paper is formatted for it.** `build_paper.py` line 624 says "Target: SN Computer Science (Springer)". SCImago widget for source 21101083109 shows Q2, SJR 0.45. A Q1 submission needs a new template, length and framing (see Section 4).

**B2. The paper does not contain the Pet study or the pre-registration.**
- The abstract and contributions (build_paper.py lines 634–700) cover three datasets and 174 models.
- `assert N_MODELS == 174` (line 141) contradicts `models.csv` (192 rows: 36 + 36 + 36 + 36 + 12 + 18 + 18).
- The Pet setting is the only one with real object masks. It is also the only pre-registered test, and it is the answer to the obvious reviewer question "does this hold beyond MNIST?". Leaving it out removes the strongest evidence.
- `README.md` ("Pending: Oxford-IIIT Pet") and `SYNOPSIS.md` are stale in the same way.

### Major

**M1. No single thesis.** The README lists 8 findings, and the outline lists 6 contributions. Q1 editors decide at desk review from the abstract and the first page. The current abstract strings together seven numeric results. In my judgement (speculation), a reviewer would summarise it as "several observations about deletion metrics". The findings already support one claim: *metrics respond to reliance (responsive) but are not calibrated to it, because their verdicts are confounded by explanation size and form.* The evidence:
- early switch points: 30–42% on MNIST against the full object map, and about 19–25% on Pet;
- the size-matched control moves the switch to 6–15%;
- the random-vs-object bias on CIFAR-10;
- form-matched controls are judged correctly.

**M2. The novelty claim does not cite two directly relevant papers.**
- **OSCAR** (Achara et al., Dec 2025, [arXiv:2512.18888](https://arxiv.org/abs/2512.18888), opened). It quantifies shortcut reliance from attribution maps on CelebA, CheXpert and ADNI, and reports that its correlations are sensitive to the *level of association* between shortcut and label in training data. A reviewer could read that as "graded shortcut strength, already done".
- **Bastings et al.**, "Will You Find These Shortcuts?" ([arXiv:2111.07367](https://arxiv.org/abs/2111.07367), opened). A protocol that injects shortcuts into partially synthetic text data to get ground truth for evaluating salience methods.
- The project's distinctions still hold: behaviourally measured reliance rather than training association; testing standard faithfulness metrics rather than proposing a new detector; images; and oracle maps independent of any method. The paper has to state these distinctions explicitly. The current sentence ("we did not find published work that varies a model's reliance… in controlled steps") is too broad.

**M3. Scope perception.** Of the 192 models, 144 are 28×28 grey images with a small CNN. All shortcuts are synthetic, and no model is pretrained.
- Pattern Recognition and TNNLS reviewers often expect at least one standard pretrained backbone. This is my experience-based expectation (speculation), not a published rule.
- Pet uses a ResNet-18 trained from scratch with 65–82% accuracy without the shortcut (`PREDICTIONS_pet.md`). A reviewer may say these are weak models.

**M4. The framing puts the weakest settings first.** The abstract opens with MNIST (ρ = 0.91–0.98) and only then mentions CIFAR. For a Q1 CV audience the order should be reversed: lead with natural images (CIFAR-10 ResNet-18, Pet ResNet-18), and present MNIST and PneumoniaMNIST as the controlled setting where the mechanism can be dissected (size-matching, the diffuse texture).

### Minor

- **m1. Title and keyword discoverability.** The field searches for "faithfulness metrics", "meta-evaluation" and "perturbation-based evaluation". The fixed title uses "Explanation Metrics". This is acceptable, and the keywords already include "Faithfulness" and "Meta-evaluation". See Section 5 for the flag.
- **m2. The main figure is a 7-panel strip** (`metric_dose_response.png`, 3913×507 px). In a single-column Elsevier or Springer layout it will print at roughly 2–3 cm high. `build_paper.py` already has a `rearrange_panels` function, so check that it produces a 2×4 grid, and add the Pet panel to the paper figure.
- **m3. The "1–2 page" supporting sanity-check study** dilutes the message. In a journal it belongs in an appendix or supplement.
- **m4. The single-author M.Tech affiliation** is fine for these journals. A cover letter that names the pre-registration and the public code helps editors (speculation based on common editorial practice).

---

## 4. Venue analysis

Requirement: Q1 in Scopus/SCImago, a no-fee subscription route, fully online, scope that covers AI / CV / PR / ML.

Quartile source for every row is the SCImago widget for the stated Scopus source id. I opened each widget image and it shows the journal name, so the id is confirmed. The SCImago journal page itself returned 403 for every journal; it is listed so the user can open it in a browser.

| Journal (publisher) | Homepage / author info (opened?) | SCImago (widget opened; page URL) | Quartile, SJR 2025 | No-fee route | Scope fit | Acceptance odds for this paper (speculative) |
|---|---|---|---|---|---|---|
| **Neurocomputing** (Elsevier) | [Wikipedia (opened)](https://en.wikipedia.org/wiki/Neurocomputing_(journal)); [journal home](https://www.sciencedirect.com/journal/neurocomputing) (login redirect, UNVERIFIED) | [widget id 24807](https://www.scimagojr.com/journal_img.php?id=24807); [page](https://www.scimagojr.com/journalsearch.php?q=24807&tip=sid) | Q1 (Artificial Intelligence), 1.47 | Elsevier hybrid, subscription route with no publication fee: **UNVERIFIED** for this journal (page blocked); Elsevier hybrid journals generally offer it | Good: empirical NN/ML studies, XAI papers common | Moderate–good |
| **Pattern Recognition** (Elsevier) | [Wikipedia (opened)](https://en.wikipedia.org/wiki/Pattern_Recognition_(journal)); [open-access options page](https://www.elsevier.com/journals/pattern-recognition/0031-3203/open-access-options) (redirect loop; search snippet says "No open access publication fee" for the subscription option, **UNVERIFIED**) | [widget id 24823](https://www.scimagojr.com/journal_img.php?id=24823); [page](https://www.scimagojr.com/journalsearch.php?q=24823&tip=sid) | Q1 (AI), 2.15 | Subscription route: snippet only, **UNVERIFIED** | Very good: CV and pattern recognition, evaluation methodology | Low–moderate; needs M3 fixed |
| **Machine Learning** (Springer) | [Wikipedia (opened)](https://en.wikipedia.org/wiki/Machine_Learning_(journal)); [how to publish](https://link.springer.com/journal/10994/how-to-publish-with-us) (bot check; snippet says OA requires an APC, which implies a non-OA option, **UNVERIFIED**) | [widget id 24775](https://www.scimagojr.com/journal_img.php?id=24775); [page](https://www.scimagojr.com/journalsearch.php?q=24775&tip=sid) | Q1 (AI), 1.03 | Hybrid with subscription route: **UNVERIFIED** (Springer page blocked) | Good: methodology and empirical evaluation; pre-registration fits ML-methods reviewers | Moderate |
| **Data Mining and Knowledge Discovery** (Springer) | [Wikipedia (opened)](https://en.wikipedia.org/wiki/Data_Mining_and_Knowledge_Discovery); [journal home](https://link.springer.com/journal/10618) (bot check, UNVERIFIED) | [widget id 13579](https://www.scimagojr.com/journal_img.php?id=13579); [page](https://www.scimagojr.com/journalsearch.php?q=13579&tip=sid) | Q1 (Computer Networks and Communications, best category), 1.2 | Hybrid: **UNVERIFIED** | Fair: publishes XAI evaluation work, but the image-CNN focus is off-centre | Moderate |
| **IEEE Trans. Neural Networks and Learning Systems** | [Wikipedia (opened; lists Open access: Hybrid)](https://en.wikipedia.org/wiki/IEEE_Transactions_on_Neural_Networks_and_Learning_Systems); [page charges (opened)](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-page-charges) | [widget id 21100235616](https://www.scimagojr.com/journal_img.php?id=21100235616); [page](https://www.scimagojr.com/journalsearch.php?q=21100235616&tip=sid) | Q1 (AI), 3.15 | **Verified:** the $110/page charge is voluntary. A mandatory $200/page applies beyond 10 pages for a full paper, so the route is free only if the paper is ≤10 IEEE pages | Good | Low: high bar, scale expectations |
| **Neural Networks** (Elsevier) | [Wikipedia (opened)](https://en.wikipedia.org/wiki/Neural_Networks_(journal)) | [widget id 24804](https://www.scimagojr.com/journal_img.php?id=24804); [page](https://www.scimagojr.com/journalsearch.php?q=24804&tip=sid) | Q1 (AI), 1.55 | Elsevier hybrid: **UNVERIFIED** | Good | Moderate–low |
| **Pattern Recognition Letters** (Elsevier) | [Wikipedia (opened)](https://en.wikipedia.org/wiki/Pattern_Recognition_Letters); [OA options](https://www.elsevier.com/journals/pattern-recognition-letters/0167-8655/open-access-options) (not opened, UNVERIFIED) | [widget id 24825](https://www.scimagojr.com/journal_img.php?id=24825); [page](https://www.scimagojr.com/journalsearch.php?q=24825&tip=sid) | Q1 (Computer Vision and Pattern Recognition), 0.79 | Elsevier hybrid: **UNVERIFIED** | Good scope, but its short-paper format would force cutting most controls | Moderate (as a letter) |
| **Computer Vision and Image Understanding** (Elsevier) | Wikipedia page 404; [journal home](https://www.sciencedirect.com/journal/computer-vision-and-image-understanding) (not opened, UNVERIFIED) | [widget id 24161](https://www.scimagojr.com/journal_img.php?id=24161); [page](https://www.scimagojr.com/journalsearch.php?q=24161&tip=sid) | Q1 (CV and PR), 0.82 | **UNVERIFIED** | Fair–good | Moderate |
| Expert Systems with Applications (Elsevier) | [Wikipedia (opened)](https://en.wikipedia.org/wiki/Expert_Systems_with_Applications) | [widget id 24201](https://www.scimagojr.com/journal_img.php?id=24201) | Q1 (AI), 1.94 | **UNVERIFIED** | Weak: application-oriented | Not recommended |
| Knowledge-Based Systems / Eng. Appl. of AI / Information Sciences (Elsevier) | Homepages not opened (UNVERIFIED) | widgets [24772](https://www.scimagojr.com/journal_img.php?id=24772), [24182](https://www.scimagojr.com/journal_img.php?id=24182), [15134](https://www.scimagojr.com/journal_img.php?id=15134) | Q1 (AI), 1.75 / 1.78 / 1.51 | **UNVERIFIED** | Weak–fair | Backups only |
| *Current target:* SN Computer Science (Springer) | [journal home](https://link.springer.com/journal/42979) (bot check, UNVERIFIED) | [widget id 21101083109](https://www.scimagojr.com/journal_img.php?id=21101083109) | **Q2** (Computational Theory and Mathematics), 0.45 | — | — | Does not meet the Q1 goal |

Before submitting, open each chosen journal's ScienceDirect or SpringerLink "Open access options" or "How to publish" page in a normal browser. Confirm two things: the subscription option says no publication fee, and the paper fits any length limits. This takes about 10 minutes and could not be done here.

### Recommended order

1. **Neurocomputing.** Best balance of scope fit and acceptance odds for an empirical meta-evaluation with ResNet-18 at small scale. Elsevier template; no strict length limit was confirmed (UNVERIFIED).
2. **Machine Learning (Springer).** Its methodology-minded readers value the pre-registration and the dose-response design more than ImageNet scale. Framing should stress "evaluation methodology for ML". Choose this if the paper is framed as a measurement-validity contribution.
3. **Pattern Recognition.** The strongest scope fit and highest prestige among realistic options, but a higher desk-reject risk unless R5 (a pretrained backbone) is done. Submit here first only if R5 is done, and fall back to option 1.

Avoid TNNLS unless the paper can be cut to 10 IEEE pages, because overlength pages are charged $200 each. Avoid ESWA, KBS and EAAI as first choices because of scope.

Sequential submission is required: one journal at a time (also in `PAPER_OUTLINE.md`). Q1 first decisions commonly take 1–4 months (speculation, not checked per journal), so the 3 weeks are for the submission package, not for acceptance.

---

## 5. Framing rewrite

### Title

The title stays as fixed by the project rules: **A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers**.

- Elsevier and Springer allow subtitles, but adding one makes this already long title (17 words) longer. I recommend **no subtitle**.
- If the user wants one anyway: *"…: Responsive but Not Calibrated"*.
- **Flag (do not change without the user):** "Faithfulness Metrics" in place of "Explanation Metrics" would match the field's search terms and make it clear that sanity checks or human-centred metrics are not the main object. This is the only title change I would argue for. The keywords already cover it, so it is optional.

### Thesis sentence (use in the abstract, intro and conclusion)

> Perturbation-based faithfulness metrics respond to how much a model relies on a shortcut, but they are not calibrated to it: their verdicts flip far too early and depend on the size and form of the explanations being compared.

### Proposed abstract (about 230 words; numbers from the current CSVs; check each one again with `build_paper.py` once it is updated)

> Deletion and insertion curves are the standard way to compare attribution methods, yet whether their verdicts follow what a model actually relies on has rarely been tested. We introduce a dose-response protocol for meta-evaluating faithfulness metrics: a signed training dose sets a CNN's behaviourally measured reliance on an injected shortcut from near 0% to near 100%, and fixed oracle explanations (shortcut, object, size-matched object, placebo and random maps) are scored at every dose, independently of any attribution method. Across 192 models on five settings, including ResNet-18 on CIFAR-10 and on Oxford-IIIT Pet with real segmentation masks, every metric's preference for the shortcut explanation rises with reliance (Spearman ρ up to 0.99). The metrics are responsive, but not calibrated. They switch to the shortcut explanation at about 20–40% reliance rather than 50%, and at 6–15% when the object explanation is matched in size to the shortcut. On CIFAR-10, black- and blur-fill deletion prefer a random pixel ordering over a compact object map 56–82% of the time when the model ignores the shortcut. Form-matched controls show that these errors come from comparing explanations of different size and form. In a pre-registered test on Pet, three of four predictions held; black-fill deletion failed to separate the real shortcut from form-matched controls (ρ = 0.73–0.75). Noisy-inpainting deletion was the only variant that responded significantly in every setting. We recommend validating a metric at several controlled reliance levels, with size- and form-matched controls, before using it to rank explanation methods.

Check before using:
- "every setting" for inpainting needs re-checking with Pet included. Pet inpaint ρ = 0.95–0.97 for the four main comparisons, but placebo-vs-random is ρ = −0.07 (that comparison is not expected to rise).
- "about 20–40%" combines MNIST 30–42% and Pet 19–25%. Use the exact CI ranges from `switch_ci_*`.

### Contribution list (replace the six-item list)

1. **Protocol.** A dose-response meta-evaluation of faithfulness metrics with behaviourally measured reliance and method-independent oracle maps (192 models, 5 settings, 2 architectures, pre-registered extension).
2. **Responsiveness ≠ calibration.** Metrics track reliance but switch too early. A size-matched control shows that part of this comes from explanation size.
3. **Form confound with a diagnostic.** Placebo and random controls separate a genuine response from a size or form bias. The bias is large with a compact Gaussian object map (CIFAR-10) and much smaller with real masks (Pet).
4. **Guidance.** Use inpainting deletion; avoid insertion for diffuse shortcuts; report metrics only with form-matched controls. Report the diverging attribution methods (Saliency, SmoothGrad and Grad-CAM move away from a diffuse shortcut) as a secondary result.

### Novelty sentence (replace the current broad claim)

> Prior work evaluates attribution *methods* against shortcuts of a fixed strength (Zhou et al.; BackX; Sun et al.; Bastings et al. for text), or derives shortcut-reliance scores from attributions and shows they vary with training-set association strength (OSCAR). We instead hold the explanations fixed and vary the model's *measured* reliance, in order to test the standard faithfulness *metrics* themselves.

---

## 6. Ranked recommendations

Costs: "days" means student working days with AI help. GPU hours are Kaggle T4 hours.

| Rank | Recommendation | Severity addressed | Cost |
|---|---|---|---|
| **R1** | **Bring the paper up to the 192-model state.** Add Pet to `SETTINGS`, change the `N_MODELS` assert, and add the Pet panel to the figures and Table 2. Add a short "Pre-registered test" subsection with the P1–P4 table from `PREDICTIONS_pet.md`. Update the stale README and SYNOPSIS "Pending" text. | B2 | 2 days, 0 GPU h |
| **R2** | **Re-target to Q1:** Neurocomputing first, or Pattern Recognition if R5 is done. Switch to the Elsevier `elsarticle` (or Springer `sn-jnl`) template, move the supporting sanity-check study to an appendix, and confirm the no-fee route in a browser. | B1, m3 | 1.5 days, 0 GPU h |
| **R3** | **Rewrite the abstract, introduction and contributions around the single thesis** ("responsive but not calibrated"; Section 5). Lead with the natural-image results, and present MNIST and PneumoniaMNIST as the mechanism-dissection setting. | M1, M4 | 1.5 days, 0 GPU h |
| **R4** | **Harden the novelty claim.** Cite and differentiate OSCAR (arXiv:2512.18888) and Bastings et al. (arXiv:2111.07367). Run the outline's own pending search ("graded spurious correlation", "meta-evaluation of faithfulness metrics", 2024+) and record the search date in the paper. | M2 | 1 day, 0 GPU h |
| **R5** | **Add one pretrained-backbone setting.** An ImageNet-pretrained ResNet-18 fine-tuned on Pet at the same 6 doses × 3 seeds, using the existing `pet_resnet18` pipeline. This answers the most predictable Q1 reviewer objection ("from-scratch models with 65–82% accuracy"). Estimate: from-scratch training took 23.2 min per model for 30 epochs; about 10 fine-tuning epochs is roughly 8–10 min per model, so about 3 GPU h for 18 models plus about 2 GPU h for evaluation. The pretrained-weights licence and the reliance range are unknowns; set doses after one pilot model and record the change as was done in the Pet amendment. | M3 | 3 days, about 5–6 GPU h (fits one Kaggle week) |
| **R6** | **Make the protocol a reusable artifact.** Tag a release and archive the code plus `results/dose_response/*.csv` on Zenodo with a DOI. Add a one-page "metric validation checklist" (test at ≥3 reliance levels, include size-matched and random controls, report switch points with CIs). Q1 journals in PR and ML value artifacts and data availability statements (speculation). | M1 | 1 day, 0 GPU h |
| **R7** | **Fix figures for print:** a 2×4 layout, the ideal-calibration line, and a Pet panel in the main figure. In Table 2, show the 95% bootstrap intervals for switch points. | m2 | 0.5 day, 0 GPU h |
| **R8** | **Cover letter and declarations.** Name the pre-registration and its commit hashes (`a962b6c` for the predictions, `d823737` for the amendment), the public repository, the AI-assistance disclosure required by the journal, and 3–5 suggested reviewers from the cited meta-evaluation papers. | m4 | 0.5 day, 0 GPU h |
| **R9** | **Decide on the title flag** ("Faithfulness" vs "Explanation"; Section 5) with the guide. This is the user's decision; do not change the title without it. | m1 | 0.1 day, 0 GPU h |

R1–R4 and R6–R8 take about 8 days with no GPU. With R5 the total is about 11 days and 6 GPU hours, which fits the 3-week window with time left for the guide's review.

---

## 7. Sources opened

- SCImago widgets (all opened as images; copies saved in `docs/audit/scratch/04/`): ids 24823, 24807, 24804, 24775, 13579, 21100235616, 24825, 24161, 24201, 24772, 24182, 15134, 21101083109, via `https://www.scimagojr.com/journal_img.php?id=<id>`. One wrong id I tried (27639) showed a different journal and was discarded.
- Wikipedia pages: [Pattern Recognition](https://en.wikipedia.org/wiki/Pattern_Recognition_(journal)), [Neural Networks](https://en.wikipedia.org/wiki/Neural_Networks_(journal)), [Machine Learning](https://en.wikipedia.org/wiki/Machine_Learning_(journal)), [DAMI](https://en.wikipedia.org/wiki/Data_Mining_and_Knowledge_Discovery), [TNNLS](https://en.wikipedia.org/wiki/IEEE_Transactions_on_Neural_Networks_and_Learning_Systems), [Neurocomputing](https://en.wikipedia.org/wiki/Neurocomputing_(journal)), [Pattern Recognition Letters](https://en.wikipedia.org/wiki/Pattern_Recognition_Letters), [ESWA](https://en.wikipedia.org/wiki/Expert_Systems_with_Applications).
- [IEEE CIS: TNNLS page charges](https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-page-charges).
- [arXiv:2512.18888 (OSCAR)](https://arxiv.org/abs/2512.18888); [arXiv:2111.07367 (Bastings et al.)](https://arxiv.org/abs/2111.07367).
- Not openable (403, bot check or login loop), so their content is UNVERIFIED: SCImago journal pages, Scopus source pages, ScienceDirect and Elsevier open-access-options pages, SpringerLink journal pages.

Content from these sources was paraphrased for compliance with licensing restrictions.

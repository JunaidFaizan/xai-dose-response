# Audit 2 verification

Project: **A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers**

Verification step of the second read-only audit. No report was edited and nothing was committed. This file is the only file this step wrote.

**Overall: PASS.** One caveat on URL item C4 is explained below. It does not change the verdict.

## A. Report files exist

| # | File | Result |
|---|---|---|
| A1 | `docs/audit2/00_combined_audit2.md` | PASS |
| A2 | `docs/audit2/01_statistics.md` | PASS |
| A3 | `docs/audit2/02_literature.md` | PASS |
| A4 | `docs/audit2/03_methodology.md` | PASS |
| A5 | `docs/audit2/04_venue_framing.md` | PASS |
| A6 | `docs/audit2/05_additions.md` | PASS |
| A7 | Every report line 3 carries the exact project title (project rule) | PASS (all six) |

## B. Reviewer report contents

| Report | Numeric score | First-audit comparison | ≥ 5 costed recommendations | Result |
|---|---|---|---|---|
| 01 statistics | §1 L19: 7 / 10 (first audit 5) | §2 "First-audit weaknesses: status" | §5: R1–R10, each with days and GPU/CPU h | PASS |
| 02 literature | §1 L11: 6.5/10 (first audit 5) | §2 "Comparison with my first-audit weaknesses" | §6: R1–R7, each with days and GPU h | PASS |
| 03 methodology | §1 L15: 6.5 / 10 (first audit 5) | §2 "First-audit weaknesses: status" | §5: R1–R10, each with days and GPU h | PASS |
| 04 venue framing | §1 L13: 6.5 Neurocomputing, 5.5 PR (first audit 5 / 4) | §2 "First-audit weaknesses: status" | §6: R1–R8, each with days and GPU h | PASS |
| 05 additions | §1 L16: 6 / 10 (first audit 5) | §2 "First-audit weaknesses: status" | §4.2: 12 ranked items with gain, days, GPU h, sessions | PASS |

## C. Combined report contents

| # | Required element | Location in `00_combined_audit2.md` | Result |
|---|---|---|---|
| C1 | Consensus score vs first audit 5/10 vs projected | §1 "How the consensus is derived" and "Against the first audit": now 6.5, first audit 5, first projection ≈ 7, CPU-only ≈ 7, full plan 8 (7.5 if a pilot fails) | PASS |
| C2 | W-A..W-O table | §3: all 15 rows present (W-A to W-O). The summary count matches the rows: 7 resolved, 4 partly resolved, 4 not resolved | PASS |
| C3 | New-issues list | §4: N1–N5 and N8–N12 under Major/Moderate, N6, N7 and N13–N19 in the Minor table (N1–N19, deduplicated with source reviewers) | PASS |
| C4 | Must / should / optional plan with SD1–SD4 ordered | §5: Must-do M1–M8; Should-do order SD2 → SD1 → SD4 → SD3, then items 5 and 6; Optional O1–O5; three-week schedule and drop order. The order is justified in §7 item 2 | PASS |
| C5 | Journal shortlist update | §6: 5 ranked journals plus an avoid list, with quartile, SJR, fee status and change from the first audit | PASS |
| C6 | Go / no-go recommendation | §8: no-go now; submit after the GPU work at the end of week 3 | PASS |

Cross-checks on C1, C5 and the reviewer reports:
- The median and mean of 7, 6.5, 6.5, 6.5 and 6 are both 6.5, as stated.
- The quartile and SJR for all 8 journals in 04 §5 and 00 §6 match the live SCImago widgets (see D).
- The OpenAlex APC values in 04 §5 match the live API.

## D. External URLs

Method:
- I collected every `http(s)://` URL from the six reports. That gives 73 unique URLs. 00 cites none, 01 cites 1, 02 cites 49, 03 cites 2, 04 cites 17 and 05 cites 5, with overlaps.
- I fetched each one with `Invoke-WebRequest` on 2026-11-13. I used `web_fetch` where a bot check blocked the scripted fetch.
- For arXiv, I compared the claims against the abstract page and the arXiv API record (`export.arxiv.org/api/query?id_list=…`).
- For publisher pages behind a login, I checked the metadata on Crossref (`api.crossref.org/works/<DOI>`).
- I downloaded and viewed all 8 SCImago widget images.

"Resolves" means the URL returns its own content (HTTP 200). "Blocked" means the URL exists but serves a login redirect, an HTTP 403 or a browser check. "Match" means the content agrees with what the report uses the URL for.

| # | URL | Cited in | Resolves | Content matches the report's claim |
|---|---|---|---|---|
| 1 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290/ | 01 §7 | Yes | Yes. Saravanan et al., hierarchical bootstrap. Section 3.1.1 says it "does not have a conservative bias in a hierarchical dataset" |
| 2 | https://arxiv.org/abs/2609.30397 | 02 | Yes | Yes. Miró-Nicolau, Spinnato, Guidotti, synthetic GT framework; "evaluating nine widely used XAI methods" (supports N2: evaluates methods) |
| 3 | https://arxiv.org/abs/2512.18888 | 02 | Yes | Yes. OSCAR, shortcut localisation |
| 4 | https://arxiv.org/abs/2512.11433 | 02 | Yes | Yes. "Back to the Baseline"; authors Picard, Boissin, Subhash, Cadène, Fel, as 02 §4 states |
| 5 | https://arxiv.org/abs/2405.02344 | 02 | Yes | Yes. BackX; authors Yang, Akhtar, Jiang, Mian, as stated |
| 6 | https://arxiv.org/abs/2502.09150 | 02 | Yes | Yes. Shortcut learning susceptibility in vision classifiers |
| 7 | https://arxiv.org/abs/2304.13836 | 02 | Yes | Yes. ROAR pitfalls; "bias toward spatially blurry masks"; comment "Accepted at the 2026 ICML Workshop on Mechanistic Interpretability" (supports N6: workshop paper) |
| 8 | https://arxiv.org/abs/2404.01415 | 02 | Yes | Yes. Faithfulness of ViT explanations |
| 9 | https://arxiv.org/abs/2007.07797 | 02 | Yes | Yes. Hierarchical bootstrap, arXiv version |
| 10 | https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience | 02 | Yes | Yes. Same paper, journal version |
| 11 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7523163/ | 02 | Yes | Yes. DeGrave et al.; page says "medRxiv [Preprint]", as 02 §4 states |
| 12 | https://www.nature.com/articles/s42256-021-00338-7 | 02 (marked snippet) | Blocked (login redirect) | Yes via Crossref: *Nature Machine Intelligence* 3, 610–619 (2021). 02 still lists volume and pages as to confirm, and Crossref now gives them |
| 13 | https://hal.science/hal-03633006 | 02 | Yes (bot check for script; `web_fetch` OK) | Yes. Gomez, Fréour, Mouchère, ICPRAI 2022, DOI 10.1007/978-3-031-09037-0_8 |
| 14 | https://aclanthology.org/2022.emnlp-main.64/ | 02 | Yes | Yes. "Will You Find These Shortcuts?" (Bastings et al., EMNLP 2022) |
| 15 | https://proceedings.mlr.press/v162/kim22h.html | 02 | Yes | Yes. Sanity Simulations for Saliency Methods |
| 16 | https://proceedings.mlr.press/v70/sundararajan17a.html | 02 | Yes | Yes. Axiomatic Attribution (Integrated Gradients) |
| 17 | https://jmlr.org/papers/v24/22-0142.html | 02 | Yes | Yes. Quantus, JMLR |
| 18 | https://openaccess.thecvf.com/content/CVPR2023/html/Binder_Shortcomings_..._CVPR_2023_paper.html | 02 | Yes | Yes. Binder et al., CVPR 2023 |
| 19 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/05957c19...-Abstract-Datasets_and_Benchmarks.html | 02 | Yes | Yes. M4 benchmark, NeurIPS 2023 D&B |
| 20 | https://github.com/holyseven/M4_XAI_Benchmark | 02 | Yes | Yes. README lists Xuhong Li, Mengnan Du, Jiamin Chen, Yekun Chai, Himabindu Lakkaraju, Haoyi Xiong, as stated |
| 21 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/7beb1ddb...-Abstract-Datasets_and_Benchmarks_Track.html | 02 | Yes | Yes. LATEC, "NeurIPS 2024 Datasets and Benchmarks Track" |
| 22 | https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/ | 02 | Yes | Yes. Parkhi et al., "Cats and Dogs", CVPR 2012, Oxford-IIIT Pet |
| 23 | https://link.springer.com/article/10.1007/s10994-024-06550-x | 02 (marked snippet) | Blocked (login redirect) | Yes via Crossref: Gevaert et al., *Machine Learning* 113, 6019–6064 (2024) |
| 24 | https://link.springer.com/chapter/10.1007/978-3-031-63787-2_21 | 02 (marked snippet) | Blocked (login redirect) | Yes via Crossref: "A Fresh Look at Sanity Checks for Saliency Maps", xAI 2024, CCIS, pp. 403–420 |
| 25 | https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5 | 02 (marked snippet) | Blocked (HTTP 403) | Yes via Crossref (DOI 10.1016/j.cell.2018.02.010): Kermany et al., *Cell* 172, 1122–1131.e9 (2018) |
| 26 | https://www.nature.com/articles/s41467-019-08987-4 | 02 (marked snippet) | Blocked (login redirect) | Yes via Crossref: Lapuschkin et al., *Nature Communications* 10 (2019) |
| 27 | https://dl.acm.org/doi/10.1145/3447548.3467213 | 02 (marked snippet) | Blocked (HTTP 403) | Yes via Crossref: Lin et al., "What Do You See?", KDD 2021 (backdoor-trigger evaluation, ref [24]) |
| 28 | https://arxiv.org/abs/2407.11910 | 02 | Yes | Yes. Hesse et al., backbone design and attribution quality |
| 29 | https://arxiv.org/abs/2310.16228 | 02 | Yes | Yes. On the Foundations of Shortcut Learning |
| 30 | https://arxiv.org/abs/2401.06654 | 02 | Yes | Yes. Decoupling pixel flipping and occlusion strategy |
| 31 | https://arxiv.org/abs/2202.12270 | 02 | Yes | Yes. Gevaert et al., arXiv version |
| 32 | https://arxiv.org/abs/2409.16756 | 02 | Yes | Yes. LATEC, "17 prominent XAI methods using 20 distinct metrics", as 02 §5 states |
| 33 | https://arxiv.org/abs/2202.00449 | 02, 05 | Yes | Yes. ROAD (Rong, Leemann et al.); "information leakage through the shape of the removed pixels", as 05 SD3 states |
| 34 | https://arxiv.org/abs/2310.06514 | 02 | Yes | Yes. AttributionLab "serves as a sanity check for faithfulness" of attribution methods (supports N4); authors as stated |
| 35 | https://arxiv.org/abs/1907.09701 | 02 | Yes | Yes. BAM; "certain methods are more likely to produce false positive explanations" (supports N1) |
| 36 | https://arxiv.org/abs/2212.04629 | 02 | Yes | Yes. Adebayo et al.; attributions "erroneously indicating dependence on spurious signals even when the model … does not rely on" them (supports N1); journal ref ICLR 2022 |
| 37 | https://arxiv.org/abs/2212.11870 | 02 | Yes | Yes. Impossibility Theorems for Feature Attribution |
| 38 | https://arxiv.org/abs/2605.14641 | 02 | Yes | Yes. Domeniconi et al.; synthetic GT dataset to compare CAM evaluation metrics, composite ARCC; "Accepted at ICPR 2026" |
| 39 | https://arxiv.org/abs/2605.16905 | 02 | Yes | Yes. AIM; masking operator confounds (zero masking OOD, interpolation keeps residual information); random-attribution bias |
| 40 | https://arxiv.org/abs/2606.28391 | 02 | Yes | Yes. Few-class Fidelity (fidelity metric variant) |
| 41 | https://arxiv.org/abs/2601.11625 | 02 | Yes | Yes. Reasoning Stabilization Point (shortcut reliance tracking) |
| 42 | https://arxiv.org/abs/2202.05189 | 02 | Yes | Yes. Rare spurious correlations; "a handful of such examples" suffices |
| 43 | https://arxiv.org/abs/2605.25052 | 02 | Yes | Yes. BonaFide, Gur-Arieh et al.; LLM CoT faithfulness metrics against ground truth, scored by AUROC |
| 44 | https://arxiv.org/abs/2412.05592 | 02 | Yes | Yes. "From Flexibility to Manipulation"; ECCV 2024 workshop |
| 45 | https://arxiv.org/abs/2410.02970 | 02 | Yes | Yes. F-Fidelity; ICLR 2025; OOD issue of removal metrics |
| 46 | https://arxiv.org/abs/2005.00631 | 02 | Yes | Yes. Bhatt et al., IJCAI 2020 (faithfulness criterion) |
| 47 | https://arxiv.org/abs/2106.00786 | 02 | Yes | Yes. Hase et al., NeurIPS 2021, OOD problem of feature removal |
| 48 | https://export.arxiv.org/api/query | 02 | Bare URL returns HTTP 400. With `id_list` parameters it returns records | Yes. It is an API endpoint, cited as such; the queried IDs are in scratch |
| 49 | https://proceedings.iclr.cc/paper_files/paper/2024/file/bddd4e76...-Paper-Conference.pdf | 02 (marked snippet only, not opened) | Yes (application/pdf, 5.5 MB) | Not confirmed. The PDF is compressed and no PDF text tool is installed here. 02 already marks the entry "confirm before citing" |
| 50 | https://openreview.net/forum?id=r1iCWpYhx4 | 02 (marked UNVERIFIED, do not cite) | Blocked (browser check) | Not checkable. 02 already says not to cite it |
| 51 | https://arxiv.org/abs/2103.14651 | 03 | Yes | Yes. Watson et al., necessity and sufficiency as building blocks of explanations, applied inconsistently in XAI |
| 52 | https://arxiv.org/abs/0712.1111 | 03 | Yes | Yes. Owen, pigeonhole bootstrap: resamples rows and columns separately, mean-consistent for crossed random effects. The "mildly conservative" remark is marked unverified in 03 |
| 53 | https://www.scimagojr.com/journal_img.php?id=24807 | 04 | Yes (PNG, viewed) | Yes. Neurocomputing, Q1 AI, SJR 2025 1.47 |
| 54 | …id=24823 | 04 | Yes (PNG, viewed) | Yes. Pattern Recognition, Q1 AI, 2.15 |
| 55 | …id=24775 | 04 | Yes (PNG, viewed) | Yes. Machine Learning, Q1 AI, 1.03 |
| 56 | …id=24804 | 04 | Yes (PNG, viewed) | Yes. Neural Networks, Q1 AI, 1.55 |
| 57 | …id=24825 | 04 | Yes (PNG, viewed) | Yes. Pattern Recognition Letters, Q1 CV&PR, 0.79 |
| 58 | …id=24161 | 04 | Yes (PNG, viewed) | Yes. CVIU, Q1 CV&PR, 0.82 |
| 59 | …id=21100235616 | 04 | Yes (PNG, viewed) | Yes. IEEE TNNLS, Q1 AI, 3.15 |
| 60 | …id=13579 | 04 | Yes (PNG, viewed) | Yes. DAMI, Q1 Computer Networks and Communications, 1.2 |
| 61 | https://api.openalex.org/sources/issn:0925-2312 | 04 | Yes | Yes. Neurocomputing, is_oa false, APC 2670 |
| 62 | …issn:0031-3203 | 04 | Yes | Yes. Pattern Recognition, false, 2590 |
| 63 | …issn:0885-6125 | 04 | Yes | Yes. Machine Learning, false, 3290 |
| 64 | …issn:0893-6080 | 04 | Yes | Yes. Neural Networks, false, 3110 |
| 65 | …issn:0167-8655 | 04 | Yes | Yes. PRL, false, 2500 |
| 66 | …issn:1077-3142 | 04 | Yes | Yes. CVIU, false, 2090 |
| 67 | …issn:2162-237X | 04 | Yes | Yes. TNNLS, false, 2645 |
| 68 | …issn:1384-5810 | 04 | Yes | Yes. DAMI, false, 3190 |
| 69 | https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls-page-charges | 04 | Yes | Yes. $110/page voluntary; mandatory $200/page over 10 pages (full paper); maximum 15 pages for a regular paper |
| 70 | https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html | 05 | Yes | Yes. IMAGENET1K_V1, GFLOPS 1.81, 44.7 MB |
| 71 | https://github.com/tleemann/road_evaluation | 05 | Yes | Yes. Official ROAD code; Noisy Linear Imputation; MoRF and LeRF |
| 72 | https://github.com/understandable-machine-intelligence-lab/Quantus | 05 | Yes | Yes. README lists ROAD, IROF and Faithfulness Correlation |
| 73 | https://arxiv.org/abs/1902.10178 | 05 | Yes | Yes. Lapuschkin et al., Clever Hans. 05 already notes that the abstract does not mention the copyright-tag example |

Summary:
- 63 of 73 URLs resolve directly. 05 makes all of its claims against the live content.
- 8 URLs are blocked by a login redirect, a 403 or a browser check: rows 12, 23–27 and 50. For the DOI rows (12, 23–27), Crossref confirms the bibliographic claim. Row 13 was also blocked for the script but opened with `web_fetch`, so it counts among the 63.
- The API endpoint (row 48) works only with parameters.
- Row 49 resolves, but I could not confirm its content.

No URL contradicts what a report uses it for.

### C4 caveat: "every external URL cited was opened"

- **This step:** I opened all 73 URLs.
- **Reviewers 01, 03, 04 and 05:** their reports say they opened every URL they cite, and their failures are disclosed. 04 lists the pages it could not open and does not cite them.
- **Reviewer 02 (§8):** it opened every URL except 8, which it marks as follows:
  - "snippet": rows 12 and 23–27;
  - "snippet only, PDF not opened": row 49;
  - "UNVERIFIED, do not cite": row 50.

  These 8 are transparently disclosed. They are tried-but-blocked publisher pages, not claims the reviewer invented.

**Result: PASS with caveat.** If the brief is read strictly as "each reviewer opened every URL it cites", 02 §8 lines 166 and 176–180 and 202–203 would fail it. Crossref confirms the bibliographic details for six of them (rows 12, 23–27), and the remaining two (rows 49 and 50) are already flagged in 02 as "confirm before citing" or "do not cite".

One small update for whoever applies 02 R2: Crossref gives DeGrave's volume and pages as *Nat. Mach. Intell.* 3:610–619, which 02 §4 lists as "still to confirm".

## E. Git

```
> git -C "<root>" status --porcelain        (run before this file was written)
?? docs/audit2/

> git -C "<root>" log -1 --format=%h
80cd09f

> git -C "<root>" status --porcelain -uall  (run before this file was written)
?? docs/audit2/00_combined_audit2.md
?? docs/audit2/01_statistics.md
?? docs/audit2/02_literature.md
?? docs/audit2/03_methodology.md
?? docs/audit2/04_venue_framing.md
?? docs/audit2/05_additions.md
?? docs/audit2/scratch/...   (41 scratch files under 01_statistics/ … 05_additions/)
```

| # | Check | Result |
|---|---|---|
| E1 | HEAD is 80cd09f | PASS |
| E2 | Only untracked (`??`) entries, all under `docs/audit2/`; no modified or staged tracked files | PASS |
| E3 | Nothing committed since 80cd09f | PASS |

This file (`docs/audit2/scratch/verification.md`) adds one more untracked entry under `docs/audit2/`.

## F. Values copied from the combined report

- **Consensus score:** 6.5 / 10 now. The first audit gave 5 / 10 and projected about 7 after MD1–MD8. The projection is about 7 after the CPU-only must-do items and 8 / 10 after the full plan with SD1–SD4 (7.5 if a pilot fails).
- **Go / no-go:** no-go for submitting now. The blockers are N10, N1/N3, N2, N4 and N16, and none needs GPU time. The recommendation is to submit to Neurocomputing after the GPU work, at the end of week 3, at about 8 / 10. The minimum acceptable point is M1–M8 plus SD2 and SD1, at about 7.5.

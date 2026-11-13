# Citation check for paper/paper_draft.docx (54 references), 2026-11-13

Source of the reference list: `paper_text.txt` in this folder (extracted from the docx with python-docx).
arXiv metadata (titles, full author lists, comments, journal refs) was pulled from the arXiv API and saved to `arxiv_api.txt`, `arxiv_api_jref.txt` and `arxiv_api_classics.txt` in this folder.

Legend. Opened: O = page opened by me (abstract page, proceedings page, journal or repository page, or the arXiv API record); S = confirmed only from a search-result snippet, because the page redirected to a login or browser check.
Entry: OK = title, authors, year and venue match; FIX = something to correct or complete. Claim: what the paper says about the work, checked against the abstract.

| # | Short | Opened | Entry | Paper's claim | Notes |
|---|---|---|---|---|---|
| 1 | Simonyan 2014 | O (arXiv API 1312.6034) | OK | Saliency method: accurate | |
| 2 | SmoothGrad | O (API 1706.03825) | OK | accurate | Full authors: Smilkov, Thorat, Kim, Viégas, Wattenberg |
| 3 | Sundararajan 2017 | O (PMLR v70) | OK | accurate | Add PMLR 70:3319–3328 |
| 4 | Grad-CAM | O (API 1610.02391) | OK | accurate | ICCV 2017 (extended version: IJCV 2019) |
| 5 | Zeiler & Fergus | O (API 1311.2901) | OK | Occlusion: accurate | |
| 6 | Samek 2017 | O (API 1509.06321) + S (TNNLS PDF) | OK | Deletion: accurate | Published author list differs from arXiv v1 (Lapuschkin instead of Bach); "et al." avoids the issue |
| 7 | RISE | O (API 1806.07421) | OK | Insertion from a blurred image: accurate | |
| 8 | Tomsett 2020 | O (API 1912.01451) | OK | "statistically unreliable": accurate | AAAI-20 confirmed in arXiv comment |
| 9 | MetaQuantus | O (API, journal ref TMLR 2023) | OK | accurate | |
| 10 | LATEC | O (arXiv + NeurIPS proceedings page) | FIX | 17 methods, 20 metrics, conflicting rankings: accurate | Venue is the NeurIPS 2024 **Datasets and Benchmarks Track** (proceedings URL) |
| 11 | Gevaert | O (arXiv) + S (Springer) | FIX | metrics measure different things; results do not transfer across datasets: accurate | Published in *Machine Learning* (Springer) 2024, DOI 10.1007/s10994-024-06550-x (Springer page redirected to login; DOI from search result) |
| 12 | ROAD | O (arXiv) | OK | leakage through the shape of the removed region: accurate | |
| 13 | Blücher 2024 | O (arXiv, TMLR journal ref) | OK | occlusion strategy decides PF rankings: accurate | |
| 14 | Back to the Baseline | O (arXiv + API) | FIX | baseline favours some methods: accurate | Authors now known: Picard, Boissin, Subhash, Cadène, Fel (2025) |
| 15 | Ross 2017 | O (API 1703.03717) | OK | Decoy MNIST: accurate | |
| 16 | Lapuschkin 2019 | O (API) + S (Nat Commun 10:1096) | OK | accurate | add 10:1096 |
| 17 | DeGrave | O (PMC7523163) + S (nature.com s42256-021-00338-7) | FIX | acquisition shortcuts: accurate | Final venue: *Nature Machine Intelligence* (2021), article s42256-021-00338-7. PMC7523163 is the medRxiv preprint. Volume and pages not verified (page redirected) |
| 18 | Sun 2023 | O (API, "Accepted to MICCAI 2023") | OK | accurate | Authors: Sun, Koch, Baumgartner |
| 19 | BAM | O (arXiv) | OK | "models trained with known relative feature importance": accurate | BAM's abstract also reports **false-positive explanations** for some methods; the paper does not use this link (see report N1) |
| 20 | SMERF | O (PMLR v162) | OK | "controls the complexity of the model's reasoning": accurate | |
| 21 | Bastings 2022 | O (ACL Anthology) | OK | accurate, pp 976–991 | |
| 22 | OSCAR | O (arXiv + API) | OK | sensitive to the strength of shortcut association: accurate | |
| 23 | Zhou 2022 | O (API, AAAI 2022) | OK | accurate | Authors: Zhou, Booth, Ribeiro, Shah |
| 24 | Lin 2021 | O (API) + S (ACM DL DOI 10.1145/3447548.3467213) | OK | backdoor triggers: accurate | |
| 25 | BackX | O (arXiv + API) | FIX | backdoor triggers: accurate | Authors now known: Yang P, Akhtar N, Jiang J, Mian A |
| 26 | Adebayo 2022 | O (arXiv, journal ref "ICLR 2022 conference paper") | OK | "ask whether explanations reveal spurious correlations": accurate but incomplete | Abstract reports attributions that **indicate spurious dependence when the model does not rely on it**, the method-level analogue of the paper's over-detection (report N1) |
| 27 | CLEVR-XAI | O (API, DOI 10.1016/j.inffus.2021.11.008) | OK | relevance mass accuracy: accurate | add Information Fusion 81 (2022) after checking the volume |
| 28 | FunnyBirds | O (API) | OK | accurate | ICCV 2023 |
| 29 | XAI-Units | O (API, DOI 10.1145/3715275.3732186) | OK | accurate | |
| 30 | AttributionLab | O (arXiv + API) | FIX | known important inputs by design: accurate | Authors now known: Zhang Y, Li Y, Brown H, Rezaei M, Bischl B, Torr P, Khakzar A, Kawaguchi K |
| 31 | Miró-Nicolau 2026 | O (arXiv + API, v1 24 Sep 2026) | OK in 2.1 / FIX in Table 1 | 2.1: "synthetic intervention-based framework": accurate | Table 1 lists it under "Meta-evaluation of faithfulness metrics". The abstract says it evaluates nine XAI **methods** (binary images, tabular, time series) and criticises fidelity scoring; it does not meta-evaluate metrics against ground truth |
| 32 | Quantus | O (JMLR page) | OK | accurate, 24(34):1–11 | author order matches JMLR |
| 33 | M4 | O (NeurIPS 2023 D&B page; GitHub README) | FIX | across modalities and networks: accurate | Authors (from repository README and NSF PAR record): Li X, Du M, Chen J, Chai Y, Lakkaraju H, Xiong H |
| 34 | Miró-Nicolau 2024 | O (API) + S (abstract) | OK | decision tree, 52,000 synthetic 128×128 images: accurate | No published version found |
| 35 | Hesse 2024 | O (arXiv) | OK | protocol avoids the OOD issue of incremental deletion: accurate | |
| 36 | Adebayo 2018 | O (API 1810.03292) | OK | accurate | |
| 37 | Binder 2023 | O (CVF open access) | OK | accurate | add pp 16143–16152 |
| 38 | Sanity checks revisited | O (API, "NeurIPS XAIA 2023") | OK | accurate | |
| 39 | Fresh look | O (API) + S (Springer chapter) | FIX | accurate | Published in xAI 2024 (Springer CCIS), DOI 10.1007/978-3-031-63787-2_21, from search result. Cite the published chapter, and consider dropping [38] or [39] as duplicates |
| 40 | ROAR | O (API 1806.10758) | OK | accurate | NeurIPS 2019 |
| 41 | ROAR pitfalls | O (arXiv v5) | OK | bias towards blurry masks, also seen in ROAD: accurate | Workshop paper (ICML 2026 MI workshop); state that it is a workshop paper |
| 42 | Gomez 2022 | O (arXiv API + HAL) | FIX | ranking only; OOD images: accurate | Published at ICPRAI 2022 (LNCS), DOI 10.1007/978-3-031-09037-0_8 (HAL record) |
| 43 | SaCo | O (arXiv) | OK | metrics struggle to separate methods from random attribution: accurate | |
| 44 | Bilodeau 2024 | O (arXiv, PNAS journal ref) | OK | end-task and repeated model evaluations: accurate | |
| 45 | Hermann 2024 | O (arXiv, ICLR 2024) | OK | predictivity and availability: accurate | |
| 46 | Suhail 2025 | O (arXiv) | OK | positional and intensity shortcuts across architectures: accurate | arXiv preprint |
| 47 | Saravanan 2020 | O (arXiv + NBDT page) | OK | hierarchical bootstrap: accurate; NBDT 3(5):1–25 confirmed | |
| 48 | Bouthillier 2021 | O (API) | OK | accurate | MLSys 2021 |
| 49 | Agarwal 2021 | O (API) | OK | accurate | NeurIPS 2021 |
| 50 | Captum | O (API) | OK | accurate | |
| 51 | Parkhi 2012 | O (Oxford VGG page) | OK | accurate | |
| 52 | MedMNIST v2 | O (API, journal ref Scientific Data 2023, DOI 10.1038/s41597-022-01721-8) | OK | accurate | |
| 53 | Kermany 2018 | S (cell.com, PubMed) | OK | accurate | Cell 172(5):1122–1131 (search result) |
| 54 | He 2015 | O (API) | FIX (minor) | accurate | Cite the CVPR 2016 version; authors He, Zhang, Ren, Sun |

Totals: 54 checked; 50 opened directly (page or arXiv API record), 4 venue details only from search snippets (Gevaert's journal, DeGrave's journal, Fresh Look's chapter, Kermany's pages). No invented or non-existent reference was found. All 54 claims about the cited works are accurate. Two are incomplete in a way that matters for positioning: [19] BAM and [26] Adebayo 2022. One placement is wrong: [31] in Table 1. Ten entries need bibliographic updates: [10], [11], [14], [17], [25], [30], [33], [39], [42], [54].

## Missing work found in this audit (pages opened)

| Work | URL opened | Why it matters |
|---|---|---|
| AIM: Adversarial Information Masking (May 2026) | https://arxiv.org/abs/2605.16905 | Masking operator confounds degradation: zero masking is OOD and interpolation keeps residual information. Assesses random-attribution bias. Matches blur leaving 20–25% of the patch and the random-over-object finding |
| How to Evaluate and Refine your CAM (ICPR 2026) | https://arxiv.org/abs/2605.14641 | A synthetic ground-truth dataset used to compare CAM **evaluation metrics**, i.e. metric meta-evaluation with ground truth. Belongs in 2.2 and Table 1 |
| F-Fidelity (ICLR 2025) | https://arxiv.org/abs/2410.02970 | OOD and leakage fix for removal metrics; recovers a ground-truth ranking of degraded explainers |
| From Flexibility to Manipulation (ECCV 2024 workshop) | https://arxiv.org/abs/2412.05592 | Hyperparameter choices of evaluation can be exploited. Supports reporting several fills. Recommended in audit 1 but not added |
| Hase et al. (NeurIPS 2021) | https://arxiv.org/abs/2106.00786 | OOD problem of removal-based metrics; compares replacement functions |
| Bhatt et al. (IJCAI 2020) | https://arxiv.org/abs/2005.00631 | Source of Faithfulness Correlation, the obvious second metric family named in Limitations |
| Understanding Rare Spurious Correlations (Yang et al. 2022) | https://arxiv.org/abs/2202.05189 | A handful of examples suffices to learn a spurious pattern. Relevant to the steep low-dose region and the A1 doses |
| BonaFide, "Faithfulness Metrics Don't Measure Faithfulness" (May 2026) | https://arxiv.org/abs/2605.25052 | Cross-domain (LLM CoT) meta-evaluation of faithfulness metrics with ground-truth labels, scored by AUROC. Supports the per-image AUROC design |
| GLBW, Towards Faithful XAI Evaluation via Generalization-Limited Backdoor Watermark (ICLR 2024) | S only (ICLR proceedings PDF and OpenReview in search results; OpenReview blocked) | Backdoor-based XAI evaluation is unreliable when triggers generalise. Relevant to [24], [25] and to the placebo map. Confirm on the proceedings page before citing |

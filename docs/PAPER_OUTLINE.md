# Paper Outline

The plan behind `paper/paper_draft.docx`, which `paper/build_paper.py` builds from the result files. Do not copy numbers from this outline into the paper by hand: the builder reads them from `results/dose_response/*.csv`. The numbers quoted here are from the current 234-model run.

## Title

"A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers"

## Target and format

- Provisional target journal: **Neurocomputing** (Elsevier), Q1 in Artificial Intelligence, SJR 1.47 (SCImago widget, `docs/audit/00_combined_audit.md` §7).
- The free subscription route (no article processing charge) is **unverified**. Confirm it in a normal browser before submitting.
- Single-column Elsevier-style manuscript (A4, Times New Roman 12 pt, 1.5 spacing), numbered references in square brackets, highlights (≤ 85 characters each), abstract ≤ 250 words (confirm both limits in the Guide for Authors). The elsarticle LaTeX version is optional and not prepared.

## Thesis

The metrics respond to reliance but are not calibrated to it, and their verdicts depend on the size and form of the explanations being compared.

## Research question

When a model's reliance on a shortcut increases in controlled steps, do faithfulness metrics shift their preference from object explanations to shortcut explanations, and at what reliance?

## Contributions

1. **A dose-response protocol for testing faithfulness metrics.** A signed training dose sets each model's reliance on an injected shortcut (234 models). Reliance is measured on conflicting test images and chance-corrected (r_c). Per-image behavioural labels come from forward passes. Fixed oracle maps keep the test independent of any attribution method.
2. **Responsiveness without calibration.** Preference rises with r_c in 27 of 28 metric–setting cells. But black and blur deletion over-detect on natural images: they prefer the shortcut map for 75–92% of the low-reliance Pet models (50–58% with a per-model margin) and, against a centred Gaussian object prior, 100% of the low-reliance CIFAR-10 models (93–100% with the margin). At the lowest dose the shortcut-vs-object preference equals the random-vs-object preference within 0.07, so much of this is the object map not beating random pixels; the excess preference still rises with r_c on Pet (ρ = 0.50–0.81). On Pet the metrics switch at 1.3–5.4% r_c, against a behavioural switch of 54.5% (not bracketed; 29.4–54.5% across labels); on CIFAR-10 the behavioural switch is not identified. Within one model, the per-image AUROC is only 0.43–0.80.
3. **Size and form dependence.** With a size-matched object map, the MNIST patch switch moves from 38.7–39.3% to 0.9–8.6% r_c. On CIFAR-10 black and blur deletion prefer a random ordering over a centred object map at low reliance (75–81% small CNN, 56–72% ResNet-18). Real Pet masks reduce this to 49–53%.
4. **A pre-registered test and amendment.** On Pet, P1, P2 and P4 hold and P3 fails for black deletion (commits a962b6c → 20694f2 → d823737 → 1e29918). Amendment A1 (8738b32) added 42 models before they were trained. Both model sets are reported.

Claims that earlier outlines made and that are now corrected:
- "Rises to 0.93–0.99 for every metric on the patch shortcut" was too general. The form-matched preference at the highest dose is 0.73–0.79 on PneumoniaMNIST patch and 0.76–1.00 on Pet (black deletion lowest).
- The old claim that inpainting deletion alone is significant in "eight" settings is replaced. There are 7 settings. Inpainting is the only variant with a positive, BH-significant ρ in all 28 shortcut-map comparisons. Black, blur and inpainting deletion are all 7/7 against the full object map. A paired test shows black deletion *better* than inpainting on PneumoniaMNIST texture.
- "Saliency, SmoothGrad and Grad-CAM move away from a diffuse shortcut" holds only on MNIST texture. On PneumoniaMNIST texture, Saliency and SmoothGrad move towards it. The Grad-CAM change on MNIST texture is negligible (0.100 → 0.098).
- MNIST inpainting switch interval. The old outline said the interval "reaches about 51%". In the legacy `responsiveness.csv` (192 models) its upper end was 49.95. In the current primary table (`responsiveness_corrected.csv`, r_c axis, two-level bootstrap) the inpainting switch is 39.2 [34.7, 52.5], status "identified". The black and blur switches before A1 lay in the 23–45% reliance gap; they are now identified, with the nearest model about 1 point away.

## Related work (positioning table in the paper, Section 2, Table 1)

| Paper | What it does | How we differ |
|---|---|---|
| BAM, Yang & Kim 2019, [arXiv:1907.09701](https://arxiv.org/abs/1907.09701) | Models trained with known relative feature importance, to evaluate attribution methods | We evaluate the metrics, with continuous measured reliance |
| SMERF, Kim, Plumb & Talwalkar, ICML 2022 ([PMLR](https://proceedings.mlr.press/v162/kim22h.html)) | Synthetic benchmark that controls the model's reasoning | Methods, not metrics |
| Bastings et al., EMNLP 2022 ([ACL](https://aclanthology.org/2022.emnlp-main.64/)) | Shortcut-injection protocol for salience methods (text) | Images; metrics |
| OSCAR, [arXiv:2512.18888](https://arxiv.org/abs/2512.18888) | Shortcut scores sensitive to association strength | Detection, not metric validation |
| Zhou et al., AAAI 2022; Lin et al., KDD 2021; BackX; Sun et al., MICCAI 2023; Adebayo et al., ICLR 2022 | Injected features or backdoors as ground truth for methods | Fixed strength; methods |
| Tomsett et al., AAAI 2020; MetaQuantus (TMLR 2023); Quantus (JMLR 2023); LATEC (Klein et al., NeurIPS 2024); M4 (NeurIPS 2023 D&B); Gevaert et al. | Meta-evaluation of metrics, mostly without ground truth | Ground truth that changes with the model |
| Miró-Nicolau et al., [arXiv:2401.10640](https://arxiv.org/abs/2401.10640) | Fidelity metrics checked against a decision tree on 52,000 synthetic 128×128 images | Learned CNNs, graded reliance |
| Miró-Nicolau, Spinnato & Guidotti 2026, [arXiv:2609.30397](https://arxiv.org/abs/2609.30397); FunnyBirds; XAI-Units; AttributionLab; Hesse et al. 2024 | Synthetic or designed ground truth; benchmarks for methods | Metrics, dose-response |
| ROAD; ROAR; ROAR pitfalls; Blücher et al. (TMLR 2024); Back to the Baseline; Gomez et al.; SaCo; Bilodeau et al. | Removal-shape, fill and ranking confounds; end-task framing | We show these confounds as reliance varies |
| Lapuschkin et al.; DeGrave et al.; Hermann et al. (ICLR 2024); Suhail et al. | Shortcuts in practice; predictivity vs availability | Motivation and limitation (we vary predictivity only) |

Citation rule: cite only papers whose page you have opened. Every reference in the draft was opened on 2026-11-13. When citing the sanity-check critique, give "Sanity Checks Revisited" (arXiv:2401.06465, NeurIPS XAIA 2023) and "A Fresh Look at Sanity Checks for Saliency Maps" (arXiv:2405.02383) as two separate entries.

## Structure (as built)

1. Introduction: problem, question, thesis, contributions (natural images first)
2. Related work: 2.1 controlled reliance and ground-truth benchmarks; 2.2 meta-evaluation; 2.3 known confounds of perturbation metrics; 2.4 shortcuts in practice; positioning table
3. Method: shortcuts; signed dose; reliance and chance correction; oracle maps with their limits; metrics; preference and per-image behavioural labels; statistics (two-level bootstrap, BH/Holm, isotonic + 4PL switch with ±10-point identification, over-detection rate and balanced accuracy, effect sizes, paired tests); methods
4. Setup: datasets, networks, model counts (from `models.csv`), amendment A1
5. Results:
   - 5.1 the dose controls reliance;
   - 5.2 natural images: respond but over-detect;
   - 5.3 pre-registered test;
   - 5.4 per-image behavioural ground truth;
   - 5.5 mechanism settings: size and switch identification;
   - 5.6 same-form controls;
   - 5.7 comparing the metrics (paired);
   - 5.8 explanation methods;
   - 5.9 reliance and object competence
6. Discussion: findings linked to known mechanisms; recommendations
7. Limitations
8. Conclusion
9. Declarations: CRediT, competing interest, data availability, generative-AI declaration, funding
10. Appendices:
    - A: sanity-check supporting study, with the exact Kendall null;
    - B: legacy analysis;
    - C: original vs amended model sets

## Before submitting

- [ ] Get your guide's approval of the claims and the venue
- [ ] Confirm the Neurocomputing no-fee route, the abstract and highlight limits, and the reference style in the Guide for Authors
- [ ] Re-run `python dose_response.py analyze` and `python paper\build_paper.py`, then check `.agents/tasks/q1-must-do/docx_check.md`
- [ ] Check every reference's authors, venue and pages against the published version, especially DeGrave et al. (cited from the PMC author manuscript) and M4 (cited by title)
- [ ] Follow the venue's rules on AI assistance and disclose it
- [ ] Put the code and result files in a public repository (Zenodo DOI recommended) and fill in the data-availability statement
- [ ] Don't submit to two venues at once

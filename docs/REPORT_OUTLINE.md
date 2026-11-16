# Report Outline

Project title: A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

A chapter-by-chapter plan for the M.Tech report (about 55–65 pages). The thesis and the paper share the same experiments, so write the paper first (`PAPER_OUTLINE.md`) and expand it here. Take the exact numbers from `results/dose_response/summary.md` (section "Corrected analysis (primary)") and `results/analysis/summary.md` after your final run, or copy them from `paper/paper_draft.docx`, which is generated from those files.

## Writing schedule (1–2 hours a day)

| Days | Write |
|---|---|
| 1–3 | The paper's Method and Results sections, which become thesis Chapters 3–5 |
| 4–5 | The paper's Introduction, Related work and Discussion |
| 6–7 | Expand into the thesis: implementation chapter, more figures, appendices |
| 8 | Abstract, conclusion, front matter |
| 9–10 | Slides, full read-through, viva practice with `VIVA_PREP.md` |

## Front matter

- Title page, certificate, declaration, acknowledgements
- An AI assistance statement, following your university's policy
- Abstract (adapt the abstract of `paper/paper_draft.docx`)
- Contents, list of figures, list of tables

## Chapter 1: Introduction (4–5 pages)

- Why explanations matter: medicine, finance, and regulation such as the EU AI Act
- Explanations are judged with metrics, but the metrics are rarely validated
- Research question: *When a model's reliance on a shortcut increases in controlled steps, do faithfulness metrics shift their preference from object explanations to shortcut explanations, and at what reliance?*
- Thesis: the metrics respond to reliance but are not calibrated to it, and their verdicts depend on explanation size and form
- Objectives:
  1. Build shortcut datasets (patch and texture) and control reliance with a signed training dose
  2. Measure reliance on conflicting images and correct it for chance
  3. Score fixed oracle maps (shortcut, object, size-matched, placebo, random) with four deletion/insertion metrics
  4. Label each scored image by forward passes (does the model follow the shortcut here?) and test the metrics against these labels
  5. Repeat on Oxford-IIIT Pet (real masks, pre-registered), CIFAR-10 (two networks), MNIST and PneumoniaMNIST; 234 models
  6. Run five explanation methods on every model; supporting sanity-check study; interactive dashboard
- Contributions (the 4 in `PAPER_OUTLINE.md`)
- Where the project came from: it extends the B.Tech digit recognition app
- Chapter overview

## Chapter 2: Literature Review (10–12 pages)

- CNNs for image classification
- Explanation methods: gradient (Saliency, SmoothGrad, Integrated Gradients), activation (Grad-CAM) and perturbation (Occlusion). One or two paragraphs each, with formulas.
- Methods out of scope: LIME, SHAP, LRP. One line each.
- Evaluating explanations:
  - Perturbation metrics and their confounds (Samek, Petsiuk, ROAD)
  - Sanity checks and their critiques (Adebayo 2018, Binder 2023, Hedström 2024)
  - Ground-truth benchmarks (CLEVR-XAI, Decoy MNIST, Adebayo 2022, FunnyBirds, XAI-Units)
  - Controlled reliance for methods: BAM, SMERF, Bastings et al., OSCAR; how this project differs (the metrics are tested, reliance is graded and measured)
  - Meta-evaluation of metrics (Tomsett, MetaQuantus, Quantus, LATEC, M4, Gevaert, Miró-Nicolau) and confounds (ROAD, ROAR and its pitfalls, Blücher, Gomez, SaCo, Bilodeau)
  - Use the positioning table in the paper (Section 2, Table 1) and the verified reference list in `docs/audit/00_combined_audit.md` §9
- Shortcut learning, and why it matters in medical imaging
- The research gap

## Chapter 3: Methodology (10–12 pages)

- **3.1 Datasets:** MNIST, Fashion-MNIST, PneumoniaMNIST (4,708 train, 524 validation, 624 test, CC BY 4.0). Show samples.
- **3.2 Decoy datasets:** patch positions (include a figure), the 50% swap in training and why it is needed, the conflicting-patch test set, and the "follows the patch" filter. Also mention the first attempt without the swap: the models then mostly ignored the patch (shortcut reliance 1–17% on Decoy MNIST), which is why the swap was introduced. That's a good design discussion.
- **3.3 Model and training:** the architecture table, Adam with learning rate 0.001, batch size 128, seeds 42, 1 and 2, and model selection on the validation set
- **3.4 Explanation methods:** formulas and settings (see `VIVA_PREP.md`)
- **3.5 Metrics:** deletion with black, blur (σ = 4, and why) or inpaint replacement; insertion; the sanity check; mass and rank accuracy
- **3.6 Dose-response design:** the signed dose, measured reliance, chance correction r_c = 100 (R − f)/(100 − f) with f = (100 − accuracy without shortcut)/(C − 1), oracle maps and their limits (PneumoniaMNIST object map is an intensity ranking; CIFAR-10 uses a Gaussian prior; blur leaves 20–25% of the patch; size-matched tie order)
- **3.7 Per-image behavioural labels:** the moved-shortcut and removed-shortcut forward passes (primary) and the Δ_S > Δ_O check; class-stratified AUROC
- **3.8 Statistics:** two-level bootstrap (seeds within dose, images within model), Benjamini–Hochberg families and Holm for headline claims, isotonic switch with 4PL check and the ±10-point identification rule, over-detection rate and balanced accuracy, median AUC gap, paired metric tests, exact Kendall null for five methods; seed stability
- **3.9 Pre-registration:** `docs/PREDICTIONS_pet.md` (P1–P4) and `docs/PREDICTIONS_amendments.md` (A1), with commit hashes

## Chapter 4: Implementation (6–8 pages)

- Tools and versions (`requirements.txt`)
- The code architecture: the `xai/` package plus four scripts, with a pipeline diagram
- The dashboard: its API, preprocessing, screenshots (including a decoy example) and response time
- Reproducibility: fixed seeds, fixed test images, one command per step (see README)

## Chapter 5: Results (12–15 pages)

Use the tables and figures of `paper/paper_draft.docx` (natural images first; the pre-registered Pet test; the behavioural test; mechanism settings; controls; paired metric tests; methods; reliance vs competence), plus these, which don't fit in the paper:
- The full `responsiveness_corrected.csv` for all 7 comparisons, and `effect_sizes.csv`
- The original-set check (`results/dose_response/original_models/`)
- Per-dataset perturbation curves (`results/metrics/<dataset>/perturbation_curves.png`)
- Per-dataset sanity curves and examples (`sanity_curve.png`, `sanity_examples.png`)
- Explanation grids for all five datasets (`results/explanations/`)
- The full significance table (appendix)

## Chapter 6: Discussion (5–6 pages)

The discussion points from `PAPER_OUTLINE.md`, in more depth, plus:
- The effect of dataset: natural images with real masks, colour images with a Gaussian prior, digits and radiographs
- Why chance correction and behavioural labels change the conclusions (lowest PneumoniaMNIST reliance about 15% raw is about 0% chance-corrected; the behavioural switch is 37–54% r_c where it is identified and bracketed, depends on the label, is not bracketed on Pet and is not identified on CIFAR-10)
- The collinearity of reliance and object competence (`confound.csv`)
- What the results mean for practitioners in medical imaging

## Chapter 7: Conclusion and Future Work (2–3 pages)

## Appendices

- A: How to run (README)
- B: Full result tables (`results/analysis/*.csv`)
- C: Key code: `xai/explain.py`, `xai/metrics.py`, `xai/data.py` (decoy construction)

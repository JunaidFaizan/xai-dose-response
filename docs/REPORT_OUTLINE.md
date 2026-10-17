# Report Outline

A chapter-by-chapter plan for the M.Tech report (about 55–65 pages). The thesis and the paper share the same experiments, so write the paper first (`PAPER_OUTLINE.md`) and expand it here. Take the exact numbers from `results/analysis/summary.md` after your final run.

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
- Abstract (adapt the draft in `PAPER_OUTLINE.md`)
- Contents, list of figures, list of tables

## Chapter 1: Introduction (4–5 pages)

- Why explanations matter: medicine, finance, and regulation such as the EU AI Act
- Explanations are judged with metrics, but the metrics are rarely validated
- Research question: *Do common explanation metrics identify the methods that point to the evidence a model really uses?*
- Objectives:
  1. Build shortcut datasets with a known ground truth
  2. Apply five explanation methods
  3. Measure them with perturbation metrics and the sanity check
  4. Test the metrics against the ground truth
  5. Repeat on standard and medical data with several seeds
  6. Build an interactive dashboard
- Contributions (the 4 in the paper outline)
- Where the project came from: it extends the B.Tech digit recognition app
- Chapter overview

## Chapter 2: Literature Review (10–12 pages)

- CNNs for image classification
- Explanation methods: gradient (Saliency, SmoothGrad, Integrated Gradients), activation (Grad-CAM) and perturbation (Occlusion). One or two paragraphs each, with formulas.
- Methods out of scope: LIME, SHAP, LRP. One line each.
- Evaluating explanations:
  - Perturbation metrics and their confounds (Samek, Petsiuk, ROAD)
  - Sanity checks and their critiques (Adebayo 2018, Binder 2023, Hedström 2024)
  - Ground-truth benchmarks (CLEVR-XAI, Decoy MNIST, Adebayo 2022)
- Shortcut learning, and why it matters in medical imaging
- The research gap

## Chapter 3: Methodology (10–12 pages)

- **3.1 Datasets:** MNIST, Fashion-MNIST, PneumoniaMNIST (4,708 train, 524 validation, 624 test, CC BY 4.0). Show samples.
- **3.2 Decoy datasets:** patch positions (include a figure), the 50% swap in training and why it is needed, the conflicting-patch test set, and the "follows the patch" filter. Also mention the first attempt without the swap: the models then mostly ignored the patch (shortcut reliance 1–17% on Decoy MNIST), which is why the swap was introduced. That's a good design discussion.
- **3.3 Model and training:** the architecture table, Adam with learning rate 0.001, batch size 128, seeds 42, 1 and 2, and model selection on the validation set
- **3.4 Explanation methods:** formulas and settings (see `VIVA_PREP.md`)
- **3.5 Metrics:** deletion with black, blur (σ = 4, and why) or inpaint replacement; insertion; the sanity check; mass and rank accuracy
- **3.6 Meta-evaluation and statistics:** Kendall τ, Spearman ρ, Wilcoxon tests with Holm correction, and seed stability

## Chapter 4: Implementation (6–8 pages)

- Tools and versions (`requirements.txt`)
- The code architecture: the `xai/` package plus four scripts, with a pipeline diagram
- The dashboard: its API, preprocessing, screenshots (including a decoy example) and response time
- Reproducibility: fixed seeds, fixed test images, one command per step (see README)

## Chapter 5: Results (12–15 pages)

Use the tables and figures listed in `PAPER_OUTLINE.md`, plus these, which don't fit in the paper:
- Per-dataset perturbation curves (`results/metrics/<dataset>/perturbation_curves.png`)
- Per-dataset sanity curves and examples (`sanity_curve.png`, `sanity_examples.png`)
- Explanation grids for all five datasets (`results/explanations/`)
- The full significance table (appendix)

## Chapter 6: Discussion (5–6 pages)

The discussion points from `PAPER_OUTLINE.md`, in more depth, plus:
- The effect of dataset: standard, medical and shortcut data
- What the results mean for practitioners in medical imaging

## Chapter 7: Conclusion and Future Work (2–3 pages)

## Appendices

- A: How to run (README)
- B: Full result tables (`results/analysis/*.csv`)
- C: Key code: `xai/explain.py`, `xai/metrics.py`, `xai/data.py` (decoy construction)

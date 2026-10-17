# Paper Outline

A plan for an 8–10 page paper in IEEE or Springer LNCS format. Take exact numbers from `results/dose_response/summary.md` (main study) and `results/analysis/summary.md` (supporting study). The numbers below are from the current runs.

## Working title

"Do XAI Metrics Notice When a Model Starts Cheating? A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance"

## Research question

When a model's reliance on a shortcut increases in controlled steps, do faithfulness metrics shift their preference from object explanations to shortcut explanations, and at what reliance?

## Contributions

1. **A dose-response protocol for meta-evaluating XAI metrics.** A signed training dose sets the model's reliance on an injected shortcut, from about 0% to 100%. Reliance is measured behaviourally, and fixed oracle explanations are scored, so the metrics are tested independently of any explanation method.
2. **Evidence that metric responsiveness depends on the data and the shortcut type.** Metrics track reliance closely on MNIST (ρ = 0.91–0.98) but weakly or not at all on chest X-rays (ρ = 0.07–0.84). Insertion fails on a diffuse shortcut.
3. **A size and shape confound in oracle-based evaluation.** Against a size-matched object map, the switch point on MNIST patch moves from 30–42% reliance to 6–15%, so metrics partly reward explanation size. The placebo control confirms it: a shortcut-shaped map at a position the model never uses is increasingly preferred over the object map as reliance grows (MNIST ρ = 0.89–0.97), and black-fill deletion on CIFAR-10 prefers it 82% of the time at 4% reliance. On the diffuse MNIST shortcut, insertion prefers the placebo over the real shortcut (ρ = −0.81).
4. **Ranking is not calibration (CIFAR-10).** On 32×32 colour images with a larger CNN, preference still tracks reliance (ρ = 0.88–0.99), but black and blur deletion prefer the shortcut explanation 74–82% of the time even at 4% reliance. A high correlation with reliance is therefore not enough to trust a metric's verdict. Inpainting deletion and insertion are far less biased.
5. **Practical guidance.** Inpainting deletion is the only variant that responds significantly in all eight settings, and some explanation methods (Saliency, SmoothGrad, Grad-CAM) move away from a diffuse shortcut as the model relies on it more.
6. **Supporting fixed-shortcut study:** five methods, five datasets and three seeds, showing that the model-randomization sanity check does not rank methods in line with the ground truth (Kendall τ = −0.20 to 0.07). Use this as a secondary section, or leave it out if space is short.

## Related work (verified links; read each before citing)

| Paper | What it does | How we differ |
|---|---|---|
| Zhou et al., AAAI 2022, [arXiv:2104.14403](https://arxiv.org/abs/2104.14403) | Induces ground-truth features to test attribution methods | We test the metrics, at graded reliance |
| BackX, [arXiv:2405.02344](https://arxiv.org/abs/2405.02344) | Backdoor-based benchmark for attribution methods | Fixed trigger strength; methods, not metrics |
| Sun et al., MICCAI 2023, [arXiv:2307.12344](https://arxiv.org/abs/2307.12344) | Can explanations detect spurious correlations in chest X-rays? | Methods, at a fixed strength |
| Adebayo et al., ICLR 2022 | Post hoc explanations and unknown spurious correlations | Methods, plus a human study |
| Tomsett et al., AAAI 2020, [arXiv:1912.01451](https://arxiv.org/abs/1912.01451) | Saliency metrics are statistically unreliable | No ground truth |
| MetaQuantus, [arXiv:2302.07265](https://arxiv.org/abs/2302.07265) | Meta-evaluates metrics by noise resilience and reactivity to randomness | No ground truth, no graded reliance |
| LATEC, NeurIPS 2024, [arXiv:2409.16756](https://arxiv.org/abs/2409.16756) | 20 metrics × 17 methods; the metrics conflict | Large benchmark, no controlled reliance |
| AttributionLab, [arXiv:2310.06514](https://arxiv.org/abs/2310.06514) (NeurIPS 2023 XAIA workshop) | Hand-set weights give an exact ground truth | Designed, not learned, models; no dose |
| [arXiv:2401.10640](https://arxiv.org/abs/2401.10640) | Fidelity metrics checked against a transparent model | Tabular/transparent models |
| Back to the Baseline, [arXiv:2512.11433](https://arxiv.org/abs/2512.11433) | The deletion/insertion baseline favours some methods | Supports our supporting study's finding |
| ROAD (Rong et al., ICML 2022); Binder et al. (CVPR 2023); Hedström et al. (2024) | Perturbation confounds; sanity check critiques | Background |
| M4, NeurIPS 2023 Datasets & Benchmarks | Faithfulness benchmark across modalities | Read the PDF to confirm its synthetic ground-truth part before citing |

Before submitting, run your own search, with your guide, for "graded spurious correlation", "dose-response" and "meta-evaluation of faithfulness metrics" from 2024 onwards. An earlier AI-generated literature summary contained invented citations, so cite only papers you've opened yourself.

## Structure

1. **Introduction** (1 page): problem, question, contributions
2. **Related work** (1 page): the table above, written as prose
3. **Method** (2 pages)
   - Shortcuts: a 3×3 patch at class-coded positions; a texture of a class-specific random 10% of pixels, brightened by 0.3
   - Signed dose: image swaps (d > 0) and shortcut noise (d < 0), capped at |d| = 0.4, and why both are needed (with d = 0 the shortcut is already easier to learn)
   - Reliance: following a conflicting shortcut on the test set
   - Oracle maps: shortcut, full object, size-matched object, placebo (the shortcut's shape at a wrong class's position)
   - Metrics: deletion (black, blur, inpaint), insertion (blur)
   - Scores: preference, responsiveness ρ, decision accuracy, switch point
4. **Setup** (0.5 page): MNIST and PneumoniaMNIST, the CNN, 12 doses × 3 seeds × 2 shortcuts × 2 datasets = 144 models; plus CIFAR-10 with a larger CNN (0.77M parameters, batch norm, 10 epochs), patch shortcut, 6 doses × 2 seeds = 12 models; 100 test images each, CPU only. For CIFAR-10 the object map is a centred Gaussian, because natural images have no clean background.
5. **Results** (3 pages)
   - Table 1: reliance per dose (`summary.md`, first table). It shows the dose controls reliance, and how much seeds vary near d = 0.
   - Figure 1: preference vs reliance, full object map (`metric_dose_response.png`). The main result.
   - Figure 2: the same with the size-matched control (`metric_dose_response_size_matched.png`)
   - Figure 2b: the placebo controls (`metric_dose_response_placebo_vs_object.png`, `metric_dose_response_shortcut_vs_placebo.png`)
   - Table 2: responsiveness, decision accuracy and switch point per metric and setting (`responsiveness.csv`)
   - Figure 3: explanation methods vs reliance (`method_dose_response.png`)
   - Optional, from the supporting study: sanity check vs ground truth (`results/analysis/gt_agreement.png`)
6. **Discussion** (1 page)
   - Why the X-ray setting is harder: textured background, 2 classes, and the lowest reliance reached was 15%
   - The size confound, and what it means for any oracle- or mask-based evaluation
   - Why insertion fails on a diffuse shortcut: restoring scattered pixels into a blurred image barely changes it
   - Why some methods move away from the texture shortcut. Possible reasons are gradient saturation, or the texture adding to bright digit pixels. State this as a hypothesis.
   - Recommendation: report inpainting-based deletion, test metrics at several reliance levels, and control for explanation size
7. **Limitations** (0.5 page): 28×28 images and a small CNN; synthetic shortcuts; reliance varies between seeds near d = 0; the 50% switch-point ideal is an assumption; one size-matching rule; ROAD-inspired, not exact ROAD
8. **Conclusion**

## Draft abstract (rewrite in your own words)

> Faithfulness metrics such as deletion and insertion curves are widely used to compare explanation methods, yet whether they detect changes in what a model actually relies on has rarely been tested. We propose a dose-response protocol: CNNs are trained on MNIST and PneumoniaMNIST with an injected local patch or diffuse texture shortcut, and a signed training dose sets their measured reliance on it from about 0% to 100% (144 models). Scoring fixed oracle explanations, we test whether each metric shifts its preference from an object explanation to a shortcut explanation as reliance grows. On MNIST, all metrics track reliance closely (Spearman ρ = 0.91–0.98), but black and blur deletion switch preference too early, at 30–35% reliance (95% bootstrap intervals 30–42%), and every metric switches at 6–15% (intervals 1–20%) when the object explanation is matched in size to the shortcut. This reveals that the metrics partly reward explanation size. On chest X-rays, responsiveness falls to ρ = 0.07–0.84, and some variants do not respond significantly. Insertion fails on the diffuse shortcut, while inpainting-based deletion is the only variant that responds significantly in all settings. Among five attribution methods, Integrated Gradients follows the shortcut in every setting, whereas Saliency, SmoothGrad and Grad-CAM put less weight on a diffuse shortcut as the model relies on it more. We recommend validating metrics at several controlled reliance levels, and controlling for explanation size, before using them to compare explanation methods.

## Before submitting

- [ ] Get your guide's approval of the claims and the venue
- [ ] Re-run `python dose_response.py analyze` and copy the final numbers
- [ ] Report the switch points and ρ with their 95% bootstrap intervals (`switch_ci_*`, `rho_ci_*` in `responsiveness.csv`). On MNIST patch, the inpaint and insertion switch-point intervals reach about 51%, so only black and blur deletion are clearly early with the full object map
- [ ] Confirm the venue is indexed (Scopus / Web of Science) and not predatory; check its deadline on its own website
- [ ] Follow the venue's rules on AI assistance and disclose it
- [ ] Put the code in a public repository and cite it
- [ ] Don't submit to two venues at once

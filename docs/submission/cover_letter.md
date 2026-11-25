# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Cover letter (draft, to be finished by the author)

[[AUTHOR: date]]

To the Editor-in-Chief, *Neurocomputing* (Elsevier)

Dear Editor,

I submit the manuscript "A Dose-Response Evaluation of Explanation Metrics under Controlled
Shortcut Reliance in CNN Image Classifiers" for consideration as a Regular article in *Neurocomputing*.

**What the paper does.** Deletion and insertion metrics decide which explanations of an image
classifier are called faithful, but they are rarely tested against a known model reliance. I train
293 CNNs in which a signed training dose sets how strongly the model relies on an injected,
class-coded shortcut, measure that reliance on held-out conflicting images with a chance correction,
and score fixed oracle explanations with four metrics. The data are Oxford-IIIT Pet (with object
masks), CIFAR-10, MNIST and PneumoniaMNIST. The metrics respond to reliance but are not calibrated
to it: black and blur deletion prefer the shortcut map for many models that barely use the shortcut,
and the verdicts depend on the size and form of the explanations compared.

**Pre-registration.** The Pet predictions were committed before any Pet model was trained.
Further predictions are pre-registered and time-stamped on Zenodo:

- version 1, amendment A2 with predictions P5–P9 (SD1 pretrained network and SD2 gap-filling doses):
  https://doi.org/10.5281/zenodo.24285532
- version 2, completion notes with the final SD1 and SD2 doses:
  https://doi.org/10.5281/zenodo.24333540
- version 3, amendment A3 with predictions P10–P13 (exact ROAD and a watermark shortcut), uploaded
  before any SD3 or SD4 model or score existed: https://doi.org/10.5281/zenodo.24416017

- version 4, completion note A3-C1 with the SD4 pilot outcome, the final SD4 doses and the ROAD
  timing, uploaded before any confirmatory SD4 model was trained:
  https://doi.org/10.5281/zenodo.24495658

The concept DOI of the record is https://doi.org/10.5281/zenodo.24285531. The manuscript reports the
outcome of every registered prediction, including the ones that failed. Of the four A3 predictions,
three held and one was refuted. Exact ROAD ranked the object map above random pixels more often than
black deletion in all four natural-image settings (P10), and it over-detected the shortcut slightly
less often on Pet (9 against 10 of 14 low-reliance models, P11). On the watermark shortcut the four
metrics tracked reliance (Spearman ρ 0.82–0.90, P12), but blur deletion over-detected the watermark
for only 2 of 9 low-reliance models, so P13 was refuted.

**Why *Neurocomputing*.** The paper is about neural networks: it studies how the faithfulness metrics used to
explain trained CNN image classifiers respond to controlled shortcut reliance in 293 trained networks, and it
gives a controlled test bed for these metrics. It is an empirical study with a new, pre-registered protocol and
not a combination of existing algorithms. [[AUTHOR: optional: add your own reasons for the journal fit after
reading the aims and scope, or delete this marker]]

**Authorship and declarations.** The manuscript has a single author, Md Junaid Faizan Shaikh (Khaja
Bandanawaz University; ORCID 0009-0002-6565-1835). It is not under consideration elsewhere
[[AUTHOR: confirm]], and the data, code and checkpoints are archived on Zenodo
([[ZENODO_RELEASE_DOI]]). Competing interests: [[AUTHOR: competing interests statement]]. Funding:
[[AUTHOR: funding statement]]. Use of AI tools in preparing the manuscript is declared in the
manuscript as the journal's policy requires.

**Keywords:** Explainable AI; Faithfulness metrics; Perturbation metrics; Shortcut learning; Meta-evaluation;
Pre-registration (the same list as in the manuscript).

**Suggested reviewers** are listed in the submission form (see `suggested_reviewers.md`).

Sincerely,

Md Junaid Faizan Shaikh
Khaja Bandanawaz University, [[AUTHOR: postal address]]
[[AUTHOR: e-mail]]

[[AUTHOR: optional line, delete if not wanted: I do not yet have an institutional e-mail address from my university
and use a personal e-mail address for this submission and for correspondence.]]

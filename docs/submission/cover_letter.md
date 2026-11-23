# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Cover letter (draft, to be finished by the author)

[[AUTHOR: date]]

To the Editor-in-Chief, *Neurocomputing* (Elsevier)

Dear Editor,

I submit the manuscript "A Dose-Response Evaluation of Explanation Metrics under Controlled
Shortcut Reliance in CNN Image Classifiers" for consideration as [[AUTHOR: article type, for
example Full length article]] in *Neurocomputing*.

**What the paper does.** Deletion and insertion metrics decide which explanations of an image
classifier are called faithful, but they are rarely tested against a known model reliance. I train
275 CNNs in which a signed training dose sets how strongly the model relies on an injected,
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

The concept DOI of the record is https://doi.org/10.5281/zenodo.24285531. The manuscript reports the
outcome of every registered prediction, including the ones that failed.
[[SD3_SD4_COVER_LETTER: one or two sentences on the SD3 and SD4 outcomes and the Zenodo version 4 DOI,
after the results exist]]

**Why *Neurocomputing*.** The work concerns the evaluation of explanations for neural-network image
classifiers and gives a controlled test bed for it. [[AUTHOR: add one sentence on fit with the scope
of the journal after reading the aims and scope]]

**Authorship and declarations.** The manuscript has a single author, Md Junaid Faizan Shaikh (Khaja
Bandanawaz University; ORCID 0009-0002-6565-1835). It is not under consideration elsewhere
[[AUTHOR: confirm]], and the data, code and checkpoints are archived on Zenodo
([[ZENODO_RELEASE_DOI]]). Competing interests: [[AUTHOR: competing interests statement]]. Funding:
[[AUTHOR: funding statement]]. Use of AI tools in preparing the manuscript is declared in the
manuscript as the journal's policy requires.

**Suggested reviewers** are listed in the submission form (see `suggested_reviewers.md`).

Sincerely,

Md Junaid Faizan Shaikh
Khaja Bandanawaz University, Kalaburagi, India
[[AUTHOR: e-mail]]

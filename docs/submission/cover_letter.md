# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Cover letter

<!-- docx: paper/build_submission_docx.py writes docs/submission/cover_letter.docx from the paragraphs below the line -->
---

[[AUTHOR: date of submission]]

The Editor-in-Chief
*Neurocomputing* (Elsevier)

Dear Editor-in-Chief,

I submit the manuscript "A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers" for consideration as a Regular article in *Neurocomputing*.

Deletion and insertion metrics decide which explanations of neural image classifiers are called faithful, but they are rarely tested against a known model reliance. I trained 293 convolutional neural networks in which a signed training dose sets the measured, chance-corrected reliance on an injected shortcut, on Oxford-IIIT Pet, CIFAR-10, MNIST and PneumoniaMNIST, and scored fixed oracle explanations with four metrics. The metrics respond to reliance but are not calibrated to it: black and blur deletion prefer the shortcut explanation for most models that barely use the shortcut, and the verdicts depend on the size and form of the explanations compared. The study gives a controlled, pre-registered test bed for the faithfulness metrics used to explain trained neural networks; it is an empirical study with a new protocol, not a combination of existing algorithms.

The predictions were pre-registered, and the amendments and their completion notes are time-stamped on Zenodo as versions 1 to 4 of one record (https://doi.org/10.5281/zenodo.24285532, https://doi.org/10.5281/zenodo.24333540, https://doi.org/10.5281/zenodo.24416017 and https://doi.org/10.5281/zenodo.24495658; concept DOI https://doi.org/10.5281/zenodo.24285531). Ten of the thirteen pre-registered predictions held, and the manuscript reports the three that did not. The code, per-model results and trained checkpoints are archived at https://doi.org/10.5281/zenodo.24551101.

The manuscript is original, has not been published before and is not under consideration for publication elsewhere. I am its sole author and I approve its submission. The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The use of generative AI and AI-assisted tools in preparing the manuscript is declared in the manuscript, as the journal's policy requires.

Sincerely,

Md Junaid Faizan Shaikh
Department of Computer Science and Engineering, Khaja Bandanawaz University, Administrative Building, Khaja Bandanawaz University Campus, Rauza-i Buzurg, Kalaburagi, Karnataka 585104, India
E-mail: junaid.faizan.codes@gmail.com · ORCID: 0009-0002-6565-1835
The author does not have an institutional e-mail address at present and uses a personal address for correspondence.

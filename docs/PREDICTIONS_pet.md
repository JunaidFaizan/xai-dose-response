# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Pre-registered predictions: Oxford-IIIT Pet extension

Date: 2026-11-10. Written and committed before any Pet model was trained.

### Design

The new base dataset `pet_resnet18` is the Oxford-IIIT Pet dataset as a binary
cat-vs-dog task, with images resized to 128x128 (shorter side, then centre
crop). A ResNet-18 is trained from scratch (no pretrained weights, 30 epochs).
The shortcut is a 12x12 white patch (3/32 of the image side, as for CIFAR-10)
in the top-left corner for cats and the top-right corner for dogs. Doses
PET_DOSES = [-0.4, -0.3, -0.2, -0.1, -0.05, 0.0] with seeds [42, 1, 2] give
18 models. The object oracle map is the real pet segmentation mask (trimap
value 1 = pet, border value 3 counted as pet), with the shortcut pixels
removed. The size-matched object map is compact: the k mask pixels closest to
the mask centroid, where k is the number of shortcut pixels (144).

Reliance is measured as in the main study: the share of conflicting test
images on which the model follows the shortcut. "Preference" for map A over
map B is the share of test images where a metric rates A as the better
explanation. All statistics below are computed by `dose_response.py analyze`
over the 18 Pet models (`responsiveness.csv`, `method_responsiveness.csv`).

### Predictions

The four metrics are Deletion (black), Deletion (blur), Deletion (inpaint)
and Insertion (blur).

- **P1.** For each of the four metrics, the preference for the shortcut map
  over the real object map rises with reliance: Spearman rho between reliance
  and preference > 0.8 (comparison "shortcut vs full object").
- **P2.** At the lowest-reliance dose, the preference for a random map over
  the real object map ("random vs full object") is higher for Deletion (black)
  and Deletion (blur) than for Deletion (inpaint) and Insertion (blur).
- **P3.** In the form-matched comparisons ("shortcut vs random" and
  "shortcut vs placebo"), preference at the lowest-reliance dose lies within
  0.5 ± 0.15 for every metric, and Spearman rho with reliance is > 0.8.
- **P4.** For Integrated Gradients and Occlusion, the share of positive
  attribution on the shortcut rises with reliance: Spearman rho > 0.8 in
  `method_responsiveness.csv`.

### Commitments

- The doses may be adjusted after the first Kaggle model if its reliance falls
  outside a useful range. Any such change will be reported together with the
  reason.
- All results will be reported whether or not these predictions hold.

## Amendment: dose change after the first model

The first Kaggle model (dose -0.4, seed 42; 30 epochs, 23.2 min on a Tesla T4)
reached a reliance of 24.8%, 84.7% test accuracy and 80.1% accuracy without the
shortcut. No metric, oracle or method results were computed or seen. Because
reliance did not reach the low end of the range, the doses were changed to:

PET_DOSES = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]

Dose -0.05 was dropped and -0.5 added. At -0.5 half the training shortcuts
point to the wrong class, so the shortcut carries no information about the
label. The predictions P1–P4 above are unchanged.

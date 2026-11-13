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

## Outcome (all 18 models)

Recorded after the full analysis (`dose_response.py analyze`, 18 Pet models,
seeds 42, 1 and 2). Reliance ranged from 17% to 99%; accuracy without the
shortcut was 65–82%. Values are from `responsiveness.csv` and
`method_responsiveness.csv`.

| Prediction | Result | Verdict |
|---|---|---|
| P1: shortcut vs full object, rho > 0.8 for all four metrics | black 0.85, blur 0.96, inpaint 0.95, insertion 0.93 | **Holds** (black's 95% bootstrap interval reaches down to 0.71) |
| P2: at the lowest dose, random-vs-object preference higher for black and blur than for inpaint and insertion | black 0.53, blur 0.49 vs inpaint 0.38, insertion 0.35 | **Holds**, with a smaller gap than on CIFAR-10 (black/blur 0.56–0.82 there) |
| P3: form-matched comparisons start within 0.5 ± 0.15 and have rho > 0.8 | Starting values 0.45–0.57 for every metric (pass). rho vs random: 0.75, 0.88, 0.96, 0.95; vs placebo: 0.73, 0.89, 0.95, 0.93 | **Fails for Deletion (black)** (rho 0.75 and 0.73); holds for the other three metrics |
| P4: IG and Occlusion shortcut mass rises, rho > 0.8 | IG 0.94 (0.6% → 22%), Occlusion 0.98 (0.4% → 60%) | **Holds** |

Summary: three of four predictions hold; P3 fails for black-fill deletion,
which separates the real shortcut from form-matched controls less reliably than
the other metrics on 128x128 photos.

Exploratory observations (not pre-registered):

- With the real pet mask, the bias of black and blur deletion towards compact or
  random maps at low reliance is much smaller than with the centred-Gaussian
  CIFAR-10 object map (about 0.49–0.53 here vs 0.56–0.82 on CIFAR-10), which
  supports the earlier hypothesis that the Gaussian object map inflated it.
- All metrics still switch their verdict early against the full object map
  (about 19–25% reliance where a switch is found).
- Against the compact size-matched object map, most metrics start near 0.5 at
  the lowest reliance (0.46–0.55), so a size-matched object region is nearly
  indistinguishable from the shortcut when the model barely uses it.

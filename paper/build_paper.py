"""Build paper/paper_draft.docx from the finished dose-response results.

Usage (from the project root):
    .venv\\Scripts\\python.exe paper\\build_paper.py

Every number in the text and tables is read from the result files; nothing is
typed in by hand:
    results/dose_response/  models.csv, preference.csv,
        responsiveness_corrected.csv (primary), method_responsiveness_corrected.csv (primary),
        behaviour_summary.csv (primary per-image test), effect_sizes.csv, metric_pairs.csv,
        pet_preregistered.csv, confound.csv,
        responsiveness.csv / method_responsiveness.csv (legacy, Appendix B),
        original_models/responsiveness_corrected.csv (192-model check set, Appendix C),
        audit-2 additions: intervals_crossed.csv (crossed and BCa intervals),
        behaviour_switch.csv (behavioural switch status, three labels),
        switch_differences.csv (paired behavioural - metric switch),
        overdetection.csv and excess_preference_models.csv (S-O / R-O / S-R,
        excess preference, Clopper-Pearson and margin FPR, r_c bands),
        behaviour_calibration.csv, confound_rc.csv, correct_only.csv, behaviour_power.csv
    results/analysis/  gt_agreement.csv, gt_instance.csv, accuracy.csv,
        decoy_mnist_summary.csv, kendall_exact.csv (supporting study, Appendix A)
Model counts are derived from models.csv. Nothing is retrained or re-evaluated.
The script only writes inside paper/ (the document and the figures in paper/figures/).
"""

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Inches, Pt, RGBColor  # noqa: E402
from PIL import Image  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
from xai.stats import bh, clopper_pearson, format_p, pava  # noqa: E402

DR = ROOT / "results" / "dose_response"
AN = ROOT / "results" / "analysis"
EX = ROOT / "results" / "explanations"
FIG_OUT = PAPER / "figures"
FIG_SUB = PAPER / "figures_submission"  # Figure_N.pdf / Figure_N.png for the journal upload (--submission only)
FIG_SRC = FIG_SUB / "_src"  # intermediate vector and 300 dpi sources of the regenerated figures (git-ignored)
# Modes: default = the guide-review draft (unchanged); --submission = journal version (main text
# plus a separate supplementary document); --final = submission build that fails while any [[...]] marker remains.
FINAL = "--final" in sys.argv
SUBMISSION = FINAL or "--submission" in sys.argv
OUTPUT = PAPER / ("paper_submission.docx" if SUBMISSION else "paper_draft.docx")
SUPP_OUTPUT = PAPER / "supplementary_submission.docx"
HIGHLIGHTS_OUTPUT = PAPER / "highlights_submission.txt"
SUPP_ORDER = ["prereg", "matched", "pairs", "legacy", "sets"]  # Supplementary Tables S1..S5


def S(draft, submission):
    """Text that differs between the guide-review draft and the journal submission."""
    return submission if SUBMISSION else draft


def tref(key, number):
    """Reference to a table that is in the main text in the draft and in the supplement in the submission."""
    return f"Supplementary Table S{SUPP_ORDER.index(key) + 1}" if SUBMISSION else f"Table {number}"

KEYWORDS_DRAFT = ["Explainable AI", "Faithfulness metrics", "Deletion and insertion", "Shortcut learning",
                  "Meta-evaluation", "Pre-registration"]
# Elsevier: 1 to 7 keywords, no multi-word keyword with "and" or "of"
KEYWORDS = ["Explainable AI", "Faithfulness metrics", "Perturbation metrics", "Shortcut learning",
            "Meta-evaluation", "Pre-registration"]

TITLE = ("A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut "
         "Reliance in CNN Image Classifiers")
DRAFT_NOTICE = "DRAFT for guide review \u2014 not for submission"
DASH = "\u2013"
MINUS = "\u2212"
RC = "r\u1d04"  # r_c (small-capital c) for chance-corrected reliance
SEARCH_AUDIT = "2019 to October 2026 (literature audit committed on 2026-11-13, docs/audit/02_literature_novelty.md)"
SEARCH_RECHECK = "2026-11-13"

# ------------------------------------------------------------------ data

MODELS = pd.read_csv(DR / "models.csv")
PREF = pd.read_csv(DR / "preference.csv")
RESP = pd.read_csv(DR / "responsiveness_corrected.csv")
RESP_ORIG = pd.read_csv(DR / "original_models" / "responsiveness_corrected.csv")
RESP_PRE_A2 = pd.read_csv(DR / "pre_a2_models" / "responsiveness_corrected.csv")
LEGACY = pd.read_csv(DR / "responsiveness.csv")
METHOD = pd.read_csv(DR / "method_responsiveness_corrected.csv")
EFFECT = pd.read_csv(DR / "effect_sizes.csv")
PAIRS = pd.read_csv(DR / "metric_pairs.csv")
# P1-P4 are judged on the 18 original Pet models only (amendment A2), i.e. without the A2 Pet models.
PET = pd.read_csv(DR / "pre_a2_models" / "pet_preregistered.csv")
CROSSED_PRE_A2 = pd.read_csv(DR / "pre_a2_models" / "intervals_crossed.csv")
BEH = pd.read_csv(DR / "behaviour_summary.csv")
BAND = pd.read_csv(DR / "behaviour_by_reliance.csv")
CONF = pd.read_csv(DR / "confound.csv")
# audit-2 additions (new files; the tables above are unchanged)
CROSSED = pd.read_csv(DR / "intervals_crossed.csv")
BSWITCH = pd.read_csv(DR / "behaviour_switch.csv")
SWDIFF = pd.read_csv(DR / "switch_differences.csv")
OVER = pd.read_csv(DR / "overdetection.csv")
EXCESS = pd.read_csv(DR / "excess_preference_models.csv")
CALIB = pd.read_csv(DR / "behaviour_calibration.csv")
CONF_RC = pd.read_csv(DR / "confound_rc.csv")
CORRECT = pd.read_csv(DR / "correct_only.csv")
POWER = pd.read_csv(DR / "behaviour_power.csv")
# amendment A3 (SD3 exact ROAD; SD4 watermark): new files of analyze (model set all); submission only
if SUBMISSION:
    ROAD_RESP = pd.read_csv(DR / "responsiveness_corrected_road.csv")
    ROAD_OVER = pd.read_csv(DR / "overdetection_road.csv")
    ROAD_MODELS = pd.read_csv(DR / "road_models.csv")
    ROAD_P10 = pd.read_csv(DR / "road_p10.csv")
    ROAD_P11 = pd.read_csv(DR / "road_p11.csv")
    ROAD_TIMING = pd.read_csv(DR / "road_timing_kaggle_s3.csv")
ROAD = ["ROAD (MoRF)", "ROAD (LeRF)"]
KENDALL = pd.read_csv(AN / "kendall_exact.csv")
GT_TAU = pd.read_csv(AN / "gt_agreement.csv")
GT_INSTANCE = pd.read_csv(AN / "gt_instance.csv")
ACCURACY = pd.read_csv(AN / "accuracy.csv").set_index("dataset")
DECOY_MNIST = pd.read_csv(AN / "decoy_mnist_summary.csv").set_index("method")

METRICS = ["Deletion (black)", "Deletion (blur)", "Deletion (inpaint)", "Insertion (blur)"]
METRIC_COLUMNS = {"Deletion (black)": "deletion_zero", "Deletion (blur)": "deletion_blur",
                  "Deletion (inpaint)": "deletion_inpaint", "Insertion (blur)": "insertion_blur"}
BB = METRICS[:2]
II = METRICS[2:]
METHODS = ["Saliency", "SmoothGrad", "Integrated Gradients", "Grad-CAM", "Occlusion"]
PRIMARY_LABEL = "follows_shortcut_causal"
BEHAVIOUR_LABELS = ["follows_shortcut_causal", "follows_shortcut", "shortcut_dominant"]  # primary first
LABEL_NAMES = {"follows_shortcut_causal": "causal (primary)", "follows_shortcut": "moved-only (check)",
               "shortcut_dominant": "shortcut dominance (check)"}
# Natural-image settings first (Pet is the primary natural-image evidence), then the
# mechanism settings.
SETTINGS = [  # (base, shortcut, label, short label)
    ("pet_resnet18", "patch", "Oxford-IIIT Pet, patch (ResNet-18)", "Pet"),
    ("cifar10_resnet18", "patch", "CIFAR-10, patch (ResNet-18)", "CIFAR ResNet-18"),
    ("cifar10", "patch", "CIFAR-10, patch (small CNN)", "CIFAR CNN"),
    ("mnist", "patch", "MNIST, patch", "MNIST patch"),
    ("mnist", "texture", "MNIST, texture", "MNIST texture"),
    ("pneumonia_mnist", "patch", "PneumoniaMNIST, patch", "Pneumonia patch"),
    ("pneumonia_mnist", "texture", "PneumoniaMNIST, texture", "Pneumonia texture"),
]
NATURAL = SETTINGS[:3]
MECHANISM = SETTINGS[3:]
# Amendment A2, SD1: the ImageNet-pretrained Pet network is a separate setting. It is never pooled
# with pet_resnet18 and is kept out of SETTINGS, so every main-study table, count and figure covers
# the seven settings above; SD1 has its own section and tables (Section 5.10).
SD1 = ("pet_resnet18_pretrained", "patch", "Oxford-IIIT Pet, patch (ImageNet-pretrained ResNet-18, SD1)",
       "Pet pretrained (SD1)")
SD1_KEY = SD1[:2]
# Amendment A3, SD4: the watermark shortcut on the SD1 network, a further separate setting (Section 5.12),
# never pooled with SD1. The guide-review draft predates A3 and keeps the models without SD4.
SD4 = ("pet_resnet18_pretrained", "watermark", "Oxford-IIIT Pet, watermark (ImageNet-pretrained ResNet-18, SD4)",
       "Pet watermark (SD4)")
SD4_KEY = SD4[:2]
MODELS_ALL = MODELS
if not SUBMISSION:
    MODELS = MODELS[[(b, s) != SD4_KEY for b, s in zip(MODELS.base, MODELS.shortcut)]].reset_index(drop=True)
LABEL = {(b, s): label for b, s, label, _ in SETTINGS + [SD1, SD4]}
SHORT = {(b, s): short for b, s, _, short in SETTINGS + [SD1, SD4]}
NETS = {"mnist": "Small CNN", "pneumonia_mnist": "Small CNN", "cifar10": "CIFAR CNN (0.77M)",
        "cifar10_resnet18": "ResNet-18 (11.2M)", "pet_resnet18": "ResNet-18 (11.2M)",
        "pet_resnet18_pretrained": "ImageNet-pretrained ResNet-18 (11.2M)"}
BASE_NAMES = {"mnist": "MNIST", "pneumonia_mnist": "PneumoniaMNIST", "cifar10": "CIFAR-10 (small CNN)",
              "cifar10_resnet18": "CIFAR-10 (ResNet-18)", "pet_resnet18": "Oxford-IIIT Pet (ResNet-18)",
              "pet_resnet18_pretrained": "Oxford-IIIT Pet (ImageNet-pretrained ResNet-18)"}
FULL, MATCHED = "shortcut vs full object", "shortcut vs size-matched object"
S_RANDOM, S_PLACEBO = "shortcut vs random", "shortcut vs placebo"
R_OBJECT, P_OBJECT, P_RANDOM = "random vs full object", "placebo vs full object", "placebo vs random"
SHORTCUT_COMPARISONS = [FULL, MATCHED, S_PLACEBO, S_RANDOM]

assert set(zip(MODELS.base, MODELS.shortcut)) == {(b, s) for b, s, _, _ in SETTINGS + [SD1] + [SD4] * SUBMISSION}, \
    "SETTINGS, SD1 and SD4 must cover every (base, shortcut) in models.csv"


def main_only(df):
    """Rows of the seven main settings (drops the separate SD1 and SD4 settings)."""
    return df[[(b, s) not in (SD1_KEY, SD4_KEY) for b, s in zip(df.base, df.shortcut)]]


def _one(df, **keys):
    sel = df
    for k, v in keys.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, (keys, len(sel))
    return sel.iloc[0]


def row(comparison, base, shortcut, metric, table=None):
    """One row of responsiveness_corrected.csv (primary, r_c axis)."""
    return _one(RESP if table is None else table, comparison=comparison, base=base,
                shortcut=shortcut, metric=metric)


def rows(comparisons, settings, metrics=METRICS, table=None):
    t = RESP if table is None else table
    keys = {(b, s) for b, s, *_ in settings}
    sel = t[t.comparison.isin(comparisons) & t.metric.isin(metrics)
            & np.array([(b, s) in keys for b, s in zip(t.base, t.shortcut)])]
    assert len(sel) == len(comparisons) * len(settings) * len(metrics), (comparisons, len(sel))
    return sel


def legacy(comparison, base, shortcut, metric):
    return _one(LEGACY, comparison=comparison, base=base, shortcut=shortcut, metric=metric)


def mrow(base, shortcut, method):
    return _one(METHOD, base=base, shortcut=shortcut, method=method)


def brow(comparison, base, shortcut, metric, label=PRIMARY_LABEL):
    return _one(BEH, comparison=comparison, base=base, shortcut=shortcut, metric=metric, label=label)


def crow(base, shortcut, metric):
    return _one(CONF, base=base, shortcut=shortcut, metric=metric)


def xrow(comparison, base, shortcut, metric):
    """One row of intervals_crossed.csv (crossed bootstrap, BCa intervals for rho)."""
    return _one(CROSSED, comparison=comparison, base=base, shortcut=shortcut, metric=metric)


def swrow(base, shortcut, label=PRIMARY_LABEL):
    """One row of behaviour_switch.csv (behavioural switch with its status)."""
    return _one(BSWITCH, base=base, shortcut=shortcut, label=label)


def drow(comparison, base, shortcut, metric, label=PRIMARY_LABEL):
    """One row of switch_differences.csv (paired behavioural - metric switch)."""
    return _one(SWDIFF, comparison=comparison, base=base, shortcut=shortcut, metric=metric, label=label)


def orow(base, shortcut, metric):
    """One row of overdetection.csv (shortcut vs full object only)."""
    return _one(OVER, base=base, shortcut=shortcut, metric=metric)


def margin_bands(base, shortcut, metric):
    """Margin over-detection per r_c band, {band: (k, n)}, from the per-model
    decisions in excess_preference_models.csv (bands as in overdetection.csv)."""
    sel = EXCESS[(EXCESS.base == base) & (EXCESS.shortcut == shortcut) & (EXCESS.metric == metric)]
    rc, says = sel.reliance_corrected, sel.says_shortcut_margin.astype(bool)
    masks = {"rc_le10": rc <= 10, "rc_10_50": (rc > 10) & (rc <= 50)}
    return {band: (int((says & mask).sum()), int(mask.sum())) for band, mask in masks.items()}


def calrow(comparison, base, shortcut, metric, label=PRIMARY_LABEL):
    return _one(CALIB, comparison=comparison, base=base, shortcut=shortcut, metric=metric, label=label)


def ccrow(base, shortcut, metric):
    """One row of confound_rc.csv (partial rho on r_c with BH)."""
    return _one(CONF_RC, base=base, shortcut=shortcut, metric=metric)


def korow(base, shortcut, metric):
    """One row of correct_only.csv (correctly classified images only)."""
    return _one(CORRECT, base=base, shortcut=shortcut, metric=metric)


def powrow(base, shortcut, label=PRIMARY_LABEL):
    return _one(POWER, base=base, shortcut=shortcut, label=label)


def prow(comparison, base, shortcut, a, b):
    return _one(PAIRS, comparison=comparison, base=base, shortcut=shortcut, metric_a=a, metric_b=b)


def erow(comparison, base, shortcut, metric, level):
    return _one(EFFECT, comparison=comparison, base=base, shortcut=shortcut, metric=metric, level=level)


def pet_row(prediction, comparison, item, statistic_start):
    sel = PET[(PET.prediction == prediction) & (PET.comparison == comparison)
              & (PET.metric_or_method == item) & PET.statistic.str.startswith(statistic_start)]
    assert len(sel) == 1, (prediction, comparison, item, statistic_start)
    return sel.iloc[0]


# ------------------------------------------------------------------ formatting


def _fmt(x, digits):
    s = f"{x:.{digits}f}"
    if float(s) == 0:  # no negative zero
        s = f"{0:.{digits}f}"
    return s.replace("-", MINUS)


def f1(x):
    return _fmt(x, 1)


def f2(x):
    return _fmt(x, 2)


def f3(x):
    return _fmt(x, 3)


def pct(x):
    """A share (0-1) as a whole percentage, rounded like f2 so text and tables agree."""
    return f"{100 * float(f'{x:.2f}'):.0f}%"


def span(values, fmt=f2, unit=""):
    """'a-b' range (a single value if both ends print the same)."""
    values = [v for v in values if not pd.isna(v)]
    lo, hi = fmt(min(values)), fmt(max(values))
    return f"{lo}{unit}" if lo == hi else f"{lo}{DASH}{hi}{unit}"


def pct_span(values):
    """'a-b%' range of shares (0-1) as whole percentages."""
    values = [v for v in values if not pd.isna(v)]
    lo, hi = pct(min(values))[:-1], pct(max(values))[:-1]
    return f"{lo}%" if lo == hi else f"{lo}{DASH}{hi}%"


def signed_span(values, fmt=f2):
    values = [v for v in values if not pd.isna(v)]
    return f"{fmt(min(values))} to {fmt(max(values))}"


def ci(lo, hi, fmt=f2):
    if pd.isna(lo) or pd.isna(hi):
        return "[\u2013]"
    return f"[{fmt(lo)}, {fmt(hi)}]"


def p_text(p):
    """'p < 0.001' or 'p = 0.012' (xai.stats.format_p)."""
    s = format_p(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


STATUS_MARK = {"identified": "", "identified at lowest-reliance edge": " (e)"}


def fmt_switch(r, column="switch_iso_rc", status="switch_status", lo="switch_iso_rc_ci_low",
               hi="switch_iso_rc_ci_high", bracketed="switch_bracketed"):
    value, st = r[column], r[status]
    if pd.isna(value):
        return {"above 0.5 at lowest reliance": "none (> 0.5 at lowest)",
                "no crossing": "none (no crossing)", "not applicable": "n/a"}.get(st, "none")
    mark = STATUS_MARK.get(st, " (n.i.)")
    if bracketed in r and r[bracketed] is not None and not pd.isna(r[bracketed]) and not bool(r[bracketed]):
        mark += " (nb)"
    interval = "" if lo is None else " " + ci(r[lo], r[hi], f1)
    return f"{f1(value)}{interval}{mark}"


def switch_text(r, unit="%", note=None):
    """behaviour_switch.csv report_text with the unit after the number:
    '54.5% (not bracketed)', '> 42.7% (not identified; no model between ...)'.
    An optional note is added inside the parentheses."""
    head, sep, tail = str(r.report_text).partition(" (")
    number = head.lstrip("> ")
    expected = r.behavioural_switch_rc - (r.distance_below if head.startswith(">") else 0)
    assert number == f"{expected:.1f}", (r.report_text, expected)
    inner = tail[:-1] if sep else ""
    if note:
        inner = f"{inner}; {note}" if inner else note
    return head.replace("-", MINUS) + unit + (f" ({inner})" if inner else "")


def sig(r, p_col="p_bh"):
    return "" if r[p_col] < 0.05 else "\u2020"


# ------------------------------------------------------------------ counts (from models.csv)

N_MODELS = len(MODELS)
N_ORIGINAL = int((MODELS.model_set == "original").sum())
N_A1 = int((MODELS.model_set == "A1").sum())
# Amendment A2 models pooled so far (scored ones only; dose_response.py analyze
# lists the pilots and unscored A2 models in pilots.csv, not in models.csv).
N_A2 = int((MODELS.model_set == "A2").sum())
N_A3 = int((MODELS.model_set == "A3").sum())  # the SD4 confirmatory models (submission only)
assert N_ORIGINAL + N_A1 + N_A2 + N_A3 == N_MODELS, set(MODELS.model_set)
SD4_MODELS = MODELS[MODELS.model_set == "A3"]
N_SD4 = len(SD4_MODELS)
assert N_SD4 == (18 if SUBMISSION else 0) and set(zip(SD4_MODELS.base, SD4_MODELS.shortcut)) <= {SD4_KEY}
A2_MODELS = MODELS[MODELS.model_set == "A2"]
A2_COUNTS = A2_MODELS.groupby("base").size().to_dict()
# The text below describes the complete A2 design: 23 SD2 models and 18 SD1 confirmatory models.
assert A2_COUNTS == {"pet_resnet18": 3, "cifar10_resnet18": 12, "cifar10": 8, "pet_resnet18_pretrained": 18}, \
    A2_COUNTS
A2_SD2 = A2_MODELS[A2_MODELS.base != SD1[0]]
N_A2_SD2 = len(A2_SD2)
N_SD1 = A2_COUNTS[SD1[0]]
N_MAIN = N_MODELS - N_SD1 - N_SD4  # models of the seven main settings
A2_PET = A2_MODELS[A2_MODELS.base == "pet_resnet18"]
N_A2_PET = len(A2_PET)
A2_PET_DOSES = sorted(A2_PET.dose.unique())
A2_CIFAR_DOSES = sorted(A2_MODELS[A2_MODELS.base == "cifar10_resnet18"].dose.unique())
assert A2_CIFAR_DOSES == sorted(A2_MODELS[A2_MODELS.base == "cifar10"].dose.unique())
PILOTS = pd.read_csv(DR / "pilots.csv") if (DR / "pilots.csv").exists() else MODELS.iloc[:0].assign(status="")
_KEYS = ["base", "shortcut", "dose", "seed"]
assert not len(PILOTS.merge(MODELS[_KEYS], on=_KEYS)), "pilots are never pooled"
PRE_A2_SWITCH = pd.read_csv(DR / "pre_a2_models" / "behaviour_switch.csv")
N_BASE = MODELS.groupby("base").size().to_dict()
N_SETTING = MODELS.groupby(["base", "shortcut"]).size().to_dict()
SEEDS = {b: sorted(MODELS[MODELS.base == b].seed.unique()) for b in MODELS.base.unique()}
DOSES = {b: sorted(MODELS[MODELS.base == b].dose.unique()) for b in MODELS.base.unique()}
EPOCHS = MODELS.groupby("base").epochs.first().to_dict()
MINUTES = MODELS.groupby("base").training_minutes.mean().to_dict()
N_SETTINGS = len(SETTINGS)
N_CELLS = N_SETTINGS * len(METRICS)
LEVELS = MODELS.groupby(["base", "shortcut", "dose"])[
    ["reliance", "reliance_corrected", "test_accuracy", "test_accuracy_no_shortcut"]].mean().reset_index()


def levels(base, shortcut):
    return LEVELS[(LEVELS.base == base) & (LEVELS.shortcut == shortcut)].sort_values("dose")


def group(base, shortcut):
    return MODELS[(MODELS.base == base) & (MODELS.shortcut == shortcut)]


def dose_list(base):
    return ", ".join(f"{d:g}" for d in DOSES[base]).replace("-", MINUS)


def behavioural_switch(base, shortcut):
    """r_c at which the per-image follow rate crosses 0.5 (same for every metric).
    Its identification status is in behaviour_switch.csv (swrow)."""
    value = brow(FULL, base, shortcut, METRICS[0]).observed_switch_rc
    assert value == swrow(base, shortcut).behavioural_switch_rc, (base, shortcut)
    return value


# ------------------------------------------------------------------ references

REFERENCES = {
    # explanation methods, metrics and tools
    "simonyan": "Simonyan K, Vedaldi A, Zisserman A (2014) Deep inside convolutional networks: visualising image classification models and saliency maps. In: ICLR Workshop",
    "smilkov": "Smilkov D, et al. (2017) SmoothGrad: removing noise by adding noise. arXiv:1706.03825",
    "sundararajan": "Sundararajan M, Taly A, Yan Q (2017) Axiomatic attribution for deep networks. In: Proceedings of the International Conference on Machine Learning (ICML)",
    "selvaraju": "Selvaraju RR, et al. (2017) Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV)",
    "zeiler": "Zeiler MD, Fergus R (2014) Visualizing and understanding convolutional networks. In: Proceedings of the European Conference on Computer Vision (ECCV)",
    "captum": "Kokhlikyan N, et al. (2020) Captum: a unified and generic model interpretability library for PyTorch. arXiv:2009.07896",
    "samek": "Samek W, et al. (2017) Evaluating the visualization of what a deep neural network has learned. IEEE Transactions on Neural Networks and Learning Systems",
    "petsiuk": "Petsiuk V, Das A, Saenko K (2018) RISE: randomized input sampling for explanation of black-box models. In: Proceedings of the British Machine Vision Conference (BMVC)",
    "adebayo2018": "Adebayo J, et al. (2018) Sanity checks for saliency maps. In: Advances in Neural Information Processing Systems (NeurIPS)",
    "he": "He K, Zhang X, Ren S, Sun J (2016) Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp 770\u2013778",
    # data
    "medmnist": "Yang J, et al. (2023) MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data",
    "kermany": "Kermany DS, et al. (2018) Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell",
    "pets": "Parkhi OM, Vedaldi A, Zisserman A, Jawahar CV (2012) Cats and dogs. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/",
    # controlled reliance and ground-truth benchmarks
    "ross": "Ross AS, Hughes MC, Doshi-Velez F (2017) Right for the right reasons: training differentiable models by constraining their explanations. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)",
    "bam": "Yang M, Kim B (2019) Benchmarking attribution methods with relative feature importance. arXiv:1907.09701",
    "smerf": "Kim JS, Plumb G, Talwalkar A (2022) Sanity simulations for saliency methods. In: Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162:11173\u201311200",
    "bastings": "Bastings J, Ebert S, Zablotskaia P, Sandholm A, Filippova K (2022) \u201cWill you find these shortcuts?\u201d A protocol for evaluating the faithfulness of input salience methods for text classification. In: Proceedings of EMNLP 2022, pp 976\u2013991",
    "zhou": "Zhou Y, et al. (2022) Do feature attribution methods correctly attribute features? In: Proceedings of the AAAI Conference on Artificial Intelligence. arXiv:2104.14403",
    "lin": "Lin Y-S, Lee W-C, Celik ZB (2021) What do you see? Evaluation of explainable artificial intelligence (XAI) interpretability through neural backdoors. In: Proceedings of KDD 2021. arXiv:2009.10639",
    "backx": "Yang P, Akhtar N, Jiang J, Mian A (2024) A backdoor-based explainable AI benchmark for high fidelity evaluation of attributions (BackX). arXiv:2405.02344",
    "adebayo2022": "Adebayo J, et al. (2022) Post hoc explanations may be ineffective for detecting unknown spurious correlation. In: International Conference on Learning Representations (ICLR)",
    "sun": "Sun S, et al. (2023) Right for the wrong reason: can interpretable ML techniques detect spurious correlations? In: Medical Image Computing and Computer Assisted Intervention (MICCAI). arXiv:2307.12344",
    "arras": "Arras L, Osman A, Samek W (2022) CLEVR-XAI: a benchmark dataset for the ground truth evaluation of neural network explanations. Information Fusion",
    "funnybirds": "Hesse R, Schaub-Meyer S, Roth S (2023) FunnyBirds: a synthetic vision dataset for a part-based analysis of explainable AI methods. In: Proceedings of ICCV 2023. arXiv:2308.06248",
    "xaiunits": "Lee JR, Emami S, Hollins MD, Wong TCH, Villalobos S\u00e1nchez CI, Toni F, Zhang D, Dejl A (2025) XAI-Units: benchmarking explainability methods with unit tests. In: Proceedings of ACM FAccT 2025. arXiv:2506.01059",
    "attributionlab": "Zhang Y, Li Y, Brown H, Rezaei M, Bischl B, Torr P, Khakzar A, Kawaguchi K (2023) AttributionLab: faithfulness of feature attribution under controllable environments. NeurIPS 2023 Workshop XAI in Action (XAIA). arXiv:2310.06514",
    "miro2026": "Mir\u00f3-Nicolau M, Spinnato F, Guidotti R (2026) A synthetic ground-truth framework for the evaluation of explainable AI methods. arXiv:2609.30397",
    "oscar": "Achara A, Triantafillou P, Puyol-Ant\u00f3n E, Hammers A, King AP (2025) Localising shortcut learning in pixel space via ordinal scoring correlations for attribution representations (OSCAR). arXiv:2512.18888",
    # meta-evaluation of metrics
    "tomsett": "Tomsett R, et al. (2020) Sanity checks for saliency metrics. In: Proceedings of the AAAI Conference on Artificial Intelligence. arXiv:1912.01451",
    "metaquantus": "Hedstr\u00f6m A, Bommer P, Wickstr\u00f8m KK, Samek W, Lapuschkin S, H\u00f6hne MMC (2023) The meta-evaluation problem in explainable AI: identifying reliable estimators with MetaQuantus. Transactions on Machine Learning Research. arXiv:2302.07265",
    "quantus": "Hedstr\u00f6m A, Weber L, Krakowczyk D, Bareeva D, Motzkus F, Samek W, Lapuschkin S, H\u00f6hne MMC (2023) Quantus: an explainable AI toolkit for responsible evaluation of neural network explanations and beyond. Journal of Machine Learning Research 24(34):1\u201311",
    "latec": "Klein L, L\u00fcth CT, Schlegel U, Bungert TJ, El-Assady M, J\u00e4ger PF (2024) Navigating the maze of explainable AI: a systematic approach to evaluating methods and metrics. In: Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track. arXiv:2409.16756",
    "m4": "Li X, Du M, Chen J, Chai Y, Lakkaraju H, Xiong H (2023) M4: a unified XAI benchmark for faithfulness evaluation of feature attribution methods across metrics, modalities and models. In: Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track",
    "gevaert": "Gevaert A, Rousseau A-J, Becker T, Valkenborg D, De Bie T, Saeys Y (2024) Evaluating feature attribution methods in the image domain. Machine Learning 113(9):6019\u20136064. https://doi.org/10.1007/s10994-024-06550-x",
    "fidelity": "Mir\u00f3-Nicolau M, Jaume-i-Cap\u00f3 A, Moy\u00e0-Alcover G (2024) A comprehensive study on fidelity metrics for XAI. arXiv:2401.10640",
    "refinecam": "Domeniconi L, Stramiglio A, Lombardi M, Salti S (2026) How to evaluate and refine your CAM. Accepted at the International Conference on Pattern Recognition (ICPR 2026). arXiv:2605.14641",
    "hesse2024": "Hesse R, Schaub-Meyer S, Roth S (2024) Benchmarking the attribution quality of vision models. In: NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2407.11910",
    "binder": "Binder A, et al. (2023) Shortcomings of top-down randomization-based sanity checks for evaluations of deep neural network explanations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    "sanityrevisited": "Hedstr\u00f6m A, Weber L, Lapuschkin S, H\u00f6hne MMC (2023) Sanity checks revisited: an exploration to repair the model parameter randomisation test. NeurIPS 2023 Workshop XAI in Action (XAIA). arXiv:2401.06465",
    "freshlook": "Hedstr\u00f6m A, Weber L, Lapuschkin S, H\u00f6hne M (2024) A fresh look at sanity checks for saliency maps. In: Explainable Artificial Intelligence (xAI 2024). Communications in Computer and Information Science, Springer, pp 403\u2013420. https://doi.org/10.1007/978-3-031-63787-2_21 (extends arXiv:2401.06465)",
    # confounds of perturbation metrics
    "rong": "Rong Y, Leemann T, Borisov V, Kasneci G, Kasneci E (2022) A consistent and efficient evaluation strategy for attribution methods (ROAD). In: Proceedings of the International Conference on Machine Learning (ICML). arXiv:2202.00449",
    "roar": "Hooker S, Erhan D, Kindermans P-J, Kim B (2019) A benchmark for interpretability methods in deep neural networks. In: Advances in Neural Information Processing Systems (NeurIPS). arXiv:1806.10758",
    "roarpitfalls": "Song J, Cha K, Seo J (2026) On pitfalls of RemOve-And-Retrain: data processing inequality perspective. ICML 2026 Workshop on Mechanistic Interpretability. arXiv:2304.13836",
    "blucher": "Bl\u00fccher S, Vielhaben J, Strodthoff N (2024) Decoupling pixel flipping and occlusion strategy for consistent XAI benchmarks. Transactions on Machine Learning Research. arXiv:2401.06654",
    "baseline": "Picard AM, Boissin T, Subhash V, Cad\u00e8ne R, Fel T (2025) Back to the baseline: examining baseline effects on explainability metrics. arXiv:2512.11433",
    "aim": "Hsieh C-Y, Fang H-Y, Wei C-S (2026) AIM: adversarial information masking for faithfulness evaluation of saliency maps. arXiv:2605.16905",
    "flexibility": "Wickstr\u00f8m K, H\u00f6hne MM-C, Hedstr\u00f6m A (2024) From flexibility to manipulation: the slippery slope of XAI evaluation. ECCV 2024 Workshop on Explainable Computer Vision. arXiv:2412.05592",
    "gomez": "Gomez T, Fr\u00e9our T, Mouch\u00e8re H (2022) Metrics for saliency map evaluation of deep learning explanation methods. In: Pattern Recognition and Artificial Intelligence (ICPRAI 2022). Lecture Notes in Computer Science, Springer, pp 84\u201395. https://doi.org/10.1007/978-3-031-09037-0_8",
    "saco": "Wu J, Kang W, Tang H, Hong Y, Yan Y (2024) On the faithfulness of vision transformer explanations. In: Proceedings of CVPR 2024. arXiv:2404.01415",
    "bilodeau": "Bilodeau B, Jaques N, Koh PW, Kim B (2024) Impossibility theorems for feature attribution. Proceedings of the National Academy of Sciences 121(2). arXiv:2212.11870",
    # shortcuts in practice
    "lapuschkin": "Lapuschkin S, W\u00e4ldchen S, Binder A, Montavon G, Samek W, M\u00fcller K-R (2019) Unmasking Clever Hans predictors and assessing what machines really learn. Nature Communications. arXiv:1902.10178",
    "degrave": "DeGrave AJ, Janizek JD, Lee S-I (2021) AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence 3(7):610\u2013619. https://doi.org/10.1038/s42256-021-00338-7",
    "hermann": "Hermann KL, Mobahi H, Fel T, Mozer MC (2024) On the foundations of shortcut learning. In: International Conference on Learning Representations (ICLR). arXiv:2310.16228",
    "suhail": "Suhail P, Goel V, Sethi A (2025) Shortcut learning susceptibility in vision classifiers. arXiv:2502.09150",
    # statistics
    "bouthillier": "Bouthillier X, et al. (2021) Accounting for variance in machine learning benchmarks. In: MLSys 2021. arXiv:2103.03098",
    "agarwal": "Agarwal R, Schwarzer M, Castro PS, Courville A, Bellemare MG (2021) Deep reinforcement learning at the edge of the statistical precipice. In: Advances in Neural Information Processing Systems (NeurIPS). arXiv:2108.13264",
    "saravanan": "Saravanan V, Berman GJ, Sober SJ (2020) Application of the hierarchical bootstrap to multi-level data in neuroscience. Neurons, Behavior, Data analysis, and Theory 3(5). arXiv:2007.07797",
}
if SUBMISSION:
    # Elsevier asks datasets and software to be cited with author, title, repository or venue, version, year and
    # identifier. Entries below were checked against the official pages (see docs/submission/DEFERRED.md D31).
    REFERENCES.update({
        "lecun": "LeCun Y, Bottou L, Bengio Y, Haffner P (1998) Gradient-based learning applied to document "
                 "recognition. Proceedings of the IEEE 86(11):2278\u20132324. https://doi.org/10.1109/5.726791. "
                 "MNIST database: https://yann.lecun.com/exdb/mnist/",
        "cifar": "Krizhevsky A (2009) Learning multiple layers of features from tiny images. Technical report, "
                 "University of Toronto. CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html",
        "pytorch": "Paszke A, et al. (2019) PyTorch: an imperative style, high-performance deep learning library. "
                   "In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019). Software version 2.14.0. "
                   "arXiv:1912.01703",
        "torchvision": "TorchVision maintainers and contributors (2016) TorchVision: PyTorch\u2019s computer vision "
                       "library. GitHub repository, software version 0.29.0. https://github.com/pytorch/vision",
        "roadcode": "Rong Y, Leemann T, Borisov V, Kasneci G, Kasneci E (2022) road_evaluation: source code of the ROAD "
                    "benchmark for feature attribution methods. GitHub repository, revision e6b9987, MIT licence. "
                    "https://github.com/tleemann/road_evaluation",
        "code": "[dataset] Shaikh MJF (2026) Data and code for: " + TITLE + ". Zenodo, version 1.0.0. "
                "https://doi.org/10.5281/zenodo.24551101",
    })
    REFERENCES["medmnist"] = ("Yang J, et al. (2023) MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D "
                              "biomedical image classification. Scientific Data 10:41. "
                              "https://doi.org/10.1038/s41597-022-01721-8. Dataset and code: https://medmnist.com/ "
                              "(software version 3.0.2)")
    REFERENCES["pets"] = ("Parkhi OM, Vedaldi A, Zisserman A, Jawahar CV (2012) Cats and dogs. In: IEEE Conference on "
                          "Computer Vision and Pattern Recognition (CVPR), pp 3498\u20133505. Oxford-IIIT Pet dataset "
                          "(CC BY-SA 4.0): https://www.robots.ox.ac.uk/~vgg/data/pets/")
    REFERENCES["captum"] = ("Kokhlikyan N, et al. (2020) Captum: a unified and generic model interpretability library "
                            "for PyTorch. arXiv:2009.07896. Software version 0.9.0, https://github.com/pytorch/captum")
CITED: list[str] = []


def cite(*keys):
    """'[n, m]' with references numbered in order of first citation."""
    numbers = []
    for key in keys:
        assert key in REFERENCES, key
        if key not in CITED:
            CITED.append(key)
        numbers.append(CITED.index(key) + 1)
    return "[" + ", ".join(str(n) for n in sorted(numbers)) + "]"


# ------------------------------------------------------------------ figures


def rearrange_panels(source, target, panels, columns=4, order=None):
    """The analyze figures put one panel per setting in a single very wide row,
    which is unreadable at page width. Cut the panels at the widest white gaps and
    lay them out in a grid below the original title. Pixels are only moved, not
    changed. Falls back to the original file if the panels can't be found."""
    image = Image.open(source).convert("RGB")
    a = np.asarray(image)
    ink = (a < 245).any(axis=2)
    rows_ink = ink.any(axis=1)
    first = int(np.argmax(rows_ink))
    gap = first + int(np.argmax(~rows_ink[first:]))
    title_end = gap + int(np.argmax(rows_ink[gap:]))
    body = ink[title_end:]
    cols_ink = body.any(axis=0)
    gaps, start = [], None
    inked = np.flatnonzero(cols_ink)
    left, right = inked.min(), inked.max()
    for x in range(left, right + 1):
        if not cols_ink[x] and start is None:
            start = x
        elif cols_ink[x] and start is not None:
            gaps.append((x - start, start, x))
            start = None
    if len(gaps) < panels - 1:
        return source
    cuts = sorted((s + e) // 2 for _, s, e in sorted(gaps, reverse=True)[:panels - 1])
    edges = [left] + cuts + [right + 1]
    pieces = [image.crop((edges[i], title_end, edges[i + 1], a.shape[0])) for i in range(panels)]
    if min(p.width for p in pieces) < 0.6 * max(p.width for p in pieces):
        return source

    def frame_x(piece):
        dark_share = (np.asarray(piece.convert("L")) < 100).mean(axis=0)
        found = np.flatnonzero(dark_share > 0.5)
        return int(found[0]) if len(found) else None

    frames = [frame_x(p) for p in pieces]
    strip = None
    if all(f is not None for f in frames):
        strip = pieces[0].crop((0, 0, frames[0], pieces[0].height))
        pieces = [p.crop((f, 0, p.width, p.height)) for p, f in zip(pieces, frames)]
    if order is not None:  # new position i shows the panel that was at index order[i]
        pieces = [pieces[i] for i in order]
    title = image.crop((0, 0, a.shape[1], title_end))
    t_ink = np.flatnonzero((np.asarray(title) < 245).any(axis=2).any(axis=0))
    title = title.crop((max(t_ink.min() - 10, 0), 0, t_ink.max() + 10, title.height))
    cell_w, cell_h = max(p.width for p in pieces), max(p.height for p in pieces)
    strip_w = strip.width if strip is not None else 0
    width = max(strip_w + columns * cell_w, title.width)
    grid_rows = int(np.ceil(len(pieces) / columns))
    canvas = Image.new("RGB", (width, title.height + grid_rows * cell_h + 10), "white")
    canvas.paste(title, ((width - title.width) // 2, 0))
    for r in range(grid_rows):
        if strip is not None:
            canvas.paste(strip, (0, title.height + 10 + r * cell_h))
    for i, piece in enumerate(pieces):
        r, c = divmod(i, columns)
        canvas.paste(piece, (strip_w + c * cell_w, title.height + 10 + r * cell_h))
    FIG_OUT.mkdir(exist_ok=True)
    canvas.save(target)
    return target


def preference_figure(comparison, target, ylabel, show_behaviour=True):
    """Preference of each metric vs chance-corrected reliance, one panel per setting
    (natural images first), with the increasing isotonic fit used for the switch
    points. Data: preference.csv (per model); behavioural switch: behaviour_summary.csv."""
    FIG_OUT.mkdir(exist_ok=True)
    if SUBMISSION:
        # journal version: 3 x 3 grid at about the printed size (text stays readable at 6.2 in), colour-blind safe
        # palette (Okabe-Ito) with a line style per metric as a second cue; the data and panels are unchanged
        FIG_SRC.mkdir(parents=True, exist_ok=True)
        target = FIG_SRC / Path(target).name
        fig, axes = plt.subplots(3, 3, figsize=(7.4, 7.0), sharey=True)
        colours = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
        styles, small = ["-", "--", "-.", ":"], 7.5
    else:
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharey=True)
        colours = ["C0", "C1", "C2", "C3"]
        styles, small = ["-"] * 4, 8
    for ax, (base, shortcut, label, _) in zip(axes.flat, SETTINGS):
        sel = PREF[(PREF.comparison == comparison) & (PREF.base == base) & (PREF.shortcut == shortcut)]
        assert len(sel) == N_SETTING[(base, shortcut)], (comparison, base, shortcut)
        for metric, colour, style in zip(METRICS, colours, styles):
            x, y = sel.reliance_corrected.to_numpy(), sel[METRIC_COLUMNS[metric]].to_numpy()
            ax.scatter(x, y, s=9, alpha=0.35, color=colour, linewidths=0)
            ux, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
            mean = np.bincount(inverse, weights=y) / counts
            ax.plot(ux, pava(mean, counts), color=colour, lw=1.6, ls=style, label=metric)
        ax.axhline(0.5, color="grey", lw=0.7)
        ax.axvline(50, color="grey", lw=0.7, ls=":")
        if show_behaviour:
            sw = behavioural_switch(base, shortcut)
            if not pd.isna(sw):
                ax.axvline(sw, color="black", lw=0.9, ls="--")
        ax.set_title(label, fontsize=small + 1)
        ax.set_xlim(-5, 103)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=small)
    xlabel_axes = axes[1] if not SUBMISSION else [axes[1][1], axes[1][2], axes[2][0]]
    for ax in xlabel_axes:
        ax.set_xlabel(f"Chance-corrected reliance {RC} (%)", fontsize=small + 0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=small + 0.5)
    legend_ax = axes.flat[len(SETTINGS)] if SUBMISSION else axes.flat[-1]
    if SUBMISSION:
        for ax in axes.flat[len(SETTINGS) + 1:]:
            ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="grey", ls=":"), plt.Line2D([], [], color="black", ls="--")]
    labels += ["assumed ideal switch (50%)", "behavioural switch (follow rate 0.5)"]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", fontsize=small + 0.5, frameon=False)
    fig.tight_layout()
    if SUBMISSION:
        fig.savefig(Path(target).with_suffix(".pdf"))  # vector, fonts embedded as TrueType
        fig.savefig(target, dpi=400)  # 7.4 in x 400 dpi = 2960 px, above the 2244 px full-page minimum
    else:
        fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


# ------------------------------------------------------------------ document helpers

# The analyze figures list the settings alphabetically; the submission puts them in the order of Table 2
# (natural images first, SD1 and SD4 last).
_ALPHA = sorted(set(zip(MODELS_ALL.base, MODELS_ALL.shortcut)))  # panels of the analyze figures (SD4 included)
N_PANELS = len(_ALPHA)
PANEL_ORDER = ([_ALPHA.index((b, s)) for b, s, *_ in SETTINGS] + [_ALPHA.index(SD1_KEY), _ALPHA.index(SD4_KEY)]
               if SUBMISSION else [i for i, k in enumerate(_ALPHA) if k != SD4_KEY])  # the draft has no SD4

doc = Document()
SUPP = Document() if SUBMISSION else None
COUNTERS = {"fig": 0, "table": 0, "stable": 0, "afig": 0, "atable": 0}  # afig, atable: Appendix A (submission)
APPENDIX = {"on": False}  # while on, the submission numbers tables and figures A.1, A.2, ...
FIGURE_FILES: list[dict] = []  # submission: one record per exported Figure_N file (checked and written to a manifest)


def setup(doc=None, header_text=None):
    """Single-column Elsevier-style manuscript: A4, 2.5 cm margins, Times New Roman
    12 pt body text with 1.5 line spacing (tables and captions single-spaced)."""
    doc = globals()["doc"] if doc is None else doc
    section = doc.sections[0]
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin"):
        setattr(section, side, Cm(2.5))
    section.top_margin = section.bottom_margin = Cm(2.5)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.5
    for level, size in ((1, 13), (2, 12)):
        style = doc.styles[f"Heading {level}"]
        style.font.name = "Times New Roman"
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.italic = level == 2
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        style.paragraph_format.space_after = Pt(4)
    header_text = DRAFT_NOTICE if header_text is None and not SUBMISSION else header_text
    if header_text:
        header = section.header.paragraphs[0]
        header.text = header_text
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        header.runs[0].font.size = Pt(8)
        header.runs[0].font.color.rgb = RGBColor(0xC0, 0, 0)


def add_runs(p, text, bold=False, italic=False, size=None):
    """Text with **bold** and _italic_ spans."""
    import re
    if SUBMISSION:  # repository paths are not journal text; the file names stay (data availability)
        text = text.replace("results/dose_response/", "").replace("results/analysis/", "")
    # Italic spans must not touch a word character or a slash, so underscores in
    # file names (results/dose_response/...) stay literal.
    for index, part in enumerate(re.split(r"(\*\*[^*]+\*\*|(?<![\w/])_[^_/]+_(?!\w))", text)):
        if not part:
            continue
        run_bold, run_italic = bold, italic
        if index % 2 == 1 and part.startswith("**"):
            part, run_bold = part[2:-2], True
        elif index % 2 == 1:
            part, run_italic = part[1:-1], True
        run = p.add_run(part)
        run.bold, run.italic = run_bold, run_italic
        if size:
            run.font.size = Pt(size)


def para(text, bold=False, italic=False, align=None, size=None, after=None, spacing=None):
    p = doc.add_paragraph()
    add_runs(p, text, bold, italic, size)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align is None else align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if spacing is not None:
        p.paragraph_format.line_spacing = spacing
    return p


def bullets(items, numbered=False):
    for item in items:
        p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
        add_runs(p, item)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def heading(text, level=1):
    doc.add_heading(text, level=level)


def export_figure(path, label):
    """Submission only: copy a figure to paper/figures_submission/Figure_<label>.png and make the matching
    .pdf: the vector file when the figure was regenerated here, otherwise the bitmap wrapped in a PDF."""
    FIG_SUB.mkdir(exist_ok=True)
    path = Path(path)
    png, pdf = FIG_SUB / f"Figure_{label}.png", FIG_SUB / f"Figure_{label}.pdf"
    image = Image.open(path)
    vector = path.with_suffix(".pdf").exists()
    if vector:
        shutil.copyfile(path, png)
        shutil.copyfile(path.with_suffix(".pdf"), pdf)
    else:
        image.convert("RGB").save(png, dpi=(300, 300))
        image.convert("RGB").save(pdf, resolution=300)
    FIGURE_FILES.append({"label": label, "png": png, "pdf": pdf, "vector": vector, "width_px": image.width,
                         "height_px": image.height, "source": path.name})


def figure(path, caption, width=Inches(6.2)):
    appendix = SUBMISSION and APPENDIX["on"]
    COUNTERS["afig" if appendix else "fig"] += 1
    label = f"A.{COUNTERS['afig']}" if appendix else str(COUNTERS["fig"])
    if SUBMISSION:
        export_figure(path, label.replace(".", ""))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(str(path), width=width)
    p.paragraph_format.keep_with_next = True
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.line_spacing = 1.0
    r = c.add_run(f"Fig. {label}. ")
    r.bold = True
    r.font.size = Pt(9)
    add_runs(c, caption, size=9)
    return label if appendix else COUNTERS["fig"]


def shade(cell, color="D9D9D9"):
    props = cell._tc.get_or_add_tcPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), color)
    props.append(fill)


def horizontal_rules(t, n_header_rows):
    """Elsevier tables: horizontal rules only (top, under the header, bottom), no vertical rules, no shading."""
    props = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        e = OxmlElement(f"w:{edge}")
        for k, v in (("val", "single"), ("sz", "8"), ("space", "0"), ("color", "000000")):
            e.set(qn(f"w:{k}"), v)
        borders.append(e)
    props.append(borders)
    for cell in t.rows[n_header_rows - 1].cells:
        tc_borders = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        for k, v in (("val", "single"), ("sz", "4"), ("space", "0"), ("color", "000000")):
            e.set(qn(f"w:{k}"), v)
        tc_borders.append(e)
        cell._tc.get_or_add_tcPr().append(tc_borders)


def table(caption, header, body, widths=None, note=None, size=8, header_rows=None, supp=False):
    """A numbered table: caption above (Elsevier style), optional note below. In the submission build a
    table with supp=True goes to the supplementary document as Table S<n>."""
    doc = SUPP if (SUBMISSION and supp) else globals()["doc"]
    key = "stable" if (SUBMISSION and supp) else ("atable" if SUBMISSION and APPENDIX["on"] else "table")
    COUNTERS[key] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.line_spacing = 1.0
    r = c.add_run(f"Table {'S' if key == 'stable' else 'A.' if key == 'atable' else ''}{COUNTERS[key]}. ")
    r.bold = True
    r.font.size = Pt(9)
    add_runs(c, caption, size=9)
    headers = header_rows or [header]
    t = doc.add_table(rows=len(headers) + len(body), cols=len(headers[-1]))
    if not SUBMISSION:
        t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, values in enumerate(headers + body):
        for j, value in enumerate(values):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(value))
            run.font.size = Pt(size)
            if i < len(headers):
                run.bold = True
                if not SUBMISSION:
                    shade(cell)
    if SUBMISSION:
        horizontal_rules(t, len(headers))
    if widths:
        for i, w in enumerate(widths):
            for cell in t.columns[i].cells:
                cell.width = Cm(w)
    if note:
        n = doc.add_paragraph()
        n.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        n.paragraph_format.line_spacing = 1.0
        add_runs(n, note, size=8)
    else:
        doc.add_paragraph()
    return COUNTERS[key]


# ------------------------------------------------------------------ table builders


def table_positioning():
    body = [
        ["Ground-truth benchmarks: inject a shortcut or build data so the evidence is known",
         "Decoy MNIST " + cite("ross") + "; Zhou et al. " + cite("zhou") + "; Lin et al. " + cite("lin")
         + "; BackX " + cite("backx") + "; Adebayo et al. " + cite("adebayo2022") + "; Sun et al. " + cite("sun")
         + "; Bastings et al. (text) " + cite("bastings") + "; synthetic intervention-based ground truth for "
         "XAI methods " + cite("miro2026"),
         "The object of evaluation is the metric, not the attribution method"],
        ["Graded or controlled model reliance",
         "BAM: models trained with known relative feature importance " + cite("bam") + "; SMERF: controlled "
         "model reasoning " + cite("smerf") + "; OSCAR: shortcut scores that react to the strength of the "
         "shortcut association " + cite("oscar"),
         "A continuous signed dose, with reliance measured behaviourally from about 0% to 100% and chance-corrected. "
         "BAM and Adebayo et al. " + cite("adebayo2022") + " report false positives of attribution methods; "
         "we show over-detection by the metrics, with oracle maps and no attribution method"],
        ["Meta-evaluation of faithfulness metrics",
         "Tomsett et al. " + cite("tomsett") + "; MetaQuantus (no ground truth) " + cite("metaquantus")
         + "; fidelity metrics vs a decision tree " + cite("fidelity") + "; CAM evaluation metrics vs synthetic "
         "ground truth " + cite("refinecam") + "; Gevaert et al. " + cite("gevaert") + "; LATEC " + cite("latec")
         + "; M4 " + cite("m4"),
         "A ground truth that changes with the model, read out as responsiveness, over-detection, a switch "
         "point and per-image AUROC"],
        ["Shape, size and fill confounds",
         "ROAD " + cite("rong") + "; ROAR pitfalls " + cite("roarpitfalls") + "; Bl\u00fccher et al. "
         + cite("blucher") + "; Back to the Baseline " + cite("baseline") + "; Gomez et al. " + cite("gomez"),
         "Size-matched, placebo and random oracle maps that show the confounds as reliance varies; real masks on Pet"],
        ["Pre-registration",
         "In our searches we did not find an XAI-evaluation benchmark that pre-registers its predictions",
         S("Pre-registered Pet predictions and a dated amendment, with commit hashes",
           "Pre-registered predictions with dated amendments, time-stamped on Zenodo")],
    ]
    return table(
        S("Positioning. What prior work already provides and what this study adds (after the audit's "
          "positioning table, docs/audit/02_literature_novelty.md \u00a72)",
          "Positioning. What prior work already provides and what this study adds"),
        ["Ingredient", "Prior work that already has it", "What this study adds"],
        body, widths=[3.2, 7.0, 5.8], size=8,
        note="The protocol is new as a combination; its individual pieces are not. BAM and SMERF already "
             "controlled what a model relies on, to evaluate attribution methods."
             + (S("", " Exact ROAD was also scored with the same oracle maps (Section 5.11): it "
                  "ranks the object map above random pixels more often than black deletion on natural images, but "
                  f"still over-detects the shortcut for {int(A3N['p11_main'].k_road)} of "
                  f"{int(A3N['p11_main'].n_below)} low-reliance Pet models.") if SUBMISSION else ""))


def table_reliance():
    body = []
    for base, shortcut, label, _ in SETTINGS + [SD1] + [SD4] * SUBMISSION:
        g, lv = group(base, shortcut), levels(base, shortcut)
        lowest, highest = lv.loc[lv.reliance.idxmin()], lv.loc[lv.reliance.idxmax()]
        a1, a2, a3 = (int((g.model_set == k).sum()) for k in ("A1", "A2", "A3"))
        count = (f"{len(g)} ({lv.dose.nunique()} doses)" + (f", incl. {a1} A1" if a1 else "")
                 + (f", incl. {a2} A2" if a2 else "") + (f", incl. {a3} A3" if a3 else ""))
        body.append([
            label, NETS[base], count,
            f"{f2(lv.dose.min())} to {f2(lv.dose.max())}",
            f"{f1(g.reliance.min())}{DASH}{f1(g.reliance.max())}",
            f"{f1(g.reliance_corrected.min())}{DASH}{f1(g.reliance_corrected.max())}",
            f"{f1(lowest.test_accuracy_no_shortcut)} \u2192 {f1(highest.test_accuracy_no_shortcut)}",
        ])
    return table(
        "Reliance reached per setting. Ranges are over single models. Accuracy without the shortcut is the "
        "per-dose mean at the dose with the lowest and the highest mean reliance",
        ["Setting", "Network", "Models", "Dose range", "Raw reliance R (%)", f"{RC} (%)",
         "Accuracy without shortcut (%), low \u2192 high reliance"],
        body, widths=[3.4, 2.2, 2.4, 1.8, 1.8, 1.8, 2.6],
        note=f"Source: results/dose_response/models.csv ({N_MODELS} models: {N_ORIGINAL} original, {N_A1} "
             + S(f"from amendment A1 and {N_A2} from amendment A2, of which {N_SD1} form the separate SD1 setting in "
                 f"the last row; the A2 pilots are not included, see ",
                 f"from amendment A1, {N_A2} from amendment A2, of which {N_SD1} form the separate SD1 setting, and "
                 f"{N_A3} from amendment A3, the separate SD4 setting in the last row; the pilots are not included, see ")
             + f"results/dose_response/pilots.csv). R: percentage of conflicting test images whose prediction follows the "
             f"shortcut. {RC} = 100 (R \u2212 f)/(100 \u2212 f) with chance floor f = (100 \u2212 accuracy "
             "without shortcut)/(C \u2212 1); it is not clipped and can be slightly negative.")


def table_main(comparison, caption, supp=False):
    body = []
    for base, shortcut, label, short in SETTINGS:
        for i, metric in enumerate(METRICS):
            r = row(comparison, base, shortcut, metric).copy()
            x = xrow(comparison, base, shortcut, metric)
            assert np.isclose(x.rho_rc, r.rho_rc) and np.isclose(x.pref_low, r.pref_low)
            assert (pd.isna(x.switch_iso_rc) and pd.isna(r.switch_iso_rc)) or np.isclose(x.switch_iso_rc, r.switch_iso_rc)
            r["switch_iso_rc_ci_low"], r["switch_iso_rc_ci_high"] = x.switch_crossed_ci_low, x.switch_crossed_ci_high
            body.append([
                short if i == 0 else "", metric,
                f"{f2(r.rho_rc)}{sig(r)} {ci(x.rho_rc_bca_ci_low, x.rho_rc_bca_ci_high)}",
                f"{f2(r.pref_low)} {ci(x.pref_low_crossed_ci_low, x.pref_low_crossed_ci_high)} \u2192 {f2(r.pref_high)}",
                f2(r.median_gap_low),
                f"{f2(r.fpr)} ({int(r.n_below)})",
                f2(r.balanced_accuracy),
                fmt_switch(r),
            ])
    return table(
        caption,
        ["Setting", "Metric", f"\u03c1 with {RC} [BCa 95% CI]", "Preference, lowest dose [crossed CI] \u2192 highest",
         "Median AUC gap, lowest dose", "Over-detection FPR (n models)", "Balanced accuracy",
         f"Switch, {RC} % [crossed CI]"],
        body, widths=[2.0, 2.2, 2.4, 2.9, 1.5, 1.6, 1.4, 2.6], size=7.5, supp=supp,
        note=f"Sources: results/dose_response/responsiveness_corrected.csv (estimates, p_bh, FPR, balanced "
             f"accuracy) and intervals_crossed.csv (intervals), comparison \u201c{comparison}\u201d. Crossed CI: "
             "2000 bootstrap replicates that resample models within dose and draw one set of images per replicate "
             "shared by all models. BCa CI for \u03c1: bias-corrected and accelerated (leave-one-model-out "
             "jackknife), on crossed replicates with tiny jitter to break ties between duplicated models. "
             "Significance comes from the model-level Spearman test, not from the intervals. "
             "\u2020: not significant after Benjamini\u2013Hochberg within the comparison family (p_bh \u2265 0.05). "
             f"Over-detection FPR: share of models with {RC} \u2264 50 for which the preference exceeds 0.5 "
             "(n = number of such models). Median AUC gap: positive means the shortcut map is rated better. "
             "Switch: first crossing of 0.5 by the increasing isotonic fit; (e) identified at the lowest-reliance "
             "edge (within 5 points of the lowest model); (nb) not bracketed by models within \u00b110 points "
             "on both sides; \u201cnone (> 0.5 at lowest)\u201d: the metric already prefers the shortcut map at "
             "the lowest reliance reached.")


def prereg_headline(r):
    """Rows of the abridged submission table: every P1 and P4 row, the P2 contrast and the Deletion (black)
    rows of P3 (the only rows that fail)."""
    return (r.prediction in ("P1", "P4") or r.statistic == "preference contrast at lowest dose"
            or (r.prediction == "P3" and r.metric_or_method == "Deletion (black)"))


def table_prereg(full=True):
    body = []
    for r in PET.itertuples():
        if not full and not prereg_headline(r):
            continue
        body.append([r.prediction, r.comparison, r.metric_or_method, r.statistic,
                     f"{f3(r.estimate)} {ci(r.ci_low, r.ci_high, f3)}",
                     "\u2013" if pd.isna(r.rho_rc) else f3(r.rho_rc), r.threshold, r.original_verdict])
    if not full:
        assert 10 <= len(body) <= 12, len(body)
    return table(
        "Pre-registered Pet predictions P1\u2013P4 with their original verdicts. The intervals and the "
        f"\u03c1 on {RC} were computed later and are additional information"
        + ("" if full else " (abridged; the full table is Supplementary Table S1)"),
        ["Pred.", "Comparison", "Metric or method", "Statistic", "Estimate [two-level 95% CI]",
         f"\u03c1 on {RC}", "Threshold", "Original verdict"],
        body, widths=[1.0, 2.4, 2.4, 2.4, 2.6, 1.2, 1.8, 2.6], size=7, supp=full and SUBMISSION,
        note=S("Source: results/dose_response/pre_a2_models/pet_preregistered.csv (the 18 original Pet models; "
               "amendment A2 commits to judging P1\u2013P4 on these only); the verdict strings are copied from the "
               "Outcome table of docs/PREDICTIONS_pet.md. ",
               "Source: pet_preregistered.csv (the 18 original Pet models; a later amendment commits to judging "
               "P1\u2013P4 on these only); the verdict strings are those recorded in the pre-registration. ")
             + ("" if full else "Rows shown: every P1 and P4 row, the P2 contrast and the Deletion (black) rows of "
                "P3, which are the rows that decide the one failed prediction; the other rows are in "
                "Supplementary Table S1. ")
             + "Verdicts were judged on point estimates, as "
             "pre-registered; the pre-registration did not say whether a point estimate or an interval decides.")


def table_behaviour():
    body = []
    for base, shortcut, label, short in SETTINGS:
        for i, metric in enumerate(METRICS):
            b = brow(FULL, base, shortcut, metric)
            cal = calrow(FULL, base, shortcut, metric)
            assert np.isclose(cal.calibration_gap, b.calibration_gap)
            body.append([
                short if i == 0 else "", metric,
                f"{f2(b.auroc_pooled_strat)} {ci(b.auroc_strat_ci_low, b.auroc_strat_ci_high)}",
                f"{f2(b.median_model_auroc_strat)} {ci(b.median_model_auroc_strat_ci_low, b.median_model_auroc_strat_ci_high)}",
                f"{f2(b.calibration_gap)} / {f2(cal.calibration_gap_determined)}",
                fmt_switch(b, "switch_on_follow_rate", "switch_on_follow_rate_status", None, None,
                           "switch_on_follow_rate_bracketed"),
                switch_text(swrow(base, shortcut)) if i == 0 else "",
            ])
    return table(
        "Per-image behavioural test (shortcut vs full object, primary label). Each scored image is labelled by "
        "forward passes as followed or not followed; the score is the per-image AUC gap",
        ["Setting", "Metric", "Pooled AUROC [CI]", "Within-model median AUROC [CI]",
         "Calibration gap, all / determined images", "Metric switch on follow rate (%)",
         f"Behavioural switch, {RC} (status)"],
        body, widths=[2.0, 2.2, 2.4, 2.6, 1.8, 2.3, 2.8], size=7.5,
        note=f"Sources: results/dose_response/behaviour_summary.csv, behaviour_calibration.csv and "
             f"behaviour_switch.csv, label {PRIMARY_LABEL}. AUROCs are "
             "class-stratified (computed within each true class and averaged) and tested one-sided against "
             "0.5. Pooled: the images of all models together, which mixes between- and within-model signal. "
             "Within-model: AUROC inside each model, median over models. Calibration gap: mean over models of "
             "(preference \u2212 follow rate), with the preference over all 100 images or over the images whose "
             "label is determined (the follow rate always uses the determined images); positive values mean "
             "over-detection. Metric switch on "
             "follow rate: where the isotonic fit of preference against 100 \u00d7 follow rate crosses 0.5 "
             "(ideal 50 by construction). Behavioural switch: the "
             f"{RC} at which the models' follow rate crosses 0.5, with the identification rule of the metric "
             "switches: \u201cnot identified\u201d when no model lies within \u00b110 points (the crossing is then "
             "only known to lie above the last model below the gap), \u201cnot bracketed\u201d when models within "
             "\u00b110 points do not lie on both sides.")


def table_labels():
    body = []
    for base, shortcut, label, short in SETTINGS:
        for i, lab in enumerate(BEHAVIOUR_LABELS):
            s = swrow(base, shortcut, lab)
            b = BEH[(BEH.comparison == FULL) & (BEH.base == base) & (BEH.shortcut == shortcut) & (BEH.label == lab)]
            assert len(b) == len(METRICS)
            und = ("none" if s.undetermined_rate_max == 0 else
                   f"{pct(s.undetermined_rate_mean)} (max {pct(s.undetermined_rate_max)}); \u03c1 = "
                   f"{f2(s.rho_undetermined_rc)}")
            body.append([short if i == 0 else "", LABEL_NAMES[lab], switch_text(s), und,
                         span(b.auroc_pooled_strat), span(b.median_model_auroc_strat)])
    return table(
        "Behavioural results under the three labels (shortcut vs full object): behavioural switch with its "
        "status, undetermined share and its Spearman \u03c1 with r\u1d04 over models, and the range over the four "
        "metrics of the pooled and within-model class-stratified AUROC",
        ["Setting", "Label", f"Behavioural switch, {RC} (status)", f"Undetermined share; \u03c1 with {RC}",
         "Pooled AUROC (4 metrics)", "Within-model median AUROC (4 metrics)"],
        body, widths=[2.0, 2.2, 4.4, 3.0, 2.2, 2.2], size=7,
        note="Sources: results/dose_response/behaviour_switch.csv and behaviour_summary.csv. What each label "
             "measures: " + "; ".join(f"{LABEL_NAMES[lab]}: {BSWITCH[BSWITCH.label == lab].measures.iloc[0]}"
                                      for lab in BEHAVIOUR_LABELS)
             + ". Only the primary label leaves images undetermined (those the model already assigns to the "
               "shortcut's class without any shortcut).")


def table_overdetection():
    body = []
    for base, shortcut, label, short in NATURAL:
        for i, metric in enumerate(METRICS):
            o = orow(base, shortcut, metric)
            bands = margin_bands(base, shortcut, metric)
            assert bands["rc_le10"][1] == o.n_rc_le10 and bands["rc_10_50"][1] == o.n_rc_10_50, (base, metric)
            body.append([
                short if i == 0 else "", metric, f2(o.pref_so_low), f2(o.pref_ro_low), f2(o.pref_sr_low),
                f"{f2(o.excess_low)} {ci(o.excess_low_ci_low, o.excess_low_ci_high)}",
                f"{f2(o.rho_excess_rc)} ({format_p(o.p_bh_excess_one_sided)})",
                f"{f2(o.fpr)} {ci(o.fpr_cp_low, o.fpr_cp_high)} ({int(o.k_fpr)}/{int(o.n_below)})",
                f"{f2(o.fpr_margin)} {ci(o.fpr_margin_cp_low, o.fpr_margin_cp_high)} "
                f"({int(o.k_fpr_margin)}/{int(o.n_below)})",
                f"{f2(o.fpr_rc_le10)} ({int(o.n_rc_le10)}) / {f2(o.fpr_rc_10_50)} ({int(o.n_rc_10_50)})",
                " / ".join(f"{f2(k / n)} {ci(*clopper_pearson(k, n))}" if n else "n/a"
                           for k, n in (bands["rc_le10"], bands["rc_10_50"])),
                f"{f2(o.tpr)} ({int(o.n_above)})",
            ])
    return table(
        "Natural images: decomposition of the over-detection (shortcut vs full object). Lowest-dose "
        "preferences for S\u2013O, R\u2013O and S\u2013R, the excess preference S\u2013O \u2212 R\u2013O, and the "
        "decision rates with intervals, a margin and a split by reliance",
        ["Setting", "Metric", "S\u2013O", "R\u2013O", "S\u2013R", "Excess, lowest dose [CI]",
         f"\u03c1(excess, {RC}) (p_bh)", "FPR [CP 95% CI] (k/n)", "Margin FPR [CP 95% CI] (k/n)",
         f"FPR at {RC} \u2264 10 (n) / 10\u201350 (n)", f"Margin FPR [CP] at {RC} \u2264 10 / 10\u201350",
         "TPR (n above 50)"],
        body, widths=[1.3, 1.6, 0.8, 0.8, 0.8, 1.8, 1.4, 2.0, 2.0, 1.6, 2.4, 1.1], size=6,
        note="Source: results/dose_response/overdetection.csv (per model: excess_preference_models.csv). S = "
             "shortcut map, O = full object map (Pet: real mask; CIFAR-10: centred Gaussian prior), R = random "
             "ordering. Excess: per-model S\u2013O minus R\u2013O, i.e. the shortcut preference beyond what a random "
             "ordering already gets against the same object map; two-level CI. \u03c1(excess, "
             f"{RC}): Spearman over models, one-sided (\u03c1 > 0), BH over all settings and metrics. FPR: share of "
             f"models with {RC} \u2264 50 whose preference exceeds 0.5, with a Clopper\u2013Pearson interval. Margin "
             "FPR: a model counts only if the lower end of its own 95% image-bootstrap interval of preference "
             "exceeds 0.5, with a Clopper\u2013Pearson interval; per band it is computed from the per-model "
             f"decisions in excess_preference_models.csv ({RC} \u2264 10 and 10 < {RC} \u2264 50). TPR: share of models with " + RC + " > 50 whose preference exceeds 0.5.")


def table_controls():
    comps = [(FULL, "S\u2013O"), (R_OBJECT, "R\u2013O"), (P_OBJECT, "P\u2013O"),
             (S_RANDOM, "S\u2013R"), (P_RANDOM, "P\u2013R")]
    body = []
    for base, shortcut, label, short in NATURAL:
        for i, metric in enumerate(METRICS):
            values = [f"{short} ({f1(group(base, shortcut).reliance_corrected.min())}%)" if i == 0 else "", metric]
            for c, _ in comps:
                r = row(c, base, shortcut, metric)
                values.append(f"{f2(r.pref_low)} {ci(r.pref_low_ci_low, r.pref_low_ci_high)}" if c == R_OBJECT
                              else f2(r.pref_low))
            body.append(values)
    return table(
        "Natural images: preference at the lowest dose for the comparisons of the main maps and the controls. "
        "Each entry is the share of images on which the metric rates the first map above the second",
        ["Setting (lowest model r\u1d04)", "Metric"] + [c for _, c in comps],
        body, widths=[3.0, 2.4, 1.6, 2.6, 1.6, 1.6, 1.6], size=7.5,
        note="S = shortcut map, O = full object map (Pet: the real segmentation mask; CIFAR-10: a centred "
             "Gaussian prior), R = random ordering, P = placebo (the shortcut's shape at a wrong class's "
             "position). At this reliance the models barely use the shortcut, so the ideal values are S\u2013O, "
             "R\u2013O and P\u2013O low, S\u2013R and P\u2013R \u2248 0.5. Source: "
             "results/dose_response/responsiveness_corrected.csv, column pref_low (R\u2013O with its two-level CI).")


def table_pairs():
    pairs = main_only(PAIRS)
    sel = pairs[pairs.comparison.isin([FULL, MATCHED, R_OBJECT]) & (pairs.p_bh < 0.05)]
    order = {(b, s): i for i, (b, s, *_) in enumerate(SETTINGS)}
    sel = sel.assign(o=[order[(b, s)] for b, s in zip(sel.base, sel.shortcut)]).sort_values(["comparison", "o"])
    body = [[r.comparison, SHORT[(r.base, r.shortcut)], r.metric_a, r.metric_b,
             f"{f3(r.delta_rho_rc)} {ci(r.delta_rho_rc_ci_low, r.delta_rho_rc_ci_high, f3)}",
             format_p(r.p_bh), format_p(r.p_holm),
             f"{f3(r.delta_pref_low)} {ci(r.delta_pref_low_ci_low, r.delta_pref_low_ci_high, f3)}"]
            for r in sel.itertuples()]
    n_tested = len(pairs[pairs.comparison.isin([FULL, MATCHED, R_OBJECT])])
    return table(
        f"Paired metric-vs-metric differences that are significant after Benjamini\u2013Hochberg ({len(sel)} of "
        f"{n_tested} pairs tested)",
        ["Comparison", "Setting", "Metric A", "Metric B", f"\u0394\u03c1 on {RC} (A \u2212 B) [CI]", "p_bh", "p_holm",
         "\u0394 preference, lowest dose [CI]"],
        body, widths=[2.6, 1.9, 2.0, 2.0, 2.8, 1.1, 1.1, 2.8], size=7, supp=True,
        note="Source: results/dose_response/metric_pairs.csv. Paired two-level bootstrap (the same resamples "
             "for both metrics); two-sided bootstrap p, adjusted within each comparison family (BH and Holm; "
             "the families also contain the SD1 pairs, which are not listed here).")


def table_methods():
    body = []
    for method in METHODS:
        values = [method]
        for base, shortcut, *_ in SETTINGS:
            r = mrow(base, shortcut, method)
            values.append(f"{f2(r.rho_rc)}{sig(r)}\n{ci(r.rho_rc_ci_low, r.rho_rc_ci_high)}\n"
                          f"{f3(r.mass_low)} \u2192 {f3(r.mass_high)}")
        body.append(values)
    return table(
        f"Explanation methods: Spearman \u03c1 between {RC} and the share of positive attribution on the shortcut "
        "pixels [two-level 95% CI], and that share at the lowest \u2192 highest dose",
        ["Method"] + [short for *_, short in SETTINGS], body, widths=[2.4] + [1.95] * 7, size=6.5,
        note="Source: results/dose_response/method_responsiveness_corrected.csv. \u2020: not significant after "
             "BH over the methods family. The shortcut covers 144 of 16,384 pixels on Pet, 9 of 1,024 on "
             "CIFAR-10, 9 of 784 on the 28\u00d728 patch settings and about 10% of the pixels for the texture.")


def table_confound():
    body = []
    for (base, shortcut), g in main_only(CONF_RC).groupby(["base", "shortcut"], sort=False):
        first = g.iloc[0]
        values = [SHORT[(base, shortcut)], str(int(first.models)), f2(first.rho_rc_competence)]
        for metric in METRICS:
            r = ccrow(base, shortcut, metric)
            assert np.isclose(r.partial_rel_given_acc, crow(base, shortcut, metric).partial_rel_given_acc)
            values.append(f"{f2(r.partial_rc_given_acc)} (p_bh {format_p(r.p_bh_rc)}) "
                          f"[raw R: {f2(r.partial_rel_given_acc)}]")
        body.append(values)
    return table(
        f"Reliance and object competence. \u03c1({RC}, acc): Spearman correlation between chance-corrected "
        "reliance and accuracy without the shortcut. Per metric (shortcut vs full object): partial \u03c1 of "
        f"preference with {RC} controlling for accuracy without the shortcut (BH-adjusted p), with the partial "
        "\u03c1 on raw reliance R in brackets",
        ["Setting", "Models", f"\u03c1({RC}, acc)"] + METRICS,
        body, widths=[2.2, 1.2, 1.4, 2.8, 2.8, 2.8, 2.8], size=7.5,
        note="Source: results/dose_response/confound_rc.csv (raw-reliance values as in confound.csv), in the row "
             "order of confound.csv (from the least collinear setting on raw reliance). Partial "
             "Spearman: Pearson correlation of rank residuals after a linear fit on the accuracy rank; p from "
             "a t distribution with n \u2212 3 degrees of freedom, adjusted by Benjamini\u2013Hochberg over the "
             f"{len(CONF_RC)} tests of confound_rc.csv (the {len(main_only(CONF_RC))} of the table and the "
             + S(f"{len(CONF_RC) - len(main_only(CONF_RC))} SD1 tests of Section 5.10).",
                 f"{int(((CONF_RC.base == SD1_KEY[0]) & (CONF_RC.shortcut == SD1_KEY[1])).sum())} SD1 and "
                 f"{int(((CONF_RC.base == SD4_KEY[0]) & (CONF_RC.shortcut == SD4_KEY[1])).sum())} SD4 tests of "
                 "Sections 5.10 and 5.12)."))


def table_kendall():
    body = []
    for r in KENDALL.itertuples():
        body.append([{"decoy_mnist": "Decoy MNIST", "decoy_pneumonia": "Decoy Pneumonia"}[r.dataset], r.metric,
                     f2(r.mean_tau), format_p(r.p_mean_one_sided),
                     ", ".join(f2(getattr(r, f"tau_seed{s}")) for s in (42, 1, 2))])
    threshold = KENDALL["min_tau_one_sided_p_below_0.05"].iloc[0]
    return table(
        "Supporting study: Kendall \u03c4 between each metric's ranking of the five methods and the "
        "ground-truth ranking, with exact p-values",
        ["Dataset", "Metric", "Mean \u03c4 (3 seeds)", "Exact one-sided p of the mean", "\u03c4 per seed (42, 1, 2)"],
        body, widths=[2.6, 3.2, 2.2, 3.0, 3.6], size=8,
        note=f"Source: results/analysis/kendall_exact.csv. With five methods \u03c4 moves in steps of 0.2, and a "
             f"single seed reaches one-sided p < 0.05 only at \u03c4 \u2265 {f1(threshold)} (exact null over all "
             "120 orderings, results/analysis/kendall_null_n5.csv). The p of the mean uses the exact null of the "
             "mean of three independent \u03c4.")


def table_legacy():
    body = []
    for base, shortcut, label, short in SETTINGS:
        for i, metric in enumerate(METRICS):
            r = legacy(FULL, base, shortcut, metric)
            point = "none" if pd.isna(r.switch_point_reliance) else f1(r.switch_point_reliance)
            body.append([short if i == 0 else "", metric,
                         f"{f2(r.responsiveness_rho)} {ci(r.rho_ci_low, r.rho_ci_high)}",
                         p_text(r.p_value).replace("p ", ""), f2(r.decision_accuracy),
                         f"{point} {ci(r.switch_ci_low, r.switch_ci_high, f1)}"])
    return table(
        "Legacy analysis (shortcut vs full object): raw reliance, seed-only bootstrap, unbalanced decision "
        "accuracy and the first-crossing switch rule, kept for comparison with "
        + S("earlier drafts", "the primary analysis"),
        ["Setting", "Metric", "\u03c1 with raw R [seed-only CI]", "p (unadjusted)", "Decision accuracy (unbalanced)",
         "Switch, raw R % [CI]"],
        body, widths=[2.2, 2.4, 3.2, 2.0, 2.4, 3.2], size=7.5, supp=True,
        note="Source: results/dose_response/responsiveness.csv (legacy). These intervals resample only the 2\u20136 "
             "seeds per dose with the images fixed, so they are too narrow; the decision accuracy is "
             "unbalanced; the switch is the first crossing of the per-dose means. Superseded by Tables in "
             "Section 5.")


def table_sets():
    body = []
    for base, shortcut, label, short in MECHANISM:
        for i, metric in enumerate(METRICS):
            a, o = row(FULL, base, shortcut, metric), row(FULL, base, shortcut, metric, RESP_ORIG)
            p = row(FULL, base, shortcut, metric, RESP_PRE_A2)
            assert all(np.isclose(a[c], p[c]) for c in ("rho_rc", "fpr", "balanced_accuracy")) and (
                (pd.isna(a.switch_iso_rc) and pd.isna(p.switch_iso_rc)) or np.isclose(a.switch_iso_rc, p.switch_iso_rc)), \
                "note: mechanism rows of all models equal those with the A1 models"
            body.append([short if i == 0 else "", metric, f"{f3(o.rho_rc)} \u2192 {f3(a.rho_rc)}",
                         f"{f3(o.fpr)} \u2192 {f3(a.fpr)}", f"{f3(o.balanced_accuracy)} \u2192 {f3(a.balanced_accuracy)}",
                         f"{fmt_switch(o, lo=None)} \u2192 {fmt_switch(a, lo=None)}"])
    return table(
        f"Model-set check promised by amendment A1 (shortcut vs full object, mechanism settings): the "
        f"{N_ORIGINAL} original models \u2192 the models including the {N_A1} A1 models",
        ["Setting", "Metric", f"\u03c1 on {RC}", "Over-detection FPR", "Balanced accuracy", f"Switch, {RC} %"],
        body, widths=[2.2, 2.4, 2.6, 2.4, 2.6, 3.8], size=7.5, supp=True,
        note="Sources: results/dose_response/original_models/responsiveness_corrected.csv and "
             "results/dose_response/responsiveness_corrected.csv (analysis of all models; the mechanism "
             "settings received no A2 model, so these rows equal those with the A1 models only). Pet and "
             "CIFAR-10 received no A1 model and are omitted; their A2 models are discussed in Sections 5.4 and 5.10.")


def table_a2_outcomes(verdicts, over, full, matched, edge, bh4, p8a_n, p8b_n, p9_rows, bsw):
    """P5-P9 (docs/PREDICTIONS_pretrained.md) with the numbers that decide them."""
    def fpr(m):
        o = over[m]
        return f"{m} {f2(o.fpr)} ({int(o.k_fpr)}/{int(o.n_below)})"

    def msw(m):
        r = matched[m]
        return f"{m} {fmt_switch(r, lo=None)}" if not pd.isna(r.switch_iso_rc) else f"{m} > 0.5 at lowest"

    p9_text = "; ".join(
        f"{SHORT[k]}: behavioural {switch_text(bsw[k])}, "
        + (", ".join(f"{r.metric} {f1(r.difference)} {ci(r.diff_ci_low, r.diff_ci_high, f1)}" for r in tested)
           or "no metric switch")
        for k, (_, tested, _) in p9_rows.items())
    body = [
        ["P5", "Over-detection rate (S vs O, " + RC + " \u2264 50) of black and of blur deletion > 0.5",
         "; ".join(fpr(m) for m in BB), verdicts["P5"]],
        ["P6", "Over-detection rate of inpaint and of insertion below both black and blur",
         "; ".join(fpr(m) for m in II), verdicts["P6"]],
        ["P7", "Size-matched switch within 5 points of the lowest-" + RC + " model for \u2265 3 of 4 metrics",
         "; ".join(msw(m) for m in METRICS) + f" ({sum(edge.values())} of 4 at the edge)", verdicts["P7"]],
        ["P8", f"(a) \u03c1({RC}, S\u2013O preference) > 0.8 for \u2265 3 of 4; (b) \u03c1({RC}, excess) > 0 "
               "with one-sided BH p < 0.05 (over the four) for \u2265 3 of 4; both must hold",
         "(a) " + ", ".join(f2(full[m].rho_rc) for m in METRICS) + f" ({p8a_n} of 4); (b) "
         + ", ".join(f"{f2(over[m].rho_excess_rc)} (p_bh {format_p(bh4[m])})" for m in METRICS)
         + f" ({p8b_n} of 4)", verdicts["P8"]],
        ["P9", "SD2 settings: (a) behavioural switch identified; (b) every identified S vs O metric switch below it, "
               "paired bootstrap interval of the difference above 0", p9_text, verdicts["P9"]],
    ]
    return table(
        "Pre-registered predictions P5\u2013P9 of amendment A2" + S(" (docs/PREDICTIONS_pretrained.md)", "")
        + " with their "
        "verdicts. P5\u2013P8: the SD1 models; P9: the three SD2 settings with their A2 models",
        ["Pred.", "Test as pre-registered", "Result", "Verdict"],
        body, widths=[1.0, 5.0, 8.4, 1.6], size=7,
        note="Sources: results/dose_response/overdetection.csv (P5, P6, P8b), responsiveness_corrected.csv "
             "(P7: shortcut vs size-matched object; P8a: shortcut vs full object), behaviour_switch.csv and "
             "switch_differences.csv (P9, primary label). P8b adjusts the four one-sided p-values of "
             "overdetection.csv (p_excess_one_sided) by Benjamini\u2013Hochberg over the four metrics, as "
             "pre-registered. Verdicts are judged on point estimates, as for P1\u2013P4. (e): switch at the "
             "lowest-reliance edge. P9 differences: behavioural \u2212 metric switch in points [paired two-level "
             "95% CI]; metrics already above 0.5 at the lowest reliance have no switch and are not tested.")


def table_sd1(bh4):
    body = []
    for key in (SD1_KEY, ("pet_resnet18", "patch")):
        for i, metric in enumerate(METRICS):
            r, x, o, m = (row(FULL, *key, metric), xrow(FULL, *key, metric), orow(*key, metric),
                          row(MATCHED, *key, metric))
            body.append([
                SHORT[key] if i == 0 else "", metric,
                f"{f2(r.rho_rc)} {ci(x.rho_rc_bca_ci_low, x.rho_rc_bca_ci_high)}",
                f"{f2(r.pref_low)} \u2192 {f2(r.pref_high)}",
                f"{f2(o.fpr)} {ci(o.fpr_cp_low, o.fpr_cp_high)} ({int(o.k_fpr)}/{int(o.n_below)})",
                f"{f2(o.fpr_margin)} ({int(o.k_fpr_margin)}/{int(o.n_below)})",
                f"{f2(o.pref_so_low)} / {f2(o.pref_ro_low)} / {f2(o.pref_sr_low)}",
                f"{f2(o.rho_excess_rc)} ({format_p(bh4[metric] if key == SD1_KEY else o.p_bh_excess_one_sided)})",
                fmt_switch(r), fmt_switch(m, lo=None),
            ])
    return table(
        "SD1 (ImageNet-pretrained ResNet-18 on Pet) next to the Pet ResNet-18 from scratch (exploratory "
        "comparison; the two settings are not pooled)",
        ["Setting", "Metric", f"\u03c1 with {RC}, S vs O [BCa CI]", "S\u2013O preference, lowest \u2192 highest dose",
         "FPR [CP 95% CI] (k/n)", "Margin FPR (k/n)", "S\u2013O / R\u2013O / S\u2013R, lowest dose",
         f"\u03c1(excess, {RC}) (p_bh)", f"Switch S vs O, {RC} % [crossed CI]", f"Switch S vs size-matched, {RC} %"],
        body, widths=[1.6, 1.8, 1.9, 1.6, 2.0, 1.2, 1.9, 1.6, 2.0, 1.4], size=6.5,
        note="Sources: results/dose_response/responsiveness_corrected.csv, intervals_crossed.csv and "
             "overdetection.csv. p_bh of the excess: for SD1 Benjamini\u2013Hochberg over its four metrics (as P8 "
             "pre-registers); for Pet the value of overdetection.csv (BH over all settings and metrics). Switch "
             "marks as in the main responsiveness table of Section 5.2.")


def table_p9(p9_rows, bsw):
    body = []
    for key, (_, tested, untested) in p9_rows.items():
        b = bsw[key]
        for i, r in enumerate(sorted(tested + untested, key=lambda r: METRICS.index(r.metric))):
            body.append([
                SHORT[key] if i == 0 else "", switch_text(b) if i == 0 else "", r.metric,
                f1(r.metric_switch_rc) + (" (e)" if r.metric_switch_status == "identified at lowest-reliance edge"
                                          else "") if not pd.isna(r.metric_switch_rc) else "none (> 0.5 at lowest)",
                "\u2013" if pd.isna(r.difference) else f"{f1(r.difference)} {ci(r.diff_ci_low, r.diff_ci_high, f1)}",
                "\u2013" if pd.isna(r.difference) else ci(r.diff_crossed_ci_low, r.diff_crossed_ci_high, f1),
                r.verdict,
            ])
    return table(
        "P9: behavioural and metric switches (shortcut vs full object, primary label) in the three SD2 settings, "
        "with the A2 models",
        ["Setting", f"Behavioural switch, {RC}", "Metric", f"Metric switch, {RC} %",
         "Difference [two-level CI]", "Difference [crossed CI]", "Verdict (switch_differences.csv)"],
        body, widths=[2.0, 2.0, 2.2, 2.0, 2.6, 2.2, 3.0], size=7,
        note="Source: results/dose_response/switch_differences.csv and behaviour_switch.csv. Difference = "
             "behavioural \u2212 metric switch on the same bootstrap resamples. (e): metric switch at the "
             "lowest-reliance edge.")


# ------------------------------------------------------------------ amendment A3 (SD3, SD4; submission only)

P10_KEYS = [("pet_resnet18", "patch"), ("cifar10_resnet18", "patch"), ("cifar10", "patch"), SD1_KEY]
SD4_DOSES = [-0.5, -0.2, 0.0, 0.1, 0.25, 0.4]  # completion note A3-C1 (rule "below 60")
A3N = {}


def _road(table, base, shortcut, metric, comparison=None):
    sel = table[(table.base == base) & (table.shortcut == shortcut) & (table.metric == metric)]
    if comparison is not None:
        sel = sel[sel.comparison == comparison]
    assert len(sel) == 1, (base, shortcut, metric, comparison, len(sel))
    return sel.iloc[0]


def a3_numbers():
    """P10-P13 of amendment A3 applied literally to the result files (road_p10.csv and road_p11.csv
    for P10 and P11; responsiveness_corrected.csv and overdetection.csv for P12 and P13), with the
    guards for the statements in the text."""
    n = A3N
    # SD3: every pooled model has exact ROAD scores, on the A3 image counts (the timing gate was not reached)
    assert len(ROAD_MODELS) == N_MODELS and not len(ROAD_MODELS.merge(PILOTS[_KEYS], on=_KEYS))
    images = ROAD_MODELS.groupby(["base", "shortcut"]).images.agg(["min", "max"])
    assert (images["min"] == images["max"]).all()
    n["images"] = images["min"].to_dict()
    assert n["images"] == {k: 100 if k[0] in ("mnist", "pneumonia_mnist", "cifar10") else 50 for k in n["images"]}
    assert len(ROAD_TIMING) == 2 and (ROAD_TIMING.minutes < 15).all() and (ROAD_TIMING.images == 50).all()
    n["timing"] = list(ROAD_TIMING.minutes)
    # P10 (ROAD MoRF decides; LeRF secondary) and P11
    p10 = {(m, b, s): r for m, b, s, r in zip(ROAD_P10.road_metric, ROAD_P10.base, ROAD_P10.shortcut.fillna(""),
                                               ROAD_P10.itertuples())}
    n["p10"] = {m: [p10[(m, b, s)] for b, s in P10_KEYS] for m in ROAD}
    n["p10_overall"] = {m: p10[(m, "overall", "")].verdict for m in ROAD}
    for m in ROAD:
        for r in n["p10"][m]:
            black, road = (_road(ROAD_OVER, r.base, r.shortcut, x) for x in ("Deletion (black)", m))
            assert np.isclose(r.dose_low, black.dose_low) and int(r.models) == int(black.models_low)
            assert np.isclose(r.pref_ro_black, black.pref_ro_low) and np.isclose(r.pref_ro_road, road.pref_ro_low)
            assert r.pref_ro_black > 0.05 and r.verdict == "holds" and r.ci_low > 0
    assert n["p10_overall"] == {"ROAD (MoRF)": "holds", "ROAD (LeRF)": "holds"}, n["p10_overall"]
    p11 = {(m, b, s): r for m, b, s, r in zip(ROAD_P11.road_metric, ROAD_P11.base, ROAD_P11.shortcut,
                                               ROAD_P11.itertuples())}
    n["p11_keys"] = [("pet_resnet18", "patch"), SD1_KEY, SD4_KEY]
    n["p11"] = {(m, k): p11[(m, *k)] for m in ROAD for k in n["p11_keys"]}
    for (m, k), r in n["p11"].items():
        black, road = (_road(ROAD_OVER, *k, x) for x in ("Deletion (black)", m))
        assert (r.k_black, r.k_road, r.n_below) == (black.k_fpr, road.k_fpr, black.n_below), (m, k)
    p11_main = n["p11"][("ROAD (MoRF)", ("pet_resnet18", "patch"))]
    assert p11_main.role == "primary" and p11_main.n_below > 0 and p11_main.k_black > 0
    n["p11_main"] = p11_main
    # SD4: pilots, doses, testability (A3 stopping rule and the two-models-above-50 rule)
    g = group(*SD4_KEY)
    assert sorted(g.dose.unique()) == SD4_DOSES and sorted(g.seed.unique()) == [1, 2, 42] and len(g) == 18
    pil = PILOTS[PILOTS.model_set == "A3_pilot"].set_index("dose")
    assert sorted(pil.index) == [0.0, 0.25] and (pil.seed == 7).all()
    assert pil.loc[0.0].reliance_corrected < 60, "A3 dose rule key 'below 60'"
    n["stopped"] = bool(pil.loc[0.25].reliance_corrected < 20)
    n["pilots"] = pil
    n["sd4"] = g
    n["above"] = int((g.reliance_corrected > 50).sum())
    n["gap"] = largest_gap(g.reliance_corrected, lo=-5, hi=105)
    assert np.isclose(n["gap"][0], g[g.dose == 0].reliance_corrected.max()) and np.isclose(
        n["gap"][1], g[g.dose == 0.1].reliance_corrected.min()), "text: the SD4 gap lies between doses 0 and +0.1"
    testable = not n["stopped"] and n["above"] >= 2
    n["full"] = {m: row(FULL, *SD4_KEY, m) for m in METRICS}
    n["over"] = {m: orow(*SD4_KEY, m) for m in METRICS}
    n["p12_n"] = sum(r.rho_rc > 0.8 for r in n["full"].values())
    p12 = ("holds" if n["p12_n"] >= 3 else "refuted") if testable else "not testable"
    assert n["over"][METRICS[0]].n_below > 0
    p13 = ("holds" if all(n["over"][m].fpr > 0.5 for m in BB) else "refuted") if not n["stopped"] else "not testable"
    n["verdicts"] = {"P10": n["p10_overall"]["ROAD (MoRF)"], "P11": p11_main.verdict, "P12": p12, "P13": p13}
    # the text below states this pattern; rebuild the wording if a verdict changes
    assert n["verdicts"] == {"P10": "holds", "P11": "holds", "P12": "holds", "P13": "refuted"}, n["verdicts"]
    assert n["p12_n"] == 4 and n["over"]["Deletion (black)"].fpr == 1 and n["over"]["Deletion (blur)"].fpr < 0.5
    n["bsw"] = {lab: swrow(*SD4_KEY, lab) for lab in BEHAVIOUR_LABELS}
    assert all(r.status.startswith("not identified") for r in n["bsw"].values())
    assert n["gap"][0] < n["bsw"][PRIMARY_LABEL].behavioural_switch_rc < n["gap"][1]
    n["beh"] = {m: brow(FULL, *SD4_KEY, m) for m in METRICS}
    n["conf"] = {m: ccrow(*SD4_KEY, m) for m in METRICS}
    n["matched"] = {m: row(MATCHED, *SD4_KEY, m) for m in METRICS}
    n["road_sd4"] = {m: (_road(ROAD_RESP, *SD4_KEY, m, FULL), _road(ROAD_OVER, *SD4_KEY, m)) for m in ROAD}
    n["held"] = [p for p, v in n["verdicts"].items() if v == "holds"]
    return n


def table_a3_sd3():
    n = A3N
    body = []
    for i, (b, s) in enumerate(P10_KEYS):
        r, lerf = n["p10"]["ROAD (MoRF)"][i], n["p10"]["ROAD (LeRF)"][i]
        body.append(["P10" if i == 0 else "", SHORT[(b, s)], f"{int(r.models)} ({int(r.images)})",
                     f2(r.pref_ro_black), f2(r.pref_ro_road),
                     f"{f2(r.difference)} {ci(r.ci_low, r.ci_high)}", r.verdict,
                     f"{f2(lerf.pref_ro_road)}; {f2(lerf.difference)} {ci(lerf.ci_low, lerf.ci_high)}"])

    def fpr(k, n_, lo, hi):
        return f"{f2(k / n_)} {ci(lo, hi)} ({int(k)}/{int(n_)})"

    for i, key in enumerate(n["p11_keys"]):
        r, lerf = n["p11"][("ROAD (MoRF)", key)], n["p11"][("ROAD (LeRF)", key)]
        body.append(["P11" if i == 0 else "", SHORT[key] + ("" if i == 0 else " (secondary)"),
                     f"{int(r.n_below)} ({int(r.images)})",
                     fpr(r.k_black, r.n_below, r.fpr_black_cp_low, r.fpr_black_cp_high),
                     fpr(r.k_road, r.n_below, r.fpr_road_cp_low, r.fpr_road_cp_high),
                     f"{f2(r.difference)}; discordant {int(r.discordant_black_only)}/{int(r.discordant_road_only)}, "
                     f"sign test {p_text(r.p_sign_two_sided)}",
                     r.verdict if i == 0 else "\u2013",
                     fpr(lerf.k_road, lerf.n_below, lerf.fpr_road_cp_low, lerf.fpr_road_cp_high)])
    return table(
        "Exact ROAD (SD3): pre-registered predictions P10 and P11, on the same images for both metrics",
        ["Pred.", "Setting", "Models (images)", "Deletion (black)", "ROAD (MoRF)",
         "Difference black \u2212 ROAD [95% CI]", "Verdict", "ROAD (LeRF), secondary"],
        body, widths=[1.0, 2.6, 1.5, 2.3, 2.3, 3.0, 1.2, 2.6], size=7,
        note="Sources: results/dose_response/road_p10.csv and road_p11.csv. P10 rows: mean preference for the random "
             "map over the full object map at the setting's lowest-dose models; difference with its paired image "
             "bootstrap interval (2000 replicates, seed 0; images resampled within each model, the same "
             "images for both metrics); LeRF column: ROAD (LeRF) preference; difference [CI]. P11 rows: "
             f"over-detection rate in the shortcut-versus-object comparison among the models with {RC} \u2264 50 "
             "(Models column: their number), with Clopper\u2013Pearson 95% intervals; discordant: models on which "
             "only black deletion / only ROAD over-detects, with an exact two-sided sign test (reported, not "
             "deciding). Verdicts are judged on the pre-registered rules, ROAD (MoRF) only; the LeRF columns and "
             "the SD1 and SD4 rows of P11 are secondary outcomes without a verdict.")


def table_sd4():
    body = []
    for key in (SD4_KEY, SD1_KEY):
        for i, metric in enumerate(METRICS):
            r, o, b = row(FULL, *key, metric), orow(*key, metric), brow(FULL, *key, metric)
            body.append([
                SHORT[key] if i == 0 else "", metric,
                f"{f2(r.rho_rc)} {ci(r.rho_rc_ci_low, r.rho_rc_ci_high)}",
                f"{f2(r.pref_low)} \u2192 {f2(r.pref_high)}",
                f"{f2(o.fpr)} {ci(o.fpr_cp_low, o.fpr_cp_high)} ({int(o.k_fpr)}/{int(o.n_below)})",
                f"{f2(o.fpr_margin)} ({int(o.k_fpr_margin)}/{int(o.n_below)})",
                f"{f2(o.pref_so_low)} / {f2(o.pref_ro_low)} / {f2(o.pref_sr_low)}",
                f"{f2(b.auroc_pooled_strat)} / {f2(b.median_model_auroc_strat)}",
                fmt_switch(r),
            ])
    return table(
        "SD4 (watermark shortcut) next to SD1 (patch shortcut) on the same ImageNet-pretrained ResNet-18 "
        "(exploratory comparison; the two settings are not pooled)",
        ["Setting", "Metric", f"\u03c1 with {RC}, S vs O [two-level CI]",
         "S\u2013O preference, lowest \u2192 highest dose", "FPR [CP 95% CI] (k/n)", "Margin FPR (k/n)",
         "S\u2013O / R\u2013O / S\u2013R, lowest dose", "Behavioural AUROC, pooled / within model",
         f"Switch S vs O, {RC} %"],
        body, widths=[1.8, 1.9, 2.1, 1.8, 2.2, 1.3, 2.0, 1.8, 1.6], size=6.5,
        note="Sources: results/dose_response/responsiveness_corrected.csv, overdetection.csv and "
             "behaviour_summary.csv (class-stratified AUROC, primary label). P12 uses the \u03c1 column of SD4 and "
             "P13 the FPR column of black and blur deletion of SD4, both on the 100 scoring images. Switch marks "
             "as in the main responsiveness table of Section 5.2.")


def table_supp_road():
    body = []
    for key in [k[:2] for k in SETTINGS] + [SD1_KEY, SD4_KEY]:
        for i, metric in enumerate(METRICS + ROAD):
            r, o = _road(ROAD_RESP, *key, metric, FULL), _road(ROAD_OVER, *key, metric)
            body.append([
                f"{SHORT[key]} ({A3N['images'][key]})" if i == 0 else "", metric,
                f"{f2(r.rho_rc)}{sig(r)} {ci(r.rho_rc_ci_low, r.rho_rc_ci_high)}",
                f"{f2(r.pref_low)} \u2192 {f2(r.pref_high)}",
                f"{f2(o.fpr)} ({int(o.k_fpr)}/{int(o.n_below)})",
                f"{f2(o.fpr_margin)} ({int(o.k_fpr_margin)}/{int(o.n_below)})",
                f2(o.pref_ro_low), fmt_switch(r),
            ])
    return table(
        "Exact ROAD next to the four existing metrics, shortcut vs full object, all settings, on the ROAD "
        "images (number per model in brackets)",
        ["Setting (images)", "Metric", f"\u03c1 with {RC} [two-level CI]", "Preference, lowest \u2192 highest dose",
         "FPR (k/n)", "Margin FPR (k/n)", "R\u2013O, lowest dose", f"Switch, {RC} %"],
        body, widths=[2.6, 2.2, 2.4, 2.0, 1.6, 1.6, 1.4, 2.2], size=6.5, supp=True,
        note="Sources: results/dose_response/responsiveness_corrected_road.csv and overdetection_road.csv. The four "
             "existing metrics are recomputed on the same first N images as ROAD, so their values can differ from "
             "the main tables (100 images). \u2020: not significant after Benjamini\u2013Hochberg within the "
             "comparison over the nine settings and six metrics of this analysis. Switch marks as in the main "
             "responsiveness table.")


def table_supp_sd4_models():
    body = []
    rows_ = [(r, "confirmatory") for r in A3N["sd4"].sort_values(["dose", "seed"]).itertuples()]
    rows_ += [(r, "pilot (excluded)") for r in PILOTS[PILOTS.model_set == "A3_pilot"].sort_values("dose").itertuples()]
    for r, role in rows_:
        body.append([f2(r.dose), str(int(r.seed)), role, f1(r.reliance), f1(r.chance_floor),
                     f1(r.reliance_corrected), f1(r.test_accuracy), f1(r.test_accuracy_no_shortcut),
                     f1(r.test_accuracy_conflicting)])
    return table(
        "SD4 models: reliance and accuracies per model (watermark shortcut, ImageNet-pretrained ResNet-18)",
        ["Dose", "Seed", "Role", "R (%)", "Chance floor (%)", f"{RC} (%)", "Test accuracy (%)",
         "Accuracy without shortcut (%)", "Accuracy, conflicting (%)"],
        body, widths=[1.2, 1.0, 2.4, 1.3, 1.6, 1.3, 1.6, 2.0, 2.0], size=7, supp=True,
        note="Sources: results/dose_response/models.csv and pilots.csv. The seed-7 pilots fixed the doses under the "
             "A3 rule and are excluded from every analysis.")


def sd3_results(n):
    """Section 5.11 text and table (P10, P11 and the secondary ROAD outcomes)."""
    v = n["verdicts"]
    morf, lerf = n["p10"]["ROAD (MoRF)"], n["p10"]["ROAD (LeRF)"]
    names = [SHORT[k] for k in P10_KEYS]
    p11, p11_lerf = n["p11_main"], n["p11"][("ROAD (LeRF)", ("pet_resnet18", "patch"))]
    sd1_11, sd4_11 = n["p11"][("ROAD (MoRF)", SD1_KEY)], n["p11"][("ROAD (MoRF)", SD4_KEY)]
    assert (p11.discordant_black_only, p11.discordant_road_only) == (1, 0) and p11.k_road == p11.k_black - 1
    assert p11_lerf.k_road == p11_lerf.k_black, "text: LeRF over-detects as often as black on Pet"
    assert p11.fpr_road > 0.5, "text: ROAD still over-detects for most low-reliance Pet models"
    t = COUNTERS["table"] + 1
    para(f"**ROAD lowers the random-map bias (P10 {v['P10']}).** At the lowest-dose models black deletion "
         f"prefers a random map over the full object map on {span([r.pref_ro_black for r in morf])} of the images "
         f"and ROAD (MoRF) on {span([r.pref_ro_road for r in morf])}. The difference black \u2212 ROAD is "
         + "; ".join(f"{f2(r.difference)} {ci(r.ci_low, r.ci_high)} on {s} ({int(r.models)} models, "
                     f"{int(r.images)} images each)" for r, s in zip(morf, names))
         + ". Every interval lies above 0, so P10 holds in all four settings, and all four are testable. ROAD "
         f"(LeRF) lowers the preference too, by {span([r.difference for r in lerf])}, with every interval above 0 "
         "(secondary).")
    para(f"**ROAD over-detects slightly less than black deletion on Pet (P11 {v['P11']}).** Among the "
         f"{int(p11.n_below)} Pet models with {RC} \u2264 50, ROAD (MoRF) prefers the shortcut map for "
         f"{int(p11.k_road)} ({f2(p11.fpr_road)} {ci(p11.fpr_road_cp_low, p11.fpr_road_cp_high)}) and black "
         f"deletion for {int(p11.k_black)} ({f2(p11.fpr_black)} {ci(p11.fpr_black_cp_low, p11.fpr_black_cp_high)}). "
         "The prediction holds on the point values, as the rule specifies, but by one model: one model "
         "over-detects only under black deletion and none only under ROAD (exact sign test "
         f"{p_text(p11.p_sign_two_sided)}), the intervals overlap almost completely, and ROAD still "
         "over-detects for most low-reliance Pet models. ROAD (LeRF) over-detects for "
         f"{int(p11_lerf.k_road)} of {int(p11_lerf.n_below)}, as often as black deletion. In the secondary repeats "
         f"ROAD (MoRF) over-detects for {int(sd1_11.k_road)} of {int(sd1_11.n_below)} SD1 models (black deletion "
         f"{int(sd1_11.k_black)}) and for {int(sd4_11.k_road)} of {int(sd4_11.n_below)} SD4 models (black deletion "
         f"{int(sd4_11.k_black)}), on the same 50 images.")
    road_rho = {m: [_road(ROAD_RESP, *k[:2], m, FULL).rho_rc for k in SETTINGS + [SD1, SD4]] for m in ROAD}
    nat = {m: [_road(ROAD_OVER, *k, m).fpr for k in P10_KEYS] for m in ROAD}
    lerf_mech = [_road(ROAD_OVER, b, "patch", "ROAD (LeRF)") for b in ("mnist", "pneumonia_mnist")]
    assert all(o.fpr == 1 for o in lerf_mech), "text: LeRF over-detects every low-reliance mechanism patch model"
    morf_rows = [_road(ROAD_RESP, *k[:2], "ROAD (MoRF)", FULL) for k in SETTINGS + [SD1, SD4]]
    assert all(r.rho_rc > 0 and r.p_bh < 0.05 for r in morf_rows), "text: MoRF rises in every setting"
    para(f"**Other comparisons (secondary).** Supplementary Table S{len(SUPP_ORDER) + 1} lists ROAD next to the "
         "four existing metrics for every setting, recomputed on the same images. Against the full object map "
         "the preference of ROAD (MoRF) rises significantly with reliance in all nine settings (\u03c1 = "
         f"{span(road_rho['ROAD (MoRF)'])}), and its over-detection rate on the four natural-image settings is "
         + ", ".join(f"{f2(x)} ({s})" for x, s in zip(nat["ROAD (MoRF)"], names))
         + ". ROAD (LeRF) behaves differently on the mechanism settings: it prefers the shortcut map for every "
         f"low-reliance model on MNIST patch ({int(lerf_mech[0].n_below)} models) and PneumoniaMNIST patch "
         f"({int(lerf_mech[1].n_below)}), and its \u03c1 spans {span(road_rho['ROAD (LeRF)'])}.")
    table_a3_sd3()
    assert COUNTERS["table"] == t


def sd4_results(n):
    """Section 5.12 text and table (P12, P13 and the secondary SD4 outcomes)."""
    v, g, full, over = n["verdicts"], n["sd4"], n["full"], n["over"]
    bb = [over[m] for m in BB]
    sd1_over = {m: orow(*SD1_KEY, m) for m in METRICS}
    below = g[g.reliance_corrected <= 50]
    assert (below.reliance_corrected <= 10).all() and over[METRICS[0]].n_rc_le10 == len(below)
    assert all(over[m].k_fpr == 0 for m in II), "text: inpainting and insertion over-detect for none"
    assert full["Deletion (black)"].rho_rc_ci_low < 0.8
    sd1_k = {m: int(orow(*SD1_KEY, m).k_fpr) for m in BB}
    assert sd1_k["Deletion (blur)"] > over["Deletion (blur)"].k_fpr and abs(
        sd1_k["Deletion (black)"] - over["Deletion (black)"].k_fpr) <= 1, "text: more often / about as often"
    bsw = n["bsw"][PRIMARY_LABEL]
    t = COUNTERS["table"] + 1
    para(f"The {len(g)} SD4 models reach {RC} from {f1(g.reliance_corrected.min())}% to "
         f"{f1(g.reliance_corrected.max())}%. The doses \u22120.5 to 0 give at most {f1(n['gap'][0])}% and every "
         f"model at +0.1 reaches at least {f1(n['gap'][1])}%, so {len(below)} models lie at {RC} \u2264 50 (all of "
         f"them at {RC} \u2264 10) and {n['above']} above; as on SD1, reliance jumps between two adjacent doses. "
         "The stopping rule was not triggered and more than two models lie above 50%, so P12 and P13 are "
         f"testable. Table {t} sets SD4 next to SD1 and Supplementary Table S{len(SUPP_ORDER) + 2} lists the models.")
    para(f"**The metrics respond to watermark reliance (P12 {v['P12']}).** Spearman \u03c1 between {RC} and the "
         "shortcut-versus-object preference is "
         + ", ".join(f"{f2(full[m].rho_rc)} {ci(full[m].rho_rc_ci_low, full[m].rho_rc_ci_high)} ({m})"
                     for m in METRICS)
         + f" on the 100 scoring images, above 0.8 for {n['p12_n']} of 4. The decision uses the point estimate, as "
         "pre-registered; the two-level interval of black deletion reaches "
         f"{f2(full['Deletion (black)'].rho_rc_ci_low)}. With no model between {f1(n['gap'][0])}% and "
         f"{f1(n['gap'][1])}%, \u03c1 here mostly measures how well the metrics separate the low-reliance from "
         "the high-reliance models.")
    para(f"**Blur deletion does not over-detect the watermark (P13 {v['P13']}).** Among the {len(below)} models "
         f"with {RC} \u2264 50, black deletion prefers the shortcut map for {int(bb[0].k_fpr)} ({f2(bb[0].fpr)} "
         f"{ci(bb[0].fpr_cp_low, bb[0].fpr_cp_high)}), but blur deletion only for {int(bb[1].k_fpr)} "
         f"({f2(bb[1].fpr)} {ci(bb[1].fpr_cp_low, bb[1].fpr_cp_high)}), so the rate of blur deletion is not "
         f"above 0.5. With a per-model margin the two rates are {f2(bb[0].fpr_margin)} and {f2(bb[1].fpr_margin)}, "
         "and inpainting and insertion over-detect for none of the models. At the lowest dose black deletion "
         f"prefers a random map over the object map on {f2(bb[0].pref_ro_low)} of the images and the shortcut map "
         f"over a random map on {f2(bb[0].pref_sr_low)}, so, as with the patch, its over-detection comes from the "
         "object map rather than from the watermark. On SD1 the same network over-detected the patch more often "
         f"under blur deletion ({int(sd1_over['Deletion (blur)'].k_fpr)} of {int(sd1_over['Deletion (blur)'].n_below)}"
         f") and about as often under black deletion ({int(sd1_over['Deletion (black)'].k_fpr)} of "
         f"{int(sd1_over['Deletion (black)'].n_below)}).")
    beh, conf = n["beh"], n["conf"]
    road = n["road_sd4"]
    assert all(o.k_fpr == 0 for _, o in road.values()), "text: ROAD over-detects for none of the SD4 models"
    para(f"**Secondary outcomes (no verdict).** The SD4 behavioural switch is not identified: the isotonic crossing "
         f"lies at {f1(bsw.behavioural_switch_rc)}%, but no model lies between {RC} {f1(n['gap'][0])}% and "
         f"{f1(n['gap'][1])}% (the same holds under both check labels). Against the primary label the "
         f"class-stratified pooled AUROC is {span([r.auroc_pooled_strat for r in beh.values()])} and the "
         f"within-model median {span([r.median_model_auroc_strat for r in beh.values()])}; "
         f"{pct(bsw.undetermined_rate_mean)} of the images are undetermined on average (up to "
         f"{pct(bsw.undetermined_rate_max)}). Reliance and object competence are collinear (\u03c1({RC}, acc) = "
         f"{f2(conf[METRICS[0]].rho_rc_competence)}); the partial \u03c1 of preference with {RC} given competence "
         "is " + ", ".join(f"{f2(conf[m].partial_rc_given_acc)} ({m})" for m in METRICS) + ". Exact ROAD on the "
         f"SD4 models (50 images) gives \u03c1 = {f2(road['ROAD (MoRF)'][0].rho_rc)} (MoRF) and "
         f"{f2(road['ROAD (LeRF)'][0].rho_rc)} (LeRF), and in neither order does it over-detect for any of the "
         f"{int(road['ROAD (MoRF)'][1].n_below)} low-reliance models. The comparison "
         "with SD1 is exploratory: SD4 changes the shortcut's shape, contrast and position within the corner at "
         "once.")
    table_sd4()
    assert COUNTERS["table"] == t


# ------------------------------------------------------------------ helpers for the text


def largest_gap(values, lo=5, hi=95):
    """(start, end) of the widest gap between consecutive sorted values inside [lo, hi]."""
    v = np.sort([x for x in values if lo <= x <= hi])
    d = np.diff(v)
    i = int(np.argmax(d))
    return v[i], v[i + 1]


def sig_positive(sel):
    return int(((sel.rho_rc > 0) & (sel.p_bh < 0.05)).sum())


def word_count(text):
    return len(text.replace("\u2013", " ").split())


# ------------------------------------------------------------------ the paper


class use_doc:
    """Write the helpers' output into another document for the length of a with-block."""

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        self.old = globals()["doc"]
        globals()["doc"] = self.target

    def __exit__(self, *exc):
        globals()["doc"] = self.old


# Every [[...]] marker that the submission build may still contain. They are listed, with what replaces
# them, in docs/submission/DEFERRED.md; `--final` fails while any marker remains in the paper or in
# docs/submission/*.md. A marker that is not declared here fails every submission build.
DECLARED_MARKERS: set[str] = set()  # the Zenodo release DOI (10.5281/zenodo.24551101) replaced the last one


def declarations():
    if SUBMISSION:
        # order before the reference list: acknowledgements, AI declaration, CRediT, competing interest, funding,
        # data availability (nothing of this is on the title page)
        heading("Acknowledgements")
        para("The author thanks Dr. Raafiya Gulmaher, Assistant Professor, Department of Computer Science and "
             "Engineering, Khaja Bandanawaz University, for serving as the guide of the author's M.Tech project.")
        heading("Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")
        para("During the preparation of this work the author used the Kiro AI coding assistant, with Anthropic's "
             "Claude models and other large language models available in Kiro, to help write and run the analysis "
             "code, to help draft and edit the manuscript text, tables and supplementary material, and to help "
             "search for and check literature. After using these tools, the author reviewed and edited the content "
             "as needed, checked the reported numbers against the result files and every reference against its "
             "source, and takes full responsibility for the content of the published article.")
        heading("CRediT authorship contribution statement")
        para("**Md Junaid Faizan Shaikh:** Conceptualization, Methodology, Software, Investigation, Formal "
             "analysis, Data curation, Visualization, Writing \u2013 original draft, Writing \u2013 review & "
             "editing. Sole author.")
        heading("Declaration of competing interest")
        para("The author declares that he has no known competing financial interests or personal relationships "
             "that could have appeared to influence the work reported in this paper.")
        heading("Funding")
        para("This research did not receive any specific grant from funding agencies in the public, commercial, "
             "or not-for-profit sectors.")
        heading("Data availability")
        para("The research data and code are deposited in the Zenodo repository " + cite("code") + " (DOI "
             "10.5281/zenodo.24551101): the code, the per-model result files, the analysis tables and the trained "
             "checkpoints. The code is released "
             "under the MIT licence and the data, results and checkpoints under CC BY 4.0. The pre-registered "
             "predictions and amendments are archived in the Zenodo record 10.5281/zenodo.24285531 (concept DOI). "
             "The raw images are not redistributed. They are available from their owners: MNIST " + cite("lecun")
             + ", PneumoniaMNIST " + cite("medmnist") + " (MedMNIST), CIFAR-10 " + cite("cifar") + " and "
             "Oxford-IIIT Pet " + cite("pets") + " (CC BY-SA 4.0).")
        return
    heading("CRediT authorship contribution statement")
    para("**Md Junaid Faizan Shaikh:** Conceptualization, Methodology, Software, Investigation, Formal analysis, "
         "Visualization, Writing \u2013 original draft, Writing \u2013 review & editing. [Guide Name]: Supervision "
         "[to be confirmed].")
    heading("Declaration of competing interest")
    para("The author declares no known competing financial interests or personal relationships that could have "
         "appeared to influence the work reported in this paper.")
    heading("Data availability")
    para("[Placeholder: link to the public code repository with result files and checkpoints.] MNIST, CIFAR-10, "
         "PneumoniaMNIST and Oxford-IIIT Pet are publicly available.")
    heading("Declaration of generative AI and AI-assisted technologies in the writing process")
    para("During the preparation of this work the author used AI assistants for code development and for "
         "drafting the text. After using these tools, the author reviewed and edited the content as needed and "
         "takes full responsibility for the content of the publication.")
    heading("Funding")
    para("This research did not receive any specific grant from funding agencies in the public, commercial, or "
         "not-for-profit sectors.")
    heading("Acknowledgements")
    para("The author thanks [Guide Name] for guidance and feedback on this work.")


def appendices_bc():
    heading(S("Appendix B. Legacy analysis", "Legacy analysis (Table S4)"))
    para(S("Earlier drafts used raw reliance, a bootstrap over seeds only, an unbalanced decision accuracy and a "
           "first-crossing switch rule. These values are kept for comparison and are superseded by Section 5.",
           "The first analysis of the models used raw reliance, a bootstrap over seeds only, an unbalanced decision "
           "accuracy and a first-crossing switch rule. These values are kept for comparison and are superseded by "
           "Section 5 of the main text."))
    table_legacy()

    # ---- Appendix C
    heading(S("Appendix C. Original and amended model sets", "Original and amended model sets (Table S5)"))
    para(f"Amendment A1 committed to reporting both the {N_ORIGINAL} original models "
         + S("(results/dose_response/original_models/)", "(analysed separately)") + " "
         f"and the {N_ORIGINAL + N_A1} models that include its {N_A1} added MNIST and "
         "PneumoniaMNIST models. The added models strengthen the rank correlations slightly and weaken the "
         "decision-level results: over-detection rises and the switch points move. Amendment A2 makes the same "
         f"commitment for its {N_A2} models: the {N_ORIGINAL + N_A1} models without them are analysed "
         + S("in results/dose_response/pre_a2_models/", "separately") + f", and the analysis of all {N_MODELS} models adds the {N_A2_SD2} "
         f"SD2 models to their settings and the {N_SD1} SD1 models as a separate setting"
         + S("", f", and the {N_SD4} SD4 models of amendment A3 as another") + ". A2 added no MNIST or "
         "PneumoniaMNIST model, so the table below compares the original models with the models including A1; "
         "the A2 changes on the natural images are in Sections 5.4 and 5.10" + S("", " of the main text") + ".")
    table_sets()
    if SUBMISSION:
        heading("Exact ROAD and the watermark shortcut (Tables S6 and S7)")
        para("Amendment A3 adds exact ROAD (SD3, Section 5.11 of the main text) and the watermark shortcut (SD4, "
             f"Section 5.12). Table S6 lists ROAD (MoRF and LeRF) next to the four existing metrics for every "
             f"setting in the shortcut-versus-object comparison, all on the ROAD images; the other comparisons are "
             "in responsiveness_corrected_road.csv and overdetection_road.csv. Table S7 lists the SD4 models and "
             "pilots.")
        table_supp_road()
        table_supp_sd4_models()


def doc_text(d):
    """All text of a python-docx document: paragraphs and table cells."""
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for r in t.rows:
            parts.extend(c.text for c in r.cells)
    return "\n".join(parts)


# Words that must not appear in the journal text (internal revision history and repository paths). "original
# draft" is a CRediT role name and is allowed.
FORBIDDEN = [r"audit", r"reviewer", r"(?<!original )(?<!help )\bdrafts?\b", r"docs/", r"guide review",
             r"(?<!serving as the )\bguide\b",
             r"commit(ted)? [0-9a-f]{7}\b", r"UNVERIFIED", r"not for submission"]
MARKER = r"\[\[[^\]]+\]\]"


# American spellings that must not appear in the (British) journal text. "Conceptualization" and "Visualization"
# are CRediT role names and are allowed.
US_SPELLINGS = (r"behavior|colou?r(?<!colour)|gray|normaliz|analyz|\bcenter|neighbor|favor|catalog|defense|"
                r"labeled|labeling|modeling|modeled|summariz|organiz|recogniz|optimiz|generaliz|initializ|"
                r"minimiz|randomiz|standardiz|characteriz|utiliz|emphasiz|parameteriz|artifact|fulfill|judgment|"
                r"\w{3,}(?<![sS])(?<!Conceptual)(?<!Visual)iz(?:e|es|ed|ing|ation)\b")
FIGURE_MIN_PX = 2244  # Elsevier: full-page width for a bitmap
FIGURE_VECTOR_STEMS = ("fig_rc_",)  # figures regenerated from the CSVs in this file (vector PDF + 400 dpi PNG)
ENGLISH_HEADINGS = ["Acknowledgements", "Declaration of generative AI and AI-assisted technologies in the manuscript "
                    "preparation process", "CRediT authorship contribution statement", "Declaration of competing "
                    "interest", "Funding", "Data availability"]


def guide_checks(main, supp, main_text, n_figs):
    """Neurocomputing / Elsevier guide for authors: formatting rules that a Word file can be checked for."""
    import re
    # tables: horizontal rules only, no shading, no vertical borders
    for name, d in (("main text", main), ("supplementary", supp)):
        assert not list(d.element.body.iter(qn("w:shd"))), f"{name}: shaded table cells"
        for tag in ("w:left", "w:right", "w:start", "w:end", "w:insideV"):
            assert not list(d.element.body.iter(qn(tag))), f"{name}: vertical table border {tag}"
        for style in d.element.body.iter(qn("w:tblStyle")):
            assert style.get(qn("w:val")) != "TableGrid", f"{name}: Table Grid style has vertical rules"
        assert not any(r.font.strike or r.font.underline for p in d.paragraphs for r in p.runs), \
            f"{name}: strikethrough or underline"
    # keywords: 1 to 7, none with "and" or "of" in a multi-word keyword
    assert 1 <= len(KEYWORDS) <= 7, KEYWORDS
    assert not [k for k in KEYWORDS if re.search(r"\b(and|of)\b", k, flags=re.I)], KEYWORDS
    assert " \u00b7 ".join(KEYWORDS) in main_text, "keywords of the front matter differ from KEYWORDS"
    # highlights: 3 to 5, at most 85 characters each
    lines = HIGHLIGHTS_OUTPUT.read_text(encoding="utf8").splitlines()
    assert 3 <= len(lines) <= 5 and all(len(x) <= 85 for x in lines), [len(x) for x in lines]
    # one English variant (British); the reference list keeps the original titles
    body = main_text.split("\nReferences\n")[0] if "\nReferences\n" in main_text else main_text
    for name, text in (("main text", body), ("supplementary", doc_text(supp))):
        hit = re.search(US_SPELLINGS, text)
        assert not hit, f"{name}: American spelling {hit.group(0)!r} near {text[max(hit.start() - 50, 0):hit.end() + 50]!r}"
    # title page: author, lowercase superscript letter, address marker or address with country, corresponding author
    head = main_text[:1500]
    assert "Corresponding author" in head and "ORCID" in head and "Khaja Bandanawaz University" in head
    # declarations: order before the reference list, none on the title page
    positions = [main_text.rfind("\n" + h + "\n") for h in ENGLISH_HEADINGS]
    assert all(p > 0 for p in positions) and positions == sorted(positions), (positions, [main_text[max(x-30,0):x+60] for x in positions])
    assert main_text.rfind("\nReferences\n") > positions[-1], "declarations must precede the reference list"
    # figures: files Figure_N.png / .pdf exist, are cited in order, are large enough, text sized; appendix as A1
    assert len(FIGURE_FILES) == n_figs + COUNTERS["afig"], (len(FIGURE_FILES), n_figs)
    for f in FIGURE_FILES:
        assert f["png"].exists() and f["pdf"].exists() and f["png"].name == f"Figure_{f['label']}.png", f
        if f["vector"]:
            assert f["width_px"] >= FIGURE_MIN_PX, f
    first = [re.search(rf"Fig\. {n}\b", main_text) for n in range(1, n_figs + 1)]
    assert all(first), "a figure is never cited"
    starts = [m.start() for m in first]
    assert starts == sorted(starts), f"figures are not first cited in numerical order {starts}"
    # appendix: its table and figure are numbered A.1, cross-referenced by number
    assert re.search(r"^Table A\.1\. ", main_text, flags=re.M) and re.search(r"^Fig\. A\.1\. ", main_text, flags=re.M)
    assert not re.search(r"\bthe (table|figure) (above|below)\b", main_text, flags=re.I), "refer to tables by number"


def submission_checks(main, supp, final=False):
    """Format checks of the submission build; `final` also fails while any [[marker]] remains."""
    import re
    texts = {"main text": doc_text(main), "supplementary": doc_text(supp)}
    for name, text in texts.items():
        assert TITLE in text, f"{name}: the official title must appear exactly"
        for pattern in FORBIDDEN:
            hit = re.search(pattern, text, flags=re.IGNORECASE)
            assert not hit, f"{name}: forbidden text {hit.group(0)!r} near {text[max(hit.start() - 60, 0):hit.end() + 60]!r}"
    old_title = re.compile(r"Do XAI Metrics Notice|Do Explanation Metrics Catch|Ground-Truth Evaluation of XAI Metrics")
    assert not any(old_title.search(t) for t in texts.values()), "an earlier title appears"
    # cross-references: every cited table or figure number exists
    main_text = texts["main text"]
    n_tables, n_figs, n_supp = COUNTERS["table"], COUNTERS["fig"], COUNTERS["stable"]
    assert n_supp == len(SUPP_ORDER) + 2, (n_supp, SUPP_ORDER)  # + S6 and S7 (amendment A3)
    plain = re.sub(r"Supplementary Table S\d+", "", main_text)
    for n in {int(x) for x in re.findall(r"\bTable (\d+)\b", plain)}:
        assert 1 <= n <= n_tables, f"Table {n} is cited but the main text has {n_tables} tables"
    for n in {int(x) for x in re.findall(r"\bFig\. (\d+)\b", main_text)}:
        assert 1 <= n <= n_figs, f"Fig. {n} is cited but the main text has {n_figs} figures"
    for text in texts.values():
        for n in {int(x) for x in re.findall(r"Table S(\d+)", text)}:
            assert 1 <= n <= n_supp, f"Table S{n} is cited but the supplement has {n_supp} tables"
    # a table that moved to the supplement is no longer cited as a main-text table
    for caption_number in range(1, n_tables + 1):
        assert re.search(rf"^Table {caption_number}\. ", main_text, flags=re.M), f"Table {caption_number} has no caption"
    guide_checks(main, supp, main_text, n_figs)
    # markers: only the declared ones; --final: none at all, in the paper and in docs/submission
    found = {m for t in texts.values() for m in re.findall(MARKER, t)}
    undeclared = found - DECLARED_MARKERS
    assert not undeclared, f"undeclared markers: {sorted(undeclared)}"
    if final:
        remaining = {f"paper: {m}" for m in found}
        folder = ROOT / "docs" / "submission"
        for md in sorted(folder.glob("*.md")) if folder.exists() else []:
            if md.name in ("DEFERRED.md", "submission_checklist.md"):
                continue
            remaining |= {f"{md.name}: {m}" for m in re.findall(MARKER, md.read_text(encoding="utf8"))}
        if remaining:
            raise SystemExit("--final: unfinished markers remain:\n  " + "\n  ".join(sorted(remaining)))
    print(f"submission checks passed: {len(found)} declared markers remain, {n_tables} tables, {n_figs} figures, "
          f"{n_supp} supplementary tables")


def build():
    setup()
    if SUBMISSION:
        setup(SUPP, header_text="Supplementary Material")
        with use_doc(SUPP):
            t = doc.add_paragraph()
            t.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = t.add_run("Supplementary Material to: " + TITLE)
            run.bold, run.font.size = True, Pt(14)
            para("Md Junaid Faizan Shaikh", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
            para("This document holds material that supports the main text: the full table of the pre-registered "
                 "Pet predictions P1\u2013P4 (Table S1; abridged in the main text), the responsiveness of the metrics "
                 "against the size-matched object map (Table S2), the paired metric-versus-metric differences "
                 "(Table S3), the legacy analysis (Table S4), the check on the original model set (Table S5), exact "
                 "ROAD next to the existing metrics (Table S6) and the models of the watermark setting (Table S7). "
                 "Section and figure numbers refer to the main text. All numbers are generated from the result "
                 "files by the analysis code.")
    figs = {
        "full": preference_figure(FULL, FIG_OUT / "fig_rc_full_object.png", "Preference for shortcut map"),
        "matched": preference_figure(MATCHED, FIG_OUT / "fig_rc_size_matched.png", "Preference for shortcut map"),
        "s_random": preference_figure(S_RANDOM, FIG_OUT / "fig_rc_shortcut_vs_random.png",
                                      "Preference for shortcut map", show_behaviour=False),
        "r_object": preference_figure(R_OBJECT, FIG_OUT / "fig_rc_random_vs_object.png",
                                      "Preference for random map", show_behaviour=False),
        "p_random": preference_figure(P_RANDOM, FIG_OUT / "fig_rc_placebo_vs_random.png",
                                      "Preference for placebo map", show_behaviour=False),
        # the analyze figures have one panel per setting in models.csv, SD1 and SD4 included
        "behaviour": rearrange_panels(DR / "behaviour_auroc.png", FIG_OUT / S("fig_behaviour_auroc.png",
                                      "fig_behaviour_auroc_submission.png"), N_PANELS, order=PANEL_ORDER),
        "methods": rearrange_panels(DR / "method_dose_response.png", FIG_OUT / S("fig_methods.png",
                                    "fig_methods_submission.png"), N_PANELS, order=PANEL_ORDER),
    }

    # ---- numbers used throughout (all from the CSVs)
    full = rows([FULL], SETTINGS)
    full_nat = rows([FULL], NATURAL)
    n_full_sig, n_full = sig_positive(full), len(full)
    full_weak = full[~((full.rho_rc > 0) & (full.p_bh < 0.05))]
    full_sig = full.drop(full_weak.index)
    n_tpr1, n_fpr_pos = int((full.tpr == 1).sum()), int((full.fpr > 0).sum())
    nat_bb_fpr = rows([FULL], NATURAL, BB).fpr
    nat_ii_fpr = rows([FULL], NATURAL, II).fpr
    cifar_bb_fpr = rows([FULL], NATURAL[1:], BB).fpr
    pet_bb_fpr = rows([FULL], NATURAL[:1], BB).fpr
    pet_full = {m: row(FULL, "pet_resnet18", "patch", m) for m in METRICS}
    pet_switch = [r.switch_iso_rc for r in pet_full.values()]
    pet_no_switch = [m for m, r in pet_full.items() if pd.isna(r.switch_iso_rc)]
    beh_switch = {(b, s): behavioural_switch(b, s) for b, s, *_ in SETTINGS}
    nat_beh_switch = [beh_switch[(b, s)] for b, s, *_ in NATURAL]
    pooled_nat = [brow(FULL, b, s, m).auroc_pooled_strat for b, s, *_ in NATURAL for m in METRICS]
    within_nat = [brow(FULL, b, s, m).median_model_auroc_strat for b, s, *_ in NATURAL for m in METRICS]
    pooled_mech = [brow(FULL, b, s, m).auroc_pooled_strat for b, s, *_ in MECHANISM for m in METRICS]
    within_mnist = [brow(FULL, b, s, m).median_model_auroc_strat for b, s, *_ in MECHANISM[:2] for m in METRICS]
    pp_pooled = [brow(FULL, "pneumonia_mnist", "patch", m).auroc_pooled_strat for m in METRICS]
    pp_black = brow(FULL, "pneumonia_mnist", "patch", "Deletion (black)")
    pp_band = BAND[(BAND.comparison == FULL) & (BAND.base == "pneumonia_mnist") & (BAND.shortcut == "patch")
                   & (BAND.label == PRIMARY_LABEL) & (BAND.band_centre >= 60)]
    assert len(pp_band) > 0
    gap_nat = [brow(FULL, b, s, m).calibration_gap for b, s, *_ in NATURAL for m in METRICS]
    assert min(gap_nat) > 0, "text says every natural-image calibration gap is positive"
    pet_gap_low = [erow(FULL, "pet_resnet18", "patch", m, "low").median_gap for m in METRICS]
    mp_full, mp_matched = rows([FULL], [MECHANISM[0]]), rows([MATCHED], [MECHANISM[0]])
    mp_orig = rows([FULL], [MECHANISM[0]], table=RESP_ORIG)
    orig_mp_rel = MODELS[(MODELS.base == "mnist") & (MODELS.shortcut == "patch") & (MODELS.model_set == "original")].reliance
    all_mp_rel = group("mnist", "patch").reliance
    gap_orig, gap_all = largest_gap(orig_mp_rel), largest_gap(all_mp_rel)
    ro = {(b, m): row(R_OBJECT, b, "patch", m) for b, *_ in NATURAL for m in METRICS}
    ro_bb = {b: [ro[(b, m)].pref_low for m in BB] for b, *_ in NATURAL}
    ro_ii = {b: [ro[(b, m)].pref_low for m in II] for b, *_ in NATURAL}
    sf_patch_settings = [s for s in SETTINGS if s[1] == "patch"]
    sf_low = rows([S_RANDOM, S_PLACEBO], sf_patch_settings).pref_low
    pr_del = rows([P_RANDOM], SETTINGS, METRICS[:3])
    pr_ins_mnist = rows([P_RANDOM], MECHANISM[:2], ["Insertion (blur)"])
    pet_pr_ins = row(P_RANDOM, "pet_resnet18", "patch", "Insertion (blur)")
    robust = {}
    for m in METRICS:
        sel = rows(SHORTCUT_COMPARISONS, SETTINGS, [m])
        robust[m] = (sig_positive(sel), len(sel), sel.rho_rc.min())
    robust_all = [m for m, (ok, n, _) in robust.items() if ok == n]
    pt_pairs = {b: prow(FULL, "pneumonia_mnist", "texture", "Deletion (black)", b) for b in METRICS[1:]}
    n_pairs_sig = int((main_only(PAIRS).p_bh < 0.05).sum())
    pt_black_sr = row(S_RANDOM, "pneumonia_mnist", "texture", "Deletion (black)")
    pt_black_sp = row(S_PLACEBO, "pneumonia_mnist", "texture", "Deletion (black)")
    pt_black_m = row(MATCHED, "pneumonia_mnist", "texture", "Deletion (black)")
    mt_ins = row(FULL, "mnist", "texture", "Insertion (blur)")
    mt_sr_blur = row(S_RANDOM, "mnist", "texture", "Deletion (blur)")
    mt_sr_ins = row(S_RANDOM, "mnist", "texture", "Insertion (blur)")
    mt_sp_ins = row(S_PLACEBO, "mnist", "texture", "Insertion (blur)")
    ig = main_only(METHOD[METHOD.method == "Integrated Gradients"]).rho_rc
    mt_away = {m: mrow("mnist", "texture", m) for m in ["Saliency", "SmoothGrad", "Grad-CAM"]}
    pt_meth = {m: mrow("pneumonia_mnist", "texture", m) for m in ["Saliency", "SmoothGrad", "Grad-CAM"]}
    assert all(r.rho_rc < 0 for r in mt_away.values()) and pt_meth["Saliency"].rho_rc > 0 \
        and pt_meth["SmoothGrad"].rho_rc > 0 and pt_meth["Grad-CAM"].rho_rc < 0
    gc_lowest = sum(min(METHODS, key=lambda m: mrow(b, s, m).mass_high) == "Grad-CAM" for b, s, *_ in sf_patch_settings)
    conf_cifar = CONF_RC[(CONF_RC.base == "cifar10")]
    conf_set = main_only(CONF_RC).groupby(["base", "shortcut"]).rho_rc_competence.first()
    cnn_pre_a2 = pd.read_csv(DR / "pre_a2_models" / "confound_rc.csv").query("base == 'cifar10'").iloc[0]
    assert cnn_pre_a2.rho_rc_competence == pd.read_csv(DR / "pre_a2_models" / "confound_rc.csv"
                                                       ).rho_rc_competence.max(), "text: least collinear before A2"
    conf_all_sig = {k for k, g in main_only(CONF_RC).groupby(["base", "shortcut"]) if (g.p_bh_rc < 0.05).all()}
    assert ("cifar10", "patch") in conf_all_sig, "text: the CIFAR CNN keeps all four partial rho significant"
    collinear = conf_set[conf_set <= -0.95]
    # ---- audit-2 numbers (M1-M4; from the new CSVs)
    bsw = {(b, s): swrow(b, s) for b, s, *_ in SETTINGS}
    pet_bsw = bsw[("pet_resnet18", "patch")]
    cifar_bsw = [bsw[(b, s)] for b, s, *_ in NATURAL[1:]]
    pet_label_sw = [swrow("pet_resnet18", "patch", lab).behavioural_switch_rc for lab in BEHAVIOUR_LABELS]
    pt_label_sw = [swrow("pneumonia_mnist", "texture", lab).behavioural_switch_rc for lab in BEHAVIOUR_LABELS]
    und_rho = main_only(BSWITCH[BSWITCH.primary & BSWITCH.undetermined_rate_max.gt(0.1)]).rho_undetermined_rc
    pp_dom_pooled = BEH[(BEH.comparison == FULL) & (BEH.base == "pneumonia_mnist") & (BEH.shortcut == "patch")
                        & (BEH.label == "shortcut_dominant")].auroc_pooled_strat
    mp_dom_black = brow(FULL, "mnist", "patch", "Deletion (black)", "shortcut_dominant")
    gap_det_nat = [calrow(FULL, b, s, m).calibration_gap_determined for b, s, *_ in NATURAL for m in METRICS]
    nat_power = [powrow(b, s) for b, s, *_ in NATURAL]
    over = {(b, m): orow(b, "patch", m) for b, *_ in NATURAL for m in METRICS}
    pet_bb_margin = [over[("pet_resnet18", m)].fpr_margin for m in BB]
    pet_ii_margin = [over[("pet_resnet18", m)].fpr_margin for m in II]
    cifar_bb_margin = [over[(b, m)].fpr_margin for b, *_ in NATURAL[1:] for m in BB]
    pet_black_over = over[("pet_resnet18", "Deletion (black)")]
    over_main = main_only(OVER)
    max_so_ro = (over_main.pref_so_low - over_main.pref_ro_low).abs().max()
    nat_over = OVER[OVER.base.isin([b for b, *_ in NATURAL])]
    nat_excess_sig = nat_over[nat_over.p_bh_excess_one_sided < 0.05]
    nat_excess_ns = nat_over[nat_over.p_bh_excess_one_sided >= 0.05]
    pet_over = OVER[OVER.base == "pet_resnet18"]
    mp_over_le10 = OVER[(OVER.base == "mnist") & (OVER.shortcut == "patch")].fpr_rc_le10
    pet_ro_bb, pet_ro_ii = ro_bb["pet_resnet18"], ro_ii["pet_resnet18"]
    pet_correct = CORRECT[CORRECT.base == "pet_resnet18"]
    pet_correct_bb = pet_correct[pet_correct.metric.isin(BB)]
    mis_low = {(b, s): korow(b, s, METRICS[0]).misclassified_share_rc_le20 for b, s, *_ in SETTINGS}
    p1_black_x = _one(CROSSED_PRE_A2, comparison=FULL, base="pet_resnet18", shortcut="patch",
                      metric="Deletion (black)")  # the 18 original Pet models, like PET
    assert p1_black_x.models == 18, "P1-P4 use the 18 original Pet models"
    full_x = main_only(CROSSED[CROSSED.comparison == FULL]).merge(
        RESP[RESP.comparison == FULL], on=["comparison", "base", "shortcut", "metric"], suffixes=("_x", ""))
    pref_width_ratio = ((full_x.pref_low_crossed_ci_high - full_x.pref_low_crossed_ci_low)
                        / (full_x.pref_low_ci_high - full_x.pref_low_ci_low))
    pp_black_x = xrow(FULL, "pneumonia_mnist", "patch", "Deletion (black)")
    pp_black_r = row(FULL, "pneumonia_mnist", "patch", "Deletion (black)")
    mp_x = CROSSED[(CROSSED.comparison == FULL) & (CROSSED.base == "mnist") & (CROSSED.shortcut == "patch")]
    sd = main_only(SWDIFF[(SWDIFF.comparison == FULL) & SWDIFF.primary])
    sd_order = {(b, s): i for i, (b, s, *_) in enumerate(SETTINGS)}
    sd = sd.assign(o=[sd_order[(b, s)] for b, s in zip(sd.base, sd.shortcut)]).sort_values(["o", "metric"])
    sd_mp = {m: drow(FULL, "mnist", "patch", m) for m in METRICS}
    sd_nat = sd[sd.base.isin([b for b, *_ in NATURAL])]
    sd_nat_test = sd_nat[~sd_nat.verdict.str.startswith("not testable")]
    sd_nd = sd[sd.verdict == "not distinguishable"]
    sd_later = sd[sd.verdict == "metric later"]
    sd_crossed_lost = sd[(sd.verdict == "metric earlier") & (sd.diff_crossed_ci_low <= 0)]
    conf_rc_sig = int((main_only(CONF_RC).p_bh_rc < 0.05).sum())
    # The per-model table agrees with the summary: margin FPR = share of low-r_c models that say shortcut
    # with a margin (natural images, every metric).
    for (b, m), o in over.items():
        sel = EXCESS[(EXCESS.base == b) & (EXCESS.metric == m) & (EXCESS.reliance_corrected <= 50)]
        assert len(sel) == o.n_below and np.isclose(sel.says_shortcut_margin.mean(), o.fpr_margin), (b, m)
    # Guards for the qualitative statements in the text.
    assert pet_bsw.status == "identified" and pet_bsw.bracketed, "Pet switch: identified and bracketed (A2)"
    pet_check_sw = [swrow("pet_resnet18", "patch", lab) for lab in BEHAVIOUR_LABELS[1:]]
    assert all(r.status == "identified" and not r.bracketed for r in pet_check_sw), "Pet check labels: nb"
    pet_pre_a2 = _one(PRE_A2_SWITCH, base="pet_resnet18", shortcut="patch", label=PRIMARY_LABEL)
    assert pet_pre_a2.status == "identified" and not pet_pre_a2.bracketed, "Pet before A2: not bracketed"
    pet_a2_rc = A2_MODELS[A2_MODELS.base == "pet_resnet18"].reliance_corrected
    pet_below = MODELS[(MODELS.base == "pet_resnet18") & (MODELS.reliance_corrected < pet_bsw.behavioural_switch_rc)
                       ].reliance_corrected.max()
    pet_above = MODELS[(MODELS.base == "pet_resnet18") & (MODELS.reliance_corrected > pet_bsw.behavioural_switch_rc)
                       ].reliance_corrected.min()
    assert np.isclose(pet_bsw.behavioural_switch_rc - pet_below, pet_bsw.distance_below)
    assert np.isclose(pet_above - pet_bsw.behavioural_switch_rc, pet_bsw.distance_above)
    assert pet_below in set(pet_a2_rc), "the model just below the Pet switch is an A2 model"
    assert (pet_a2_rc < pet_bsw.behavioural_switch_rc).sum() == 2 and len(pet_a2_rc) == 3, "text: two below, one above"
    assert all(r.status == "identified" and r.bracketed and r.behavioural_switch_rc > 50 for r in cifar_bsw), \
        "CIFAR-10 switch identified and bracketed above 50 (A2)"
    cifar_pre_a2 = [_one(PRE_A2_SWITCH, base=b, shortcut=s, label=PRIMARY_LABEL) for b, s, *_ in NATURAL[1:]]
    assert all(r.status.startswith("not identified") for r in cifar_pre_a2), "CIFAR-10 before A2: not identified"
    cifar_check_sw = {(b, s, lab): swrow(b, s, lab) for b, s, *_ in NATURAL[1:] for lab in BEHAVIOUR_LABELS[1:]}
    assert all(r.status == "identified" for r in cifar_check_sw.values()), "CIFAR-10 check labels identified"
    cifar_check_nb = [
        " and ".join(f"the {LABEL_NAMES[lab].replace(' (check)', ' check')}" for lab in BEHAVIOUR_LABELS[1:]
                     if not cifar_check_sw[(b, s, lab)].bracketed) + f" on {short}"
        for b, s, _, short in NATURAL[1:]
        if any(not cifar_check_sw[(b, s, lab)].bracketed for lab in BEHAVIOUR_LABELS[1:])]
    mp_bsw = bsw[("mnist", "patch")]
    assert mp_bsw.status == "identified" and mp_bsw.bracketed and 45 <= mp_bsw.behavioural_switch_rc <= 60, \
        "text: the assumed 50% is close to the bracketed MNIST patch switch"
    assert all(bsw[("pneumonia_mnist", s)].status == "identified" and bsw[("pneumonia_mnist", s)].bracketed
               and beh_switch[("pneumonia_mnist", s)] < 45 for s in ("patch", "texture")), "50% too high there"
    assert min(gap_det_nat) > 0, "text: calibration gap on determined images positive on natural images"
    assert max(abs(v - 0.5) for v in pet_ro_bb) < 0.05, "text: Pet black/blur rank the mask like random pixels"
    assert max(pet_ro_ii) < 0.45, "text: inpainting and insertion rank the mask above random, weakly"
    assert min(pet_bb_margin) >= 0.5 and min(cifar_bb_margin) >= 0.9, "text: at least half / nearly all"
    assert set(nat_excess_sig.base) >= {"pet_resnet18"} and len(pet_over[pet_over.p_bh_excess_one_sided < 0.05]) == 4
    assert (mp_over_le10 == 0).all(), "text: no MNIST patch over-detection at r_c <= 10"
    assert (sd_nat_test.verdict == "metric earlier").all() and (sd_nat_test.diff_crossed_ci_low > 0).all(), \
        "text: every testable natural-image cell is 'metric earlier', also crossed"
    assert sd_nat[sd_nat.verdict.str.startswith("not testable")].metric_switch_status.eq(
        "above 0.5 at lowest reliance").all()
    assert (pref_width_ratio >= 1).all(), "text: crossed level intervals are wider"
    assert mp_x.switch_crossed_ci_high.max() < mp_bsw.behavioural_switch_rc
    assert pet_pr_ins.rho_rc < 0 and (pr_ins_mnist.rho_rc > 0).all()
    assert all(robust_full == N_SETTINGS for robust_full in
               [sig_positive(rows([FULL], SETTINGS, [m])) for m in METRICS[:3]])
    assert pt_pairs["Deletion (inpaint)"].delta_rho_rc_ci_low > 0
    assert max(ro_bb["pet_resnet18"]) < min(ro_bb["cifar10"])
    assert pt_black_sr.rho_rc < 0 and pt_black_sp.rho_rc < 0 and mt_sr_ins.rho_rc < 0
    assert all(beh_switch[(b, s)] > max(r.switch_iso_rc for r in rows([FULL], [(b, s)]).itertuples()
                                        if not pd.isna(r.switch_iso_rc)) for b, s, *_ in NATURAL)
    pet_edge = [m for m, r in pet_full.items() if r.switch_status == "identified at lowest-reliance edge"]
    pneu_low = {s: levels("pneumonia_mnist", s).sort_values("reliance").iloc[0] for s in ("patch", "texture")}
    pneu_ties = RESP[RESP.base == "pneumonia_mnist"].tie_rate.max()
    mnist_ties = RESP[RESP.base == "mnist"].tie_rate.max()
    p1_black = pet_row("P1", FULL, "Deletion (black)", "spearman")
    p2_contrast = pet_row("P2", R_OBJECT, "mean(black, blur) - mean(inpaint, insertion)", "preference contrast")
    p3_black_sr = pet_row("P3", S_RANDOM, "Deletion (black)", "spearman")
    p3_black_sp = pet_row("P3", S_PLACEBO, "Deletion (black)", "spearman")
    assert np.isclose(p1_black_x.rho_raw, p1_black.estimate), "P1 black estimate is the raw-reliance rho"
    assert p1_black.ci_low < 0.8, "text: the two-level interval reaches below 0.8"
    kmn = KENDALL.set_index(["dataset", "metric"])
    sanity = [kmn.loc[(d, "Sanity (object px)")] for d in ("decoy_mnist", "decoy_pneumonia")]
    tau_threshold = KENDALL["min_tau_one_sided_p_below_0.05"].iloc[0]
    gt_tau = GT_TAU.groupby(["dataset", "metric"]).tau_with_ground_truth.mean()
    del_tau_mnist = [gt_tau[("decoy_mnist", m)] for m in METRICS]
    del_tau_pneu = [gt_tau[("decoy_pneumonia", m)] for m in METRICS]
    inst = GT_INSTANCE.set_index(["dataset", "metric"]).spearman_with_ground_truth
    del_inst = [inst[(d, m)] for d in ("decoy_mnist", "decoy_pneumonia") for m in METRICS[:3]]
    ins_inst = [inst[(d, "Insertion (blur)")] for d in ("decoy_mnist", "decoy_pneumonia")]
    # ---- amendment A2: P5-P9 exactly as written in docs/PREDICTIONS_pretrained.md
    sd1_g = group(*SD1_KEY)
    assert len(sd1_g) == N_SD1 and (sd1_g.reliance_corrected > 50).sum() >= 2, "SD1 testable (A2 rule)"
    sd1_over = {m: orow(*SD1_KEY, m) for m in METRICS}
    sd1_full = {m: row(FULL, *SD1_KEY, m) for m in METRICS}
    sd1_matched = {m: row(MATCHED, *SD1_KEY, m) for m in METRICS}
    sd1_bh4 = dict(zip(METRICS, bh([sd1_over[m].p_excess_one_sided for m in METRICS])))  # BH over the four
    sd1_bsw = swrow(*SD1_KEY)
    sd1_gap = largest_gap(sd1_g.reliance_corrected, lo=-5, hi=105)
    assert sd1_bsw.status.startswith("not identified") and sd1_gap[0] < sd1_bsw.behavioural_switch_rc < sd1_gap[1]
    assert all(swrow(*SD1_KEY, lab).status.startswith("not identified") for lab in BEHAVIOUR_LABELS)
    assert np.isclose(sd1_gap[1], sd1_g[sd1_g.dose == 0.1].reliance_corrected.min()) and np.isclose(
        sd1_gap[0], sd1_g[sd1_g.dose == 0].reliance_corrected.max()), "text: the gap lies between doses 0 and +0.1"
    p5 = all(sd1_over[m].fpr > 0.5 for m in BB)
    p6 = all(sd1_over[m].fpr < min(sd1_over[b].fpr for b in BB) for m in II)
    sd1_low = sd1_g.reliance_corrected.min()
    sd1_edge = {m: r.switch_status in ("identified at lowest-reliance edge", "above 0.5 at lowest reliance")
                for m, r in sd1_matched.items()}
    assert all(not e or pd.isna(sd1_matched[m].switch_iso_rc) or sd1_matched[m].switch_iso_rc - sd1_low <= 5
               for m, e in sd1_edge.items()), "edge = within 5 r_c points of the lowest-r_c model"
    p7 = sum(sd1_edge.values()) >= 3
    p8a_n = sum(r.rho_rc > 0.8 for r in sd1_full.values())
    p8b_n = sum(sd1_over[m].rho_excess_rc > 0 and sd1_bh4[m] < 0.05 for m in METRICS)
    p8 = p8a_n >= 3 and p8b_n >= 3
    p9_settings = NATURAL  # the three SD2 settings: Pet, CIFAR ResNet-18, CIFAR CNN
    assert {b for b, *_ in p9_settings} == set(A2_SD2.base)
    p9_rows = {}
    for b, s, *_ in p9_settings:
        a_ok = bsw[(b, s)].status == "identified"
        cells = [drow(FULL, b, s, m) for m in METRICS]
        found = ("identified", "identified at lowest-reliance edge")
        p9_rows[(b, s)] = (a_ok, [r for r in cells if r.metric_switch_status in found],
                           [r for r in cells if r.metric_switch_status not in found])
    p9_testable = all(a for a, *_ in p9_rows.values())
    p9 = p9_testable and all(r.diff_ci_low > 0 for _, tested, _ in p9_rows.values() for r in tested)
    verdict = {True: "holds", False: "fails"}
    p_verdicts = {"P5": verdict[p5], "P6": verdict[p6], "P7": verdict[p7], "P8": verdict[p8],
                  "P9": verdict[p9] if p9_testable else "not testable"}
    held = [p for p, v in p_verdicts.items() if v == "holds"]
    failed = [p for p, v in p_verdicts.items() if v == "fails"]
    # the text below states this pattern; rebuild the wording if a verdict changes
    assert held == ["P5", "P6", "P7", "P9"] and failed == ["P8"], p_verdicts
    assert p8a_n == 4 and p8b_n == 1, (p8a_n, p8b_n)
    sd1_pet = {m: row(FULL, "pet_resnet18", "patch", m) for m in METRICS}
    # ---- amendment A3: P10-P13 exactly as written in docs/PREDICTIONS_sd3_sd4.md (submission only)
    a3 = a3_numbers() if SUBMISSION else None

    # ---- front matter
    if not SUBMISSION:
        notice = para(DRAFT_NOTICE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
        notice.runs[0].font.color.rgb = RGBColor(0xC0, 0, 0)
        para("Provisional target journal: Neurocomputing (Elsevier), Q1 in Artificial Intelligence, SJR 1.47 according "
             "to the SCImago widget recorded in docs/audit/00_combined_audit.md \u00a77. The free (subscription, no "
             "article processing charge) route is UNVERIFIED and must be confirmed by the author in a normal browser "
             "before submission. Layout: single-column Elsevier-style draft (A4, Times New Roman 12 pt, 1.5 line "
             "spacing); an elsarticle LaTeX version was not prepared. All numbers are generated from the project's "
             "result files by paper/build_paper.py.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9, spacing=1.0)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(12)
    run = t.add_run(TITLE)
    run.bold, run.font.size = True, Pt(16)
    if SUBMISSION:
        # Elsevier title page: given and family name, affiliation with its full postal address and country (letter
        # in superscript), corresponding author with e-mail
        a = para("Md Junaid Faizan Shaikh", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
        for mark in ("a", ",*"):
            a.add_run(mark).font.superscript = True
        b = para("", align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=0)
        b.add_run("a").font.superscript = True
        b.add_run(" Department of Computer Science and Engineering, Khaja Bandanawaz University, Administrative "
                  "Building, Khaja Bandanawaz University Campus, Rauza-i Buzurg, Kalaburagi, Karnataka 585104, "
                  "India").font.size = Pt(10)
        para("* Corresponding author. E-mail: junaid.faizan.codes@gmail.com  \u00b7  ORCID: 0009-0002-6565-1835",
             align=WD_ALIGN_PARAGRAPH.CENTER, size=10)
    else:
        para("Md Junaid Faizan Shaikh", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
        para("Department of Computer Science and Engineering, Khaja Bandanawaz University, Kalaburagi, India",
             align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=0)
        para("E-mail: [email]", align=WD_ALIGN_PARAGRAPH.CENTER, size=10)

    n_word = {1: "one", 2: "two", 3: "three", 4: "four"}
    pet_ident = [m for m in METRICS if m not in pet_no_switch]
    pet_abs_tail = ""  # submission only; the lowest-reliance-edge note was cut for the A3 sentence (250 words)
    if SUBMISSION and pet_no_switch:
        assert len(pet_no_switch) == 1, "abstract wording covers exactly one metric without a switch"
        pet_abs_tail += f"; {pet_no_switch[0]} prefers the shortcut map from the lowest reliance"
    abstract = (
        S("Deletion and insertion curves decide which explanations count as faithful, "
          "but whether they track a model's reliance is rarely tested. ",
          "Whether deletion and insertion metrics track a model's reliance is rarely tested. ") + "We trained "
        f"{N_MODELS} CNNs in which a signed training dose sets the measured reliance on an injected class-coded "
        "shortcut, on Oxford-IIIT Pet with object masks, CIFAR-10, MNIST and "
        "PneumoniaMNIST, and scored " + S("fixed ", "") + "oracle explanations with four metrics. The metrics respond to "
        "reliance: preference for the shortcut map over the object map rises with chance-corrected reliance "
        f"in {n_full_sig} of {n_full} cells (\u03c1 = {span(full_sig.rho_rc)}). They are not calibrated to it. Among models with chance-corrected reliance "
        + S("at most", "\u2264") + " 50%, "
        f"black and blur deletion prefer the shortcut map for {pct_span(pet_bb_fpr)} of Pet models "
        f"({pct_span(pet_bb_margin)} with a " + S("per-model ", "") + "margin) and, against a centred "
        f"Gaussian object prior, {pct_span(cifar_bb_fpr)} of CIFAR-10 models. " + S("On Pet the metrics switch at ",
                                     f"On Pet {n_word[len(pet_ident)]} metrics switch at ") +
        f"{span(pet_switch, f1)}%{S(' chance-corrected reliance', '')}{S('', pet_abs_tail)}, while the models "
        f"follow {S('the shortcut', 'it')} on half the images at "
        f"{switch_text(pet_bsw, note=S(f'{span(pet_label_sw, f1)}% across behavioural labels', None))}. Against "
        + S("per-image ", "") +
        f"behavioural labels the {S('class-stratified', 'stratified')} AUROC is {span(pooled_nat)} "
        f"on natural images, but within a model only {span(within_nat)}. {S('Their verdicts', 'Verdicts')} depend on "
        "explanation size and form: a size-matched object map moves the MNIST switch points from "
        f"{span(mp_full.switch_iso_rc, f1)}% to {span(mp_matched.switch_iso_rc, f1)}%, and on CIFAR-10 black "
        "deletion prefers a random ordering over a centred object map "
        f"{pct_span([ro[(b, 'Deletion (black)')].pref_low for b, *_ in NATURAL[1:]])} "
        "of the time at low reliance. " + S("A pre-registered test on Pet confirmed three of four "
        "predictions; black deletion failed the form-matched control. A second, on an ImageNet-pretrained "
        "network and gap-filled settings, confirmed four of five.",
        "Pre-registered tests confirmed three of four Pet predictions (black deletion failed a form-matched "
        "control), four of five on an ImageNet-pretrained network with gap-filled settings, and "
        f"{n_word[len(a3['held'])] if a3 else ''} of four on exact ROAD and a watermark, which blur deletion did "
        "not over-detect."))
    assert word_count(abstract) <= 250, (word_count(abstract), abstract)
    p = para("")
    add_runs(p, "**Abstract** " + abstract)
    highlights = [
        f"{N_MODELS} CNNs with measured, chance-corrected shortcut reliance test four metrics",
        S("Black and blur deletion over-detect shortcuts on natural images",
          "Black and blur deletion over-detect patch shortcuts on natural images"),  # P13: not the watermark
        "Per-image behavioural labels replace the assumed 50% switch point",
        "Explanation size and form change which map the metrics prefer",
        S("Pre-registered Pet test: three of four predictions held; black deletion failed",
          f"Pre-registered tests: {3 + len(held) + len(a3['held']) if a3 else ''} of 13 predictions held, "
          "including exact ROAD and a watermark"),
    ]
    assert all(len(h) <= 85 for h in highlights), [len(h) for h in highlights]
    if not SUBMISSION:
        para("**Highlights** (at most 85 characters each, following the usual Elsevier rule; confirm in the "
             "Neurocomputing Guide for Authors)", after=0)
        bullets(highlights)
    else:
        HIGHLIGHTS_OUTPUT.write_text("\n".join(highlights) + "\n", encoding="utf8")
    para("**Keywords** " + " \u00b7 ".join(S(KEYWORDS_DRAFT, KEYWORDS)))

    # ---- 1 Introduction
    heading("1. Introduction")
    para("Attribution methods such as Saliency " + cite("simonyan") + ", SmoothGrad " + cite("smilkov")
         + ", Integrated Gradients " + cite("sundararajan") + ", Grad-CAM " + cite("selvaraju") + " and Occlusion "
         + cite("zeiler") + " explain an image classifier's prediction with a heatmap. Because the methods "
         "disagree, they are compared with evaluation metrics, most often perturbation metrics: deletion "
         + cite("samek") + " removes the pixels a map ranks first and measures how fast the model's confidence "
         "falls, and insertion " + cite("petsiuk") + " restores them into a blurred image. These metrics decide "
         "which methods are trusted, yet they are rarely validated, because the true explanation of a trained "
         "network is normally unknown. Metrics disagree with one another " + cite("tomsett", "metaquantus", "latec")
         + ", their results do not transfer across datasets " + cite("gevaert") + ", and they depend on how "
         "removed pixels are filled " + cite("rong", "blucher", "baseline") + ".")
    para("One of the main uses of explanations is to find shortcut learning, where a model uses a cue that is "
         "unrelated to the task, such as a marker on a radiograph " + cite("ross", "lapuschkin", "degrave", "sun")
         + ". A metric that is fit for this use should change its verdict when the model's reasoning moves from "
         "the object to a shortcut. Earlier work controlled what a model relies on in order to test attribution "
         "methods " + cite("bam", "smerf", "bastings") + ", or measured how strongly a model uses a shortcut "
         + cite("oscar") + ". In our searches we did not find a study that varies a model's measured reliance in "
         "controlled steps to test the evaluation metrics themselves. We ask:")
    para("_When a model's reliance on a shortcut increases in controlled steps, do faithfulness metrics shift "
         "their preference from object explanations to shortcut explanations, and at what reliance?_",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para("Our answer is one thesis: **the metrics respond to reliance but are not calibrated to it, and their "
         "verdicts depend on the size and form of the explanations being compared.** A training dose sets each "
         "model's reliance, reliance is measured on held-out conflicting images and corrected for chance, and "
         "fixed oracle maps are scored, so that the test does not depend on any attribution method. We lead "
         "with the natural-image settings, Oxford-IIIT Pet with real segmentation masks and CIFAR-10 with two "
         "networks, and use MNIST and PneumoniaMNIST to study the mechanisms. The contributions are:")
    bullets([
        f"**A dose-response protocol for testing faithfulness metrics** with {N_MODELS} models, chance-corrected "
        "reliance, per-image behavioural labels from forward passes, and a two-level bootstrap over seeds and "
        "images.",
        f"**Evidence of responsiveness without calibration.** Preference rises with reliance in {n_full_sig} of "
        f"{n_full} cells, but black and blur deletion over-detect the shortcut on natural images "
        f"(over-detection rate {span(nat_bb_fpr)}, {span(pet_bb_margin + cifar_bb_margin)} with a per-model "
        "margin), in every testable natural-image cell the metrics switch earlier than the models follow the "
        "shortcut on half of the images (paired bootstrap), and within a single model the metrics "
        "separate followed from non-followed images only weakly.",
        "**Size and form dependence.** Matching the object map's size to the shortcut moves the switch points "
        "towards 0%, and on CIFAR-10 black and blur deletion prefer a random pixel ordering over a centred "
        "object map. Real masks on Pet reduce this form bias but do not remove the over-detection.",
        "**A pre-registered test and a pre-registered amendment.** The Pet predictions P1\u2013P4 were committed "
        f"before training; {N_A1} models that fill reliance gaps were added under a dated amendment (A1), and a "
        f"second one (A2) added {N_A2_SD2} natural-image models in the remaining reliance gaps and {N_SD1} "
        "ImageNet-pretrained Pet models with predictions P5\u2013P9. Every model set is reported.",
    ])
    para("Section 2 positions the work, Sections 3 and 4 describe the protocol and setup, Section 5 reports the "
         "results (natural images first), and Sections 6\u20138 discuss them, state the limitations and conclude. "
         + S("Appendix A holds a supporting study of the model-randomisation sanity check, Appendix B the legacy "
             "analysis of earlier drafts, and Appendix C the check on the original model set.",
             "Appendix A holds a supporting study of the model-randomisation sanity check. The Supplementary "
             "Material holds the full pre-registered outcome table, the size-matched responsiveness table, the "
             "paired metric comparisons, the legacy analysis and the check on the original model set."))

    # ---- 2 Related work
    heading("2. Related work")
    para(S(f"The literature audit behind this section searched {SEARCH_AUDIT}; ",
           "The literature search behind this section covered 2019 to October 2026; ")
         + f"every cited page was re-opened on "
         f"{SEARCH_RECHECK}, when a further search found no study of graded reliance as ground truth for "
         f"faithfulness metrics. Table {COUNTERS['table'] + 1} summarises the positioning.")
    heading("2.1. Controlled reliance and ground-truth benchmarks", 2)
    para("Decoy MNIST " + cite("ross") + " adds a class-coded patch so that the model's reason is known. Zhou et "
         "al. " + cite("zhou") + " induce ground-truth features, Lin et al. " + cite("lin") + " and BackX "
         + cite("backx") + " use backdoor triggers, Adebayo et al. " + cite("adebayo2022") + " and Sun et al. "
         + cite("sun") + " ask whether explanations reveal spurious correlations, and Bastings et al. "
         + cite("bastings") + " give a shortcut-injection protocol for salience methods on text. CLEVR-XAI "
         + cite("arras") + ", FunnyBirds " + cite("funnybirds") + ", XAI-Units " + cite("xaiunits") + ", "
         "AttributionLab " + cite("attributionlab") + " and a recent synthetic intervention-based framework "
         + cite("miro2026") + " build data or models whose important inputs are known by design. Closest to our "
         "design, BAM trains models with known relative feature importance " + cite("bam") + ", and SMERF "
         "controls the complexity of the model's reasoning " + cite("smerf") + "; OSCAR derives shortcut scores "
         "that are sensitive to the strength of the shortcut association " + cite("oscar") + ". All of these "
         "evaluate attribution methods or detect shortcuts. We evaluate the metrics, against reliance that is "
         "varied in steps and measured. BAM reports that some attribution methods give false-positive "
         "explanations " + cite("bam") + ", and Adebayo et al. " + cite("adebayo2022") + " that attributions can "
         "indicate a spurious signal the model does not rely on. These are false positives at the level of the "
         "method. We show the counterpart at the level of the evaluation metric: scored with oracle maps, so that "
         "no attribution method is involved, the metrics over-detect a shortcut the model barely uses.")
    heading("2.2. Meta-evaluation of faithfulness metrics", 2)
    para("Tomsett et al. " + cite("tomsett") + " found saliency metrics statistically unreliable. MetaQuantus "
         + cite("metaquantus") + " rates metrics by their resilience to noise and reactivity to randomness "
         "without ground truth, and Quantus " + cite("quantus") + " collects many metrics in one toolkit. LATEC "
         + cite("latec") + " evaluates 17 methods with 20 metrics and finds conflicting rankings; M4 "
         + cite("m4") + " benchmarks faithfulness across modalities and networks. Gevaert et al. " + cite("gevaert")
         + " report that metrics measure different properties and that results on one dataset need not hold on "
         "another. Mir\u00f3-Nicolau et al. " + cite("fidelity") + " check fidelity metrics against a decision tree "
         "on 52,000 synthetic 128\u00d7128 images, and Hesse et al. " + cite("hesse2024") + " propose a protocol "
         "that avoids the out-of-domain problem of incremental deletion. Domeniconi et al. " + cite("refinecam")
         + " compare CAM evaluation metrics on a synthetic dataset with ground-truth attributions and propose a "
         "composite metric. The model-randomisation sanity check "
         + cite("adebayo2018") + " has been criticised " + cite("binder") + " and revised "
         + cite("sanityrevisited", "freshlook") + ". None of these varies the model's reliance and reads the "
         "metrics' response to it.")
    heading("2.3. Known confounds of perturbation metrics", 2)
    para("ROAD " + cite("rong") + " shows that pixel-perturbation results are driven by information leaking "
         "through the shape of the removed region; ROAR " + cite("roar") + " retrains after removal, and a later "
         "analysis traces ROAR's and ROAD's rankings to a bias towards spatially blurry masks "
         + cite("roarpitfalls") + " (a workshop paper). AIM " + cite("aim") + " shows that the masking operator "
         "confounds removal-based scores: zero masking creates out-of-distribution inputs and interpolation-based "
         "masking leaves predictive information in place " + cite("rong", "aim") + ". Bl\u00fccher et al. "
         + cite("blucher") + " show that the occlusion "
         "strategy decides pixel-flipping rankings, and \u201cBack to the Baseline\u201d " + cite("baseline")
         + " that the replacement baseline favours some methods. \u201cFrom Flexibility to Manipulation\u201d "
         + cite("flexibility") + " shows that such free evaluation choices can be exploited to change an evaluation's outcome, which is "
         "why we report several fills side by side and compare oracle maps in pairs within a fill. Gomez et al. " + cite("gomez") + " note that "
         "deletion and insertion use only the ranking of the map and present out-of-distribution images. SaCo "
         + cite("saco") + " reports that common metrics struggle to separate methods from random attribution, "
         "and Bilodeau et al. " + cite("bilodeau") + " argue that a concrete end task with repeated model "
         "evaluations is the proper reference. Section 6 links each of our findings to one of these mechanisms.")
    heading("2.4. Shortcuts in practice", 2)
    para("Clever-Hans behaviour has been found in standard vision models " + cite("lapuschkin") + ", and "
         "radiograph classifiers have been shown to use acquisition shortcuts " + cite("degrave") + ". Which "
         "feature a network uses depends on its predictivity and on its availability " + cite("hermann")
         + "; our dose varies predictivity only. Architectures differ in how readily they pick up positional "
         "and intensity shortcuts " + cite("suhail") + ".")
    table_positioning()

    # ---- 3 Method
    heading("3. Method")
    heading("3.1. Shortcuts", 2)
    para("A **patch** shortcut is a white square (3\u00d73 pixels on the 28\u00d728 and 32\u00d732 images, 12\u00d712 "
         "on the 128\u00d7128 Pet images) whose position encodes the class: along the top and bottom borders "
         "for ten classes, and in the two top corners for two classes, like markers burned into radiographs. A "
         "**texture** shortcut gives each class a fixed random set of about 10% of the pixels, brightened by "
         "0.3 (clipped at 1); it is used on the two 28\u00d728 datasets.")
    heading("3.2. Controlling reliance with a signed dose", 2)
    para("A signed dose _d_ combines two mechanisms. For _d_ > 0, a share _d_ of training images is replaced by "
         "an image of another class while the label and the shortcut are kept, so the image becomes less "
         "reliable. For _d_ < 0, the shortcut of a share |_d_| of training images points to a randomly chosen "
         "wrong class, so the shortcut becomes less reliable. On MNIST and PneumoniaMNIST |_d_| \u2264 0.4. "
         f"CIFAR-10 doses reach {f1(min(DOSES['cifar10']))}, and the Pet doses include {f1(min(DOSES['pet_resnet18']))}, "
         "where half of the training shortcuts point to the wrong class and the shortcut carries no label "
         f"information. The doses are {dose_list('mnist')} (MNIST); {dose_list('pneumonia_mnist')} "
         f"(PneumoniaMNIST); {dose_list('cifar10')} (CIFAR-10); and {dose_list('pet_resnet18')} (Pet).")
    heading("3.3. Measuring reliance and correcting it for chance", 2)
    para("Every test image receives the shortcut of a different, randomly chosen class, and raw reliance _R_ is "
         "the percentage of these conflicting images on which the prediction follows the shortcut. _R_ has a "
         "floor: a model that ignores the shortcut still lands on the shortcut's class whenever it errs, always "
         "on a binary task and, if errors spread evenly, with probability 1/(_C_ \u2212 1) on a task with _C_ "
         "classes. We therefore use the chance floor _f_ = (100 \u2212 accuracy without the shortcut)/(_C_ \u2212 1) "
         f"and the chance-corrected reliance {RC} = 100 (_R_ \u2212 _f_)/(100 \u2212 _f_), in percent and not "
         "clipped. The correction is an approximation, because images without the shortcut are out of "
         f"distribution for a model trained with it. All primary analyses use {RC}; raw reliance is reported "
         "where labelled.")
    heading("3.4. Oracle explanations", 2)
    para("Five fixed maps are scored on the same 100 aligned test images per model (shortcut and image agree), "
         "explaining the true label:")
    bullets([
        "**Shortcut map**: 1 on the shortcut pixels, 0 elsewhere.",
        "**Full object map**: on Pet, the real segmentation mask (pet and border band) without the shortcut "
        "pixels. On the " + S("grayscale", "greyscale") + " datasets, the image's own intensities without the shortcut pixels; on "
        "PneumoniaMNIST, where 97.5% of the pixels exceed 0.1, this is an intensity ranking of the whole "
        "radiograph rather than an object segmentation. On CIFAR-10, a centred Gaussian prior (\u03c3 = image "
        "size/4), because the images have no masks. Pet is therefore the primary natural-image evidence.",
        "**Size-matched object map**: _k_ object pixels, where _k_ is the number of shortcut pixels. On the "
        "intensity-based maps they are the first _k_ pixels of a stable descending sort; on MNIST many saturated "
        "pixels tie, so the topmost saturated pixels in row-major order are taken. On Pet they are the _k_ mask "
        "pixels nearest the mask centroid.",
        "**Placebo map**: the shortcut's shape at the position (or texture pattern) of a wrong class.",
        "**Random map**: an independent uniform score per pixel, i.e. a random pixel ordering.",
    ])
    heading("3.5. Metrics under test", 2)
    para("Each metric runs in 20 steps; step _i_ changes the top _i_/20 of the pixels in the map's ranking (ties "
         "broken at random), and the score is the area under the model's probability for the explained class. "
         "**Deletion** replaces pixels (lower area is better) with black, with a blurred copy of the image "
         "(Gaussian kernel 25 and \u03c3 = 4 at 28 and 32 px, scaled to 97 and 16 on Pet), or by inpainting: the "
         "per-channel mean of the remaining pixels refined by repeated averaging of four neighbours (100 "
         "iterations, 1600 on Pet) plus Gaussian noise (standard deviation 0.05). The inpainting is inspired by "
         "ROAD " + cite("rong") + " but is not the ROAD procedure. **Insertion** starts from the blurred image "
         "and restores pixels (higher area is better). Two properties matter for interpretation: blur fill "
         "leaves about 20% (28 and 32 px) to 25% (Pet) of the patch's intensity in place, so blur deletion "
         "does not fully remove a patch shortcut; and at the last inpainting step every pixel is masked, so the "
         "curve ends on noise around black.")
    heading("3.6. Preference and per-image behavioural labels", 2)
    para("For maps A and B, a metric's **preference** for A is the share of images on which it rates A better; "
         "area differences of at most 10\u207b\u2074 count as ties (one half). The per-image **AUC gap** is "
         "the signed difference of the two areas, positive when A is rated better. For the shortcut versus the "
         "object map an ideal metric gives a preference near 0 at low reliance and near 1 at high reliance.")
    para("Preference is scored on aligned images, while reliance is measured on conflicting images, so the "
         "assumption that a metric should switch at 50% reliance is not derived. We therefore label each "
         "scored image from forward passes only. The primary label" + S(" (after the audit's reviewer 05)", "") + " is 1 if the "
         "model predicts the shortcut's class when the same image's shortcut is moved to a wrong class's "
         "position and does not predict that class when the shortcut is removed; it is 0 if the moved shortcut "
         "does not win; images that the model already assigns to the shortcut's class without any shortcut are "
         "left undetermined, which removes the same error floor as the chance correction. A check label "
         + S("(reviewer 03) ", "") + "marks an image as shortcut-dominant when removing the shortcut lowers the true-class "
         "probability more than replacing the image by one of another class that carries the same shortcut. "
         "Each metric's AUC gap is then scored against the label by an AUROC computed within each true class "
         "and averaged (class-stratified), because on the binary tasks the label and the gap both depend on "
         "the class. The **behavioural switch** is the reliance at which the models' follow rate crosses 0.5; "
         "where the models identify it (Section 3.7), it replaces the assumed 50% ideal.")
    heading("3.7. Statistics", 2)
    bullets([
        f"**Responsiveness**: Spearman \u03c1 between {RC} and preference over the models of a setting.",
        f"**Decisions**: a model counts as relying on the shortcut when {RC} > 50, and the metric \u201csays "
        "shortcut\u201d when its preference exceeds 0.5. We report the **over-detection rate** (the false-positive "
        f"rate among models with {RC} \u2264 50) and balanced accuracy. The over-detection rate is given with a "
        "Clopper\u2013Pearson interval, with a margin (a model counts only if the lower end of its own 95% "
        f"image-bootstrap interval of preference exceeds 0.5), and split into {RC} \u2264 10 and 10\u201350; the "
        f"true-positive rate is given with the number of models above {RC} = 50.",
        "**Excess preference**: per model, the shortcut-vs-object preference minus the random-vs-object "
        "preference, i.e. what the shortcut map gains over a random ordering compared against the same object "
        f"map; its Spearman \u03c1 with {RC} is tested one-sided.",
        f"**Switch point**: the first {RC} at which an increasing isotonic fit of preference reaches 0.5; a "
        "four-parameter logistic with free asymptotes is the sensitivity check. A switch is \u201cidentified\u201d "
        "only if a model lies within \u00b110 points of it, and \u201cbracketed\u201d if such models lie on both "
        "sides; it is reported only for the comparisons whose ideal curve crosses 0.5 (shortcut vs full or "
        "size-matched object). The same rule is applied to the behavioural switch, and the difference "
        "behavioural \u2212 metric switch is bootstrapped on the same resamples; a metric is called earlier "
        "than behaviour only if that 95% interval excludes 0.",
        "**Uncertainty**: a two-level bootstrap with 2000 replicates that resamples models within each dose and "
        "the 100 images within each model " + cite("saravanan") + ", following the advice to account for "
        "both sources of variance in machine-learning comparisons " + cite("bouthillier", "agarwal") + ". The "
        "same resamples are reused for every metric, so metric-versus-metric differences are paired. Because "
        "every model of a setting is scored on the same 100 images, images are a crossed factor; for "
        "preference levels and switch points we therefore also use a **crossed bootstrap** that resamples "
        "models within dose and draws one set of images per replicate for all models, which gives wider "
        "intervals. Intervals for \u03c1 are **BCa** (bias-corrected and accelerated, with a leave-one-model-out "
        "jackknife) computed on the crossed replicates after adding a tiny jitter that breaks the ties between "
        "duplicated models. Significance of \u03c1 always comes from the model-level Spearman test; the "
        "intervals describe uncertainty. The main responsiveness tables give crossed and BCa intervals; the "
        "other tables give two-level intervals.",
        "**Multiplicity**: Benjamini\u2013Hochberg within each family (each comparison, the methods, the metric "
        "pairs, each behavioural label), and Holm over the union of the headline tests; the families of the "
        "analysis of all models also contain the SD1 setting" + S(". ", " and the SD4 setting (amendment A3). The "
        "exact-ROAD analysis (Section 5.11) has its own families: Benjamini\u2013Hochberg within each comparison "
        "over its nine settings and six metrics, and over all its rows for the excess preference. The "
        "pre-registered verdicts on P10\u2013P13 use their own point-value and interval rules, which include no "
        "multiplicity adjustment. ") + "Correlation p-values "
        "use Spearman's t approximation; p-values below 0.001 are printed as p < 0.001.",
        "**Effect size**: the median per-image AUC gap at the lowest and the highest dose.",
    ])
    heading("3.8. Explanation methods", 2)
    para("Five methods are run on every model with Captum " + cite("captum") + ": Saliency (absolute gradient), "
         "SmoothGrad (25 samples, noise 0.15), Integrated Gradients (50 steps, black baseline), Grad-CAM on the "
         "last convolutional feature maps (6\u00d76 for the 28\u00d728 CNN, 16\u00d716 for the CIFAR CNN, 4\u00d74 "
         "for the CIFAR ResNet-18, 16\u00d716 for the Pet ResNet-18 and 4\u00d74 for the pretrained Pet ResNet-18 "
         "of SD1, upsampled bilinearly), and Occlusion "
         "(window and stride 4 and 2 at 28 and 32 px, 16 and 8 on Pet). For each method we compute the share of "
         "positive attribution on the shortcut pixels (relevance mass accuracy " + cite("arras") + ").")

    if SUBMISSION:
        heading("3.9. Pre-registered extensions: exact ROAD and a watermark shortcut", 2)
        para("Two further tests were pre-registered in amendment A3 (Zenodo version 3, 10.5281/zenodo.24416017) "
             "before any model or score for them existed. **SD3** scores the same oracle and control maps with the "
             "Noisy Linear Imputation operator of ROAD " + cite("rong") + ", in two removal orders: most relevant "
             "first (MoRF, a lower area is better) and least relevant first (LeRF, a higher area is better). The "
             "removed shares of the pixels are 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7 and 0.9, and the scoring images are "
             "the first 100 (28 and 32 px small CNNs) or 50 (CIFAR-10 ResNet-18, Pet, SD1 and SD4) of the existing "
             "scoring images. "
             "Each removed pixel is imputed as the weighted mean of its eight neighbours (weight 1/6 for the edge "
             "and 1/12 for the diagonal neighbours), solved jointly for all removed pixels of an image, step and "
             "channel, and Gaussian noise with standard deviation 0.01 is added to the imputed pixels; the "
             "operator is ported from the reference implementation of Rong et al. "
             "(https://github.com/tleemann/road_evaluation " + cite("roadcode") + ", MIT licence, whose copyright "
             "notice is kept in the code). The score is the area under the probability-of-label curve over the removed shares divided "
             "by 0.9; ties in the binary maps are broken as for the four existing metrics, and the noise is "
             "seeded per model and image, so the scores are deterministic. Every one of the "
             f"{len(ROAD_MODELS)} pooled models was scored (the pilots were not). The pre-registered timing gate, "
             "which would have cut the Pet-sized settings to 25 images if one Pet model took more than 15 minutes, "
             f"was not reached: ROAD on 50 images took {span(a3['timing'], f2)} minutes per Pet model on a Kaggle "
             "T4 GPU, so those settings kept 50 images. When ROAD is compared with an existing metric, the existing "
             "metric is recomputed from its stored per-image scores on the same images.")
        sd4_pil = a3["pilots"]
        para("**SD4** replaces the patch of SD1 by a semi-transparent watermark in one image corner, whose stroke "
             "arrangement codes the class, on the ImageNet-pretrained Pet ResNet-18, with 18 confirmatory models "
             "(six doses, three seeds). The doses follow a pilot rule fixed in A3 before the pilot. "
             "Both marks are white with opacity 0.4 and lie in the same 24\u00d712-pixel box in the bottom-right "
             "corner: two vertical 4-pixel bars for cats and two horizontal 4-pixel bars for dogs, 96 stroke pixels "
             "each. The strokes of the two marks occupy disjoint pixels, so the class is coded by the stroke "
             "arrangement and also by which pixels in the corner are brightened: the shortcut is partly "
             "position-coded within the corner, not purely content-coded. Compared with SD1, SD4 changes the "
             "shortcut's shape, contrast and layout within one shared corner; the network, training and metrics "
             "are those of SD1. The shortcut oracle is the stroke mask of the shown mark, the placebo the other "
             "class's stroke mask in the same box, the object oracle the pet mask without the shown strokes, and "
             "the size-matched map the 96 object pixels nearest the mask centroid. Two seed-7 pilots at doses 0 and +0.25 "
             f"reached {RC} {f2(sd4_pil.loc[0.0].reliance_corrected)}% and "
             f"{f2(sd4_pil.loc[0.25].reliance_corrected)}%. The dose-0 pilot was below 60%, so the A3 rule set the "
             "doses to " + ", ".join(("+" if d > 0 else "") + f"{d:g}" for d in SD4_DOSES).replace("-", MINUS)
             + "; the stopping rule (+0.25 pilot below 20%) was not triggered. The completion note recording this "
             "was archived on Zenodo before any confirmatory SD4 model was trained (Section 4). The pilots are "
             "excluded from every analysis.")

    # ---- 4 Setup
    heading("4. Experimental setup")
    para("**Datasets.** Oxford-IIIT Pet " + cite("pets") + " as a cat-versus-dog task at 128\u00d7128 with the "
         "dataset's trimap masks; CIFAR-10" + (" " + cite("cifar") if SUBMISSION else "") + " (10 classes, 32\u00d732 "
         "colour); MNIST" + (" " + cite("lecun") if SUBMISSION else "") + " (10 classes, 28\u00d728); and "
         "PneumoniaMNIST " + cite("medmnist") + " (normal vs pneumonia, 28\u00d728, derived from "
         + cite("kermany") + ").")
    if SUBMISSION:
        para("**Software.** The models are implemented in PyTorch " + cite("pytorch") + " with torchvision "
             + cite("torchvision") + " (versions 2.14.0 and 0.29.0), and the explanation methods are run with "
             "Captum " + cite("captum") + " (version 0.9.0). The code of this study is archived on Zenodo "
             + cite("code") + ".")
    para("**Networks and training.** A small CNN with four convolutional layers on the 28\u00d728 data; on "
         "CIFAR-10 a CNN with batch normalisation (0.77M parameters) and a ResNet-18 " + cite("he") + " (11.2M "
         "parameters) with a 3\u00d73 stride-1 first convolution and no initial max-pooling; on Pet the same "
         "ResNet-18, also with this CIFAR-style stem. Apart from the separate pretrained SD1 network (amendment A2, "
         "below), all networks are trained from scratch with Adam (learning "
         f"rate 10\u207b\u00b3, batch size 128) for {EPOCHS['mnist']} (MNIST), {EPOCHS['pneumonia_mnist']} "
         f"(PneumoniaMNIST), {EPOCHS['cifar10']} (CIFAR CNN), {EPOCHS['cifar10_resnet18']} (CIFAR ResNet-18) and "
         f"{EPOCHS['pet_resnet18']} epochs (Pet), keeping the best validation checkpoint, without augmentation. "
         f"The ResNet-18 models were trained on a free Kaggle T4 GPU (about {MINUTES['cifar10_resnet18']:.0f} "
         f"minutes per CIFAR model and {MINUTES['pet_resnet18']:.0f} per Pet model); all others on a laptop CPU.")
    counts = "; ".join(f"{BASE_NAMES[b]} {N_BASE[b]}" for b in ["pet_resnet18", "cifar10_resnet18", "cifar10",
                                                                 "mnist", "pneumonia_mnist"])
    counts += "; " + S(f"{BASE_NAMES[SD1[0]]} {N_BASE[SD1[0]]}",
                       f"{BASE_NAMES[SD1[0]]} {N_SD1} with the patch (SD1) and {N_SD4} with the watermark (SD4)")
    para(f"**Models.** {N_MODELS} models in total ({counts}). Seeds: "
         + "; ".join(f"{BASE_NAMES[b]} {', '.join(str(s) for s in SEEDS[b])}" for b in
                     ["pet_resnet18", "cifar10_resnet18", "cifar10", "mnist", "pneumonia_mnist",
                      "pet_resnet18_pretrained"])
         + f". {N_ORIGINAL} models existed before amendment A1 (" + S("docs/PREDICTIONS_amendments.md, commit 8738b32, ", "") + ""
         f"committed before any of its models was trained); A1 added {N_A1} MNIST and PneumoniaMNIST models at "
         "new doses near the steep part of the dose-response curve, with six seeds at the two steepest MNIST "
         "doses, to fill reliance gaps. A1 made no predictions about the metrics, and both model sets are "
         "reported (" + S("Appendix C", tref("sets", 0)) + ").")
    # pilots.csv lists only never-pooled pilots: the SD1 pilots (A2) and the SD4 pilots (A3); every A2 model,
    # the CIFAR-10 ResNet-18 pilot included, is scored and pooled.
    assert set(PILOTS.model_set) <= {"A2_pilot", "A3_pilot"}, set(PILOTS.model_set)
    pilot_text = "; ".join(
        f"{BASE_NAMES.get(p.base, p.base)} dose {f2(p.dose)}, seed {int(p.seed)} ({RC} {f1(p.reliance_corrected)}%, "
        "an SD1 pilot)" for p in PILOTS[PILOTS.model_set == "A2_pilot"].itertuples())
    sd4_pilots = PILOTS[PILOTS.model_set == "A3_pilot"].sort_values("dose")
    def a2_seeds(base):
        return ", ".join(str(s) for s in sorted(A2_MODELS[A2_MODELS.base == base].seed.unique(),
                                                key=[42, 1, 2].index))

    sd1_doses = ", ".join(("+" if d > 0 else "") + f"{d:g}" for d in sorted(group(*SD1_KEY).dose.unique())
                          ).replace("-", MINUS)
    para("**Amendment A2.** A second dated amendment (" + S("docs/PREDICTIONS_amendments.md, ", "") + "A2, with predictions "
         "P5\u2013P9" + S(" in docs/PREDICTIONS_pretrained.md; ", "; ") + "Zenodo record 10.5281/zenodo.24285532) adds natural-image "
         "doses inside the reliance gaps (SD2) and an ImageNet-pretrained ResNet-18 on Pet (SD1). Pilots fixed "
         "the final doses under the amendment's rules (completion notes A2-C1 and A2-C2, 2026-11-17, Zenodo "
         "10.5281/zenodo.24333540). SD2 adds "
         f"{N_A2_SD2} models: Pet from scratch at dose {f2(A2_PET_DOSES[0])} ({N_A2_PET} models), and CIFAR-10 at "
         f"doses {', '.join(f2(d) for d in A2_CIFAR_DOSES)} with ResNet-18 ({A2_COUNTS['cifar10_resnet18']} "
         f"models, seeds {a2_seeds('cifar10_resnet18')}) "
         f"and the small CNN ({A2_COUNTS['cifar10']} models, seeds {a2_seeds('cifar10')}). "
         "The CIFAR-10 ResNet-18 pilot (dose \u22120.05, seed 42) is one of these models, as A2 specifies. The SD2 "
         "models are pooled with the existing models of their setting, and results without any A2 model are "
         + S("written to results/dose_response/pre_a2_models/ (", "analysed separately (") + f"{N_ORIGINAL + N_A1} models). SD1 has {N_SD1} "
         f"confirmatory models (doses {sd1_doses}, seeds {a2_seeds(SD1[0])}), the ImageNet-pretrained ResNet-18 with the "
         "standard ImageNet stem, fine-tuned for "
         f"{EPOCHS[SD1[0]]} epochs with Adam (learning rate 10\u207b\u2074, batch size 128) on the same Pet data, "
         "shortcut, oracle maps and metrics. SD1 is a separate setting: it is never pooled with the Pet "
         "ResNet-18 from scratch, and it is reported in Section 5.10 only. The pilots are listed separately and "
         f"are not pooled (results/dose_response/pilots.csv): {pilot_text}"
         + S("", f", and the {len(sd4_pilots)} SD4 pilots of amendment A3 (Section 3.9)")
         + ". The SD1 Grad-CAM maps are coarser "
         "than those of the Pet ResNet-18 from scratch: "
         "with the standard ImageNet stem the last residual stage is 4\u00d74 at 128\u00d7128, against 16\u00d716 "
         "with the CIFAR-style stem, so method-level comparisons between the two mix pretraining with map "
         "resolution. The oracle-map metrics do not use Grad-CAM and are not affected.")
    para("**Evaluation.** Every model is scored on 100 aligned test images, the same images for every model of a "
         "setting. Reliance and accuracy use the full test set. Re-evaluating a cached model reproduces its "
         "stored scores exactly.")
    if SUBMISSION:
        para("**Pre-registration.** The Pet predictions P1\u2013P4 were committed to the project repository before any "
             "Pet model was trained. Amendment A1 (gap-filling MNIST and PneumoniaMNIST models) was committed "
             "before its models were trained. Amendment A2 and the predictions P5\u2013P9 were deposited on Zenodo "
             "as version 1 of the record (DOI 10.5281/zenodo.24285532, 2026-11-16) before any confirmatory SD1 or "
             "SD2 model was trained; the two SD1 pilot models had been trained shortly before the upload, and only "
             "their reliance and accuracy were read, a deviation recorded in the amendment. The completion notes with "
             "the final SD1 and SD2 doses are version 2 (10.5281/zenodo.24333540, 2026-11-18). Amendment A3 and the "
             "predictions P10\u2013P13 for exact ROAD and the watermark shortcut are version 3 "
             "(10.5281/zenodo.24416017, 2026-11-21), uploaded before any SD3 or SD4 model or score was produced. The "
             "completion note A3-C1 with the SD4 pilot outcome, the final SD4 doses and the ROAD timing is version "
             "4 (10.5281/zenodo.24495658, 2026-11-24), uploaded before any confirmatory SD4 model was trained. The "
             "concept DOI of the record is 10.5281/zenodo.24285531.")

    # ---- 5 Results
    heading("5. Results")
    heading("5.1. The dose controls reliance", 2)
    t_rel = COUNTERS["table"] + 1
    pet_g = group("pet_resnet18", "patch")
    pet_min = pet_g.loc[pet_g.reliance.idxmin()]
    para(f"Table {t_rel} shows that the dose moves reliance across most of the range in every setting. On Pet "
         f"the lowest raw reliance of a model is {f1(pet_min.reliance)}%, but its chance floor is about as large, "
         f"so its {RC} is {f1(pet_min.reliance_corrected)}% (the lowest {RC} of any Pet model is "
         f"{f1(pet_g.reliance_corrected.min())}%). The same "
         f"holds on PneumoniaMNIST: the lowest per-dose mean raw reliance is {f1(pneu_low['patch'].reliance)}% "
         f"(patch) and {f1(pneu_low['texture'].reliance)}% (texture), which is {RC} = "
         f"{f1(pneu_low['patch'].reliance_corrected)}% and {f1(pneu_low['texture'].reliance_corrected)}%. " + S("Earlier "
         "drafts reported \u201cabout 15%, not 0\u201d here; that 15% is the error floor. ",
         "A raw reliance of about 15% at the lowest dose is the error floor, not use of the shortcut. ") + "Near _d_ = 0 reliance "
         "varies strongly between seeds, so every analysis uses measured reliance, not the dose.")
    table_reliance()

    heading("5.2. Natural images: the metrics respond but over-detect", 2)
    f_full = COUNTERS["fig"] + 1
    t_full = COUNTERS["table"] + 1
    para(f"Fig. {f_full} and Table {t_full} give the main comparison, the shortcut map against the full object "
         f"map. Over all settings, preference rises with {RC} in {n_full_sig} of {n_full} cells (\u03c1 = "
         f"{span(S(full.rho_rc, full_sig.rho_rc))}); the exception is "
         + "; ".join(f"{r.metric} on {LABEL[(r.base, r.shortcut)]} (\u03c1 = {f2(r.rho_rc)}, p_bh = "
                     f"{format_p(r.p_bh)})" for r in full_weak.itertuples())
         + f". On the natural images \u03c1 = {span(full_nat.rho_rc)}.")
    para("Responsiveness is not calibration. Among the models with "
         f"{RC} \u2264 50, black and blur deletion still prefer the shortcut map for {pct_span(pet_bb_fpr)} of the "
         f"Pet models and {pct_span(cifar_bb_fpr)} of the CIFAR-10 models; inpainting deletion and insertion do "
         f"so for {pct_span(nat_ii_fpr)}. Across all settings the true-positive rate is 1.00 in {n_tpr1} of "
         f"{n_full} cells and the over-detection rate is above 0 in {n_fpr_pos} of {n_full}: the metrics rarely miss a "
         "shortcut the model uses, but they often report one the model barely uses. On the natural images the "
         f"true-positive rate rests on few models (models with {RC} > 50: "
         + ", ".join(f"{int(over[(b, METRICS[0])].n_above)} on {short}" for b, _, _, short in NATURAL)
         + "). On Pet, the isotonic "
         f"switch lies at {span(pet_switch, f1)}% {RC} for "
         + ", ".join(m for m in METRICS if m not in pet_no_switch)
         + (f" ({', '.join(pet_edge)} at the lowest-reliance edge)" if pet_edge else "")
         + (f", and {', '.join(pet_no_switch)} prefers the shortcut map already at the lowest reliance" if pet_no_switch else "")
         + f". The Pet models themselves follow the shortcut on half of the images only at {RC} = "
         f"{switch_text(pet_bsw, note='Section 5.4')}. The effect sizes temper this: at the "
         f"lowest Pet dose the median per-image AUC gap is {signed_span(pet_gap_low, f3)}, so the low-reliance "
         "preferences there are close to ties.")
    figure(figs["full"],
           f"Preference for the shortcut map over the full object map against chance-corrected reliance {RC}, for "
           "the four metrics in the seven settings (natural images first). Dots are single models; lines are the "
           "increasing isotonic fits used for the switch points. Dotted line: the assumed 50% ideal; dashed line: "
           "the behavioural switch, where the models' per-image follow rate crosses 0.5 (primary label"
           + "".join("; not identified on " + " and ".join(ni)
                     + ", where the line only marks the isotonic crossing inside a reliance gap" for ni in
                     [[SHORT[(r.base, r.shortcut)] for r in bsw.values() if r.status.startswith("not identified")]]
                     if ni)
           + "".join(", and not bracketed on " + " and ".join(nb) for nb in
                     [[SHORT[(r.base, r.shortcut)] for r in bsw.values()
                       if r.status == "identified" and not r.bracketed]] if nb)
           + "; Section 5.4). Data: results/dose_response/preference.csv, behaviour_summary.csv and "
           "behaviour_switch.csv")
    table_main(FULL, f"Metric responsiveness on {RC}, shortcut map vs full object map")
    t_ctrl = COUNTERS["table"] + 1
    f_ro = COUNTERS["fig"] + 1
    para(f"Where does the over-detection come from? Table {t_ctrl} and Fig. {f_ro} compare the object map with a "
         "random pixel ordering at the lowest dose, where the model barely uses the shortcut. On CIFAR-10 black "
         f"and blur deletion prefer the random ordering over the centred object map {pct_span(ro_bb['cifar10'])} "
         f"of the time with the small CNN and {pct_span(ro_bb['cifar10_resnet18'])} with ResNet-18, against "
         f"{pct_span(ro_ii['cifar10'] + ro_ii['cifar10_resnet18'])} for inpainting deletion and insertion. With "
         f"the real Pet masks the bias is smaller ({pct_span(ro_bb['pet_resnet18'])} against "
         f"{pct_span(ro_ii['pet_resnet18'])}), which supports the " + S("earlier ", "") + "hypothesis that the Gaussian object map "
         "inflated it on CIFAR-10. On Pet, black and blur deletion do not rank the real object mask above random "
         f"pixels (random-vs-object preference {span(pet_ro_bb)}); inpainting deletion and insertion do, but "
         f"only weakly ({span(pet_ro_ii)}).")
    table_controls()
    t_over = COUNTERS["table"] + 1
    o_bb = [over[("pet_resnet18", m)] for m in BB]
    para(f"Table {t_over} decomposes the over-detection. At the lowest dose the shortcut-vs-object preference "
         "(S\u2013O) equals the random-vs-object preference (R\u2013O) within "
         f"{f2(max_so_ro)} in all {len(over_main)} metric\u2013setting cells, and the shortcut-vs-random preference "
         f"(S\u2013R) is {span(nat_over.pref_sr_low)} on the natural images. At low reliance the metrics thus rate "
         "the shortcut pixels about like random pixels, which is correct, and the preference above 0.5 comes "
         "from the object map not beating a random ordering. On CIFAR-10 that object map is a centred Gaussian "
         "prior; on Pet it is the real mask. The excess preference (S\u2013O \u2212 R\u2013O) is what the shortcut "
         f"itself adds. At the lowest dose it is {signed_span(nat_over.excess_low)} on the natural images, so the "
         "over-detection at low reliance is the object map not beating random pixels. As reliance grows the "
         f"excess rises (one-sided p_bh < 0.05) in {len(nat_excess_sig)} of {len(nat_over)} natural-image cells: "
         f"all four Pet metrics (\u03c1 = {span(pet_over.rho_excess_rc)})"
         + "".join(f"; {r.metric} on {SHORT[(r.base, r.shortcut)]} (\u03c1 = {f2(r.rho_excess_rc)})"
                   for r in nat_excess_sig[nat_excess_sig.base != "pet_resnet18"].itertuples())
         + ". It does not rise significantly for "
         + ", ".join(f"{r.metric} on {SHORT[(r.base, r.shortcut)]} (\u03c1 = {f2(r.rho_excess_rc)})"
                     for r in nat_excess_ns.itertuples())
         + ".")
    para("The over-detection rates are imprecise and depend on the threshold. With Clopper\u2013Pearson intervals, "
         f"Pet black deletion over-detects for {int(pet_black_over.k_fpr)} of {int(pet_black_over.n_below)} models, "
         f"{f2(pet_black_over.fpr)} {ci(pet_black_over.fpr_cp_low, pet_black_over.fpr_cp_high)}. If a model "
         "counts only when the lower end of its own 95% image-bootstrap interval of preference exceeds 0.5, "
         f"Pet black and blur deletion fall to {span(pet_bb_margin)} and inpainting deletion and insertion to "
         f"{span(pet_ii_margin)}, while CIFAR-10 black and blur deletion stay at {span(cifar_bb_margin)}. Split by "
         f"reliance, Pet black and blur deletion over-detect for {span([o.fpr_rc_le10 for o in o_bb])} of the "
         f"{int(o_bb[0].n_rc_le10)} models with {RC} \u2264 10 and inpainting deletion and insertion for "
         f"{span([over[('pet_resnet18', m)].fpr_rc_le10 for m in II])}; among the {int(o_bb[0].n_rc_10_50)} Pet "
         f"models with 10 < {RC} \u2264 50 every metric gives {span(pet_over.fpr_rc_10_50)}. The Pet inpainting "
         "and insertion over-detection therefore comes from models that already use the shortcut partly. On "
         f"MNIST patch no metric over-detects at {RC} \u2264 10. Misclassified images do not explain the result: at "
         f"{RC} \u2264 20, {pct(mis_low[('pet_resnet18', 'patch')])} of the aligned Pet images, "
         f"{pct(mis_low[('cifar10', 'patch')])} (CIFAR CNN) and {pct(mis_low[('pneumonia_mnist', 'patch')])} "
         "(PneumoniaMNIST patch) are misclassified, and scoring only "
         f"the correctly classified images gives Pet \u03c1 = {span(pet_correct.rho_rc_correct)} (all images "
         f"{span(pet_correct.rho_rc)}) and a Pet black and blur over-detection rate of "
         f"{span(pet_correct_bb.fpr_correct)} (all images {span(pet_correct_bb.fpr)}).")
    table_overdetection()
    figure(figs["r_object"],
           f"Preference for a random ordering over the full object map against {RC}. Ideal: low at low reliance; a "
           "rise at high reliance is correct, because the object map then explains the model worse than random. "
           "Data: results/dose_response/preference.csv")

    heading("5.3. Pre-registered test on Oxford-IIIT Pet", 2)
    t_pre = COUNTERS["table"] + 1
    para("The Pet predictions were committed before any Pet model was trained" + S(" (commit a962b6c)", "") + ", followed by the "
         "Pet code" + S(" (20694f2)", "") + ", a documented dose change after the first model and before any metric result "
         + S("(d823737), ", "") + "and the results with the recorded verdicts" + S(" (1e29918)", "") + ". P1 predicted \u03c1 > 0.8 between "
         "reliance and the shortcut-vs-object preference for all four metrics; P2 a larger random-over-object "
         "preference for black and blur deletion than for inpainting and insertion at the lowest dose; P3 "
         "starting values within 0.5 \u00b1 0.15 and \u03c1 > 0.8 in the form-matched comparisons; P4 \u03c1 > 0.8 "
         f"for the shortcut mass of Integrated Gradients and Occlusion. Table {t_pre} lists the outcomes" + S("", " (the full table is Supplementary Table S1)") + ", on the "
         f"18 original Pet models only, as amendment A2 commits (the {N_A2_PET} A2 Pet models are not used here). "
         "P1, P2 "
         "and P4 hold and P3 fails for black deletion (\u03c1 = "
         f"{f2(p3_black_sr.estimate)} against the random map and {f2(p3_black_sp.estimate)} against the placebo). "
         "Verdicts were judged on point estimates, as pre-registered; the intervals are added information. They "
         f"qualify P1 for black deletion, whose estimate {f2(p1_black.estimate)} has a two-level percentile "
         f"interval of {ci(p1_black.ci_low, p1_black.ci_high)} and a BCa interval of "
         f"{ci(p1_black_x.rho_raw_bca_ci_low, p1_black_x.rho_raw_bca_ci_high)}; "
         + ("both reach below the 0.8 threshold, so the data do not establish \u03c1 > 0.8 for black deletion."
            if p1_black_x.rho_raw_bca_ci_low < 0.8 else
            "the bias-corrected interval stays above the 0.8 threshold.")
         + " The P2 contrast, "
         f"{f3(p2_contrast.estimate)} {ci(p2_contrast.ci_low, p2_contrast.ci_high, f3)}, excludes zero.")
    if SUBMISSION:
        table_prereg(full=False)
        table_prereg(full=True)  # Supplementary Table S1
    else:
        table_prereg()

    heading("5.4. Per-image behavioural ground truth", 2)
    t_beh = COUNTERS["table"] + 1
    f_beh = COUNTERS["fig"] + 1
    para(f"Table {t_beh} and Fig. {f_beh} score the per-image AUC gap against the behavioural label. Pooled over "
         f"all images of a setting, the class-stratified AUROC is {span(pooled_nat)} on the natural images and "
         f"{span(pooled_mech)} on the mechanism settings. Under this primary label PneumoniaMNIST patch is weak "
         f"({span(pp_pooled)}; black deletion {f2(pp_black.auroc_pooled_strat)} "
         f"{ci(pp_black.auroc_strat_ci_low, pp_black.auroc_strat_ci_high)}, Holm-adjusted "
         f"{p_text(pp_black.p_holm_headline_strat)}), but this depends on the label: under the "
         f"shortcut-dominance label its pooled AUROC is {span(pp_dom_pooled)} (Table {t_beh + 1}). The pooled "
         "AUROC mixes models of different reliance. Within "
         f"a single model the median AUROC is {span(within_nat)} on the natural images and {span(within_mnist)} on "
         "MNIST, so most of the pooled signal separates high- from low-reliance models rather than followed from "
         "non-followed images of one model. These within-model values are imprecise: on the natural images the "
         f"smaller label group of a model has a median of {span([p.median_minority_count for p in nat_power], lambda v: f'{v:g}')} "
         "images, which gives a single model's AUROC a standard error of about "
         f"{span([p.median_hanley_mcneil_se for p in nat_power])} under the null (Hanley\u2013McNeil). Within {RC} "
         f"bands (Fig. {f_beh}) the PneumoniaMNIST patch AUROC "
         f"falls to {span(pp_band.auroc_strat)} above {RC} = 60%, where the metrics rank the images the "
         f"wrong way round (with only {span(pp_band.n_neg, lambda v: f'{v:.0f}')} non-followed images per band).")
    for b in ("mnist", "pneumonia_mnist"):
        sel = BEH[(BEH.comparison == FULL) & (BEH.base == b) & (BEH.shortcut == "patch")]
        assert sel[sel.label == "shortcut_dominant"].auroc_pooled_strat.min() > \
            sel[sel.label != "shortcut_dominant"].auroc_pooled_strat.max(), "text: dominance gives highest AUROC"
    para("The behavioural labels also give a derived ideal switch, judged by the same identification rule as "
         "the metric switches. Under the primary label the models follow the shortcut on half of the images at "
         f"the following {RC}: "
         + "; ".join(f"{switch_text(bsw[(b, s)])} on {short}" for b, s, _, short in SETTINGS)
         + ". On CIFAR-10 it is identified and bracketed with the amendment-A2 models at doses "
         f"{', '.join(f2(d) for d in A2_CIFAR_DOSES)} ({A2_COUNTS['cifar10_resnet18']} ResNet-18 and "
         f"{A2_COUNTS['cifar10']} CNN models). Without them" + S(" (results/dose_response/pre_a2_models/)", "") + " the crossing "
         "fell in a reliance gap without models and was only known to lie above the last model below the gap: "
         + " and ".join(f"{switch_text(r)} on {SHORT[(r.base, r.shortcut)]}" for r in cifar_pre_a2)
         + ". On Pet it is identified and, with the "
         f"{N_A2_PET} amendment-A2 models at dose {f2(A2_PET_DOSES[0])}, also bracketed: the nearest models lie "
         f"{f1(pet_bsw.distance_below)} points below ({RC} {f1(pet_below)}, an A2 model) and "
         f"{f1(pet_bsw.distance_above)} points above. Without the A2 models (the {N_ORIGINAL + N_A1} models" + S(", "
         f"results/dose_response/pre_a2_models/", "") + f") it was {switch_text(pet_pre_a2)}. The A2 models at this dose "
         f"reach {RC} {span(pet_a2_rc, f1)}: two seeds lie below the switch and one above it, so the seed "
         "spread at this dose is wide. The two check "
         "labels are still not bracketed on Pet. The assumed 50% is close to "
         f"the bracketed MNIST patch switch ({f1(mp_bsw.behavioural_switch_rc)}%) and too high for PneumoniaMNIST; "
         "on CIFAR-10 the bracketed switches lie above it ("
         + " and ".join(f"{f1(r.behavioural_switch_rc)}% on {SHORT[(r.base, r.shortcut)]}" for r in cifar_bsw)
         + "). Measured against the behaviour the over-detection "
         f"remains: on the natural images the calibration gap (preference minus follow rate) is positive in every "
         f"cell ({span(gap_nat)}), and {span(gap_det_nat)} when the preference is computed on the same "
         "determined images as the follow rate.")
    t_lab = COUNTERS["table"] + 2
    pet_und = bsw[("pet_resnet18", "patch")]
    pp_und = bsw[("pneumonia_mnist", "patch")]
    para(f"The behavioural results depend on the label (Table {t_lab}). The primary label asks whether the "
         "shortcut is sufficient under conflict once errors are removed: the moved shortcut wins, and the image "
         "without any shortcut is not already assigned to the shortcut's class. The moved-only check drops the "
         "second condition, so it also counts errors; the shortcut-dominance check asks whether the shortcut is "
         "more necessary than the object on the aligned image. The Pet switch is "
         f"{span(pet_label_sw, f1)}% depending on the label, and the PneumoniaMNIST texture switch "
         f"{span(pt_label_sw, f1)}%; on CIFAR-10 every label identifies it, at "
         + " and ".join(f"{span([swrow(b, s, lab).behavioural_switch_rc for lab in BEHAVIOUR_LABELS], f1)}% on "
                        f"{short}" for b, s, _, short in NATURAL[1:])
         + (", but it is not bracketed under " + "; ".join(cifar_check_nb) if cifar_check_nb else "")
         + ". Only the primary label leaves images "
         f"undetermined, on average {pct(pet_und.undetermined_rate_mean)} on Pet (up to "
         f"{pct(pet_und.undetermined_rate_max)} for one model) and {pct(pp_und.undetermined_rate_mean)} on "
         "PneumoniaMNIST patch, and the share rises with reliance (Spearman \u03c1 with "
         f"{RC} = {span(und_rho)} where it exceeds 10% for some model), so the label discards more images at high "
         "reliance. The shortcut-dominance label gives the highest pooled AUROCs on the patch mechanism settings "
         f"(MNIST patch black deletion {f2(mp_dom_black.auroc_pooled_strat)}), but for black deletion on MNIST "
         "patch it is partly circular: the image without the patch, which defines the label, is on the black "
         "MNIST background nearly the image that black deletion produces when it removes the shortcut pixels.")
    table_behaviour()
    table_labels()
    figure(figs["behaviour"],
           f"Class-stratified AUROC of each metric's per-image AUC gap against the behavioural label, within {RC} "
           + S("bands (shortcut vs full object). Panels rearranged from results/dose_response/behaviour_auroc.png "
             "(alphabetical setting order; the pet_resnet18_pretrained panel is the separate SD1 setting, "
             "Section 5.10)",
             "bands (shortcut vs full object). Panels are in the order of Table 2. Titles give the dataset or "
             "network and the shortcut: pet_resnet18 is the ResNet-18 trained from scratch on Pet, cifar10 the "
             "small CNN, and pet_resnet18_pretrained the separate SD1 (patch, Section 5.10) and SD4 (watermark, "
             "Section 5.12) settings"))

    heading("5.5. Mechanism settings: size and switch identification", 2)
    f_matched = COUNTERS["fig"] + 1
    t_matched = COUNTERS["table"] + 1
    mp_low_full, mp_low_matched = mp_full.pref_low, mp_matched.pref_low
    para("The full object map marks far more pixels than the shortcut map. Replacing it by a size-matched object "
         f"map (Fig. {f_matched}, {tref('matched', t_matched)}) raises the MNIST patch preference for the shortcut at the "
         f"lowest dose from {span(mp_low_full, f3)} to {span(mp_low_matched, f3)}, moves the switch from "
         f"{span(mp_full.switch_iso_rc, f1)}% to {span(mp_matched.switch_iso_rc, f1)}% {RC}, and raises the "
         f"over-detection rate from {span(mp_full.fpr)} to {span(mp_matched.fpr)}. The metrics still respond "
         f"(\u03c1 = {span(mp_matched.rho_rc)}), but much of what looked like \u201cthe metric recognises the "
         "object\u201d is the object map being larger. On PneumoniaMNIST texture, black deletion barely responds "
         f"under this control (\u03c1 = {f2(pt_black_m.rho_rc)}, p_bh = {format_p(pt_black_m.p_bh)}).")
    para("Before amendment A1, the original MNIST patch models left a gap in raw reliance from "
         f"{f1(gap_orig[0])}% to {f1(gap_orig[1])}%, and the switches against the full object map "
         f"({span(mp_orig.switch_iso_rc, f1)}% {RC}) lay inside it, so " + S("earlier drafts' statement that black and "
         "blur deletion switch \u201cclearly before 50%\u201d was not supported",
         "the original models could not show that black and blur deletion switch \u201cclearly before 50%\u201d")
         + ". With the A1 models the widest "
         f"gap between 5% and 95% raw reliance is {f1(gap_all[0])}% to {f1(gap_all[1])}%, every switch has a "
         f"model within {S(f1(mp_full.nearest_model_distance.max()) + ' points', 'about 1 point' if abs(mp_full.nearest_model_distance.max() - 1) < 0.1 else f1(mp_full.nearest_model_distance.max()) + ' points')}, and the switches are "
         f"{span(mp_full.switch_iso_rc, f1)}% {RC} (4-parameter logistic {span(mp_full.switch_4pl_rc, f1)}%), "
         f"against a behavioural switch of {f1(beh_switch[('mnist', 'patch')])}%. The added models moved the "
         "switches up and closer together, and raised the over-detection rate; " + S("Appendix C", tref("sets", 0))
         + " lists both model sets.")

    def cell(r):
        return f"{r.metric} on {SHORT[(r.base, r.shortcut)]}"

    mp_earlier = [m for m, r in sd_mp.items() if r.verdict == "metric earlier"]
    mp_nd = [m for m, r in sd_mp.items() if r.verdict == "not distinguishable"]
    para("Is the metric switch earlier than the behavioural switch? A paired bootstrap of the difference "
         "(behavioural \u2212 metric switch, primary label) on the same resamples answers this per cell. On MNIST "
         "patch it is "
         + "; ".join(f"{f1(sd_mp[m].difference)} {ci(sd_mp[m].diff_ci_low, sd_mp[m].diff_ci_high, f1)} points "
                     f"for {m}" for m in METRICS)
         + f", so {' and '.join(mp_earlier)} switch earlier than behaviour"
         + (f", while {' and '.join(mp_nd)} cannot be distinguished from it" if mp_nd else "")
         + ". Over all settings the cells that are not distinguishable are "
         + ", ".join(f"{cell(r)} ({f1(r.difference)} {ci(r.diff_ci_low, r.diff_ci_high, f1)})"
                     for r in sd_nd.itertuples())
         + "; the metric switches later than behaviour for "
         + ", ".join(cell(r) for r in sd_later.itertuples())
         + ". On the natural images all "
         f"{len(sd_nat_test)} testable cells are earlier (also under the crossed bootstrap), and the other "
         f"{len(sd_nat) - len(sd_nat_test)} cannot be tested because the metric already prefers the shortcut map "
         "at the lowest reliance."
         + (" Under the crossed bootstrap " + ", ".join(cell(r) for r in sd_crossed_lost.itertuples())
            + " also becomes indistinguishable from behaviour." if len(sd_crossed_lost) else ""))
    para("The crossed bootstrap, which respects that all models of a setting share the same images, widens the "
         "level intervals: the interval of the lowest-dose preference against the full object map is "
         f"{f2(pref_width_ratio.median())} times as wide as the two-level one in the median cell (range "
         f"{f2(pref_width_ratio.min())}\u2013{f2(pref_width_ratio.max())}), and the PneumoniaMNIST patch "
         f"black-deletion switch interval widens from {ci(pp_black_r.switch_iso_rc_ci_low, pp_black_r.switch_iso_rc_ci_high, f1)} "
         f"to {ci(pp_black_x.switch_crossed_ci_low, pp_black_x.switch_crossed_ci_high, f1)}. The crossed upper "
         f"bounds of the MNIST patch metric switches are {span(mp_x.switch_crossed_ci_high, f1)}%, against a "
         f"behavioural switch of {f1(mp_bsw.behavioural_switch_rc)}%; the paired test above is the direct "
         "comparison.")
    figure(figs["matched"],
           f"Preference for the shortcut map over the size-matched object map against {RC}. Layout as in Fig. "
           f"{f_full}. Data: results/dose_response/preference.csv")
    table_main(MATCHED, f"Metric responsiveness on {RC}, shortcut map vs size-matched object map", supp=True)
    para("The diffuse texture shows the form dependence from the other side. On MNIST texture, insertion "
         f"switches only at {f1(mt_ins.switch_iso_rc)}% {RC} and reaches a preference of only "
         f"{f2(mt_ins.pref_high)} at the highest dose. Against a random ordering, which has the same scattered "
         f"form, blur deletion does not respond (\u03c1 = {f2(mt_sr_blur.rho_rc)}, p_bh = {format_p(mt_sr_blur.p_bh)}) "
         f"and insertion moves the wrong way (\u03c1 = {f2(mt_sr_ins.rho_rc)}; against the placebo "
         f"{f2(mt_sp_ins.rho_rc)}). On PneumoniaMNIST texture black deletion also moves the wrong way in these "
         f"form-matched comparisons (\u03c1 = {f2(pt_black_sr.rho_rc)} against random and {f2(pt_black_sp.rho_rc)} "
         "against the placebo).")

    heading("5.6. Same-form controls", 2)
    f_sr = COUNTERS["fig"] + 1
    f_pr = f_sr + 1
    para(f"When the compared maps have the same form, the metrics mostly behave correctly (Fig. {f_sr}). On the "
         "patch shortcut, the preference for the shortcut map over a random ordering or a placebo starts at "
         f"{span(sf_low)} at the lowest dose and rises to "
         + "; ".join(f"{span(rows([S_RANDOM, S_PLACEBO], [s]).pref_high)} ({s[3]})" for s in sf_patch_settings)
         + " at the highest dose. The rise is smallest on PneumoniaMNIST patch and for black deletion on Pet, so "
         + S("\u201crises to 0.93\u20130.99 for every metric\u201d in earlier drafts was too general. ",
             "the rise is not the same for every metric. ") + "The placebo "
         f"behaves like a random ordering for the deletion variants (Fig. {f_pr}): placebo vs random stays at "
         f"{span(list(pr_del.pref_low) + list(pr_del.pref_high))} at the lowest and highest dose in every setting. "
         "Insertion is the exception: its placebo-over-random preference rises with reliance on MNIST (\u03c1 = "
         f"{span(pr_ins_mnist.rho_rc)}) and falls on Pet (\u03c1 = {f2(pet_pr_ins.rho_rc)}).")
    figure(figs["s_random"],
           f"Same-form control: preference for the shortcut map over a random ordering against {RC}. Ideal: about "
           "0.5 at 0% reliance, rising to 1. Data: results/dose_response/preference.csv")
    figure(figs["p_random"],
           f"Placebo control: preference for the placebo map over the random ordering against {RC}. Ideal: about "
           "0.5 at every reliance. Data: results/dose_response/preference.csv")

    heading("5.7. Comparing the metrics", 2)
    t_pairs = COUNTERS["table"] + 1
    ok_i, n_i, min_i = robust["Deletion (inpaint)"]
    para(f"Across the {len(SETTINGS)} settings and the {len(SHORTCUT_COMPARISONS)} comparisons of the shortcut "
         "map with another map (full object, size-matched object, placebo, random), "
         + (", ".join(robust_all) if robust_all else "no variant")
         + f" had a positive, BH-significant \u03c1 in all {n_i} cases (lowest \u03c1 = {f2(min_i)}). The other "
         "variants missed in " + ", ".join(f"{n_ - ok_} of {n_} cases ({m})" for m, (ok_, n_, _) in robust.items()
                                            if ok_ < n_) + ". Against the full object map alone, black, blur and "
         f"inpainting deletion are positive and significant in all {N_SETTINGS} settings. A paired test does "
         f"not make inpainting the best variant ({tref('pairs', t_pairs)}): {n_pairs_sig} of {len(main_only(PAIRS))} metric pairs "
         f"differ at p_bh < 0.05, and on PneumoniaMNIST texture black deletion has the higher \u03c1 against "
         f"the full object map (black \u2212 inpaint \u0394\u03c1 = {f3(pt_pairs['Deletion (inpaint)'].delta_rho_rc)} "
         f"{ci(pt_pairs['Deletion (inpaint)'].delta_rho_rc_ci_low, pt_pairs['Deletion (inpaint)'].delta_rho_rc_ci_high, f3)}, "
         f"Holm-adjusted {p_text(pt_pairs['Deletion (inpaint)'].p_holm)}). The supported claims are narrower: "
         "inpainting deletion responds in every comparison we ran, and black and blur deletion carry the "
         "largest form bias on natural images.")
    table_pairs()

    heading("5.8. Explanation methods", 2)
    f_m = COUNTERS["fig"] + 1
    t_m = COUNTERS["table"] + 1
    gc = mt_away["Grad-CAM"]
    para(f"Fig. {f_m} and Table {t_m} show how much positive attribution each method places on the shortcut. "
         f"Integrated Gradients follows the shortcut in every setting (\u03c1 = {span(ig)}). On MNIST texture, "
         f"Saliency and SmoothGrad move away from the shortcut as reliance grows (\u03c1 = "
         f"{f2(mt_away['Saliency'].rho_rc)} and {f2(mt_away['SmoothGrad'].rho_rc)}); Grad-CAM's trend is also "
         f"negative (\u03c1 = {f2(gc.rho_rc)}, Holm-adjusted {p_text(gc.p_holm)}), but its shortcut mass changes "
         f"only from {f3(gc.mass_low)} to {f3(gc.mass_high)}, which is negligible. This pattern is specific to "
         "MNIST texture: on PneumoniaMNIST texture Saliency and SmoothGrad move towards the shortcut (\u03c1 = "
         f"{f2(pt_meth['Saliency'].rho_rc)} and {f2(pt_meth['SmoothGrad'].rho_rc)}) and only Grad-CAM moves away "
         f"(\u03c1 = {f2(pt_meth['Grad-CAM'].rho_rc)}). Grad-CAM places the least attribution on the patch at the "
         f"highest dose in {gc_lowest} of the {len(sf_patch_settings)} patch settings.")
    figure(figs["methods"],
           S("Share of positive attribution that each explanation method places on the shortcut pixels, against raw "
             "reliance. Panels rearranged from results/dose_response/method_dose_response.png (alphabetical setting "
             "order; the pet_resnet18_pretrained panel is the separate SD1 setting, whose Grad-CAM maps are 4\u00d74)",
             "Share of positive attribution that each explanation method places on the shortcut pixels, against raw "
             "reliance. Panels are in the order of Table 2; titles are as in Fig. 3, and the two "
             "pet_resnet18_pretrained panels are the separate SD1 (patch) and SD4 (watermark) settings, whose "
             "Grad-CAM maps are 4\u00d74"))
    table_methods()

    heading("5.9. Reliance and object competence", 2)
    t_conf = COUNTERS["table"] + 1
    cnn = conf_cifar.iloc[0]
    least = [k for k, v in conf_set.sort_values(ascending=False).items() if v > cnn.rho_rc_competence]
    para("A model that relies more on the shortcut is also worse at the object, so a metric that responds to "
         "reliance may be responding to object competence. Table "
         f"{t_conf} separates the two on {RC}. Reliance and accuracy without the shortcut are least collinear "
         "on " + " and ".join(f"{SHORT[k]} (\u03c1 = {f2(conf_set[k])})" for k in least)
         + f", then the CIFAR small CNN (\u03c1 = {f2(cnn.rho_rc_competence)}; "
         f"{f2(cnn_pre_a2.rho_rc_competence)} without its {A2_COUNTS['cifar10']} A2 models, which fill its "
         f"reliance gap). The partial \u03c1 of preference with {RC} given competence stays significant for all "
         "four metrics on "
         + " and ".join(f"{SHORT[k]} ({span([ccrow(*k, m).partial_rc_given_acc for m in METRICS])})"
                        for k in sorted(conf_all_sig, key=lambda k: list(SHORT).index(k)))
         + ". In "
         f"{len(collinear)} of {len(conf_set)} settings the two are collinear (\u03c1 \u2264 \u22120.95), and there "
         "the partial correlations are smaller and in places not significant (MNIST texture "
         f"{signed_span(CONF_RC[(CONF_RC.base == 'mnist') & (CONF_RC.shortcut == 'texture')].partial_rc_given_acc)}). "
         f"After Benjamini\u2013Hochberg over the {len(CONF_RC)} tests, {conf_rc_sig} of the {len(main_only(CONF_RC))} "
         "partial correlations of the main settings remain significant. Where reliance and competence are "
         "collinear, our design cannot fully separate the two.")
    table_confound()

    heading("5.10. Pre-registered amendment A2: pretrained Pet (SD1) and the filled reliance gaps (SD2)", 2)
    t_a2 = COUNTERS["table"] + 1
    sd1_bb = [sd1_over[m] for m in BB]
    sd1_ii = [sd1_over[m] for m in II]
    sd1_conf = {m: ccrow(*SD1_KEY, m) for m in METRICS}
    sd1_beh = {m: brow(FULL, *SD1_KEY, m) for m in METRICS}
    sd1_levels = levels(*SD1_KEY)
    sd1_dose0 = sd1_levels[sd1_levels.dose == 0].iloc[0]
    para("Amendment A2 and its predictions P5\u2013P9" + S(" (docs/PREDICTIONS_pretrained.md)", "") + " were committed and "
         "timestamped on Zenodo before any SD1 or SD2 model was trained; the verdicts below apply the "
         f"pre-registered rules literally, to point estimates, and are summarised in Table {t_a2}. P5\u2013P8 use "
         f"the {N_SD1} SD1 models, an ImageNet-pretrained ResNet-18 fine-tuned on Pet with the same patch "
         f"shortcut (Section 4). Their {RC} spans {f1(sd1_low)}% to {f1(sd1_g.reliance_corrected.max())}%: at "
         f"dose 0 the pretrained models reach only {f1(sd1_dose0.reliance_corrected)}% on average, while every "
         f"model at the next dose (+0.1) has at least {f1(sd1_gap[1])}%, so {int(sd1_over[METRICS[0]].n_below)} "
         f"models lie at or below {RC} = 50 and {int(sd1_over[METRICS[0]].n_above)} above it.")
    para(f"**Over-detection persists after pretraining (P5 {p_verdicts['P5']}).** Among the "
         f"{int(sd1_bb[0].n_below)} SD1 models with {RC} \u2264 50, black deletion prefers the shortcut map for "
         f"{int(sd1_bb[0].k_fpr)} ({f2(sd1_bb[0].fpr)} {ci(sd1_bb[0].fpr_cp_low, sd1_bb[0].fpr_cp_high)}) and "
         f"blur deletion for {int(sd1_bb[1].k_fpr)} ({f2(sd1_bb[1].fpr)} "
         f"{ci(sd1_bb[1].fpr_cp_low, sd1_bb[1].fpr_cp_high)}), so both rates exceed 0.5 as predicted; the "
         "Clopper\u2013Pearson interval of blur deletion still reaches well below 0.5, and with a per-model "
         f"margin both fall to {span([o.fpr_margin for o in sd1_bb])}. **Inpainting and insertion over-detect "
         f"less (P6 {p_verdicts['P6']}):** {f2(sd1_ii[0].fpr)} and {f2(sd1_ii[1].fpr)}, each below both black "
         f"and blur deletion. Split by reliance, the over-detection at {RC} \u2264 10 is "
         + ", ".join(f"{f2(sd1_over[m].fpr_rc_le10)} ({m})" for m in METRICS)
         + f" over {int(sd1_bb[0].n_rc_le10)} models, and every metric over-detects for all "
         f"{int(sd1_bb[0].n_rc_10_50)} models with 10 < {RC} \u2264 50. "
         f"**The size-matched comparison switches at the edge (P7 {p_verdicts['P7']}):** against the "
         f"size-matched object map {sum(sd1_edge.values())} of 4 metrics switch within 5 points of the "
         f"lowest-{RC} model or already prefer the shortcut map there.")
    para(f"**The metrics respond to reliance, but the excess preference does not rise (P8 "
         f"{p_verdicts['P8']}).** Part (a) holds: Spearman \u03c1 between {RC} and the shortcut-vs-object "
         f"preference is {', '.join(f2(sd1_full[m].rho_rc) for m in METRICS)} for the four metrics, above 0.8 "
         f"in {p8a_n} of 4. Part (b) fails: \u03c1 between {RC} and the excess shortcut preference (S\u2013O "
         "\u2212 R\u2013O) is "
         + ", ".join(f2(sd1_over[m].rho_excess_rc) for m in METRICS)
         + " (one-sided p_bh over the four metrics "
         + ", ".join(format_p(sd1_bh4[m]) for m in METRICS)
         + f"), positive and significant in {p8b_n} of 4 (inpainting deletion only), against the required 3. "
         "For the other three metrics the rise of the shortcut-vs-object preference with reliance is not "
         "significantly larger than the rise of the random-vs-object preference, so the shortcut pixels add "
         f"little beyond a random ordering. At the lowest dose S\u2013O, R\u2013O and S\u2013R are "
         + "; ".join(f"{f2(sd1_over[m].pref_so_low)}, {f2(sd1_over[m].pref_ro_low)} and "
                     f"{f2(sd1_over[m].pref_sr_low)} ({m})" for m in METRICS)
         + ".")
    p9_tested = [r for _, tested, _ in p9_rows.values() for r in tested]
    p9_untested = [r for *_, untested in p9_rows.values() for r in untested]
    assert all(r.metric_switch_status == "above 0.5 at lowest reliance" for r in p9_untested)
    para(f"**Metrics switch before behaviour (P9 {p_verdicts['P9']}).** With the A2 models the behavioural "
         "switch is identified and bracketed in all three SD2 settings: "
         + "; ".join(f"{switch_text(bsw[(b, s)])} on {short}" for b, s, _, short in p9_settings)
         + f". In each of the {len(p9_tested)} cells with an identified metric switch against the full object "
         "map, the paired bootstrap interval of the difference (behavioural \u2212 metric switch) lies above 0 "
         f"(differences {span([r.difference for r in p9_tested], f1)} points, lower interval ends "
         f"{span([r.diff_ci_low for r in p9_tested], f1)}; Table {t_a2 + 2}). The other {len(p9_untested)} "
         "cells have no metric switch because the metric already prefers the shortcut map at the lowest "
         "reliance, which P9 does not test.")
    para(f"**Secondary outcomes (no verdict).** The SD1 behavioural switch is not identified: the isotonic "
         f"crossing lies at {f1(sd1_bsw.behavioural_switch_rc)}%, but no model lies between {RC} "
         f"{f1(sd1_gap[0])}% and {f1(sd1_gap[1])}%, so it is only known to lie above {f1(sd1_gap[0])}% "
         "(the same holds under both check labels). Against the primary label, the class-stratified pooled "
         f"AUROC is {span([r.auroc_pooled_strat for r in sd1_beh.values()])} and the within-model median "
         f"{span([r.median_model_auroc_strat for r in sd1_beh.values()])}; "
         f"{pct(sd1_bsw.undetermined_rate_mean)} of the images are undetermined on average (up to "
         f"{pct(sd1_bsw.undetermined_rate_max)}), with Spearman \u03c1 = {f2(sd1_bsw.rho_undetermined_rc)} "
         f"between that share and {RC}. Reliance and object competence are collinear on SD1 "
         f"(\u03c1({RC}, acc) = {f2(sd1_conf[METRICS[0]].rho_rc_competence)}), and the partial \u03c1 of "
         f"preference with {RC} given competence is "
         + ", ".join(f"{f2(sd1_conf[m].partial_rc_given_acc)} ({m})" for m in METRICS)
         + f". Compared exploratorily with the Pet ResNet-18 from scratch (Table {t_a2 + 1}), the pretrained "
         f"network gives a similar responsiveness (\u03c1 = {span([r.rho_rc for r in sd1_full.values()])} against "
         f"{span([r.rho_rc for r in sd1_pet.values()])}) and lower over-detection by blur deletion, inpainting "
         "and insertion. Method-level comparisons between the two networks are confounded by the 4\u00d74 "
         "Grad-CAM maps of the pretrained network (16\u00d716 from scratch, Section 4).")
    assert all(sd1_over[m].fpr < over[("pet_resnet18", m)].fpr for m in METRICS[1:]), "text: lower over-detection"
    assert all(sd1_over[m].fpr_rc_10_50 == 1 for m in METRICS), "text: every metric over-detects at 10-50"
    assert [m for m in METRICS if sd1_over[m].rho_excess_rc > 0 and sd1_bh4[m] < 0.05] == ["Deletion (inpaint)"]
    table_a2_outcomes(p_verdicts, sd1_over, sd1_full, sd1_matched, sd1_edge, sd1_bh4, p8a_n, p8b_n, p9_rows, bsw)
    table_sd1(sd1_bh4)
    table_p9(p9_rows, bsw)

    if SUBMISSION:
        heading("5.11. Exact ROAD (SD3: pre-registered predictions P10 and P11)", 2)
        para("Predictions P10 and P11 were registered in amendment A3 before any SD3 score existed. P10: in each of "
             "the four natural-image settings, at its lowest-dose models, the mean preference for a random map over "
             "the object map is lower under ROAD (MoRF) than under black deletion, on the same images. P11: on the "
             "Pet ResNet-18 from scratch, the over-detection rate of ROAD in the shortcut-versus-object comparison "
             "is lower than that of black deletion, on the same models and images. ROAD LeRF is reported next to "
             f"ROAD MoRF and decides no verdict. Table {COUNTERS['table'] + 1} gives the numbers that decide both.")
        sd3_results(a3)
        heading("5.12. Watermark shortcut (SD4: pre-registered predictions P12 and P13)", 2)
        para(f"Predictions P12 and P13 were registered in amendment A3 before any SD4 model existed. P12: Spearman "
             f"\u03c1 between {RC} and the shortcut-versus-object preference is above 0.8 for at least three of the "
             "four existing metrics over the 18 confirmatory SD4 models. P13: the over-detection rate of black "
             f"deletion and of blur deletion exceeds 0.5 among the SD4 models with {RC} \u2264 50.")
        sd4_results(a3)

    # ---- 6 Discussion
    heading("6. Discussion")
    para("**Responsive but not calibrated.** Every metric ranks models by reliance, but a ranking is not a "
         "verdict. A practitioner who reads \u201cthe metric prefers the shortcut explanation\u201d as \u201cthe "
         "model uses the shortcut\u201d would be wrong, under black and blur deletion, for at least half of the "
         "low-reliance Pet models and nearly all low-reliance CIFAR-10 models, even when a per-model margin is "
         "required. At low reliance this error comes from the object map, which these metrics do not rank above "
         "random pixels, rather than from the shortcut map itself. Bilodeau et al. " + cite("bilodeau") + " argue that the end task must be "
         "defined; our behavioural labels give one such definition, and against it the metrics over-detect. "
         "BAM " + cite("bam") + " and Adebayo et al. " + cite("adebayo2022") + " found false positives in "
         "attribution methods; our oracle maps show the same kind of error in the metrics used to evaluate "
         "those methods, with no attribution method involved.")
    para("**Size and form.** The size-matched control removes much of the preference for the object map at low "
         "reliance, and the random control shows black and blur deletion preferring scattered removal over "
         "compact removal on natural images. Both fit known mechanisms: removal shape leaks class information "
         + cite("rong") + ", removal-based benchmarks favour spatially blurry masks " + cite("roarpitfalls")
         + ", interpolation-based masking leaves predictive information in place " + cite("rong", "aim")
         + ", the fill strategy decides rankings " + cite("blucher") + ", deletion and insertion read only the "
         "ranking of a map " + cite("gomez") + ", and common metrics struggle to separate methods from random "
         "attribution " + cite("saco") + ". The real Pet masks reduce the CIFAR-10 form bias, so part of it was "
         "the Gaussian prior; the rest is a property of the metrics. Any mask-based evaluation that compares a "
         "small marker with a large object region inherits the size confound.")
    para("**Why the radiograph setting is harder.** On PneumoniaMNIST the \u201cobject\u201d map is an intensity "
         "ranking of the whole radiograph, there are two classes, and the model's probability saturates often: "
         f"the share of tied comparisons reaches {pct(pneu_ties)} against at most {100 * mnist_ties:.1f}% on "
         "MNIST. Results that hold on MNIST should not be transferred to medical images without checking "
         + cite("gevaert") + ".")
    para("**Why insertion fails on a diffuse shortcut.** Restoring scattered pixels into a blurred image changes "
         "it little, because the blur has already spread the texture's brightness over neighbouring pixels. "
         "This explanation was not tested separately.")
    para("**Why some methods move away from the texture.** On MNIST texture, Saliency and SmoothGrad put less "
         "weight on the shortcut as reliance grows, possibly from gradient saturation once the model is confident "
         "or because the texture adds to bright digit pixels. The opposite trend on PneumoniaMNIST texture shows "
         "that this is not a general property of these methods.")
    para("**Recommendations.** (1) Test a metric at several measured, chance-corrected reliance levels before "
         "using it to rank explanation methods, and report its over-detection rate, not only a correlation. (2) "
         "Derive the ideal switch from behaviour rather than assuming 50%. (3) Compare maps of matched size and "
         "form, or report size-matched and random controls. (4) Do not use insertion for diffuse evidence. (5) "
         "Report paired tests before calling one metric variant better than another.")

    # ---- 7 Limitations
    heading("7. Limitations")
    bullets([
        "All metrics come from one family (ordered pixel removal or insertion with three fills). Other families "
        "such as faithfulness correlation or region perturbation were not tested, and the inpainting variant "
        "is inspired by ROAD but is not ROAD" + S(".", "; exact ROAD is tested separately (Section 5.11)."),
        "The networks of the main study are trained from scratch; the Pet ResNet-18 uses a CIFAR-style stem at "
        f"128\u00d7128. The only pretrained network is the ImageNet ResNet-18 of SD1 ({N_SD1} models, one "
        "dataset and one shortcut), and no transformer backbone was tested.",
        "The shortcut is synthetic (a white patch or a brightened texture). The dose varies the shortcut's "
        "predictivity, not its availability " + cite("hermann") + ".",
        "The oracle maps have limits: on PneumoniaMNIST the object map is an intensity ranking of the whole "
        "radiograph; on CIFAR-10 it is a centred Gaussian prior, so Pet is the primary natural-image evidence; "
        "the size-matched map on MNIST takes the topmost tied saturated pixels; blur deletion leaves about "
        "20\u201325% of the patch intensity in place; and the inpainting curve ends on noise around black.",
        f"Reliance and object competence are collinear in {len(collinear)} of {len(conf_set)} settings (Section 5.9).",
        "The chance correction assumes that errors spread evenly over the wrong classes.",
        "The behavioural switches on the natural images rest on the A2 models in the former reliance gaps: "
        "on Pet they bracket the switch under the primary label only, the A2 Pet dose has only three seeds "
        "with a wide spread, and the CIFAR-10 small CNN has two seeds per dose. On SD1 the reliance jumps from "
        f"{RC} {f1(sd1_gap[0])}% to {f1(sd1_gap[1])}% between adjacent doses, so its behavioural switch is not "
        "identified. Each model is scored on 100 "
        "images, the within-model AUROCs on natural images have null standard errors of about "
        f"{span([p.median_hanley_mcneil_se for p in nat_power])}, and the low-reliance preferences "
        "on Pet are near ties.",
        "The behavioural switch and the per-image AUROCs depend on which behavioural label is used; all three "
        "labels are reported.",
        "The switch-identification window (\u00b110 points), the causal behavioural label and the class-stratified "
        "AUROC were chosen " + S("after the audit and ", "") + "after the original-set results had been inspected. The amendment-A1 "
        "models had been trained by then, but they entered no analysis until these choices were committed.",
    ] + ([
        "The tests of the class-stratified AUROC against 0.5 are one-sided (H1: AUROC > 0.5). A metric that ranks "
        "the images the wrong way round, as the PneumoniaMNIST patch metrics do above 60% reliance, is therefore "
        "not tested in that direction; such cases are read from the AUROC and its interval (Section 5.4).",
        "Exact ROAD (SD3) was run with one imputation noise level and one removal grid, on 50 of the 100 images "
        "for the ResNet-18 settings, and its LeRF grid differs from the reference implementation's. The P11 "
        "verdict rests on one model of "
        f"{int(A3N['p11_main'].n_below)}, with overlapping intervals.",
        f"The watermark of SD4 is partly position-coded within the corner, so SD4 is not a purely content-coded "
        f"shortcut, and it changes the shape, contrast and layout of the shortcut at once. Its reliance jumps "
        f"from {RC} {f1(A3N['gap'][0])}% to {f1(A3N['gap'][1])}% between adjacent doses, so its behavioural switch "
        "is not identified and its \u03c1 values mostly reflect two groups of models.",
    ] if SUBMISSION else []))

    # ---- 8 Conclusion
    heading("8. Conclusion")
    para(f"Across {N_MODELS} models with controlled and measured shortcut reliance, deletion and insertion metrics "
         "respond to reliance but are not calibrated to it, and their verdicts depend on the size and form of "
         "the explanations they compare. On natural images black and blur deletion report a shortcut for at "
         "least half of the Pet models and nearly all CIFAR-10 models that barely use it, the metrics switch far below the reliance at which the models follow the "
         "shortcut on half of the images, and within a model they separate followed from non-followed images "
         "only weakly. The pre-registered Pet test confirmed three of four predictions and showed black "
         "deletion failing a form-matched control. The pre-registered amendment confirmed four of five further "
         "predictions: black and blur deletion also over-detect on an ImageNet-pretrained network, and the "
         "metrics switch before behaviour once the natural-image reliance gaps are filled; the excess shortcut "
         "preference of the pretrained network did not rise significantly with reliance for three of the four "
         "metrics. Metrics should be validated at several measured reliance "
         "levels, against behavioural ground truth and with size- and form-matched controls, before they are "
         "used to rank explanation methods. Future work should test other metric families, other pretrained "
         "backbones and realistic shortcuts." + (S("", (
             f" A third pre-registered amendment confirmed {n_word[len(a3['held'])]} of four predictions: exact ROAD "
             "ranked the object map above random pixels more often than black deletion in all four natural-image "
             "settings and over-detected slightly less on Pet, and the four metrics tracked reliance on a watermark "
             "shortcut; the prediction that blur deletion over-detects the watermark was refuted "
             f"({int(a3['over']['Deletion (blur)'].k_fpr)} of {int(a3['over']['Deletion (blur)'].n_below)} "
             "low-reliance models), while black deletion over-detected it for "
             f"{int(a3['over']['Deletion (black)'].k_fpr)} of {int(a3['over']['Deletion (black)'].n_below)}."))
             if SUBMISSION else ""))

    if not SUBMISSION:
        declarations()

    # ---- Appendix A
    heading("Appendix A. Supporting study: the sanity check against ground truth")
    t_k = S(COUNTERS["table"] + 1, "A.1")  # the submission numbers the appendix Table A.1 and Fig. A.1
    f_gt = S(COUNTERS["fig"] + 1, "A.1")
    APPENDIX["on"] = SUBMISSION
    para("A separate fixed-shortcut study trained three seeds on each of five datasets, including two decoy "
         "datasets with a class-coded patch " + cite("ross") + ". The models follow the patch "
         f"{ACCURACY.loc['decoy_mnist', 'shortcut_reliance_mean']:.0f}% (Decoy MNIST) and "
         f"{ACCURACY.loc['decoy_pneumonia', 'shortcut_reliance_mean']:.0f}% (Decoy Pneumonia) of the time, so on "
         "test images whose patch points to a wrong class the patch is the ground-truth evidence. On Decoy MNIST "
         f"Integrated Gradients puts {DECOY_MNIST.loc['Integrated Gradients', 'mass_accuracy']:.0%} of its positive "
         f"attribution on the patch and Occlusion {DECOY_MNIST.loc['Occlusion', 'mass_accuracy']:.0%}, against at "
         f"most {max(DECOY_MNIST.loc[m, 'mass_accuracy'] for m in ['Saliency', 'SmoothGrad', 'Grad-CAM']):.0%} for "
         f"the other methods. Table {t_k} and Fig. {f_gt} compare each metric's ranking of the five methods with "
         "the ground-truth ranking. The model-randomisation sanity check " + cite("adebayo2018") + " is **not "
         "shown to agree** with the ground truth: its mean Kendall \u03c4 is "
         f"{f2(sanity[0].mean_tau)} on Decoy MNIST (exact one-sided {p_text(sanity[0].p_mean_one_sided)}) and "
         f"{f2(sanity[1].mean_tau)} on Decoy Pneumonia ({p_text(sanity[1].p_mean_one_sided)}). With five methods, "
         f"a single seed reaches one-sided p < 0.05 only at \u03c4 \u2265 {f1(tau_threshold)}, so the study cannot "
         "show disagreement either. The perturbation metrics agree with the ground truth on Decoy MNIST (mean "
         f"\u03c4 = {span(del_tau_mnist)}) but not on Decoy Pneumonia ({signed_span(del_tau_pneu)}); per image, "
         f"deletion scores correlate with correct localisation (Spearman \u03c1 = {span(del_inst)}) and insertion "
         f"less so ({span(ins_inst)}).")
    table_kendall()
    figure(AN / "gt_agreement.png",
           "Supporting study. Left: Kendall \u03c4 between each metric's ranking of the five methods and the "
           "ground-truth ranking (mean \u00b1 standard deviation over three seeds). Right: per-image Spearman "
           "\u03c1 between metric scores and ground-truth localisation. Source: results/analysis/gt_agreement.png",
           width=Inches(6.0))
    APPENDIX["on"] = False

    if SUBMISSION:
        declarations()
        with use_doc(SUPP):
            appendices_bc()
    else:
        appendices_bc()

    # ---- References
    heading("References")
    if not SUBMISSION:
        para("[Note for the guide: each entry was checked against the paper's own page (arXiv, proceedings, journal "
             "or PubMed Central) on " + SEARCH_RECHECK + "; entries without authors are cited by title. Page numbers "
             "and final venues must be checked before submission.]", italic=True, size=9, spacing=1.0)
    for i, key in enumerate(CITED, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(f"[{i}] {REFERENCES[key]}")
        run.font.size = Pt(9.5)
    unused = [k for k in REFERENCES if k not in CITED]
    assert not unused, f"references listed but never cited: {unused}"
    if SUBMISSION:
        submission_checks(doc, SUPP, final=FINAL)
        SUPP.save(SUPP_OUTPUT)
        print(f"wrote {SUPP_OUTPUT} ({COUNTERS['stable']} supplementary tables) and {HIGHLIGHTS_OUTPUT}")
    doc.save(OUTPUT)
    print(f"wrote {OUTPUT} ({COUNTERS['fig']} figures, {COUNTERS['table']} tables, {len(CITED)} references, "
          f"{N_MODELS} models, abstract {word_count(abstract)} words)")


if __name__ == "__main__":
    build()

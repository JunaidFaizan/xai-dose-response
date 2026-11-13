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
        original_models/responsiveness_corrected.csv (192-model check set, Appendix C)
    results/analysis/  gt_agreement.csv, gt_instance.csv, accuracy.csv,
        decoy_mnist_summary.csv, kendall_exact.csv (supporting study, Appendix A)
Model counts are derived from models.csv. Nothing is retrained or re-evaluated.
The script only writes inside paper/ (the document and the figures in paper/figures/).
"""

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
from xai.stats import format_p, pava  # noqa: E402

DR = ROOT / "results" / "dose_response"
AN = ROOT / "results" / "analysis"
EX = ROOT / "results" / "explanations"
FIG_OUT = PAPER / "figures"
OUTPUT = PAPER / "paper_draft.docx"

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
LEGACY = pd.read_csv(DR / "responsiveness.csv")
METHOD = pd.read_csv(DR / "method_responsiveness_corrected.csv")
EFFECT = pd.read_csv(DR / "effect_sizes.csv")
PAIRS = pd.read_csv(DR / "metric_pairs.csv")
PET = pd.read_csv(DR / "pet_preregistered.csv")
BEH = pd.read_csv(DR / "behaviour_summary.csv")
BAND = pd.read_csv(DR / "behaviour_by_reliance.csv")
CONF = pd.read_csv(DR / "confound.csv")
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
LABEL = {(b, s): label for b, s, label, _ in SETTINGS}
SHORT = {(b, s): short for b, s, _, short in SETTINGS}
NETS = {"mnist": "Small CNN", "pneumonia_mnist": "Small CNN", "cifar10": "CIFAR CNN (0.77M)",
        "cifar10_resnet18": "ResNet-18 (11.2M)", "pet_resnet18": "ResNet-18 (11.2M)"}
BASE_NAMES = {"mnist": "MNIST", "pneumonia_mnist": "PneumoniaMNIST", "cifar10": "CIFAR-10 (small CNN)",
              "cifar10_resnet18": "CIFAR-10 (ResNet-18)", "pet_resnet18": "Oxford-IIIT Pet (ResNet-18)"}
FULL, MATCHED = "shortcut vs full object", "shortcut vs size-matched object"
S_RANDOM, S_PLACEBO = "shortcut vs random", "shortcut vs placebo"
R_OBJECT, P_OBJECT, P_RANDOM = "random vs full object", "placebo vs full object", "placebo vs random"
SHORTCUT_COMPARISONS = [FULL, MATCHED, S_PLACEBO, S_RANDOM]

assert set(zip(MODELS.base, MODELS.shortcut)) == {(b, s) for b, s, _, _ in SETTINGS}, \
    "SETTINGS must cover every (base, shortcut) in models.csv"


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
    """A share (0-1) as a whole percentage."""
    return f"{100 * x:.0f}%"


def span(values, fmt=f2, unit=""):
    """'a-b' range (a single value if both ends print the same)."""
    values = [v for v in values if not pd.isna(v)]
    lo, hi = fmt(min(values)), fmt(max(values))
    return f"{lo}{unit}" if lo == hi else f"{lo}{DASH}{hi}{unit}"


def pct_span(values):
    """'a-b%' range of shares (0-1) as whole percentages."""
    values = [v for v in values if not pd.isna(v)]
    lo, hi = f"{100 * min(values):.0f}", f"{100 * max(values):.0f}"
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


def sig(r, p_col="p_bh"):
    return "" if r[p_col] < 0.05 else "\u2020"


# ------------------------------------------------------------------ counts (from models.csv)

N_MODELS = len(MODELS)
N_ORIGINAL = int((MODELS.model_set == "original").sum())
N_A1 = int((MODELS.model_set == "A1").sum())
assert N_ORIGINAL + N_A1 == N_MODELS
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
    """r_c at which the per-image follow rate crosses 0.5 (same for every metric)."""
    return brow(FULL, base, shortcut, METRICS[0]).observed_switch_rc


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
    "he": "He K, et al. (2015) Deep residual learning for image recognition. arXiv:1512.03385",
    # data
    "medmnist": "Yang J, et al. (2023) MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data",
    "kermany": "Kermany DS, et al. (2018) Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell",
    "pets": "Parkhi OM, Vedaldi A, Zisserman A, Jawahar CV (2012) The Oxford-IIIT Pet dataset. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://www.robots.ox.ac.uk/~vgg/data/pets/",
    # controlled reliance and ground-truth benchmarks
    "ross": "Ross AS, Hughes MC, Doshi-Velez F (2017) Right for the right reasons: training differentiable models by constraining their explanations. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)",
    "bam": "Yang M, Kim B (2019) Benchmarking attribution methods with relative feature importance. arXiv:1907.09701",
    "smerf": "Kim JS, Plumb G, Talwalkar A (2022) Sanity simulations for saliency methods. In: Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162:11173\u201311200",
    "bastings": "Bastings J, Ebert S, Zablotskaia P, Sandholm A, Filippova K (2022) \u201cWill you find these shortcuts?\u201d A protocol for evaluating the faithfulness of input salience methods for text classification. In: Proceedings of EMNLP 2022, pp 976\u2013991",
    "zhou": "Zhou Y, et al. (2022) Do feature attribution methods correctly attribute features? In: Proceedings of the AAAI Conference on Artificial Intelligence. arXiv:2104.14403",
    "lin": "Lin Y-S, Lee W-C, Celik ZB (2021) What do you see? Evaluation of explainable artificial intelligence (XAI) interpretability through neural backdoors. In: Proceedings of KDD 2021. arXiv:2009.10639",
    "backx": "A backdoor-based explainable AI benchmark for high fidelity evaluation of attributions (BackX). arXiv:2405.02344",
    "adebayo2022": "Adebayo J, et al. (2022) Post hoc explanations may be ineffective for detecting unknown spurious correlation. In: International Conference on Learning Representations (ICLR)",
    "sun": "Sun S, et al. (2023) Right for the wrong reason: can interpretable ML techniques detect spurious correlations? In: Medical Image Computing and Computer Assisted Intervention (MICCAI). arXiv:2307.12344",
    "arras": "Arras L, Osman A, Samek W (2022) CLEVR-XAI: a benchmark dataset for the ground truth evaluation of neural network explanations. Information Fusion",
    "funnybirds": "Hesse R, Schaub-Meyer S, Roth S (2023) FunnyBirds: a synthetic vision dataset for a part-based analysis of explainable AI methods. In: Proceedings of ICCV 2023. arXiv:2308.06248",
    "xaiunits": "Lee JR, Emami S, Hollins MD, Wong TCH, Villalobos S\u00e1nchez CI, Toni F, Zhang D, Dejl A (2025) XAI-Units: benchmarking explainability methods with unit tests. In: Proceedings of ACM FAccT 2025. arXiv:2506.01059",
    "attributionlab": "AttributionLab: faithfulness of feature attribution under controllable environments (2023). NeurIPS XAI in Action Workshop. arXiv:2310.06514",
    "miro2026": "Mir\u00f3-Nicolau M, Spinnato F, Guidotti R (2026) A synthetic ground-truth framework for the evaluation of explainable AI methods. arXiv:2609.30397",
    "oscar": "Achara A, Triantafillou P, Puyol-Ant\u00f3n E, Hammers A, King AP (2025) Localising shortcut learning in pixel space via ordinal scoring correlations for attribution representations (OSCAR). arXiv:2512.18888",
    # meta-evaluation of metrics
    "tomsett": "Tomsett R, et al. (2020) Sanity checks for saliency metrics. In: Proceedings of the AAAI Conference on Artificial Intelligence. arXiv:1912.01451",
    "metaquantus": "Hedstr\u00f6m A, Bommer P, Wickstr\u00f8m KK, Samek W, Lapuschkin S, H\u00f6hne MMC (2023) The meta-evaluation problem in explainable AI: identifying reliable estimators with MetaQuantus. Transactions on Machine Learning Research. arXiv:2302.07265",
    "quantus": "Hedstr\u00f6m A, Weber L, Krakowczyk D, Bareeva D, Motzkus F, Samek W, Lapuschkin S, H\u00f6hne MMC (2023) Quantus: an explainable AI toolkit for responsible evaluation of neural network explanations and beyond. Journal of Machine Learning Research 24(34):1\u201311",
    "latec": "Klein L, L\u00fcth CT, Schlegel U, Bungert TJ, El-Assady M, J\u00e4ger PF (2024) Navigating the maze of explainable AI: a systematic approach to evaluating methods and metrics. In: Advances in Neural Information Processing Systems (NeurIPS). arXiv:2409.16756",
    "m4": "M4: a unified XAI benchmark for faithfulness evaluation of feature attribution methods across metrics, modalities and models (2023). In: NeurIPS 2023 Datasets and Benchmarks Track",
    "gevaert": "Gevaert A, Rousseau A-J, Becker T, Valkenborg D, De Bie T, Saeys Y (2022) Evaluating feature attribution methods in the image domain. arXiv:2202.12270",
    "fidelity": "Mir\u00f3-Nicolau M, Jaume-i-Cap\u00f3 A, Moy\u00e0-Alcover G (2024) A comprehensive study on fidelity metrics for XAI. arXiv:2401.10640",
    "hesse2024": "Hesse R, Schaub-Meyer S, Roth S (2024) Benchmarking the attribution quality of vision models. In: NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2407.11910",
    "binder": "Binder A, et al. (2023) Shortcomings of top-down randomization-based sanity checks for evaluations of deep neural network explanations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    "sanityrevisited": "Hedstr\u00f6m A, Weber L, Lapuschkin S, H\u00f6hne MMC (2023) Sanity checks revisited: an exploration to repair the model parameter randomisation test. NeurIPS 2023 Workshop XAI in Action (XAIA). arXiv:2401.06465",
    "freshlook": "Hedstr\u00f6m A, Weber L, Lapuschkin S, H\u00f6hne M (2024) A fresh look at sanity checks for saliency maps. arXiv:2405.02383 (text overlaps with arXiv:2401.06465)",
    # confounds of perturbation metrics
    "rong": "Rong Y, Leemann T, Borisov V, Kasneci G, Kasneci E (2022) A consistent and efficient evaluation strategy for attribution methods (ROAD). In: Proceedings of the International Conference on Machine Learning (ICML). arXiv:2202.00449",
    "roar": "Hooker S, Erhan D, Kindermans P-J, Kim B (2019) A benchmark for interpretability methods in deep neural networks. In: Advances in Neural Information Processing Systems (NeurIPS). arXiv:1806.10758",
    "roarpitfalls": "Song J, Cha K, Seo J (2026) On pitfalls of RemOve-And-Retrain: data processing inequality perspective. ICML 2026 Workshop on Mechanistic Interpretability. arXiv:2304.13836",
    "blucher": "Bl\u00fccher S, Vielhaben J, Strodthoff N (2024) Decoupling pixel flipping and occlusion strategy for consistent XAI benchmarks. Transactions on Machine Learning Research. arXiv:2401.06654",
    "baseline": "Back to the baseline: examining baseline effects on explainability metrics (2025). arXiv:2512.11433",
    "gomez": "Gomez T, Fr\u00e9our T, Mouch\u00e8re H (2022) Metrics for saliency map evaluation of deep learning explanation methods. arXiv:2201.13291",
    "saco": "Wu J, Kang W, Tang H, Hong Y, Yan Y (2024) On the faithfulness of vision transformer explanations. In: Proceedings of CVPR 2024. arXiv:2404.01415",
    "bilodeau": "Bilodeau B, Jaques N, Koh PW, Kim B (2024) Impossibility theorems for feature attribution. Proceedings of the National Academy of Sciences 121(2). arXiv:2212.11870",
    # shortcuts in practice
    "lapuschkin": "Lapuschkin S, W\u00e4ldchen S, Binder A, Montavon G, Samek W, M\u00fcller K-R (2019) Unmasking Clever Hans predictors and assessing what machines really learn. Nature Communications. arXiv:1902.10178",
    "degrave": "DeGrave AJ, et al. (2021) AI for radiographic COVID-19 detection selects shortcuts over signal. Author manuscript, PubMed Central PMC7523163 [final venue to be confirmed from the published version]",
    "hermann": "Hermann KL, Mobahi H, Fel T, Mozer MC (2024) On the foundations of shortcut learning. In: International Conference on Learning Representations (ICLR). arXiv:2310.16228",
    "suhail": "Suhail P, Goel V, Sethi A (2025) Shortcut learning susceptibility in vision classifiers. arXiv:2502.09150",
    # statistics
    "bouthillier": "Bouthillier X, et al. (2021) Accounting for variance in machine learning benchmarks. In: MLSys 2021. arXiv:2103.03098",
    "agarwal": "Agarwal R, Schwarzer M, Castro PS, Courville A, Bellemare MG (2021) Deep reinforcement learning at the edge of the statistical precipice. In: Advances in Neural Information Processing Systems (NeurIPS). arXiv:2108.13264",
    "saravanan": "Saravanan V, Berman GJ, Sober SJ (2020) Application of the hierarchical bootstrap to multi-level data in neuroscience. Neurons, Behavior, Data analysis, and Theory 3(5). arXiv:2007.07797",
}
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


def rearrange_panels(source, target, panels, columns=4):
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
    title = image.crop((0, 0, a.shape[1], title_end))
    t_ink = np.flatnonzero((np.asarray(title) < 245).any(axis=2).any(axis=0))
    title = title.crop((max(t_ink.min() - 10, 0), 0, t_ink.max() + 10, title.height))
    cell_w, cell_h = max(p.width for p in pieces), max(p.height for p in pieces)
    strip_w = strip.width if strip is not None else 0
    width = max(strip_w + columns * cell_w, title.width)
    grid_rows = int(np.ceil(panels / columns))
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
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharey=True)
    colours = ["C0", "C1", "C2", "C3"]
    for ax, (base, shortcut, label, _) in zip(axes.flat, SETTINGS):
        sel = PREF[(PREF.comparison == comparison) & (PREF.base == base) & (PREF.shortcut == shortcut)]
        assert len(sel) == N_SETTING[(base, shortcut)], (comparison, base, shortcut)
        for metric, colour in zip(METRICS, colours):
            x, y = sel.reliance_corrected.to_numpy(), sel[METRIC_COLUMNS[metric]].to_numpy()
            ax.scatter(x, y, s=9, alpha=0.35, color=colour, linewidths=0)
            ux, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
            mean = np.bincount(inverse, weights=y) / counts
            ax.plot(ux, pava(mean, counts), color=colour, lw=1.6, label=metric)
        ax.axhline(0.5, color="grey", lw=0.7)
        ax.axvline(50, color="grey", lw=0.7, ls=":")
        if show_behaviour:
            sw = behavioural_switch(base, shortcut)
            if not pd.isna(sw):
                ax.axvline(sw, color="black", lw=0.9, ls="--")
        ax.set_title(label, fontsize=9)
        ax.set_xlim(-5, 103)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel(f"Chance-corrected reliance {RC} (%)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8.5)
    legend_ax = axes.flat[-1]
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="grey", ls=":"), plt.Line2D([], [], color="black", ls="--")]
    labels += ["assumed ideal switch (50%)", "behavioural switch (follow rate 0.5)"]
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


# ------------------------------------------------------------------ document helpers

doc = Document()
COUNTERS = {"fig": 0, "table": 0}


def setup():
    """Single-column Elsevier-style manuscript: A4, 2.5 cm margins, Times New Roman
    12 pt body text with 1.5 line spacing (tables and captions single-spaced)."""
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
    header = section.header.paragraphs[0]
    header.text = DRAFT_NOTICE
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor(0xC0, 0, 0)


def add_runs(p, text, bold=False, italic=False, size=None):
    """Text with **bold** and _italic_ spans."""
    import re
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


def figure(path, caption, width=Inches(6.2)):
    COUNTERS["fig"] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.add_run().add_picture(str(path), width=width)
    p.paragraph_format.keep_with_next = True
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.line_spacing = 1.0
    r = c.add_run(f"Fig. {COUNTERS['fig']}. ")
    r.bold = True
    r.font.size = Pt(9)
    add_runs(c, caption, size=9)
    return COUNTERS["fig"]


def shade(cell, color="D9D9D9"):
    props = cell._tc.get_or_add_tcPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), color)
    props.append(fill)


def table(caption, header, body, widths=None, note=None, size=8, header_rows=None):
    """A numbered table: caption above (Elsevier style), optional note below."""
    COUNTERS["table"] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.keep_with_next = True
    c.paragraph_format.line_spacing = 1.0
    r = c.add_run(f"Table {COUNTERS['table']}. ")
    r.bold = True
    r.font.size = Pt(9)
    add_runs(c, caption, size=9)
    headers = header_rows or [header]
    t = doc.add_table(rows=len(headers) + len(body), cols=len(headers[-1]))
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
                shade(cell)
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
    return COUNTERS["table"]


# ------------------------------------------------------------------ table builders


def table_positioning():
    body = [
        ["Inject a shortcut so the evidence is known",
         "Decoy MNIST " + cite("ross") + "; Zhou et al. " + cite("zhou") + "; Lin et al. " + cite("lin")
         + "; BackX " + cite("backx") + "; Adebayo et al. " + cite("adebayo2022") + "; Sun et al. " + cite("sun")
         + "; Bastings et al. (text) " + cite("bastings"),
         "The object of evaluation is the metric, not the attribution method"],
        ["Graded or controlled model reliance",
         "BAM: models trained with known relative feature importance " + cite("bam") + "; SMERF: controlled "
         "model reasoning " + cite("smerf") + "; OSCAR: shortcut scores that react to the strength of the "
         "shortcut association " + cite("oscar"),
         "A continuous signed dose, with reliance measured behaviourally from about 0% to 100% and chance-corrected"],
        ["Meta-evaluation of faithfulness metrics",
         "Tomsett et al. " + cite("tomsett") + "; MetaQuantus (no ground truth) " + cite("metaquantus")
         + "; fidelity metrics vs a decision tree " + cite("fidelity") + "; synthetic intervention-based ground "
         "truth " + cite("miro2026") + "; Gevaert et al. " + cite("gevaert") + "; LATEC " + cite("latec")
         + "; M4 " + cite("m4"),
         "A ground truth that changes with the model, read out as responsiveness, over-detection, a switch "
         "point and per-image AUROC"],
        ["Shape, size and fill confounds",
         "ROAD " + cite("rong") + "; ROAR pitfalls " + cite("roarpitfalls") + "; Bl\u00fccher et al. "
         + cite("blucher") + "; Back to the Baseline " + cite("baseline") + "; Gomez et al. " + cite("gomez"),
         "Size-matched, placebo and random oracle maps that show the confounds as reliance varies; real masks on Pet"],
        ["Pre-registration",
         "In our searches we did not find an XAI-evaluation benchmark that pre-registers its predictions",
         "Pre-registered Pet predictions and a dated amendment, with commit hashes"],
    ]
    return table(
        "Positioning. What prior work already provides and what this study adds (after the audit's "
        "positioning table, docs/audit/02_literature_novelty.md \u00a72)",
        ["Ingredient", "Prior work that already has it", "What this study adds"],
        body, widths=[3.2, 7.0, 5.8], size=8,
        note="The protocol is new as a combination; its individual pieces are not. BAM and SMERF already "
             "controlled what a model relies on, to evaluate attribution methods.")


def table_reliance():
    body = []
    for base, shortcut, label, _ in SETTINGS:
        g, lv = group(base, shortcut), levels(base, shortcut)
        lowest, highest = lv.loc[lv.reliance.idxmin()], lv.loc[lv.reliance.idxmax()]
        a1 = int((g.model_set == "A1").sum())
        count = f"{len(g)} ({lv.dose.nunique()} doses)" + (f", incl. {a1} A1" if a1 else "")
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
        note=f"Source: results/dose_response/models.csv ({N_MODELS} models: {N_ORIGINAL} original and {N_A1} "
             f"from amendment A1). R: percentage of conflicting test images whose prediction follows the "
             f"shortcut. {RC} = 100 (R \u2212 f)/(100 \u2212 f) with chance floor f = (100 \u2212 accuracy "
             "without shortcut)/(C \u2212 1); it is not clipped and can be slightly negative.")


def table_main(comparison, caption):
    body = []
    for base, shortcut, label, short in SETTINGS:
        for i, metric in enumerate(METRICS):
            r = row(comparison, base, shortcut, metric)
            body.append([
                short if i == 0 else "", metric,
                f"{f2(r.rho_rc)}{sig(r)} {ci(r.rho_rc_ci_low, r.rho_rc_ci_high)}",
                f"{f2(r.pref_low)} {ci(r.pref_low_ci_low, r.pref_low_ci_high)} \u2192 {f2(r.pref_high)}",
                f2(r.median_gap_low),
                f"{f2(r.fpr)} ({int(r.n_below)})",
                f2(r.balanced_accuracy),
                fmt_switch(r),
            ])
    return table(
        caption,
        ["Setting", "Metric", f"\u03c1 with {RC} [95% CI]", "Preference, lowest dose [CI] \u2192 highest",
         "Median AUC gap, lowest dose", "Over-detection FPR (n models)", "Balanced accuracy",
         f"Switch, {RC} % [CI]"],
        body, widths=[2.0, 2.2, 2.4, 2.9, 1.5, 1.6, 1.4, 2.6], size=7.5,
        note=f"Source: results/dose_response/responsiveness_corrected.csv, comparison \u201c{comparison}\u201d. "
             "CI: two-level bootstrap (2000 replicates; models resampled within dose, images within model). "
             "\u2020: not significant after Benjamini\u2013Hochberg within the comparison family (p_bh \u2265 0.05). "
             f"Over-detection FPR: share of models with {RC} \u2264 50 for which the preference exceeds 0.5 "
             "(n = number of such models). Median AUC gap: positive means the shortcut map is rated better. "
             "Switch: first crossing of 0.5 by the increasing isotonic fit; (e) identified at the lowest-reliance "
             "edge (within 5 points of the lowest model); (nb) not bracketed by models within \u00b110 points "
             "on both sides; \u201cnone (> 0.5 at lowest)\u201d: the metric already prefers the shortcut map at "
             "the lowest reliance reached.")


def table_prereg():
    body = []
    for r in PET.itertuples():
        body.append([r.prediction, r.comparison, r.metric_or_method, r.statistic,
                     f"{f3(r.estimate)} {ci(r.ci_low, r.ci_high, f3)}",
                     "\u2013" if pd.isna(r.rho_rc) else f3(r.rho_rc), r.threshold, r.original_verdict])
    return table(
        "Pre-registered Pet predictions P1\u2013P4 with their original verdicts. The intervals and the "
        f"\u03c1 on {RC} were computed later and are additional information",
        ["Pred.", "Comparison", "Metric or method", "Statistic", "Estimate [two-level 95% CI]",
         f"\u03c1 on {RC}", "Threshold", "Original verdict"],
        body, widths=[1.0, 2.4, 2.4, 2.4, 2.6, 1.2, 1.8, 2.6], size=7,
        note="Source: results/dose_response/pet_preregistered.csv; the verdict strings are copied from the "
             "Outcome table of docs/PREDICTIONS_pet.md. Verdicts were judged on point estimates, as "
             "pre-registered; the pre-registration did not say whether a point estimate or an interval decides.")


def table_behaviour():
    body = []
    for base, shortcut, label, short in SETTINGS:
        for i, metric in enumerate(METRICS):
            b = brow(FULL, base, shortcut, metric)
            body.append([
                short if i == 0 else "", metric,
                f"{f2(b.auroc_pooled_strat)} {ci(b.auroc_strat_ci_low, b.auroc_strat_ci_high)}",
                f"{f2(b.median_model_auroc_strat)} {ci(b.median_model_auroc_strat_ci_low, b.median_model_auroc_strat_ci_high)}",
                f2(b.calibration_gap),
                fmt_switch(b, "switch_on_follow_rate", "switch_on_follow_rate_status", None, None,
                           "switch_on_follow_rate_bracketed"),
                f1(b.observed_switch_rc) if i == 0 else "",
            ])
    return table(
        "Per-image behavioural test (shortcut vs full object). Each scored image is labelled by forward "
        "passes as followed or not followed; the score is the per-image AUC gap",
        ["Setting", "Metric", "Pooled AUROC [CI]", "Within-model median AUROC [CI]",
         "Calibration gap", "Metric switch on follow rate (%)", f"Behavioural switch, {RC} (%)"],
        body, widths=[2.0, 2.3, 2.6, 2.8, 1.6, 2.6, 2.2], size=7.5,
        note=f"Source: results/dose_response/behaviour_summary.csv, label {PRIMARY_LABEL}. AUROCs are "
             "class-stratified (computed within each true class and averaged) and tested one-sided against "
             "0.5. Pooled: the images of all models together, which mixes between- and within-model signal. "
             "Within-model: AUROC inside each model, median over models. Calibration gap: mean of "
             "(preference \u2212 follow rate) over models; positive values mean over-detection. Metric switch on "
             "follow rate: where the isotonic fit of preference against 100 \u00d7 follow rate crosses 0.5 "
             "(ideal 50 by construction). Behavioural switch: the "
             f"{RC} at which the models' follow rate crosses 0.5, the derived ideal that replaces the assumed 50%.")


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
    sel = PAIRS[PAIRS.comparison.isin([FULL, MATCHED, R_OBJECT]) & (PAIRS.p_bh < 0.05)]
    order = {(b, s): i for i, (b, s, *_) in enumerate(SETTINGS)}
    sel = sel.assign(o=[order[(b, s)] for b, s in zip(sel.base, sel.shortcut)]).sort_values(["comparison", "o"])
    body = [[r.comparison, SHORT[(r.base, r.shortcut)], r.metric_a, r.metric_b,
             f"{f3(r.delta_rho_rc)} {ci(r.delta_rho_rc_ci_low, r.delta_rho_rc_ci_high, f3)}",
             format_p(r.p_bh), format_p(r.p_holm),
             f"{f3(r.delta_pref_low)} {ci(r.delta_pref_low_ci_low, r.delta_pref_low_ci_high, f3)}"]
            for r in sel.itertuples()]
    n_tested = len(PAIRS[PAIRS.comparison.isin([FULL, MATCHED, R_OBJECT])])
    return table(
        f"Paired metric-vs-metric differences that are significant after Benjamini\u2013Hochberg ({len(sel)} of "
        f"{n_tested} pairs tested)",
        ["Comparison", "Setting", "Metric A", "Metric B", f"\u0394\u03c1 on {RC} (A \u2212 B) [CI]", "p_bh", "p_holm",
         "\u0394 preference, lowest dose [CI]"],
        body, widths=[2.6, 1.9, 2.0, 2.0, 2.8, 1.1, 1.1, 2.8], size=7,
        note="Source: results/dose_response/metric_pairs.csv. Paired two-level bootstrap (the same resamples "
             "for both metrics); two-sided bootstrap p, adjusted within each comparison family (BH and Holm).")


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
    for (base, shortcut), g in CONF.groupby(["base", "shortcut"], sort=False):
        first = g.iloc[0]
        values = [SHORT[(base, shortcut)], str(int(first.models)), f2(first.rho_reliance_competence)]
        for metric in METRICS:
            r = crow(base, shortcut, metric)
            values.append(f"{f2(r.rho_rel)} / {f2(r.partial_rel_given_acc)} ({format_p(r.p_partial_rel_given_acc)})")
        body.append(values)
    return table(
        "Reliance and object competence. \u03c1(R, acc): Spearman correlation between reliance and accuracy "
        "without the shortcut. Per metric (shortcut vs full object): \u03c1 with reliance / partial \u03c1 with "
        "reliance controlling for accuracy without the shortcut (p)",
        ["Setting", "Models", "\u03c1(R, acc)"] + METRICS,
        body, widths=[2.2, 1.2, 1.4, 2.8, 2.8, 2.8, 2.8], size=7.5,
        note="Source: results/dose_response/confound.csv, sorted from the least collinear setting. Partial "
             "Spearman: Pearson correlation of rank residuals after a linear fit on the accuracy rank; p from "
             "a t distribution with n \u2212 3 degrees of freedom.")


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
        "accuracy and the first-crossing switch rule, kept for comparison with earlier drafts",
        ["Setting", "Metric", "\u03c1 with raw R [seed-only CI]", "p (unadjusted)", "Decision accuracy (unbalanced)",
         "Switch, raw R % [CI]"],
        body, widths=[2.2, 2.4, 3.2, 2.0, 2.4, 3.2], size=7.5,
        note="Source: results/dose_response/responsiveness.csv (legacy). These intervals resample only the 2\u20136 "
             "seeds per dose with the images fixed, so they are too narrow; the decision accuracy is "
             "unbalanced; the switch is the first crossing of the per-dose means. Superseded by Tables in "
             "Section 5.")


def table_sets():
    body = []
    for base, shortcut, label, short in MECHANISM:
        for i, metric in enumerate(METRICS):
            a, o = row(FULL, base, shortcut, metric), row(FULL, base, shortcut, metric, RESP_ORIG)
            body.append([short if i == 0 else "", metric, f"{f3(o.rho_rc)} \u2192 {f3(a.rho_rc)}",
                         f"{f3(o.fpr)} \u2192 {f3(a.fpr)}", f"{f3(o.balanced_accuracy)} \u2192 {f3(a.balanced_accuracy)}",
                         f"{fmt_switch(o, lo=None)} \u2192 {fmt_switch(a, lo=None)}"])
    return table(
        f"Model-set check promised by amendment A1 (shortcut vs full object): the {N_ORIGINAL} original models "
        f"\u2192 all {N_MODELS} models",
        ["Setting", "Metric", f"\u03c1 on {RC}", "Over-detection FPR", "Balanced accuracy", f"Switch, {RC} %"],
        body, widths=[2.2, 2.4, 2.6, 2.4, 2.6, 3.8], size=7.5,
        note="Sources: results/dose_response/original_models/responsiveness_corrected.csv and "
             "results/dose_response/responsiveness_corrected.csv. Pet and CIFAR-10 received no new models, so "
             "their rows are identical in both sets and are omitted.")


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


def build():
    setup()
    figs = {
        "full": preference_figure(FULL, FIG_OUT / "fig_rc_full_object.png", "Preference for shortcut map"),
        "matched": preference_figure(MATCHED, FIG_OUT / "fig_rc_size_matched.png", "Preference for shortcut map"),
        "s_random": preference_figure(S_RANDOM, FIG_OUT / "fig_rc_shortcut_vs_random.png",
                                      "Preference for shortcut map", show_behaviour=False),
        "r_object": preference_figure(R_OBJECT, FIG_OUT / "fig_rc_random_vs_object.png",
                                      "Preference for random map", show_behaviour=False),
        "p_random": preference_figure(P_RANDOM, FIG_OUT / "fig_rc_placebo_vs_random.png",
                                      "Preference for placebo map", show_behaviour=False),
        "behaviour": rearrange_panels(DR / "behaviour_auroc.png", FIG_OUT / "fig_behaviour_auroc.png", N_SETTINGS),
        "methods": rearrange_panels(DR / "method_dose_response.png", FIG_OUT / "fig_methods.png", N_SETTINGS),
    }

    # ---- numbers used throughout (all from the CSVs)
    full = rows([FULL], SETTINGS)
    full_nat = rows([FULL], NATURAL)
    n_full_sig, n_full = sig_positive(full), len(full)
    full_weak = full[~((full.rho_rc > 0) & (full.p_bh < 0.05))]
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
    n_pairs_sig = int((PAIRS.p_bh < 0.05).sum())
    pt_black_sr = row(S_RANDOM, "pneumonia_mnist", "texture", "Deletion (black)")
    pt_black_sp = row(S_PLACEBO, "pneumonia_mnist", "texture", "Deletion (black)")
    pt_black_m = row(MATCHED, "pneumonia_mnist", "texture", "Deletion (black)")
    mt_ins = row(FULL, "mnist", "texture", "Insertion (blur)")
    mt_sr_blur = row(S_RANDOM, "mnist", "texture", "Deletion (blur)")
    mt_sr_ins = row(S_RANDOM, "mnist", "texture", "Insertion (blur)")
    mt_sp_ins = row(S_PLACEBO, "mnist", "texture", "Insertion (blur)")
    ig = METHOD[METHOD.method == "Integrated Gradients"].rho_rc
    mt_away = {m: mrow("mnist", "texture", m) for m in ["Saliency", "SmoothGrad", "Grad-CAM"]}
    pt_meth = {m: mrow("pneumonia_mnist", "texture", m) for m in ["Saliency", "SmoothGrad", "Grad-CAM"]}
    assert all(r.rho_rc < 0 for r in mt_away.values()) and pt_meth["Saliency"].rho_rc > 0 \
        and pt_meth["SmoothGrad"].rho_rc > 0 and pt_meth["Grad-CAM"].rho_rc < 0
    gc_lowest = sum(min(METHODS, key=lambda m: mrow(b, s, m).mass_high) == "Grad-CAM" for b, s, *_ in sf_patch_settings)
    conf_first = CONF.iloc[0]
    assert (conf_first.base, conf_first.shortcut) == ("cifar10", "patch"), "text leads with the CIFAR CNN"
    conf_cifar = CONF[(CONF.base == "cifar10")]
    conf_set = CONF.groupby(["base", "shortcut"]).rho_reliance_competence.first()
    collinear = conf_set[conf_set <= -0.95]
    # Guards for the qualitative statements in the text.
    assert all(45 <= beh_switch[k] <= 60 for k in [("pet_resnet18", "patch"), ("cifar10", "patch"),
                                                     ("cifar10_resnet18", "patch"), ("mnist", "patch")])
    assert all(beh_switch[("pneumonia_mnist", s)] < 45 for s in ("patch", "texture"))
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
    kmn = KENDALL.set_index(["dataset", "metric"])
    sanity = [kmn.loc[(d, "Sanity (object px)")] for d in ("decoy_mnist", "decoy_pneumonia")]
    tau_threshold = KENDALL["min_tau_one_sided_p_below_0.05"].iloc[0]
    gt_tau = GT_TAU.groupby(["dataset", "metric"]).tau_with_ground_truth.mean()
    del_tau_mnist = [gt_tau[("decoy_mnist", m)] for m in METRICS]
    del_tau_pneu = [gt_tau[("decoy_pneumonia", m)] for m in METRICS]
    inst = GT_INSTANCE.set_index(["dataset", "metric"]).spearman_with_ground_truth
    del_inst = [inst[(d, m)] for d in ("decoy_mnist", "decoy_pneumonia") for m in METRICS[:3]]
    ins_inst = [inst[(d, "Insertion (blur)")] for d in ("decoy_mnist", "decoy_pneumonia")]

    # ---- front matter
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
    para("Md Junaid Faizan Shaikh", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    para("Department of Computer Science and Engineering, Khaja Bandanawaz University, Kalaburagi, India",
         align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=0)
    para("E-mail: [email]", align=WD_ALIGN_PARAGRAPH.CENTER, size=10)

    abstract = (
        "Deletion and insertion curves are used to decide which explanation of an image classifier is faithful, "
        "but whether their verdicts follow what a model relies on is rarely tested. We trained "
        f"{N_MODELS} CNNs in which a signed training dose sets the measured reliance on an injected class-coded "
        "shortcut, on Oxford-IIIT Pet with real object masks, CIFAR-10, MNIST and "
        "PneumoniaMNIST, and scored fixed oracle explanations with four metric variants. The metrics respond to "
        "reliance: the preference for the shortcut map over the object map rises with chance-corrected reliance "
        f"in {n_full_sig} of {n_full} metric\u2013setting cells (Spearman \u03c1 = {span(full.rho_rc)}). They are "
        "not calibrated to it. Among models that follow the shortcut on at most half of the conflicting images, "
        f"black and blur deletion still prefer the shortcut map for {pct_span(pet_bb_fpr)} of Pet models and "
        f"{pct_span(cifar_bb_fpr)} of CIFAR-10 models, and on Pet the metrics switch at "
        f"{span(pet_switch, f1)}% chance-corrected reliance, while the models follow the "
        f"shortcut on half of the images only at {f1(beh_switch[('pet_resnet18', 'patch')])}%. Against per-image "
        f"behavioural labels, the metrics reach a class-stratified AUROC of {span(pooled_nat)} "
        f"on natural images, but within a model only {span(within_nat)}. Their verdicts depend on "
        "explanation size and form: a size-matched object map moves the MNIST switch points from "
        f"{span(mp_full.switch_iso_rc, f1)}% to {span(mp_matched.switch_iso_rc, f1)}%, and on CIFAR-10 black "
        f"deletion prefers a random ordering over a centred object map {pct(ro[('cifar10', 'Deletion (black)')].pref_low)} "
        "of the time at low reliance. A pre-registered test on Pet confirmed three of four "
        "predictions; black deletion failed the form-matched control.")
    assert word_count(abstract) <= 250, word_count(abstract)
    p = para("")
    add_runs(p, "**Abstract** " + abstract)
    highlights = [
        f"{N_MODELS} CNNs with measured, chance-corrected shortcut reliance test four metrics",
        "Deletion and insertion track reliance but over-detect shortcuts models barely use",
        "Per-image behavioural labels replace the assumed 50% switch point",
        "Explanation size and form change which map the metrics prefer",
        "Pre-registered Pet test: three of four predictions held; black deletion failed",
    ]
    assert all(len(h) <= 85 for h in highlights), [len(h) for h in highlights]
    para("**Highlights** (at most 85 characters each, following the usual Elsevier rule; confirm in the "
         "Neurocomputing Guide for Authors)", after=0)
    bullets(highlights)
    para("**Keywords** Explainable AI \u00b7 Faithfulness metrics \u00b7 Deletion and insertion \u00b7 "
         "Shortcut learning \u00b7 Meta-evaluation \u00b7 Pre-registration")

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
        f"(over-detection rate {span(nat_bb_fpr)}), the metrics' switch points lie well below the reliance at "
        "which the models follow the shortcut on half of the images, and within a single model the metrics "
        "separate followed from non-followed images only weakly.",
        "**Size and form dependence.** Matching the object map's size to the shortcut moves the switch points "
        "towards 0%, and on CIFAR-10 black and blur deletion prefer a random pixel ordering over a centred "
        "object map. Real masks on Pet reduce this form bias but do not remove the over-detection.",
        "**A pre-registered test and a pre-registered amendment.** The Pet predictions P1\u2013P4 were committed "
        f"before training; {N_A1} models that fill reliance gaps were added under a dated amendment, and both "
        "model sets are reported.",
    ])
    para("Section 2 positions the work, Sections 3 and 4 describe the protocol and setup, Section 5 reports the "
         "results (natural images first), and Sections 6\u20138 discuss them, state the limitations and conclude. "
         "Appendix A holds a supporting study of the model-randomisation sanity check, Appendix B the legacy "
         "analysis of earlier drafts, and Appendix C the check on the original model set.")

    # ---- 2 Related work
    heading("2. Related work")
    para(f"The literature audit behind this section searched {SEARCH_AUDIT}; every cited page was re-opened on "
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
         "varied in steps and measured.")
    heading("2.2. Meta-evaluation of faithfulness metrics", 2)
    para("Tomsett et al. " + cite("tomsett") + " found saliency metrics statistically unreliable. MetaQuantus "
         + cite("metaquantus") + " rates metrics by their resilience to noise and reactivity to randomness "
         "without ground truth, and Quantus " + cite("quantus") + " collects many metrics in one toolkit. LATEC "
         + cite("latec") + " evaluates 17 methods with 20 metrics and finds conflicting rankings; M4 "
         + cite("m4") + " benchmarks faithfulness across modalities and networks. Gevaert et al. " + cite("gevaert")
         + " report that metrics measure different properties and that results on one dataset need not hold on "
         "another. Mir\u00f3-Nicolau et al. " + cite("fidelity") + " check fidelity metrics against a decision tree "
         "on 52,000 synthetic 128\u00d7128 images, and Hesse et al. " + cite("hesse2024") + " propose a protocol "
         "that avoids the out-of-domain problem of incremental deletion. The model-randomisation sanity check "
         + cite("adebayo2018") + " has been criticised " + cite("binder") + " and revised "
         + cite("sanityrevisited", "freshlook") + ". None of these varies the model's reliance and reads the "
         "metrics' response to it.")
    heading("2.3. Known confounds of perturbation metrics", 2)
    para("ROAD " + cite("rong") + " shows that pixel-perturbation results are driven by information leaking "
         "through the shape of the removed region; ROAR " + cite("roar") + " retrains after removal, and a later "
         "analysis traces ROAR's and ROAD's rankings to a bias towards spatially blurry masks "
         + cite("roarpitfalls") + ". Bl\u00fccher et al. " + cite("blucher") + " show that the occlusion "
         "strategy decides pixel-flipping rankings, and \u201cBack to the Baseline\u201d " + cite("baseline")
         + " that the replacement baseline favours some methods. Gomez et al. " + cite("gomez") + " note that "
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
        "pixels. On the grayscale datasets, the image's own intensities without the shortcut pixels; on "
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
         "scored image from forward passes only. The primary label (after the audit's reviewer 05) is 1 if the "
         "model predicts the shortcut's class when the same image's shortcut is moved to a wrong class's "
         "position and does not predict that class when the shortcut is removed; it is 0 if the moved shortcut "
         "does not win; images that the model already assigns to the shortcut's class without any shortcut are "
         "left undetermined, which removes the same error floor as the chance correction. A check label "
         "(reviewer 03) marks an image as shortcut-dominant when removing the shortcut lowers the true-class "
         "probability more than replacing the image by one of another class that carries the same shortcut. "
         "Each metric's AUC gap is then scored against the label by an AUROC computed within each true class "
         "and averaged (class-stratified), because on the binary tasks the label and the gap both depend on "
         "the class. The **behavioural switch** is the reliance at which the models' follow rate crosses 0.5; "
         "it replaces the assumed 50% ideal.")
    heading("3.7. Statistics", 2)
    bullets([
        f"**Responsiveness**: Spearman \u03c1 between {RC} and preference over the models of a setting.",
        f"**Decisions**: a model counts as relying on the shortcut when {RC} > 50, and the metric \u201csays "
        "shortcut\u201d when its preference exceeds 0.5. We report the **over-detection rate** (the false-positive "
        f"rate among models with {RC} \u2264 50) and balanced accuracy.",
        f"**Switch point**: the first {RC} at which an increasing isotonic fit of preference reaches 0.5; a "
        "four-parameter logistic with free asymptotes is the sensitivity check. A switch is \u201cidentified\u201d "
        "only if a model lies within \u00b110 points of it; it is reported only for the comparisons whose ideal "
        "curve crosses 0.5 (shortcut vs full or size-matched object).",
        "**Uncertainty**: a two-level bootstrap with 2000 replicates that resamples models within each dose and "
        "the 100 images within each model " + cite("saravanan") + ", following the advice to account for "
        "both sources of variance in machine-learning comparisons " + cite("bouthillier", "agarwal") + ". The "
        "same resamples are reused for every metric, so metric-versus-metric differences are paired.",
        "**Multiplicity**: Benjamini\u2013Hochberg within each family (each comparison, the methods, the metric "
        "pairs, each behavioural label), and Holm over the union of the headline tests. Correlation p-values "
        "use Spearman's t approximation; p-values below 0.001 are printed as p < 0.001.",
        "**Effect size**: the median per-image AUC gap at the lowest and the highest dose.",
    ])
    heading("3.8. Explanation methods", 2)
    para("Five methods are run on every model with Captum " + cite("captum") + ": Saliency (absolute gradient), "
         "SmoothGrad (25 samples, noise 0.15), Integrated Gradients (50 steps, black baseline), Grad-CAM on the "
         "last convolutional feature maps (6\u00d76 for the 28\u00d728 CNN, 16\u00d716 for the CIFAR CNN, 4\u00d74 "
         "for the CIFAR ResNet-18 and 16\u00d716 for the Pet ResNet-18, upsampled bilinearly), and Occlusion "
         "(window and stride 4 and 2 at 28 and 32 px, 16 and 8 on Pet). For each method we compute the share of "
         "positive attribution on the shortcut pixels (relevance mass accuracy " + cite("arras") + ").")

    # ---- 4 Setup
    heading("4. Experimental setup")
    para("**Datasets.** Oxford-IIIT Pet " + cite("pets") + " as a cat-versus-dog task at 128\u00d7128 with the "
         "dataset's trimap masks; CIFAR-10 (10 classes, 32\u00d732 colour); MNIST (10 classes, 28\u00d728); and "
         "PneumoniaMNIST " + cite("medmnist") + " (normal vs pneumonia, 28\u00d728, derived from "
         + cite("kermany") + ").")
    para("**Networks and training.** A small CNN with four convolutional layers on the 28\u00d728 data; on "
         "CIFAR-10 a CNN with batch normalisation (0.77M parameters) and a ResNet-18 " + cite("he") + " (11.2M "
         "parameters) with a 3\u00d73 stride-1 first convolution and no initial max-pooling; on Pet the same "
         "ResNet-18, also with this CIFAR-style stem. All networks are trained from scratch with Adam (learning "
         f"rate 10\u207b\u00b3, batch size 128) for {EPOCHS['mnist']} (MNIST), {EPOCHS['pneumonia_mnist']} "
         f"(PneumoniaMNIST), {EPOCHS['cifar10']} (CIFAR CNN), {EPOCHS['cifar10_resnet18']} (CIFAR ResNet-18) and "
         f"{EPOCHS['pet_resnet18']} epochs (Pet), keeping the best validation checkpoint, without augmentation. "
         f"The ResNet-18 models were trained on a free Kaggle T4 GPU (about {MINUTES['cifar10_resnet18']:.0f} "
         f"minutes per CIFAR model and {MINUTES['pet_resnet18']:.0f} per Pet model); all others on a laptop CPU.")
    counts = "; ".join(f"{BASE_NAMES[b]} {N_BASE[b]}" for b in ["pet_resnet18", "cifar10_resnet18", "cifar10",
                                                                 "mnist", "pneumonia_mnist"])
    para(f"**Models.** {N_MODELS} models in total ({counts}). Seeds: "
         + "; ".join(f"{BASE_NAMES[b]} {', '.join(str(s) for s in SEEDS[b])}" for b in
                     ["pet_resnet18", "cifar10_resnet18", "cifar10", "mnist", "pneumonia_mnist"])
         + f". {N_ORIGINAL} models existed before amendment A1 (docs/PREDICTIONS_amendments.md, commit 8738b32, "
         f"committed before any of its models was trained); A1 added {N_A1} MNIST and PneumoniaMNIST models at "
         "new doses near the steep part of the dose-response curve, with six seeds at the two steepest MNIST "
         "doses, to fill reliance gaps. A1 made no predictions about the metrics, and both model sets are "
         "reported (Appendix C).")
    para("**Evaluation.** Every model is scored on 100 aligned test images, the same images for every model of a "
         "setting. Reliance and accuracy use the full test set. Re-evaluating a cached model reproduces its "
         "stored scores exactly.")

    # ---- 5 Results
    heading("5. Results")
    heading("5.1. The dose controls reliance", 2)
    t_rel = COUNTERS["table"] + 1
    para(f"Table {t_rel} shows that the dose moves reliance across most of the range in every setting. On Pet "
         f"the lowest raw reliance is {f1(group('pet_resnet18', 'patch').reliance.min())}%, but its chance floor "
         f"is as large, so {RC} starts at {f1(group('pet_resnet18', 'patch').reliance_corrected.min())}%. The same "
         f"holds on PneumoniaMNIST: the lowest per-dose mean raw reliance is {f1(pneu_low['patch'].reliance)}% "
         f"(patch) and {f1(pneu_low['texture'].reliance)}% (texture), which is {RC} = "
         f"{f1(pneu_low['patch'].reliance_corrected)}% and {f1(pneu_low['texture'].reliance_corrected)}%. Earlier "
         "drafts reported \u201cabout 15%, not 0\u201d here; that 15% is the error floor. Near _d_ = 0 reliance "
         "varies strongly between seeds, so every analysis uses measured reliance, not the dose.")
    table_reliance()

    heading("5.2. Natural images: the metrics respond but over-detect", 2)
    f_full = COUNTERS["fig"] + 1
    t_full = COUNTERS["table"] + 1
    para(f"Fig. {f_full} and Table {t_full} give the main comparison, the shortcut map against the full object "
         f"map. Over all settings, preference rises with {RC} in {n_full_sig} of {n_full} cells (\u03c1 = "
         f"{span(full.rho_rc)}); the exception is "
         + "; ".join(f"{r.metric} on {LABEL[(r.base, r.shortcut)]} (\u03c1 = {f2(r.rho_rc)}, p_bh = "
                     f"{format_p(r.p_bh)})" for r in full_weak.itertuples())
         + f". On the natural images \u03c1 = {span(full_nat.rho_rc)}.")
    para("Responsiveness is not calibration. Among the models with "
         f"{RC} \u2264 50, black and blur deletion still prefer the shortcut map for {pct_span(pet_bb_fpr)} of the "
         f"Pet models and {pct_span(cifar_bb_fpr)} of the CIFAR-10 models; inpainting deletion and insertion do "
         f"so for {pct_span(nat_ii_fpr)}. Across all settings the true-positive rate is 1.00 in {n_tpr1} of "
         f"{n_full} cells and the over-detection rate is above 0 in {n_fpr_pos} of {n_full}: the metrics rarely miss a "
         "shortcut the model uses, but they often report one the model does not use. On Pet, the isotonic "
         f"switch lies at {span(pet_switch, f1)}% {RC} for "
         + ", ".join(m for m in METRICS if m not in pet_no_switch)
         + (f" ({', '.join(pet_edge)} at the lowest-reliance edge)" if pet_edge else "")
         + (f", and {', '.join(pet_no_switch)} prefers the shortcut map already at the lowest reliance" if pet_no_switch else "")
         + f". The Pet models themselves follow the shortcut on half of the images only at {RC} = "
         f"{f1(beh_switch[('pet_resnet18', 'patch')])}% (Section 5.4). The effect sizes temper this: at the "
         f"lowest Pet dose the median per-image AUC gap is {signed_span(pet_gap_low, f3)}, so the low-reliance "
         "preferences there are close to ties.")
    figure(figs["full"],
           f"Preference for the shortcut map over the full object map against chance-corrected reliance {RC}, for "
           "the four metrics in the seven settings (natural images first). Dots are single models; lines are the "
           "increasing isotonic fits used for the switch points. Dotted line: the assumed 50% ideal; dashed line: "
           "the behavioural switch, where the models' per-image follow rate crosses 0.5. Data: "
           "results/dose_response/preference.csv and behaviour_summary.csv")
    table_main(FULL, f"Metric responsiveness on {RC}, shortcut map vs full object map")
    t_ctrl = COUNTERS["table"] + 1
    f_ro = COUNTERS["fig"] + 1
    para(f"Where does the over-detection come from? Table {t_ctrl} and Fig. {f_ro} compare the object map with a "
         "random pixel ordering at the lowest dose, where the model barely uses the shortcut. On CIFAR-10 black "
         f"and blur deletion prefer the random ordering over the centred object map {pct_span(ro_bb['cifar10'])} "
         f"of the time with the small CNN and {pct_span(ro_bb['cifar10_resnet18'])} with ResNet-18, against "
         f"{pct_span(ro_ii['cifar10'] + ro_ii['cifar10_resnet18'])} for inpainting deletion and insertion. With "
         f"the real Pet masks the bias is smaller ({pct_span(ro_bb['pet_resnet18'])} against "
         f"{pct_span(ro_ii['pet_resnet18'])}), which supports the earlier hypothesis that the Gaussian object map "
         "inflated it on CIFAR-10, but black and blur deletion still treat the object mask about like a random "
         "ordering.")
    table_controls()
    figure(figs["r_object"],
           f"Preference for a random ordering over the full object map against {RC}. Ideal: low at low reliance; a "
           "rise at high reliance is correct, because the object map then explains the model worse than random. "
           "Data: results/dose_response/preference.csv")

    heading("5.3. Pre-registered test on Oxford-IIIT Pet", 2)
    t_pre = COUNTERS["table"] + 1
    para("The Pet predictions were committed before any Pet model was trained (commit a962b6c), followed by the "
         "Pet code (20694f2), a documented dose change after the first model and before any metric result "
         "(d823737), and the results with the recorded verdicts (1e29918). P1 predicted \u03c1 > 0.8 between "
         "reliance and the shortcut-vs-object preference for all four metrics; P2 a larger random-over-object "
         "preference for black and blur deletion than for inpainting and insertion at the lowest dose; P3 "
         "starting values within 0.5 \u00b1 0.15 and \u03c1 > 0.8 in the form-matched comparisons; P4 \u03c1 > 0.8 "
         f"for the shortcut mass of Integrated Gradients and Occlusion. Table {t_pre} lists the outcomes. P1, P2 "
         "and P4 hold and P3 fails for black deletion (\u03c1 = "
         f"{f2(p3_black_sr.estimate)} against the random map and {f2(p3_black_sp.estimate)} against the placebo). "
         "Verdicts were judged on point estimates, as pre-registered; the intervals are added information. They "
         f"qualify P1 for black deletion, whose estimate {f2(p1_black.estimate)} has a two-level interval of "
         f"{ci(p1_black.ci_low, p1_black.ci_high)} that reaches below the 0.8 threshold. The P2 contrast, "
         f"{f3(p2_contrast.estimate)} {ci(p2_contrast.ci_low, p2_contrast.ci_high, f3)}, excludes zero.")
    table_prereg()

    heading("5.4. Per-image behavioural ground truth", 2)
    t_beh = COUNTERS["table"] + 1
    f_beh = COUNTERS["fig"] + 1
    para(f"Table {t_beh} and Fig. {f_beh} score the per-image AUC gap against the behavioural label. Pooled over "
         f"all images of a setting, the class-stratified AUROC is {span(pooled_nat)} on the natural images and "
         f"{span(pooled_mech)} on the mechanism settings; PneumoniaMNIST patch is the weak case "
         f"({span(pp_pooled)}; black deletion {f2(pp_black.auroc_pooled_strat)} "
         f"{ci(pp_black.auroc_strat_ci_low, pp_black.auroc_strat_ci_high)}, Holm-adjusted "
         f"{p_text(pp_black.p_holm_headline_strat)}). The pooled AUROC mixes models of different reliance. Within "
         f"a single model the median AUROC is {span(within_nat)} on the natural images and {span(within_mnist)} on "
         "MNIST, so most of the pooled signal separates high- from low-reliance models rather than followed from "
         f"non-followed images of one model. Within {RC} bands (Fig. {f_beh}) the PneumoniaMNIST patch AUROC "
         f"falls to {span(pp_band.auroc_strat)} above {RC} = 60%, where the metrics rank the images the "
         f"wrong way round (with only {span(pp_band.n_neg, lambda v: f'{v:.0f}')} non-followed images per band).")
    para("The behavioural labels also locate the ideal switch. The models follow the shortcut on half of the "
         "images at " + ", ".join(f"{RC} = {f1(beh_switch[(b, s)])}% ({short})" for b, s, _, short in SETTINGS)
         + ". The assumed 50% is therefore about right for Pet, CIFAR-10 and MNIST patch and too high for "
         "PneumoniaMNIST. Measured against this derived ideal the over-detection remains: on the natural images "
         f"the calibration gap (preference minus follow rate) is positive in every cell ({span(gap_nat)}).")
    table_behaviour()
    figure(figs["behaviour"],
           f"Class-stratified AUROC of each metric's per-image AUC gap against the behavioural label, within {RC} "
           "bands (shortcut vs full object). Panels rearranged from results/dose_response/behaviour_auroc.png "
           "(alphabetical setting order)")

    heading("5.5. Mechanism settings: size and switch identification", 2)
    f_matched = COUNTERS["fig"] + 1
    t_matched = COUNTERS["table"] + 1
    mp_low_full, mp_low_matched = mp_full.pref_low, mp_matched.pref_low
    para("The full object map marks far more pixels than the shortcut map. Replacing it by a size-matched object "
         f"map (Fig. {f_matched}, Table {t_matched}) raises the MNIST patch preference for the shortcut at the "
         f"lowest dose from {span(mp_low_full, f3)} to {span(mp_low_matched, f3)}, moves the switch from "
         f"{span(mp_full.switch_iso_rc, f1)}% to {span(mp_matched.switch_iso_rc, f1)}% {RC}, and raises the "
         f"over-detection rate from {span(mp_full.fpr)} to {span(mp_matched.fpr)}. The metrics still respond "
         f"(\u03c1 = {span(mp_matched.rho_rc)}), but much of what looked like \u201cthe metric recognises the "
         "object\u201d is the object map being larger. On PneumoniaMNIST texture, black deletion barely responds "
         f"under this control (\u03c1 = {f2(pt_black_m.rho_rc)}, p_bh = {format_p(pt_black_m.p_bh)}).")
    para("Before amendment A1, the original MNIST patch models left a gap in raw reliance from "
         f"{f1(gap_orig[0])}% to {f1(gap_orig[1])}%, and the switches against the full object map "
         f"({span(mp_orig.switch_iso_rc, f1)}% {RC}) lay inside it, so earlier drafts' statement that black and "
         "blur deletion switch \u201cclearly before 50%\u201d was not supported. With the A1 models the widest "
         f"gap between 5% and 95% raw reliance is {f1(gap_all[0])}% to {f1(gap_all[1])}%, every switch has a "
         f"model within {f1(mp_full.nearest_model_distance.max())} points, and the switches are "
         f"{span(mp_full.switch_iso_rc, f1)}% {RC} (4-parameter logistic {span(mp_full.switch_4pl_rc, f1)}%), "
         f"against a behavioural switch of {f1(beh_switch[('mnist', 'patch')])}%. The added models moved the "
         "switches up and closer together, and raised the over-detection rate; Appendix C lists both model sets.")
    figure(figs["matched"],
           f"Preference for the shortcut map over the size-matched object map against {RC}. Layout as in Fig. "
           f"{f_full}. Data: results/dose_response/preference.csv")
    table_main(MATCHED, f"Metric responsiveness on {RC}, shortcut map vs size-matched object map")
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
         "\u201crises to 0.93\u20130.99 for every metric\u201d in earlier drafts was too general. The placebo "
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
         f"not make inpainting the best variant (Table {t_pairs}): {n_pairs_sig} of {len(PAIRS)} metric pairs "
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
           "Share of positive attribution that each explanation method places on the shortcut pixels, against raw "
           "reliance. Panels rearranged from results/dose_response/method_dose_response.png (alphabetical setting "
           "order)")
    table_methods()

    heading("5.9. Reliance and object competence", 2)
    t_conf = COUNTERS["table"] + 1
    cnn = conf_cifar.iloc[0]
    para("A model that relies more on the shortcut is also worse at the object, so a metric that responds to "
         "reliance may be responding to object competence. Table "
         f"{t_conf} separates the two. Reliance and accuracy without the shortcut are least collinear for the "
         f"CIFAR small CNN (\u03c1 = {f2(cnn.rho_reliance_competence)}), where the partial \u03c1 of preference "
         f"with reliance given competence stays at {span(conf_cifar.partial_rel_given_acc)}. In "
         f"{len(collinear)} of {len(conf_set)} settings the two are collinear (\u03c1 \u2264 \u22120.95), and there "
         "the partial correlations are smaller and in places not significant (MNIST texture "
         f"{signed_span(CONF[(CONF.base == 'mnist') & (CONF.shortcut == 'texture')].partial_rel_given_acc)}). "
         "Outside the CIFAR small CNN our design cannot fully separate reliance from object competence.")
    table_confound()

    # ---- 6 Discussion
    heading("6. Discussion")
    para("**Responsive but not calibrated.** Every metric ranks models by reliance, but a ranking is not a "
         "verdict. A practitioner who reads \u201cthe metric prefers the shortcut explanation\u201d as \u201cthe "
         "model uses the shortcut\u201d would be wrong for most of the low-reliance Pet and CIFAR-10 models under "
         "black and blur deletion. Bilodeau et al. " + cite("bilodeau") + " argue that the end task must be "
         "defined; our behavioural labels give one such definition, and against it the metrics over-detect.")
    para("**Size and form.** The size-matched control removes much of the preference for the object map at low "
         "reliance, and the random control shows black and blur deletion preferring scattered removal over "
         "compact removal on natural images. Both fit known mechanisms: removal shape leaks class information "
         + cite("rong") + ", removal-based benchmarks favour spatially blurry masks " + cite("roarpitfalls")
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
        "is inspired by ROAD but is not ROAD.",
        "All networks are trained from scratch; the Pet ResNet-18 uses a CIFAR-style stem at 128\u00d7128, and no "
        "pretrained or transformer backbone was tested.",
        "The shortcut is synthetic (a white patch or a brightened texture). The dose varies the shortcut's "
        "predictivity, not its availability " + cite("hermann") + ".",
        "The oracle maps have limits: on PneumoniaMNIST the object map is an intensity ranking of the whole "
        "radiograph; on CIFAR-10 it is a centred Gaussian prior, so Pet is the primary natural-image evidence; "
        "the size-matched map on MNIST takes the topmost tied saturated pixels; blur deletion leaves about "
        "20\u201325% of the patch intensity in place; and the inpainting curve ends on noise around black.",
        f"Reliance and object competence are collinear in {len(collinear)} of {len(conf_set)} settings (Section 5.9).",
        "The chance correction assumes that errors spread evenly over the wrong classes.",
        "Reliance gaps remain on CIFAR-10 and Pet, where no models were added; each model is scored on 100 "
        "images, and the low-reliance preferences on Pet are near ties.",
        "The switch-identification window (\u00b110 points), the causal behavioural label and the class-stratified "
        "AUROC were chosen after the audit and after the original-set results had been inspected. The amendment-A1 "
        "models had been trained by then, but they entered no analysis until these choices were committed.",
    ])

    # ---- 8 Conclusion
    heading("8. Conclusion")
    para(f"Across {N_MODELS} models with controlled and measured shortcut reliance, deletion and insertion metrics "
         "respond to reliance but are not calibrated to it, and their verdicts depend on the size and form of "
         "the explanations they compare. On natural images black and blur deletion report a shortcut for most "
         "models that barely use it, the metrics switch far below the reliance at which the models follow the "
         "shortcut on half of the images, and within a model they separate followed from non-followed images "
         "only weakly. The pre-registered Pet test confirmed three of four predictions and showed black "
         "deletion failing a form-matched control. Metrics should be validated at several measured reliance "
         "levels, against behavioural ground truth and with size- and form-matched controls, before they are "
         "used to rank explanation methods. Future work should test other metric families, pretrained "
         "backbones and realistic shortcuts.")

    # ---- Declarations
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

    # ---- Appendix A
    heading("Appendix A. Supporting study: the sanity check against ground truth")
    t_k = COUNTERS["table"] + 1
    f_gt = COUNTERS["fig"] + 1
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

    # ---- Appendix B
    heading("Appendix B. Legacy analysis")
    para("Earlier drafts used raw reliance, a bootstrap over seeds only, an unbalanced decision accuracy and a "
         "first-crossing switch rule. These values are kept for comparison and are superseded by Section 5.")
    table_legacy()

    # ---- Appendix C
    heading("Appendix C. Original and amended model sets")
    para(f"Amendment A1 committed to reporting both the {N_ORIGINAL} original models and all {N_MODELS} models. "
         "The added models strengthen the rank correlations slightly and weaken the decision-level results: "
         "over-detection rises and the switch points move.")
    table_sets()

    # ---- References
    heading("References")
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

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT} ({COUNTERS['fig']} figures, {COUNTERS['table']} tables, {len(CITED)} references, "
          f"{N_MODELS} models, abstract {word_count(abstract)} words)")


if __name__ == "__main__":
    build()

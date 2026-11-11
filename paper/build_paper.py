"""Build paper/paper_draft.docx from the finished dose-response results.

Usage (from the project root):
    .venv\\Scripts\\python.exe paper\\build_paper.py

Every number in the text and tables is read from the result files:
    results/dose_response/models.csv, responsiveness.csv, method_responsiveness.csv
    results/analysis/gt_agreement.csv, gt_instance.csv, accuracy.csv, decoy_mnist_summary.csv
Nothing is retrained or re-evaluated. The script only writes inside paper/
(the document and rearranged copies of the wide figures in paper/figures/).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401 (kept for easy landscape tables)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from PIL import Image

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
DR = ROOT / "results" / "dose_response"
AN = ROOT / "results" / "analysis"
EX = ROOT / "results" / "explanations"
FIG_OUT = PAPER / "figures"
OUTPUT = PAPER / "paper_draft.docx"

TITLE = ("A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut "
         "Reliance in CNN Image Classifiers")
DRAFT_NOTICE = "DRAFT for guide review \u2014 not for submission"
DASH = "\u2013"

# ------------------------------------------------------------------ data

MODELS = pd.read_csv(DR / "models.csv")
RESP = pd.read_csv(DR / "responsiveness.csv")
METHOD_RESP = pd.read_csv(DR / "method_responsiveness.csv")
GT_TAU = pd.read_csv(AN / "gt_agreement.csv")
GT_INSTANCE = pd.read_csv(AN / "gt_instance.csv")
ACCURACY = pd.read_csv(AN / "accuracy.csv").set_index("dataset")
DECOY_MNIST = pd.read_csv(AN / "decoy_mnist_summary.csv").set_index("method")

METRICS = ["Deletion (black)", "Deletion (blur)", "Deletion (inpaint)", "Insertion (blur)"]
DELETION = METRICS[:3]
METHODS = ["Saliency", "SmoothGrad", "Integrated Gradients", "Grad-CAM", "Occlusion"]
SETTINGS = [  # (base, shortcut, label)
    ("mnist", "patch", "MNIST, patch"),
    ("mnist", "texture", "MNIST, texture"),
    ("pneumonia_mnist", "patch", "PneumoniaMNIST, patch"),
    ("pneumonia_mnist", "texture", "PneumoniaMNIST, texture"),
    ("cifar10", "patch", "CIFAR-10, patch (small CNN)"),
    ("cifar10_resnet18", "patch", "CIFAR-10, patch (ResNet-18)"),
]
FULL, MATCHED = "shortcut vs full object", "shortcut vs size-matched object"
S_RANDOM, S_PLACEBO = "shortcut vs random", "shortcut vs placebo"
R_OBJECT, P_OBJECT, P_RANDOM = "random vs full object", "placebo vs full object", "placebo vs random"


def row(comparison, base, shortcut, metric):
    """One row of responsiveness.csv (exactly one must match)."""
    sel = RESP[(RESP.comparison == comparison) & (RESP.base == base)
               & (RESP.shortcut == shortcut) & (RESP.metric == metric)]
    assert len(sel) == 1, (comparison, base, shortcut, metric)
    return sel.iloc[0]


def rows(comparisons, bases, shortcuts=("patch", "texture"), metrics=METRICS):
    sel = RESP[RESP.comparison.isin(comparisons) & RESP.base.isin(bases)
               & RESP.shortcut.isin(shortcuts) & RESP.metric.isin(metrics)]
    assert len(sel) > 0
    return sel


def mrow(base, shortcut, method):
    sel = METHOD_RESP[(METHOD_RESP.base == base) & (METHOD_RESP.shortcut == shortcut)
                      & (METHOD_RESP.method == method)]
    assert len(sel) == 1
    return sel.iloc[0]


def f2(x):
    return f"{x:.2f}"


def f1(x):
    return f"{x:.1f}"


def pct(x):
    """A preference share (0-1) as a whole percentage."""
    return f"{100 * x:.0f}%"


def span(values, fmt=f2, unit=""):
    """'a-b' range of values, formatted (a single value if they are equal)."""
    lo, hi = fmt(min(values)), fmt(max(values))
    return f"{lo}{unit}" if lo == hi else f"{lo}{DASH}{hi}{unit}"


def pct_span(values):
    # Round half up (0.225 -> 23%), not to even.
    lo, hi = int(np.floor(100 * min(values) + 0.5 + 1e-9)), int(np.floor(100 * max(values) + 0.5 + 1e-9))
    return f"{lo}%" if lo == hi else f"{lo}{DASH}{hi}%"


def signed(x):
    return f"{x:.2f}".replace("-", "\u2212")


def signed_span(values):
    return f"{signed(min(values))} to {signed(max(values))}"


# Per-dose means (as in summary.md, first table).
LEVELS = MODELS.groupby(["base", "shortcut", "dose"])[
    ["reliance", "test_accuracy", "test_accuracy_no_shortcut"]].mean().reset_index()


def levels(base, shortcut):
    return LEVELS[(LEVELS.base == base) & (LEVELS.shortcut == shortcut)].sort_values("dose")


def lowest_reliance(base, shortcut):
    return levels(base, shortcut).reliance.min()


N_MODELS = len(MODELS)
N_SMALL = len(MODELS[MODELS.base.isin(["mnist", "pneumonia_mnist"])])
N_CIFAR = len(MODELS[MODELS.base == "cifar10"])
N_RESNET = len(MODELS[MODELS.base == "cifar10_resnet18"])
RESNET_MINUTES = MODELS[MODELS.base == "cifar10_resnet18"].training_minutes.mean()
CIFAR_MINUTES = MODELS[MODELS.base == "cifar10"].training_minutes.mean()
SEEDS = {b: sorted(MODELS[MODELS.base == b].seed.unique()) for b in MODELS.base.unique()}
DOSES = {b: sorted(MODELS[MODELS.base == b].dose.unique()) for b in MODELS.base.unique()}
assert N_MODELS == 174, N_MODELS

# ---- numbers used in the text (all derived from the CSVs)
MNIST_RHO = rows([FULL, MATCHED], ["mnist"]).responsiveness_rho
PNEU = rows([FULL, MATCHED], ["pneumonia_mnist"])
PNEU_RHO = PNEU.responsiveness_rho
PNEU_NONSIG = PNEU[PNEU.p_value >= 0.05]
PNEU_WORST = PNEU.loc[PNEU.responsiveness_rho.idxmin()]

MP_FULL = rows([FULL], ["mnist"], ["patch"])
MP_MATCHED = rows([MATCHED], ["mnist"], ["patch"])
MP_FULL_EARLY = rows([FULL], ["mnist"], ["patch"], ["Deletion (black)", "Deletion (blur)"])
MP_FULL_LATE = rows([FULL], ["mnist"], ["patch"], ["Deletion (inpaint)", "Insertion (blur)"])

MT_INS = row(FULL, "mnist", "texture", "Insertion (blur)")
MT_DEL_BI = rows([FULL], ["mnist"], ["texture"], ["Deletion (black)", "Deletion (inpaint)"])
MT_BLUR_RANDOM = row(S_RANDOM, "mnist", "texture", "Deletion (blur)")
MT_INS_RANDOM = row(S_RANDOM, "mnist", "texture", "Insertion (blur)")
MT_INS_PLACEBO = row(S_PLACEBO, "mnist", "texture", "Insertion (blur)")


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {f2(p)}"

CIFAR_FULL = rows([FULL], ["cifar10"], ["patch"])
RESNET_FULL = rows([FULL], ["cifar10_resnet18"], ["patch"])
BB = ["Deletion (black)", "Deletion (blur)"]
II = ["Deletion (inpaint)", "Insertion (blur)"]


def low(comparison, base, metrics):
    return [row(comparison, base, "patch", m).preference_low_reliance for m in metrics]


CIFAR_RAND_BB = low(R_OBJECT, "cifar10", BB)
CIFAR_RAND_II = low(R_OBJECT, "cifar10", II)
CIFAR_SHORT_BB = low(FULL, "cifar10", BB)
RESNET_RAND_BB = low(R_OBJECT, "cifar10_resnet18", BB)
RESNET_RAND_II = low(R_OBJECT, "cifar10_resnet18", II)
RESNET_SHORT_BB = low(FULL, "cifar10_resnet18", BB)
ALL_RAND_II = CIFAR_RAND_II + RESNET_RAND_II
ALL_RAND_BB = CIFAR_RAND_BB + RESNET_RAND_BB

# Same-form comparisons on the patch shortcut.
SAME_FORM = rows([S_RANDOM, S_PLACEBO], ["mnist", "cifar10", "cifar10_resnet18", "pneumonia_mnist"], ["patch"])
SF_CLEAN = SAME_FORM[SAME_FORM.base != "pneumonia_mnist"]
SF_PNEU = SAME_FORM[SAME_FORM.base == "pneumonia_mnist"]
# Placebo vs random, deletion variants, every setting.
PR_DEL = rows([P_RANDOM], ["mnist", "pneumonia_mnist", "cifar10", "cifar10_resnet18"], metrics=DELETION)
PR_INS_MNIST = rows([P_RANDOM], ["mnist"], metrics=["Insertion (blur)"])

# Robustness: significant positive rho when the shortcut map is compared with every other map.
SHORTCUT_COMPARISONS = [FULL, MATCHED, S_PLACEBO, S_RANDOM]
ROBUST = {}
for m in METRICS:
    sel = rows(SHORTCUT_COMPARISONS, [b for b, _, _ in SETTINGS], metrics=[m])
    sel = sel[sel.apply(lambda r: (r.base, r.shortcut) in {(b, s) for b, s, _ in SETTINGS}, axis=1)]
    ROBUST[m] = (int(((sel.responsiveness_rho > 0) & (sel.p_value < 0.05)).sum()), len(sel),
                 sel.responsiveness_rho.min())
ROBUST_ALL = [m for m, (ok, n, _) in ROBUST.items() if ok == n]

IG_RHO = METHOD_RESP[METHOD_RESP.method == "Integrated Gradients"].responsiveness_rho
MT_AWAY = [mrow("mnist", "texture", m).responsiveness_rho for m in ["Saliency", "SmoothGrad", "Grad-CAM"]]
PT_GRADCAM = mrow("pneumonia_mnist", "texture", "Grad-CAM").responsiveness_rho
PT_SALIENCY = mrow("pneumonia_mnist", "texture", "Saliency").responsiveness_rho
PATCH_SETTINGS = [(b, s) for b, s, _ in SETTINGS if s == "patch"]
GC_LOWEST = sum(
    min(METHODS, key=lambda m: mrow(b, s, m).mass_high_reliance) == "Grad-CAM" for b, s in PATCH_SETTINGS)
PNEU_TIES = RESP[RESP.base == "pneumonia_mnist"].tie_rate.max()
MNIST_TIES = RESP[RESP.base == "mnist"].tie_rate.max()

# Supporting study.
TAU = GT_TAU.groupby(["dataset", "metric"]).tau_with_ground_truth.mean()
SANITY_TAU = [TAU[("decoy_mnist", "Sanity (object px)")], TAU[("decoy_pneumonia", "Sanity (object px)")]]
DEL_TAU_MNIST = [TAU[("decoy_mnist", m)] for m in METRICS]
DEL_TAU_PNEU = [TAU[("decoy_pneumonia", m)] for m in METRICS]
INST = GT_INSTANCE.set_index(["dataset", "metric"]).spearman_with_ground_truth
DEL_INST = [INST[(d, m)] for d in ("decoy_mnist", "decoy_pneumonia") for m in DELETION]
INS_INST = [INST[(d, "Insertion (blur)")] for d in ("decoy_mnist", "decoy_pneumonia")]

# ------------------------------------------------------------------ references

REFERENCES = {
    "selvaraju": "Selvaraju RR, et al. (2017) Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV)",
    "sundararajan": "Sundararajan M, Taly A, Yan Q (2017) Axiomatic attribution for deep networks. In: Proceedings of the International Conference on Machine Learning (ICML)",
    "simonyan": "Simonyan K, Vedaldi A, Zisserman A (2014) Deep inside convolutional networks: visualising image classification models and saliency maps. In: ICLR Workshop",
    "zeiler": "Zeiler MD, Fergus R (2014) Visualizing and understanding convolutional networks. In: Proceedings of the European Conference on Computer Vision (ECCV)",
    "adebayo2018": "Adebayo J, et al. (2018) Sanity checks for saliency maps. In: Advances in Neural Information Processing Systems (NeurIPS)",
    "samek": "Samek W, et al. (2017) Evaluating the visualization of what a deep neural network has learned. IEEE Transactions on Neural Networks and Learning Systems",
    "captum": "Kokhlikyan N, et al. (2020) Captum: a unified and generic model interpretability library for PyTorch. arXiv:2009.07896",
    "petsiuk": "Petsiuk V, Das A, Saenko K (2018) RISE: randomized input sampling for explanation of black-box models. In: Proceedings of the British Machine Vision Conference (BMVC)",
    "rong": "Rong Y, et al. (2022) A consistent and efficient evaluation strategy for attribution methods (ROAD). In: Proceedings of the International Conference on Machine Learning (ICML)",
    "tomsett": "Tomsett R, et al. (2020) Sanity checks for saliency metrics. In: Proceedings of the AAAI Conference on Artificial Intelligence. arXiv:1912.01451",
    "binder": "Binder A, et al. (2023) Shortcomings of top-down randomization-based sanity checks for evaluations of deep neural network explanations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    "hedstrom2024": "Hedstr\u00f6m A, et al. (2024) A fresh look at sanity checks for saliency maps. arXiv:2405.02383",
    "metaquantus": "Hedstr\u00f6m A, et al. (2023) The meta-evaluation problem in explainable AI: identifying reliable estimators with MetaQuantus. arXiv:2302.07265",
    "baseline": "Back to the baseline: examining baseline effects on explainability metrics (2025). arXiv:2512.11433",
    "ross": "Ross AS, Hughes MC, Doshi-Velez F (2017) Right for the right reasons: training differentiable models by constraining their explanations. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)",
    "sun": "Sun S, et al. (2023) Right for the wrong reason: can interpretable ML techniques detect spurious correlations? In: Medical Image Computing and Computer Assisted Intervention (MICCAI). arXiv:2307.12344",
    "zhou": "Zhou Y, et al. (2022) Do feature attribution methods correctly attribute features? In: Proceedings of the AAAI Conference on Artificial Intelligence. arXiv:2104.14403",
    "backx": "A backdoor-based explainable AI benchmark for high fidelity evaluation of attributions (BackX). arXiv:2405.02344",
    "attributionlab": "AttributionLab: faithfulness of feature attribution under controllable environments (2023). NeurIPS XAI in Action Workshop. arXiv:2310.06514",
    "medmnist": "Yang J, et al. (2023) MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data",
    "smilkov": "Smilkov D, et al. (2017) SmoothGrad: removing noise by adding noise. arXiv:1706.03825",
    "adebayo2022": "Adebayo J, et al. (2022) Post hoc explanations may be ineffective for detecting unknown spurious correlation. In: International Conference on Learning Representations (ICLR)",
    "arras": "Arras L, Osman A, Samek W (2022) CLEVR-XAI: a benchmark dataset for the ground truth evaluation of neural network explanations. Information Fusion",
    "kermany": "Kermany DS, et al. (2018) Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell",
    "latec": "Klein et al. (2024) LATEC. In: Advances in Neural Information Processing Systems (NeurIPS). arXiv:2409.16756 [full title and author list to be added from the paper]",
    "fidelity": "A comprehensive study on fidelity metrics for XAI (2024). arXiv:2401.10640",
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


def rearrange_panels(source, target, columns=3):
    """The dose-response figures put six panels in one very wide row (3354x507 px),
    which is unreadable at page width. Cut the panels at the widest white gaps and
    lay them out in a 2x3 grid below the original title. Pixels are only moved, not
    changed. Falls back to the original file if the panels can't be found."""
    image = Image.open(source).convert("RGB")
    a = np.asarray(image)
    ink = (a < 245).any(axis=2)
    rows_ink = ink.any(axis=1)
    # Title band: from the top to the first blank row after the first inked rows.
    first = int(np.argmax(rows_ink))
    gap = first + int(np.argmax(~rows_ink[first:]))
    title_end = gap + int(np.argmax(rows_ink[gap:]))
    body = ink[title_end:]
    cols_ink = body.any(axis=0)
    # Runs of blank columns between inked columns.
    gaps, start = [], None
    inked = np.flatnonzero(cols_ink)
    left, right = inked.min(), inked.max()
    for x in range(left, right + 1):
        if not cols_ink[x] and start is None:
            start = x
        elif cols_ink[x] and start is not None:
            gaps.append((x - start, start, x))
            start = None
    panels = 6
    if len(gaps) < panels - 1:
        return source
    cuts = sorted((s + e) // 2 for _, s, e in sorted(gaps, reverse=True)[:panels - 1])
    edges = [left] + cuts + [right + 1]
    pieces = [image.crop((edges[i], title_end, edges[i + 1], a.shape[0])) for i in range(panels)]
    if min(p.width for p in pieces) < 0.6 * max(p.width for p in pieces):
        return source
    # The y axis (label and tick labels) is drawn only left of the first panel
    # (shared y). Cut it off at the first panel's left frame line and repeat it
    # at the start of every grid row.
    # Every panel is then cropped at its own left frame line, so the panels line up.
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
        # Left-align so every panel's frame starts right after the y-axis strip.
        canvas.paste(piece, (strip_w + c * cell_w, title.height + 10 + r * cell_h))
    FIG_OUT.mkdir(exist_ok=True)
    canvas.save(target)
    return target


# ------------------------------------------------------------------ document helpers

doc = Document()
COUNTERS = {"fig": 0, "table": 0}


def setup():
    section = doc.sections[0]
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin"):
        setattr(section, side, Cm(2.5))
    section.top_margin = section.bottom_margin = Cm(2.5)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for level, size in ((1, 13), (2, 11.5)):
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


def para(text, bold=False, italic=False, align=None, size=None, after=None):
    p = doc.add_paragraph()
    add_runs(p, text, bold, italic, size)
    if align is not None:
        p.alignment = align
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


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
    p.add_run().add_picture(str(path), width=width)
    p.paragraph_format.keep_with_next = True
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = c.add_run(f"Fig. {COUNTERS['fig']} ")
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
    """A numbered table: caption above (Springer style), optional note below."""
    COUNTERS["table"] += 1
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.keep_with_next = True
    r = c.add_run(f"Table {COUNTERS['table']} ")
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
    if header_rows:  # merge identical neighbouring cells in the top header row
        top = header_rows[0]
        j = 0
        while j < len(top):
            k = j
            while k + 1 < len(top) and top[k + 1] == top[j] and top[j] != "":
                k += 1
            if k > j:
                merged = t.cell(0, j).merge(t.cell(0, k))
                merged.text = ""
                run = merged.paragraphs[0].add_run(top[j])
                run.bold, run.font.size = True, Pt(size)
                merged.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            j = k + 1
        if len(header_rows) == 2 and header_rows[0][0] == header_rows[1][0]:
            merged = t.cell(0, 0).merge(t.cell(1, 0))
            merged.text = ""
            run = merged.paragraphs[0].add_run(header_rows[0][0])
            run.bold, run.font.size = True, Pt(size)
    if widths:
        for i, w in enumerate(widths):
            for cell in t.columns[i].cells:
                cell.width = Cm(w)
    if note:
        n = doc.add_paragraph()
        n.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(n, note, size=8)
    else:
        doc.add_paragraph()
    return COUNTERS["table"]


# ------------------------------------------------------------------ table builders


def table_reliance():
    body = []
    nets = {"mnist": "Small CNN (28\u00d728)", "pneumonia_mnist": "Small CNN (28\u00d728)",
            "cifar10": "CIFAR CNN, 0.77M", "cifar10_resnet18": "ResNet-18, 11.2M"}
    for base, shortcut, label in SETTINGS:
        lv = levels(base, shortcut)
        group = MODELS[(MODELS.base == base) & (MODELS.shortcut == shortcut)]
        lowest, highest = lv.loc[lv.reliance.idxmin()], lv.loc[lv.reliance.idxmax()]
        body.append([
            label.split(" (")[0], nets[base],
            f"{len(group)} ({lv.dose.nunique()} \u00d7 {group.seed.nunique()})",
            f"{f1(lv.dose.min())} to {f1(lv.dose.max())}".replace("-", "\u2212"),
            f"{f1(lv.reliance.min())}{DASH}{f1(lv.reliance.max())}",
            f"{f1(lv.test_accuracy.min())}{DASH}{f1(lv.test_accuracy.max())}",
            f"{f1(lowest.test_accuracy_no_shortcut)} \u2192 {f1(highest.test_accuracy_no_shortcut)}",
        ])
    return table(
        "Reliance reached per setting. Values are means over seeds at each dose; the reliance range is the "
        "lowest and highest per-dose mean. Accuracy without the shortcut is shown at the dose with the lowest "
        "and the highest mean reliance",
        ["Data, shortcut", "Network", "Models (doses \u00d7 seeds)", "Dose range", "Reliance (%)",
         "Test accuracy with shortcut (%)", "Accuracy without shortcut (%), low \u2192 high reliance"],
        body, widths=[3.0, 2.6, 2.0, 1.9, 1.9, 2.1, 2.6],
        note="Source: results/dose_response/models.csv (per-dose means as in summary.md). Reliance is the "
             "percentage of test predictions that follow a shortcut placed at a wrong class's position.")


def format_rho(r):
    mark = "" if r.p_value < 0.05 else "\u2020"
    return f"{signed(r.responsiveness_rho)}{mark} [{signed(r.rho_ci_low)}, {signed(r.rho_ci_high)}]"


def format_switch(r):
    point = "none" if pd.isna(r.switch_point_reliance) else f1(r.switch_point_reliance)
    if pd.isna(r.switch_ci_low):
        return point
    return f"{point} [{f1(r.switch_ci_low)}, {f1(r.switch_ci_high)}]"


def table_responsiveness(comparison, caption):
    body = []
    for base, shortcut, label in SETTINGS:
        for i, metric in enumerate(METRICS):
            r = row(comparison, base, shortcut, metric)
            body.append([label if i == 0 else "", metric, format_rho(r), f2(r.decision_accuracy),
                         format_switch(r),
                         f"{f2(r.preference_low_reliance)} \u2192 {f2(r.preference_high_reliance)}"])
    return table(
        caption,
        ["Setting", "Metric", "Responsiveness \u03c1 [95% CI]", "Decision accuracy",
         "Switch point, % reliance [95% CI]", "Preference, lowest \u2192 highest reliance"],
        body, widths=[3.4, 2.6, 3.4, 1.7, 3.2, 2.5],
        note=f"Source: results/dose_response/responsiveness.csv, comparison = \u201c{comparison}\u201d. "
             "\u2020: not significant (p \u2265 0.05). CI: 95% bootstrap interval, 2000 resamples of the seeds "
             "within each dose. \u201cnone\u201d: the mean preference curve does not cross 0.5 in the measured "
             "range; an interval is still shown when at least half of the bootstrap resamples cross. "
             "Ideal metric: \u03c1 near 1, decision accuracy 1, switch point near 50%.")


def table_controls():
    comps = [(FULL, "S\u2013O"), (R_OBJECT, "R\u2013O"), (P_OBJECT, "P\u2013O"),
             (S_RANDOM, "S\u2013R"), (P_RANDOM, "P\u2013R")]
    small = f"Small CNN (reliance {f1(lowest_reliance('cifar10', 'patch'))}%)"
    resnet = f"ResNet-18 (reliance {f1(lowest_reliance('cifar10_resnet18', 'patch'))}%)"
    top = ["Metric"] + [small] * len(comps) + [resnet] * len(comps)
    second = ["Metric"] + [c for _, c in comps] * 2
    body = []
    for metric in METRICS:
        values = [metric]
        for base in ("cifar10", "cifar10_resnet18"):
            values += [f2(row(c, base, "patch", metric).preference_low_reliance) for c, _ in comps]
        body.append(values)
    return table(
        "CIFAR-10 patch shortcut: preference at the lowest reliance level for the comparisons of the main "
        "maps and the controls. Each entry is the share of images on which the metric rates the first map "
        "above the second",
        None, body, widths=[2.8] + [1.25] * 10, size=8, header_rows=[top, second],
        note="S = shortcut map, O = full object map (for CIFAR-10 a centred Gaussian), R = random ordering, "
             "P = placebo (the shortcut's shape at a wrong class's position). At this reliance the model "
             "barely uses the shortcut, so the ideal values are S\u2013O low, R\u2013O low, P\u2013O low, "
             "S\u2013R \u2248 0.5, P\u2013R \u2248 0.5. Source: results/dose_response/responsiveness.csv, "
             "column preference_low_reliance.")


def table_methods():
    body = []
    for method in METHODS:
        values = [method]
        for base, shortcut, _ in SETTINGS:
            r = mrow(base, shortcut, method)
            mark = "" if r.p_value < 0.05 else "\u2020"
            values.append(f"{signed(r.responsiveness_rho)}{mark}\n({f2(r.mass_low_reliance)} \u2192 "
                          f"{f2(r.mass_high_reliance)})")
        body.append(values)
    short = ["MNIST patch", "MNIST texture", "Pneumonia patch", "Pneumonia texture",
             "CIFAR-10 small CNN", "CIFAR-10 ResNet-18"]
    return table(
        "Explanation methods: Spearman \u03c1 between reliance and the share of positive attribution on the "
        "shortcut pixels, with that share at the lowest and the highest dose in brackets",
        ["Method"] + short, body, widths=[3.2] + [2.2] * 6,
        note="Source: results/dose_response/method_responsiveness.csv. \u2020: not significant (p \u2265 0.05). "
             "For comparison, the shortcut covers 9 pixels (patch) or about 10% of the pixels (texture).")


# ------------------------------------------------------------------ the paper


def build():
    setup()
    figs = {
        "main": rearrange_panels(DR / "metric_dose_response.png", FIG_OUT / "fig_full_object.png"),
        "matched": rearrange_panels(DR / "metric_dose_response_size_matched.png", FIG_OUT / "fig_size_matched.png"),
        "s_random": rearrange_panels(DR / "metric_dose_response_shortcut_vs_random.png", FIG_OUT / "fig_shortcut_vs_random.png"),
        "r_object": rearrange_panels(DR / "metric_dose_response_random_vs_object.png", FIG_OUT / "fig_random_vs_object.png"),
        "p_random": rearrange_panels(DR / "metric_dose_response_placebo_vs_random.png", FIG_OUT / "fig_placebo_vs_random.png"),
        "methods": rearrange_panels(DR / "method_dose_response.png", FIG_OUT / "fig_methods.png"),
    }

    # ---- front matter
    notice = para(DRAFT_NOTICE, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12)
    notice.runs[0].font.color.rgb = RGBColor(0xC0, 0, 0)
    para("Target: SN Computer Science (Springer), single-column draft. All numbers are generated from the "
         "project's result files by paper/build_paper.py.", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(12)
    run = t.add_run(TITLE)
    run.bold, run.font.size = True, Pt(16)
    para("Md Junaid Faizan Shaikh", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    para("Department of Computer Science and Engineering, Khaja Bandanawaz University, Kalaburagi, India",
         align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=0)
    para("E-mail: [email]", align=WD_ALIGN_PARAGRAPH.CENTER, size=10)

    mnist_rho = span(MNIST_RHO)
    pneu_rho = span(PNEU_RHO)
    abstract = (
        "Faithfulness metrics such as deletion and insertion curves are widely used to compare explanation "
        "methods, but whether they notice a change in what a model actually relies on has rarely been tested. "
        "We propose a dose-response protocol for testing the metrics themselves: a signed training dose sets "
        "a network's measured reliance on an injected class-coded shortcut (a local patch or a diffuse texture) "
        f"from near 0% to near 100%, and fixed oracle explanations are scored. We trained {N_MODELS} models on "
        "MNIST, PneumoniaMNIST and CIFAR-10, including a standard ResNet-18. On MNIST, every metric's "
        "preference for the shortcut explanation tracks "
        f"reliance closely (Spearman \u03c1 = {mnist_rho}). Against the full object explanation, black and blur "
        f"deletion switch to the shortcut at {span(MP_FULL_EARLY.switch_point_reliance, f1)}% reliance instead "
        f"of 50%, and against a size-matched object explanation all four metrics switch at "
        f"{span(MP_MATCHED.switch_point_reliance, f1)}%: the metrics partly reward explanation size. Placebo "
        "and random-map controls show that explanations of the same form are ranked correctly, but forms that "
        "differ cause bias: on CIFAR-10 at the lowest reliance, black and blur "
        f"deletion prefer a random pixel ordering over a centred object map {pct_span(CIFAR_RAND_BB)} of the time "
        f"with a small CNN and {pct_span(RESNET_RAND_BB)} with ResNet-18, while inpainting deletion and insertion "
        f"do so only {pct_span(ALL_RAND_II)} of the time. On chest X-rays responsiveness falls to "
        f"\u03c1 = {pneu_rho}. Insertion fails on the diffuse shortcut, and inpainting-based deletion is the only "
        "variant that responds significantly in every setting and comparison. Metrics should be tested at "
        "several controlled reliance levels, with controls for explanation size and form, before they are "
        "used to compare explanation methods.")
    p = para("")
    add_runs(p, "**Abstract** " + abstract)
    para("**Keywords** Explainable AI \u00b7 Evaluation metrics \u00b7 Faithfulness \u00b7 Deletion and "
         "insertion \u00b7 Shortcut learning \u00b7 Meta-evaluation")

    # ---- 1 Introduction
    heading("1 Introduction")
    para("Convolutional neural networks reach high accuracy on image classification, but they do not say why "
         "they make a prediction. Attribution methods such as Saliency " + cite("simonyan") + ", SmoothGrad "
         + cite("smilkov") + ", Integrated Gradients " + cite("sundararajan") + ", Grad-CAM " + cite("selvaraju")
         + " and Occlusion " + cite("zeiler") + " answer this with a heatmap over the input pixels. Because these "
         "methods often disagree, researchers compare them with evaluation metrics. The most common are "
         "perturbation metrics: deletion curves " + cite("samek") + ", which remove the pixels a map marks as "
         "important and measure how fast the model's confidence falls, and insertion curves " + cite("petsiuk")
         + ", which restore them into a blurred image. The model-randomization sanity check " + cite("adebayo2018")
         + " is a further widely used test.")
    para("The metrics decide which explanation methods are trusted, yet they are rarely validated themselves, "
         "because the true explanation of a trained network is normally unknown. Recent work shows that metrics "
         "disagree with one another " + cite("tomsett", "metaquantus", "latec") + ", that they depend on hidden "
         "choices such as the value that replaces removed pixels " + cite("rong", "baseline") + ", and that the "
         "sanity check has conceptual problems " + cite("binder", "hedstrom2024") + ".")
    para("A common failure of neural networks is shortcut learning: the model uses an irrelevant cue, such as "
         "a marker on an X-ray, instead of the object " + cite("ross", "sun") + ". Detecting such reliance is "
         "one of the main reasons explanations are used in practice. A useful metric should therefore notice "
         "when a model's reasoning moves from the object to a shortcut. Existing studies test explanation "
         "methods against a shortcut of one fixed strength " + cite("zhou", "backx", "sun") + ", or test the "
         "metrics without a ground truth " + cite("tomsett", "metaquantus") + ". This paper asks a different "
         "question:")
    para("_When a model's reliance on a shortcut increases in controlled steps, do XAI evaluation metrics shift "
         "their preference from object explanations to shortcut explanations, and at what level of reliance?_",
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para("To answer it we borrow the dose-response idea: a training dose sets how much each model relies on "
         "an injected shortcut, the reliance is measured on held-out data, and fixed oracle explanations are "
         "scored by every metric at every dose. In our literature searches we did not find published work that "
         "varies a model's reliance on a shortcut in controlled steps to test the evaluation metrics rather "
         "than the explanation methods; this will be confirmed by a full search before submission. Our "
         "contributions are:")
    bullets([
        "**A dose-response protocol for meta-evaluating XAI metrics.** A signed training dose sets the measured "
        f"reliance on a patch or texture shortcut from near 0% to near 100% ({N_MODELS} models on three "
        "datasets and two network sizes). Fixed oracle maps make the test independent of any explanation method.",
        "**Evidence that metric responsiveness depends on the data.** The metrics track reliance closely on "
        f"MNIST (\u03c1 = {mnist_rho}) but weakly, and in {len(PNEU_NONSIG)} of {len(PNEU)} cases not "
        f"significantly, on chest X-rays (\u03c1 = {pneu_rho}).",
        "**A size and form confound in oracle-based evaluation.** Matching the object map's size to the shortcut "
        f"moves the MNIST patch switch point from {span(MP_FULL.switch_point_reliance, f1)}% to "
        f"{span(MP_MATCHED.switch_point_reliance, f1)}% reliance. Placebo and random-map controls show that "
        "maps of the same form are ranked correctly, and that on CIFAR-10 black and blur deletion prefer a "
        "random ordering over a compact object map when the model ignores the shortcut.",
        "**Practical guidance.** Inpainting-based deletion was the most robust variant, insertion fails on a "
        "diffuse shortcut, and some attribution methods put less weight on a diffuse shortcut as the model "
        "relies on it more.",
    ])
    para("A smaller supporting study with a fixed shortcut checks the sanity check against ground truth. "
         "Section 2 reviews related work, Sections 3 and 4 describe the protocol and the setup, Section 5 "
         "reports the results, and Sections 6\u20138 discuss them, state the limitations and conclude.")

    # ---- 2 Related work
    heading("2 Related Work")
    heading("2.1 Explanation methods and faithfulness metrics", 2)
    para("Gradient-based methods attribute a prediction to pixels through the gradient of the class score "
         + cite("simonyan") + ", averaged over noisy copies of the input " + cite("smilkov") + " or "
         "accumulated along a path from a baseline image " + cite("sundararajan") + ". Grad-CAM weights the "
         "last convolutional feature maps by their gradients " + cite("selvaraju") + ", and Occlusion measures "
         "the change in output when image patches are hidden " + cite("zeiler") + ". We compute all five with "
         "Captum " + cite("captum") + ". Faithfulness is commonly measured by perturbation: deletion "
         + cite("samek") + " and insertion " + cite("petsiuk") + ". ROAD " + cite("rong") + " showed that the "
         "shape of the removed region can leak class information and proposed replacing removed pixels by "
         "noisy inpainting; our inpainting variant is inspired by it. \u201cBack to the Baseline\u201d "
         + cite("baseline") + " shows that the replacement baseline of deletion and insertion favours some "
         "methods.")
    heading("2.2 Testing the metrics", 2)
    para("Tomsett et al. " + cite("tomsett") + " found saliency metrics statistically unreliable. MetaQuantus "
         + cite("metaquantus") + " meta-evaluates metrics by their resilience to noise and their reaction to "
         "randomisation, without a ground truth. LATEC " + cite("latec") + " compares 20 metrics on 17 methods "
         "and finds that the metrics conflict. A study of fidelity metrics " + cite("fidelity") + " checks them "
         "against a transparent model. AttributionLab " + cite("attributionlab") + " sets network weights by "
         "hand so the true explanation is known. The model-randomization sanity check " + cite("adebayo2018")
         + " has been criticised for conceptual shortcomings " + cite("binder", "hedstrom2024") + ". None of "
         "these varies what a trained model relies on in controlled steps.")
    heading("2.3 Shortcuts as ground truth", 2)
    para("Decoy MNIST " + cite("ross") + " adds a class-coded patch to digits so that the model's reason is "
         "known. Zhou et al. " + cite("zhou") + " induce ground-truth features to test attribution methods, "
         "BackX " + cite("backx") + " uses backdoor triggers as a benchmark, Sun et al. " + cite("sun")
         + " ask whether explanations detect spurious correlations in chest X-rays, and Adebayo et al. "
         + cite("adebayo2022") + " study post hoc explanations under unknown spurious correlations. Arras et "
         "al. " + cite("arras") + " define relevance mass and rank accuracy against a ground-truth mask. These "
         "works evaluate explanation methods at a fixed shortcut strength. We instead evaluate the metrics, "
         "and vary the strength.")

    # ---- 3 Method
    heading("3 Method")
    heading("3.1 Shortcuts", 2)
    para("Two shortcut types are added to every image. The **patch** is a 3\u00d73 square set to the maximum "
         "value (white in every channel) whose position encodes the class. With ten classes, the positions "
         "lie along the top and bottom borders (five each), where MNIST digits never are; with two classes "
         "(PneumoniaMNIST) they are the two top corners, like the markers often burned into real X-rays. The "
         "**texture** gives each class a fixed random set of about 10% of all pixels, spread over the whole "
         "image, which are brightened by 0.3 (clipped at 1). The texture is used on the two 28\u00d728 "
         "datasets only.")
    heading("3.2 Controlling reliance with a signed dose", 2)
    para("Because the shortcut is easier to learn than the image, models trained with an always-correct "
         "shortcut already rely on it heavily. Two mechanisms are therefore combined into one signed dose _d_. "
         "For _d_ > 0, a share _d_ of training images is replaced by an image of another class while the label "
         "and the shortcut are kept, so the image becomes less reliable and the model relies on the shortcut "
         "more. For _d_ < 0, the shortcut of a share |_d_| of training images points to a randomly chosen "
         "wrong class, so the shortcut becomes less reliable and the model relies on it less. On MNIST and "
         "PneumoniaMNIST |_d_| is capped at 0.4, so neither cue ever points away from the label more often "
         "than towards it on the binary task. The doses are "
         + ", ".join(f"{d:g}" for d in DOSES["mnist"]).replace("-", "\u2212")
         + "; the small steps near 0 cover the range in which reliance changes fastest. CIFAR-10 models learn "
         "the patch almost at once, so their doses reach further down ("
         + ", ".join(f"{d:g}" for d in DOSES["cifar10"]).replace("-", "\u2212")
         + "). With ten classes, a shortcut noise of 0.8 still points to the true class 20% of the time, "
         "against about 9% for each wrong class.")
    heading("3.3 Measuring reliance", 2)
    para("Reliance is measured, not assumed. Every test image receives the shortcut of a different, randomly "
         "chosen class (a conflicting shortcut), and reliance is the percentage of predictions that follow the "
         "shortcut's class. A model that ignores the shortcut has a reliance near 0%, and one that only reads "
         "the shortcut has a reliance near 100%. We also report test accuracy with the correct shortcut and "
         "with no shortcut. All analyses use measured reliance rather than the dose, because reliance varies "
         "between seeds near _d_ = 0.")
    heading("3.4 Oracle explanations", 2)
    para("To test the metrics independently of any explanation method, every metric scores five fixed maps "
         "on the same aligned test images (shortcut and image agree), explaining the true label:")
    bullets([
        "**Shortcut map**: 1 on the shortcut pixels, 0 elsewhere.",
        "**Full object map**: on the grayscale datasets, the image's own pixel intensities with the shortcut "
        "pixels set to 0. CIFAR-10 images have no clean background, so the object map is a centred Gaussian "
        "(\u03c3 = image size / 4) with the shortcut pixels set to 0, since CIFAR-10 objects are mostly centred. "
        "This is a modelling choice.",
        "**Size-matched object map**: a binary map of the _k_ highest-valued pixels of the object map, where "
        "_k_ is the number of shortcut pixels. It has the same size and form as the shortcut map.",
        "**Placebo map**: the shortcut's shape at the position (or texture pattern) of a randomly chosen wrong "
        "class. The image has no shortcut there, so the placebo never carries evidence for the true label.",
        "**Random map**: independent uniform scores for every pixel, i.e. a fully random pixel ordering.",
    ])
    para("Metrics rank pixels by their map value and break ties randomly. A binary map therefore orders its "
         "marked pixels first, in random order, followed by all other pixels in random order. For the placebo, "
         "the marked pixels carry no evidence, so it is expected to behave almost like the random map; the "
         "random map tests exactly this.")
    heading("3.5 Metrics under test", 2)
    para("Four perturbation metrics are tested. Each runs in 20 steps; step _i_ changes the top _i_/20 of the "
         "pixels in the map's ranking, and the score is the area under the curve of the model's probability "
         "for the explained class (trapezoidal rule over [0, 1]). **Deletion** replaces pixels and a lower area "
         "is better. Three replacement values are used: black (0), a blurred copy of the image (Gaussian blur, "
         "kernel 25, \u03c3 = 4, chosen so a fully blurred image gives near-chance confidence), and inpainting. "
         "The inpainting variant fills removed pixels with the per-channel mean of the remaining pixels, refines "
         "them by 100 iterations of averaging their four neighbours, and adds Gaussian noise (standard deviation "
         "0.05). It is inspired by ROAD " + cite("rong") + " but is not the exact ROAD procedure. **Insertion** "
         "starts from the blurred image and restores pixels; a higher area is better.")
    heading("3.6 Scores for each metric", 2)
    para("For two maps A and B, the **preference** of a metric for A is the share of test images on which it "
         "rates A as the better explanation. Area differences of at most 10\u207b\u2074 are ties and count as "
         "one half; tie rates are reported in the result files. The main comparison is the shortcut map against "
         "the object map. For a metric that responds to the model, this preference should be near 0 when "
         "reliance is low, near 1 when it is high, and cross 0.5 near 50% reliance. Over all models of one "
         "setting we compute:")
    bullets([
        "**Responsiveness**: the Spearman correlation \u03c1 between reliance and preference, with each model one "
        "observation (ideal: close to 1).",
        "**Decision accuracy**: the share of models for which (preference > 0.5) equals (reliance > 50%).",
        "**Switch point**: the reliance at which the mean preference per dose, ordered by mean reliance, first "
        "crosses 0.5 (linear interpolation). The 50% ideal is a modelling assumption.",
    ])
    para("Uncertainty is estimated by a bootstrap with 2000 resamples in which, within each dose, the seeds are "
         "drawn with replacement; the dose design stays fixed and only the seed-to-seed variation is resampled. "
         "We report 95% percentile intervals for \u03c1 and the switch point. The switch-point interval is "
         "computed over the resamples that cross 0.5, and only if at least half of them do.")
    heading("3.7 Explanation methods", 2)
    para("Five methods are run on every model with Captum " + cite("captum") + ": Saliency (absolute gradient) "
         + cite("simonyan") + ", SmoothGrad (25 noisy samples, noise standard deviation 0.15) " + cite("smilkov")
         + ", Integrated Gradients (50 steps, black baseline) " + cite("sundararajan") + ", Grad-CAM on the "
         "last convolutional feature maps, bilinearly upsampled " + cite("selvaraju") + ", and Occlusion "
         "(4\u00d74 black window, stride 2) " + cite("zeiler") + ". For colour images the channel attributions "
         "are summed. For each method we compute the share of positive attribution that falls on the shortcut "
         "pixels (relevance mass accuracy " + cite("arras") + "), average it over the test images of a model, "
         "and correlate it with reliance.")

    # ---- 4 Setup
    heading("4 Experimental Setup")
    para("**Datasets.** MNIST (handwritten digits, 10 classes, 28\u00d728 grayscale), PneumoniaMNIST "
         + cite("medmnist") + " (paediatric chest X-rays derived from " + cite("kermany") + ", normal vs "
         "pneumonia, 28\u00d728 grayscale) and CIFAR-10 (natural images, 10 classes, 32\u00d732 colour). For "
         "MNIST and CIFAR-10, 5000 training images are held out for validation with a fixed split; "
         "PneumoniaMNIST uses its official splits. Validation images receive the same dose as the training images.")
    para("**Networks and training.** The 28\u00d728 datasets use a small CNN with four convolutional layers "
         "(5\u00d75 with 16 and 32 filters, then 3\u00d73 with 64 and 128 filters, two max-pooling layers, dropout "
         "0.25) and two fully connected layers. CIFAR-10 uses a larger CNN with four 3\u00d73 convolutional layers "
         "with batch normalisation (0.77M parameters) and a standard ResNet-18 (11.2M parameters) adapted to "
         "32\u00d732 inputs with a 3\u00d73 stride-1 first convolution and no initial max-pooling. All models are "
         f"trained with Adam (learning rate 10\u207b\u00b3, batch size 128) for {MODELS[MODELS.base == 'mnist'].epochs.iloc[0]} "
         f"epochs (MNIST), {MODELS[MODELS.base == 'pneumonia_mnist'].epochs.iloc[0]} (PneumoniaMNIST), "
         f"{MODELS[MODELS.base == 'cifar10'].epochs.iloc[0]} (CIFAR-10 CNN) and "
         f"{MODELS[MODELS.base == 'cifar10_resnet18'].epochs.iloc[0]} (ResNet-18), keeping the checkpoint with "
         "the best validation accuracy. No data augmentation is used on CIFAR-10: in a short test it slowed "
         "learning too much for the 10-epoch budget.")
    para(f"**Models.** On MNIST and PneumoniaMNIST: 12 doses \u00d7 3 seeds \u00d7 2 shortcuts \u00d7 2 datasets "
         f"= {N_SMALL} models. On CIFAR-10 with the patch shortcut: 6 doses \u00d7 {len(SEEDS['cifar10'])} seeds "
         f"with the CIFAR CNN ({N_CIFAR} models) and 6 doses \u00d7 {len(SEEDS['cifar10_resnet18'])} seeds with "
         f"ResNet-18 ({N_RESNET} models), {N_MODELS} models in total. ResNet-18 was trained on a free Kaggle "
         f"GPU (NVIDIA T4), about {RESNET_MINUTES:.0f} minutes per model; all other models were trained on a "
         f"laptop CPU (about {CIFAR_MINUTES:.0f} minutes per CIFAR-10 CNN, under 4 minutes per 28\u00d728 model).")
    para("**Evaluation.** Every model is scored on 100 aligned test images, the same images for every model of "
         "a setting. Reliance and accuracy are measured on the full test set. The statistics use SciPy "
         "(Spearman \u03c1 and its p-value).")

    # ---- 5 Results
    heading("5 Results")
    heading("5.1 The dose controls reliance", 2)
    t_rel = COUNTERS["table"] + 1
    mt = MODELS[(MODELS.base == "mnist") & (MODELS.shortcut == "texture") & (MODELS.dose == 0.005)].reliance
    para(f"Table {t_rel} shows that the signed dose moves reliance across almost the full range. On MNIST the "
         f"per-dose mean reliance runs from {f1(lowest_reliance('mnist', 'patch'))}% to "
         f"{f1(levels('mnist', 'patch').reliance.max())}% with the patch and from "
         f"{f1(lowest_reliance('mnist', 'texture'))}% to {f1(levels('mnist', 'texture').reliance.max())}% with "
         f"the texture; on CIFAR-10 it runs from {f1(lowest_reliance('cifar10', 'patch'))}% to "
         f"{f1(levels('cifar10', 'patch').reliance.max())}% (small CNN) and from "
         f"{f1(lowest_reliance('cifar10_resnet18', 'patch'))}% to "
         f"{f1(levels('cifar10_resnet18', 'patch').reliance.max())}% (ResNet-18). On PneumoniaMNIST the lowest reliance reached was "
         f"{f1(lowest_reliance('pneumonia_mnist', 'patch'))}% (patch) and "
         f"{f1(lowest_reliance('pneumonia_mnist', 'texture'))}% (texture), not 0. As reliance rises, accuracy "
         "without the shortcut falls, which confirms that the models use the shortcut in place of the image. "
         "On CIFAR-10 no model fell between about 45% and 96% reliance. Near _d_ = 0 reliance varies strongly "
         f"between seeds; for example, on MNIST texture at _d_ = 0.005 the three seeds reached "
         + ", ".join(f"{v:.0f}%" for v in sorted(mt)) + ". This is why all results are analysed against "
         "measured reliance.")
    table_reliance()
    fig_examples = figure(EX / "decoy_mnist_examples.png",
                          "Examples of the patch shortcut and the five explanation methods (supporting study, Decoy "
                          "MNIST). Each test image carries the 3\u00d73 patch of a wrong class; the model predicts "
                          "the patch's class, and the methods differ in how much of their attribution they place "
                          "on the patch. Source: results/explanations/decoy_mnist_examples.png",
                          width=Inches(4.3))
    para(f"Fig. {fig_examples} illustrates the patch shortcut and how differently the five methods treat it.")

    heading("5.2 Responsiveness against the full object map", 2)
    f_main = COUNTERS["fig"] + 1
    t_full = COUNTERS["table"] + 1
    mp_low = MP_FULL.preference_low_reliance
    para(f"Fig. {f_main} and Table {t_full} show the main result. On MNIST the preference for the shortcut "
         f"map rises steadily with reliance for every metric (\u03c1 = "
         f"{span(rows([FULL], ['mnist']).responsiveness_rho)}). At the lowest reliance the metrics strongly prefer "
         f"the object map (preference {span(mp_low)} on the patch shortcut), which is correct. However, black and "
         f"blur deletion switch to the shortcut map at {span(MP_FULL_EARLY.switch_point_reliance, f1)}% reliance "
         f"(95% intervals {f1(MP_FULL_EARLY.switch_ci_low.min())}\u2013{f1(MP_FULL_EARLY.switch_ci_high.max())}%), "
         "clearly before 50%. Inpainting deletion and insertion switch at "
         f"{span(MP_FULL_LATE.switch_point_reliance, f1)}%, but their intervals reach "
         f"{span(MP_FULL_LATE.switch_ci_high, f1)}%, so they are not clearly early.")
    para(f"On the chest X-rays responsiveness is much weaker. Across the full and size-matched comparisons, "
         f"\u03c1 ranges from {f2(PNEU_RHO.min())} to {f2(PNEU_RHO.max())}, and {len(PNEU_NONSIG)} of the "
         f"{len(PNEU)} metric\u2013comparison combinations are not significant: "
         + "; ".join(f"{r.metric} on the {r.shortcut} shortcut, {r.comparison} (\u03c1 = {f2(r.responsiveness_rho)}, "
                     f"p = {f2(r.p_value)})" for r in PNEU_NONSIG.itertuples()) + ". "
         "The decision accuracies are also lower, partly because the lowest reliance reached was about 15%.")
    cf, rf = CIFAR_FULL, RESNET_FULL
    para(f"On CIFAR-10 every metric's preference rises with reliance (\u03c1 = {span(cf.responsiveness_rho)} for "
         f"the small CNN and {span(rf.responsiveness_rho)} for ResNet-18). But black and blur deletion prefer the "
         f"shortcut map even at the lowest reliance ({pct_span(CIFAR_SHORT_BB)} of images with the small CNN, "
         f"{pct_span(RESNET_SHORT_BB)} with ResNet-18), so their preference never crosses 0.5 and their decision "
         "accuracy is only "
         f"{span(rows([FULL], ['cifar10', 'cifar10_resnet18'], ['patch'], BB).decision_accuracy)}. Inpainting "
         f"deletion and insertion start low ({pct_span(low(FULL, 'cifar10', II) + low(FULL, 'cifar10_resnet18', II))}) "
         f"and switch at {span(rows([FULL], ['cifar10', 'cifar10_resnet18'], ['patch'], II).switch_point_reliance, f1)}% "
         "reliance. Because no CIFAR-10 model fell between 45% and 96% reliance, the CIFAR-10 switch points are "
         "interpolated over a wide gap. Section 5.4 shows where the early preference of black and blur deletion "
         "comes from.")
    figure(figs["main"],
           "Preference for the shortcut map over the full object map as measured reliance grows, for the four "
           "metrics in the six settings. Small dots are single models; lines connect the per-dose means. Ideal: "
           "a rise from 0 to 1 that crosses 0.5 (horizontal line) at 50% reliance (dotted line). Panels "
           "rearranged from results/dose_response/metric_dose_response.png")
    table_responsiveness(FULL, "Metric responsiveness, shortcut map vs full object map")

    heading("5.3 Explanation size is a confound", 2)
    f_matched = COUNTERS["fig"] + 1
    t_matched = COUNTERS["table"] + 1
    para("The full object map marks far more pixels than the shortcut map, so part of the preference for it at "
         f"low reliance may come from its size. Fig. {f_matched} and Table {t_matched} replace it with a "
         "size-matched object map. On MNIST patch the preference for the shortcut at the lowest reliance rises "
         f"from {span(mp_low)} to {span(MP_MATCHED.preference_low_reliance)}, and all four metrics switch at "
         f"{span(MP_MATCHED.switch_point_reliance, f1)}% reliance (intervals "
         f"{f1(MP_MATCHED.switch_ci_low.min())}\u2013{f1(MP_MATCHED.switch_ci_high.max())}%) instead of "
         f"{span(MP_FULL.switch_point_reliance, f1)}%. The metrics still respond to reliance (\u03c1 = "
         f"{span(MP_MATCHED.responsiveness_rho)}), but much of what looks like \u201cthe metric recognises the "
         "object\u201d is the object map simply being larger. On PneumoniaMNIST texture, black deletion does not "
         f"respond to reliance at all under this control (\u03c1 = {f2(row(MATCHED, 'pneumonia_mnist', 'texture', 'Deletion (black)').responsiveness_rho)}, "
         f"p = {f2(row(MATCHED, 'pneumonia_mnist', 'texture', 'Deletion (black)').p_value)}). On CIFAR-10 the "
         "size-matched map is a compact block of central pixels, and all metrics prefer the shortcut over it "
         f"about half the time or more at the lowest reliance "
         f"({span(rows([MATCHED], ['cifar10', 'cifar10_resnet18'], ['patch']).preference_low_reliance)}).")
    figure(figs["matched"],
           "Preference for the shortcut map over the size-matched object map (same number of pixels as the "
           "shortcut). Layout as in Fig. " + str(f_main) + ". Panels rearranged from "
           "results/dose_response/metric_dose_response_size_matched.png")
    table_responsiveness(MATCHED, "Metric responsiveness, shortcut map vs size-matched object map")

    heading("5.4 Placebo and random controls: same form is ranked correctly", 2)
    f_sr = COUNTERS["fig"] + 1
    t_ctrl = COUNTERS["table"] + 1
    sf_low = SAME_FORM.preference_low_reliance
    para(f"Fig. {f_sr} compares the shortcut map with a random ordering. On the patch shortcut, every metric "
         f"starts near 0.5 at the lowest reliance (shortcut vs random and shortcut vs placebo: {span(sf_low)}) "
         "and rises as reliance grows, to "
         f"{span(SF_CLEAN.preference_high_reliance)} on MNIST and CIFAR-10 and "
         f"{span(SF_PNEU.preference_high_reliance)} on PneumoniaMNIST, whose highest-reliance models still use "
         "the image to some extent. This is the correct behaviour: when the model ignores the shortcut, the "
         "shortcut map is as uninformative as a random ordering, and when the model reads the shortcut, the "
         "shortcut map is the better explanation. When the compared maps have the same form, the metrics "
         "respond to the model.")
    f_ro = f_sr + 1
    f_pr = f_sr + 2
    para(f"The placebo behaves like a random ordering, as Section 3.4 predicts (Fig. {f_pr}). For the three "
         f"deletion variants, the preference for the placebo over the random map stays between "
         f"{f2(min(PR_DEL.preference_low_reliance.min(), PR_DEL.preference_high_reliance.min()))} and "
         f"{f2(max(PR_DEL.preference_low_reliance.max(), PR_DEL.preference_high_reliance.max()))} at the lowest "
         "and highest reliance in every setting. Insertion on MNIST is an exception: its preference rises from "
         f"{span(PR_INS_MNIST.preference_low_reliance)} to {span(PR_INS_MNIST.preference_high_reliance)} "
         f"(\u03c1 = {span(PR_INS_MNIST.responsiveness_rho)}). Consequently the rise of the placebo, and of the "
         f"random map, over the full object map at high reliance (Fig. {f_ro}) is also correct: once the model "
         "reads only the shortcut, the object map explains the model worse than a random ordering.")
    para(f"The errors appear when maps of different form are compared. Table {t_ctrl} shows the CIFAR-10 "
         "comparisons at the lowest reliance, where the model barely uses the shortcut. Black and blur deletion "
         f"prefer a fully random ordering over the object map {pct_span(CIFAR_RAND_BB)} of the time with the small "
         f"CNN and {pct_span(RESNET_RAND_BB)} with ResNet-18, about as often as they prefer the shortcut map "
         f"({pct_span(CIFAR_SHORT_BB)} and {pct_span(RESNET_SHORT_BB)}). Their apparent detection of a shortcut "
         "that the model ignores is therefore a preference for scattered maps over the compact object map, not "
         "a response to the model. Inpainting deletion and insertion do not show this bias: they prefer the "
         f"random map over the object map only {pct_span(CIFAR_RAND_II)} (small CNN) and "
         f"{pct_span(RESNET_RAND_II)} (ResNet-18) of the time. The pattern is the same for both networks, so it "
         "is not specific to small networks. A possible cause is that removing scattered pixels damages a "
         "natural image more than removing a compact region, the information-leakage effect described by ROAD "
         + cite("rong") + ". Because the CIFAR-10 object map is a centred Gaussian and not a real object mask, "
         "this cause is a hypothesis.")
    figure(figs["s_random"],
           "Control 1: preference for the shortcut map over a random pixel ordering (same-form comparison). "
           "Ideal: about 0.5 at 0% reliance, rising to 1. Panels rearranged from "
           "results/dose_response/metric_dose_response_shortcut_vs_random.png")
    figure(figs["r_object"],
           "Control 2: preference for the random ordering over the full object map. Ideal: low at low "
           "reliance; a rise at high reliance is correct, because the object map then explains the model worse "
           "than random. Panels rearranged from results/dose_response/metric_dose_response_random_vs_object.png")
    figure(figs["p_random"],
           "Control 3: preference for the placebo map over the random ordering. Ideal: about 0.5 at every "
           "reliance, since the placebo carries no evidence. Panels rearranged from "
           "results/dose_response/metric_dose_response_placebo_vs_random.png")
    table_controls()

    heading("5.5 A diffuse shortcut", 2)
    mt_ins_random = MT_INS_RANDOM
    para("The texture shortcut is spread over the image. On MNIST texture, insertion switches to the shortcut "
         f"map only at {f1(MT_INS.switch_point_reliance)}% reliance and its preference reaches only "
         f"{f2(MT_INS.preference_high_reliance)} at the highest reliance, while black and inpainting deletion "
         f"switch at {span(MT_DEL_BI.switch_point_reliance, f1)}%. A possible reason is discussed in Section 6. "
         "Against a random map, blur deletion does not separate the real texture shortcut from random "
         f"(\u03c1 = {f2(MT_BLUR_RANDOM.responsiveness_rho)}, {p_text(MT_BLUR_RANDOM.p_value)}; preference "
         f"{f2(MT_BLUR_RANDOM.preference_low_reliance)} \u2192 {f2(MT_BLUR_RANDOM.preference_high_reliance)}). "
         "Insertion moves the wrong way: the more the model relies on the texture, the more often it rates "
         f"a random ordering above the true shortcut map (\u03c1 = {signed(mt_ins_random.responsiveness_rho)}, "
         f"{p_text(mt_ins_random.p_value)}; preference {f2(mt_ins_random.preference_low_reliance)} \u2192 "
         f"{f2(mt_ins_random.preference_high_reliance)}). Against the placebo map the same pattern is stronger "
         f"(\u03c1 = {signed(MT_INS_PLACEBO.responsiveness_rho)}, {p_text(MT_INS_PLACEBO.p_value)}). The texture "
         "is itself a scattered set of pixels, so it has the same form as a random ordering.")
    robust_text = ", ".join(ROBUST_ALL) if ROBUST_ALL else "no variant"
    ok, n, min_rho = ROBUST["Deletion (inpaint)"]
    para(f"Across all six settings and the four comparisons of the shortcut map with another map (full object, "
         f"size-matched object, placebo, random), {robust_text} had a significant positive \u03c1 in every one "
         f"of the {n} cases (lowest \u03c1 = {f2(min_rho)}). The number of cases without a significant positive "
         "\u03c1 was " + ", ".join(f"{n_ - ok_} for {m}" for m, (ok_, n_, _) in ROBUST.items() if ok_ < n_)
         + ". "
         "Inpainting deletion is therefore the most robust variant we tested, although on CIFAR-10 it also "
         "switches early against the full object map (Table " + str(t_full) + ").")

    heading("5.6 Explanation methods", 2)
    f_m = COUNTERS["fig"] + 1
    t_m = COUNTERS["table"] + 1
    para(f"Fig. {f_m} and Table {t_m} show how much attribution each method places on the shortcut as reliance "
         f"grows. Integrated Gradients follows the shortcut in every setting (\u03c1 = {span(IG_RHO)}), and on "
         f"MNIST patch it places {f2(mrow('mnist', 'patch', 'Integrated Gradients').mass_high_reliance)} of its "
         "positive attribution on the 9 shortcut pixels at the highest dose. On the MNIST texture shortcut, "
         "Saliency, SmoothGrad and Grad-CAM move the wrong way: they put less weight on the shortcut as the "
         f"model relies on it more (\u03c1 = {signed_span(MT_AWAY)}). On PneumoniaMNIST texture only Grad-CAM "
         f"does so (\u03c1 = {signed(PT_GRADCAM)}), while Saliency follows the shortcut (\u03c1 = {signed(PT_SALIENCY)}). "
         f"Grad-CAM, with its coarse feature maps, places the least attribution of all methods on the patch at "
         f"the highest dose in {GC_LOWEST} of the {len(PATCH_SETTINGS)} patch settings.")
    figure(figs["methods"],
           "Share of positive attribution that each explanation method places on the shortcut pixels, as "
           "measured reliance grows. Panels rearranged from results/dose_response/method_dose_response.png")
    table_methods()

    heading("5.7 Supporting study: the sanity check against ground truth", 2)
    f_gt = COUNTERS["fig"] + 1
    para("A separate fixed-shortcut study trained three seeds on each of five datasets (MNIST, Fashion-MNIST, "
         "PneumoniaMNIST and two decoy datasets with a class-coded patch, following " + cite("ross") + "). On the "
         f"decoy datasets the models follow the patch {ACCURACY.loc['decoy_mnist', 'shortcut_reliance_mean']:.0f}% "
         f"(Decoy MNIST) and {ACCURACY.loc['decoy_pneumonia', 'shortcut_reliance_mean']:.0f}% (Decoy Pneumonia) of "
         "the time, so the patch is the ground-truth evidence on test images whose patch points to a wrong class. "
         f"On Decoy MNIST, Integrated Gradients puts {DECOY_MNIST.loc['Integrated Gradients', 'mass_accuracy']:.0%} "
         f"of its positive attribution on the patch and Occlusion {DECOY_MNIST.loc['Occlusion', 'mass_accuracy']:.0%}, "
         f"against at most {max(DECOY_MNIST.loc[m, 'mass_accuracy'] for m in ['Saliency', 'SmoothGrad', 'Grad-CAM']):.0%} "
         f"for the other methods. Fig. {f_gt} compares each metric's ranking of the five methods with the "
         "ground-truth ranking. The model-randomization sanity check " + cite("adebayo2018") + " does not rank the "
         f"methods in line with the ground truth (Kendall \u03c4 = {signed(SANITY_TAU[0])} on Decoy MNIST and "
         f"{signed(SANITY_TAU[1])} on Decoy Pneumonia). The perturbation metrics agree with the ground truth on "
         f"Decoy MNIST (\u03c4 = {span(DEL_TAU_MNIST)}) but not on Decoy Pneumonia (\u03c4 = "
         f"{signed_span(DEL_TAU_PNEU)}); per image, deletion scores correlate with correct localisation "
         f"(Spearman \u03c1 = {span(DEL_INST)}) and insertion less so ({span(INS_INST)}).")
    figure(AN / "gt_agreement.png",
           "Supporting study. Left: Kendall \u03c4 between each metric's ranking of the five methods and the "
           "ground-truth ranking (mean \u00b1 standard deviation over three seeds). Right: per-image Spearman "
           "\u03c1 between metric scores and ground-truth localisation. Source: results/analysis/gt_agreement.png",
           width=Inches(6.0))

    # ---- 6 Discussion
    heading("6 Discussion")
    para("**Why the X-ray setting is harder.** PneumoniaMNIST images have a textured background, there are only "
         "two classes, and the lowest reliance reached was about 15%. Removing or restoring pixels of a chest "
         "X-ray changes the image less predictably than on a dark-background digit, and with two classes the "
         "model's probability is more often saturated: the share of tied comparisons reaches "
         f"{PNEU_TIES:.0%} on PneumoniaMNIST against at most {MNIST_TIES:.1%} on MNIST. The supporting study points the "
         "same way: on Decoy Pneumonia the perturbation metrics' rankings disagree with the ground truth. "
         "Metric results obtained on MNIST-like data should therefore not be transferred to medical images "
         "without checking.")
    para("**The size confound.** A metric can prefer a larger explanation simply because removing more "
         "pixels, or the right pixels at the start of the curve, lowers confidence faster. The size-matched "
         "control shows that a large part of the preference for the object map at low reliance comes from this. "
         "Any oracle- or mask-based evaluation that compares maps of different sizes, such as an object mask "
         "against a small marker, inherits the confound and should include a size-matched control.")
    para("**Form, not only size.** The same-form controls show that the metrics are not blind: they rank the "
         "shortcut map against a random or placebo map correctly. The failures come from comparing a compact "
         "map with a scattered one. On CIFAR-10, black and blur deletion prefer scattered removal over compact "
         "removal of central pixels even when the model ignores the shortcut. Inpainting removes this effect, "
         "which is consistent with ROAD's argument that the shape of the removed region can leak information "
         + cite("rong") + ", but our object map is a Gaussian prior, so the explanation is a hypothesis to be "
         "tested with real object masks.")
    para("**Why insertion fails on a diffuse shortcut.** A likely reason is that restoring scattered pixels "
         "into a blurred image barely changes it: the blur spreads the texture's brightness over the "
         "neighbouring pixels, so the restored pixels add little new information. Insertion then gives the texture map little "
         "credit, and as reliance grows it rates a random ordering above the texture map more often "
         "(Section 5.5). This explanation was not tested separately.")
    para("**Why some methods move away from the texture shortcut.** As reliance on the texture grows, "
         "Saliency, SmoothGrad and Grad-CAM put less weight on it. Possible reasons are gradient saturation once "
         "the model is confident, or the texture adding to bright digit pixels so that gradient methods "
         "attribute it to the digit. We state these as hypotheses; Integrated Gradients, which integrates along "
         "a path from a black baseline, does not show the effect.")
    para("**Recommendations.** (1) Before using a metric to compare explanation methods, test it at several "
         "controlled reliance levels, not at one shortcut strength. (2) Compare maps of matched size and form, "
         "or report a size-matched and a random control. (3) Prefer inpainting-based deletion over black or blur "
         "replacement, which was the most robust variant in our tests. (4) Do not rely on insertion for diffuse "
         "evidence. (5) Use the model-randomization sanity check with caution as a ranking criterion; in our "
         "supporting study its ranking of methods did not agree with the ground truth.")

    # ---- 7 Limitations
    heading("7 Limitations")
    bullets([
        "The images are small (28\u00d728 grayscale and 32\u00d732 colour), and the largest network is ResNet-18. "
        "The results may not hold for high-resolution images or other architectures.",
        "The shortcuts are synthetic (a white patch and a brightened texture) and simpler than real-world "
        "shortcuts such as hospital markers or background correlations.",
        "The inpainting deletion is inspired by ROAD but is not the exact ROAD procedure.",
        "The 50% switch point is a modelling assumption about when a model \u201crelies\u201d on a shortcut, and "
        "the size-matched control uses the highest-valued object pixels, which is only one way to match size.",
        "The ResNet-18 check covers only the CIFAR-10 patch shortcut. On CIFAR-10 (both networks) there is one "
        "shortcut type, no augmentation, and no model between about 45% and 96% reliance; the small CNN uses "
        "two seeds.",
        "The CIFAR-10 object map is a centred Gaussian, not a real object mask, so the cause of the form bias "
        "remains a hypothesis.",
        "On PneumoniaMNIST the lowest reliance reached was about 15%, not 0%, and near d = 0 reliance varies "
        "strongly between seeds. Each model is scored on 100 test images, and three seeds per dose limit the "
        "precision of the switch points.",
        "The literature search behind the novelty statement was our own and must be completed before submission.",
    ])

    # ---- 8 Conclusion
    heading("8 Conclusion")
    para("We tested whether XAI faithfulness metrics notice when a model starts relying on a shortcut, by "
         f"setting the reliance of {N_MODELS} models in controlled steps and scoring fixed oracle explanations. "
         f"The metrics follow reliance closely on MNIST (\u03c1 = {mnist_rho}) but weakly on chest X-rays "
         f"(\u03c1 = {pneu_rho}). They switch to the shortcut explanation too early when the competing object "
         "explanation differs in size, and on natural images black and blur deletion prefer a random ordering "
         "over a compact object map when the model ignores the shortcut. When the compared explanations have "
         "the same form, the metrics behave correctly. Inpainting-based deletion was the most robust variant, "
         "and insertion failed on a diffuse shortcut. Evaluation metrics should themselves be validated at "
         "several controlled reliance levels, with controls for explanation size and form, before they are "
         "used to rank explanation methods. Future work should use real object masks, real-world shortcuts "
         "and larger images.")

    # ---- Declarations
    heading("Declarations")
    para("**Funding** This research received no funding.")
    para("**Competing interests** The author declares no competing interests.")
    para("**Data and code availability** [Placeholder: link to the public code repository and the result "
         "files.] MNIST, CIFAR-10 and PneumoniaMNIST are publicly available.")
    para("**Use of AI tools** AI assistants were used for code development and for drafting the text of this "
         "manuscript. The author designed the study, ran and verified the experiments, checked the results and "
         "the text, and takes full responsibility for the content.")
    heading("Acknowledgements")
    para("The author thanks [Guide Name] for guidance and feedback on this work.")

    # ---- References
    heading("References")
    para("[Note for the guide: every entry is taken from the project's verified reference list. Authors, "
         "venues and page numbers must be checked against the original papers before submission; entries "
         "without authors are cited by title and arXiv ID.]", italic=True, size=9)
    for i, key in enumerate(CITED, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(f"{i}. {REFERENCES[key]}")
        run.font.size = Pt(9.5)
    unused = [k for k in REFERENCES if k not in CITED]
    assert not unused, f"references listed but never cited: {unused}"

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT} ({COUNTERS['fig']} figures, {COUNTERS['table']} tables, {len(CITED)} references)")


if __name__ == "__main__":
    build()

"""Statistics for the corrected dose-response analysis (numpy/scipy only).

Chance floor and chance-corrected reliance
    Reliance R is the share of conflicting test images whose prediction
    follows the shortcut. A model that ignores the shortcut still lands on the
    shortcut's class when it makes an error: on a binary task every error on a
    conflicting image hits the shortcut class; with C classes an error hits the
    one shortcut class with probability 1/(C - 1) if errors spread uniformly.
        floor f = (100 - accuracy without shortcut) / (C - 1)
        r_c     = 100 * (R - f) / (100 - f)          (percent, not clipped)
    This is an approximation: images without the shortcut are out of
    distribution for a model trained with it.

Per-image win score and AUC gap
    w = 1 if the metric rates map A above map B, 0 if below, 0.5 for a tie
    (|AUC_A - AUC_B| within the tie tolerance). "Above" is a lower AUC for
    deletion and a higher AUC for insertion. The gap g is positive when map A
    is rated better: g = AUC_B - AUC_A (deletion), AUC_A - AUC_B (insertion).

Switch points
    isotonic_switch: an increasing isotonic fit (PAVA, ties in x pooled) of
    preference on reliance; the switch is the first reliance where the fitted
    step function reaches 0.5, by linear interpolation between consecutive
    distinct x values. switch_identification: a switch counts as identified
    only if some model lies within +-10 reliance points of it.
    switch_bracketing (sensitivity): models within +-10 points on BOTH sides.
    logistic4_switch: p(r) = a + (b - a) / (1 + exp(-(r - c) / s)) with free
    asymptotes 0 <= a < b <= 1, s > 0; the switch solves p(r) = 0.5.

Decision rates (replacing the unbalanced decision accuracy)
    positive class: reliance > 50; "metric says shortcut": preference > 0.5.
    TPR = P(says shortcut | above), TNR = P(says object | below),
    balanced accuracy = (TPR + TNR) / 2, over-detection rate fpr = 1 - TNR.

Two-level bootstrap
    two_level_picks resamples models (seeds) within each dose and, for every
    picked model, its image rows, independently per model.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import false_discovery_control, norm, pearsonr, rankdata, t as t_dist


# ------------------------------------------------------------ reliance

def chance_floor(acc_no_shortcut, classes):
    """Reliance expected from errors alone: (100 - accuracy without shortcut) / (C - 1)."""
    return (100.0 - np.asarray(acc_no_shortcut, dtype=float)) / (np.asarray(classes, dtype=float) - 1)


def corrected_reliance(reliance, floor):
    """Chance-corrected reliance in percent, 100 (R - f) / (100 - f); not clipped."""
    floor = np.asarray(floor, dtype=float)
    return 100.0 * (np.asarray(reliance, dtype=float) - floor) / (100.0 - floor)


# ------------------------------------------------------------ per-image scores

def win_scores(a, b, higher, tol):
    """1 if A is rated above B, 0 if below, 0.5 for a tie (as preference_table)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    better = a > b if higher else a < b
    ties = np.isclose(a, b, atol=tol)
    return np.where(ties, 0.5, better.astype(float))


def auc_gap(a, b, higher):
    """Per-image AUC gap, positive when map A is rated better."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a - b if higher else b - a


# ------------------------------------------------------------ switch points

def pava(y, w=None):
    """Increasing isotonic regression (pool adjacent violators)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    values, weights, sizes = [], [], []
    for yi, wi in zip(y, w):
        values.append(yi)
        weights.append(wi)
        sizes.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            wsum = weights[-2] + weights[-1]
            merged = (values[-2] * weights[-2] + values[-1] * weights[-1]) / wsum
            size = sizes[-2] + sizes[-1]
            values[-2:], weights[-2:], sizes[-2:] = [merged], [wsum], [size]
    return np.repeat(values, sizes)


def _pooled(x, y):
    """Distinct sorted x with the mean y and count at each."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    ux, inverse, counts = np.unique(x[keep], return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y[keep], minlength=len(ux))
    return ux, sums / np.maximum(counts, 1), counts


def isotonic_switch(x, y, level=0.5):
    """(switch, status) of an increasing isotonic fit of y on x.

    status: "crossed", "no crossing" (the fit stays below level), or
    "above 0.5 at lowest reliance" (already >= level at the smallest x).
    """
    ux, my, counts = _pooled(x, y)
    if len(ux) == 0:
        return float("nan"), "no crossing"
    fit = pava(my, counts)
    if fit[0] >= level:
        return float("nan"), "above 0.5 at lowest reliance"
    above = np.flatnonzero(fit >= level)
    if len(above) == 0:
        return float("nan"), "no crossing"
    i = above[0]
    x0, x1, f0, f1 = ux[i - 1], ux[i], fit[i - 1], fit[i]
    return float(x0 + (level - f0) * (x1 - x0) / (f1 - f0)), "crossed"


def switch_identification(switch, x, window=10):
    """(nearest_model_distance, status): identified if a model lies within +-window."""
    x = np.asarray(x, dtype=float)
    if np.isnan(switch) or len(x) == 0:
        return float("nan"), None
    distance = float(np.nanmin(np.abs(x - switch)))
    status = "identified" if distance <= window else f"not identified (no model within ±{window:g} points)"
    return distance, status


def switch_bracketing(switch, x, window=10):
    """(distance below, distance above, bracketed): the distance from the switch
    to the nearest model at or below it and at or above it; bracketed when both
    are within +-window. A switch inside a gap wider than 2 x window can pass
    switch_identification from one edge but is never bracketed."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    nan = float("nan")
    if np.isnan(switch) or len(x) == 0:
        return nan, nan, False
    below, above = x[x <= switch], x[x >= switch]
    d_below = float(switch - below.max()) if len(below) else nan
    d_above = float(above.min() - switch) if len(above) else nan
    bracketed = bool(len(below) and len(above) and d_below <= window and d_above <= window)
    return d_below, d_above, bracketed


def _logistic4(r, a, b, c, s):
    with np.errstate(over="ignore"):
        return a + (b - a) / (1 + np.exp(-(r - c) / s))


def logistic4_switch(x, y, level=0.5):
    """(switch, params dict, fit_ok) of a 4-parameter logistic with free asymptotes."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    nan = float("nan")
    params = {"a": nan, "b": nan, "c": nan, "s": nan}
    if len(x) < 5 or np.ptp(x) == 0:
        return nan, params, False
    lo, hi = float(x.min()), float(x.max())
    p0 = [max(min(y.min(), 0.99), 0.0), min(max(y.max(), 0.01), 1.0), float(np.median(x)), max(np.ptp(x) / 10, 1.0)]
    p0[1] = max(p0[1], p0[0] + 1e-3)
    try:
        fitted, _ = curve_fit(_logistic4, x, y, p0=p0, maxfev=20000,
                              bounds=([0, 0, lo - 50, 1e-3], [1, 1, hi + 50, 200]))
    except (RuntimeError, ValueError):
        return nan, params, False
    a, b, c, s = (float(v) for v in fitted)
    params = {"a": a, "b": b, "c": c, "s": s}
    ok = a < b
    if not (ok and a < level < b):
        return nan, params, bool(ok)
    return float(c - s * np.log((b - a) / (level - a) - 1)), params, True


# ------------------------------------------------------------ decisions

def decision_rates(preference, reliance, threshold=50):
    """TPR, TNR, over-detection rate (fpr = 1 - TNR) and balanced accuracy of
    (preference > 0.5) as a test for (reliance > threshold)."""
    preference, reliance = np.asarray(preference, dtype=float), np.asarray(reliance, dtype=float)
    says, above = preference > 0.5, reliance > threshold
    n_above, n_below = int(above.sum()), int((~above).sum())
    tpr = float(says[above].mean()) if n_above else float("nan")
    tnr = float((~says[~above]).mean()) if n_below else float("nan")
    balanced = (tpr + tnr) / 2 if n_above and n_below else float("nan")
    return {"tpr": tpr, "tnr": tnr, "fpr": 1 - tnr if n_below else float("nan"),
            "balanced_accuracy": balanced, "n_above": n_above, "n_below": n_below}


# ------------------------------------------------------------ multiplicity

def bh(p):
    """Benjamini-Hochberg adjusted p-values; NaN entries stay NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    keep = ~np.isnan(p)
    if keep.any():
        out[keep] = false_discovery_control(p[keep], method="bh")
    return out


def holm(p):
    """Holm step-down adjusted p-values; NaN entries stay NaN."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    keep = np.flatnonzero(~np.isnan(p))
    m = len(keep)
    if m:
        order = keep[np.argsort(p[keep], kind="stable")]
        adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
        out[order] = adjusted
    return out


def bootstrap_z_p(estimate, boot, null=0.0):
    """Two-sided normal-approximation p-value 2 * sf(|estimate - null| / SE),
    with SE the standard deviation of the bootstrap replicates. Unlike a
    percentile bootstrap p it is not floored at 1/(reps + 1)."""
    boot = np.asarray(boot, dtype=float)
    boot = boot[~np.isnan(boot)]
    if np.isnan(estimate) or len(boot) < 2:
        return float("nan")
    se = float(boot.std(ddof=1))
    if se == 0:
        return 0.0 if estimate != null else 1.0
    return float(2 * norm.sf(abs(estimate - null) / se))


def format_p(p):
    """'< 0.001' below 0.001, else 3 decimals ('–' for NaN)."""
    if p is None or np.isnan(p):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------ correlations

def auroc(scores, labels):
    """Area under the ROC curve via midrank Mann-Whitney U; NaN if a class is empty."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z: Pearson correlation of
    the x and y rank residuals after a linear fit on the z rank; p from a t
    distribution with n - 3 degrees of freedom."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def residual(a):
        return a - np.polyval(np.polyfit(rz, a, 1), rz)

    ex, ey = residual(rx), residual(ry)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan"), float("nan")
    rho = float(pearsonr(ex, ey)[0])
    n = len(rx)
    if n <= 3 or abs(rho) >= 1:
        return rho, 0.0 if abs(rho) >= 1 else float("nan")
    stat = rho * np.sqrt((n - 3) / (1 - rho ** 2))
    return rho, float(2 * t_dist.sf(abs(stat), n - 3))


def auroc_rows(scores, labels, valid=None):
    """Row-wise AUROC of two (reps, n) matrices (midrank); NaN where a row has one class.

    valid: optional boolean (reps, n) mask; entries outside it are ignored.
    They are given score +inf, which ranks them above every valid entry and
    so leaves the ranks of the valid entries unchanged."""
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    labels = np.atleast_2d(labels)
    if valid is not None:
        valid = np.atleast_2d(valid).astype(bool)
        scores = np.where(valid, scores, np.inf)
        labels = np.where(valid, labels, 0)
    labels = labels.astype(bool)
    ranks = rankdata(scores, axis=1)
    pos = labels.sum(axis=1).astype(float)
    neg = (labels.shape[1] if valid is None else valid.sum(axis=1)) - pos
    u = (ranks * labels).sum(axis=1) - pos * (pos + 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((pos > 0) & (neg > 0), u / (pos * neg), np.nan)


def spearman_rows(x, y):
    """Row-wise Spearman correlation of two (reps, n) matrices (x may be 1-D)."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    x = np.asarray(x, dtype=float)
    x = np.broadcast_to(x, y.shape) if x.ndim == 1 else x
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    denominator = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (rx * ry).sum(axis=1) / denominator, np.nan)


# ------------------------------------------------------------ exact Kendall null

def _inversion_counts(n):
    counts = np.zeros(n * (n - 1) // 2 + 1, dtype=np.int64)
    for perm in itertools.permutations(range(n)):
        k = sum(perm[i] > perm[j] for i in range(n) for j in range(i + 1, n))
        counts[k] += 1
    return counts


def kendall_null(n=5):
    """Exact null distribution of Kendall's tau (no ties) over all n! orderings."""
    counts = _inversion_counts(n)
    pairs = n * (n - 1) / 2
    tau = 1 - 2 * np.arange(len(counts)) / pairs
    return pd.DataFrame({"tau": np.round(tau, 10), "count": counts, "prob": counts / counts.sum()})


def kendall_exact_p(tau, n=5):
    """(one-sided P(T >= tau), two-sided P(|T| >= |tau|)) under the exact null."""
    null = kendall_null(n)
    eps = 1e-9
    one = float(null.prob[null.tau >= tau - eps].sum())
    two = float(null.prob[null.tau.abs() >= abs(tau) - eps].sum())
    return one, two


def mean_tau_null(n=5, k=3):
    """Exact null of the mean of k independent Kendall taus (each over n items)."""
    counts = _inversion_counts(n).astype(float)
    dist = counts / counts.sum()
    total = np.array([1.0])
    for _ in range(k):
        total = np.convolve(total, dist)
    pairs = n * (n - 1) / 2
    mean_tau = 1 - 2 * np.arange(len(total)) / (pairs * k)
    return pd.DataFrame({"tau_mean": np.round(mean_tau, 10), "prob": total})


# ------------------------------------------------------------ bootstrap

def two_level_picks(dose_of_model, n_images, reps, seed):
    """(P, I) for a two-level bootstrap.

    P[r, m]: the model placed in slot m in replicate r, drawn with replacement
    from the models of slot m's dose (so the dose design stays fixed).
    I[r, m, :]: image rows drawn with replacement for that slot, independently
    per slot.
    """
    doses = np.asarray(dose_of_model)
    rng = np.random.default_rng(seed)
    P = np.empty((reps, len(doses)), dtype=np.int64)
    for dose in pd.unique(doses):
        members = np.flatnonzero(np.isclose(doses.astype(float), float(dose)))
        P[:, members] = members[rng.integers(0, len(members), size=(reps, len(members)))]
    I = rng.integers(0, n_images, size=(reps, len(doses), n_images)).astype(np.int32)
    return P, I

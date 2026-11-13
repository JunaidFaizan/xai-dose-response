"""Corrected dose-response analysis: DataFrame in, DataFrame out (no file I/O).

Called by `dose_response.py analyze`, which writes the outputs. The formulas
(chance floor, chance-corrected reliance r_c, win scores, AUC gaps, isotonic
and 4PL switch points, decision rates, BH/Holm) are documented in xai/stats.py.

Unit of analysis
    One Setting = one base dataset x shortcut type. Its models are sorted by
    (dose, seed), its per-image scores by (dose, seed, image index); every
    model is scored on the same images, so per-image scores form a
    (models x images) matrix.

Two-level bootstrap (MD3)
    Per replicate, models are resampled within each dose and each picked
    model's image rows are resampled independently (xai.stats.two_level_picks).
    The picks are drawn once per setting (seed BOOTSTRAP_SEED) and reused for
    every comparison, metric and method, so metric-vs-metric differences are
    paired. 95% intervals are percentiles. Switch intervals use only the
    replicates in which a switch is found and are left empty if fewer than
    half find one (switch_found_share).

p-values and families
    Correlation p-values are Spearman's (t approximation over models).
    Bootstrap p-values (metric pairs, AUROC vs 0.5) are two-sided:
    min(1, 2 min(P(d <= 0), P(d >= 0))), floored at 1/(reps + 1).
    BH and Holm are applied within each family:
        responsiveness:<comparison>   all settings x metrics of one comparison
        methods                       all settings x methods
        metric_pairs:<comparison>     all settings x metric pairs
        behaviour:<label>             all settings x comparisons x metrics
    p_holm_headline is Holm over the union of the headline rows
    ("shortcut vs full object" and "shortcut vs random" responsiveness, and
    the primary-label behaviour AUROC for "shortcut vs full object").
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import stats
from .explain import METHODS
from .metrics import FAITHFULNESS_METRICS

KEYS = ["base", "shortcut", "dose", "seed"]
# Comparisons whose ideal curve crosses 0.5 at 50% reliance: switch points
# are only computed for these ("not applicable" elsewhere).
SWITCH_COMPARISONS = ("shortcut vs full object", "shortcut vs size-matched object")
PAIR_COMPARISONS = ("shortcut vs full object", "shortcut vs size-matched object", "random vs full object")
HEADLINE_COMPARISONS = ("shortcut vs full object", "shortcut vs random")
BEHAVIOUR_COMPARISONS = SWITCH_COMPARISONS
BEHAVIOUR_LABELS = ("follows_shortcut", "shortcut_dominant")  # primary, check
PRIMARY_LABEL = "follows_shortcut"
IDENTIFY_WINDOW = 10
RELIANCE_BANDS = [-np.inf, 20, 40, 60, 80, np.inf]
BAND_LABELS = ["<20", "20-40", "40-60", "60-80", ">=80"]
BAND_CENTRES = [10, 30, 50, 70, 90]
CHUNK = 250
NAN = float("nan")


def _ci(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return NAN, NAN
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _boot_p(values, null=0.0):
    """Two-sided bootstrap p-value of values against null, floored at 1/(reps + 1)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return NAN
    p = 2 * min(np.mean(values <= null), np.mean(values >= null))
    return float(max(min(1.0, p), 1 / (len(values) + 1)))


def _spearman(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = spearmanr(x, y, nan_policy="omit")
    return float(rho), float(p)


def _first_crossing(reliance, preference):
    """The legacy switch rule (dose_response.switch_point): first linear
    crossing of 0.5 by the dose means, sorted by reliance."""
    frame = pd.DataFrame({"r": reliance, "p": preference}).sort_values("r")
    r, p = frame.r.to_numpy(), frame.p.to_numpy()
    for i in range(len(r) - 1):
        if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0 and p[i] != p[i + 1]:
            return float(r[i] + (0.5 - p[i]) * (r[i + 1] - r[i]) / (p[i + 1] - p[i]))
    return NAN


def _adjust(frame, family_column="family", p_column="p"):
    """Append p_bh and p_holm, computed within each family."""
    frame = frame.copy()
    frame["p_bh"], frame["p_holm"] = np.nan, np.nan
    for _, index in frame.groupby(family_column).groups.items():
        p = frame.loc[index, p_column].to_numpy(dtype=float)
        frame.loc[index, "p_bh"] = stats.bh(p)
        frame.loc[index, "p_holm"] = stats.holm(p)
    return frame


class Setting:
    """Per-image score matrices and bootstrap picks of one base x shortcut."""

    def __init__(self, base, shortcut, models, oracle, methods, reps, seed, tie_tolerance):
        self.base, self.shortcut, self.reps, self.tie = base, shortcut, reps, tie_tolerance
        self.models = models.sort_values(["dose", "seed"]).reset_index(drop=True)
        order = ["dose", "seed", "index"]
        self.oracle = oracle.merge(self.models[KEYS], on=KEYS).sort_values(order).reset_index(drop=True)
        self.n_models = len(self.models)
        counts = self.oracle.groupby(KEYS).size()
        if len(counts) != self.n_models or counts.nunique() != 1:
            raise ValueError(f"{base}/{shortcut}: every model needs the same number of scored images")
        self.n_images = int(counts.iloc[0])
        self.indices = self.oracle["index"].to_numpy().reshape(self.n_models, self.n_images)
        if not (self.indices == self.indices[0]).all():
            raise ValueError(f"{base}/{shortcut}: models were scored on different images")
        self.methods = None
        if methods is not None:
            self.methods = methods.merge(self.models[KEYS], on=KEYS).sort_values(order + ["method"])
        self.rc = self.models.reliance_corrected.to_numpy(dtype=float)
        self.raw = self.models.reliance.to_numpy(dtype=float)
        self.dose = self.models.dose.to_numpy(dtype=float)
        self.low = np.flatnonzero(np.isclose(self.dose, self.dose.min()))
        self.high = np.flatnonzero(np.isclose(self.dose, self.dose.max()))
        self.P, self.I = stats.two_level_picks(self.dose, self.n_images, reps, seed)
        self._cache = {}

    def has(self, column):
        return column in self.oracle and not self.oracle[column].isna().any()

    def matrix(self, column):
        return self.oracle[column].to_numpy(dtype=float).reshape(self.n_models, self.n_images)

    def method_matrix(self, method):
        frame = self.methods[self.methods.method == method].sort_values(["dose", "seed", "index"])
        values = frame.shortcut_mass.to_numpy(dtype=float).reshape(self.n_models, self.n_images)
        if not (frame["index"].to_numpy().reshape(self.n_models, self.n_images) == self.indices).all():
            raise ValueError(f"{self.base}/{self.shortcut}: method scores use other images")
        return values

    def resample(self, values, reducer):
        """Apply reducer(sub, axis=2) to values[P, I] in chunks: (reps, models)."""
        out = np.empty(self.P.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for s in range(0, self.reps, CHUNK):
                sub = values[self.P[s:s + CHUNK, :, None], self.I[s:s + CHUNK]]
                out[s:s + CHUNK] = reducer(sub, axis=2)
        return out

    def pooled(self, values, slots, reducer):
        """reducer over the pooled resampled images of the given slots: (reps,)."""
        out = np.empty(self.reps)
        for s in range(0, self.reps, CHUNK):
            sub = values[self.P[s:s + CHUNK, slots, None], self.I[s:s + CHUNK][:, slots]]
            out[s:s + CHUNK] = reducer(sub.reshape(sub.shape[0], -1))
        return out

    def comparison(self, first, second, metric):
        """Cached per-image wins/gaps, model preferences and their bootstrap."""
        key = (first, second, metric)
        if key not in self._cache:
            higher = FAITHFULNESS_METRICS[metric][3]
            a, b = self.matrix(f"{metric}_{first}"), self.matrix(f"{metric}_{second}")
            wins = stats.win_scores(a, b, higher, self.tie)
            gaps = stats.auc_gap(a, b, higher)
            boot = self.resample(wins, np.mean)
            self._cache[key] = {
                "wins": wins, "gaps": gaps, "ties": np.isclose(a, b, atol=self.tie),
                "pref": wins.mean(axis=1), "boot": boot,
                "rho_rc_boot": stats.spearman_rows(self.rc[self.P], boot),
                "rho_raw_boot": stats.spearman_rows(self.raw[self.P], boot),
            }
        return self._cache[key]

    def median_gap(self, first, second, metric, level):
        key = ("median", first, second, metric, level)
        if key not in self._cache:
            gaps = self.comparison(first, second, metric)["gaps"]
            slots = self.low if level == "low" else self.high
            self._cache[key] = self.pooled(gaps, slots, lambda m: np.median(m, axis=1))
        return self._cache[key]


def build_settings(oracle, methods, models, reps, seed, tie_tolerance):
    """One Setting per base x shortcut that has scored models."""
    scored = models.merge(oracle[KEYS].drop_duplicates(), on=KEYS)
    settings = []
    for (base, shortcut), group in scored.groupby(["base", "shortcut"]):
        settings.append(Setting(base, shortcut, group,
                                oracle[(oracle.base == base) & (oracle.shortcut == shortcut)],
                                None if methods is None else
                                methods[(methods.base == base) & (methods.shortcut == shortcut)],
                                reps, seed, tie_tolerance))
    return settings


def _available(settings, comparisons):
    for first, second, label, *_ in comparisons:
        column = f"deletion_zero_{first}", f"deletion_zero_{second}"
        if all(s.has(column[0]) and s.has(column[1]) for s in settings):
            yield first, second, label


def _switch_columns(s, c, label):
    nan_row = {"switch_iso_rc": NAN, "switch_iso_rc_ci_low": NAN, "switch_iso_rc_ci_high": NAN,
               "switch_found_share": NAN, "switch_status": "not applicable",
               "nearest_model_distance": NAN, "switch_4pl_rc": NAN, "fit_4pl_ok": False,
               "switch_iso_raw": NAN, "switch_interp_rc": NAN}
    if label not in SWITCH_COMPARISONS:
        return nan_row
    pref = c["pref"]
    switch, status = stats.isotonic_switch(s.rc, pref)
    distance = NAN
    if status == "crossed":
        distance, status = stats.switch_identification(switch, s.rc, IDENTIFY_WINDOW)
    boot = np.array([stats.isotonic_switch(s.rc[s.P[r]], c["boot"][r])[0] for r in range(s.reps)])
    found = ~np.isnan(boot)
    share = float(found.mean())
    low, high = _ci(boot) if share >= 0.5 else (NAN, NAN)
    switch_4pl, _, ok = stats.logistic4_switch(s.rc, pref)
    levels = pd.DataFrame({"dose": s.dose, "rc": s.rc, "p": pref}).groupby("dose").mean()
    return {"switch_iso_rc": switch, "switch_iso_rc_ci_low": low, "switch_iso_rc_ci_high": high,
            "switch_found_share": share, "switch_status": status,
            "nearest_model_distance": distance, "switch_4pl_rc": switch_4pl, "fit_4pl_ok": bool(ok),
            "switch_iso_raw": stats.isotonic_switch(s.raw, pref)[0],
            "switch_interp_rc": _first_crossing(levels.rc, levels.p)}


def responsiveness_corrected(settings, comparisons, metric_titles):
    """PRIMARY metric table: one row per comparison x setting x metric."""
    rows = []
    for first, second, label in _available(settings, comparisons):
        for s in settings:
            for metric in FAITHFULNESS_METRICS:
                c = s.comparison(first, second, metric)
                pref = c["pref"]
                rho_rc, p_rc = _spearman(s.rc, pref)
                rho_raw, p_raw = _spearman(s.raw, pref)
                low_boot, high_boot = c["boot"][:, s.low].mean(axis=1), c["boot"][:, s.high].mean(axis=1)
                on_rc = stats.decision_rates(pref, s.rc)
                on_raw = stats.decision_rates(pref, s.raw)
                row = {"comparison": label, "base": s.base, "shortcut": s.shortcut,
                       "metric": metric_titles[metric], "models": s.n_models,
                       "tie_rate": float(c["ties"].mean()),
                       "rho_rc": rho_rc, "p_rho_rc": p_rc}
                row["rho_rc_ci_low"], row["rho_rc_ci_high"] = _ci(c["rho_rc_boot"])
                row["rho_raw"], row["p_rho_raw"] = rho_raw, p_raw
                row["rho_raw_ci_low"], row["rho_raw_ci_high"] = _ci(c["rho_raw_boot"])
                row["pref_low"] = float(pref[s.low].mean())
                row["pref_low_ci_low"], row["pref_low_ci_high"] = _ci(low_boot)
                row["pref_high"] = float(pref[s.high].mean())
                row["pref_high_ci_low"], row["pref_high_ci_high"] = _ci(high_boot)
                row["median_gap_low"] = float(np.median(c["gaps"][s.low]))
                row["median_gap_high"] = float(np.median(c["gaps"][s.high]))
                row.update({"balanced_accuracy": on_rc["balanced_accuracy"], "tpr": on_rc["tpr"],
                            "fpr": on_rc["fpr"], "n_above": on_rc["n_above"], "n_below": on_rc["n_below"],
                            "balanced_accuracy_raw": on_raw["balanced_accuracy"], "tpr_raw": on_raw["tpr"],
                            "fpr_raw": on_raw["fpr"]})
                row.update(_switch_columns(s, c, label))
                row["rc_range"] = f"{s.rc.min():.1f} to {s.rc.max():.1f}"
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["family"] = "responsiveness:" + frame.comparison
    frame = _adjust(frame, p_column="p_rho_rc")
    frame["headline"] = frame.comparison.isin(HEADLINE_COMPARISONS)
    return frame


def effect_sizes(settings, comparisons, metric_titles):
    """Preference and median per-image AUC gap at the lowest and highest dose."""
    rows = []
    for first, second, label in _available(settings, comparisons):
        for s in settings:
            for metric in FAITHFULNESS_METRICS:
                c = s.comparison(first, second, metric)
                for level, slots in (("low", s.low), ("high", s.high)):
                    gaps = c["gaps"][slots].ravel()
                    pref_boot = c["boot"][:, slots].mean(axis=1)
                    gap_boot = s.median_gap(first, second, metric, level)
                    row = {"comparison": label, "base": s.base, "shortcut": s.shortcut,
                           "metric": metric_titles[metric], "level": level,
                           "dose": float(s.dose[slots][0]), "models": len(slots),
                           "reliance_corrected_mean": float(s.rc[slots].mean()),
                           "preference": float(c["pref"][slots].mean())}
                    row["preference_ci_low"], row["preference_ci_high"] = _ci(pref_boot)
                    row["median_gap"] = float(np.median(gaps))
                    row["median_gap_ci_low"], row["median_gap_ci_high"] = _ci(gap_boot)
                    row["gap_q25"], row["gap_q75"] = (float(v) for v in np.percentile(gaps, [25, 75]))
                    rows.append(row)
    return pd.DataFrame(rows)


def metric_pairs(settings, comparisons, metric_titles):
    """Paired metric-vs-metric differences in rho_rc and lowest-dose preference."""
    rows = []
    for first, second, label in _available(settings, comparisons):
        if label not in PAIR_COMPARISONS:
            continue
        for s in settings:
            for m_a, m_b in itertools.combinations(FAITHFULNESS_METRICS, 2):
                ca, cb = s.comparison(first, second, m_a), s.comparison(first, second, m_b)
                diff = ca["rho_rc_boot"] - cb["rho_rc_boot"]
                diff_raw = ca["rho_raw_boot"] - cb["rho_raw_boot"]
                low = ca["boot"][:, s.low].mean(axis=1) - cb["boot"][:, s.low].mean(axis=1)
                row = {"comparison": label, "base": s.base, "shortcut": s.shortcut,
                       "metric_a": metric_titles[m_a], "metric_b": metric_titles[m_b],
                       "delta_rho_rc": _spearman(s.rc, ca["pref"])[0] - _spearman(s.rc, cb["pref"])[0]}
                row["delta_rho_rc_ci_low"], row["delta_rho_rc_ci_high"] = _ci(diff)
                row["p"] = _boot_p(diff)
                row["delta_rho_raw"] = _spearman(s.raw, ca["pref"])[0] - _spearman(s.raw, cb["pref"])[0]
                row["delta_rho_raw_ci_low"], row["delta_rho_raw_ci_high"] = _ci(diff_raw)
                row["delta_pref_low"] = float(ca["pref"][s.low].mean() - cb["pref"][s.low].mean())
                row["delta_pref_low_ci_low"], row["delta_pref_low_ci_high"] = _ci(low)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["family"] = "metric_pairs:" + frame.comparison
    return _adjust(frame)


def method_responsiveness_corrected(settings, method_titles):
    """PRIMARY method table: shortcut mass vs reliance, with two-level CIs."""
    rows = []
    for s in settings:
        for method in METHODS:
            values = s.method_matrix(method)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mass = np.nanmean(values, axis=1)
            boot = s.resample(values, np.nanmean)
            rho_rc, p_rc = _spearman(s.rc, mass)
            rho_raw, p_raw = _spearman(s.raw, mass)
            row = {"base": s.base, "shortcut": s.shortcut, "method": method_titles[method],
                   "models": s.n_models, "rho_rc": rho_rc, "p": p_rc}
            row["rho_rc_ci_low"], row["rho_rc_ci_high"] = _ci(stats.spearman_rows(s.rc[s.P], boot))
            row["rho_raw"], row["p_rho_raw"] = rho_raw, p_raw
            row["rho_raw_ci_low"], row["rho_raw_ci_high"] = _ci(stats.spearman_rows(s.raw[s.P], boot))
            row["mass_low"] = float(np.nanmean(mass[s.low]))
            row["mass_low_ci_low"], row["mass_low_ci_high"] = _ci(np.nanmean(boot[:, s.low], axis=1))
            row["mass_high"] = float(np.nanmean(mass[s.high]))
            row["mass_high_ci_low"], row["mass_high_ci_high"] = _ci(np.nanmean(boot[:, s.high], axis=1))
            row["nan_share"] = float(np.isnan(values).mean())
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["family"] = "methods"
    return _adjust(frame)


# Verdicts copied from the Outcome table of docs/PREDICTIONS_pet.md.
PET_VERDICTS = {"P1": "Holds", "P2": "Holds",
                "P3": "Fails for Deletion (black); holds for the other three metrics",
                "P4": "Holds"}
PET_NOTE = ("Original verdict judged on the point estimate (the pre-registration did not say whether the "
            "point estimate or an interval decides); the two-level 95% CI and rho_rc are additional information.")


def pet_preregistered(settings, comparisons, metric_titles, method_titles, base="pet_resnet18"):
    """Pre-registered Pet predictions P1-P4: original statistics (raw reliance)
    with two-level CIs; the original verdicts are kept unchanged."""
    pet = [s for s in settings if s.base == base]
    if not pet:
        return pd.DataFrame()
    s = pet[0]
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    rows = []

    def add(prediction, comparison, item, statistic, estimate, boot, threshold, rho_rc=NAN):
        low, high = _ci(boot)
        rows.append({"prediction": prediction, "comparison": comparison, "metric_or_method": item,
                     "statistic": statistic, "estimate": estimate, "ci_low": low, "ci_high": high,
                     "rho_rc": rho_rc, "threshold": threshold,
                     "original_verdict": PET_VERDICTS[prediction], "note": PET_NOTE})

    def rho_rows(prediction, label):
        first, second = maps[label]
        for metric in FAITHFULNESS_METRICS:
            c = s.comparison(first, second, metric)
            add(prediction, label, metric_titles[metric], "spearman rho (raw reliance)",
                _spearman(s.raw, c["pref"])[0], c["rho_raw_boot"], "> 0.8", _spearman(s.rc, c["pref"])[0])

    def low_rows(prediction, label, threshold):
        first, second = maps[label]
        boots = {}
        for metric in FAITHFULNESS_METRICS:
            c = s.comparison(first, second, metric)
            boots[metric] = c["boot"][:, s.low].mean(axis=1)
            add(prediction, label, metric_titles[metric], f"preference at lowest dose ({s.dose.min():g})",
                float(c["pref"][s.low].mean()), boots[metric], threshold)
        return boots

    if "shortcut vs full object" in maps:
        rho_rows("P1", "shortcut vs full object")
    if "random vs full object" in maps:
        boots = low_rows("P2", "random vs full object", "black, blur > inpaint, insertion")
        first, second = maps["random vs full object"]
        point = {m: float(s.comparison(first, second, m)["pref"][s.low].mean()) for m in FAITHFULNESS_METRICS}
        contrast = ((point["deletion_zero"] + point["deletion_blur"]) / 2
                    - (point["deletion_inpaint"] + point["insertion_blur"]) / 2)
        contrast_boot = ((boots["deletion_zero"] + boots["deletion_blur"]) / 2
                         - (boots["deletion_inpaint"] + boots["insertion_blur"]) / 2)
        add("P2", "random vs full object", "mean(black, blur) - mean(inpaint, insertion)",
            "preference contrast at lowest dose", contrast, contrast_boot, "> 0")
    for label in ("shortcut vs random", "shortcut vs placebo"):
        if label in maps:
            low_rows("P3", label, "0.5 ± 0.15")
            rho_rows("P3", label)
    for method in ("integrated_gradients", "occlusion"):
        values = s.method_matrix(method)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mass = np.nanmean(values, axis=1)
        boot = s.resample(values, np.nanmean)
        add("P4", "shortcut mass", method_titles[method], "spearman rho (raw reliance)",
            _spearman(s.raw, mass)[0], stats.spearman_rows(s.raw[s.P], boot), "> 0.8", _spearman(s.rc, mass)[0])
    return pd.DataFrame(rows)


# ------------------------------------------------------------ MD4 behaviour

def _label_matrix(s, behaviour, column):
    frame = behaviour.merge(s.models[KEYS], on=KEYS).sort_values(["dose", "seed", "index"])
    values = frame[column].to_numpy(dtype=float)
    if len(values) != s.n_models * s.n_images or not (
            frame["index"].to_numpy().reshape(s.n_models, s.n_images) == s.indices).all():
        raise ValueError(f"{s.base}/{s.shortcut}: behaviour labels do not match the scored images")
    return values.reshape(s.n_models, s.n_images)


def _boot_auroc(s, gaps, labels):
    """Pooled AUROC per two-level bootstrap replicate: (reps,)."""
    out = []
    for r0 in range(0, s.reps, CHUNK):
        picks, images = s.P[r0:r0 + CHUNK, :, None], s.I[r0:r0 + CHUNK]
        n = len(images)
        out.append(stats.auroc_rows(gaps[picks, images].reshape(n, -1), labels[picks, images].reshape(n, -1)))
    return np.concatenate(out)


def _soft_accuracy(wins, labels):
    """accuracy = mean(1 - |w - y|); balanced = mean of mean(w | y = 1) and mean(1 - w | y = 0)."""
    accuracy = float(np.mean(1 - np.abs(wins - labels)))
    pos, neg = labels == 1, labels == 0
    balanced = (float(wins[pos].mean()) + float((1 - wins[neg]).mean())) / 2 if pos.any() and neg.any() else NAN
    return accuracy, balanced


def behaviour_tables(settings, behaviour, comparisons, metric_titles):
    """(behaviour_models, behaviour_summary, behaviour_by_reliance) DataFrames.

    The per-image label replaces the assumed 50% ideal. AUROC uses the
    per-image AUC gap (positive = shortcut map rated better) as the score of
    "the model follows the shortcut on this image"; accuracy uses the win
    score w (1, 0 or 0.5 for a tie).
    """
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    model_rows, summary_rows, band_rows = [], [], []
    for s in settings:
        labels = {name: _label_matrix(s, behaviour, name) for name in BEHAVIOUR_LABELS}
        rates = {name: values.mean(axis=1) for name, values in labels.items()}
        base_frame = s.models[KEYS + ["reliance", "reliance_corrected"]].copy()
        base_frame["follow_rate"] = rates["follows_shortcut"]
        base_frame["dominant_rate"] = rates["shortcut_dominant"]
        band = pd.cut(s.rc, RELIANCE_BANDS, right=False, labels=BAND_LABELS)
        for label in BEHAVIOUR_COMPARISONS:
            if label not in maps:
                continue
            first, second = maps[label]
            tag = "full" if label == "shortcut vs full object" else "matched"
            for metric in FAITHFULNESS_METRICS:
                c = s.comparison(first, second, metric)
                base_frame[f"pref_{metric}_{tag}"] = c["pref"]
                for name in BEHAVIOUR_LABELS:
                    y, rate = labels[name], rates[name]
                    per_model = np.array([stats.auroc(c["gaps"][m], y[m]) for m in range(s.n_models)])
                    base_frame[f"auroc_{name}_{metric}_{tag}"] = per_model
                    boot = _boot_auroc(s, c["gaps"], y)
                    accuracy, balanced = _soft_accuracy(c["wins"].ravel(), y.ravel())
                    switch, status = stats.isotonic_switch(100 * rate, c["pref"])
                    distance = NAN
                    if status == "crossed":
                        distance, status = stats.switch_identification(switch, 100 * rate, IDENTIFY_WINDOW)
                    observed, observed_status = stats.isotonic_switch(s.rc, rate)
                    row = {"comparison": label, "base": s.base, "shortcut": s.shortcut,
                           "metric": metric_titles[metric], "label": name, "models": s.n_models,
                           "label_rate": float(y.mean()),
                           "auroc_pooled": stats.auroc(c["gaps"].ravel(), y.ravel())}
                    row["auroc_ci_low"], row["auroc_ci_high"] = _ci(boot)
                    row["p_auroc"] = _boot_p(boot, 0.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        row["median_model_auroc"] = float(np.nanmedian(per_model))
                    row["models_with_both_classes"] = int((~np.isnan(per_model)).sum())
                    row.update({"accuracy": accuracy, "balanced_accuracy": balanced,
                                "calibration_gap": float(np.mean(c["pref"] - rate)),
                                "calibration_mae": float(np.mean(np.abs(c["pref"] - rate))),
                                "rho_pref_follow": _spearman(rate, c["pref"])[0],
                                "switch_on_follow_rate": switch, "switch_on_follow_rate_status": status,
                                "switch_on_follow_rate_distance": distance,
                                "observed_switch_rc": observed, "observed_switch_rc_status": observed_status})
                    summary_rows.append(row)
                    for band_label, centre in zip(BAND_LABELS, BAND_CENTRES):
                        inside = np.flatnonzero(band == band_label)
                        if len(inside) == 0:
                            continue
                        yb, gb, wb = y[inside].ravel(), c["gaps"][inside].ravel(), c["wins"][inside].ravel()
                        acc_b, bal_b = _soft_accuracy(wb, yb)
                        band_rows.append({"comparison": label, "base": s.base, "shortcut": s.shortcut,
                                          "metric": metric_titles[metric], "label": name,
                                          "band": band_label, "band_centre": centre,
                                          "models": len(inside), "n_images": len(yb),
                                          "n_pos": int(yb.sum()), "n_neg": int(len(yb) - yb.sum()),
                                          "auroc": stats.auroc(gb, yb), "accuracy": acc_b,
                                          "balanced_accuracy": bal_b})
        model_rows.append(base_frame)
    summary = pd.DataFrame(summary_rows)
    summary["family"] = "behaviour:" + summary.label
    summary = _adjust(summary, p_column="p_auroc")
    summary["headline"] = (summary.label == PRIMARY_LABEL) & (summary.comparison == "shortcut vs full object")
    return pd.concat(model_rows, ignore_index=True), summary, pd.DataFrame(band_rows)


def headline_holm(responsiveness, behaviour_summary=None):
    """Holm over the union of headline rows; adds p_holm_headline to both tables."""
    responsiveness = responsiveness.copy()
    p = [responsiveness.loc[responsiveness.headline, "p_rho_rc"].to_numpy(dtype=float)]
    if behaviour_summary is not None:
        behaviour_summary = behaviour_summary.copy()
        p.append(behaviour_summary.loc[behaviour_summary.headline, "p_auroc"].to_numpy(dtype=float))
    adjusted = stats.holm(np.concatenate(p))
    responsiveness["p_holm_headline"] = np.nan
    n = int(responsiveness.headline.sum())
    responsiveness.loc[responsiveness.headline, "p_holm_headline"] = adjusted[:n]
    if behaviour_summary is not None:
        behaviour_summary["p_holm_headline"] = np.nan
        behaviour_summary.loc[behaviour_summary.headline, "p_holm_headline"] = adjusted[n:]
    return responsiveness, behaviour_summary


# ------------------------------------------------------------ MD7 confound

def confound_table(settings, comparisons, metric_titles):
    """Reliance vs object competence (accuracy without the shortcut) per setting,
    with partial Spearman correlations ("shortcut vs full object"). Sorted so
    the least collinear setting comes first."""
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    first, second = maps["shortcut vs full object"]
    rows = []
    for s in settings:
        accuracy = s.models.test_accuracy_no_shortcut.to_numpy(dtype=float)
        rel_acc = _spearman(s.raw, accuracy)[0]
        rc_acc = _spearman(s.rc, accuracy)[0]
        for metric in FAITHFULNESS_METRICS:
            pref = s.comparison(first, second, metric)["pref"]
            partial, p_partial = stats.partial_spearman(s.raw, pref, accuracy)
            partial_acc, p_acc = stats.partial_spearman(accuracy, pref, s.raw)
            partial_rc, p_rc = stats.partial_spearman(s.rc, pref, accuracy)
            rows.append({"base": s.base, "shortcut": s.shortcut, "models": s.n_models,
                         "rho_reliance_competence": rel_acc, "rho_rc_competence": rc_acc,
                         "metric": metric_titles[metric],
                         "rho_rel": _spearman(s.raw, pref)[0], "rho_acc": _spearman(accuracy, pref)[0],
                         "partial_rel_given_acc": partial, "p_partial_rel_given_acc": p_partial,
                         "partial_acc_given_rel": partial_acc, "p_partial_acc_given_rel": p_acc,
                         "partial_rc_given_acc": partial_rc, "p_partial_rc_given_acc": p_rc})
    frame = pd.DataFrame(rows)
    frame["_order"] = frame.rho_reliance_competence.abs()
    return frame.sort_values(["_order", "base", "shortcut"], kind="stable").drop(columns="_order").reset_index(drop=True)

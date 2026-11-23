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
    Correlation p-values (responsiveness, methods) are Spearman's t
    approximation over the models of a setting; they do NOT come from the
    two-level bootstrap (only their intervals do).
    Metric-pair p-values are percentile bootstrap: two-sided
    min(1, 2 min(P(d <= 0), P(d >= 0))), floored at 1/(reps + 1).
    Behaviour AUROC p-values (vs 0.5) are normal-approximation z-tests with the
    two-level bootstrap SE (stats.bootstrap_z_p), so they are not floored; the
    floored percentile value is kept as p_auroc_boot. The class-stratified
    AUROCs (*_strat, PRIMARY) use a one-sided test (H1: AUROC > 0.5); the
    unstratified ones are two-sided and carry a direction column.
    BH and Holm are applied within each family:
        responsiveness:<comparison>   all settings x metrics of one comparison
        methods                       all settings x methods
        metric_pairs:<comparison>     all settings x metric pairs
        behaviour:<label>             all settings x comparisons x metrics
                                      (each AUROC column adjusted separately:
                                      p_bh, p_bh_within, p_bh_strat,
                                      p_bh_within_strat)
    Holm over the union of the headline rows ("shortcut vs full object" and
    "shortcut vs random" responsiveness, and both the pooled and the
    within-model class-stratified one-sided primary-label behaviour AUROC for
    "shortcut vs full object"): p_holm_headline in the responsiveness table,
    p_holm_headline_strat and p_holm_headline_within_strat in the behaviour
    summary.

Switch status
    "identified": a model within +-IDENTIFY_WINDOW of the switch (as specified).
    switch_bracketed (sensitivity): models within +-IDENTIFY_WINDOW on both
    sides. Identified switches within EDGE_WINDOW points of the lowest-reliance
    model get the status "identified at lowest-reliance edge": the metric
    already prefers the shortcut near the lowest reliance (over-detection),
    which is not a mid-range switch. Switch CIs are empty when the full-data
    switch is NaN.

Per-image behaviour labels (MD4)
    follows_shortcut_causal (PRIMARY): 1 if the moved shortcut wins (the
    conflicting image is predicted as the shortcut's class) and the image
    without shortcut is not predicted as that class; 0 if the moved shortcut
    does not win; undetermined (NaN, excluded and counted in
    undetermined_rate) if the model predicts the shortcut's class already
    without the shortcut. Without this rule an ordinary error on a binary task
    counts as shortcut use (the error floor that r_c removes from reliance).
    follows_shortcut (check): moved-shortcut pass only.
    shortcut_dominant (check): delta_s > delta_o.

Audit-2 additions (M1-M4; new tables only, existing ones unchanged)
    Crossed bootstrap (N4): Setting.crossed holds picks from
    stats.crossed_picks (seed + 1): models within dose, ONE image resample per
    replicate shared by all models, because every model is scored on the same
    images. comparison_crossed gives the crossed preference replicates and
    rho replicates computed after jittering ties (rng seed + 2); rho intervals
    are BCa with a leave-one-model-out jackknife. Significance still comes
    from the model-level Spearman test. Table: intervals_crossed.
    Behavioural switch (N1, N2, N5): behaviour_switch applies switch_status
    (+-IDENTIFY_WINDOW identification and bracketing) to the follow-rate
    switch of every label, with two-level and crossed CIs and the
    undetermined share; switch_differences is the paired bootstrap of
    (behavioural - metric) switch on the same replicates.
    Over-detection (N3): overdetection splits the lowest-dose preference into
    S-O (shortcut vs object), R-O (random vs object) and S-R (shortcut vs
    random); excess = S-O - R-O; FPR/TPR with Clopper-Pearson intervals, a
    margin FPR (a model over-detects only if its image-bootstrap 95% lower
    bound of preference exceeds 0.5) and FPR in the r_c bands <= 10 and
    10-50. excess_preference_models has the per-model values.
    Small analyses: behaviour_calibration (N6, calibration on determined
    images), confound_rc (N7, partial rho on r_c with BH), correct_only (N8,
    correctly classified aligned images only), behaviour_power (N9, minority
    label counts and the Hanley-McNeil SE of a per-model AUROC at 0.5).
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import stats
from .explain import METHODS
from .metrics import FAITHFULNESS_METRICS, ROAD_METRICS

KEYS = ["base", "shortcut", "dose", "seed"]
# Comparisons whose ideal curve crosses 0.5 at 50% reliance: switch points
# are only computed for these ("not applicable" elsewhere).
SWITCH_COMPARISONS = ("shortcut vs full object", "shortcut vs size-matched object")
PAIR_COMPARISONS = ("shortcut vs full object", "shortcut vs size-matched object", "random vs full object")
HEADLINE_COMPARISONS = ("shortcut vs full object", "shortcut vs random")
BEHAVIOUR_COMPARISONS = SWITCH_COMPARISONS
BEHAVIOUR_LABELS = ("follows_shortcut_causal", "follows_shortcut", "shortcut_dominant")  # primary, checks
PRIMARY_LABEL = "follows_shortcut_causal"
IDENTIFY_WINDOW = 10
EDGE_WINDOW = 5
RELIANCE_BANDS = [-np.inf, 20, 40, 60, 80, np.inf]
BAND_LABELS = ["<20", "20-40", "40-60", "60-80", ">=80"]
BAND_CENTRES = [10, 30, 50, 70, 90]
CHUNK = 250
NAN = float("nan")


def _higher(metric):
    """Whether a higher AUC is better (existing metrics, then exact ROAD)."""
    if metric in FAITHFULNESS_METRICS:
        return FAITHFULNESS_METRICS[metric][3]
    return ROAD_METRICS[metric][1]


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


def _adjust(frame, family_column="family", p_column="p", suffix=""):
    """Append p_bh<suffix> and p_holm<suffix>, computed within each family."""
    frame = frame.copy()
    bh_col, holm_col = f"p_bh{suffix}", f"p_holm{suffix}"
    frame[bh_col], frame[holm_col] = np.nan, np.nan
    for _, index in frame.groupby(family_column).groups.items():
        p = frame.loc[index, p_column].to_numpy(dtype=float)
        frame.loc[index, bh_col] = stats.bh(p)
        frame.loc[index, holm_col] = stats.holm(p)
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
        self.seed = seed
        self._crossed = None
        self._cache = {}

    @property
    def crossed(self):
        """(P, I) of the crossed bootstrap (seed + 1), drawn on first use."""
        if self._crossed is None:
            self._crossed = stats.crossed_picks(self.dose, self.n_images, self.reps, self.seed + 1)
        return self._crossed

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

    def resample(self, values, reducer, picks=None):
        """Apply reducer(sub, axis=2) to values[P, I] in chunks: (reps, models).
        picks: optional (P, I), default the two-level (self.P, self.I)."""
        P, I = (self.P, self.I) if picks is None else picks
        out = np.empty(P.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for s in range(0, self.reps, CHUNK):
                sub = values[P[s:s + CHUNK, :, None], I[s:s + CHUNK]]
                out[s:s + CHUNK] = reducer(sub, axis=2)
        return out

    def pooled(self, values, slots, reducer, picks=None):
        """reducer over the pooled resampled images of the given slots: (reps,)."""
        P, I = (self.P, self.I) if picks is None else picks
        out = np.empty(self.reps)
        for s in range(0, self.reps, CHUNK):
            sub = values[P[s:s + CHUNK, slots, None], I[s:s + CHUNK][:, slots]]
            out[s:s + CHUNK] = reducer(sub.reshape(sub.shape[0], -1))
        return out

    def comparison(self, first, second, metric):
        """Cached per-image wins/gaps, model preferences and their bootstrap."""
        key = (first, second, metric)
        if key not in self._cache:
            higher = _higher(metric)
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

    def comparison_crossed(self, first, second, metric):
        """Crossed-bootstrap model preferences (boot) and jittered rho replicates."""
        key = ("crossed", first, second, metric)
        if key not in self._cache:
            wins = self.comparison(first, second, metric)["wins"]
            P = self.crossed[0]
            boot = self.resample(wins, np.mean, self.crossed)
            rng = np.random.default_rng(self.seed + 2)
            self._cache[key] = {
                "boot": boot,
                "rho_rc_boot": stats.spearman_rows(stats.jitter(self.rc[P], rng), stats.jitter(boot, rng)),
                "rho_raw_boot": stats.spearman_rows(stats.jitter(self.raw[P], rng), stats.jitter(boot, rng)),
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


EDGE_STATUS = "identified at lowest-reliance edge"


def switch_status(x, y):
    """Isotonic switch of y on x with its identification columns:
    (switch, status, nearest distance, distance below, distance above, bracketed, at_edge)."""
    x = np.asarray(x, dtype=float)
    switch, status = stats.isotonic_switch(x, y)
    distance, below, above, bracketed, edge = NAN, NAN, NAN, False, False
    if status == "crossed":
        distance, status = stats.switch_identification(switch, x, IDENTIFY_WINDOW)
        below, above, bracketed = stats.switch_bracketing(switch, x, IDENTIFY_WINDOW)
        edge = bool(switch - np.nanmin(x) <= EDGE_WINDOW)
        if status == "identified" and edge:
            status = EDGE_STATUS
    return switch, status, distance, below, above, bracketed, edge


def _switch_columns(s, c, label):
    nan_row = {"switch_iso_rc": NAN, "switch_iso_rc_ci_low": NAN, "switch_iso_rc_ci_high": NAN,
               "switch_found_share": NAN, "switch_status": "not applicable",
               "nearest_model_distance": NAN, "switch_distance_below": NAN, "switch_distance_above": NAN,
               "switch_bracketed": False, "switch_at_low_edge": False,
               "switch_4pl_rc": NAN, "fit_4pl_ok": False,
               "switch_iso_raw": NAN, "switch_interp_rc": NAN}
    if label not in SWITCH_COMPARISONS:
        return nan_row
    pref = c["pref"]
    switch, status, distance, below, above, bracketed, edge = switch_status(s.rc, pref)
    boot = np.array([stats.isotonic_switch(s.rc[s.P[r]], c["boot"][r])[0] for r in range(s.reps)])
    found = ~np.isnan(boot)
    share = float(found.mean())
    # No interval without a full-data switch (a CI from the crossing replicates
    # alone would be conditional on crossing).
    low, high = _ci(boot) if share >= 0.5 and not np.isnan(switch) else (NAN, NAN)
    switch_4pl, _, ok = stats.logistic4_switch(s.rc, pref)
    levels = pd.DataFrame({"dose": s.dose, "rc": s.rc, "p": pref}).groupby("dose").mean()
    return {"switch_iso_rc": switch, "switch_iso_rc_ci_low": low, "switch_iso_rc_ci_high": high,
            "switch_found_share": share, "switch_status": status,
            "nearest_model_distance": distance, "switch_distance_below": below,
            "switch_distance_above": above, "switch_bracketed": bracketed, "switch_at_low_edge": edge,
            "switch_4pl_rc": switch_4pl, "fit_4pl_ok": bool(ok),
            "switch_iso_raw": stats.isotonic_switch(s.raw, pref)[0],
            "switch_interp_rc": _first_crossing(levels.rc, levels.p)}


def responsiveness_corrected(settings, comparisons, metric_titles, metrics=None):
    """PRIMARY metric table: one row per comparison x setting x metric
    (metrics: default the four existing ones)."""
    rows = []
    for first, second, label in _available(settings, comparisons):
        for s in settings:
            for metric in (metrics or FAITHFULNESS_METRICS):
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

def add_causal_label(behaviour):
    """Add follows_shortcut_causal (PRIMARY) from the stored passes; no new forward passes.

    1 = the moved shortcut wins and the image without shortcut is not
    predicted as the shortcut's class; 0 = the moved shortcut does not win;
    NaN (undetermined) = the image without shortcut is already predicted as
    the shortcut's class, so the moved pass cannot separate shortcut use from
    an ordinary error."""
    behaviour = behaviour.copy()
    moved = behaviour.pred_conflicting.to_numpy() == behaviour.conflict_class.to_numpy()
    undetermined = behaviour.pred_none.to_numpy() == behaviour.conflict_class.to_numpy()
    behaviour["follows_shortcut_causal"] = np.where(undetermined, np.nan, moved.astype(float))
    return behaviour


def _label_matrix(s, behaviour, column):
    frame = behaviour.merge(s.models[KEYS], on=KEYS).sort_values(["dose", "seed", "index"])
    values = frame[column].to_numpy(dtype=float)
    if len(values) != s.n_models * s.n_images or not (
            frame["index"].to_numpy().reshape(s.n_models, s.n_images) == s.indices).all():
        raise ValueError(f"{s.base}/{s.shortcut}: behaviour labels do not match the scored images")
    return values.reshape(s.n_models, s.n_images)


def _stratified_rows(scores, labels, valid, classes, class_values):
    """Row-wise class-stratified AUROC: mean over the true classes that have
    both labels in a row of the within-class AUROC; NaN if no class has both."""
    total, count = np.zeros(len(scores)), np.zeros(len(scores))
    for c in class_values:
        a = stats.auroc_rows(scores, labels, valid & (classes == c))
        ok = ~np.isnan(a)
        total += np.where(ok, a, 0.0)
        count += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _boot_auroc(s, gaps, labels, classes):
    """Two-level bootstrap AUROC per replicate, ignoring undetermined (NaN) labels.

    Returns a dict of (reps,) arrays: pooled = AUROC over the resampled images
    of all slots together; within = median over slots of the per-slot AUROC
    (slots with one label only are skipped); pooled_strat and within_strat =
    the same with the AUROC computed within each true class (classes, same
    shape as labels; images keep their class when resampled) and averaged
    over the classes that have both labels."""
    class_values = np.unique(classes)
    out = {"pooled": [], "within": [], "pooled_strat": [], "within_strat": []}
    for r0 in range(0, s.reps, CHUNK):
        picks, images = s.P[r0:r0 + CHUNK, :, None], s.I[r0:r0 + CHUNK]
        n = len(images)
        g, y, k = gaps[picks, images], labels[picks, images], classes[picks, images]
        valid = ~np.isnan(y)
        y = np.nan_to_num(y)
        flat = [a.reshape(n, -1) for a in (g, y, valid, k)]
        slot = [a.reshape(-1, s.n_images) for a in (g, y, valid, k)]
        out["pooled"].append(stats.auroc_rows(*flat[:3]))
        out["pooled_strat"].append(_stratified_rows(*flat, class_values))
        per_slot = stats.auroc_rows(*slot[:3]).reshape(n, s.n_models)
        per_slot_strat = _stratified_rows(*slot, class_values).reshape(n, s.n_models)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["within"].append(np.nanmedian(per_slot, axis=1))
            out["within_strat"].append(np.nanmedian(per_slot_strat, axis=1))
    return {name: np.concatenate(values) for name, values in out.items()}


def _auroc_masked(scores, labels):
    keep = ~np.isnan(labels)
    return stats.auroc(scores[keep], labels[keep])


def _auroc_stratified(scores, labels, classes):
    """Mean over true classes (with both labels) of the within-class AUROC; NaN if none."""
    values = [_auroc_masked(scores[classes == c], labels[classes == c]) for c in np.unique(classes)]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else NAN


def _direction(estimate):
    if np.isnan(estimate):
        return ""
    return "above 0.5" if estimate > 0.5 else ("below 0.5" if estimate < 0.5 else "at 0.5")


def _soft_accuracy(wins, labels):
    """accuracy = mean(1 - |w - y|); balanced = mean of mean(w | y = 1) and
    mean(1 - w | y = 0). Undetermined (NaN) labels are ignored."""
    keep = ~np.isnan(labels)
    wins, labels = wins[keep], labels[keep]
    if len(labels) == 0:
        return NAN, NAN
    accuracy = float(np.mean(1 - np.abs(wins - labels)))
    pos, neg = labels == 1, labels == 0
    balanced = (float(wins[pos].mean()) + float((1 - wins[neg]).mean())) / 2 if pos.any() and neg.any() else NAN
    return accuracy, balanced


def _nanmean_rows(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=1)


def behaviour_tables(settings, behaviour, comparisons, metric_titles):
    """(behaviour_models, behaviour_summary, behaviour_by_reliance) DataFrames.

    The per-image label replaces the assumed 50% ideal. AUROC uses the
    per-image AUC gap (positive = shortcut map rated better) as the score of
    "the model follows the shortcut on this image"; accuracy uses the win
    score w (1, 0 or 0.5 for a tie). Undetermined labels (NaN) are excluded
    everywhere; label rates are over the determined images of each model.

    auroc_pooled mixes between-model and within-model signal (high-reliance
    models have both larger gaps and more label-1 images). median_model_auroc
    (median over models of the per-model AUROC) measures detection inside a
    model; both have two-level CIs and two-sided z-test p-values, with a
    direction column (above / below 0.5).

    On binary tasks the label is tied to the true class (a model that fails
    without its shortcut predicts one class), and the AUC gap also differs by
    class, so the unstratified AUROCs can be pushed far below 0.5 by class
    differences alone (Simpson's paradox). The *_strat columns (PRIMARY,
    headline) compute the AUROC within each true class and average over the
    classes that have both labels: auroc_pooled_strat over the pooled images,
    median_model_auroc_strat per model then median over models. Their
    p-values are one-sided z-tests (H1: AUROC > 0.5, "the metric detects
    followed images"), so a result below 0.5 is never significant.
    """
    if "follows_shortcut_causal" not in behaviour:
        behaviour = add_causal_label(behaviour)
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    model_rows, summary_rows, band_rows = [], [], []
    for s in settings:
        labels = {name: _label_matrix(s, behaviour, name) for name in BEHAVIOUR_LABELS}
        classes = _label_matrix(s, behaviour, "label")  # true class of each scored image
        rates = {name: _nanmean_rows(values) for name, values in labels.items()}
        undetermined = np.isnan(labels[PRIMARY_LABEL]).mean(axis=1)
        base_frame = s.models[KEYS + ["reliance", "reliance_corrected"]].copy()
        base_frame["follow_rate"] = rates[PRIMARY_LABEL]
        base_frame["undetermined_rate"] = undetermined
        base_frame["follow_rate_moved_only"] = rates["follows_shortcut"]
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
                    gaps = c["gaps"]
                    per_model = np.array([_auroc_masked(gaps[m], y[m]) for m in range(s.n_models)])
                    per_model_strat = np.array([_auroc_stratified(gaps[m], y[m], classes[m])
                                                for m in range(s.n_models)])
                    base_frame[f"auroc_{name}_{metric}_{tag}"] = per_model
                    base_frame[f"auroc_strat_{name}_{metric}_{tag}"] = per_model_strat
                    boot = _boot_auroc(s, gaps, y, classes)
                    accuracy, balanced = _soft_accuracy(c["wins"].ravel(), y.ravel())
                    switch, status, distance, _, _, bracketed, edge = switch_status(100 * rate, c["pref"])
                    observed, observed_status = stats.isotonic_switch(s.rc, rate)
                    pooled = _auroc_masked(gaps.ravel(), y.ravel())
                    pooled_strat = _auroc_stratified(gaps.ravel(), y.ravel(), classes.ravel())
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        within = float(np.nanmedian(per_model))
                        within_strat = float(np.nanmedian(per_model_strat))
                    row = {"comparison": label, "base": s.base, "shortcut": s.shortcut,
                           "metric": metric_titles[metric], "label": name, "models": s.n_models,
                           "label_rate": float(np.nanmean(y)),
                           "undetermined_rate": float(np.isnan(y).mean()),
                           "auroc_pooled": pooled}
                    row["auroc_ci_low"], row["auroc_ci_high"] = _ci(boot["pooled"])
                    row["p_auroc"] = stats.bootstrap_z_p(pooled, boot["pooled"], 0.5)
                    row["p_auroc_boot"] = _boot_p(boot["pooled"], 0.5)
                    row["direction"] = _direction(pooled)
                    row["median_model_auroc"] = within
                    row["median_model_auroc_ci_low"], row["median_model_auroc_ci_high"] = _ci(boot["within"])
                    row["p_auroc_within"] = stats.bootstrap_z_p(within, boot["within"], 0.5)
                    row["direction_within"] = _direction(within)
                    row["models_with_both_classes"] = int((~np.isnan(per_model)).sum())
                    # class-stratified (PRIMARY for the headline test): one-sided H1 AUROC > 0.5
                    row["auroc_pooled_strat"] = pooled_strat
                    row["auroc_strat_ci_low"], row["auroc_strat_ci_high"] = _ci(boot["pooled_strat"])
                    row["p_auroc_strat"] = stats.bootstrap_z_p(pooled_strat, boot["pooled_strat"], 0.5, "greater")
                    row["direction_strat"] = _direction(pooled_strat)
                    row["median_model_auroc_strat"] = within_strat
                    row["median_model_auroc_strat_ci_low"], row["median_model_auroc_strat_ci_high"] = _ci(
                        boot["within_strat"])
                    row["p_auroc_within_strat"] = stats.bootstrap_z_p(
                        within_strat, boot["within_strat"], 0.5, "greater")
                    row["direction_within_strat"] = _direction(within_strat)
                    row["models_with_both_labels_in_a_class"] = int((~np.isnan(per_model_strat)).sum())
                    row.update({"accuracy": accuracy, "balanced_accuracy": balanced,
                                "calibration_gap": float(np.nanmean(c["pref"] - rate)),
                                "calibration_mae": float(np.nanmean(np.abs(c["pref"] - rate))),
                                "rho_pref_follow": _spearman(rate, c["pref"])[0],
                                "switch_on_follow_rate": switch, "switch_on_follow_rate_status": status,
                                "switch_on_follow_rate_distance": distance,
                                "switch_on_follow_rate_bracketed": bracketed,
                                "observed_switch_rc": observed, "observed_switch_rc_status": observed_status})
                    summary_rows.append(row)
                    for band_label, centre in zip(BAND_LABELS, BAND_CENTRES):
                        inside = np.flatnonzero(band == band_label)
                        if len(inside) == 0:
                            continue
                        yb, gb, wb = y[inside].ravel(), c["gaps"][inside].ravel(), c["wins"][inside].ravel()
                        determined = ~np.isnan(yb)
                        acc_b, bal_b = _soft_accuracy(wb, yb)
                        band_rows.append({"comparison": label, "base": s.base, "shortcut": s.shortcut,
                                          "metric": metric_titles[metric], "label": name,
                                          "band": band_label, "band_centre": centre,
                                          "models": len(inside), "n_images": int(determined.sum()),
                                          "n_undetermined": int((~determined).sum()),
                                          "n_pos": int(np.nansum(yb)), "n_neg": int(determined.sum() - np.nansum(yb)),
                                          "auroc": _auroc_masked(gb, yb),
                                          "auroc_strat": _auroc_stratified(gb, yb, classes[inside].ravel()),
                                          "accuracy": acc_b,
                                          "balanced_accuracy": bal_b})
        model_rows.append(base_frame)
    summary = pd.DataFrame(summary_rows)
    summary["family"] = "behaviour:" + summary.label
    summary = _adjust(summary, p_column="p_auroc")
    summary = _adjust(summary, p_column="p_auroc_within", suffix="_within")
    summary = _adjust(summary, p_column="p_auroc_strat", suffix="_strat")
    summary = _adjust(summary, p_column="p_auroc_within_strat", suffix="_within_strat")
    summary["headline"] = (summary.label == PRIMARY_LABEL) & (summary.comparison == "shortcut vs full object")
    return pd.concat(model_rows, ignore_index=True), summary, pd.DataFrame(band_rows)


def headline_holm(responsiveness, behaviour_summary=None):
    """Holm over the union of headline rows; adds p_holm_headline to the
    responsiveness table and, for the behaviour test, p_holm_headline_strat
    (pooled) and p_holm_headline_within_strat (within-model) on the
    class-stratified one-sided AUROC p-values."""
    responsiveness = responsiveness.copy()
    p = [responsiveness.loc[responsiveness.headline, "p_rho_rc"].to_numpy(dtype=float)]
    if behaviour_summary is not None:
        behaviour_summary = behaviour_summary.copy()
        p.append(behaviour_summary.loc[behaviour_summary.headline, "p_auroc_strat"].to_numpy(dtype=float))
        p.append(behaviour_summary.loc[behaviour_summary.headline, "p_auroc_within_strat"].to_numpy(dtype=float))
    adjusted = stats.holm(np.concatenate(p))
    responsiveness["p_holm_headline"] = np.nan
    n = int(responsiveness.headline.sum())
    responsiveness.loc[responsiveness.headline, "p_holm_headline"] = adjusted[:n]
    if behaviour_summary is not None:
        k = int(behaviour_summary.headline.sum())
        behaviour_summary["p_holm_headline_strat"] = np.nan
        behaviour_summary["p_holm_headline_within_strat"] = np.nan
        behaviour_summary.loc[behaviour_summary.headline, "p_holm_headline_strat"] = adjusted[n:n + k]
        behaviour_summary.loc[behaviour_summary.headline, "p_holm_headline_within_strat"] = adjusted[n + k:]
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


# ------------------------------------------------------------ audit-2 additions (M1-M4)

LABEL_MEASURES = {
    "follows_shortcut_causal": "shortcut sufficient under conflict, errors removed",
    "follows_shortcut": "shortcut sufficient under conflict (moved-shortcut pass only)",
    "shortcut_dominant": "shortcut more necessary than the object on aligned images",
}
OVERDETECTION_MAPS = {"so": "shortcut vs full object", "ro": "random vs full object", "sr": "shortcut vs random"}
KINDS = ("two_level", "crossed")


def _picks(s, kind):
    return (s.P, s.I) if kind == "two_level" else s.crossed


def _switch_boot(s, values_boot, picks):
    """Isotonic switch on r_c in every replicate of a (reps, models) array."""
    P = picks[0]
    return np.array([stats.isotonic_switch(s.rc[P[r]], values_boot[r])[0] for r in range(s.reps)])


def _switch_ci(switch, boot):
    """(low, high, found_share) by the _switch_columns rule: no interval when
    the full-data switch is NaN or fewer than half of the replicates cross."""
    share = float((~np.isnan(boot)).mean())
    low, high = _ci(boot) if share >= 0.5 and not np.isnan(switch) else (NAN, NAN)
    return low, high, share


def _metric_switch_boot(s, first, second, metric, kind):
    key = ("switch_boot", first, second, metric, kind)
    if key not in s._cache:
        boot = (s.comparison(first, second, metric) if kind == "two_level"
                else s.comparison_crossed(first, second, metric))["boot"]
        s._cache[key] = _switch_boot(s, boot, _picks(s, kind))
    return s._cache[key]


def _follow_boot(s, name, labels, kind):
    """(follow-rate replicates, switch replicates) of one label matrix; the
    follow rate of a slot is the nanmean of its resampled images' labels.
    Cached per label name: one behaviour frame per run."""
    key = ("follow_boot", name, kind)
    if key not in s._cache:
        picks = _picks(s, kind)
        rate_boot = s.resample(labels, np.nanmean, picks)
        s._cache[key] = (rate_boot, _switch_boot(s, rate_boot, picks))
    return s._cache[key]


def _with_causal(behaviour):
    return behaviour if "follows_shortcut_causal" in behaviour else add_causal_label(behaviour)


def _jackknife_rho(x, y):
    """Leave-one-model-out Spearman correlations."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.array([_spearman(np.delete(x, m), np.delete(y, m))[0] for m in range(len(x))])


def _report_text(switch, status, below, above, bracketed):
    """How a switch should be quoted: '54.5', '54.5 (not bracketed)' or
    '> 42.7 (not identified; no model between 42.7 and 96.0)'."""
    if np.isnan(switch):
        return status
    if status.startswith("not identified"):
        low, high = switch - below, switch + above
        return f"> {low:.1f} (not identified; no model between {low:.1f} and {high:.1f})"
    if status == EDGE_STATUS:
        return f"{switch:.1f} (at lowest-reliance edge" + ("" if bracketed else "; not bracketed") + ")"
    return f"{switch:.1f}" if bracketed else f"{switch:.1f} (not bracketed)"


def intervals_crossed(settings, comparisons, metric_titles):
    """N4: crossed (models x images) bootstrap intervals for levels and switch
    points, BCa intervals (jittered, crossed replicates) for rho. Point values
    are those of responsiveness_corrected."""
    rows = []
    for first, second, label in _available(settings, comparisons):
        for s in settings:
            for metric in FAITHFULNESS_METRICS:
                c = s.comparison(first, second, metric)
                cc = s.comparison_crossed(first, second, metric)
                pref = c["pref"]
                rho_rc, p_rc = _spearman(s.rc, pref)
                rho_raw = _spearman(s.raw, pref)[0]
                row = {"comparison": label, "base": s.base, "shortcut": s.shortcut,
                       "metric": metric_titles[metric], "models": s.n_models,
                       "rho_rc": rho_rc, "p_rho_rc": p_rc}
                row["rho_rc_bca_ci_low"], row["rho_rc_bca_ci_high"] = stats.bca_interval(
                    rho_rc, cc["rho_rc_boot"], _jackknife_rho(s.rc, pref))
                boot = cc["rho_rc_boot"][~np.isnan(cc["rho_rc_boot"])]
                row["rho_rc_share_boot_below"] = (float(np.mean(boot < rho_rc))
                                                  if len(boot) and not np.isnan(rho_rc) else NAN)
                row["rho_raw"] = rho_raw
                row["rho_raw_bca_ci_low"], row["rho_raw_bca_ci_high"] = stats.bca_interval(
                    rho_raw, cc["rho_raw_boot"], _jackknife_rho(s.raw, pref))
                row["pref_low"] = float(pref[s.low].mean())
                row["pref_low_crossed_ci_low"], row["pref_low_crossed_ci_high"] = _ci(
                    cc["boot"][:, s.low].mean(axis=1))
                row["pref_high"] = float(pref[s.high].mean())
                row["pref_high_crossed_ci_low"], row["pref_high_crossed_ci_high"] = _ci(
                    cc["boot"][:, s.high].mean(axis=1))
                if label in SWITCH_COMPARISONS:
                    switch, status = switch_status(s.rc, pref)[:2]
                    low, high, share = _switch_ci(switch, _metric_switch_boot(s, first, second, metric, "crossed"))
                else:
                    switch, status, low, high, share = NAN, "not applicable", NAN, NAN, NAN
                row.update({"switch_iso_rc": switch, "switch_status": status, "switch_crossed_ci_low": low,
                            "switch_crossed_ci_high": high, "switch_crossed_found_share": share})
                rows.append(row)
    return pd.DataFrame(rows)


def behaviour_switch(settings, behaviour):
    """N1/N2: behavioural (follow-rate) switch on r_c per setting x label, with
    the identification and bracketing rule, two-level and crossed CIs and the
    undetermined share. The switch equals behaviour_summary's observed_switch_rc."""
    behaviour = _with_causal(behaviour)
    rows = []
    for s in settings:
        for name in BEHAVIOUR_LABELS:
            labels = _label_matrix(s, behaviour, name)
            rate = _nanmean_rows(labels)
            switch, status, distance, below, above, bracketed, _ = switch_status(s.rc, rate)
            undetermined = np.isnan(labels).mean(axis=1)
            rho_u, p_u = _spearman(s.rc, undetermined)
            row = {"base": s.base, "shortcut": s.shortcut, "label": name, "primary": name == PRIMARY_LABEL,
                   "measures": LABEL_MEASURES[name], "models": s.n_models,
                   "follow_rate_min": float(np.nanmin(rate)), "follow_rate_max": float(np.nanmax(rate)),
                   "behavioural_switch_rc": switch, "status": status, "nearest_model_distance": distance,
                   "distance_below": below, "distance_above": above, "bracketed": bracketed,
                   "report_text": _report_text(switch, status, below, above, bracketed)}
            for kind, prefix in (("two_level", "switch"), ("crossed", "switch_crossed")):
                low, high, share = _switch_ci(switch, _follow_boot(s, name, labels, kind)[1])
                row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"] = low, high
                if kind == "two_level":
                    row["switch_found_share"] = share
            row.update({"undetermined_rate_mean": float(undetermined.mean()),
                        "undetermined_rate_max": float(undetermined.max()),
                        "rho_undetermined_rc": rho_u, "p_undetermined_rc": p_u})
            rows.append(row)
    return pd.DataFrame(rows)


def _verdict(metric_switch, metric_status, behavioural_switch, behavioural_status, low, high):
    missing = []
    if np.isnan(metric_switch):
        missing.append(f"metric switch: {metric_status}")
    if np.isnan(behavioural_switch):
        missing.append(f"behavioural switch: {behavioural_status}")
    if missing:
        return f"not testable ({'; '.join(missing)})"
    if np.isnan(low):
        return "not testable (both switches found in fewer than half of the replicates)"
    if low > 0:
        return "metric earlier"
    if high < 0:
        return "metric later"
    return "not distinguishable"


def switch_differences(settings, behaviour, comparisons, metric_titles):
    """N5: paired bootstrap of (behavioural - metric) switch on r_c. In every
    replicate the metric's preference and the follow rate use the same model
    and image picks, so the difference is paired."""
    behaviour = _with_causal(behaviour)
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    rows = []
    for s in settings:
        labels = {name: _label_matrix(s, behaviour, name) for name in BEHAVIOUR_LABELS}
        for label in SWITCH_COMPARISONS:
            if label not in maps:
                continue
            first, second = maps[label]
            for metric in FAITHFULNESS_METRICS:
                pref = s.comparison(first, second, metric)["pref"]
                m_switch, m_status = switch_status(s.rc, pref)[:2]
                for name in BEHAVIOUR_LABELS:
                    b_switch, b_status = switch_status(s.rc, _nanmean_rows(labels[name]))[:2]
                    row = {"comparison": label, "base": s.base, "shortcut": s.shortcut,
                           "metric": metric_titles[metric], "label": name, "primary": name == PRIMARY_LABEL,
                           "models": s.n_models, "metric_switch_rc": m_switch, "metric_switch_status": m_status,
                           "behavioural_switch_rc": b_switch, "behavioural_status": b_status,
                           "difference": b_switch - m_switch}
                    defined = not (np.isnan(m_switch) or np.isnan(b_switch))
                    for kind, prefix in (("two_level", "diff"), ("crossed", "diff_crossed")):
                        diff = (_follow_boot(s, name, labels[name], kind)[1]
                                - _metric_switch_boot(s, first, second, metric, kind))
                        share = float((~np.isnan(diff)).mean())
                        low, high = _ci(diff) if defined and share >= 0.5 else (NAN, NAN)
                        row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"] = low, high
                        if kind == "two_level":
                            row["both_found_share"] = share
                    row["verdict"] = _verdict(m_switch, m_status, b_switch, b_status,
                                              row["diff_ci_low"], row["diff_ci_high"])
                    rows.append(row)
    return pd.DataFrame(rows)


def _overdetection_models(s, maps, metric):
    """Per-model S-O, R-O, S-R preferences, excess and the image-only
    bootstrap CI of S-O (model fixed, images from the two-level I)."""
    key = ("overdetection", metric)
    if key not in s._cache:
        c = {k: s.comparison(*maps[label], metric) for k, label in OVERDETECTION_MAPS.items()}
        fixed = (np.broadcast_to(np.arange(s.n_models), s.P.shape), s.I)
        image_boot = s.resample(c["so"]["wins"], np.mean, fixed)
        low, high = np.percentile(image_boot, [2.5, 97.5], axis=0)
        s._cache[key] = {"c": c, "pref_so": c["so"]["pref"], "pref_ro": c["ro"]["pref"],
                         "pref_sr": c["sr"]["pref"], "excess": c["so"]["pref"] - c["ro"]["pref"],
                         "so_ci_low": low, "so_ci_high": high}
    return s._cache[key]


def _rate_cp(says, mask, prefix):
    k, n = int((says & mask).sum()), int(mask.sum())
    low, high = stats.clopper_pearson(k, n)
    return {prefix: k / n if n else NAN, f"k_{prefix}": k, f"n_{prefix}": n,
            f"{prefix}_cp_low": low, f"{prefix}_cp_high": high}


def overdetection(settings, comparisons, metric_titles, metrics=None):
    """N3: what the low-reliance over-detection (shortcut vs full object) is
    made of, and its rates with intervals, a margin rule and r_c bands
    (metrics: default the four existing ones)."""
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    if not all(label in maps for label in OVERDETECTION_MAPS.values()):
        return pd.DataFrame()
    rows = []
    for s in settings:
        for metric in (metrics or FAITHFULNESS_METRICS):
            d = _overdetection_models(s, maps, metric)
            boots = {k: d["c"][k]["boot"] for k in OVERDETECTION_MAPS}
            row = {"comparison": "shortcut vs full object", "base": s.base, "shortcut": s.shortcut,
                   "metric": metric_titles[metric], "models": s.n_models,
                   "dose_low": float(s.dose[s.low][0]), "models_low": len(s.low)}
            for k in OVERDETECTION_MAPS:
                row[f"pref_{k}_low"] = float(d[f"pref_{k}"][s.low].mean())
                row[f"pref_{k}_low_ci_low"], row[f"pref_{k}_low_ci_high"] = _ci(boots[k][:, s.low].mean(axis=1))
            excess_boot = boots["so"] - boots["ro"]
            row["excess_low"] = float(d["excess"][s.low].mean())
            row["excess_low_ci_low"], row["excess_low_ci_high"] = _ci(excess_boot[:, s.low].mean(axis=1))
            rho, p = _spearman(s.rc, d["excess"])
            row["rho_excess_rc"], row["p_excess_two_sided"] = rho, p
            row["p_excess_one_sided"] = NAN if np.isnan(p) else (p / 2 if rho > 0 else 1 - p / 2)
            row["rho_excess_ci_low"], row["rho_excess_ci_high"] = _ci(stats.spearman_rows(s.rc[s.P], excess_boot))
            rates = stats.decision_rates(d["pref_so"], s.rc)
            says, above = d["pref_so"] > 0.5, s.rc > 50
            margin = d["so_ci_low"] > 0.5
            fpr, tpr = _rate_cp(says, ~above, "fpr"), _rate_cp(says, above, "tpr")
            row.update({"fpr": rates["fpr"], "k_fpr": fpr["k_fpr"], "n_below": rates["n_below"],
                        "fpr_cp_low": fpr["fpr_cp_low"], "fpr_cp_high": fpr["fpr_cp_high"],
                        "tpr": rates["tpr"], "n_above": rates["n_above"],
                        "tpr_cp_low": tpr["tpr_cp_low"], "tpr_cp_high": tpr["tpr_cp_high"]})
            m = _rate_cp(margin, ~above, "fpr_margin")
            row.update({"fpr_margin": m["fpr_margin"], "k_fpr_margin": m["k_fpr_margin"],
                        "fpr_margin_cp_low": m["fpr_margin_cp_low"], "fpr_margin_cp_high": m["fpr_margin_cp_high"]})
            for band, mask in (("rc_le10", s.rc <= 10), ("rc_10_50", (s.rc > 10) & (s.rc <= 50))):
                b = _rate_cp(says, mask, f"fpr_{band}")
                row.update({f"fpr_{band}": b[f"fpr_{band}"], f"n_{band}": b[f"n_fpr_{band}"],
                            f"fpr_{band}_cp_low": b[f"fpr_{band}_cp_low"],
                            f"fpr_{band}_cp_high": b[f"fpr_{band}_cp_high"]})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.insert(frame.columns.get_loc("p_excess_one_sided") + 1, "p_bh_excess_one_sided",
                 stats.bh(frame.p_excess_one_sided.to_numpy(dtype=float)))
    return frame


def excess_preference_models(settings, comparisons, metric_titles):
    """N3: per-model S-O, R-O, S-R preferences, excess = S-O - R-O and both
    over-detection decisions (point > 0.5; image-bootstrap lower bound > 0.5)."""
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    if not all(label in maps for label in OVERDETECTION_MAPS.values()):
        return pd.DataFrame()
    frames = []
    for s in settings:
        for metric in FAITHFULNESS_METRICS:
            d = _overdetection_models(s, maps, metric)
            frame = s.models[KEYS + ["reliance_corrected"]].copy()
            frame["metric"] = metric_titles[metric]
            for column in ("pref_so", "pref_ro", "pref_sr", "excess"):
                frame[column] = d[column]
            frame["pref_so_ci_low"], frame["pref_so_ci_high"] = d["so_ci_low"], d["so_ci_high"]
            frame["says_shortcut"] = d["pref_so"] > 0.5
            frame["says_shortcut_margin"] = d["so_ci_low"] > 0.5
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def behaviour_calibration(settings, behaviour, comparisons, metric_titles):
    """N6: calibration of the preference against the follow rate on the
    determined images only (the existing calibration_gap mixes in the win
    scores of undetermined images)."""
    behaviour = _with_causal(behaviour)
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    rows = []
    for label in SWITCH_COMPARISONS:
        if label not in maps:
            continue
        for s in settings:
            labels = {name: _label_matrix(s, behaviour, name) for name in BEHAVIOUR_LABELS}
            for metric in FAITHFULNESS_METRICS:
                c = s.comparison(*maps[label], metric)
                for name, y in labels.items():
                    rate = _nanmean_rows(y)
                    determined = ~np.isnan(y)
                    gap = _nanmean_rows(np.where(determined, c["wins"], np.nan)) - rate
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        rows.append({"comparison": label, "base": s.base, "shortcut": s.shortcut,
                                     "metric": metric_titles[metric], "label": name, "models": s.n_models,
                                     "calibration_gap": float(np.nanmean(c["pref"] - rate)),
                                     "calibration_gap_determined": float(np.nanmean(gap)),
                                     "calibration_mae_determined": float(np.nanmean(np.abs(gap))),
                                     "determined_share": float(determined.mean())})
    return pd.DataFrame(rows)


def confound_rc(confound):
    """N7: the confound table's partial correlations on r_c and on raw
    reliance, each with BH over all rows (same row order as confound)."""
    frame = confound[["base", "shortcut", "models", "metric", "rho_rc_competence", "partial_rc_given_acc",
                      "p_partial_rc_given_acc", "partial_rel_given_acc", "p_partial_rel_given_acc"]].copy()
    frame.insert(frame.columns.get_loc("p_partial_rc_given_acc") + 1, "p_bh_rc",
                 stats.bh(frame.p_partial_rc_given_acc.to_numpy(dtype=float)))
    frame["p_bh_raw"] = stats.bh(frame.p_partial_rel_given_acc.to_numpy(dtype=float))
    return frame.reset_index(drop=True)


def correct_only(settings, behaviour, comparisons, metric_titles):
    """N8: shortcut vs full object on correctly classified aligned images only
    (pred_aligned == label), next to the all-image values."""
    maps = {label: (first, second) for first, second, label in _available(settings, comparisons)}
    first, second = maps["shortcut vs full object"]
    rows = []
    for s in settings:
        correct = _label_matrix(s, behaviour, "pred_aligned") == _label_matrix(s, behaviour, "label")
        low20 = s.rc <= 20
        for metric in FAITHFULNESS_METRICS:
            c = s.comparison(first, second, metric)
            pref_c = _nanmean_rows(np.where(correct, c["wins"], np.nan))
            keep = ~np.isnan(pref_c)
            rho_c, p_c = _spearman(s.rc, pref_c)
            rates_c = stats.decision_rates(pref_c[keep], s.rc[keep])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pref_low_c = float(np.nanmean(pref_c[s.low]))
            rows.append({"comparison": "shortcut vs full object", "base": s.base, "shortcut": s.shortcut,
                         "metric": metric_titles[metric], "models": s.n_models,
                         "misclassified_share": float(1 - correct.mean()),
                         "misclassified_share_rc_le20": float(1 - correct[low20].mean()) if low20.any() else NAN,
                         "rho_rc_correct": rho_c, "p_rho_rc_correct": p_c, "pref_low_correct": pref_low_c,
                         "fpr_correct": rates_c["fpr"], "n_below_correct": rates_c["n_below"],
                         "rho_rc": _spearman(s.rc, c["pref"])[0], "pref_low": float(c["pref"][s.low].mean()),
                         "fpr": stats.decision_rates(c["pref"], s.rc)["fpr"]})
    return pd.DataFrame(rows)


def behaviour_power(settings, behaviour):
    """N9: how many images a per-model behaviour AUROC rests on: median over
    models of the minority-label count (determined images), of the smallest
    within-class minority count (classes with both labels), and of the
    Hanley-McNeil SE at AUROC 0.5, sqrt((n1 + n0 + 1) / (12 n1 n0))."""
    behaviour = _with_causal(behaviour)
    rows = []
    for s in settings:
        classes = _label_matrix(s, behaviour, "label")
        for name in BEHAVIOUR_LABELS:
            y = _label_matrix(s, behaviour, name)
            minority, within, se = [], [], []
            for m in range(s.n_models):
                determined = ~np.isnan(y[m])
                n1 = int((y[m][determined] == 1).sum())
                n0 = int(determined.sum()) - n1
                minority.append(min(n1, n0))
                se.append(np.sqrt((n1 + n0 + 1) / (12 * n1 * n0)) if n1 and n0 else NAN)
                counts = []
                for c in np.unique(classes[m]):
                    inside = determined & (classes[m] == c)
                    c1 = int((y[m][inside] == 1).sum())
                    c0 = int(inside.sum()) - c1
                    if c1 and c0:
                        counts.append(min(c1, c0))
                within.append(min(counts) if counts else NAN)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rows.append({"base": s.base, "shortcut": s.shortcut, "label": name, "models": s.n_models,
                             "median_minority_count": float(np.median(minority)),
                             "median_within_class_minority_count": float(np.nanmedian(within)),
                             "median_hanley_mcneil_se": float(np.nanmedian(se)),
                             "models_with_both_labels": int(np.sum(~np.isnan(se)))})
    return pd.DataFrame(rows)

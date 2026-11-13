"""Audit 2, reviewer 01: behavioural switch identification and a paired two-level
bootstrap of (behavioural switch - metric switch). Read-only; imports xai.stats only
for its pure functions. Run from the project root."""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from xai import stats  # noqa: E402

R = "results/dose_response/"
REPS, SEED, TOL = 1000, 123, 1e-4
models = pd.read_csv(R + "models.csv")
beh_sum = pd.read_csv(R + "behaviour_summary.csv")
oracle = pd.read_csv(R + "scores_oracle.csv")
beh = pd.concat([pd.read_csv(f) for f in glob.glob(R + "per_model/*_behaviour.csv")], ignore_index=True)
KEYS = ["base", "shortcut", "dose", "seed"]
moved = beh.pred_conflicting == beh.conflict_class
undet = beh.pred_none == beh.conflict_class
beh["causal"] = np.where(undet, np.nan, moved.astype(float))
METRICS = {"deletion_zero": False, "deletion_blur": False, "deletion_inpaint": False, "insertion_blur": True}


def h(t):
    print("\n" + "=" * 8, t, "=" * 8)


h("A. behavioural switch (follow rate crosses 0.5 on r_c): identification by the paper's own +-10 rule")
bm = pd.read_csv(R + "behaviour_models.csv")
rows = []
for (b, s), g in bm.groupby(["base", "shortcut"]):
    sw, st = stats.isotonic_switch(g.reliance_corrected, g.follow_rate)
    d, ident = stats.switch_identification(sw, g.reliance_corrected, 10)
    lo, hi, br = stats.switch_bracketing(sw, g.reliance_corrected, 10)
    sw_m, _ = stats.isotonic_switch(g.reliance_corrected, g.follow_rate_moved_only)
    sw_d, _ = stats.isotonic_switch(g.reliance_corrected, g.dominant_rate)
    rows.append([b, s, round(sw, 1), st, round(d, 1), ident, round(lo, 1), round(hi, 1), br, round(sw_m, 1), round(sw_d, 1)])
print(pd.DataFrame(rows, columns=["base", "shortcut", "switch", "status", "nearest", "ident", "dist_below", "dist_above",
                                  "bracketed", "switch_moved_only_label", "switch_dominant_label"]).to_string())
print("\nCSV observed_switch_rc_status values:", beh_sum.observed_switch_rc_status.unique())

h("B. follow rate (primary label) of every model with 10 < r_c <= 60, natural images")
for b in ["pet_resnet18", "cifar10_resnet18", "cifar10"]:
    g = bm[(bm.base == b) & (bm.reliance_corrected > 10) & (bm.reliance_corrected <= 60)].sort_values("reliance_corrected")
    print(b, [(round(r, 1), round(f, 2), round(u, 2)) for r, f, u in zip(g.reliance_corrected, g.follow_rate, g.undetermined_rate)])

h("C. undetermined share vs r_c (Spearman, per setting)")
from scipy.stats import spearmanr  # noqa: E402
for (b, s), g in bm.groupby(["base", "shortcut"]):
    print(f"{b:17s}{s:8s} mean undetermined {g.undetermined_rate.mean():.3f}  max {g.undetermined_rate.max():.2f}  "
          f"rho(undetermined, r_c) {spearmanr(g.undetermined_rate, g.reliance_corrected).correlation:.2f}")

h("D. paired two-level bootstrap of behavioural switch minus metric switch (shortcut vs full object)")
merged = oracle.merge(beh[KEYS + ["index", "causal"]], on=KEYS + ["index"], how="inner")
merged = merged.merge(models[KEYS + ["reliance_corrected"]], on=KEYS)
out = []
for (b, s), g in merged.groupby(["base", "shortcut"]):
    g = g.sort_values(["dose", "seed", "index"])
    mk = g[KEYS + ["reliance_corrected"]].drop_duplicates(KEYS)
    nm = len(mk)
    ni = len(g) // nm
    rc = mk.reliance_corrected.to_numpy()
    dose = mk.dose.to_numpy()
    lab = g.causal.to_numpy().reshape(nm, ni)
    P, I = stats.two_level_picks(dose, ni, REPS, SEED)
    lab_b = lab[P[:, :, None], I]
    with np.errstate(invalid="ignore"):
        fr_b = np.nanmean(lab_b, axis=2)
    beh_b = np.array([stats.isotonic_switch(rc[P[r]], fr_b[r])[0] for r in range(REPS)])
    beh_full = stats.isotonic_switch(rc, np.nanmean(lab, axis=1))[0]
    for m, higher in METRICS.items():
        a = g[f"{m}_shortcut"].to_numpy().reshape(nm, ni)
        o = g[f"{m}_object"].to_numpy().reshape(nm, ni)
        w = stats.win_scores(a, o, higher, TOL)
        pref_b = w[P[:, :, None], I].mean(axis=2)
        met_b = np.array([stats.isotonic_switch(rc[P[r]], pref_b[r])[0] for r in range(REPS)])
        met_full, status = stats.isotonic_switch(rc, w.mean(axis=1))
        # a metric that is already above 0.5 at the lowest reliance switches "at or below" the lowest model
        met_b_filled = np.where(np.isnan(met_b), rc.min(), met_b)
        diff = beh_b - met_b_filled
        ok = ~np.isnan(diff)
        lo, hi = np.percentile(diff[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
        out.append([b, s, m, round(beh_full, 1), round(met_full, 1), status,
                    round(np.nanpercentile(beh_b, 2.5), 1), round(np.nanpercentile(beh_b, 97.5), 1),
                    round(float(np.mean(np.isnan(beh_b))), 3), round(lo, 1), round(hi, 1), round(float(np.mean(diff[ok] <= 0)), 4)])
res = pd.DataFrame(out, columns=["base", "shortcut", "metric", "beh_switch", "metric_switch", "metric_status",
                                 "beh_ci_lo", "beh_ci_hi", "beh_nan_share", "diff_ci_lo", "diff_ci_hi", "P(diff<=0)"])
print(res.to_string())
print("\nNote: metric switch NaN in a replicate (already > 0.5 at the lowest reliance) is set to the lowest model r_c.")

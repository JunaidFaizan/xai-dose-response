"""Audit 2, reviewer 01: band claims, calibration gap on determined images only. Read-only."""
import glob

import numpy as np
import pandas as pd

R = "results/dose_response/"
pd.set_option("display.width", 220, "display.max_columns", 30)
bands = pd.read_csv(R + "behaviour_by_reliance.csv")
b = bands[(bands.base == "pneumonia_mnist") & (bands.shortcut == "patch") & (bands.label == "follows_shortcut_causal")
          & (bands.comparison == "shortcut vs full object") & bands.band.isin(["60-80", ">=80"])]
print("== Pneumonia patch bands above r_c 60 ==")
print(b[["metric", "band", "models", "n_pos", "n_neg", "auroc_strat"]].round(3).to_string())

models = pd.read_csv(R + "models.csv")
oracle = pd.read_csv(R + "scores_oracle.csv")
beh = pd.concat([pd.read_csv(f) for f in glob.glob(R + "per_model/*_behaviour.csv")], ignore_index=True)
KEYS = ["base", "shortcut", "dose", "seed"]
beh["causal"] = np.where(beh.pred_none == beh.conflict_class, np.nan,
                         (beh.pred_conflicting == beh.conflict_class).astype(float))
m = oracle.merge(beh[KEYS + ["index", "causal"]], on=KEYS + ["index"])
print("\n== calibration gap: all images (as in paper) vs determined images only ==")
rows = []
for (base, sc), g in m.groupby(["base", "shortcut"]):
    for met, higher in {"deletion_zero": False, "deletion_blur": False, "deletion_inpaint": False, "insertion_blur": True}.items():
        a, o = g[f"{met}_shortcut"], g[f"{met}_object"]
        w = np.where(np.isclose(a, o, atol=1e-4), 0.5, ((a > o) if higher else (a < o)).astype(float))
        gg = g.assign(w=w)
        per = gg.groupby(KEYS).agg(pref_all=("w", "mean"), follow=("causal", "mean"))
        per["pref_det"] = gg[gg.causal.notna()].groupby(KEYS).w.mean()
        rows.append([base, sc, met, round((per.pref_all - per.follow).mean(), 3), round((per.pref_det - per.follow).mean(), 3)])
print(pd.DataFrame(rows, columns=["base", "shortcut", "metric", "gap_all_images", "gap_determined_only"]).to_string())

"""Second read-only check (audit2 reviewer 03): label redundancy, S-O vs R-O, misclassified images, CSV switch CIs."""
import glob
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
D = "results/dose_response/"
KEYS = ["base", "shortcut", "dose", "seed"]
models = pd.read_csv(D + "models.csv")
beh = pd.concat([pd.read_csv(p) for p in glob.glob(D + "per_model/*_behaviour.csv")], ignore_index=True)
und = beh.pred_none == beh.conflict_class
beh["causal"] = np.where(und, np.nan, (beh.pred_conflicting == beh.conflict_class).astype(float))

print("=" * 30, "A. label-1 images: is the object alone sufficient (pred_none == label)? redundancy share")
for (b, s), g in beh.groupby(["base", "shortcut"]):
    pos = g[g.causal == 1]
    print(f"{b:18s} {s:8s} n_pos={len(pos):5d} share pred_none==label: {np.mean(pos.pred_none == pos.label):.3f}  "
          f"median p_label_none={pos.p_label_none.median():.3f}  share delta_s<0.1: {np.mean(pos.delta_s < 0.1):.3f}")

print("\n", "=" * 30, "B. headline AUROC by label (shortcut vs full object), pooled strat / within strat")
bs = pd.read_csv(D + "behaviour_summary.csv")
h = bs[bs.comparison == "shortcut vs full object"]
t = h.pivot_table(index=["base", "shortcut", "metric"], columns="label",
                  values=["auroc_pooled_strat", "median_model_auroc_strat"]).round(3)
print(t.to_string())

print("\n", "=" * 30, "C. low-dose preference: shortcut vs object (S-O) vs random vs object (R-O)")
rc = pd.read_csv(D + "responsiveness_corrected.csv")
so = rc[rc.comparison == "shortcut vs full object"].set_index(["base", "shortcut", "metric"]).pref_low
ro = rc[rc.comparison == "random vs full object"].set_index(["base", "shortcut", "metric"]).pref_low
sr = rc[rc.comparison == "shortcut vs random"].set_index(["base", "shortcut", "metric"]).pref_low
print(pd.DataFrame({"S-O": so, "R-O": ro, "S-R": sr, "S-O minus R-O": so - ro}).round(3).to_string())

print("\n", "=" * 30, "D. FPR on shortcut vs RANDOM (form-matched) among r_c<=50, vs FPR on shortcut vs full object")
srf = rc[rc.comparison == "shortcut vs random"].set_index(["base", "shortcut", "metric"])[["fpr", "n_below"]]
sof = rc[rc.comparison == "shortcut vs full object"].set_index(["base", "shortcut", "metric"])[["fpr"]]
print(srf.join(sof, rsuffix="_full").round(3).to_string())

print("\n", "=" * 30, "E. aligned images misclassified (explained label = true label), by base and dose band")
m = beh.merge(models[KEYS + ["reliance_corrected"]], on=KEYS)
m["wrong"] = m.pred_aligned != m.label
m["band"] = pd.cut(m.reliance_corrected, [-5, 20, 50, 80, 101])
print(m.groupby(["base", "shortcut", "band"], observed=True).wrong.mean().round(3).to_string())

print("\n", "=" * 30, "F. project's switch CIs (shortcut vs full object)")
print(rc[rc.comparison == "shortcut vs full object"][["base", "shortcut", "metric", "switch_iso_rc", "switch_iso_rc_ci_low",
                                                     "switch_iso_rc_ci_high", "switch_found_share", "switch_4pl_rc",
                                                     "nearest_model_distance", "switch_bracketed"]].round(2).to_string(index=False))

print("\n", "=" * 30, "G. r_c distribution of models per setting (gaps > 15 points)")
for (b, s), g in models.groupby(["base", "shortcut"]):
    x = np.sort(g.reliance_corrected.to_numpy())
    gaps = [(round(a, 1), round(c, 1)) for a, c in zip(x[:-1], x[1:]) if c - a > 15]
    print(f"{b:18s} {s:8s} n={len(x)} gaps>15: {gaps}")

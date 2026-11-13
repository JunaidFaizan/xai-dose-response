"""Read-only: size of AUC differences between shortcut and random maps, per base,
at low and high reliance. Shows how small the signal is that decides 'preference'."""
import pandas as pd

s = pd.read_csv("results/dose_response/scores_oracle.csv")
m = pd.read_csv("results/dose_response/models.csv")[["base", "shortcut", "dose", "seed", "reliance"]]
s = s.merge(m, on=["base", "shortcut", "dose", "seed"])
rows = []
for metric, sign in [("deletion_zero", 1), ("deletion_blur", 1), ("deletion_inpaint", 1), ("insertion_blur", -1)]:
    s[f"{metric}_gap_rand"] = sign * (s[f"{metric}_random"] - s[f"{metric}_shortcut"])
    s[f"{metric}_gap_obj"] = sign * (s[f"{metric}_object"] - s[f"{metric}_shortcut"])
for (b, sc), g in s.groupby(["base", "shortcut"]):
    lo = g[g.reliance <= g.reliance.quantile(0.2)]
    hi = g[g.reliance >= g.reliance.quantile(0.8)]
    for metric in ["deletion_zero", "deletion_blur", "deletion_inpaint", "insertion_blur"]:
        rows.append({"base": b, "shortcut": sc, "metric": metric,
                     "n_img": len(g),
                     "gap_vs_random_low": lo[f"{metric}_gap_rand"].median(),
                     "gap_vs_random_high": hi[f"{metric}_gap_rand"].median(),
                     "gap_vs_object_low": lo[f"{metric}_gap_obj"].median(),
                     "gap_vs_object_high": hi[f"{metric}_gap_obj"].median()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nimages per model:", s.groupby(["base", "shortcut", "dose", "seed"]).size().describe().to_dict())

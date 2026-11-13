"""Read-only audit analysis (reviewer 3). Reads existing CSVs, writes nothing outside this folder."""
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr, rankdata
from torchvision.transforms.functional import gaussian_blur

R = "results/dose_response/"
models = pd.read_csv(R + "models.csv")
pref = pd.read_csv(R + "preference.csv")
resp = pd.read_csv(R + "responsiveness.csv")
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)

print("== 1. Reliance floor vs base error (lowest dose) ==")
classes = {"mnist": 10, "pneumonia_mnist": 2, "cifar10": 10, "cifar10_resnet18": 10, "pet_resnet18": 2}
models["err_none"] = 100 - models.test_accuracy_no_shortcut
models["chance_floor"] = models.apply(lambda r: r.err_none / (classes[r.base] - 1), axis=1)
low = models.loc[models.groupby(["base", "shortcut"]).dose.transform("min") == models.dose]
print(low.groupby(["base", "shortcut"])[["reliance", "chance_floor", "test_accuracy_conflicting"]].mean().round(2))
# chance-corrected reliance: (R - floor) / (100 - floor)
models["reliance_corr"] = 100 * (models.reliance - models.chance_floor) / (100 - models.chance_floor)
print(models.groupby(["base", "shortcut"]).reliance_corr.agg(["min", "max"]).round(1))

print("\n== 2. Is preference tracking reliance or object competence (no-shortcut acc)? ==")
keys = ["base", "shortcut", "dose", "seed"]
p = pref[pref.comparison == "shortcut vs full object"].merge(
    models[keys + ["test_accuracy_no_shortcut"]], on=keys)
def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    return pearsonr(res(rx, rz), res(ry, rz))[0]
rows = []
for (b, s), g in p.groupby(["base", "shortcut"]):
    r_ra = spearmanr(g.reliance, g.test_accuracy_no_shortcut)[0]
    for m in ["deletion_zero", "deletion_blur", "deletion_inpaint", "insertion_blur"]:
        rows.append({"base": b, "shortcut": s, "metric": m,
                     "rho_rel": spearmanr(g.reliance, g[m])[0],
                     "rho_accnone": spearmanr(g.test_accuracy_no_shortcut, g[m])[0],
                     "partial_rel|acc": partial_spearman(g.reliance, g[m], g.test_accuracy_no_shortcut),
                     "partial_acc|rel": partial_spearman(g.test_accuracy_no_shortcut, g[m], g.reliance),
                     "rho(rel,acc)": r_ra})
print(pd.DataFrame(rows).round(3))

print("\n== 3. Significance (p<0.05, rho>0) per comparison/metric across the 7 settings ==")
resp["sig"] = (resp.p_value < 0.05) & (resp.responsiveness_rho > 0)
print(resp.pivot_table(index=["comparison", "metric"], values="sig", aggfunc="sum"))
print("\nNumber of settings:", resp[["base", "shortcut"]].drop_duplicates().shape[0])

print("\n== 4. Form-matched comparisons: low/high preference ==")
for c in ["shortcut vs random", "shortcut vs placebo", "placebo vs random"]:
    print(resp[resp.comparison == c][["base", "shortcut", "metric", "responsiveness_rho", "p_value",
                                      "preference_low_reliance", "preference_high_reliance"]].to_string(index=False))

print("\n== 5. Residual shortcut after blur fill (patch pixels mean value) ==")
for size, p_, k, sg in [(28, 3, 25, 4.0), (32, 3, 25, 4.0), (128, 12, 97, 16.0)]:
    x = torch.zeros(1, 1, size, size)
    x[..., 0:p_, 0:p_] = 1.0
    b = gaussian_blur(x, [k, k], [sg, sg])
    print(f"{size}px patch {p_}: blurred patch mean {b[..., 0:p_, 0:p_].mean():.3f}, "
          f"max elsewhere-corner {b[..., 0:p_, -p_:].mean():.4f}")
# texture: class pattern after blur, contrast between pattern and non-pattern pixels
rng = np.random.default_rng(123)
pat = torch.from_numpy((rng.random((28, 28)) < 0.10).astype(np.float32))[None, None] * 0.3
b = gaussian_blur(pat, [25, 25], [4.0, 4.0])
m = pat[0, 0] > 0
print(f"texture 28px: blurred value on pattern {b[0,0][m].mean():.4f} vs off {b[0,0][~m].mean():.4f}")

print("\n== 6. Ties and sample size ==")
o = pd.read_csv(R + "scores_oracle.csv")
print(o.groupby(["base", "shortcut", "dose", "seed"]).size().describe())
print("unique test indices per base/shortcut:", o.groupby(["base", "shortcut"]).index.nunique().to_dict())

print("\n== 7. Dose-reliance monotonicity: seed spread near d=0 ==")
print(models.groupby(["base", "shortcut", "dose"]).reliance.agg(["min", "max"]).query("max - min > 30"))

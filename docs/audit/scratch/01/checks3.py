"""Audit 01, part 3: why the interpolated and the logistic switch points differ on MNIST patch.
Model-level points near the crossing, and an isotonic (monotone, nonparametric) fit."""
import numpy as np
import pandas as pd

pref = pd.read_csv("results/dose_response/preference.csv")
METRICS = ["deletion_zero", "deletion_blur", "deletion_inpaint", "insertion_blur"]


def pava(y):
    """Pool-adjacent-violators: non-decreasing fit to y (already sorted by x)."""
    blocks = [[v, 1] for v in y]
    out = []
    for v, w in blocks:
        out.append([v, w])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop(); v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    return np.concatenate([[v] * w for v, w in out])


def crossing(x, f):
    for i in range(len(x) - 1):
        if (f[i] - .5) * (f[i + 1] - .5) <= 0 and f[i] != f[i + 1]:
            return x[i] + (.5 - f[i]) * (x[i + 1] - x[i]) / (f[i + 1] - f[i])
    return np.nan


for b, s in [("mnist", "patch"), ("mnist", "texture"), ("pet_resnet18", "patch"), ("pneumonia_mnist", "patch")]:
    g = pref[(pref.comparison == "shortcut vs full object") & (pref.base == b) & (pref.shortcut == s)].sort_values("reliance")
    res = []
    for m in METRICS:
        iso = pava(g[m].to_numpy())
        res.append(f"{m[:12]} isotonic {crossing(g.reliance.to_numpy(), iso):.1f}")
    print(b, s, " | ".join(res))
    if b == "mnist" and s == "patch":
        print(g[(g.reliance > 5) & (g.reliance < 80)][["dose", "seed", "reliance"] + METRICS].round(3).to_string())

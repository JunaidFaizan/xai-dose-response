"""Audit 01, part 4: is reliance confounded with object competence (accuracy without the shortcut)?
Partial Spearman of preference on reliance given accuracy without the shortcut. Read-only."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

pref = pd.read_csv("results/dose_response/preference.csv")
models = pd.read_csv("results/dose_response/models.csv")
KEYS = ["base", "shortcut", "dose", "seed"]
pref = pref.merge(models[KEYS + ["test_accuracy_no_shortcut"]], on=KEYS)
METRICS = ["deletion_zero", "deletion_blur", "deletion_inpaint", "insertion_blur"]


def partial(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    rxy, rxz, ryz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)), rxz


for comp in ["shortcut vs full object", "shortcut vs random"]:
    print("==", comp)
    for (b, s), g in pref[pref.comparison == comp].groupby(["base", "shortcut"]):
        out = []
        for m in METRICS:
            pr, rxz = partial(g.reliance, g[m], g.test_accuracy_no_shortcut)
            raw = np.corrcoef(rankdata(g.reliance), rankdata(g[m]))[0, 1]
            out.append(f"{m[:12]} raw {raw:+.2f} partial {pr:+.2f}")
        print(f"{b:16s} {s:7s} rho(reliance, acc_no_shortcut)={rxz:+.2f} | " + " | ".join(out))

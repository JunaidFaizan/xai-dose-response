"""Read-only statistical checks for audit 01 (statistics and rigour).
Reads results/dose_response/*.csv; writes nothing outside docs/audit/scratch/01."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata


def multipletests(p, method):
    """Holm and Benjamini-Hochberg adjusted p-values (no statsmodels in the venv)."""
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); adj = np.empty(n)
    if method == "holm":
        a = np.maximum.accumulate((n - np.arange(n)) * p[order])
    else:
        a = np.minimum.accumulate((n / np.arange(n, 0, -1)) * p[order][::-1])[::-1]
    adj[order] = np.minimum(a, 1.0)
    return None, adj

R = "results/dose_response/"
models = pd.read_csv(R + "models.csv")
resp = pd.read_csv(R + "responsiveness.csv")
mresp = pd.read_csv(R + "method_responsiveness.csv")
pref = pd.read_csv(R + "preference.csv")
oracle = pd.read_csv(R + "scores_oracle.csv")
methods = pd.read_csv(R + "scores_methods.csv")
METRICS = ["deletion_zero", "deletion_blur", "deletion_inpaint", "insertion_blur"]
HIGHER = {"deletion_zero": False, "deletion_blur": False, "deletion_inpaint": False, "insertion_blur": True}
CLASSES = {"mnist": 10, "cifar10": 10, "cifar10_resnet18": 10, "pneumonia_mnist": 2, "pet_resnet18": 2}
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 30)

print("=== A. chance-corrected reliance (errors that land on the shortcut class by chance) ===")
m = models.copy()
q = m.base.map(lambda b: 1.0 if CLASSES[b] == 2 else 1.0 / (CLASSES[b] - 1))
m["chance_floor"] = (100 - m.test_accuracy_no_shortcut) * q
m["excess"] = m.reliance - m.chance_floor
lvl = m.groupby(["base", "shortcut", "dose"])[["reliance", "chance_floor", "excess"]].mean().round(1)
print(lvl[lvl.index.get_level_values(0).isin(["pet_resnet18", "pneumonia_mnist"])])
print(lvl.loc[["cifar10", "cifar10_resnet18"]])

print("\n=== B. reliance coverage per setting ===")
for (b, s), g in m.groupby(["base", "shortcut"]):
    r = np.sort(g.reliance.to_numpy())
    gaps = np.diff(r); i = gaps.argmax()
    print(f"{b:18s} {s:8s} n={len(r):2d} in[35,65]={((r>=35)&(r<=65)).sum():2d} >50={(r>50).sum():2d} "
          f"largest gap {r[i]:.1f}->{r[i+1]:.1f}  min={r.min():.1f}")

print("\n=== C. multiple comparisons over all reported Spearman tests ===")
allp = pd.concat([resp.assign(kind="metric")[["kind", "comparison", "base", "shortcut", "metric", "responsiveness_rho", "p_value"]],
                  mresp.assign(kind="method", comparison="method", metric=mresp.method)[["kind", "comparison", "base", "shortcut", "metric", "responsiveness_rho", "p_value"]]])
print("tests:", len(allp), "metric tests:", (allp.kind == "metric").sum())
allp["holm"] = multipletests(allp.p_value.clip(lower=1e-300), method="holm")[1]
allp["bh"] = multipletests(allp.p_value.clip(lower=1e-300), method="fdr_bh")[1]
raw = allp.p_value < 0.05
print("significant raw:", raw.sum(), " holm:", (allp.holm < 0.05).sum(), " BH:", (allp.bh < 0.05).sum())
print("lose significance under Holm:")
print(allp[raw & (allp.holm >= 0.05)].round(3).to_string())
# p_value column in CSV is rounded to 3 dp; recompute exact p for those near threshold
print("note: p_value in responsiveness.csv is rounded to 3 decimals; recomputing exact p from preference.csv")
exact = []
for (c, b, s), g in pref.groupby(["comparison", "base", "shortcut"]):
    for mt in METRICS:
        rho, p = spearmanr(g.reliance, g[mt]); exact.append((c, b, s, mt, rho, p))
ex = pd.DataFrame(exact, columns=["comparison", "base", "shortcut", "metric", "rho", "p"])
ex["holm"] = multipletests(ex.p, method="holm")[1]; ex["bh"] = multipletests(ex.p, method="fdr_bh")[1]
print("exact metric tests: raw<.05", (ex.p < .05).sum(), "holm", (ex.holm < .05).sum(), "bh", (ex.bh < .05).sum())
print(ex[(ex.p < .05) & (ex.holm >= .05)].round(4).to_string())

print("\n=== C2. rho on dose-level means (the doses are the design points) ===")
for (b, s), g in pref[pref.comparison == "shortcut vs full object"].groupby(["base", "shortcut"]):
    lv = g.groupby("dose")[["reliance"] + METRICS].mean()
    out = []
    for mt in METRICS:
        rho, p = spearmanr(lv.reliance, lv[mt]); out.append(f"{mt[:12]} {rho:.2f} (p={p:.3f}, n={len(lv)})")
    print(b, s, " | ".join(out))

print("\n=== G. decision accuracy vs balanced decision accuracy (shortcut vs full object) ===")
rows = []
for (b, s), g in pref[pref.comparison == "shortcut vs full object"].groupby(["base", "shortcut"]):
    hi = g.reliance > 50
    for mt in METRICS:
        say = g[mt] > 0.5
        acc = (say == hi).mean()
        tpr = say[hi].mean() if hi.any() else np.nan
        tnr = (~say[~hi]).mean() if (~hi).any() else np.nan
        rows.append((b, s, mt, hi.mean(), acc, (tpr + tnr) / 2, tpr, tnr))
print(pd.DataFrame(rows, columns=["base", "shortcut", "metric", "share_hi", "dec_acc", "balanced", "TPR", "TNR"]).round(3).to_string())

print("\n=== I. tie rates ===")
for mt in METRICS:
    t = pref[f"{mt}_ties"]
    print(mt, "mean", round(t.mean(), 3), "max", round(t.max(), 3))
tt = pref.groupby(["comparison", "base", "shortcut"])[[f"{mt}_ties" for mt in METRICS]].mean().round(3)
print(tt[tt.max(axis=1) > 0.05])

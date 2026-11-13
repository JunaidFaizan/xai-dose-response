"""Audit 01, part 2: image-level uncertainty, switch-point model sensitivity,
paired metric differences, method-responsiveness CIs. Read-only."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, rankdata

R = "results/dose_response/"
models = pd.read_csv(R + "models.csv")
oracle = pd.read_csv(R + "scores_oracle.csv")
methods = pd.read_csv(R + "scores_methods.csv")
METRICS = ["deletion_zero", "deletion_blur", "deletion_inpaint", "insertion_blur"]
HIGHER = {"deletion_zero": False, "deletion_blur": False, "deletion_inpaint": False, "insertion_blur": True}
KEYS = ["base", "shortcut", "dose", "seed"]
TOL = 1e-4
rng = np.random.default_rng(1)
oracle = oracle.merge(models[KEYS + ["reliance", "test_accuracy_no_shortcut"]], on=KEYS)
CLASSES = {"mnist": 10, "cifar10": 10, "cifar10_resnet18": 10, "pneumonia_mnist": 2, "pet_resnet18": 2}


def per_image(first, second, metric):
    a, b = oracle[f"{metric}_{first}"], oracle[f"{metric}_{second}"]
    better = (a > b) if HIGHER[metric] else (a < b)
    tie = np.isclose(a, b, atol=TOL)
    return np.where(tie, 0.5, better.astype(float))


def spear(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return np.nan if rx.std() == 0 or ry.std() == 0 else np.corrcoef(rx, ry)[0, 1]


def switch(r, p):
    o = np.argsort(r); r, p = np.asarray(r)[o], np.asarray(p)[o]
    for i in range(len(r) - 1):
        if (p[i] - .5) * (p[i + 1] - .5) <= 0 and p[i] != p[i + 1]:
            return r[i] + (.5 - p[i]) * (r[i + 1] - r[i]) / (p[i + 1] - p[i])
    return np.nan


def logistic_switch(rel, wins, n):
    """Binomial logistic fit of per-model win counts on reliance; reliance at p = 0.5."""
    x = rel / 100.0
    def nll(t):
        z = t[0] + t[1] * x; pr = 1 / (1 + np.exp(-z)); pr = np.clip(pr, 1e-9, 1 - 1e-9)
        return -(wins * np.log(pr) + (n - wins) * np.log(1 - pr)).sum()
    t = minimize(nll, [0, 1], method="Nelder-Mead").x
    return -t[0] / t[1] * 100 if t[1] != 0 else np.nan, t


def two_level(sub, metric_col, reps=1000):
    """Bootstrap: seeds within dose, then images within each chosen model."""
    groups = {k: g[metric_col].to_numpy() for k, g in sub.groupby(KEYS)}
    rel = {k: g.reliance.iloc[0] for k, g in sub.groupby(KEYS)}
    by_dose = {}
    for k in groups:
        by_dose.setdefault(k[2], []).append(k)
    rhos, sws, rhos_seed, sws_seed = [], [], [], []
    for _ in range(reps):
        rr, vv, lr, lv, vv0, lv0 = [], [], [], [], [], []
        for d, ks in by_dose.items():
            pick = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            r_d, v_d, v0_d = [], [], []
            for k in pick:
                x = groups[k]
                v_d.append(x[rng.integers(0, len(x), len(x))].mean()); v0_d.append(x.mean()); r_d.append(rel[k])
            rr += r_d; vv += v_d; vv0 += v0_d
            lr.append(np.mean(r_d)); lv.append(np.mean(v_d)); lv0.append(np.mean(v0_d))
        rhos.append(spear(rr, vv)); sws.append(switch(lr, lv))
        rhos_seed.append(spear(rr, vv0)); sws_seed.append(switch(lr, lv0))
    return map(np.array, (rhos, sws, rhos_seed, sws_seed))


def ci(a):
    a = a[~np.isnan(a)]
    return (np.nan, np.nan, 0.0) if len(a) == 0 else (*np.percentile(a, [2.5, 97.5]).round(1 if a.max() > 1.5 else 3), len(a))


print("=== D. seed-only vs seed+image bootstrap (shortcut vs full object) ===")
for m in METRICS:
    oracle[f"w_{m}"] = per_image("shortcut", "object", m)
for (b, s), sub in oracle.groupby(["base", "shortcut"]):
    if b not in ("mnist", "pet_resnet18", "pneumonia_mnist"):
        continue
    for m in METRICS:
        rh, sw, rh0, sw0 = two_level(sub, f"w_{m}", reps=500)
        a, b_, n = ci(sw); a0, b0, n0 = ci(sw0)
        print(f"{b:16s} {s:7s} {m:17s} switch seed-only [{a0},{b0}] found {n0/5:.0f}% | seed+image [{a},{b_}] found {n/5:.0f}%"
              f" | rho seed-only {ci(rh0)[:2]} seed+image {ci(rh)[:2]}")

print("\n=== K. switch point: interpolation (paper) vs logistic fit on model-level wins ===")
for (b, s), sub in oracle.groupby(["base", "shortcut"]):
    g = sub.groupby(KEYS)
    rel = g.reliance.first().to_numpy()
    floor = (100 - g.test_accuracy_no_shortcut.first().to_numpy()) * (1.0 if CLASSES[b] == 2 else 1 / 9)
    rel_norm = np.clip((rel - floor) / (100 - floor) * 100, 0, 100)
    out = []
    for m in METRICS:
        wins = g[f"w_{m}"].sum().to_numpy(); n = g[f"w_{m}"].count().to_numpy()
        lv = pd.DataFrame({"d": g.dose.first().to_numpy(), "r": rel, "rn": rel_norm, "p": wins / n}).groupby("d").mean()
        sp = switch(lv.r.to_numpy(), lv.p.to_numpy()); spn = switch(lv.rn.to_numpy(), lv.p.to_numpy())
        ls, t = logistic_switch(rel, wins, n)
        out.append(f"{m[:10]} interp {sp:.1f} logit {ls:.1f} interp(norm-rel) {spn:.1f}")
    print(f"{b:16s} {s:7s} " + " | ".join(out))

print("\n=== E. lowest-dose preferences with seed+image CI (P2, P3 on Pet; CIFAR for comparison) ===")
for first, second in [("random", "object"), ("shortcut", "random"), ("shortcut", "placebo"), ("shortcut", "object")]:
    for b in ["pet_resnet18", "cifar10", "cifar10_resnet18"]:
        sub = oracle[(oracle.base == b) & (oracle.dose == oracle[oracle.base == b].dose.min())]
        row = []
        for m in METRICS:
            w = per_image(first, second, m)[sub.index] if False else None
            a, bb = sub[f"{m}_{first}"], sub[f"{m}_{second}"]
            better = (a > bb) if HIGHER[m] else (a < bb)
            v = np.where(np.isclose(a, bb, atol=TOL), .5, better.astype(float))
            seeds = sub.seed.to_numpy(); us = np.unique(seeds)
            boots = []
            for _ in range(2000):
                ps = rng.choice(us, len(us))
                vals = np.concatenate([v[seeds == q][rng.integers(0, (seeds == q).sum(), (seeds == q).sum())] for q in ps])
                boots.append(vals.mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row.append(f"{m[:10]} {v.mean():.3f} [{lo:.3f},{hi:.3f}]")
        print(f"{first} vs {second:8s} {b:16s} " + " | ".join(row))

print("\n=== E2. P2 contrast on Pet lowest dose: mean(black, blur) - mean(inpaint, insertion), random vs object ===")
sub = oracle[(oracle.base == "pet_resnet18") & (oracle.dose == -0.5)]
vals = {}
for m in METRICS:
    a, bb = sub[f"{m}_random"], sub[f"{m}_object"]
    better = (a > bb) if HIGHER[m] else (a < bb)
    vals[m] = np.where(np.isclose(a, bb, atol=TOL), .5, better.astype(float))
diff = (vals["deletion_zero"] + vals["deletion_blur"]) / 2 - (vals["deletion_inpaint"] + vals["insertion_blur"]) / 2
seeds = sub.seed.to_numpy(); us = np.unique(seeds); boots = []
for _ in range(2000):
    ps = rng.choice(us, len(us))
    boots.append(np.concatenate([diff[seeds == q][rng.integers(0, 100, 100)] for q in ps]).mean())
print("contrast", round(diff.mean(), 3), "95% CI", np.percentile(boots, [2.5, 97.5]).round(3))
for m1 in ["deletion_zero", "deletion_blur"]:
    for m2 in ["deletion_inpaint", "insertion_blur"]:
        d = vals[m1] - vals[m2]; bs = []
        for _ in range(2000):
            ps = rng.choice(us, len(us))
            bs.append(np.concatenate([d[seeds == q][rng.integers(0, 100, 100)] for q in ps]).mean())
        print(f"  {m1} - {m2}: {d.mean():.3f} CI {np.percentile(bs, [2.5, 97.5]).round(3)}")

print("\n=== F. paired difference in rho, inpaint minus black/blur (shortcut vs full object), seed bootstrap ===")
for (b, s), sub in oracle.groupby(["base", "shortcut"]):
    g = sub.groupby(KEYS)
    tab = pd.DataFrame({m: g[f"w_{m}"].mean() for m in METRICS}); tab["rel"] = g.reliance.first()
    tab = tab.reset_index()
    by = [x for _, x in tab.groupby("dose")]
    out = []
    for other in ["deletion_zero", "deletion_blur", "insertion_blur"]:
        ds = []
        for _ in range(2000):
            pk = pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in by])
            ds.append(spear(pk.rel, pk.deletion_inpaint) - spear(pk.rel, pk[other]))
        ds = np.array(ds); ds = ds[~np.isnan(ds)]
        out.append(f"inpaint-{other[:12]} {spear(tab.rel, tab.deletion_inpaint) - spear(tab.rel, tab[other]):+.3f} "
                   f"[{np.percentile(ds, 2.5):+.3f},{np.percentile(ds, 97.5):+.3f}]")
    print(f"{b:16s} {s:7s} " + " | ".join(out))

print("\n=== H. method responsiveness with seed-bootstrap CI ===")
mass = methods.groupby(KEYS + ["method"]).shortcut_mass.mean().reset_index().merge(models[KEYS + ["reliance"]], on=KEYS)
for (b, s, me), g in mass.groupby(["base", "shortcut", "method"]):
    by = [x for _, x in g.groupby("dose")]
    bs = []
    for _ in range(2000):
        pk = pd.concat([x.iloc[rng.integers(0, len(x), len(x))] for x in by])
        bs.append(spear(pk.reliance, pk.shortcut_mass))
    bs = np.array(bs); bs = bs[~np.isnan(bs)]
    if s == "texture" or me in ("gradcam",):
        print(f"{b:16s} {s:7s} {me:12s} rho {spear(g.reliance, g.shortcut_mass):+.3f} CI [{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")

print("\n=== J. seed spread of reliance at the steepest doses ===")
mm = models.groupby(["base", "shortcut", "dose"]).reliance.agg(["min", "max", "std"]).round(1)
print(mm[mm["std"] > 10])

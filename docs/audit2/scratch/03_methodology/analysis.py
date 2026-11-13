"""Read-only checks for audit2 reviewer 03 (methodology). Reads results/dose_response CSVs only.
Run: .venv\\Scripts\\python.exe docs\\audit2\\scratch\\03_methodology\\analysis.py > docs\\audit2\\scratch\\03_methodology\\analysis_out.txt
"""
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
D = "results/dose_response/"
models = pd.read_csv(D + "models.csv")
KEYS = ["base", "shortcut", "dose", "seed"]

beh = pd.concat([pd.read_csv(p) for p in glob.glob(D + "per_model/*_behaviour.csv")], ignore_index=True)
und = beh.pred_none == beh.conflict_class
beh["causal"] = np.where(und, np.nan, (beh.pred_conflicting == beh.conflict_class).astype(float))

print("=" * 30, "1. follow rate (primary label) vs r_c, per setting")
per = beh.groupby(KEYS).agg(follow=("causal", "mean"), und=("causal", lambda s: s.isna().mean()),
                            moved=("follows_shortcut", "mean"), dom=("shortcut_dominant", "mean")).reset_index()
per = per.merge(models[KEYS + ["reliance", "reliance_corrected", "test_accuracy_no_shortcut"]], on=KEYS)
for (b, s), g in per.groupby(["base", "shortcut"]):
    r = spearmanr(g.reliance_corrected, 100 * g.follow)[0]
    mae = np.mean(np.abs(g.reliance_corrected - 100 * g.follow))
    rd = spearmanr(g.reliance_corrected, 100 * g.dom)[0]
    lab = g[["causal" if False else "follow", "dom"]]
    print(f"{b:18s} {s:8s} n={len(g):3d} rho(rc, follow)={r:.3f}  MAE|rc-100*follow|={mae:5.1f} pts  "
          f"rho(rc, dominant)={rd:.3f}  mean undetermined={g.und.mean():.3f} max={g.und.max():.3f}")

print("\n", "=" * 30, "2. agreement of primary label with shortcut_dominant (determined images)")
det = beh[~beh.causal.isna()]
for (b, s), g in det.groupby(["base", "shortcut"]):
    agree = (g.causal == g.shortcut_dominant).mean()
    print(f"{b:18s} {s:8s} agreement={agree:.3f}  P(dom=1|causal=1)={g[g.causal==1].shortcut_dominant.mean():.3f}  "
          f"P(dom=1|causal=0)={g[g.causal==0].shortcut_dominant.mean():.3f}")

print("\n", "=" * 30, "3. behavioural switch on dominant label (isotonic of dominant rate on r_c) vs primary")
import sys
sys.path.insert(0, ".")
from xai import stats
for (b, s), g in per.groupby(["base", "shortcut"]):
    sw_f = stats.isotonic_switch(g.reliance_corrected, g.follow)
    sw_d = stats.isotonic_switch(g.reliance_corrected, g.dom)
    print(f"{b:18s} {s:8s} switch(follow)={sw_f[0]:.1f} {sw_f[1]}  switch(dominant)={sw_d[0]:.1f} {sw_d[1]}")

print("\n", "=" * 30, "4. binary tasks: class composition of determined images within high-reliance models")
for b in ["pet_resnet18", "pneumonia_mnist"]:
    g = beh[beh.base == b].copy()
    g = g.merge(models[KEYS + ["reliance_corrected"]], on=KEYS)
    g["band"] = pd.cut(g.reliance_corrected, [-5, 20, 50, 80, 101])
    t = g.groupby(["shortcut", "band"], observed=True).apply(
        lambda h: pd.Series({"models": h.groupby(["dose", "seed"]).ngroups,
                             "undetermined": h.causal.isna().mean(),
                             "share_determined_in_majority_class": h[~h.causal.isna()].groupby(["dose", "seed"]).label.apply(
                                 lambda l: l.value_counts(normalize=True).max()).mean()}))
    print(b); print(t.round(3))

print("\n", "=" * 30, "5. Pet: models with r_c <= 50, preference (shortcut vs full object) and near-tie share")
oracle = pd.read_csv(D + "scores_oracle.csv")
pet = oracle[oracle.base == "pet_resnet18"].merge(models[KEYS + ["reliance_corrected"]], on=KEYS)
rows = []
rng = np.random.default_rng(0)
for key, g in pet.groupby(KEYS):
    rc = g.reliance_corrected.iloc[0]
    row = dict(dose=key[2], seed=key[3], rc=round(rc, 1))
    for m, higher in [("deletion_zero", False), ("deletion_blur", False), ("deletion_inpaint", False), ("insertion_blur", True)]:
        a, o = g[f"{m}_shortcut"].to_numpy(), g[f"{m}_object"].to_numpy()
        gap = (a - o) if higher else (o - a)
        w = np.where(np.isclose(a, o, atol=1e-4), 0.5, (gap > 0).astype(float))
        boot = np.array([w[rng.integers(0, len(w), len(w))].mean() for _ in range(2000)])
        row[m] = round(w.mean(), 2)
        row[m + "_ci"] = f"[{np.quantile(boot, .025):.2f},{np.quantile(boot, .975):.2f}]"
        row[m + "_|gap|<.01"] = round(np.mean(np.abs(gap) < 0.01), 2)
    rows.append(row)
pt = pd.DataFrame(rows).sort_values("rc")
print(pt.to_string(index=False))
low = pt[pt.rc <= 50]
for m in ["deletion_zero", "deletion_blur", "deletion_inpaint", "insertion_blur"]:
    lo = low[m + "_ci"].str.strip("[]").str.split(",").str[0].astype(float)
    print(f"{m:18s} FPR(pref>0.5)={np.mean(low[m] > 0.5):.2f}  FPR(image-CI lower bound >0.5)={np.mean(lo > 0.5):.2f}  n={len(low)}")

print("\n", "=" * 30, "6. nested (per-model images) vs crossed (shared images) bootstrap, shortcut vs full object")


def pref_matrix(base, shortcut, metric, higher):
    g = oracle[(oracle.base == base) & (oracle.shortcut == shortcut)].sort_values(["dose", "seed", "index"])
    a, o = g[f"{metric}_shortcut"].to_numpy(), g[f"{metric}_object"].to_numpy()
    better = a > o if higher else a < o
    w = np.where(np.isclose(a, o, atol=1e-4), 0.5, better.astype(float))
    keys = g[KEYS].drop_duplicates()
    n_img = g.groupby(KEYS).size().iloc[0]
    W = w.reshape(len(keys), n_img)
    idx = g["index"].to_numpy().reshape(len(keys), n_img)
    assert (idx == idx[0]).all(), "images not shared across models"
    keys = keys.merge(models[KEYS + ["reliance_corrected"]], on=KEYS)
    return W, keys


def boot(W, keys, crossed, reps=2000, seed=1):
    rng = np.random.default_rng(seed)
    doses = keys.dose.to_numpy()
    rc = keys.reliance_corrected.to_numpy()
    low = np.isclose(doses, doses[np.argmin(np.abs(doses - doses.min()))]) if False else None
    n_m, n_i = W.shape
    rhos, sws, lows = [], [], []
    low_dose = keys.groupby("dose").reliance_corrected.mean().idxmin()
    for _ in range(reps):
        P = np.empty(n_m, dtype=int)
        for d in np.unique(doses):
            mem = np.flatnonzero(doses == d)
            P[mem] = mem[rng.integers(0, len(mem), len(mem))]
        if crossed:
            I = np.broadcast_to(rng.integers(0, n_i, n_i), (n_m, n_i))
        else:
            I = rng.integers(0, n_i, (n_m, n_i))
        pref = W[P[:, None], I].mean(axis=1)
        x = rc[P]
        rhos.append(spearmanr(x, pref)[0])
        sws.append(stats.isotonic_switch(x, pref)[0])
        lows.append(pref[np.isclose(doses, low_dose)].mean())
    q = lambda v: np.nanquantile(v, [0.025, 0.975])
    return q(rhos), q(sws), np.mean(~np.isnan(sws)), q(lows)


for base, sc in [("pet_resnet18", "patch"), ("cifar10", "patch"), ("cifar10_resnet18", "patch"), ("mnist", "patch"),
                 ("pneumonia_mnist", "patch")]:
    for metric, higher in [("deletion_zero", False), ("deletion_inpaint", False)]:
        W, keys = pref_matrix(base, sc, metric, higher)
        nested, crossed = boot(W, keys, False), boot(W, keys, True)
        f = lambda t: f"[{t[0]:.3f},{t[1]:.3f}] w={t[1]-t[0]:.3f}"
        print(f"{base:16s} {metric:16s} rho nested {f(nested[0])} crossed {f(crossed[0])} | "
              f"switch nested {f(nested[1])} crossed {f(crossed[1])} | pref@lowdose nested {f(nested[3])} crossed {f(crossed[3])}")

print("\n", "=" * 30, "7. within-model AUROC: which models contribute (Pet, CIFAR, MNIST patch), deletion_zero full")
bm = pd.read_csv(D + "behaviour_models.csv")
col = "auroc_strat_follows_shortcut_causal_deletion_zero_full"
for b in ["pet_resnet18", "cifar10", "cifar10_resnet18", "mnist"]:
    g = bm[(bm.base == b) & (bm.shortcut == "patch")].sort_values("reliance_corrected")
    ok = g[~g[col].isna()]
    print(f"{b:16s} models with defined within-AUROC: {len(ok)}/{len(g)}; their r_c range "
          f"{ok.reliance_corrected.min():.1f}-{ok.reliance_corrected.max():.1f}; follow-rate range {ok.follow_rate.min():.2f}-{ok.follow_rate.max():.2f}; "
          f"min(#pos,#neg) median = "
          f"{np.median([min(r * (1 - u) * 100, (1 - r) * (1 - u) * 100) for r, u in zip(ok.follow_rate, ok.undetermined_rate)]):.0f}")

print("\n", "=" * 30, "8. spot-check of paper numbers against CSVs")
rc = pd.read_csv(D + "responsiveness_corrected.csv")
full = rc[rc.comparison == "shortcut vs full object"]
print(full[["base", "shortcut", "metric", "rho_rc", "p_bh", "fpr", "n_below", "balanced_accuracy", "switch_iso_rc", "switch_status",
            "pref_low", "median_gap_low"]].round(3).to_string(index=False))
sm = rc[rc.comparison == "shortcut vs size-matched object"]
print(sm[sm.base.isin(["mnist"])][["base", "shortcut", "metric", "rho_rc", "fpr", "switch_iso_rc", "pref_low"]].round(3).to_string(index=False))
ro = rc[rc.comparison == "random vs full object"]
print(ro[ro.base.isin(["pet_resnet18", "cifar10", "cifar10_resnet18"])][["base", "metric", "pref_low", "pref_low_ci_low", "pref_low_ci_high"]].round(3).to_string(index=False))
bs = pd.read_csv(D + "behaviour_summary.csv")
h = bs[bs.headline]
print(h[["base", "shortcut", "metric", "calibration_gap", "observed_switch_rc", "observed_switch_rc_status"]].round(3).to_string(index=False))
cf = pd.read_csv(D + "confound.csv")
print(cf[["base", "shortcut", "metric", "rho_reliance_competence", "partial_rel_given_acc", "p_partial_rel_given_acc", "partial_rc_given_acc"]].round(3).to_string(index=False))
print("n significant cells (full object, p_bh<0.05 & rho>0):", int(((full.p_bh < 0.05) & (full.rho_rc > 0)).sum()), "of", len(full),
      " rho range sig:", full[(full.p_bh < 0.05)].rho_rc.min().round(3), full.rho_rc.max().round(3))

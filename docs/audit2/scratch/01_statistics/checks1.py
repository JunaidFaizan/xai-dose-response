"""Audit 2, reviewer 01 (statistics): read-only checks on results/dose_response/*.csv.
Run from the project root: .venv\\Scripts\\python.exe docs\\audit2\\scratch\\01_statistics\\checks1.py
Writes nothing outside docs/audit2/scratch/01_statistics/ (stdout is redirected there)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, beta

R = "results/dose_response/"
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 400)
models = pd.read_csv(R + "models.csv")
resp = pd.read_csv(R + "responsiveness_corrected.csv")
meth = pd.read_csv(R + "method_responsiveness_corrected.csv")
pairs = pd.read_csv(R + "metric_pairs.csv")
beh = pd.read_csv(R + "behaviour_summary.csv")
pet = pd.read_csv(R + "pet_preregistered.csv")
conf = pd.read_csv(R + "confound.csv")
pref = pd.read_csv(R + "preference.csv")

def h(t):
    print("\n" + "=" * 8, t, "=" * 8)

h("A. model counts")
print(models.groupby(["base", "shortcut", "model_set"]).size().unstack(fill_value=0))
print("total", len(models))

h("B. recompute rho_rc from preference.csv (shortcut vs full object)")
cols = {"deletion_zero": "Deletion (black)", "deletion_blur": "Deletion (blur)",
        "deletion_inpaint": "Deletion (inpaint)", "insertion_blur": "Insertion (blur)"}
p = pref[pref.comparison == "shortcut vs full object"]
maxdiff = 0
for (b, s), g in p.groupby(["base", "shortcut"]):
    for c, t in cols.items():
        rho = spearmanr(g.reliance_corrected, g[c]).correlation
        row = resp[(resp.comparison == "shortcut vs full object") & (resp.base == b) & (resp.shortcut == s) & (resp.metric == t)]
        maxdiff = max(maxdiff, abs(rho - row.rho_rc.iloc[0]))
print("max |rho recomputed - csv| =", maxdiff, "(preference.csv r_c is rounded to 4 dp)")

h("C. headline counts, shortcut vs full object")
sf = resp[resp.comparison == "shortcut vs full object"]
sig = sf[(sf.p_bh < 0.05) & (sf.rho_rc > 0)]
print("cells", len(sf), "BH-sig positive", len(sig), "rho range all", sf.rho_rc.min().round(3), sf.rho_rc.max().round(3),
      "rho range sig", sig.rho_rc.min().round(3), sig.rho_rc.max().round(3))
print(sf[sf.p_bh >= 0.05][["base", "shortcut", "metric", "rho_rc", "p_rho_rc", "p_bh", "rho_rc_ci_low", "rho_rc_ci_high"]])
print("Holm headline significant:", (sf.p_holm_headline < 0.05).sum(), "of", len(sf))

h("D. point estimate outside its own two-level percentile CI (rho_rc)")
for comp, g in resp.groupby("comparison"):
    out_hi = (g.rho_rc > g.rho_rc_ci_high).sum()
    out_lo = (g.rho_rc < g.rho_rc_ci_low).sum()
    near = ((g.rho_rc_ci_high - g.rho_rc) < 0.02).sum()
    print(f"{comp:35s} n={len(g)} est>CIhigh={out_hi} est<CIlow={out_lo} est within 0.02 of CIhigh={near}")
x = resp.assign(asym=(resp.rho_rc - resp.rho_rc_ci_low) / (resp.rho_rc_ci_high - resp.rho_rc_ci_low))
print("median position of estimate inside CI (0=low,1=high):", round(float(x.asym.median()), 3))
print(sf[["base", "shortcut", "metric", "rho_rc", "rho_rc_ci_low", "rho_rc_ci_high"]].round(3).head(28))
print("methods: est>CIhigh", (meth.rho_rc > meth.rho_rc_ci_high).sum(), "of", len(meth))
print("pet P rows est outside CI:")
print(pet[(pet.estimate > pet.ci_high) | (pet.estimate < pet.ci_low)][["prediction", "comparison", "metric_or_method", "estimate", "ci_low", "ci_high"]])

h("E. p-value vs CI disagreement (Spearman t p from models vs two-level CI)")
dis = resp[((resp.p_bh < 0.05) & (resp.rho_rc_ci_low <= 0) & (resp.rho_rc_ci_high >= 0)) |
           ((resp.p_bh >= 0.05) & ((resp.rho_rc_ci_low > 0) | (resp.rho_rc_ci_high < 0)))]
print(dis[["comparison", "base", "shortcut", "metric", "rho_rc", "p_bh", "rho_rc_ci_low", "rho_rc_ci_high"]].round(3))
dm = meth[((meth.p_bh < 0.05) & (meth.rho_rc_ci_low <= 0) & (meth.rho_rc_ci_high >= 0)) |
          ((meth.p_bh >= 0.05) & ((meth.rho_rc_ci_low > 0) | (meth.rho_rc_ci_high < 0)))]
print("methods:\n", dm[["base", "shortcut", "method", "rho_rc", "p", "p_bh", "p_holm", "rho_rc_ci_low", "rho_rc_ci_high"]].round(4))
print("methods p_bh >= 0.05:\n", meth[meth.p_bh >= 0.05][["base", "shortcut", "method", "rho_rc", "p_bh"]])
print("Grad-CAM MNIST texture:\n", meth[(meth.method == "Grad-CAM") & (meth.base == "mnist") & (meth.shortcut == "texture")][["rho_rc", "p", "p_bh", "p_holm", "mass_low", "mass_high"]])

h("F. over-detection FPR: denominators, Clopper-Pearson CIs (models as iid; ignores dose clustering), n_above")
def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi
rows = []
for _, r in sf.iterrows():
    k = round(r.fpr * r.n_below)
    lo, hi = cp(k, int(r.n_below))
    rows.append([r.base, r.shortcut, r.metric, r.fpr, int(r.n_below), int(r.n_above), lo, hi, r.tpr])
print(pd.DataFrame(rows, columns=["base", "shortcut", "metric", "fpr", "n_below", "n_above", "cp_lo", "cp_hi", "tpr"]).round(3))

h("G. FPR design dependence: FPR among r_c<=50 vs r_c<=10 vs 10<r_c<=50 (shortcut vs full object)")
p2 = p.copy()
rows = []
for (b, s), g in p2.groupby(["base", "shortcut"]):
    for c, t in cols.items():
        says = g[c] > 0.5
        rc = g.reliance_corrected
        def rate(m):
            return (says[m].mean(), int(m.sum())) if m.sum() else (np.nan, 0)
        a, na = rate(rc <= 50)
        lo, nlo = rate(rc <= 10)
        mid, nmid = rate((rc > 10) & (rc <= 50))
        rows.append([b, s, t, round(a, 3), na, round(lo, 3), nlo, round(mid, 3), nmid])
print(pd.DataFrame(rows, columns=["base", "shortcut", "metric", "fpr<=50", "n", "fpr<=10", "n", "fpr10-50", "n"]))
print("r_c distribution of models with r_c<=50 per setting:")
for (b, s), g in models.groupby(["base", "shortcut"]):
    v = np.sort(g.reliance_corrected[g.reliance_corrected <= 50].round(1).to_numpy())
    print(b, s, v)

h("H. switch identification claims, MNIST patch full object; gaps in raw reliance")
mp = resp[(resp.comparison == "shortcut vs full object") & (resp.base == "mnist") & (resp.shortcut == "patch")]
print(mp[["metric", "switch_iso_rc", "switch_iso_rc_ci_low", "switch_iso_rc_ci_high", "switch_found_share", "switch_status",
          "nearest_model_distance", "switch_distance_below", "switch_distance_above", "switch_4pl_rc"]].round(2))
for (b, s), g in models.groupby(["base", "shortcut"]):
    r = np.sort(g.reliance.to_numpy())
    r = r[(r >= 5) & (r <= 95)]
    if len(r) > 1:
        i = np.argmax(np.diff(r))
        rc = np.sort(g.reliance_corrected.to_numpy())
        j = np.argmax(np.diff(rc))
        print(f"{b:16s}{s:8s} widest raw gap within 5-95: {r[i]:.1f}->{r[i+1]:.1f}; widest r_c gap overall {rc[j]:.1f}->{rc[j+1]:.1f}")

h("I. metric pairs and inpaint claims")
print("pairs tested", len(pairs), "p_bh<0.05:", (pairs.p_bh < 0.05).sum(), "p_holm<0.05:", (pairs.p_holm < 0.05).sum())
print("min p", pairs.p.min(), "= 1/(reps+1)?", 1 / 2001)
shortcut_comps = ["shortcut vs full object", "shortcut vs size-matched object", "shortcut vs placebo", "shortcut vs random"]
sc = resp[resp.comparison.isin(shortcut_comps)]
for t in cols.values():
    g = sc[sc.metric == t]
    ok = (g.p_bh < 0.05) & (g.rho_rc > 0)
    print(f"{t:20s} positive BH-sig {ok.sum()}/{len(g)}  lowest rho among ok {g[ok].rho_rc.min():.3f}")
print("comparisons present:", resp.comparison.unique())

h("J. behaviour summary, headline rows (primary label, shortcut vs full object)")
bh = beh[beh.headline]
print(bh[["base", "shortcut", "metric", "auroc_pooled_strat", "p_holm_headline_strat", "median_model_auroc_strat",
          "median_model_auroc_strat_ci_low", "median_model_auroc_strat_ci_high", "p_holm_headline_within_strat",
          "models_with_both_labels_in_a_class", "observed_switch_rc", "calibration_gap"]].round(3))
print("within-model strat AUROC significant after headline Holm:", (bh.p_holm_headline_within_strat < 0.05).sum(), "of", len(bh))
print("label rate / undetermined per setting:")
print(bh.groupby(["base", "shortcut"])[["label_rate", "undetermined_rate"]].first().round(3))

h("K. Pet P2 contrast and P1")
print(pet[pet.prediction.isin(["P1", "P2"])][["prediction", "metric_or_method", "estimate", "ci_low", "ci_high"]].round(3))

h("L. confound table")
print(conf[["base", "shortcut", "models", "rho_reliance_competence", "metric", "partial_rc_given_acc", "p_partial_rc_given_acc",
            "partial_rel_given_acc", "p_partial_rel_given_acc"]].round(3))

h("M. tie rates by setting (shortcut vs full object)")
print(sf.groupby(["base", "shortcut"]).tie_rate.max().round(4))
print("max tie rate any comparison:", resp.groupby(["base", "shortcut"]).tie_rate.max().round(4))

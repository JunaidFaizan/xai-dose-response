"""Read-only: how well is the behavioural switch (follow rate = 0.5) supported by models near it?
Applies the project's own +/-10-point identification and bracketing rule to observed_switch_rc.
Also: per-model follow rate and image counts near the switch; images per model with both labels."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.dirname(__file__)
R = os.path.join(ROOT, "results", "dose_response")
bm = pd.read_csv(os.path.join(R, "behaviour_models.csv"))
bs = pd.read_csv(os.path.join(R, "behaviour_summary.csv"))
lab = bs[(bs["label"] == "follows_shortcut_causal") & (bs["comparison"] == "shortcut vs full object")]
lines = []
for (b, s), g in bm.groupby(["base", "shortcut"]):
    g = g.sort_values("reliance_corrected")
    row = lab[(lab["base"] == b) & (lab["shortcut"] == s)].iloc[0]
    sw = float(row["observed_switch_rc"])
    lines.append(f"[status in CSV: {row['observed_switch_rc_status']}]")
    rc = g["reliance_corrected"].values
    below = rc[rc <= sw]; above = rc[rc > sw]
    db = sw - below.max() if len(below) else np.nan
    da = above.min() - sw if len(above) else np.nan
    near = min(db, da)
    lines.append(f"{b}/{s}: observed_switch_rc={sw:.1f}; nearest below {db:.1f} pts, above {da:.1f} pts; "
                 f"identified(<=10)={near <= 10}; bracketed(both<=10)={(db <= 10) and (da <= 10)}")
    rows = g[(g.reliance_corrected > sw - 30) & (g.reliance_corrected < sw + 45)][["dose", "seed", "reliance_corrected", "follow_rate", "undetermined_rate"]]
    lines.append(rows.round(3).to_string(index=False))
txt = "\n".join(lines)
open(os.path.join(OUT, "behaviour_switch_support.txt"), "w", encoding="utf-8").write(txt)
print(bs.columns.tolist()[:8])
print(txt)

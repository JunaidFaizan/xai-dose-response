"""Read-only: reliance coverage, gaps, training time per setting, and docx text extraction.
Outputs stay in docs/audit2/scratch/05_additions/."""
import os, sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.dirname(__file__)
m = pd.read_csv(os.path.join(ROOT, "results", "dose_response", "models.csv"))
lines = []
for (b, s), g in m.groupby(["base", "shortcut"]):
    rc = np.sort(g["reliance_corrected"].values)
    gaps = np.diff(rc)
    i = int(np.argmax(gaps))
    in_mid = ((rc > 30) & (rc < 70)).sum()
    lines.append(f"{b}/{s}: n={len(g)} rc min={rc.min():.1f} max={rc.max():.1f} "
                 f"largest gap {rc[i]:.1f}->{rc[i+1]:.1f} ({gaps[i]:.1f} pts); models with 30<rc<70: {in_mid}; "
                 f"train min mean={g['training_minutes'].mean():.1f}; acc_noSC range {g['test_accuracy_no_shortcut'].min():.1f}-{g['test_accuracy_no_shortcut'].max():.1f}")
    per_dose = g.groupby("dose")["reliance_corrected"].agg(["mean", "min", "max", "count"]).round(1)
    lines.append(per_dose.to_string())
txt = "\n".join(lines)
open(os.path.join(OUT, "coverage.txt"), "w", encoding="utf-8").write(txt)
print(txt)

try:
    import docx
    d = docx.Document(os.path.join(ROOT, "paper", "paper_draft.docx"))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for r in t.rows:
            parts.append(" | ".join(c.text for c in r.cells))
    open(os.path.join(OUT, "paper_text.txt"), "w", encoding="utf-8").write("\n".join(parts))
    print("docx paragraphs", len(d.paragraphs), "tables", len(d.tables))
except Exception as e:
    print("docx failed", e)

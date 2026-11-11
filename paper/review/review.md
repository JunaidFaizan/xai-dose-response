# Review: paper\paper_draft.docx (pass 2)

Verdict: APPROVED (no blockers)

## Prior finding F1: resolved
- `build_paper.py` lines 158–164 add `MT_INS_PLACEBO` (the "shortcut vs placebo" row for mnist / texture / Insertion (blur)) and `p_text()`. Lines 1010–1018 now report blur deletion and insertion separately. All values come from the CSV rows.
- The rendered docx text of Section 5.5 matches `results\dose_response\responsiveness.csv`:
  - **Blur deletion, shortcut vs random:** ρ = 0.09, p = 0.61, preference 0.44 → 0.44. The CSV has 0.089, 0.607 and 0.443 → 0.443.
  - **Insertion, shortcut vs random:** ρ = −0.59, p < 0.001, preference 0.43 → 0.36. The CSV has −0.593, 0.0 and 0.43 → 0.363. The text correctly describes this as a significant response in the wrong direction.
  - **Insertion, shortcut vs placebo:** ρ = −0.81, p < 0.001. The CSV has −0.81 and 0.0. Its preference falls from 0.467 to 0.30, which supports "same pattern is stronger".
- **Path affected by the fix:** the Discussion paragraph "Why insertion fails on a diffuse shortcut" now says insertion "gives the texture map little credit, and as reliance grows it rates a random ordering above the texture map more often". This agrees with the CSV, and the paragraph says the explanation was not tested separately.
- **Other text on the same result:** the Abstract, Practical guidance and Conclusion still say "insertion fails on the diffuse shortcut". This fits the switch point of 98.8% and the negative ρ values, so no change is needed.

## Unchanged since pass 1
- Pass 1 checked the structure, references, method details, controls interpretation and the other numbers. The edit did not touch any of these, and none of them were reopened.
- The coder reports a successful rebuild with 8 figures, 5 tables and 26 references. The docx timestamp (17:17:52) is later than the script's (17:17:37).

## Files outside paper\
- No file outside `paper\`, `.venv`, `.git` and `__pycache__` has a modification time after 01-11-2026.
- This agrees with the coder's identical before/after `git diff` hash.

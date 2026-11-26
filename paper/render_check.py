"""Rendered check of the submission Word files (needs Microsoft Word on Windows and PyMuPDF).

Exports paper/paper_submission.docx and paper/supplementary_submission.docx (or the files given) to PDF with
Word, invisibly and read-only (the docx is never saved), renders every page to PNG and reports pages, words,
text or images outside the margins and tables wider than the text block.

    .venv/Scripts/python.exe paper/render_check.py .agents/render/after [file.docx ...]
"""
import subprocess
import sys
from pathlib import Path

import pymupdf as fitz
from docx import Document

PAPER = Path(__file__).resolve().parent
DEFAULT = [PAPER / "paper_submission.docx", PAPER / "supplementary_submission.docx"]
DPI = 90
TOL = 5.0  # points (glyph boxes of justified and right-aligned lines overshoot the margin by about 3 pt)


PS = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
try {
  $word.Visible = $false; $word.DisplayAlerts = 0
  foreach ($pair in $args) {
    $src, $pdf = $pair -split '\|'
    $d = $word.Documents.Open($src, $false, $true, $false)
    try {
      $d.Repaginate()
      Write-Output ("STATS|" + $src + "|" + $d.ComputeStatistics(2) + "|" + $d.ComputeStatistics(0))
      $d.ExportAsFixedFormat($pdf, 17)
    } finally { $d.Close(0) }
  }
} finally { $word.Quit([ref]0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
"""


def export(docs, out):
    """Word -> PDF through PowerShell COM (Open read-only, never saved). Returns {docx: (pages, words)}."""
    script = out / "_export.ps1"
    script.write_text(PS, encoding="utf8")
    pairs = [f"{p.resolve()}|{(out / (p.stem + '.pdf')).resolve()}" for p in docs]
    run = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *pairs],
                         capture_output=True, text=True)
    script.unlink()
    if run.returncode:
        raise SystemExit(run.stdout + run.stderr)
    stats = {}
    for line in run.stdout.splitlines():
        if line.startswith("STATS|"):
            _, src, pages, words = line.split("|")
            stats[Path(src)] = (int(pages), int(words))
    return {p: stats[p.resolve()] for p in docs}


def text_block(path):
    s = Document(path).sections[0]
    pt = lambda emu: emu / 12700
    return pt(s.left_margin), pt(s.page_width) - pt(s.right_margin), pt(s.top_margin), \
        pt(s.page_height) - pt(s.bottom_margin), pt(s.page_width) - pt(s.left_margin) - pt(s.right_margin)


def check(path, out):
    pdf = fitz.open(out / (path.stem + ".pdf"))
    left, right, top, bottom, width = text_block(path)
    findings = []
    for i, page in enumerate(pdf, 1):
        page.get_pixmap(dpi=DPI).save(out / f"{path.stem}_p{i:02d}.png")
        pw = page.rect.width
        # landscape pages: use the rotated text block
        l, r = (left, right) if pw < page.rect.height else (top, pw - (page.rect.height - bottom))
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
            if x0 < l - TOL or x1 > r + TOL:
                findings.append(f"p{i}: text outside margins x={x0:.0f}-{x1:.0f} (block {l:.0f}-{r:.0f}): "
                                f"{txt.strip()[:50]!r}")
        for img in page.get_image_info():
            x0, _, x1, _ = img["bbox"]
            if x0 < l - TOL or x1 > r + TOL:
                findings.append(f"p{i}: image outside margins x={x0:.0f}-{x1:.0f}")
        for tab in page.find_tables(strategy="lines").tables:
            x0, _, x1, _ = tab.bbox
            if x1 - x0 > (r - l) + TOL:
                findings.append(f"p{i}: table wider than text block ({x1 - x0:.0f} > {r - l:.0f} pt)")
        # values split over two lines (an interval or an arrow at a line end); prose can match too: inspect
        for line in page.get_text("text").splitlines():
            s = line.strip()
            if s.endswith(("\u2192", "[", "(", "incl.")) or s.count("[") > s.count("]") \
                    or s.startswith((")", "]")) or s.endswith(("(pBH", "(pHolm", "pBH", "pHolm")) \
                    or (len(s) <= 30 and s.count("(") > s.count(")")):
                findings.append(f"p{i}: split value? {s[-60:]!r}")
    return len(pdf), findings


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER.parent / ".agents" / "render" / "check"
    docs = [Path(p) for p in sys.argv[2:]] or DEFAULT
    out.mkdir(parents=True, exist_ok=True)
    stats = export(docs, out)
    for path in docs:
        pages, findings = check(path, out)
        print(f"{path.name}: Word pages {stats[path][0]}, words {stats[path][1]}, PDF pages {pages}, "
              f"margin/width findings {len(findings)}")
        for f in findings:
            print("  " + f)


if __name__ == "__main__":
    main()

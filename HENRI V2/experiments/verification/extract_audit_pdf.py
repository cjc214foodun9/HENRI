#!/usr/bin/env python3
"""Extract the audit PDF, including pages that the standard reader cannot decode.

read_file reported: NeedsOcrError on pages 3, 6-7, 10 of 18. Those are exactly the
pages most likely to hold the directive tables, so a text-only extraction would
silently drop them and I would be summarizing a document I had not read.

Strategy: PyMuPDF text extraction per page. Report per-page character counts so
the gap is VISIBLE rather than silently absent. If a page yields almost no text,
dump it as PNG so it can be viewed directly instead of guessed at.
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_PDF = os.path.join(os.path.expanduser("~"), "Downloads",
                           "Project HENRI_ Codebase Audit, Intelligence "
                           "Extrapolation, & Missing Systems Specification.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=DEFAULT_PDF,
                    help="path to the audit PDF (defaults to ~/Downloads)")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "audit_pdf_extract"))
    args = ap.parse_args()
    PDF, OUT = args.pdf, args.outdir

    import fitz  # pymupdf
    os.makedirs(OUT, exist_ok=True)
    IMG = os.path.join(OUT, "pages")
    os.makedirs(IMG, exist_ok=True)
    doc = fitz.open(PDF)
    total = []
    sparse = []
    for i, page in enumerate(doc):
        txt = page.get_text("text") or ""
        total.append(txt)
        n = len(txt.strip())
        flag = "" if n >= 400 else "  <-- SPARSE"
        print(f"page {i+1:2d}: {n:6d} chars{flag}")
        if n < 400:
            sparse.append(i + 1)
            pix = page.get_pixmap(dpi=200)
            p = os.path.join(IMG, f"page_{i+1:02d}.png")
            pix.save(p)
            print(f"         rendered -> {p}")

    body = "\n\n".join(
        f"===== PAGE {i+1} =====\n{t}" for i, t in enumerate(total))
    txt_path = os.path.join(OUT, "audit_text.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print()
    print(f"pages={len(total)} sparse_pages={sparse}")
    print(f"wrote {txt_path} ({len(body)} chars)")
    return 0

if __name__ == "__main__":
    sys.exit(main())

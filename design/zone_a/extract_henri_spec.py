#!/usr/bin/env python
"""Extract Project HENRI.pdf -- the authoritative architecture spec.

WHY THIS RUNS IN THE WORKTREE
    Keeps provenance reproducible. Raw source is never committed. Output stays
    inside the worktree so the artifact is traceable to this revision.

D70 (self-caught): inline long commands hit the bash parse wall (exit 2).
    Always write a script file, then run it with `bash <path>` or `python <path>`.

Reports per-page text density. A diagram-heavy spec yields thin text; we must
know which pages carry the architecture before reading.
"""
import hashlib
import os
import sys

SRC = r"C:/Users/chan/Downloads/Project HENRI.pdf"
OUTDIR = r"C:/Users/chan/henri-worktrees/phase1-transduction/design/zone_a/spec"
OUT = os.path.join(OUTDIR, "Project_HENRI.pdf.txt")

os.makedirs(OUTDIR, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.path.exists(SRC):
        print("MISSING_SOURCE", SRC)
        return 2

    size = os.path.getsize(SRC)
    digest = sha256(SRC)
    print(f"SOURCE {SRC}")
    print(f"  bytes  {size}")
    print(f"  sha256 {digest}")

    import fitz  # pymupdf

    doc = fitz.open(SRC)
    n_pages = doc.page_count
    print(f"  pages  {n_pages}")

    parts = []
    per_page = []
    for i, page in enumerate(doc):
        txt = page.get_text("text") or ""
        parts.append(f"\n\n===== PAGE {i + 1} =====\n{txt}")
        per_page.append((i + 1, len(txt.strip())))

    full = "".join(parts)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(full)

    total = len(full)
    print(f"\nWROTE {OUT}")
    print(f"  chars  {total}")
    print(f"  words  {len(full.split())}")

    thin = [p for p, n in per_page if n < 200]
    print(f"\n  DENSE pages (>=2500 chars): "
          f"{[p for p, n in per_page if n >= 2500]}")
    print(f"  THIN pages (<200 chars):    {len(thin)} -> {thin[:40]}")
    print("\n  per-page chars:")
    for p, n in per_page:
        bar = "#" * min(60, n // 120)
        print(f"    p{p:>3} {n:>6} {bar}")

    doc.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

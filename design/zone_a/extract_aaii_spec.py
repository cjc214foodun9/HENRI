#!/usr/bin/env python
"""Extract the five operator-supplied PDFs to text and build a MANIFEST.

WHY THIS RUNS IN THE WORKTREE, NOT THE MAIN CHECKOUT
    Session lesson (HANDOFF-2026-10-04.md sec 3.1): the ak4_corpus lived only in
    C:/Users/chan/AppData/Local/Temp and git tracked 0 of its files. A temp cleaner
    would have destroyed a PROVEN result. Extracted spec text that a receipt cites
    MUST live inside the worktree and be committed.

WHAT IT PINS
    - sha256 of each source PDF (the authoritative identity).
    - sha256 + char count + page count of each extracted text.
    - A term scan for the identifiers the operator claims: AAII, v4.3, 4.3.2.
    - A structural scan for dataset / harness / scoring language.

Run:  python -u C:/Users/chan/henri-worktrees/phase1-transduction/design/zone_a/extract_aaii_spec.py
"""
import hashlib
import json
import os
import re
import sys

SRC = r"C:/Users/chan/Downloads"
ROOT = r"C:/Users/chan/henri-worktrees/phase1-transduction"
OUT = os.path.join(ROOT, "HENRI V2", "benchmarks", "spec")

FILES = [
    "AGENT benchmarking.pdf",
    "Intelligence Evaluation Principles.pdf",
    "SCIENTIFIC REASONING Benchmarking.pdf",
    "GENERAL Benchmarking.pdf",
    "CODING Benchmarking.pdf",
]

# The identifiers the operator asserts. If these are absent, the premise is unverified.
TERMS = [
    "AAII", "aaii", "v4.3", "4.3.2", "4.3",
    "version", "VERSION",
    "dataset", "benchmark", "harness", "suite",
    "scoring", "score", "metric", "accuracy", "pass@", "pass@k",
    "split", "train", "test", "holdout", "held-out",
    "leaderboard", "baseline", "reference implementation",
    "methodology", "protocol", "standard",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text(path):
    """Return (text, pages). Prefer pymupdf; fall back to pypdf."""
    try:
        import pymupdf  # modern name
        doc = pymupdf.open(path)
        pages = [doc[i].get_text() for i in range(doc.page_count)]
        n = doc.page_count
        doc.close()
        return "\n".join(pages), n
    except Exception as exc_mu:
        try:
            import pypdf
            rd = pypdf.PdfReader(path)
            pages = [(p.extract_text() or "") for p in rd.pages]
            return "\n".join(pages), len(rd.pages)
        except Exception as exc_py:
            raise RuntimeError(
                f"both extractors failed: pymupdf={exc_mu!r} pypdf={exc_py!r}")


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    ok = True

    for fname in FILES:
        path = os.path.join(SRC, fname)
        rec = {"file": fname, "src_path": path}
        if not os.path.isfile(path):
            rec["error"] = "NOT FOUND"
            ok = False
            manifest.append(rec)
            print(f"[MISS] {fname}")
            continue

        pdf_sha = sha256_file(path)
        size = os.path.getsize(path)
        try:
            text, npages = extract_text(path)
        except Exception as exc:
            rec.update({"pdf_sha256": pdf_sha, "bytes": size,
                        "error": f"extract failed: {exc}"})
            ok = False
            manifest.append(rec)
            print(f"[FAIL] {fname}: {exc}")
            continue

        txt_name = re.sub(r"[^A-Za-z0-9._-]+", "_", fname) + ".txt"
        txt_path = os.path.join(OUT, txt_name)
        with open(txt_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)

        # Term scan with first-hit context, case-insensitive where sensible.
        hits = {}
        for term in TERMS:
            flags = 0 if any(c.isupper() for c in term) else re.IGNORECASE
            pat = re.compile(re.escape(term), flags)
            found = pat.findall(text)
            if found:
                m = pat.search(text)
                ctx = text[max(0, m.start() - 90):m.start() + 90]
                ctx = re.sub(r"\s+", " ", ctx).strip()
                hits[term] = {"count": len(found), "first_context": ctx}

        rec.update({
            "pdf_sha256": pdf_sha,
            "bytes": size,
            "pages": npages,
            "text_chars": len(text),
            "text_sha256": sha256_file(txt_path),
            "text_out": os.path.relpath(txt_path, ROOT).replace("\\", "/"),
            "term_hits": hits,
        })
        manifest.append(rec)

        key_terms = [t for t in ("AAII", "aaii", "v4.3", "4.3.2") if t in hits]
        print(f"[OK]   {fname}")
        print(f"         {size:>8} bytes  {npages:>4} pages  {len(text):>7} chars")
        print(f"         pdf_sha256 {pdf_sha[:16]}...")
        print(f"         claim terms present: {key_terms or 'NONE'}")
        print(f"         hits: " + ", ".join(f"{k}={v['count']}"
              for k, v in sorted(hits.items())))

    mpath = os.path.join(OUT, "MANIFEST.json")
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print(f"manifest -> {os.path.relpath(mpath, ROOT)}")
    print(f"extract dir -> {os.path.relpath(OUT, ROOT)}")

    # Global verdict on the operator's premise.
    joined = " ".join(
        open(os.path.join(OUT, r["text_out"].split("/")[-1]), encoding="utf-8").read()
        for r in manifest if r.get("text_out"))
    gv = {}
    for t in ("AAII", "aaii", "v4.3", "4.3.2", "version"):
        gv[t] = len(re.findall(re.escape(t), joined, re.IGNORECASE if t.islower() or t == "aaii" else 0))
    print()
    print("=== PREMISE CHECK (all five docs combined) ===")
    for t, n in gv.items():
        print(f"   {t:<8} {n}")
    print(f"   total chars {len(joined)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

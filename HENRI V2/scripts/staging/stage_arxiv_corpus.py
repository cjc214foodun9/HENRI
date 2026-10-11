#!/usr/bin/env python3
"""Stage the three directive-defining arXiv papers into the local corpus.

WHY THIS EXISTS
    The operator asked for the accompanying research paper to be committed to the local
    corpus. The three papers below are the PRIMARY SOURCES the engineering directives cite:

        2610.12349  SplitJEPA          -> directive 1 (invariant/variant block encoder)
        2510.04618  ACE                -> directive 2 (itemized playbook, delta KV)
        2609.38879  Fold2Reason        -> D_inv grounding (protein structural dataset)

WHAT LANDS IN THE CORPUS
    Per paper, the extracted TEXT and a provenance row. Not the PDF binary: the text is
    what a retrieval consumer reads, and the PDF's sha256 in the manifest makes the source
    auditable without carrying a large opaque blob in git history.

FAIL-CLOSED DESIGN
    Three guards, because a silent corpus error is worse than an absent corpus:
      1. HTTP must return a real PDF body (not an HTML error page).
      2. pymupdf must extract at least MIN_CHARS of text.
      3. Each paper must contain a REQUIRED PHRASE from its own abstract. arXiv rate
         limiting or a redirect to an error page yields a syntactically valid PDF or a
         stub; the phrase check is what proves the RIGHT paper arrived. A wrong paper that
         passed a length check would be a silent contamination of the corpus.

DETERMINISM
    No timestamp is written into a text file, so the .md bytes are reproducible from the
    same upstream PDF. The retrieval time lives only in the manifest's `retrieval` block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
# BUG CAUGHT BY RUNNING IT: `HERE.parent.parent` is "HENRI V2", NOT the repo root, so the
# first run wrote to HENRI V2/design/... -- a path git does NOT track (0 files there),
# while every evidence receipt lives in the repo-root design/zone_a/evidence/. Three
# levels up is correct: staging -> scripts -> HENRI V2 -> repo root.
REPO = HERE.parents[2]
DEFAULT_OUT = REPO / "design" / "zone_a" / "evidence" / "arxiv_corpus"


def _assert_repo_root(p: pathlib.Path) -> None:
    """Fail closed if the resolved repo root is not the tracked tree.

    A silent corpus write into an UNTRACKED directory is worse than a crash: the run
    reports success, the artifacts are invisible to git, and a later consumer finds no
    corpus. This guard is what would have caught the parents[2] bug immediately.
    """
    marker = p / "design" / "zone_a" / "evidence"
    if not marker.is_dir():
        raise RuntimeError(
            f"resolved repo root {p} has no design/zone_a/evidence/ -- refusing to write "
            f"a corpus into an untracked location. Check the HERE.parents index.")

UA = ("Mozilla/5.0 (compatible; HENRI-corpus-stager/1.0; "
      "+https://github.com/cjc214foodun9/HENRI)")
MIN_CHARS = 20000
PDF_URL = "https://arxiv.org/pdf/{aid}"

# arxiv_id, short role, title, REQUIRED PHRASE that must appear in THIS paper
PAPERS = [
    ("2610.12349", "splitjepa",
     "SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction",
     "SplitJEPA"),
    ("2510.04618", "ace",
     "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models",
     "Agentic Context Engineering"),
    ("2609.38879", "fold2reason",
     "Does Learning Protein Folding Generalize to Broader Reasoning?",
     "FoldingCorpus"),
]

CHUNK_CHARS = 4000


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_pdf(aid: str, timeout: int = 120) -> bytes:
    url = PDF_URL.format(aid=aid)
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "application/pdf"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(
            f"{aid}: response is not a PDF (first bytes {data[:16]!r}). arXiv may be "
            f"rate-limiting or redirecting to an HTML page.")
    return data


def extract_pages(pdf_bytes: bytes) -> list[str]:
    import pymupdf                                        # noqa: PLC0415
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def compose_text(pages: list[str]) -> str:
    """Page markers are kept so page attribution survives into the stored text."""
    parts = []
    for i, p in enumerate(pages, start=1):
        parts.append(f"<!-- page {i} -->\n{p.strip()}\n")
    return "\n".join(parts)


def chunk_index(text: str, pages: list[str]) -> list[dict]:
    """Fixed-window chunks over the COMPOSED text, with a page attribution.

    Page attribution is computed from the composed markers so a consumer can map any
    character offset back to a page without re-parsing the PDF.
    """
    bounds, cursor = [], 0
    for i, p in enumerate(pages, start=1):
        marker = f"<!-- page {i} -->\n"
        start = text.find(marker, cursor)
        if start < 0:
            continue
        content_start = start + len(marker)
        content_end = content_start + len(p.strip())
        bounds.append((i, content_start, content_end))
        cursor = content_end

    def page_of(off: int) -> int:
        for pg, a, b in bounds:
            if a <= off < b:
                return pg
        return bounds[-1][0] if bounds else 1

    out = []
    for n, start in enumerate(range(0, max(1, len(text)), CHUNK_CHARS)):
        seg = text[start:start + CHUNK_CHARS]
        if not seg.strip():
            continue
        out.append({
            "chunk": n,
            "char_start": start,
            "char_end": start + len(seg),
            "page_start": page_of(start),
            "page_end": page_of(max(start, start + len(seg) - 1)),
            "sha256": sha256_bytes(seg.encode("utf-8")),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="corpus directory (created if missing)")
    ap.add_argument("--only", default=None,
                    help="comma-separated arxiv ids to stage (default: all)")
    ap.add_argument("--delay", type=float, default=4.0,
                    help="seconds between arxiv requests; stay polite")
    args = ap.parse_args()

    _assert_repo_root(REPO)
    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"repo root = {REPO}")
    print(f"corpus    = {out_dir}")

    wanted = set(a.strip() for a in args.only.split(",")) if args.only else None
    rows, failures = [], []

    for aid, role, title, phrase in PAPERS:
        if wanted and aid not in wanted:
            continue
        print(f"--- {aid} ({role}) {title}")
        try:
            pdf = fetch_pdf(aid)
        except Exception as exc:                          # noqa: BLE001
            failures.append((aid, f"fetch: {type(exc).__name__}: {exc}"))
            print(f"    FAIL fetch: {exc}")
            time.sleep(args.delay)
            continue

        pages = extract_pages(pdf)
        text = compose_text(pages)
        chars = len(text)
        if chars < MIN_CHARS:
            failures.append((aid, f"extraction too small: {chars} chars < {MIN_CHARS}"))
            print(f"    FAIL extraction: {chars} chars")
            time.sleep(args.delay)
            continue

        # GUARD 3: prove we received THIS paper, not an error page or a sibling.
        if phrase not in text:
            failures.append((aid, f"required phrase {phrase!r} ABSENT -- wrong document?"))
            print(f"    FAIL required phrase {phrase!r} absent")
            time.sleep(args.delay)
            continue

        text_path = out_dir / f"{aid}.md"
        body = (f"---\n"
                f"arxiv_id: {aid}\n"
                f"role: {role}\n"
                f"title: {title}\n"
                f"pdf_sha256: {sha256_bytes(pdf)}\n"
                f"n_pages: {len(pages)}\n"
                f"required_phrase: {phrase}\n"
                f"---\n\n")
        text_path.write_text(body + text, encoding="utf-8", newline="\n")

        rows.append({
            "arxiv_id": aid,
            "role": role,
            "title": title,
            "pdf_sha256": sha256_bytes(pdf),
            "pdf_bytes": len(pdf),
            "n_pages": len(pages),
            "text_chars": chars,
            "text_sha256": sha256_bytes(text_path.read_bytes()),
            "text_path": str(text_path.relative_to(REPO)).replace("\\", "/"),
            "required_phrase": phrase,
            "n_chunks": len(chunk_index(text, pages)),
            "chunks": chunk_index(text, pages),
        })
        print(f"    OK {len(pages)} pages, {chars} chars, "
              f"{rows[-1]['n_chunks']} chunks, pdf {rows[-1]['pdf_bytes']} bytes")
        time.sleep(args.delay)

    manifest = {
        "schema": "henri.arxiv.corpus.manifest.v1",
        "evidence_class": "OBSERVED_PRIMARY_SOURCE",
        "what": ("Text of the arXiv papers the 2026-10 engineering directives cite. The PDFs "
                 "are NOT stored; each row carries the source PDF sha256 so the extraction "
                 "is auditable and the upstream document is identifiable."),
        "guards": ["body must start with %PDF",
                   f"extracted text must exceed {MIN_CHARS} chars",
                   "each paper must contain a required phrase from its own abstract"],
        "retrieval": {"url_template": PDF_URL, "user_agent": UA,
                      "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "papers": sorted(rows, key=lambda r: r["arxiv_id"]),
        "failures": [{"arxiv_id": a, "reason": r} for a, r in failures],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print("=" * 72)
    print(f"staged {len(rows)}/{len(rows) + len(failures)} papers into {out_dir}")
    for a, r in failures:
        print(f"  FAILED {a}: {r}")
    print("NOT CLAIMED: staging is not understanding. This corpus is text retrieval "
          "material. It establishes no execution and no model capability.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

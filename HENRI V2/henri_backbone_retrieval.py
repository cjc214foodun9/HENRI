"""Default-OFF retrieval augmentation for the CLASS51 frozen backbone (P3(a)).

Read-only System-3 memory layer: deterministic BM25-style lexical retrieval
over a provenance-pinned public corpus (CPython Doc/library, pinned commit),
injected into the prompt before frozen-backbone generation.

Constraints:
- Default OFF via HENRI_BACKBONE_RETRIEVAL=1.
- No learned embeddings, no wave codec (qFHRR text codec FALSIFIED as
  semantic: run20 random-ring baseline ~0.0039).
- Fail closed: missing corpus, manifest mismatch, contamination hit, or
  retrieval error -> RetrievalBlockedError (no silent fallback).
- Contamination gate: any corpus snippet sharing a 5-gram with the
  benchmark task set (prompts, solutions, docstrings, unit tests) blocks.
- No Zone C reads or writes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re

HENRI_BACKBONE_RETRIEVAL = "HENRI_BACKBONE_RETRIEVAL"
_CONTAMINATION_LINES: set[str] = set()   # code-bearing normalized lines
_CONTAMINATION_IDENT4: set[str] = set()  # identifier 4-grams (len >= 3)

# Identifiers with >= 3 chars: single letters and bare numeric literals are
# NOT contamination signals (vacuous 5-gram evidence: '1 2 3 4 5', 'a b c d e',
# '[2, 2, 2]' fired on clean CPython docs; verbatim-line total was 1 and that
# hit was a doctest OUTPUT literal coinciding with a task data literal).
# English prose words are ALSO not contamination signals: the v2 detector
# fired on IDENT4 'the end the string' (HumanEval/10 vs re.rst prose).
# v3.2 rule (Amendment A2, RATIFIED 2026-08-22): same structure as v3.1 plus
# two calibrations against vacuous hits on the full 264-item surface:
#   (a) bare 'import x' / 'from y' lines WITHOUT underscore or compound
#       structure do NOT fire (random.rst 'import random', statistics.rst
#       'import math' were verbatim-line hits);
#   (b) shingles/lines whose ONLY code signal is a single prose-common
#       keyword ('from'/'return'/'del') do NOT fire (collections.rst
#       'from the right side', re.rst 'return tuple containing all').
# Compound code structure (def/class/=/>>>/underscore identifiers, import
# inside a compound line) still fires. Corpus: itertools.rst Recipes section
# excised (sole genuine overlap: is_prime vs HE/31); new aggregate
# 2ced6f6f1afefb4d54129b9fff74d4edf08a91efe38c4b639ef8301144786f04.
_IDENT_RE = re.compile(r"[a-z_][a-z0-9_]{2,}")
# Compound code-dominant keywords: appear in English prose only rarely.
_COMPOUND_KEYWORDS = {
    "def", "class", "import", "lambda", "yield", "assert",
    "raise", "except", "finally", "while", "elif", "global", "nonlocal",
    "pass", "break", "continue",
}
# Prose-common keywords: a lone occurrence does NOT make a line/shingle
# code-like (v3.2 calibration b).
_PROSE_KEYWORDS = {"from", "return", "del"}
_CODE_MARKERS = (">>>", "=")
_IMPORT_LEAD = {"import", "from"}


def _is_code_like(tokens: list[str], line: str | None = None) -> bool:
    """True if the token list / line carries compound code-dominant structure.

    v3.2: underscore identifier, compound keyword, or code marker fires.
    A bare import/from lead with no compound structure does not (a).
    A signal set made solely of prose-common keywords does not (b).
    """
    if line is not None and tokens and tokens[0] in _IMPORT_LEAD:
        rest = tokens[1:]
        if not any("_" in t or t in _COMPOUND_KEYWORDS for t in rest):
            return False  # bare import line (v3.2 a)
    code_sig = [t for t in tokens if "_" in t or t in _COMPOUND_KEYWORDS]
    if code_sig:
        return not all(t in _PROSE_KEYWORDS for t in code_sig)
    if line is not None:
        return any(m in line for m in _CODE_MARKERS)
    return False


class RetrievalBlockedError(RuntimeError):
    """Raised when retrieval cannot run: corpus, manifest, contamination."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _contamination_lines(text: str) -> set[str]:
    """Normalized lines that carry code semantics (>=1 real identifier AND
    code-like structure: underscore token, keyword, or syntax char)."""
    out = set()
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 8 or line.startswith("#"):
            continue
        toks = _IDENT_RE.findall(line.lower())
        if not toks:
            continue
        if not _is_code_like(toks, line):
            continue
        out.add(re.sub(r"\s+", " ", line))
    return out


def _ident_shingles(text: str, n: int = 4) -> set[str]:
    """Shingles over identifiers of length >= 3; only code-like shingles
    (>=1 underscore token or Python keyword) count."""
    toks = _IDENT_RE.findall(text.lower())
    out = set()
    for i in range(max(0, len(toks) - n + 1)):
        window = toks[i:i + n]
        if _is_code_like(window):
            out.add(" ".join(window))
    return out


def add_contamination_shingles(text: str) -> int:
    """Register benchmark task text (prompt/solution/test) for the gate."""
    before = len(_CONTAMINATION_LINES) + len(_CONTAMINATION_IDENT4)
    _CONTAMINATION_LINES.update(_contamination_lines(text))
    _CONTAMINATION_IDENT4.update(_ident_shingles(text))
    return len(_CONTAMINATION_LINES) + len(_CONTAMINATION_IDENT4) - before


class BackboneRetrieval:
    """Deterministic lexical retrieval over a pinned corpus directory."""

    def __init__(self, corpus_dir: str | pathlib.Path, top_k: int = 3,
                 snippet_chars: int = 900, enabled: bool | None = None):
        if enabled is None:
            enabled = os_environ_flag()
        if not enabled:
            self.enabled = False
            self._snippets: list[dict] = []
            return
        self.enabled = True
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.corpus_dir = pathlib.Path(corpus_dir)
        self._load_corpus()

    def _load_corpus(self) -> None:
        manifest_path = self.corpus_dir / "manifest.json"
        if not manifest_path.exists():
            raise RetrievalBlockedError(f"corpus manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_id") != "henri.corpus-manifest.v1":
            raise RetrievalBlockedError("corpus manifest schema mismatch")
        entries = manifest["files"]
        self._documents: list[dict] = []
        self._doc_freqs: dict[str, int] = {}
        for entry in entries:
            path = self.corpus_dir / entry["file"]
            if not path.exists():
                raise RetrievalBlockedError(f"corpus file missing: {entry['file']}")
            raw = path.read_bytes()
            # canonical LF bytes
            lf = raw.replace(b"\r\n", b"\n")
            if hashlib.sha256(lf).hexdigest() != entry["sha256"]:
                raise RetrievalBlockedError(f"corpus file hash mismatch: {entry['file']}")
            text = lf.decode("utf-8", errors="replace")
            toks = _tokenize(text)
            if not toks:
                raise RetrievalBlockedError(f"empty corpus file: {entry['file']}")
            self._documents.append({"entry": entry, "text": text, "tokens": toks})
            for t in set(toks):
                self._doc_freqs[t] = self._doc_freqs.get(t, 0) + 1
        if not self._documents:
            raise RetrievalBlockedError("empty corpus")

    # -- contamination gate -------------------------------------------------
    def scan_contamination(self) -> list[str]:
        """Return list of contaminated snippet ids (empty = clean).

        Signal: verbatim-line overlap containing >=1 real identifier, OR
        identifier-4-gram overlap (tokens len >= 3). Bare literals and
        single-letter variable sequences are not leakage signals (vacuous
        detector evidence documented in module docstring).
        """
        hits = []
        for doc in self._documents:
            text = doc["text"]
            if _contamination_lines(text) & _CONTAMINATION_LINES:
                hits.append(doc["entry"]["file"])
                continue
            if _ident_shingles(text) & _CONTAMINATION_IDENT4:
                hits.append(doc["entry"]["file"])
        return hits

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, query: str) -> list[dict]:
        """Top-k snippets with provenance tags. Raises on contamination."""
        contaminated = self.scan_contamination()
        if contaminated:
            raise RetrievalBlockedError(
                f"contamination gate fired: {contaminated[:5]}")
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        n_docs = len(self._documents)
        scored = []
        for doc in self._documents:
            doc_len = len(doc["tokens"])
            tf = {}
            for t in q_toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t, count in tf.items():
                if t not in self._doc_freqs:
                    continue
                idf = math.log(1.0 + n_docs / self._doc_freqs[t])
                denom = count + 1.5 * (0.25 + 0.75 * doc_len / 4000.0) + 0.75
                score += idf * count * 2.2 / denom
            if score > 0.0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        snippets = []
        for score, doc in scored[:self.top_k]:
            text = doc["text"][:self.snippet_chars]
            snippets.append({
                "source_file": doc["entry"]["file"],
                "module": doc["entry"]["module"],
                "sha256": doc["entry"]["sha256"],
                "bm25_score": round(score, 4),
                "snippet": text,
            })
        return snippets

    def build_prompt(self, task_prompt: str) -> tuple[str, dict]:
        """Return (augmented_prompt, telemetry). Fail closed."""
        if not self.enabled:
            raise RetrievalBlockedError("retrieval disabled")
        snippets = self.retrieve(task_prompt)
        if not snippets:
            raise RetrievalBlockedError("retrieval returned no snippets")
        block_parts = [
            "### Retrieved reference material (provenance-tagged; not task data)",
        ]
        for s in snippets:
            block_parts.append(
                f"[source: {s['source_file']} sha256:{s['sha256'][:12]} "
                f"score:{s['bm25_score']}]\n{s['snippet']}")
        retrieval_block = "\n\n".join(block_parts)
        augmented = (
            "You are solving a programming task. Reference material from the "
            "Python standard-library documentation is provided below; use it "
            "only if it helps. Do not copy it verbatim.\n\n"
            f"{retrieval_block}\n\n"
            "### Task\n"
            f"{task_prompt}"
        )
        telemetry = {
            "retrieval_engaged": True,
            "snippets": [s["source_file"] for s in snippets],
            "prompt_sha256": hashlib.sha256(augmented.encode()).hexdigest(),
            "retrieval_block_bytes": len(retrieval_block),
        }
        return augmented, telemetry


def os_environ_flag() -> bool:
    return os.environ.get(HENRI_BACKBONE_RETRIEVAL, "0") == "1"


def build_arm_a_prompt(task_prompt: str) -> tuple[str, dict]:
    """Arm A: frozen-backbone prompt (no retrieval block)."""
    prompt = (
        "You are solving a programming task. Provide only the final Python "
        "solution.\n\n### Task\n" + task_prompt
    )
    return prompt, {
        "retrieval_engaged": False,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }

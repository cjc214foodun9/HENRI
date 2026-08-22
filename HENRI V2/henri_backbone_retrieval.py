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
_CONTAMINATION_SET = set()  # populated by add_contamination_shingles


class RetrievalBlockedError(RuntimeError):
    """Raised when retrieval cannot run: corpus, manifest, contamination."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _shingles(text: str, n: int = 5) -> set[str]:
    toks = _tokenize(text)
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def add_contamination_shingles(text: str) -> int:
    """Register benchmark task text (prompt/solution/test) for the gate."""
    before = len(_CONTAMINATION_SET)
    _CONTAMINATION_SET.update(_shingles(text))
    return len(_CONTAMINATION_SET) - before


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
        """Return list of contaminated snippet ids (empty = clean)."""
        hits = []
        for doc in self._documents:
            for shingle in _shingles(doc["text"]):
                if shingle in _CONTAMINATION_SET:
                    hits.append(doc["entry"]["file"])
                    break
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

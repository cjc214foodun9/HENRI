"""G6 v6 — count-aware separable text codec + directed bigram-chain decode.

STATUS (2026-09-07): carrier/g6-count-aware-egress. Prereg sealed pre-result:
experiments/verification/g6_v6_count_aware_prereg.md.

Motivating measured failure (v5, g5_v5_kill_receipt.json): exact sequence
recovery 0.025 < 0.70 -> V1 kill. Root causes (direct source reads):
  (1) features_of() dedups -> multiplicity destroyed before accumulation;
  (2) per-row L2 normalization erases cross-row magnitudes;
  (3) v5 beam forbids repeats (`if w in path: continue`).
v6 is a NEW codec: multiplicity-preserving accumulation, count-bearing payload
(NOT row-unit — documented contract deviation), count estimate from per-cell
magnitude median, directed bigram chain with capacity counts.

Evidence class: REPRESENTATION_CAPACITY_EVIDENCE (self-consistent round-trip).
NOT a task score; no DB writes; default-OFF; zero trainable parameters.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from zone_c_world_knowledge_codec import (
    NUM_BLOCKS, BLOCK_DIM, WAVE_DIM, PROJ_DIM, MAX_WORDS,
    _proj_accum, _l2, features_of, tokenize,
)
from g5_separable_codec import v5_feature_cells

WAVE_EXPAND = 16
MIN_MATCHED = 16       # exact support gate
EDGE_SUPPORT_MIN = 12  # directed bigram edge evidence (of 16)
COUNT_CLAMP_MAX = 64


@dataclass
class V6DecodeResult:
    text: str | None = None
    confidence: float = 0.0
    status: str = "ABSTAIN_LOW_CONF"
    words: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    path: list[str] = field(default_factory=list)
    n_matched: int = 0


class CountAwareCodec:
    """Deterministic, zero-trainable, multiplicity-preserving codec."""

    def __init__(self, vocab: list[str], beam: int = 8,
                 min_matched: int = MIN_MATCHED,
                 edge_support_min: int = EDGE_SUPPORT_MIN):
        self.vocab = sorted(set(vocab))
        self.beam = int(beam)
        self.min_matched = int(min_matched)
        self.edge_support_min = int(edge_support_min)
        uni = [f"w:{w}" for w in self.vocab]
        self._uni_cells = (np.stack([v5_feature_cells(f) for f in uni])
                           if uni else np.zeros((0, WAVE_EXPAND), dtype=np.int64))
        self._edge_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    # -- encoding -------------------------------------------------------------
    def encode(self, text: str) -> tuple[bytes, np.ndarray]:
        words = tokenize(text)[:MAX_WORDS]
        feats = [f"w:{w}" for w in words] + [
            f"b:{words[i]} {words[i + 1]}" for i in range(len(words) - 1)
        ]
        acc = np.zeros(WAVE_DIM, dtype=np.float32)
        for f in feats:
            cells = v5_feature_cells(f)
            signs = np.where((cells % 2) == 0, 1.0, -1.0).astype(np.float32)
            np.add.at(acc, cells, signs)
        rows = acc.reshape(NUM_BLOCKS, BLOCK_DIM)
        proj = _l2(_proj_accum(features_of(text, ngram_max=2), PROJ_DIM)).astype(np.float32)
        return rows.astype(np.float32).tobytes(), proj

    # -- decoding -------------------------------------------------------------
    def _sig(self, flat: np.ndarray, cells: np.ndarray) -> tuple[int, np.ndarray]:
        c = cells.astype(np.int64)
        v = flat[c]
        s = np.where((c % 2) == 0, 1.0, -1.0).astype(np.float32)
        return int(np.sum((v * s) > 0.0)), v * s

    def _count_of(self, est: np.ndarray) -> int:
        pos = est[est > 0.0]
        if pos.size == 0:
            return 0
        m = float(np.median(pos))
        if m < 1.0:
            return 0 if m < 0.5 else 1
        return int(min(COUNT_CLAMP_MAX, max(1, int(round(m)))))

    def _edge(self, flat: np.ndarray, a: str, b: str) -> float:
        k = (a, b)
        if k not in self._edge_cache:
            self._edge_cache[k] = (v5_feature_cells(f"b:{a} {b}"), None)
        cells = self._edge_cache[k][0]
        matched, _ = self._sig(flat, cells)
        return matched / WAVE_EXPAND

    def decode(self, wave_rows: np.ndarray) -> V6DecodeResult:
        flat = np.asarray(wave_rows, dtype=np.float32).ravel()
        if flat.size != WAVE_DIM or self._uni_cells.shape[0] == 0:
            return V6DecodeResult(status="ABSTAIN_INVALID_INPUT")
        admitted: list[str] = []
        counts: dict[str, int] = {}
        for i in range(self._uni_cells.shape[0]):
            matched, est = self._sig(flat, self._uni_cells[i])
            if matched >= self.min_matched:
                w = self.vocab[i]
                cnt = self._count_of(est)
                counts[w] = cnt if cnt >= 1 else 1
                admitted.append(w)
        if not admitted:
            return V6DecodeResult(status="ABSTAIN_LOW_CONF")
        total = sum(counts.values())
        if total <= 0:
            return V6DecodeResult(status="ABSTAIN_NO_ORDER", words=admitted,
                                  counts=counts, n_matched=len(admitted))
        thresh = self.edge_support_min / WAVE_EXPAND
        paths: list[tuple[list[str], float]] = [([w], 0.0) for w in admitted]
        for _ in range(total - 1):
            nxt: list[tuple[list[str], float]] = []
            for path, sc in paths:
                used: dict[str, int] = {}
                for w in path:
                    used[w] = used.get(w, 0) + 1
                for w in admitted:
                    if used.get(w, 0) >= counts[w]:
                        continue
                    ev = self._edge(flat, path[-1], w)
                    s = ev if ev >= thresh else 0.0
                    nxt.append((path + [w], sc + s))
            if not nxt:
                break
            nxt.sort(key=lambda t: (-t[1], t[0]))
            paths = nxt[: self.beam]
        best = None
        if paths:
            best = max(paths, key=lambda t: (len(t[0]) / total, t[1]))
        if best is None or len(best[0]) != total:
            return V6DecodeResult(status="ABSTAIN_NO_ORDER", words=admitted,
                                  counts=counts, n_matched=len(admitted))
        path, sc = best
        conf = round(sc / max(1, total - 1), 4) if total > 1 else 1.0
        return V6DecodeResult(text=" ".join(path), confidence=conf, status="OK",
                              words=admitted, counts=counts, path=path,
                              n_matched=len(admitted))


def build_vocab(texts: list[str], max_words: int = 100000) -> list[str]:
    seen: set[str] = set()
    for t in texts:
        for w in tokenize(t):
            seen.add(w)
            if len(seen) >= max_words:
                return sorted(seen)
    return sorted(seen)

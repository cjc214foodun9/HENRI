"""G7 v7 — high-order count-aware codec with EXACT walk enumeration decode.

STATUS (2026-09-08): carrier/g7-highorder-egress. Prereg sealed pre-result:
experiments/verification/g7_v7_highorder_walk_prereg.md @ 7224b64.

Inherits from v6 (measured, sealed g6_v6_kill_receipt.json @ 8a03f6b):
  * admission 16/16 support gate (V2 P/R 1.0/1.0 PASS)
  * per-cell median-magnitude counts (V6 multiplicity-exact 1.0 PASS)
  * zero fabrication + OOV abstain (V3/V5 PASS)
  * order information IS in the wave (true adjacent bigram evidence 60/60)
v7 ADDS: trigram (2nd-order) + 4-gram (3rd-order) features, multiplicity-
preserving, and replaces the beam with EXACT walk enumeration over the
strong-edge multigraph: edges must pass 2nd-order trigram evidence >= 12/16,
then 3rd-order 4-gram evidence >= 12/16. Deterministic lexicographic order.
OK iff EXACTLY ONE walk consumes every admitted word exactly its count.

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
MIN_MATCHED = 16        # exact support gate (v6-verified exact)
EDGE_SUPPORT_MIN = 12   # strong n-gram evidence out of 16
COUNT_CLAMP_MAX = 64
NODE_BUDGET = 5_000_000  # hard per-window DFS node budget (honest cap)


@dataclass
class V7DecodeResult:
    text: str | None = None
    confidence: float = 0.0
    status: str = "ABSTAIN_LOW_CONF"
    words: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    path: list[str] = field(default_factory=list)
    n_matched: int = 0
    n_walks: int = 0
    cap_hit: bool = False


class HighOrderCodec:
    """Deterministic, zero-trainable, order-k count-aware codec (k = 1..3)."""

    def __init__(self, vocab: list[str], min_matched: int = MIN_MATCHED,
                 edge_support_min: int = EDGE_SUPPORT_MIN,
                 node_budget: int = NODE_BUDGET):
        self.vocab = sorted(set(vocab))
        self.min_matched = int(min_matched)
        self.edge_support_min = int(edge_support_min)
        self.thresh = self.edge_support_min / WAVE_EXPAND
        self.node_budget = int(node_budget)
        uni = [f"w:{w}" for w in self.vocab]
        self._uni_cells = (np.stack([v5_feature_cells(f) for f in uni])
                           if uni else np.zeros((0, WAVE_EXPAND), dtype=np.int64))
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- encoding -------------------------------------------------------------
    def encode(self, text: str) -> tuple[bytes, np.ndarray]:
        words = tokenize(text)[:MAX_WORDS]
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
        feats += [f"t:{words[i]} {words[i + 1]} {words[i + 2]}"
                  for i in range(len(words) - 2)]
        feats += [f"q:{words[i]} {words[i + 1]} {words[i + 2]} {words[i + 3]}"
                  for i in range(len(words) - 3)]
        acc = np.zeros(WAVE_DIM, dtype=np.float32)
        for f in feats:
            cells = v5_feature_cells(f)
            signs = np.where((cells % 2) == 0, 1.0, -1.0).astype(np.float32)
            np.add.at(acc, cells, signs)
        rows = acc.reshape(NUM_BLOCKS, BLOCK_DIM)
        proj = _l2(_proj_accum(features_of(text, ngram_max=2), PROJ_DIM)).astype(np.float32)
        return rows.astype(np.float32).tobytes(), proj

    # -- evidence -------------------------------------------------------------
    def _cells(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = (v5_feature_cells(key), np.array([]))
        return self._cache[key][0]

    def evidence(self, flat: np.ndarray, key: str) -> float:
        cells = self._cells(key)
        c = cells.astype(np.int64)
        v = flat[c]
        s = np.where((c % 2) == 0, 1.0, -1.0).astype(np.float32)
        return float(np.sum((v * s) > 0.0)) / WAVE_EXPAND

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

    # -- decode ---------------------------------------------------------------
    def decode(self, wave_rows: np.ndarray) -> V7DecodeResult:
        flat = np.asarray(wave_rows, dtype=np.float32).ravel()
        if flat.size != WAVE_DIM or self._uni_cells.shape[0] == 0:
            return V7DecodeResult(status="ABSTAIN_INVALID_INPUT")
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
            return V7DecodeResult(status="ABSTAIN_LOW_CONF")
        total = sum(counts.values())
        if total <= 0:
            return V7DecodeResult(status="ABSTAIN_NO_ORDER", words=admitted,
                                  counts=counts, n_matched=len(admitted))
        starts = sorted(w for w in counts if counts[w] > 0)

        def enumerate_walks(max_order: int) -> tuple[list[list[str]], bool]:
            """Exact DFS under constraints up to max_order
            (2=bigram+trigram, 3=+4-gram). Returns (found_paths, cap_hit)
            where found_paths is capped at 2 paths (uniqueness judgement only
            needs 0/1/many)."""
            found: list[list[str]] = []
            nodes = [0]

            def dfs(path: list[str], used: dict[str, int]) -> None:
                nodes[0] += 1
                if nodes[0] > self.node_budget:
                    return
                if len(path) == total:
                    found.append(list(path))
                    return
                for w in starts:
                    if used.get(w, 0) >= counts[w]:
                        continue
                    if path:
                        if self.evidence(flat, f"b:{path[-1]} {w}") < self.thresh:
                            continue
                        if max_order >= 2 and len(path) >= 2 and \
                           self.evidence(flat, f"t:{path[-2]} {path[-1]} {w}") < self.thresh:
                            continue
                        if max_order >= 3 and len(path) >= 3 and \
                           self.evidence(flat, f"q:{path[-3]} {path[-2]} {path[-1]} {w}") < self.thresh:
                            continue
                    used[w] = used.get(w, 0) + 1
                    path.append(w)
                    dfs(path, used)
                    path.pop()
                    used[w] -= 1
                    if nodes[0] > self.node_budget or len(found) >= 2:
                        return

            dfs([], {})
            return found, nodes[0] > self.node_budget

        # Prereg: constraints of INCREASING order. 2nd-order first; escalate to
        # 3rd-order only if ambiguity remains.
        found, cap_hit = enumerate_walks(max_order=2)
        if cap_hit or not found:
            return V7DecodeResult(status="ABSTAIN_NO_ORDER", words=admitted,
                                  counts=counts, n_matched=len(admitted),
                                  n_walks=len(found), cap_hit=cap_hit)
        order_used = 2
        if len(found) > 1:
            found3, cap3 = enumerate_walks(max_order=3)
            if cap3 or not found3:
                return V7DecodeResult(status="ABSTAIN_NO_ORDER", words=admitted,
                                      counts=counts, n_matched=len(admitted),
                                      n_walks=len(found3), cap_hit=cap3)
            found, cap_hit, order_used = found3, False, 3
        if len(found) != 1:
            return V7DecodeResult(status="AMBIGUOUS", words=admitted,
                                  counts=counts, n_matched=len(admitted),
                                  n_walks=len(found), cap_hit=False)
        path = found[0]
        return V7DecodeResult(text=" ".join(path), confidence=1.0, status="OK",
                              words=admitted, counts=counts, path=path,
                              n_matched=len(admitted), n_walks=1, cap_hit=False)


def build_vocab(texts: list[str], max_words: int = 100000) -> list[str]:
    seen: set[str] = set()
    for t in texts:
        for w in tokenize(t):
            seen.add(w)
            if len(seen) >= max_words:
                return sorted(seen)
    return sorted(seen)

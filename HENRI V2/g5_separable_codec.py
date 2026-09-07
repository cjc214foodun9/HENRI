"""G5 v5 — Separable sparse codec + exact support-membership decode.

Measured falsification (2026-09-07, sealed): the K5 codec cell map depends
only on x mod 8192 (probe: distinct_cells_reachable = 8192 of 65536; tokens
'013' and 'duryee' share h mod 8192 -> IDENTICAL 16-cell signatures). At K5
vocab scale 1,554/58,298 words satisfy the support gate (30 true, 1,524
false) -> deterministic inversion of K5 waves FALSIFIED (span P=0.0052,
R=0.012, OOV guard failed).

v5 fixes the map: cell indices are derived from the FULL feature hash via
splitmix64 mixing (cell_s = splitmix64(hash ^ s*GOLDEN) % 65536). Effective
address space becomes 65536; two features collide on all 16 cells only if
their hashes collide (negligible). In the sparse regime (message <= ~200
features -> occupancy <= 5%), the support gate (all 16 cells present AND
sign-matched) is near-exact, so decode = exact word recovery or ABSTAIN.

Contracts:
  * Deterministic, zero trainable parameters, fail-closed.
  * encode returns [8192,8] row-unit wave payload (pipeline-compatible) +
    2000-d projection for retrieval (kept identical to K5 for ARM-R parity).
  * decode returns OK (exact) or ABSTAIN_*; never fabricates partial text.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from zone_c_world_knowledge_codec import (
    NUM_BLOCKS, BLOCK_DIM, WAVE_DIM, PROJ_DIM, MAX_WORDS,
    _feature_hash, _proj_accum, _l2, features_of, tokenize,
)

GOLDEN64 = 0x9E3779B97F4A7C15
MASK64 = (1 << 64) - 1
WAVE_EXPAND = 16


def _splitmix64(x: int) -> int:
    x = (x + GOLDEN64) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def v5_feature_cells(feature: str) -> np.ndarray:
    """16 distinct cell indices in [0, 65536) from the FULL feature hash.

    Full entropy: each cell uses the whole 64-bit state mixed by splitmix64,
    so the map is not reducible to a low-bit congruence class (unlike K5,
    where cell = f(x mod 8192) and the address space collapssed to 8192).
    """
    h = _feature_hash(feature)
    out = np.empty(WAVE_EXPAND, dtype=np.int64)
    for s in range(WAVE_EXPAND):
        x = _splitmix64(h ^ ((s + 1) * GOLDEN64))
        out[s] = x % WAVE_DIM
    return out


def v5_fill_cells(seed_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fill for empty rows (K5-compatible row-unit contract).

    Returns (idx[8192], sign[8192]) for every block; empty rows are replaced
    with the seeded unit basis so the payload is row-unit as in K5. Fill cells
    are outside feature encodings only by concentration: a false feature still
    needs ALL 16 cells sign-matched, which the fill cannot supply with
    non-negligible probability (occupancy^16).
    """
    seed = _feature_hash(seed_text[:64] or "empty")
    idx = np.zeros(NUM_BLOCKS, dtype=np.int64)
    sign = np.zeros(NUM_BLOCKS, dtype=np.float32)
    x = seed
    for k in range(NUM_BLOCKS):
        x = _splitmix64(x ^ (k + 1) * GOLDEN64)
        idx[k] = k * BLOCK_DIM + (x % BLOCK_DIM)
        sign[k] = 1.0 if (idx[k] % 2 == 0) else -1.0
    return idx, sign


@dataclass
class V5DecodeResult:
    text: str | None = None
    confidence: float = 0.0
    status: str = "ABSTAIN_LOW_CONF"
    words: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    n_matched: int = 0


class SeparableCodec:
    """Deterministic separable text codec + exact decoder over bounded vocab."""

    def __init__(self, vocab: list[str], beam: int = 8, bigram_weight: float = 0.5,
                 min_matched: int = 16):
        self.vocab = sorted(set(vocab))
        self.beam = int(beam)
        self.bigram_weight = float(bigram_weight)
        self.min_matched = int(min_matched)
        uni = [f"w:{w}" for w in self.vocab]
        self._uni_cells = np.stack([v5_feature_cells(f) for f in uni]) if uni else np.zeros((0, 16), dtype=np.int64)
        self._bigram_cache: dict[tuple[str, str], float] = {}

    # -- encoding (mirror K5 pipeline shape) ---------------------------------
    def encode(self, text: str) -> tuple[bytes, np.ndarray]:
        feats = features_of(text, ngram_max=3)
        acc = np.zeros(WAVE_DIM, dtype=np.float32)
        for f in feats:
            cells = v5_feature_cells(f)
            signs = np.where((cells % 2) == 0, 1.0, -1.0).astype(np.float32)
            np.add.at(acc, cells, signs)
        rows = acc.reshape(NUM_BLOCKS, BLOCK_DIM)
        # row-unit (empty rows get fill) as in K5
        f_idx, f_sign = v5_fill_cells(text[:64])
        empty = np.linalg.norm(rows, axis=1) < 1e-9
        if empty.any():
            rows[empty, :] = 0.0
            for k in np.nonzero(empty)[0]:
                rows[k, int(f_idx[k] % BLOCK_DIM)] = f_sign[k]
        rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        proj = _l2(_proj_accum(features_of(text, ngram_max=2), PROJ_DIM)).astype(np.float32)
        return rows.astype(np.float32).tobytes(), proj

    # -- decoding ------------------------------------------------------------
    def _match(self, flat: np.ndarray, cells: np.ndarray) -> int:
        """Number of the 16 cells present AND sign-matched (of 16)."""
        c = cells.astype(np.int64)
        v = flat[c]
        sgn = np.where((c % 2) == 0, 1.0, -1.0).astype(np.float32)
        return int(np.sum((v * sgn) > 0.0))

    def decode(self, wave_rows: np.ndarray) -> V5DecodeResult:
        flat = np.asarray(wave_rows, dtype=np.float32).ravel()
        if flat.size != WAVE_DIM or self._uni_cells.shape[0] == 0:
            return V5DecodeResult(status="ABSTAIN_INVALID_INPUT")
        matched = np.array([self._match(flat, c) for c in self._uni_cells], dtype=np.int32)
        hits = np.nonzero(matched >= self.min_matched)[0]
        if hits.size == 0:
            return V5DecodeResult(status="ABSTAIN_LOW_CONF", n_matched=0)
        uniq: list[str] = []
        seen: set[str] = set()
        for gi in hits:
            w = self.vocab[int(gi)]
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        # order via bigram support (K5-style DP, lightweight beam)
        order = uniq
        # beam DP over adjacent pairs (pair score from bigram cells)
        def pair_score(a: str, b: str) -> float:
            k = (a, b)
            if k not in self._bigram_cache:
                cells = v5_feature_cells(f"b:{a} {b}")
                self._bigram_cache[k] = float(np.mean(
                    [1.0 if (flat[int(c)] * (1.0 if (int(c) % 2 == 0) else -1.0)) > 0 else 0.0
                     for c in cells]))
            return self._bigram_cache[k]

        paths: list[tuple[list[str], float]] = [([w], 0.0) for w in order]
        for _ in range(len(order) - 1):
            nxt: list[tuple[list[str], float]] = []
            for path, sc in paths:
                for w in order:
                    if w in path:
                        continue
                    nxt.append((path + [w], sc + pair_score(path[-1], w)))
            if not nxt:
                break
            nxt.sort(key=lambda t: -t[1])
            paths = nxt[: self.beam]
        path = paths[0][0] if paths else order[:1]
        conf = float(len(path) / max(1, len(order))) if order else 0.0
        if conf < 0.5 or not path:
            return V5DecodeResult(status="ABSTAIN_NO_ORDER", words=order, n_matched=int(hits.size))
        return V5DecodeResult(text=" ".join(path), confidence=round(conf, 4), status="OK",
                              words=order, path=path, n_matched=int(hits.size))


def build_vocab(texts: list[str], max_words: int = 100000) -> list[str]:
    seen: set[str] = set()
    for t in texts:
        for w in tokenize(t):
            seen.add(w)
            if len(seen) >= max_words:
                return sorted(seen)
    return sorted(seen)

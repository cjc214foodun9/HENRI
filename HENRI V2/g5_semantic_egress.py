"""G5 — Deterministic Sparse Code Inversion (DSCI) semantic egress.

FALSIFIED AT SCALE (2026-09-07, sealed): the K5 codec cell map depends only
on x mod 8192 (distinct_cells_reachable = 8192 of 65536; tokens '013' and
'duryee' share h mod 8192 -> identical 16-cell signatures), so at K5 vocab
scale (58,298 words) 1,554/58,298 words satisfy the support gate for one
40-word window (30 true + 1,524 false; span P=0.0052, R=0.012). Deterministic
inversion of K5 waves is impossible. This module is retained as the measured
fail artifact; the successor is g5_separable_codec.py (v5, full-entropy
cells). Do not use DSCI for production egress.

Inverts the K5 compositional codec (CompositionalTextCodec v4) wave into
bounded-vocabulary text using matching pursuit over feature codebooks plus
n-gram evidence DP for word order. Zero trainable parameters, zero
pretrained-LM dependency, fail-closed ABSTAIN.

Status vocabulary: OK | ABSTAIN_LOW_CONF | ABSTAIN_NO_ORDER | ABSTAIN_INVALID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from zone_c_world_knowledge_codec import (
    NUM_BLOCKS,
    BLOCK_DIM,
    WAVE_DIM,
    MAX_WORDS,
    _feature_hash,
    LCG_MUL,
    LCG_ADD,
    tokenize,
)

_UNIT = 0.25  # 1/sqrt(16): unit-norm code for a 16-cell feature vector


def feature_code(feature: str) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the codec cell expansion for one feature.

    Returns (idx[16], sign[16]) identical to the intra-feature expansion in
    ``_wave_accum`` (LCG chain, block = x % NUM_BLOCKS, dim = (x >> 8) % 8,
    sign = parity of the final linear cell index).
    """
    x = _feature_hash(feature)
    idx = np.empty(16, dtype=np.int64)
    for s in range(16):
        x = (x * LCG_MUL + LCG_ADD) & 0xFFFFFFFF
        block = x % NUM_BLOCKS
        dim8 = (x >> 8) % BLOCK_DIM
        idx[s] = block * BLOCK_DIM + dim8
    signs = np.where((idx % 2) == 0, 1.0, -1.0).astype(np.float32)
    return idx, signs


@dataclass
class DecodeResult:
    text: Optional[str] = None
    confidence: float = 0.0
    status: str = "ABSTAIN_LOW_CONF"
    words: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    n_pursued: int = 0


class DSCIEngine:
    """Deterministic sparse-code inversion over a bounded vocabulary.

    Parameters
    ----------
    vocab : list[str]  bounded word vocabulary (e.g. K5 corpus words).
    max_decode_iter : int  matching-pursuit cap (default 64; a hard cap
        prevents dense chunks from being over-explained by false positives).
    beam : int  order-DP beam width.
    bigram_weight : float  weight of bigram evidence vs unigram evidence.
    tau_accept : float  minimum unigram correlation to accept a pursuit hit.
    """

    def __init__(
        self,
        vocab: list[str],
        max_decode_iter: int = 64,
        beam: int = 8,
        bigram_weight: float = 0.5,
        tau_accept: float = 0.05,
    ):
        self.vocab = sorted(set(vocab))
        self.max_decode_iter = int(max_decode_iter)
        self.beam = int(beam)
        self.bigram_weight = float(bigram_weight)
        self.tau_accept = float(tau_accept)
        feats = [f"w:{w}" for w in self.vocab]
        self._uni_idx, self._uni_sign = self._codebook(feats)

    @staticmethod
    def _codebook(features: list[str]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.zeros((len(features), 16), dtype=np.int64)
        sign = np.zeros((len(features), 16), dtype=np.float32)
        for i, f in enumerate(features):
            idx[i], sign[i] = feature_code(f)
        return idx, sign

    @staticmethod
    def _pair_score(w1: str, w2: str, flat: np.ndarray) -> float:
        idx, sign = feature_code(f"b:{w1} {w2}")
        return float(flat[idx] @ sign) * _UNIT

    def decode(self, wave_rows: np.ndarray) -> DecodeResult:
        flat = np.asarray(wave_rows, dtype=np.float32).ravel()
        if flat.size != WAVE_DIM or self._uni_idx.shape[0] == 0:
            return DecodeResult(status="ABSTAIN_INVALID_INPUT")

        # v2 (corrected): block-address code inversion requires EXACT support
        # membership. For sparse text the codec wave has exactly one nonzero
        # cell per block (fill occupies empty rows; feature cells replace them),
        # so a present feature's 16 cells are ALL in the wave support with
        # sign match (rs = 4.0). Fill-seeded cells can alias PART of an
        # in-vocab code (13-14/16 cells -> conf 0.84 on an OOV string), so a
        # score threshold alone fabricates. Gate: every cell of the candidate
        # code must be nonzero in the wave AND sign-match (16/16 membership).
        support = np.nonzero(flat)[0]
        support_set = set(int(i) for i in support)
        chosen: list[tuple[int, float]] = []
        for gi in range(self._uni_idx.shape[0]):
            cells = self._uni_idx[gi]
            if all(int(c) in support_set for c in cells):
                rs = float((flat[cells] * self._uni_sign[gi]).sum() * _UNIT)
                if rs >= 3.999:
                    chosen.append((int(gi), rs))

        if not chosen:
            return DecodeResult(status="ABSTAIN_LOW_CONF", n_pursued=0)

        cand = [self.vocab[i] for i, _ in chosen]
        uniq = list(dict.fromkeys(cand))[:64]
        uni_score: dict[str, float] = {}
        for (i, s), w in zip(chosen, cand):
            uni_score[w] = max(uni_score.get(w, 0.0), s)

        # 2) Order recovery: beam DP over bigram code support.
        pair_cache: dict[tuple[str, str], float] = {}
        beam_paths: list[tuple[list[str], float, str]] = [
            ([w], uni_score.get(w, 0.0), w) for w in uniq
        ]
        for _ in range(len(uniq) - 1):
            nxt: list[tuple[list[str], float, str]] = []
            for path, sc, last in beam_paths:
                for w in uniq:
                    if w in path:
                        continue
                    k = (last, w)
                    if k not in pair_cache:
                        pair_cache[k] = self._pair_score(last, w, flat)
                    ns = sc + uni_score.get(w, 0.0) + self.bigram_weight * pair_cache[k]
                    nxt.append((path + [w], ns, w))
            if not nxt:
                break
            nxt.sort(key=lambda t: -t[1])
            beam_paths = nxt[: self.beam]

        if not beam_paths:
            return DecodeResult(status="ABSTAIN_NO_ORDER", words=uniq, n_pursued=len(chosen))

        path, sc, _ = beam_paths[0]
        n = max(1, len(path))
        # v2 confidence: mean matched-cell fraction over the emitted path.
        # Max per-word score is 4.0 (16/16 cells); rs values are on that scale.
        conf = float(np.clip(np.mean([uni_score.get(w, 0.0) for w in path]) / 4.0, 0.0, 1.0))
        if conf < 0.5:
            return DecodeResult(status="ABSTAIN_NO_ORDER", words=uniq, path=path, n_pursued=len(chosen))
        return DecodeResult(
            text=" ".join(path),
            confidence=round(conf, 4),
            status="OK",
            words=uniq,
            path=path,
            n_pursued=len(chosen),
        )


def build_vocab(texts: list[str], max_words: int = 100000) -> list[str]:
    """Bounded vocabulary from raw text (lowercased codec tokenization)."""
    seen: set[str] = set()
    for t in texts:
        for w in tokenize(t):
            seen.add(w)
            if len(seen) >= max_words:
                return sorted(seen)
    return sorted(seen)

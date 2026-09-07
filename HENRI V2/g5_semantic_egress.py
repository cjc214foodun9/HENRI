"""G5 — Deterministic Sparse Code Inversion (DSCI) semantic egress.

Inverts the K5 compositional codec (CompositionalTextCodec v4) wave into
bounded-vocabulary text using matching pursuit over feature codebooks plus
n-gram evidence DP for word order. Zero trainable parameters, zero
pretrained-LM dependency, fail-closed ABSTAIN.

Evidence lineage (measured 2026-09-05, sealed #e0099722):
  ARM-R retrieval P@1=1.0; ARM-H Hopfield exact 0.9255@10k, edit1=0;
  ARM-U unbinder token collapse (16 waves -> token 29674), i.e.
  SEMANTIC_CAPACITY_BLOCKED on the trained egress path. This module is the
  deterministic inversion candidate for the K5 codec (encode-only today).

Codec math (zone_c_world_knowledge_codec): each feature f (w:/b:/t: n-gram)
maps to 16 signed cells via an LCG over _feature_hash(f); the wave is the
signed sum of feature codes, rows L2-normalized per 8-dim block. DSCI
recovers features by unit-code correlation (matching pursuit), then orders
words by bigram code support (beam DP).

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

        # 1) Matching pursuit: recover unigram feature codes.
        residual = flat.copy()
        initial_norm = float(np.linalg.norm(residual)) or 1.0
        chosen: list[tuple[int, float]] = []
        active = np.ones(self._uni_idx.shape[0], dtype=bool)
        for _ in range(self.max_decode_iter):
            ai = np.nonzero(active)[0]
            if ai.size == 0:
                break
            vals = residual[self._uni_idx[ai]] * self._uni_sign[ai]
            rs = vals.sum(axis=1) * _UNIT
            best_local = int(np.argmax(rs))
            if float(rs[best_local]) < self.tau_accept:
                break
            gi = int(ai[best_local])
            chosen.append((gi, float(rs[best_local])))
            u = np.zeros(WAVE_DIM, dtype=np.float32)
            u[self._uni_idx[gi]] = self._uni_sign[gi] * _UNIT
            residual -= float(rs[best_local]) * u
            active[gi] = False

        if not chosen:
            return DecodeResult(status="ABSTAIN_LOW_CONF", n_pursued=0)

        # Fail-closed coverage gate: if the pursued features explain less than
        # 50% of the wave energy, ABSTAIN rather than emit a partial guess.
        coverage = 1.0 - float(np.linalg.norm(residual)) / initial_norm
        if coverage < 0.5:
            return DecodeResult(status="ABSTAIN_LOW_CONF", n_pursued=len(chosen))

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
        conf = float(np.clip((sc / n - self.tau_accept) / (1.0 - self.tau_accept), 0.0, 1.0))
        if conf < 0.1:
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

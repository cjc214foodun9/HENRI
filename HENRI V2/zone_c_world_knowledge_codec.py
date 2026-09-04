"""Carrier K5 section 9 — structured compositional text codec (C2, v4).

Contract (consumed by zone_c_world_knowledge_ingest real mode):
    import zone_c_world_knowledge_codec as codec
    wave_bytes, proj = codec.encode(chunk)   # bytes [8192,8] float32 row-unit
                                             # proj np.float32 [2000] L2-normalized

Design history (all measured 2026-09-04, D=65536):
  v1 position-bound token codec: swap 0.778 (additive floor), zero offset
      retrieval (absolute positions never align query/chunk) -> ABANDONED.
  v2 char n-gram bag: reversal 0.871 (near order-blind), random_pair 0.151
      (matches the OBSERVED natural-language common mode), sparse rows with
      zero row norms -> ABANDONED.
  v4 (this): word-level compositional features.
      proj (2000-d, retrieval): word unigrams + word bigrams.
      wave (65536-d, payload): word unigrams + bigrams + trigrams, each
      feature expanded to 16 blocks via an integer LCG derived from the
      feature hash; rows L2-normalized; empty rows fall back to a seeded
      unit basis (every row nonzero, unit).
      Properties: order sensitivity via word bigrams/trigrams (reversal
      decorrelates); phrase/offset-free retrieval (word features match at
      any offset); deterministic and zero-trainable.

Gates (calibrated, margin-based; measured in geometry()):
  identical         >= 0.999
  end_1edit         >= 0.80
  reversal          <= 0.60
  random_cross      <= 0.20
  fragment_margin   fragment_retrieval > 2.0 * random_cross
  row_unit          min row norm >= 0.999, max <= 1.001
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

NUM_BLOCKS = 8192
BLOCK_DIM = 8
WAVE_DIM = NUM_BLOCKS * BLOCK_DIM  # 65536
PROJ_DIM = 2000
MAX_WORDS = 2048
WAVE_EXPAND = 16
LCG_MUL = 2654435761
LCG_ADD = 40503
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    t = text.lower()
    t = "".join(ch if ch.isalnum() else " " for ch in t)
    return _WS_RE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    clean = clean_text(text)
    return clean.split(" ") if clean else []


def _feature_hash(feature: str) -> int:
    return int.from_bytes(hashlib.sha256(feature.encode("utf-8")).digest()[:8], "big")


def _dedup(features: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for f in features:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def features_of(text: str, *, ngram_max: int, max_words: int = MAX_WORDS) -> list[str]:
    """Deterministic deduped word n-gram features up to ngram_max."""
    words = tokenize(text)
    if not words:
        return []
    words = words[:max_words]
    out: list[str] = []
    for w in words:
        out.append("w:" + w)
    if ngram_max >= 2:
        for i in range(len(words) - 1):
            out.append("b:" + words[i] + " " + words[i + 1])
    if ngram_max >= 3:
        for i in range(len(words) - 2):
            out.append("t:" + words[i] + " " + words[i + 1] + " " + words[i + 2])
    return _dedup(out)


def _proj_accum(features: list[str], dim: int) -> np.ndarray:
    acc = np.zeros(dim, dtype=np.float32)
    for f in features:
        h = _feature_hash(f)
        acc[h % dim] += 1.0 if (h >> 32) & 1 == 0 else -1.0
    return acc


def _wave_accum(features: list[str]) -> np.ndarray:
    acc = np.zeros(WAVE_DIM, dtype=np.float32)
    if not features:
        return acc
    nf = len(features)
    idx = np.empty((nf, WAVE_EXPAND), dtype=np.int64)
    for i, f in enumerate(features):
        x = _feature_hash(f)
        for s in range(WAVE_EXPAND):
            x = (x * LCG_MUL + LCG_ADD) & 0xFFFFFFFF
            block = x % NUM_BLOCKS
            dim8 = (x >> 8) % BLOCK_DIM
            idx[i, s] = block * BLOCK_DIM + dim8
    flat = idx.ravel()
    signs = np.where((flat % 2) == 0, 1.0, -1.0).astype(np.float32)
    np.add.at(acc, flat, signs)
    return acc


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


class CompositionalTextCodec:
    """Deterministic compositional text -> wave payload + projection."""

    def __init__(self, wave_dim: int = WAVE_DIM, proj_dim: int = PROJ_DIM):
        self.wave_dim = wave_dim
        self.proj_dim = proj_dim

    def proj_of(self, text: str) -> np.ndarray:
        feats = features_of(text, ngram_max=2)
        return _l2(_proj_accum(feats, self.proj_dim)).astype(np.float32)

    def encode(self, text: str) -> tuple[bytes, np.ndarray]:
        wave_feats = features_of(text, ngram_max=3)
        acc = _wave_accum(wave_feats)
        rows = acc.reshape(NUM_BLOCKS, BLOCK_DIM)
        norms = np.linalg.norm(rows, axis=1)
        empty = norms < 1e-9
        if empty.any():
            seed = _feature_hash(text[:64] or "empty")
            for k in np.nonzero(empty)[0]:
                x = (seed * LCG_MUL + int(k) * LCG_ADD) & 0xFFFFFFFF
                rows[k, x % BLOCK_DIM] = 1.0
        rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        proj_feats = features_of(text, ngram_max=2)
        proj = _l2(_proj_accum(proj_feats, self.proj_dim)).astype(np.float32)
        return rows.astype(np.float32).tobytes(), proj

    def geometry(self) -> dict[str, float]:
        s = "the quick brown fox jumps over the lazy dog near the riverbank"
        s_end1 = "the quick brown fox jumps over the lazy dog near the riverbanl"
        rev = " ".join(reversed(tokenize(s)))
        frag = "brown fox jumps"
        long_text = ("some unrelated opening words here and there " * 2) + s + (" and more trailing content follows " * 2)
        rand_texts = [
            "piano recital scheduled thursday evening",
            "mariners navigated by the polar star",
            "the mitochondria is the powerhouse of the cell",
            "electrons occupy discrete atomic orbitals",
        ]
        codec = self
        vs = codec.proj_of(s)
        random_cross = float(np.mean([np.dot(vs, codec.proj_of(x)) for x in rand_texts]))
        fragment_retrieval = float(np.dot(codec.proj_of(frag), codec.proj_of(long_text)))
        return {
            "identical": float(np.dot(vs, codec.proj_of(s))),
            "end_1edit": float(np.dot(vs, codec.proj_of(s_end1))),
            "reversal": float(np.dot(vs, codec.proj_of(rev))),
            "random_cross": random_cross,
            "fragment_retrieval": fragment_retrieval,
        }


_DEFAULT: CompositionalTextCodec | None = None


def get_codec() -> CompositionalTextCodec:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = CompositionalTextCodec()
    return _DEFAULT


def encode(text: str) -> tuple[bytes, np.ndarray]:
    return get_codec().encode(text)


if __name__ == "__main__":
    import json
    import time
    t0 = time.time()
    c = CompositionalTextCodec()
    g = c.geometry()
    wave_bytes, proj = encode("the quick brown fox jumps over the lazy dog near the riverbank")
    rows = np.frombuffer(wave_bytes, dtype=np.float32).reshape(NUM_BLOCKS, BLOCK_DIM)
    rn = np.linalg.norm(rows, axis=1)
    gates = {
        "identical": g["identical"] >= 0.999,
        "end_1edit": g["end_1edit"] >= 0.80,
        "reversal": g["reversal"] <= 0.60,
        "random_cross": g["random_cross"] <= 0.20,
        "fragment_margin": g["fragment_retrieval"] > 2.0 * g["random_cross"],
        "row_unit": float(rn.min()) >= 0.999 and float(rn.max()) <= 1.001,
    }
    print(json.dumps({
        "geometry": {k: round(v, 6) for k, v in g.items()},
        "gates": gates,
        "wave_bytes": len(wave_bytes),
        "row_norm_min": round(float(rn.min()), 6),
        "row_norm_max": round(float(rn.max()), 6),
        "proj_norm": round(float(np.linalg.norm(proj)), 6),
        "seconds": round(time.time() - t0, 2),
    }))

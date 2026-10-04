"""HENRI typed egress: wave -> strongly typed decision manifold.

Blueprint HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3, sec 2.4 item 3:
    "Restrict outputs to strongly typed decision manifolds (Boolean flags,
     validated spatial coordinates, discrete primitive tool IDs) rather than
     32,000 unconstrained text tokens."

MEASURED BASIS (all local CPU, $0, held-out templates, controls at chance)
    readout_head_to_head.py, V=64, ONE corpus, FOUR readouts:
        CENT  (training-free nearest class mean)  0.9141   <- 58.5x chance
        RIDGE (closed-form linear optimum)        0.8984
        GRAD  (Adam)                              0.4062   <- what Path A used
        HOP   (Hopfield snap, beta=26.10)         0.0312
      -> OPTIMIZATION_DEFICIT. The egress deficit was TRAINING, not the codec
         and not the readout family. So this head is deliberately TRAINING-FREE.
    typed_manifold_egress_test.py, training-free centroid:
        tool 32-way 0.9502 | arg 16-way 0.9951 | flat 512-way 0.8857
    ingredient_count_law_probe.py, manifold fixed at 512:
        513 ingredients 0.5146 -> 72 ingredients 0.9287

WHY TRAINING-FREE IS THE RIGHT DEFAULT
    Gradient training on few templates per class memorizes (train accuracy 1.0000)
    and fails to generalize (0.4062). A class mean has no such capacity and
    generalizes (0.9141). Few-shot egress at small K wants the closed form.

WHAT THIS MODULE DOES NOT DO
    It does not claim wave->text over 32,000 tokens. It emits typed fields. That
    is the point: the decision manifold is small by construction, and the capacity
    sweep shows the decodable region is V <= 128.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

# The codec lives one level up (HENRI V2/). Resolve without assuming cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_H_CODEC = None


def _codec_mod():
    global _H_CODEC
    if _H_CODEC is None:
        import zone_c_world_knowledge_codec as C  # noqa: N813
        _H_CODEC = C
    return _H_CODEC


def typed_vocab(names) -> dict:
    """Bijective, reproducible name -> id map. USE THIS for a closed manifold.

    D20 SELF-CAUGHT DEFECT (by reasoning, before the smoke ran). The first draft
    exposed `content_id(text, n) = int(sha256(text)[:4]) % n` and used it for a
    32-token manifold. That is a hash-and-mod, so it is NOT injective: with 32
    tokens into 32 buckets the birthday bound gives ~50% chance of at least one
    collision, and two distinct words would share a label. The head would then be
    scored against an inconsistent target, and nothing in the output would say so.

    Ranking by sha256 gives a BIJECTION over the supplied names and is still
    reproducible across processes (unlike python hash(), which is salted).

    Deterministic tie-break on the name itself, so two names whose digests are
    equal still get distinct, stable ids.
    """
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("typed_vocab: duplicate names supplied")
    return {t: i for i, t in enumerate(
        sorted(names, key=lambda t: (hashlib.sha256(t.encode()).digest(), t)))}


def content_id(text: str, n: int) -> int:
    """Open-vocabulary hash id. NOT injective -- collisions possible.

    Kept for streaming/open vocabularies where the name set is unknown up front.
    For a CLOSED typed manifold use typed_vocab(), which is bijective. Callers who
    need injectivity must check it: len({content_id(t, n) for t in names}) == n.
    """
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % n


def collision_count(names, n: int) -> int:
    """How many names collide under content_id for this manifold width.

    Exposed so a caller can ASSERT injectivity instead of assuming it:
        assert collision_count(names, n) == 0
    """
    return len(names) - len({content_id(t, n) for t in names})


def wave_features(rows: np.ndarray) -> np.ndarray:
    """[NB, BD] float32 codec rows -> [DC, 2] real/imag features, L2-normalized.

    features() preserves the codec's exact float order, so a plain reshape is the
    true block layout. Established after two self-caught rotation defects (D13,D14).
    """
    C = _codec_mod()
    nb, bd = int(C.NUM_BLOCKS), int(C.BLOCK_DIM)
    a = np.ascontiguousarray(rows, dtype="<f4")
    z = a.reshape(nb, bd // 2, 2).view(np.complex64).reshape(nb * bd // 2)
    f = np.stack([z.real, z.imag], axis=-1).astype(np.float32)
    n = float(np.linalg.norm(f))
    return f / max(n, 1e-12)


class TypedEgressHead:
    """Nearest-class-mean egress over several typed fields. Training-free.

    fit()  accumulates one mean feature vector per (field, id) from REAL corpus
           waves. snapshot() snaps a wave to the nearest id in each field and
           reports the margin to the runner-up. No optimizer, no learning rate,
           no early stopping, no random init -- so there is nothing to overfit.
    """

    def __init__(self, fields: dict):
        if not fields:
            raise ValueError("fields must be a non-empty {name: n_classes} mapping")
        for name, n in fields.items():
            if not isinstance(n, int) or n < 2:
                raise ValueError(f"field {name!r}: n_classes must be an int >= 2, got {n!r}")
        self.fields = dict(fields)
        self._sum = {f: None for f in self.fields}
        self._cnt = {f: np.zeros(self.fields[f], dtype=np.int64) for f in self.fields}
        self._means = {f: None for f in self.fields}
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, samples):
        """samples: iterable of (wave_rows, {field: id}).

        wave_rows is the codec's [NB, BD] array (from encode_egress).
        """
        C = _codec_mod()
        dc = int(C.NUM_BLOCKS) * int(C.BLOCK_DIM) // 2
        for f in self.fields:
            self._sum[f] = np.zeros((self.fields[f], dc, 2), dtype=np.float64)
        for rows, labels in samples:
            f = wave_features(rows)
            for name in self.fields:
                if name not in labels:
                    continue
                k = int(labels[name])
                if not (0 <= k < self.fields[name]):
                    raise ValueError(f"{name} label {k} out of range 0..{self.fields[name]-1}")
                self._sum[name][k] += f
                self._cnt[name][k] += 1
        for name in self.fields:
            miss = np.nonzero(self._cnt[name] == 0)[0]
            if miss.size:
                raise ValueError(
                    f"field {name!r}: {miss.size} class(es) never seen: {miss[:8].tolist()}"
                )
            m = self._sum[name] / self._cnt[name][:, None, None]
            flat = m.reshape(self.fields[name], -1)
            flat /= np.clip(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12, None)
            self._means[name] = flat.astype(np.float32)
        self._fitted = True
        return self

    # --------------------------------------------------------------- query
    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("TypedEgressHead is not fitted; call fit() first")

    def snap(self, rows):
        """One wave -> {field: (id, margin)}. margin = top1 sim - top2 sim."""
        self._require_fitted()
        f = wave_features(rows).reshape(-1)
        out = {}
        for name, m in self._means.items():
            sim = m @ f
            order = np.argsort(sim)[::-1]
            top = int(order[0])
            margin = float(sim[order[0]] - sim[order[1]]) if sim.size > 1 else float("inf")
            out[name] = (top, margin)
        return out

    def score(self, samples):
        """Accuracy per field plus joint accuracy. samples align with fit()."""
        self._require_fitted()
        hit = {f: 0 for f in self.fields}
        joint = 0
        n = 0
        for rows, labels in samples:
            pred = self.snap(rows)
            ok = True
            for name in self.fields:
                if name not in labels:
                    continue
                correct = pred[name][0] == int(labels[name])
                hit[name] += int(correct)
                ok = ok and correct
            joint += int(ok)
            n += 1
        n = max(n, 1)
        return {"per_field": {k: v / n for k, v in hit.items()},
                "joint": joint / n, "n": n}


__all__ = ["TypedEgressHead", "wave_features", "typed_vocab", "content_id",
           "collision_count"]

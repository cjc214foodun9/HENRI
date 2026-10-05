"""Benchmark staging: expose HENRI as an agent for the AAII harness runner.

Purpose
    "Staged for benchmarking" means the model is runnable through the existing
    verified harness, with gates pre-registered. It does NOT mean a score.

Contract
    The runner's Agent protocol:
        fit(items: list[Item]) -> None
        predict(item: Item) -> int          # a class index in [0, K)

    HENRI's ingress produces a 65,536-d wave. Classification reads the wave with
    the training-free centroid snap: assign each item to the nearest stored
    wave centroid. That is exactly the "training-free nearest-class centroid
    evaluation" the document specifies for the Action head (doc p25, G-D5).

Disclosed status
    spec_status stays CLASSIFIED-AS-NOT-A-BENCHMARK-RESULT. No AAII dataset is
    pinned for this adapter, so the harness verdict refuses to emit a score.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                     # noqa: E402
from henri_core.system import TriModelSystem                # noqa: E402
from henri_core.tokenizer import ByteBPE                    # noqa: E402

ADAPTER_ID = "henri.tri-model.wave-centroid.v1"

# A fixed calibration corpus. The tokenizer is fitted here, so the vocabulary is
# reproducible from source alone; no external corpus or asset is required.
CALIBRATION = [
    "entity acts on object in context",
    "retrieval transfer result on the ladder",
    "the axiom holds the boundary",
    "measuring the energy saddle",
    "phase interference clears the port",
    "the crystal stores the engram",
    "gradient descent reaches the attractor",
    "the swarm finds the lower energy",
    "memory decays with the relaxation time",
    "the veto rejects the inverted phase",
    "the resonator couples the modes",
    "the spectral projector filters drift",
]


def build_encoder(small: bool = True, vocab: int = 512, seed: int = 20261004):
    """Return (encode_fn, system, tokenizer). encode_fn(text) -> [D] complex."""
    tok = ByteBPE().train(CALIBRATION, vocab_size=vocab)
    system = TriModelSystem(vocab=tok.vocab_size, small=small, seed=seed)
    system.eval()

    def encode_fn(text: str) -> torch.Tensor:
        with torch.no_grad():
            return system.wave_of(text, tok)

    return encode_fn, system, tok


class HenriWaveCentroidAgent:
    """Nearest-wave-centroid classifier over HENRI's 65,536-d ingress.

    Args:
        K:           number of classes in the task
        encode_fn:   text -> unit-norm complex wave
        system:      the TriModelSystem, used for the swarm refinement pass
        use_swarm:   if True, settle each wave through Zone B before centring
        oov:         policy name, kept for interface parity with the runner
    """

    def __init__(self, K: int, encode_fn, system=None, use_swarm: bool = False,
                 oov: str = "refuse", **kw):
        self.K = int(K)
        self.encode_fn = encode_fn
        self.system = system
        self.use_swarm = bool(use_swarm)
        self.oov = oov
        self.centroids: torch.Tensor | None = None
        self.classes_: list[int] = []
        self.oov_rates: list[float] = []

    # ------------------------------------------------------------------ settle
    def _wave(self, text: str) -> torch.Tensor:
        """Ingress, then optionally settle the wave through the Zone B swarm."""
        psi = self.encode_fn(text)
        if self.use_swarm and self.system is not None:
            with torch.no_grad():
                out = self.system.swarm(psi, patterns=self.system.axiom_bank)
                psi = out["psi"]
        return sub.unit_norm(psi)

    # --------------------------------------------------------------------- fit
    def fit(self, items) -> None:
        by_class: dict[int, list[torch.Tensor]] = {}
        for it in items:
            by_class.setdefault(int(it.label), []).append(self._wave(it.text))
        self.classes_ = sorted(by_class)
        rows = []
        for c in self.classes_:
            # mean wave, then re-project to the manifold: a wave centroid
            m = torch.stack(by_class[c]).sum(dim=0)
            rows.append(sub.unit_norm(m))
        self.centroids = torch.stack(rows)                 # [n_classes, D]

    # ----------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, item) -> int:
        if self.centroids is None:
            raise RuntimeError("fit() must run before predict()")
        w = self._wave(item.text)
        cos = (self.centroids.conj() @ w).real             # [n_classes]
        return int(self.classes_[int(cos.argmax())])

    @property
    def mean_oov_rate(self) -> float:
        return float(sum(self.oov_rates) / len(self.oov_rates)) if self.oov_rates else 0.0


def make_agent(K: int, small: bool = True, use_swarm: bool = False):
    """Factory used by the harness runner."""
    encode_fn, system, _tok = build_encoder(small=small)
    return HenriWaveCentroidAgent(K, encode_fn, system=system, use_swarm=use_swarm)


if __name__ == "__main__":
    # self-check: the adapter must separate classes above chance on a toy task
    import json

    from henri_core.bench_agent import make_agent as _mk

    K = 8
    texts = {i: [f"fact number {i} about entity {chr(97 + i)} attempt {j}"
                 for j in range(6)] for i in range(K)}
    train = [(i, t) for i in range(K) for t in texts[i][:3]]
    test = [(i, t) for i in range(K) for t in texts[i][3:]]

    class It:
        def __init__(self, label, text):
            self.label, self.text = label, text

    agent = _mk(K)
    agent.fit([It(l, t) for l, t in train])
    correct = sum(1 for l, t in test if agent.predict(It(l, t)) == l)
    acc = correct / len(test)
    print(json.dumps({
        "schema": "henri.tri-model.bench-adapter.v1",
        "adapter_id": ADAPTER_ID,
        "K": K,
        "n_train": len(train), "n_test": len(test),
        "accuracy": acc,
        "chance": 1.0 / K,
        "beats_chance": acc > 1.0 / K,
        "spec_status": "NO_PINNED_DATASET_ADAPTER_STAGED_ONLY",
    }, indent=2))

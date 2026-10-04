"""HENRI collision-managed egress feature map (ADDITIVE, opt-in).

MEASURED BASIS -- design/zone_a/evidence/egress_ceiling_resolved_receipt.json
    K=513, 8 salts, held-out accuracy, single-seed receipt reproduced exactly:
        shipped encode_egress          0.4298  (seed0 0.5146 = receipt)
        greedy least-loaded, b=16      0.8252
        collision-free (b_eff=D//F)    1.0000
    Pairwise off-diagonal |cos| stayed FLAT (0.4967 -> 0.4771) while accuracy
    moved +0.57. Crosstalk is NOT the channel. AGGREGATE SLOT COLLISION is.
    cfree = 1.0000 at EVERY configuration tested, including load 2.004.

THE GEOMETRY (pinned from zone_c_world_knowledge_codec.py, not from docs)
    M = NUM_BLOCKS = 8192, BD = BLOCK_DIM = 8, b = WAVE_EXPAND = 16, D = 65536.
    A feature writes b signed SLOTS (block, dim8), one dim per block.
    Features interfere IFF they share a SLOT. Sharing only a block at different
    dims is ORTHOGONAL, so the collision unit is the slot, not the block.
    Collision-free FEATURE capacity at b slots is D // b.

WHY THIS IS A SEPARATE MODULE
    encode() and encode_egress() stay BYTE-UNCHANGED. This module re-implements
    the slot-ADDRESSING step only. Retrieval, ingestion, and every stored engram
    keep identical bytes. Import this module to opt in; nothing else changes.

WHY ASSIGNMENT AND NOT MORE BLOCKS
    Raising M is not deployable: the wave carries M=8192 blocks by contract.
    Managing the assignment reaches 1.0000 at the SHIPPED M without touching
    encode_egress() bytes.

WHY NOT THE BLUEPRINT'S 4-SLOT PARTITION
    Directive 2 confines blocks and multiplies per-slot load. Measured at K=513:
    partition 0.1780 vs shipped 0.4298 (-0.2518), while its V-H4 gate reports
    inter-slot collisions = 0. That gate passes while accuracy collapses.
    Partitioning is not the lever. SLOT LOAD is.

CONTRACT
    build_allocator(features, b=16)  -> Allocator (raises if collision-free is
                                        impossible at b and D)
    encode_managed(text, alloc)      -> [M, BD] float32, row-L2-normalized,
                                        same layout and dtype as encode_egress
    Allocator.collision_report()     -> realized crowding; assert zero when
                                        F*b <= D
    Fail closed: an unknown feature raises; a crowding allocator raises.

NOT CLAIMED
    No wave->text egress. No AAII v4.3 result. No GPU run. This is a feature map
    for typed decision manifolds, measured on template and arbitrary corpora.
"""
from __future__ import annotations

import heapq
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import zone_c_world_knowledge_codec as C  # noqa: E402

NUM_BLOCKS = int(C.NUM_BLOCKS)
BLOCK_DIM = int(C.BLOCK_DIM)
WAVE_EXPAND = int(C.WAVE_EXPAND)
WAVE_DIM = int(C.WAVE_DIM)
TOTAL_SLOTS = NUM_BLOCKS * BLOCK_DIM


class AllocationError(ValueError):
    """Raised when a requested allocation cannot satisfy its stated contract."""


class Allocator:
    """Deterministic feature -> slot map with a realized-crowding report.

    Built once from the CLOSED feature set of a corpus. Greedy least-loaded:
    each feature takes the b currently-least-used slots. When len(features)*b
    <= TOTAL_SLOTS this is collision-free by construction; otherwise it
    minimizes the maximum slot load.
    """

    def __init__(self, features, b: int = WAVE_EXPAND, seed: int = 0):
        feats = list(dict.fromkeys(features))            # dedupe, keep order
        if b < 1:
            raise AllocationError(f"b must be >= 1, got {b}")
        self.b = int(b)
        self.seed = int(seed)
        self.features = feats
        self._slots = {}
        rng = np.random.default_rng(seed)
        # D58 SELF-CAUGHT DEFECT. The first draft built the heap as (load, slot).
        # Heap pop returns the MINIMUM tuple, so with every load at 0 the order
        # was simply ascending slot index -- the seed permutation was DISCARDED
        # and the map was seed-INDEPENDENT. A dead parameter promises randomness
        # that does not exist, and it made the probe 'managed'/'cfree' arms
        # deterministic (structural-zero SE). Fix: tie-break equal-load slots by
        # the seed permutation rank, so the seed genuinely selects among them.
        # Collision-free is PRESERVED: greedy still pops the minimum LOAD, and
        # the tie-break only chooses among slots that share that load.
        perm = rng.permutation(TOTAL_SLOTS)
        rank = np.empty(TOTAL_SLOTS, dtype=np.int64)
        rank[perm] = np.arange(TOTAL_SLOTS, dtype=np.int64)
        load = np.zeros(TOTAL_SLOTS, dtype=np.int64)
        heap = [(0, int(rank[s]), int(s)) for s in range(TOTAL_SLOTS)]
        heapq.heapify(heap)
        for f in feats:
            picked = []
            for _ in range(self.b):
                ld, _, s = heapq.heappop(heap)
                picked.append(s)
                heapq.heappush(heap, (ld + 1, int(rank[s]), int(s)))
            self._slots[f] = (np.array(picked, dtype=np.int64), None)
        self._report = self._measure()

    # ------------------------------------------------------------- allocation
    def slots_of(self, feature: str):
        if feature not in self._slots:
            raise AllocationError(
                f"feature {feature!r} is not in this allocator; build the "
                "allocator from the full closed feature set")
        return self._slots[feature]

    @property
    def max_features_collision_free(self) -> int:
        return TOTAL_SLOTS // self.b

    @property
    def collision_free(self) -> bool:
        return self._report["zero_collision_frac"] >= 1.0

    def _measure(self) -> dict:
        fid = {f: i for i, f in enumerate(self.features)}
        slot = np.concatenate([v[0] for v in self._slots.values()])
        feat = np.concatenate([np.full(self.b, fid[f]) for f in self.features])
        order = np.argsort(slot, kind="stable")
        slot, feat = slot[order], feat[order]
        uniq, start, cnt = np.unique(slot, return_index=True, return_counts=True)
        partners = np.zeros(len(self.features), dtype=np.int64)
        for k in np.nonzero(cnt > 1)[0]:
            fl = feat[start[k]:start[k] + cnt[k]]
            partners[np.unique(fl)] += (fl.size - 1)
        return {
            "n_features": len(self.features), "b": self.b,
            "slots_used": int(uniq.size), "slots_total": TOTAL_SLOTS,
            "max_multiplicity": int(cnt.max()) if cnt.size else 0,
            "mean_collision_partners": round(float(partners.mean()), 3),
            "max_collision_partners": int(partners.max()) if partners.size else 0,
            "zero_collision_frac": round(float((partners == 0).mean()), 4),
        }

    def collision_report(self) -> dict:
        return dict(self._report)

    def assert_collision_free(self):
        """Fail closed: refuse to serve a crowding map as if it were clean."""
        if not self.collision_free:
            raise AllocationError(
                "allocator is not collision-free: "
                f"zero_collision_frac={self._report['zero_collision_frac']}, "
                f"n_features={len(self.features)} > capacity "
                f"{self.max_features_collision_free} at b={self.b}. "
                "Lower b or reduce the feature count.")
        return self


def collision_free_b(features, b_max: int = WAVE_EXPAND) -> int:
    """Largest b <= b_max that admits a collision-free map for this feature set.

    This is the cheap deployable lever: b_eff = min(b_max, D // F). Measured
    1.0000 held-out at every configuration where the resulting map is disjoint.
    """
    F = len(set(features))
    if F == 0:
        return int(b_max)
    return int(max(1, min(b_max, TOTAL_SLOTS // F)))


def build_allocator(features, b: int | None = None, seed: int = 0,
                    require_collision_free: bool = True) -> Allocator:
    """Build an allocator. b=None picks the largest collision-free b."""
    feats = list(dict.fromkeys(features))
    if b is None:
        b = collision_free_b(feats, WAVE_EXPAND)
    alloc = Allocator(feats, b=b, seed=seed)
    if require_collision_free:
        alloc.assert_collision_free()
    return alloc


def _signs(slots: np.ndarray) -> np.ndarray:
    return np.where((slots % 2) == 0, 1.0, -1.0).astype(np.float32)


def encode_managed(text: str, alloc: Allocator,
                   ngram_max: int = 3) -> np.ndarray:
    """Text -> [M, BD] float32 rows, exactly the encode_egress layout.

    ADDITIVE variant of encode_egress. The only change is slot ADDRESSING:
    managed slots instead of the sha256-LCG hash. Row L2-normalization, dtype,
    and shape are byte-identical to encode_egress.
    """
    feats = C.features_of(text, ngram_max=ngram_max)
    acc = np.zeros(WAVE_DIM, dtype=np.float32)
    for f in feats:
        sl, sg = alloc.slots_of(f)
        np.add.at(acc, sl, _signs(sl) if sg is None else sg)
    rows = acc.reshape(NUM_BLOCKS, BLOCK_DIM).copy()
    nrm = np.linalg.norm(rows, axis=1, keepdims=True)
    np.divide(rows, nrm, out=rows, where=nrm > 1e-9)
    return rows.astype(np.float32)


def managed_vocabulary(texts, b: int | None = None, seed: int = 0) -> Allocator:
    """Build one allocator from every feature of every text. Use for a corpus."""
    feats = []
    for tx in texts:
        feats.extend(C.features_of(tx, ngram_max=3))
    return build_allocator(feats, b=b, seed=seed)


def wave_features(rows: np.ndarray) -> np.ndarray:
    """[M, BD] float32 rows -> [DC, 2] L2-normalized features (typed-egress
    compatible). Plain reshape: features() preserves the codec float order."""
    nb, bd = NUM_BLOCKS, BLOCK_DIM
    a = np.ascontiguousarray(rows, dtype="<f4")
    z = a.reshape(nb, bd // 2, 2).view(np.complex64).reshape(nb * bd // 2)
    f = np.stack([z.real, z.imag], axis=-1).astype(np.float32)
    n = float(np.linalg.norm(f))
    return f / max(n, 1e-12)


__all__ = ["Allocator", "AllocationError", "build_allocator",
           "collision_free_b", "encode_managed", "managed_vocabulary",
           "wave_features", "NUM_BLOCKS", "BLOCK_DIM", "WAVE_EXPAND",
           "TOTAL_SLOTS"]


if __name__ == "__main__":
    import json
    import time
    t0 = time.time()
    texts = [f"run tool{i:04d} on arg0001" for i in range(64)]
    alloc = managed_vocabulary(texts)
    rep = alloc.collision_report()
    same_layout = None
    r_man = encode_managed("run tool0007 on arg0001", alloc)
    r_ref = C.get_codec().encode_egress("run tool0007 on arg0007")
    same_layout = (r_man.shape == r_ref.shape and r_man.dtype == r_ref.dtype)
    print(json.dumps({
        "allocator": rep,
        "collision_free": alloc.collision_free,
        "b_eff_for_64": collision_free_b(texts),
        "encode_managed_shape": list(r_man.shape),
        "encode_egress_shape": list(r_ref.shape),
        "layout_identical": bool(same_layout),
        "row_norm_min": round(float(np.linalg.norm(r_man, axis=1).min()), 6),
        "row_norm_max": round(float(np.linalg.norm(r_man, axis=1).max()), 6),
        "seconds": round(time.time() - t0, 3),
    }, indent=2))

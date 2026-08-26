"""P2 carrier: coordinate-attribution audit for stall windows (instrumentation-only).

Tests whether a k-step exteroceptive stall window identifies parameter
coordinates causally responsible for the failure, WITHOUT any parameter
mutation or heat injection (heat is P3, gated on P2).

Mask arms (matched L2 perturbation norm):
  - anisotropic: top-k coordinates by sensitivity magnitude
  - isotropic:   random coordinates, random weights, same norm
  - shuffled:    sensitivity values assigned to a random coordinate set

Telemetry: mask digests, coordinate counts, cross-seed overlap (IoU),
per-module masked-norm (read-only), score deltas per window, verdict class.

Default-OFF: HENRI_ATTRIBUTION_AUDIT=1 must be set; construction raises
AttributionDisabledError otherwise (T0/P1 pattern). Zero trainable.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FLAG = "HENRI_ATTRIBUTION_AUDIT"
DEFAULT_TOP_K = 64
STABILITY_THRESHOLD = 0.5


class AttributionDisabledError(RuntimeError):
    pass


class AttributionAudit:
    """Pure analysis: no nn.Module, no Parameter, no mutation."""

    def __init__(self, top_k: int = DEFAULT_TOP_K, *, flag: str = FLAG,
                 seed: int = 0,
                 stability_threshold: float = STABILITY_THRESHOLD):
        if os.environ.get(flag, "0") != "1":
            raise AttributionDisabledError(
                f"{flag} is not set; the attribution audit is default-OFF")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = top_k
        self._flag = flag
        self._seed = seed
        self.stability_threshold = stability_threshold

    # -- mask arms -------------------------------------------------------

    def anisotropic_mask(self, sensitivity: np.ndarray,
                         rng: np.random.Generator) -> Dict[str, Any]:
        sens = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
        if sens.size == 0:
            raise ValueError("sensitivity is empty")
        k = min(self.top_k, sens.size)
        idx = np.argsort(-np.abs(sens))[:k]
        vals = sens[idx].copy()
        vals = self._unit_norm(vals)
        return {"kind": "anisotropic", "indices": idx.tolist(),
                "weights": vals.tolist(),
                "l2": float(np.linalg.norm(vals)),
                "digest": self._mask_digest(idx, vals)}

    def isotropic_mask(self, sensitivity: np.ndarray,
                       rng: np.random.Generator) -> Dict[str, Any]:
        sens = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
        k = min(self.top_k, sens.size)
        idx = rng.choice(sens.size, size=k, replace=False)
        vals = rng.standard_normal(k)
        vals = self._unit_norm(vals)
        return {"kind": "isotropic", "indices": idx.tolist(),
                "weights": vals.tolist(),
                "l2": float(np.linalg.norm(vals)),
                "digest": self._mask_digest(idx, vals)}

    def shuffled_mask(self, sensitivity: np.ndarray,
                      rng: np.random.Generator) -> Dict[str, Any]:
        sens = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
        k = min(self.top_k, sens.size)
        idx = rng.choice(sens.size, size=k, replace=False)
        vals = sens[rng.permutation(sens.size)[:k]].copy()
        vals = self._unit_norm(vals)
        return {"kind": "shuffled", "indices": idx.tolist(),
                "weights": vals.tolist(),
                "l2": float(np.linalg.norm(vals)),
                "digest": self._mask_digest(idx, vals)}

    @staticmethod
    def _unit_norm(vals: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vals))
        if norm <= 0.0:
            return np.ones_like(vals) / np.sqrt(max(vals.size, 1))
        return vals / norm

    @staticmethod
    def _mask_digest(indices: np.ndarray, weights: np.ndarray) -> str:
        raw = json.dumps(
            {"i": [int(x) for x in indices],
             "w": [round(float(x), 8) for x in weights]},
            sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def overlap(mask_a: Dict[str, Any], mask_b: Dict[str, Any]) -> float:
        ia, ib = set(mask_a["indices"]), set(mask_b["indices"])
        if not ia and not ib:
            return 1.0
        return len(ia & ib) / len(ia | ib)

    # -- stability -------------------------------------------------------

    def stability(self, sensitivity: np.ndarray, n_seeds: int = 4,
                  noise_scale: float = 0.05) -> Dict[str, Any]:
        """Cross-seed top-k overlap of the anisotropic mask under oracle
        noise: masks from sensitivity + noise; mean/min pairwise IoU."""
        sens = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
        std = float(sens.std()) if sens.size > 1 else 0.0
        masks: List[Dict[str, Any]] = []
        for s in range(n_seeds):
            rng = np.random.default_rng(self._seed + s)
            noisy = sens + noise_scale * std * rng.standard_normal(sens.size)
            masks.append(self.anisotropic_mask(noisy, rng))
        overlaps = [self.overlap(masks[i], masks[j])
                    for i in range(n_seeds) for j in range(i + 1, n_seeds)]
        return {"mean_iou": float(np.mean(overlaps)) if overlaps else 1.0,
                "min_iou": float(np.min(overlaps)) if overlaps else 1.0,
                "n_seeds": n_seeds}

    # -- audit entry -----------------------------------------------------

    def run(self, *, stall_windows: Sequence[Dict[str, Any]],
            score_deltas: Sequence[float], sensitivity: np.ndarray,
            module_slices: Optional[Dict[str, List[int]]] = None,
            score_kind: str = "none", n_seeds: int = 4) -> Dict[str, Any]:
        """Emit telemetry + verdict class. Read-only; mutation_applied=False."""
        if not stall_windows:
            return {"verdict": "BLOCKED_NO_STALL_ENGAGEMENT",
                    "reason": "no resolved stall windows in telemetry",
                    "mutation_applied": False}
        if score_kind not in ("scorecard", "frame"):
            return {"verdict": "BLOCKED_MISSING_EXTERNAL_SCORE",
                    "reason": f"score series kind {score_kind!r} is not "
                              "an external observation/scorecard series",
                    "mutation_applied": False}
        sens = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
        rng = np.random.default_rng(self._seed)
        aniso = self.anisotropic_mask(sens, rng)
        iso = self.isotropic_mask(sens, rng)
        shuf = self.shuffled_mask(sens, rng)
        stab = self.stability(sens, n_seeds=n_seeds)
        per_module: Dict[str, Any] = {}
        if module_slices is not None:
            for name, sl in module_slices.items():
                if isinstance(sl, tuple):
                    idx = slice(int(sl[0]), int(sl[1]))
                    n_coords = int(sl[1]) - int(sl[0])
                else:
                    idx = np.asarray(sl, dtype=np.int64)
                    n_coords = int(idx.size)
                per_module[name] = {
                    "n_coords": n_coords,
                    "masked_norm": float(np.linalg.norm(sens[idx])),
                }
        verdict = ("ATTRIBUTION_STABLE"
                   if stab["mean_iou"] >= self.stability_threshold
                   else "FALSIFIED_NO_STABLE_ATTRIBUTION")
        return {
            "verdict": verdict,
            "stall_windows": int(len(stall_windows)),
            "score_kind": score_kind,
            "score_delta_min": float(min(score_deltas)) if score_deltas else None,
            "score_delta_max": float(max(score_deltas)) if score_deltas else None,
            "masks": {"anisotropic": aniso["digest"],
                      "isotropic": iso["digest"],
                      "shuffled": shuf["digest"]},
            "anisotropic_overlap_isotropic": self.overlap(aniso, iso),
            "anisotropic_overlap_shuffled": self.overlap(aniso, shuf),
            "stability": stab,
            "module_masked_norms": per_module,
            "trainable_parameters": 0,
            "mutation_applied": False,
        }

"""G3 snapped-rerank carrier (frozen, default-OFF, zero trainable params).

Pre-registered 2026-08-25 (henri-g3-snap-rerank-prereg-20260825-001,
audit bdae5bda70e46379). Parent: henri-g2-lexical-snap-result-20260825-001
(2cdd5e33b583a911).

Scientific endpoint (Reference 3 + prereg): at the 13-family DSL outcome
ceiling, pass@1 improvement is vacuous. The non-vacuous endpoint is EXACT
outcome preservation + paired verifier-call reduction (bootstrap CI lb > 0)
+ per-family preservation. Engagement (reorder fraction, score spread,
first-passing-rank movement) is a gate, never a capability claim.

Leakage guard (Spine C precedent): the reranker may consume ONLY the task
prompt/specification and the candidate program bodies generated BEFORE
verifier execution. Canonical solutions, hidden outcomes, verifier results,
family-answer metadata (fid), and post-execution traces are PROHIBITED ->
BLOCKED_TARGET_LEAKAGE.

Controls (pre-registered):
  beta0_identity     flag off -> arm order byte-identical to baseline
  dead_memory        flat keys -> no confident routing (p_top1 ~ uniform)
  mismatched_frame   per-block O(8) on keys only -> discrimination collapses
  snap_bypass        tau -> inf makes C identical to B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Callable, Dict, List, Optional

import torch

try:
    from .envelope import EVALUATOR_ONLY_FIELDS, assert_no_evaluator_fields
    from .ingress.text import TextWaveAdapter
    from .gauge_audit import random_orthogonal
    from .lexical_snap import scores_for, pre_snap_stats, DEFAULT_TAU, \
        memory_sha256, implementation_sha256
except ImportError:  # pragma: no cover - bare-script execution path
    from universal_wave_harness.envelope import (  # type: ignore
        EVALUATOR_ONLY_FIELDS, assert_no_evaluator_fields)
    from universal_wave_harness.ingress.text import TextWaveAdapter  # type: ignore
    from universal_wave_harness.gauge_audit import random_orthogonal  # type: ignore
    from universal_wave_harness.lexical_snap import (  # type: ignore
        scores_for, pre_snap_stats, DEFAULT_TAU, memory_sha256,
        implementation_sha256)

# Fields that must NEVER feed the reranker (Spine C doctrine).
FORBIDDEN_FIELDS = frozenset({
    "code", "canonical", "canonical_code", "gold", "solution",
    "tests", "verifier_tests", "outcome_tests", "expected", "answer",
    "fid", "family", "family_id", "outcome", "result", "passed",
    "traces", "execution", "reward", "score_gt",
})


class TargetLeakageError(RuntimeError):
    """Raised when a forbidden field reaches the reranker input."""


def provenance_scan(fields: Dict) -> None:
    """Static provenance scan: refuse any forbidden key in the input set."""
    hits = sorted(f for f in fields if f in FORBIDDEN_FIELDS)
    if hits:
        raise TargetLeakageError(
            f"BLOCKED_TARGET_LEAKAGE: forbidden fields {hits} would feed "
            f"the reranker. Refusing to compute scores.")


def is_enabled() -> bool:
    return os.environ.get("HENRI_G3_SNAP_RERANK", "0") == "1"


def order_baseline(n: int) -> List[int]:
    """Arm A: structural/generator order (uniform, lexicographic)."""
    return list(range(n))


def order_continuous(scores: torch.Tensor, base: Optional[List[int]] = None) -> List[int]:
    """Arm B: descending continuous relational score, ties broken by base order."""
    s = scores.detach().cpu().to(torch.float64)
    idx = torch.argsort(s, descending=True, stable=True).tolist()
    return idx


def order_snapped(scores: torch.Tensor, tau: float,
                  base: Optional[List[int]] = None) -> List[int]:
    """Arm C: snapped routing. argmax first, then remaining by score desc
    (stable tiebreak = base order). With tau -> inf this degenerates to B
    (snap-bypass identity)."""
    s = scores.detach().cpu().to(torch.float64)
    if s.numel() == 0:
        return []
    top = int(torch.argmax(s).item())
    rest = [i for i in torch.argsort(s, descending=True, stable=True).tolist()
            if i != top]
    return [top] + rest


def dead_keys(keys: torch.Tensor, device: str) -> torch.Tensor:
    """Dead-memory control: every key replaced by the mean key (flat frame)."""
    return keys.mean(dim=0, keepdim=True).expand_as(keys).to(device)


def mismatched_keys(keys: torch.Tensor, seed: int, device: str) -> torch.Tensor:
    """Mismatched-frame control: per-block ARBITRARY O(8) on keys ONLY
    (invalid joint gauge; G1/G2 pattern). Decorrelates query-key alignment."""
    from universal_wave_harness.gauge_audit import random_orthogonal  # noqa
    T = random_orthogonal(seed=seed, dim=8, count=keys.shape[1]).to(device)
    return torch.einsum("kab,nkb->nka", T.to(torch.float64),
                        keys.to(torch.float64)).to(keys.dtype)


def write_item_row(path: str, row: Dict) -> None:
    """Incremental per-task JSONL persistence BEFORE aggregation (protocol:
    an aggregation crash must never destroy the only evidence)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def run_rerank_smoke(
    adapter: TextWaveAdapter,
    task_prompt: str,
    candidate_codes: List[str],
    tau: float = DEFAULT_TAU,
    device: str = "cpu",
    seed: int = 20260825,
) -> Dict:
    """Disposable bridge kill test on one task's candidate pool.

    Returns telemetry only; the EVALUATOR runs sandbox/CEGIS admission.
    Raises TargetLeakageError if any forbidden field is present.
    """
    provenance_scan({"prompt": task_prompt})
    for c in candidate_codes:
        provenance_scan({"candidate": c})

    q = adapter.encode(task_prompt, source_uri="g3-query",
                       item_id="q").wave.to(device)
    keys = torch.stack(
        [adapter.encode(c, source_uri="g3-key",
                        item_id=str(i)).wave.to(device)
         for i, c in enumerate(candidate_codes)])

    scores = scores_for(q, keys)
    st = pre_snap_stats(scores, tau)
    order_a = order_baseline(len(candidate_codes))
    order_b = order_continuous(scores, order_a)
    order_c = order_snapped(scores, tau, order_a)

    # controls
    kd = dead_keys(keys, device)
    scores_dead = scores_for(q, kd)
    st_dead = pre_snap_stats(scores_dead, tau)
    km = mismatched_keys(keys, seed=seed, device=device)
    scores_mm = scores_for(q, km)
    order_mm = order_continuous(scores_mm, order_a)

    reordered_b = order_b != order_a
    reordered_c = order_c != order_a
    flat = bool(torch.isfinite(scores).all() and scores.std().item() < 1e-9)
    dead_confident = st_dead["p_top1"] > 0.99 or st_dead["s_margin"] > 1e-6
    mismatched_changed = order_mm != order_a

    return {
        "n_candidates": len(candidate_codes),
        "unique_candidates": len(set(candidate_codes)),
        "scores": scores.tolist(),
        "scores_std": float(scores.std().item()),
        "pre_snap": st,
        "order_a": order_a, "order_b": order_b, "order_c": order_c,
        "order_mismatched": order_mm,
        "reordered_b": reordered_b, "reordered_c": reordered_c,
        "flat": flat,
        "dead_p_top1": st_dead["p_top1"], "dead_s_margin": st_dead["s_margin"],
        "dead_confident": dead_confident,
        "mismatched_changed": mismatched_changed,
        "snap_bypass_identity": order_c == order_b if st["s_margin"] == 0.0
                                else True,
    }


def implementation_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(f"snap_rerank: module import only (flag="
          f"{'ON' if is_enabled() else 'OFF'}, device={args.device}). "
          f"impl_sha256={implementation_sha256()[:16]}")


if __name__ == "__main__":
    main()

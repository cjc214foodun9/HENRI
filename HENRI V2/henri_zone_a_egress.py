#!/usr/bin/env python
"""henri_zone_a_egress.py -- wire the orphaned fail-closed egress into a runner.

Defect 3 (SPEC-2026-10-02-ZONE-A.md section 7): `henri_hopfield_egress.py` is
default-OFF and imported only by its own test.  The fail-closed egress layer is
wired into nothing.  This module de-orphans it by providing the missing CALLER.

What this adds
    ZoneAEgressRunner  -- registration + candidate snapping over
                          CanonicalCodebookEgress, with per-call accounting and
                          a fail-closed batch path.
    main()             -- `--smoke` exercises the real path end to end and prints
                          a JSON verdict, so de-orphaning is demonstrable by
                          execution rather than by inspection.

Fail-closed discipline (inherited, not weakened)
    A snapped prototype outside the canonical allowlist is REJECTED, never
    emitted.  `decode_valid` raises in strict mode.  This runner adds no
    fallback of its own: on rejection it reports REJECTED and emits nothing.

Default-OFF: HENRI_ZONE_A_EGRESS=1 is required.  The runner never replaces the
CLASS51 adapter path; it is additive and coexists with it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

_HERE = Path(__file__).resolve()
_V2 = _HERE.parent
if str(_V2) not in sys.path:
    sys.path.insert(0, str(_V2))

from henri_hopfield_egress import (                    # noqa: E402
    CanonicalCodebookEgress,
    EgressSyntaxRejectedError,
    ValidatedEgressResult,
    noise_floor_sigma,
)

ENV_ENABLE_FLAG = "HENRI_ZONE_A_EGRESS"


def zone_a_egress_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_FLAG, "").strip() in {"1", "true", "True", "yes"}


class ZoneAEgressError(RuntimeError):
    """Base class for Zone A egress runner failures."""


class ZoneAEgressDisabledError(ZoneAEgressError):
    """Raised when the runner is used with HENRI_ZONE_A_EGRESS unset or 0."""


@dataclass
class EgressBatchReport:
    snapped: int = 0
    rejected: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    noise_floor: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "snapped": self.snapped,
            "rejected": self.rejected,
            "noise_floor": self.noise_floor,
            "results": self.results,
        }


class ZoneAEgressRunner:
    """Caller for the fail-closed Modern Hopfield egress.

    The account of snapped vs rejected candidates is the telemetry the live
    engine lacked; nothing is fabricated on rejection.
    """

    def __init__(self, dim: int, beta: float = 8.0, eps: float = 0.15) -> None:
        if not zone_a_egress_enabled():
            raise ZoneAEgressDisabledError(
                f"{ENV_ENABLE_FLAG} is not set; Zone A egress runner is disabled"
            )
        if dim <= 0:
            raise ZoneAEgressError(f"dim must be positive; got {dim}")
        self.dim = int(dim)
        self.beta = float(beta)
        self.eps = float(eps)
        self.egress = CanonicalCodebookEgress(dim=self.dim, beta=self.beta)
        self.noise_floor = noise_floor_sigma(self.dim, self.eps)
        self._calls = 0

    @property
    def calls(self) -> int:
        """Number of candidates actually routed through the egress."""
        return self._calls

    @torch.no_grad()
    def register(
        self,
        code_waves: torch.Tensor,
        canonical_ids: Sequence[int],
        validator: Optional[Callable[[int], bool]] = None,
    ) -> int:
        return int(self.egress.register(code_waves, canonical_ids, validator))

    def snap(self, wave: torch.Tensor) -> ValidatedEgressResult:
        """Snap one candidate; REJECTED propagates, never a fallback payload."""
        self._calls += 1
        return self.egress.decode(wave)

    def snap_strict(self, wave: torch.Tensor) -> Tuple[int, float]:
        """Strict path for score-bearing use: raises on rejection."""
        self._calls += 1
        return self.egress.decode_valid(wave)

    def run_batch(self, waves: torch.Tensor) -> EgressBatchReport:
        """Snap a batch of candidate waves [N, D] and account for each outcome."""
        if waves.ndim != 2 or waves.shape[-1] != self.dim:
            raise ZoneAEgressError(
                f"waves must be [N, {self.dim}]; got {tuple(waves.shape)}"
            )
        report = EgressBatchReport(noise_floor=self.noise_floor)
        for i in range(waves.shape[0]):
            res = self.snap(waves[i])
            entry = {
                "index": i,
                "status": res.status,
                "snapped_index": res.snapped_index,
                "similarity": res.similarity,
                "reason": res.reason,
            }
            report.results.append(entry)
            if res.status == "SNAPPED":
                report.snapped += 1
            else:
                report.rejected += 1
        return report


# ------------------------------------------------------------------ self-test
def _smoke() -> Dict[str, Any]:
    """Exercise the real path: register a codebook, snap clean + perverse inputs."""
    os.environ.setdefault(ENV_ENABLE_FLAG, "1")
    dim, m = 64, 6
    g = torch.Generator().manual_seed(20261002)

    # Orthonormal codebook -> distinct canonical ids.
    raw = torch.randn(m, dim, generator=g)
    q, _ = torch.linalg.qr(raw.transpose(0, 1))
    code_waves = q.transpose(0, 1).contiguous()
    canonical_ids = list(range(100, 100 + m))

    runner = ZoneAEgressRunner(dim=dim, beta=8.0)
    registered = runner.register(code_waves, canonical_ids)

    # 1. clean recall: exact prototype must snap to its own canonical id
    clean_ok = []
    for k in range(m):
        res = runner.snap(code_waves[k])
        clean_ok.append(res.status == "SNAPPED" and res.snapped_index == canonical_ids[k])

    # 2. out-of-allowlist: the egress must never emit an unlisted id. Register a
    #    codebook whose ids are all off-allowlist and confirm REJECTED.
    runner2 = ZoneAEgressRunner(dim=dim, beta=8.0)
    runner2.register(code_waves, canonical_ids,
                     validator=lambda cid: cid in canonical_ids[:3])
    rejected = runner2.snap(code_waves[5])  # id 105 fails the validator

    # 3. fail-closed on empty codebook
    runner3 = ZoneAEgressRunner(dim=dim, beta=8.0)
    empty = runner3.snap(code_waves[0])

    # 4. noise floor arithmetic: eps / sqrt(D)
    expected_floor = 0.15 / (dim ** 0.5)

    return {
        "registered": registered,
        "clean_all_snapped_correctly": all(clean_ok),
        "validator_rejected_unlisted": rejected.status == "REJECTED",
        "empty_codebook_fail_closed": empty.status == "REJECTED",
        "noise_floor_matches_formula": abs(runner.noise_floor - expected_floor) < 1e-12,
        "noise_floor": runner.noise_floor,
        "calls_routed": runner.calls + runner2.calls + runner3.calls,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Zone A egress runner / de-orphan smoke")
    parser.add_argument("--smoke", action="store_true", help="run the end-to-end smoke")
    args = parser.parse_args(argv)
    if not args.smoke:
        parser.print_help()
        return 0
    out = _smoke()
    ok = all(v for k, v in out.items() if isinstance(v, bool))
    out["smoke_ok"] = bool(ok)
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

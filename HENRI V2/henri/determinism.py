"""Deterministic seeding standard (specification refinements doc, sec. 1.2 item 2).

SPECIFICATION
-------------
"All pseudo-random number generation (PRNG) must derive from an immutable run
manifest (`seed_seq`). Any test case displaying non-zero entropy between
repeated executions must trigger an immediate harness error."

WHAT THIS MODULE DOES
---------------------
Derives a per-component seed from a single immutable manifest seed, so that
component seeds are reproducible AND independent, and exposes a guard that
detects nondeterministic draws.

WHY DERIVED RATHER THAN FIXED-AT-1
----------------------------------
Setting every component to the same constant seed makes two components draw
IDENTICAL streams, which silently correlates arms and can manufacture a false
"agreement" between an active arm and its control. `spawn` derives distinct,
stable child seeds from a parent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict

import torch

DEFAULT_MANIFEST_SEED = 20261001
_UINT31 = 2**31 - 1


def derive_seed(manifest_seed: int, component: str) -> int:
    """Stable, independent child seed. Same inputs -> same seed, always."""
    h = hashlib.sha256(f"{int(manifest_seed)}::{component}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % _UINT31


@dataclass
class RunManifest:
    """Immutable run manifest. The single source of PRNG seeding."""

    seed_seq: int = DEFAULT_MANIFEST_SEED
    components: Dict[str, int] = field(default_factory=dict)

    def seed_for(self, component: str) -> int:
        if component not in self.components:
            self.components[component] = derive_seed(self.seed_seq, component)
        return self.components[component]

    def apply(self, component: str) -> int:
        """Seed torch globally AND return the seed for component-local use.

        CUDA deterministic flags are set only when CUDA is available; setting
        them on a CPU build is a no-op that would otherwise raise.
        """
        s = self.seed_for(component)
        torch.manual_seed(s)
        if torch.cuda.is_available():          # pragma: no cover - no GPU on host
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        return s

    def as_dict(self) -> dict:
        return {"seed_seq": self.seed_seq, "components": dict(self.components)}


def assert_deterministic(draw_fn) -> None:
    """Raise unless two repeated calls to `draw_fn` agree bit-for-bit.

    CONTRACT: `draw_fn` owns its own seeding, derived from the run manifest. The
    harness must NOT re-seed between the two draws.

    WHY — a measured vacuity (2026-10-01). An earlier version of this function
    applied the manifest itself before each draw. That made the check
    worthless: re-seeding with the same seed before both calls makes the two
    draws identical even when `draw_fn` contains wholly unseeded randomness.
    The guard passed for a `lambda: torch.randn(8)` with no seeding at all.
    Seeding is therefore the caller's obligation and only the comparison lives
    here.

    LIMIT: this detects nondeterminism WITHIN one process. Cross-process
    entropy is caught by the receipt-hash comparison in the gate runner, not
    here.
    """
    a = draw_fn()
    b = draw_fn()
    if not torch.equal(a, b):
        raise AssertionError(
            "NONDETERMINISTIC SEEDING DETECTED: two repeat draws disagreed. "
            "Seed from the run manifest inside draw_fn (doc sec. 1.2 item 2)."
        )

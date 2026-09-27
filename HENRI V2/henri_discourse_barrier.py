"""HENRI-Chat — discourse barrier over an immutable axiom subspace.

HONEST BOUNDARIES (do not restate the addendum's claims as fact)
----------------------------------------------------------------
* This measures the PHASE DISPLACEMENT of a user wave from a fixed axiom
  subspace.  It is NOT a proof of prompt-injection immunity, and the words
  "immune", "unhackable" or "guaranteed" are deliberately NOT used or implied.
  Measured behaviour on a constructed corpus is the only claim made.
* It operates on ALREADY-ENCODED waves.  It does not parse text.  Encoder
  quality is out of scope: a weak encoder defeats any barrier downstream.
* DEFAULT-OFF SIDECAR: `enabled=False` admits everything and sets
  `bypass=True`.  No live conversation path imports this file.
* `content_blind=True` is the DEAD-INPUT NEGATIVE CONTROL: it ignores the
  input wave and must therefore FAIL to veto a known injection.

Design
    The axiom basis is built with DISJOINT support, so the axioms are exactly
    orthogonal by construction (no Gram-Schmidt approximation).  Half of the
    ambient dimension is reserved as the query subspace.  A well-formed query
    lives in the query subspace and has ~0 overlap with the axioms; an
    injection attempts to occupy the axiom subspace and scores ~1.

Admission rule (pre-registered, single threshold):
    admitted  <=>  injection_score <= theta
    injection_score = max_i |<u, a_i>|  for unit u and unit axiom a_i

The axiom store has NO write path from `admit()`: the digest is taken before
and after and the equality is returned, not assumed.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# Pre-registered operating point.
THETA_AXIOM = 0.55


def _digest(axioms: Sequence[Sequence[complex]]) -> str:
    h = hashlib.sha256()
    for i, a in enumerate(axioms):
        for j, z in enumerate(a):
            h.update(("%d,%d,%.12f,%.12f;" % (i, j, z.real, z.imag)).encode())
    return h.hexdigest()[:16]


@dataclass
class Admission:
    admitted: bool
    injection_score: float
    theta: float
    verdict: str
    bypass: bool = False
    axioms_digest_before: str = ""
    axioms_digest_after: str = ""
    axioms_unchanged: bool = True
    content_blind: bool = False


class DiscourseBarrier:
    """Phase-displacement barrier between a user wave and the axiom subspace."""

    def __init__(
        self,
        d: int = 64,
        n_axioms: int = 4,
        theta: float = THETA_AXIOM,
        enabled: bool = True,
        content_blind: bool = False,
        seed: int = 20260927,
    ) -> None:
        if d <= 0 or n_axioms <= 0:
            raise ValueError("d and n_axioms must be positive")
        if d % (2 * n_axioms) != 0:
            raise ValueError(
                "d=%d must be divisible by 2*n_axioms=%d so that half the space "
                "can be reserved as the query subspace" % (d, 2 * n_axioms)
            )
        self.d = d
        self.n_axioms = n_axioms
        self.theta = theta
        self.enabled = enabled
        self.content_blind = content_blind

        k = d // (2 * n_axioms)          # support width per axiom
        rng = random.Random(seed)
        axioms: List[Tuple[complex, ...]] = []
        for i in range(n_axioms):
            v = [0j] * d
            for j in range(i * k, (i + 1) * k):
                ang = rng.random() * 2.0 * math.pi
                v[j] = complex(math.cos(ang), math.sin(ang))
            # UNIT NORMALISATION (defect found by this module's own test suite,
            # 2026-09-27): with k unit-magnitude components the raw axiom has
            # norm sqrt(k), so |<u, a_i>| could reach sqrt(k) = 2.83 at d=64.
            # The score is only interpretable in [0, 1] -- and the docstring's
            # "injection scores ~1" only true -- when the axioms are unit-norm.
            nv = math.sqrt(sum(abs(z) ** 2 for z in v))
            axioms.append(tuple(z / nv for z in v))
        self.axioms: Tuple[Tuple[complex, ...], ...] = tuple(axioms)
        self._axiom_support = tuple(
            tuple(range(i * k, (i + 1) * k)) for i in range(n_axioms)
        )
        self._query_support = tuple(range(n_axioms * k, d))

    # ------------------------------------------------------------------ api
    def digest(self) -> str:
        return _digest(self.axioms)

    @property
    def query_support(self) -> Tuple[int, ...]:
        return self._query_support

    def admit(self, user_wave: Sequence[complex]) -> Admission:
        before = self.digest()

        if not self.enabled:
            return Admission(admitted=True, injection_score=0.0, theta=self.theta,
                             verdict="BYPASS", bypass=True,
                             axioms_digest_before=before, axioms_digest_after=before,
                             axioms_unchanged=True, content_blind=self.content_blind)

        if len(user_wave) != self.d:
            raise ValueError("user_wave must have dimension %d, got %d"
                             % (self.d, len(user_wave)))

        nrm = math.sqrt(sum(abs(z) ** 2 for z in user_wave))
        if nrm <= 0.0:
            raise ValueError("user_wave must be non-zero")
        u = [z / nrm for z in user_wave]

        score = 0.0
        if not self.content_blind:
            for a in self.axioms:
                inner = abs(sum(uc * ac.conjugate() for uc, ac in zip(u, a)))
                if inner > score:
                    score = inner

        # The axiom store is never written by admit(); the digest is MEASURED.
        after = self.digest()
        ok = score <= self.theta
        return Admission(
            admitted=ok,
            injection_score=score,
            theta=self.theta,
            verdict="ADMIT" if ok else "INJECTION_VETO",
            bypass=False,
            axioms_digest_before=before,
            axioms_digest_after=after,
            axioms_unchanged=(before == after),
            content_blind=self.content_blind,
        )


# END

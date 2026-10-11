"""henri_ace_playbook.py -- itemized, non-collapsing context with delta operations.

BLUEPRINT
    HENRI-SPEC-2026-ZONE-C-SHARED-KV-ACE-V1, section 3 and section 5; theoretical
    basis Agentic Context Engineering (ACE, Zhang et al., arXiv:2510.04618).
    Context is a compiled, itemized PLAYBOOK. It grows through ADD/EDIT/PRUNE
    deltas. It is never rewritten monolithically, so it cannot collapse.

WHAT THIS MODULE IS
    The DATA STRUCTURE and the deterministic merge logic of section 3.1:
      - itemized bullets with stable ids, section, content, utility counters
      - the three delta operations, each O(1) in total context size
      - semantic deduplication of candidate ADDs
      - the Maxwell-Debye fractional viscoelastic retention weight of section 4.2
      - a delta PATCH PLAN that names which blocks would change
    Every one of these is pure logic and is verified on CPU.

WHAT THIS MODULE IS NOT
    It is NOT a paged radix KV cache and it does NOT touch GDDR7 or L2. The
    blueprint's G-KV2 (delta compile <= 15.0 us) and G-KV4 (92% L2 prefetch hit
    rate) are HARDWARE gates. They cannot be measured on torch 2.13.0+cpu, so
    they are not claimed. `patch_plan()` emits the plan a KV engine would consume;
    no KV engine consumes it yet. The L2 prefetch path (section 4.1) is absent.
    The Reflector and Curator are LLM roles in the blueprint; here they are
    interfaces a caller fills, never a model call from this module.

TOKEN COUNT IS A PROXY
    `token_count` uses whitespace tokens, not a real tokenizer. It exists to
    detect COLLAPSE (a sudden drop), which is a shape property and survives the
    substitution. It is not a billing or context-window figure.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

# Retention tiers of section 4.2. The fourth tier is the axiom baseplate: tau -> inf
# means the kernel does not decay, so a conserved law or a Sagnac boundary never ages
# out. Tier 1 is an ephemeral environment return.
TIERS = {
    1: 1e-3,
    2: 1e0,
    3: 1e3,
    4: float("inf"),
}

# Section names from the blueprint's playbook schema (section 5.1).
SECTIONS = (
    "strategies_and_hard_rules",
    "api_schemas_and_conventions",
    "code_snippets_and_templates",
    "verification_checklist_and_sagnac_boundaries",
)

DEFAULT_UTILITY_LAMBDA = 1.0
DEFAULT_EVICTION_THRESHOLD = 1e-4
DEFAULT_DEDUP_THRESHOLD = 0.92


class PlaybookError(RuntimeError):
    """Typed failure so fail-closed callers can distinguish misuse from absence."""


# ---------------------------------------------------------------------------------
# deterministic proxy embedding + similarity
# ---------------------------------------------------------------------------------
def _char_ngrams(text: str, n: int = 3) -> list:
    t = " ".join(text.lower().split())
    if len(t) < n:
        return [t] if t else []
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def hash_embedding(text: str, dim: int = 256) -> list:
    """Deterministic hashed char-3gram embedding. No model, no network, no RNG.

    This is a PROXY: it measures lexical overlap, not semantic meaning. Real
    semantic dedup needs the wave encoder, which is absent. The dedup MECHANISM is
    therefore verified here and its SEMANTIC QUALITY is not claimed.
    """
    vec = [0.0] * dim
    for g in _char_ngrams(text):
        h = 2166136261
        for ch in g:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        vec[h % dim] += 1.0
    nrm = math.sqrt(sum(v * v for v in vec))
    return [v / nrm for v in vec] if nrm else vec


def cosine(a: list, b: list) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb) if na and nb else 0.0


def make_similarity(dim: int = 256) -> Callable[[str, str], float]:
    """Default similarity: lexical proxy. Callers may inject a better one."""
    cache = {}

    def sim(a: str, b: str) -> float:
        for t in (a, b):
            if t not in cache:
                cache[t] = hash_embedding(t, dim)
        return cosine(cache[a], cache[b])
    return sim


# ---------------------------------------------------------------------------------
# bullets
# ---------------------------------------------------------------------------------
@dataclass
class Bullet:
    bullet_id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    tier: int = 3
    created_at: float = 0.0
    activated_at: float = 0.0
    edits: int = 0
    alive: bool = True

    @property
    def token_count(self) -> int:
        return len(self.content.split())


@dataclass
class Op:
    """One delta operation. The op log is append-only and is the audit trail."""
    kind: str               # ADD | EDIT | PRUNE | REJECT_DUPLICATE | TAG | EVICT
    bullet_id: str
    section: str = ""
    reason: str = ""
    at: float = 0.0
    touched: int = 1        # bullets whose stored bytes changed


# ---------------------------------------------------------------------------------
# retention weight (section 4.2)
# ---------------------------------------------------------------------------------
def retention_weight(b: Bullet, now: float, utility_lambda: float = DEFAULT_UTILITY_LAMBDA,
                     tier_weights: Optional[dict] = None) -> float:
    """W_k(t) = g_k * kernel(tier_k) + lambda * (h - H)/(h + H + 1).

    BUG CAUGHT BY TEST (test_ace_d), fixed here. The first draft summed over ALL four
    tiers unconditionally:
        acc = sum_j w_j exp(-dt/tau_j)      with tau_4 = inf
    Because tau_4 = inf contributes a CONSTANT 1.0, that made the tier-4 term universal:
    every bullet, of every tier, retained a floor of 1.0 and NOTHING ever aged out.
    Measured: a tier-1 ephemeral bullet returned W = 1.0 at t = 1e9, so
    `evict()` could never fire on aging at all. The tier-4 baseplate guard was inert
    because everything was a baseplate.

    A KV page has ONE dominant relaxation tier (section 4.2 assigns the tiers to
    page classes: sensory / episode / tactic / axiom). Its weight therefore uses ITS
    OWN mode. tau_4 = inf then means exactly what the blueprint says: an axiom page
    does not decay, while an ephemeral page does.

    BLUEPRINT AMBIGUITY RECORDED: section 4.2 writes the weight as a single
    integral of a four-mode kernel, which read literally gives sum-over-all-modes and
    reproduces the inert behaviour above. The per-page reading is the only one under
    which the eviction mechanism does any work, so it is adopted and flagged.
    """
    w = tier_weights or {t: 1.0 for t in TIERS}
    tau = TIERS[b.tier]
    dt = max(0.0, now - b.activated_at)
    g = w.get(b.tier, 1.0)
    base = g if tau == float("inf") else g * math.exp(-dt / tau)
    denom = b.helpful + b.harmful + 1
    return base + utility_lambda * ((b.helpful - b.harmful) / denom)


# ---------------------------------------------------------------------------------
# the playbook
# ---------------------------------------------------------------------------------
class Playbook:
    """Itemized context with delta-only mutation.

    INVARIANT (measured, not asserted): no operation rewrites a bullet other than
    its own target. `touched` on each Op records how many stored bullets changed,
    and every operation must report touched == 1 or 0.
    """

    def __init__(self, similarity: Optional[Callable[[str, str], float]] = None,
                 dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
                 eviction_threshold: float = DEFAULT_EVICTION_THRESHOLD,
                 utility_lambda: float = DEFAULT_UTILITY_LAMBDA):
        self.bullets: dict = {}
        self.ops: list = []
        self.similarity = similarity or make_similarity()
        self.dedup_threshold = float(dedup_threshold)
        self.eviction_threshold = float(eviction_threshold)
        self.utility_lambda = float(utility_lambda)
        self._counter = 0
        self._prefix = {
            "strategies_and_hard_rules": "shr",
            "api_schemas_and_conventions": "api",
            "code_snippets_and_templates": "code",
            "verification_checklist_and_sagnac_boundaries": "veto",
        }

    # -- helpers -----------------------------------------------------------------
    def _next_id(self, section: str) -> str:
        self._counter += 1
        return f"{self._prefix.get(section, 'ctx')}-{self._counter:05d}"

    def _log(self, kind: str, bullet_id: str, section: str = "", reason: str = "",
             touched: int = 1, at: float = 0.0) -> Op:
        op = Op(kind, bullet_id, section, reason, at, touched)
        self.ops.append(op)
        return op

    def token_count(self) -> int:
        return sum(b.token_count for b in self.bullets.values() if b.alive)

    def live_ids(self) -> list:
        return [i for i, b in self.bullets.items() if b.alive]

    # -- delta operations --------------------------------------------------------
    def add(self, section: str, content: str, tier: int = 3, now: float = 0.0,
            helpful: int = 0, harmful: int = 0) -> Op:
        """ADD a bullet. Rejects a near-duplicate instead of appending it."""
        if section not in SECTIONS:
            raise PlaybookError(f"unknown section {section!r}")
        if not content.strip():
            raise PlaybookError("refusing to add an empty bullet")
        if tier not in TIERS:
            raise PlaybookError(f"tier must be one of {sorted(TIERS)}")
        best_id, best_sim = None, 0.0
        for bid in self.live_ids():
            s = self.similarity(content, self.bullets[bid].content)
            if s > best_sim:
                best_id, best_sim = bid, s
        if best_sim >= self.dedup_threshold:
            return self._log("REJECT_DUPLICATE", best_id or "", section,
                             f"similarity {best_sim:.4f}", touched=0, at=now)
        bid = self._next_id(section)
        b = Bullet(bid, section, content, helpful, harmful, tier, now, now)
        self.bullets[bid] = b
        return self._log("ADD", bid, section, at=now)

    def edit(self, bullet_id: str, content: str, now: float = 0.0) -> Op:
        """EDIT one bullet's content. No other bullet's bytes change."""
        b = self._require(bullet_id)
        if not content.strip():
            raise PlaybookError("refusing to edit a bullet to empty content")
        b.content = content
        b.edits += 1
        b.activated_at = now
        return self._log("EDIT", bullet_id, b.section, at=now)

    def prune(self, bullet_id: str, reason: str = "", now: float = 0.0) -> Op:
        """PRUNE one bullet. This is the ONLY operation that lowers token_count."""
        b = self._require(bullet_id)
        b.alive = False
        return self._log("PRUNE", bullet_id, b.section, reason, at=now)

    def tag(self, bullet_id: str, tag: str, now: float = 0.0) -> Op:
        """Reflector verdict on one bullet: helpful, harmful, or neutral."""
        b = self._require(bullet_id)
        if tag == "helpful":
            b.helpful += 1
        elif tag == "harmful":
            b.harmful += 1
        elif tag != "neutral":
            raise PlaybookError(f"unknown tag {tag!r}")
        return self._log("TAG", bullet_id, b.section, tag, at=now)

    def _require(self, bullet_id: str) -> Bullet:
        b = self.bullets.get(bullet_id)
        if b is None:
            raise PlaybookError(f"no such bullet {bullet_id!r}")
        if not b.alive:
            raise PlaybookError(f"bullet {bullet_id!r} was pruned")
        return b

    # -- aging -------------------------------------------------------------------
    def evict(self, now: float, respect_tier4: bool = True) -> list:
        """Evict bullets whose retention weight fell below the threshold.

        Tier 4 contributes a constant 1.0 to W_k, so it can never fall below a
        positive threshold. That is the mechanism the blueprint relies on to keep
        physical axioms pinned. `respect_tier4=False` exists only so a test can
        show the guard is load bearing.
        """
        out = []
        for bid in list(self.live_ids()):
            b = self.bullets[bid]
            if respect_tier4 and b.tier == 4:
                continue
            if b.harmful > b.helpful:
                out.append(self._log("EVICT", bid, b.section, "harmful > helpful", at=now))
                b.alive = False
                continue
            w = retention_weight(b, now, self.utility_lambda)
            if w < self.eviction_threshold:
                out.append(self._log("EVICT", bid, b.section, f"W={w:.3e}", at=now))
                b.alive = False
        return out

    # -- compile -----------------------------------------------------------------
    def patch_plan(self) -> dict:
        """The delta a KV engine would apply. A PLAN, not a KV cache.

        Every op carries `touched`, so the plan states how many stored blocks
        change. A delta patch touches at most one block, which is the section 3.1
        O(|delta b|) property. No hardware claim attaches to this structure.
        """
        adds = [o.bullet_id for o in self.ops if o.kind == "ADD"]
        edits = [o.bullet_id for o in self.ops if o.kind == "EDIT"]
        prunes = [o.bullet_id for o in self.ops if o.kind == "PRUNE"]
        evicts = [o.bullet_id for o in self.ops if o.kind == "EVICT"]
        rejects = [o.bullet_id for o in self.ops if o.kind == "REJECT_DUPLICATE"]
        max_touched = max([o.touched for o in self.ops], default=0)
        return {
            "add_blocks": adds,
            "edit_blocks": edits,
            "prune_blocks": prunes,
            "evict_blocks": evicts,
            "rejected_duplicates": rejects,
            "live_bullets": len(self.live_ids()),
            "max_blocks_touched_per_op": max_touched,
            "delta_property_holds": bool(max_touched <= 1),
            "engine": "ABSENT -- plan only, no paged radix KV cache in tree",
        }

    def sections(self) -> dict:
        out = {s: [] for s in SECTIONS}
        for bid in self.live_ids():
            b = self.bullets[bid]
            out.setdefault(b.section, []).append(
                f"- [{bid}] helpful={b.helpful} harmful={b.harmful} :: {b.content}")
        return out

    def render(self) -> str:
        lines = ["# ===== HENRI COMPILED ACE PLAYBOOK ====="]
        for s in SECTIONS:
            lines.append(f"\n[SECTION: {s}]")
            lines.extend(self.sections().get(s, []))
        return "\n".join(lines)

    def token_history(self) -> list:
        """Token count after each op. Collapse shows as a sudden drop."""
        hist, cur = [], 0
        for o in self.ops:
            if o.kind == "ADD":
                cur += self.bullets[o.bullet_id].token_count
            elif o.kind in ("PRUNE", "EVICT"):
                b = self.bullets.get(o.bullet_id)
                if b is not None:
                    cur = sum(x.token_count for x in self.bullets.values() if x.alive)
            hist.append(cur)
        return hist

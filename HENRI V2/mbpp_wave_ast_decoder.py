"""Wave -> AST structural decoder (non-autoregressive) for the MBPP pilot.

Measured design constraint (run13 debug, 2026-08-02): slot-level wave
scoring (encode(op-name) vs the predicted wave) is NOISE at qFHRR scale
(cosines 0.02-0.08 — fragments are orthogonal to full-program waves).
The signal that works is FULL-PROGRAM wave ranking (run11d; the exact
solution wave ranks #1 when it IS the prediction source, sim ~ 1.0).

Therefore the decoder is: a bounded, MBPP-simple AST grammar enumerator
(return-expression programs over the item's parsed signature) whose
candidates are ranked by transformation-relative full-program wave
similarity and verified by CEGIS in the sandbox. The wave guides
selection; the grammar bounds expression; the sandbox decides.

Non-autoregressive: candidates are enumerated in full and scored in
parallel from the SAME predicted wave. No token-by-token generation.

Honest boundary: this grammar expresses single-return solutions
(unary/binary collection ops, slices, int-conversion, list
comprehensions). DP/regex/heapq multi-statement solutions are beyond it;
the external run measures what this space covers.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Optional

import torch

# ---- grammar: expression templates, {a}=first arg, {b}=second arg ----
EXPRS_UNARY = [
    "sorted({a})", "sum({a})", "len({a})", "min({a})", "max({a})",
    "list({a})", "tuple({a})", "set({a})", "reversed({a})", "abs({a})",
    "str({a})", "int({a})", "list(reversed({a}))", "sorted({a}, reverse=True)",
    "{a}[::-1]", "{a}[0]", "{a}[-1]",
]
EXPRS_UNARY_COMP = [
    "[x for x in {a}]", "[x ** 2 for x in {a}]",
    "sum([x ** 2 for x in {a}])", "[len(x) for x in {a}]",
    "[x for x in {a} if x > 0]",
]
EXPRS_BINARY = [
    "{a} + {b}", "{a} - {b}", "{a} * {b}", "{a} // {b}", "{a} % {b}",
    "abs({a} - {b})", "set({a}) & set({b})", "set({a}) | set({b})",
    "sorted({a} + {b})", "len({a}) + len({b})", "{a}.count({b})",
    "{a}.index({b})", "{a}[:{b}]", "{a}[{b}:]", "sorted({a})[:{b}]",
    "{a}[:len({b})]",
]
EXPRS_CONST = [
    "int({a}, 2)", "int({a}, 16)", "{a} ** 2", "{a} + 1", "{a} - 1",
    "{a} * 2", "len({a}) + 1",
]

# ---- multi-statement body templates (bounded; each complete + parseable) ----
# {entry} is the item's function name (needed by the recursion shape).
IFELSE_CONDS = [
    "{a} > {b}", "{a} < {b}", "{a} == {b}", "{a} >= {b}",
    "len({a}) < len({b})", "len({a}) > len({b})",
]
IFELSE_BRANCHES = [
    ("{a}", "{b}"),
    ("sorted({a})", "sorted({b})"),
    ("sum({a})", "sum({b})"),
    ("len({a})", "len({b})"),
]
LOOP_APPENDS = [
    "x", "x ** 2", "len(x)", "x[0] if x else None",
]
LOOP_SUMS = ["x", "x ** 2"]
COUNT_CONDS = ["x > 0", "x < 0", "x % 2 == 0"]


def _ifelse_bodies(a: str, b: str) -> list[str]:
    out = []
    for cond in IFELSE_CONDS:
        for br1, br2 in IFELSE_BRANCHES:
            out.append(
                f"    if {cond.format(a=a, b=b)}:\n"
                f"        return {br1.format(a=a, b=b)}\n"
                f"    return {br2.format(a=a, b=b)}")
    return out


def _loop_bodies(a: str) -> list[str]:
    out = []
    for expr in LOOP_APPENDS:
        out.append(
            f"    result = []\n"
            f"    for x in {a}:\n"
            f"        result.append({expr})\n"
            f"    return result")
    for expr in LOOP_SUMS:
        out.append(
            f"    t = 0\n"
            f"    for x in {a}:\n"
            f"        t += {expr}\n"
            f"    return t")
    for cond in COUNT_CONDS:
        out.append(
            f"    c = 0\n"
            f"    for x in {a}:\n"
            f"        if {cond}:\n"
            f"            c += 1\n"
            f"    return c")
    return out


def _recursive_bodies(a: str, entry: str) -> list[str]:
    return [
        f"    if {a} <= 1:\n        return {a}\n"
        f"    return {a} * {entry}({a} - 1)",
        f"    if {a} == 0:\n        return 1\n"
        f"    return {a} * {entry}({a} - 1)",
    ]


def _index_bodies(a: str, b: str) -> list[str]:
    return [
        f"    for i in range(len({a})):\n"
        f"        if {a}[i] == {b}:\n"
        f"            return i\n"
        f"    return -1",
    ]


class WaveASTDecoder:
    """Grammar-enumerating wave-guided AST decoder (single-return programs)."""

    def __init__(self, codec, device: str = "cuda"):
        self.codec = codec
        self.device = device

    def _wave(self, text: str) -> torch.Tensor:
        ring = self.codec.encode_text(text).to(torch.float32)
        return torch.nn.functional.normalize(
            (ring / (self.codec.k_bins - 1) * 2.0 - 1.0).view(-1).to(self.device), p=2, dim=0)

    def _instantiate(self, entry: str, args: list[str]) -> list[str]:
        """Return BODY strings (already indented) for the item's signature."""
        bodies: list[str] = []
        if len(args) >= 2:
            a0, a1 = args[0], args[1]
            for t in EXPRS_UNARY:
                bodies.append(f"    return {t.format(a=a0)}")
                bodies.append(f"    return {t.format(a=a1)}")
            for t in EXPRS_BINARY:
                bodies.append(f"    return {t.format(a=a0, b=a1)}")
            for t in EXPRS_UNARY_COMP:
                bodies.append(f"    return {t.format(a=a0)}")
            for t in EXPRS_CONST:
                bodies.append(f"    return {t.format(a=a0)}")
            bodies.extend(_ifelse_bodies(a0, a1))
            bodies.extend(_loop_bodies(a0))
            bodies.extend(_loop_bodies(a1))
            bodies.extend(_recursive_bodies(a0, entry))
            bodies.extend(_index_bodies(a0, a1))
        elif len(args) == 1:
            a0 = args[0]
            for t in EXPRS_UNARY:
                bodies.append(f"    return {t.format(a=a0)}")
            for t in EXPRS_UNARY_COMP:
                bodies.append(f"    return {t.format(a=a0)}")
            for t in EXPRS_CONST:
                bodies.append(f"    return {t.format(a=a0)}")
            bodies.extend(_loop_bodies(a0))
            bodies.extend(_recursive_bodies(a0, entry))
        else:
            return []
        # dedupe, preserve order
        seen: set[str] = set()
        out: list[str] = []
        for b in bodies:
            if b not in seen:
                seen.add(b)
                out.append(b)
        return out

    def decode(
        self, pred_wave: torch.Tensor, prompt_wave: torch.Tensor,
        entry: str, args: list[str],
        manifold_proj: Optional[torch.Tensor] = None,
        complexity_lambda: float = 0.15,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Enumerate the grammar under the item signature, rank every complete
        program by transformation-relative wave similarity MINUS a structural-
        complexity penalty (run15: the run14 exemplar-bias correction).

        manifold_proj: the R-EDMD low-rank basis V ([D, r], orthonormal
        columns) — the task-manifold bottleneck. Candidates whose phase
        geometry is off-manifold (high residual energy ||(I - VV^T) psi||)
        incur the proportional penalty, restoring selection fidelity against
        exemplar-biased multi-statement distractors."""
        prompt_wave = torch.nn.functional.normalize(
            prompt_wave.view(-1).to(torch.float32), p=2, dim=0)
        pn = torch.nn.functional.normalize(
            pred_wave.view(-1).to(torch.float32) - prompt_wave, p=2, dim=0)
        proj = None
        if manifold_proj is not None:
            proj = manifold_proj.view(-1, manifold_proj.shape[-1]).to(torch.float32).to(self.device)

        candidates: list[tuple[str, dict[str, Any]]] = []
        for body in self._instantiate(entry, args):
            src = f"def {entry}({', '.join(args)}):\n{body}"
            try:
                ast.parse(src)
            except SyntaxError:
                continue
            candidates.append((src, {"decoder": True, "body": body.splitlines()[0].strip()}))

        scored = []
        for src, meta in candidates:
            v = self._wave(src)
            v_rel = v - prompt_wave * torch.dot(v, prompt_wave).clamp(min=0.0)
            v_rel = torch.nn.functional.normalize(v_rel, p=2, dim=0)
            sim = float(torch.dot(v_rel, pn).item())
            penalty = 0.0
            if proj is not None:
                # structural-complexity penalty: off-manifold residual energy,
                # dimension-normalized per the architecture invariant
                # (L2 / sqrt(d)).
                coeffs = v @ proj  # [r]
                on_manifold = coeffs @ proj.t()  # [D]
                resid = torch.norm(v - on_manifold) / math.sqrt(v.numel())
                penalty = float(complexity_lambda * resid)
            scored.append((src, meta, sim - penalty))
        scored.sort(key=lambda t: t[2], reverse=True)
        return [(src, meta) for src, meta, _ in scored]

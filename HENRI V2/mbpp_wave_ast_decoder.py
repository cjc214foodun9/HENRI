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
from typing import Any

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
        """Return expression strings for the item's signature (arity-aware)."""
        exprs: list[str] = []
        if len(args) >= 2:
            a0, a1 = args[0], args[1]
            for t in EXPRS_UNARY:
                exprs.append(t.format(a=a0))
                exprs.append(t.format(a=a1))
            for t in EXPRS_BINARY:
                exprs.append(t.format(a=a0, b=a1))
            for t in EXPRS_UNARY_COMP:
                exprs.append(t.format(a=a0))
            for t in EXPRS_CONST:
                exprs.append(t.format(a=a0))
        elif len(args) == 1:
            a0 = args[0]
            for t in EXPRS_UNARY:
                exprs.append(t.format(a=a0))
            for t in EXPRS_UNARY_COMP:
                exprs.append(t.format(a=a0))
            for t in EXPRS_CONST:
                exprs.append(t.format(a=a0))
        else:
            return []
        # dedupe, preserve order
        seen: set[str] = set()
        out: list[str] = []
        for e in exprs:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def decode(
        self, pred_wave: torch.Tensor, prompt_wave: torch.Tensor,
        entry: str, args: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Enumerate the grammar under the item signature, rank every complete
        program by transformation-relative wave similarity, return the list."""
        prompt_wave = torch.nn.functional.normalize(
            prompt_wave.view(-1).to(torch.float32), p=2, dim=0)
        pn = torch.nn.functional.normalize(
            pred_wave.view(-1).to(torch.float32) - prompt_wave, p=2, dim=0)

        candidates: list[tuple[str, dict[str, Any]]] = []
        for expr in self._instantiate(entry, args):
            src = f"def {entry}({', '.join(args)}):\n    return {expr}"
            try:
                ast.parse(src)
            except SyntaxError:
                continue
            candidates.append((src, {"decoder": True, "expr": expr}))

        scored = []
        for src, meta in candidates:
            v = self._wave(src)
            v_rel = v - prompt_wave * torch.dot(v, prompt_wave).clamp(min=0.0)
            v_rel = torch.nn.functional.normalize(v_rel, p=2, dim=0)
            scored.append((src, meta, float(torch.dot(v_rel, pn).item())))
        scored.sort(key=lambda t: t[2], reverse=True)
        return [(src, meta) for src, meta, _ in scored]

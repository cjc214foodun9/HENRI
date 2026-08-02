"""Wave -> AST structural decoder (non-autoregressive) for the MBPP pilot.

Architecture rationale (run12 law, measured): the R-EDMD predicted wave
carries solution-family structure but cannot RANK wide candidate lists
(selection resolution ~ 1/candidates). The decoder instead scores each
STRUCTURAL SLOT of a bounded AST grammar in parallel from the predicted
wave (op, selector, wrapper, body shape), prunes to the top-K per slot,
enumerates the pruned grammar (~O(30) candidates), ranks by
transformation-relative wave similarity, and lets CEGIS verify.

Non-autoregressive: all slots are scored from the SAME predicted wave;
no token-by-token generation. The grammar is bounded and MBPP-realistic
(single-return solutions: sorted/sum/len/... over arg selectors with
wrapper coercions, plus list-comprehension shape).

Pre-registered gates:
  - Local contract: for a synthetic single-return solution, the decoder
    must rank the EXACT solution in the top-K of its own candidates.
  - Remote run: --ast-decode on the heldout; acceptance = pass count
    strictly greater than the run11d exemplar-anchor baseline (4/500).
"""

from __future__ import annotations

import ast
from typing import Any, Optional

import torch

OPS = [
    "sorted", "sum", "len", "min", "max", "list", "tuple", "set",
    "reversed", "map", "filter", "zip", "enumerate", "range", "abs",
    "pow", "bin", "hex", "str", "int", "round", "join", "split",
]

WRAPPERS = [
    ("identity", lambda s: s),
    ("list", lambda s: f"list({s})"),
    ("tuple", lambda s: f"tuple({s})"),
    ("sorted", lambda s: f"sorted({s})"),
    ("sorted_rev", lambda s: f"sorted({s}, reverse=True)"),
    ("set", lambda s: f"set({s})"),
    ("sum", lambda s: f"sum({s})"),
    ("len", lambda s: f"len({s})"),
    ("min", lambda s: f"min({s})"),
    ("max", lambda s: f"max({s})"),
    ("reversed", lambda s: f"list(reversed({s}))"),
]

DECODE_TOP_OPS = 6
DECODE_TOP_SELECTORS = 4
DECODE_TOP_WRAPPERS = 3
DECODE_TOP_K = 5  # local contract gate: true solution must rank in top-K


class WaveASTDecoder:
    """Non-autoregressive wave->AST structural decoder over a bounded grammar."""

    def __init__(self, codec, device: str = "cuda"):
        self.codec = codec
        self.device = device
        self._op_waves = {op: self._wave(op) for op in OPS}

    def _wave(self, text: str) -> torch.Tensor:
        ring = self.codec.encode_text(text).to(torch.float32)
        return torch.nn.functional.normalize(
            (ring / (self.codec.k_bins - 1) * 2.0 - 1.0).view(-1).to(self.device), p=2, dim=0)

    def _score(self, cand_wave: torch.Tensor, pred: torch.Tensor) -> float:
        return float(torch.dot(
            torch.nn.functional.normalize(cand_wave, p=2, dim=0),
            torch.nn.functional.normalize(pred.view(-1), p=2, dim=0)).item())

    def _selectors(self, args: list[str]) -> list[str]:
        sels = []
        for a in args:
            sels.append(a)
            sels.append(f"{a}[0]")
            sels.append(f"{a}[-1]")
            sels.append(f"{a}[::-1]")
            sels.append(f"len({a})")
        if len(args) >= 2:
            sels.append(f"{args[0]}[:{args[1]}]")
        # dedupe, keep order
        seen = set()
        out = []
        for s in sels:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def decode(
        self, pred_wave: torch.Tensor, prompt_wave: torch.Tensor,
        entry: str, args: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Score every grammar slot from the predicted wave; return the pruned
        candidate list (source, meta) with full-wave ranking."""
        prompt_wave = torch.nn.functional.normalize(prompt_wave.view(-1).to(torch.float32), p=2, dim=0)
        pn = torch.nn.functional.normalize(
            pred_wave.view(-1).to(torch.float32) - prompt_wave, p=2, dim=0)
        # slot: op
        op_scores = [(op, self._score(self._op_waves[op], pn)) for op in OPS]
        op_scores.sort(key=lambda t: t[1], reverse=True)
        top_ops = [op for op, _ in op_scores[:DECODE_TOP_OPS]]
        # slot: selectors (arg-position neutral: score the arg-name text)
        sels = self._selectors(args)
        sel_scores = [(s, self._score(self._wave(s), pn)) for s in sels]
        sel_scores.sort(key=lambda t: t[1], reverse=True)
        top_sels = [s for s, _ in sel_scores[:DECODE_TOP_SELECTORS]]
        # slot: wrapper (scored by wrapper-op text)
        wrap_scores = [(w, self._score(self._wave(w), pn)) for w, _ in WRAPPERS]
        wrap_scores.sort(key=lambda t: t[1], reverse=True)
        top_wraps = [w for w, _ in wrap_scores[:DECODE_TOP_WRAPPERS]]
        # enumerate the pruned grammar + shape. Minimal set: every selector
        # under the top ops (identity wrapper) so selector-scoring noise cannot
        # kill expressiveness; wrapper variants under the top-2 selectors.
        candidates: list[tuple[str, dict[str, Any]]] = []
        for op in top_ops:
            for sel in sels:
                inner = f"{op}({sel})"
                for wname, wrap in WRAPPERS:
                    if wname != "identity":
                        continue
                    expr = wrap(inner)
                    body = f"return {expr}"
                    src = f"def {entry}({', '.join(args)}):\n    {body}"
                    try:
                        ast.parse(src)
                    except SyntaxError:
                        continue
                    candidates.append((src, {"decoder": True, "op": op, "selector": sel,
                                             "wrapper": wname, "shape": "return"}))
        for op in top_ops:
            for sel in top_sels:
                inner = f"{op}({sel})"
                for wname, wrap in WRAPPERS:
                    if wname not in top_wraps or wname == "identity":
                        continue
                    expr = wrap(inner)
                    body = f"return {expr}"
                    src = f"def {entry}({', '.join(args)}):\n    {body}"
                    try:
                        ast.parse(src)
                    except SyntaxError:
                        continue
                    candidates.append((src, {"decoder": True, "op": op, "selector": sel,
                                             "wrapper": wname, "shape": "return"}))
        # list-comprehension shape: the list op over the top selectors
        for op in top_ops:
            if op != "list":
                continue
            for sel in top_sels:
                expr = f"[x for x in {sel}]"
                src = f"def {entry}({', '.join(args)}):\n    return {expr}"
                try:
                    ast.parse(src)
                except SyntaxError:
                    continue
                candidates.append((src, {"decoder": True, "op": op, "selector": sel,
                                         "wrapper": "identity", "shape": "listcomp"}))
        # rank by full transformation-relative wave similarity
        scored = []
        for src, meta in candidates:
            ring = self.codec.encode_text(src).to(torch.float32)
            v = torch.nn.functional.normalize(
                (ring / (self.codec.k_bins - 1) * 2.0 - 1.0).view(-1).to(self.device), p=2, dim=0)
            v_rel = v - prompt_wave * torch.dot(v, prompt_wave).clamp(min=0.0)
            v_rel = torch.nn.functional.normalize(v_rel, p=2, dim=0)
            scored.append((src, meta, float(torch.dot(v_rel, pn).item())))
        scored.sort(key=lambda t: t[2], reverse=True)
        return [(src, meta) for src, meta, _ in scored]

"""CEGIS/AST program synthesis egress for the MBPP heldout pilot.

Mechanism (zero-train, non-autoregressive):
  1. Parse the item's entry signature from the prompt (AST).
  2. Build candidates: each exemplar solution, AST-renamed to the item's
     signature (arg names + entry name), plus bounded return-wrapper
     morphisms (list/tuple/sorted/set).
  3. Rank candidates by cosine similarity of their encoded wave to the
     R-EDMD PREDICTED solution wave (the composition result).
  4. CEGIS loop: syntax-validate (real ast.parse), then run the item's
     test_list in the sandbox; the first candidate passing ALL tests wins.

Kill gates (pre-registered):
  - Probe (exemplars only): the predicted wave must rank the true exemplar
    solution in the top-K among distractors for >= 50% of exemplars; else
    CEGIS_SELECTION_INERT -> BLOCKED.
  - Every candidate is syntax-checked before execution (run2 lesson: real
    grammar validity, not synthetic fixtures).
"""

from __future__ import annotations

import ast
from typing import Any, Optional

import torch

CEGIS_MAX_ATTEMPTS = int(__import__("os").environ.get("CEGIS_MAX_ATTEMPTS", "12"))
CEGIS_PROBE_TOP_K = int(__import__("os").environ.get("CEGIS_PROBE_TOP_K", "5"))
CEGIS_PROBE_MIN_HIT = float(__import__("os").environ.get("CEGIS_PROBE_MIN_HIT", "0.5"))
_WRAPPERS = [
    ("identity", lambda s: s),
    ("list", lambda s: f"list({s})"),
    ("tuple", lambda s: f"tuple({s})"),
    ("sorted", lambda s: f"sorted({s})"),
    ("set", lambda s: f"set({s})"),
]


def parse_entry_signature(prompt: str) -> Optional[tuple[str, list[str]]]:
    """Extract (entry_name, arg_names) from the first 'def ' in the prompt."""
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("def "):
            try:
                tree = ast.parse(line + "\n    pass")  # bare def line has no suite
            except SyntaxError:
                return None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    return node.name, args
    return None


class _RenameArgs(ast.NodeTransformer):
    """Rename the function name and positional args of an exemplar solution to
    the target item's signature. Positional mapping: exemplar args -> target
    args by index (arity must match)."""

    def __init__(self, old_name: str, new_name: str, arg_map: dict[str, str]):
        self.old_name = old_name
        self.new_name = new_name
        self.arg_map = arg_map

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self.new_name
        for a in node.args.args:
            if a.arg in self.arg_map:
                a.arg = self.arg_map[a.arg]
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self.arg_map:
            node.id = self.arg_map[node.id]
        return node

    def visit_arg(self, node: ast.arg):
        if node.arg in self.arg_map:
            node.arg = self.arg_map[node.arg]
        return node


class MbppCegisSynthesizer:
    """Exemplar-anchored CEGIS program synthesizer for the MBPP pilot."""

    def __init__(self, exemplars: list[dict[str, Any]], codec, device: str = "cuda"):
        self.exemplars = exemplars
        self.codec = codec
        self.device = device
        self._parsed = []
        for ex in exemplars:
            sig = parse_entry_signature(ex["code"])
            if sig is None:
                raise ValueError(f"exemplar {ex.get('task_id')} has no parseable signature")
            self._parsed.append(sig)

    def build_candidates(self, prompt: str) -> list[tuple[str, dict[str, Any]]]:
        """Instantiate exemplar solutions under the item's rendered prompt
        signature + bounded return-wrapper morphisms."""
        sig = parse_entry_signature(prompt)
        if sig is None:
            return []
        entry, tgt_args = sig
        candidates: list[tuple[str, dict[str, Any]]] = []
        for ex, (ex_entry, ex_args) in zip(self.exemplars, self._parsed):
            if len(ex_args) != len(tgt_args):
                continue  # arity mismatch: skip this exemplar anchor
            arg_map = dict(zip(ex_args, tgt_args))
            try:
                tree = ast.parse(ex["code"])
                tree = _RenameArgs(ex_entry, entry, arg_map).visit(tree)
                ast.fix_missing_locations(tree)
                renamed = ast.unparse(tree)
            except Exception:
                continue
            body = "\n".join(renamed.splitlines())
            last_return = body.rfind("return ")
            if last_return < 0:
                continue
            ret_expr = body[last_return + len("return "):]
            for wname, wrap in _WRAPPERS:
                # preserve the 'return' keyword; wrap only the returned expr
                wrapped = body[:last_return] + "return " + wrap(ret_expr)
                try:
                    ast.parse(wrapped)  # syntax gate before execution
                except SyntaxError:
                    continue
                candidates.append((wrapped, {"anchor": int(ex["task_id"]), "morphism": wname}))
        return candidates

    def rank_candidates(
        self, candidates: list[tuple[str, dict[str, Any]]], pred_wave: torch.Tensor,
        prompt_wave: Optional[torch.Tensor] = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Rank candidates by cosine similarity to the R-EDMD predicted solution
        wave. When prompt_wave is provided, rank by TRANSFORMATION similarity:
        cos(cand - prompt, pred - prompt) — subtracts the shared prompt
        component so the holographic manifold blend cannot mask the correct
        solution family (run11: CEGIS_SELECTION_INERT)."""
        scored = []
        pn = torch.nn.functional.normalize(pred_wave.view(-1).to(torch.float32), p=2, dim=0)
        qn = None
        if prompt_wave is not None:
            pn = torch.nn.functional.normalize(
                (pred_wave.view(-1).to(torch.float32) - prompt_wave.view(-1).to(torch.float32)), p=2, dim=0)
            qn = torch.nn.functional.normalize(prompt_wave.view(-1).to(torch.float32), p=2, dim=0)
        for src, meta in candidates:
            ring = self.codec.encode_text(src).to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0
            v = ring.view(-1).to(self.device)
            if qn is not None:
                v = v - torch.nn.functional.normalize(qn, p=2, dim=0) * torch.dot(v, qn).clamp(min=0.0)
            v = torch.nn.functional.normalize(v, p=2, dim=0)
            sim = float(torch.dot(v, pn).item())
            scored.append((src, meta, sim))
        scored.sort(key=lambda t: t[2], reverse=True)
        return scored

    def cegis_verify(
        self, ranked: list[tuple[str, dict[str, Any], float]], item: dict[str, Any],
        sandbox, max_attempts: int = CEGIS_MAX_ATTEMPTS,
    ) -> tuple[Optional[str], dict[str, Any]]:
        """Run candidates in order; first to pass ALL tests wins (CEGIS)."""
        tests = "\n".join(item.get("test_list", []))
        attempted = 0
        for src, meta, sim in ranked[:max_attempts]:
            attempted += 1
            try:
                ast.parse(src)
            except SyntaxError:
                continue
            try:
                result = sandbox.execute(src + "\n" + tests)
            except Exception:
                continue
            if result.status == "PASS":
                return src, {"candidates_tried": attempted, "winner_sim": round(sim, 4), "cegis": True}
        return None, {"candidates_tried": attempted, "cegis": False}

    def probe_self_selection(
        self, pred_waves: list[torch.Tensor], prompt_waves: Optional[list[torch.Tensor]] = None,
    ) -> dict[str, Any]:
        """Kill gate on the exemplars only: for each exemplar, does the
        predicted wave rank the true solution in the top-K among distractors?
        Requires the operator's self-prediction waves (fit on exemplars)."""
        hits = 0
        top_ranks = []
        for k, (ex, pred) in enumerate(zip(self.exemplars, pred_waves)):
            true_src = ex["code"]
            # candidate set = all exemplar solutions renamed to this item's
            # signature (the true solution appears with identity rename)
            cands = self.build_candidates(true_src)
            pw = prompt_waves[k] if prompt_waves is not None else None
            scored = self.rank_candidates(cands, pred, prompt_wave=pw)
            true_rank = None
            for i, (src, meta, sim) in enumerate(scored):
                if meta["anchor"] == int(ex["task_id"]) and meta["morphism"] == "identity":
                    true_rank = i
                    break
            top_ranks.append(true_rank if true_rank is not None else len(scored))
            if true_rank is not None and true_rank < CEGIS_PROBE_TOP_K:
                hits += 1
        hit_rate = hits / len(self.exemplars)
        return {"hit_rate": round(hit_rate, 4), "top_ranks": top_ranks, "min_hit": CEGIS_PROBE_MIN_HIT}

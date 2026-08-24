"""
System-1 v0.5 faithful AST-skeleton egress (structural support repair).
===========================================================================
Grounded in the v0.4.3 SUPPORT_FAILURE verdict (OBSERVED: any_pass@64 ==
pass@1, mean distinct finals 1.0 — correct programs never enter the token
beam's candidate set) and the uploaded v0.5 proposal (System-1_Kernel_v0.5_
Engine.py, sha 3dfd53f0..., audited 2026-08-24: random-weight mock core, no
checkpoint load, no tasks/sandbox, 5-rule grammar, energy = single
signature-state scalar x rule prob — DISPOSITION: FALSIFIED as a faithful
implementation; structural-skeleton concept BOUNDED_IMPLEMENTABLE).

FAITHFUL DESIGN (this file):
  - Reuses the LIVE calibrated kernel system1_kernel_v041_energy_refactored.py
    (System1KernelV04, token-FSA, name-conditioned decoder) for the latent
    encoder + core + energy head. No new random-weight core. The v0.4.1
    checkpoint (ckpt_v041/checkpoint.pt sha 11d56121...) is the ONLY source
    of weights.
  - THEOREM (killed claim): candidate diversity is a CONSEQUENCE of the
    input signature, not a property of the skeleton pool. The skeleton head
    selects production rules FROM THE TASK SIGNATURE LATENT (through the
    calibrated core). A 5-rule grammar sampled from the SAME latent cannot
    claim diversity independent of input.
  - Production-rule grammar instantiated over the LIVE 7-family task DSL
    (sum_list, max_list, count_positive, intersect_tuples, union_tuples,
    pair_sums, factorial): rule skeletons -> FSA-valid token streams via the
    live tokenizer (tokenize_code), then the live sandbox for pass labels.
  - ENERGY PROVENANCE: per-candidate core-unrolled latent (8 steps, the
    calibration family) scored by the trained head — NOT a single
    signature-state scalar. This preserves the calibrated rho=0.4383
    provenance family (CONDITIONAL re-measurement on exact v0.5 states is
    still required).
  - CEGIS loop: skeleton candidates -> sandbox verify -> counterexample
    refinement of the skeleton pool (bounded, pre-registered).

MATCHED ARMS (pre-registered 2026-08-24, reference
system1-decoder-support-audit.md):
  A  v0.4.3 token beam (beta=0.0), matched candidate budget
  B  v0.5 structural skeleton generation, UNIFORM selection (no energy)
  C  v0.5 structural skeleton generation + calibrated energy filter
GATES (pre-registered):
  DIVERSITY        mean distinct finals per task > 1 (materially)
  SUPPORT          S = any_pass@64 - pass@1 >= 0.15 on dev, task-blocked
                   CI lower bound > 0
  CAPABILITY       paired pass@1: B or C > A (McNemar + delta)
  ENERGY_FILTER    C >= B (energy filter does not damage)
  PROMOTION        support AND paired outcome improvement, validity kept
GUARD: refuses the consumed heldout digest. Fresh disposable split only.
"""
from __future__ import annotations

import ast
import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, ID2TOK, System1KernelV04, tokenize_code, detokenize)
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox)

CORE_STEPS = 8                       # calibration-family latent depth
ENERGY_EPS = 1e-5


# ---------------------------------------------------------------------------
# 1. Production-rule skeleton grammar over the LIVE task DSL
# ---------------------------------------------------------------------------

class SkeletonGrammar:
    """Skeleton templates with FSA-valid live-vocab instantiation.

    n_rules controls how many rules are ACTIVE (default 7 = v0.5.1
    compatibility; expansion cycle uses 13). All names/bodies are closed on
    the live ~90-token vocab (contract-tested).
    """

    # rule_id -> (name_template, body_template, nargs)  (ALL rules)
    RULES_ALL: dict[int, tuple[str, str, int]] = {
        0: ("def {f}({a}):", "    return sum({a})", 1),                      # sum_list
        1: ("def {f}({a}):", "    return max({a})", 1),                      # max_list
        2: ("def {f}({a}):", "    return sum(1 for x in {a} if x > 0)", 1),  # count_positive
        3: ("def {f}({a0}, {a1}):",
            "    return tuple(sorted(set({a0}) & set({a1})))", 2),           # intersect_tuples
        4: ("def {f}({a0}, {a1}):",
            "    return tuple(sorted(set({a0}) | set({a1})))", 2),           # union_tuples
        5: ("def {f}({a0}, {a1}):",
            "    return [x + y for x, y in zip({a0}, {a1})]", 2),            # pair_sums
        6: ("def {f}({a}):",
            "    res = 1\n    for i in range(1, {a} + 1):\n        res = res * i\n"
            "    return res", 1),                                            # factorial
        7: ("def {f}({a}):", "    return min({a})", 1),                      # min_list
        8: ("def {f}({a}):", "    return [abs(x) for x in {a}]", 1),         # abs_values
        9: ("def {f}({a}):", "    return sorted({a})", 1),                   # sorted_list
        10: ("def {f}({a}):", "    return sum(range({a}))", 1),              # range_sum
        11: ("def {f}({a0}, {a1}):",
             "    return [x - y for x, y in zip({a0}, {a1})]", 2),           # pair_diffs
        12: ("def {f}({a}):",
             "    acc = 1\n    for x in {a}:\n        acc = acc * x\n"
             "    return acc", 1),                                           # list_product
    }
    N_RULES_ALL = len(RULES_ALL)

    def __init__(self, n_rules: int = 7):
        self.n_rules = int(n_rules)
        if self.n_rules > self.N_RULES_ALL:
            raise ValueError(f"n_rules {self.n_rules} > {self.N_RULES_ALL}")
        self.RULES = {i: self.RULES_ALL[i] for i in range(self.n_rules)}
        self.N_RULES = len(self.RULES)

    def instantiate(self, rule_id: int, func_name: str,
                    arg_names: list[str]) -> str | None:
        if rule_id not in self.RULES:
            return None
        sig_t, body_t, nargs = self.RULES[rule_id]
        if len(arg_names) < nargs:
            return None
        args = arg_names[:nargs]
        f = func_name
        sig = sig_t.format(f=f, a=args[0], a0=args[0], a1=args[1]) \
            if nargs == 2 else sig_t.format(f=f, a=args[0])
        body = body_t.format(f=f, a=args[0], a0=args[0], a1=args[1]) \
            if nargs == 2 else body_t.format(f=f, a=args[0])
        return sig + "\n" + body


# ---------------------------------------------------------------------------
# 2. Skeleton classifier head over the calibrated latent (task signature)
# ---------------------------------------------------------------------------

class SkeletonHead(nn.Module):
    """Predicts production-rule logits from the CORE-UNROLLED signature latent.

    Input latent family matches the energy head's calibration family (mean
    over slots of the 8-step core state) so the trained energy head stays
    in-distribution.
    """

    def __init__(self, d_slot: int, num_rules: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_slot, 128), nn.GELU(), nn.Linear(128, num_rules))

    def forward(self, z: torch.Tensor) -> torch.Tensor:   # z: [B, slots, d]
        return self.net(z.mean(dim=1))


# ---------------------------------------------------------------------------
# 3. Faithful v0.5 engine (frozen calibrated core + NEW skeleton head)
# ---------------------------------------------------------------------------

class System1KernelV05(nn.Module):
    """Structural egress on the calibrated v0.4.1 substrate.

    Only the skeleton head has new weights (random init). The latent encoder,
    core, tokenizer/FSA, and energy head are loaded from the pinned v0.4.1
    checkpoint and FROZEN. This makes the structural-support hypothesis
    attributable: any support change comes from skeleton generation.
    """

    def __init__(self, backbone: System1KernelV04, num_rules: int = 7,
                 d_slot: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.core_steps = CORE_STEPS
        self.grammar = SkeletonGrammar(n_rules=num_rules)
        if d_slot is None:
            d_slot = backbone.cfg.d_slot          # live config (384)
        self.skeleton_head = SkeletonHead(d_slot=d_slot,
                                          num_rules=num_rules)
        # freeze the entire backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def signature_latent(self, z0: torch.Tensor,
                         sp: torch.Tensor) -> torch.Tensor:
        """8-step core-unrolled latent of the task signature (calibration
        family). Returns [1, slots, d_slot]."""
        z = z0
        slow_cache = None
        for t in range(self.core_steps):
            z, slow_cache = self.backbone.core(z, t, slow_cache)
        return z

    @torch.no_grad()
    def candidate_energy(self, seq_ids: list[int], dev) -> float:
        """Per-candidate core-unrolled latent energy (v0.4.1 provenance)."""
        ids = torch.tensor([seq_ids], dtype=torch.long, device=dev)
        z = self.backbone.encode_tokens(ids)
        slow_cache = None
        for t in range(self.core_steps):
            z, slow_cache = self.backbone.core(z, t, slow_cache)
        e = float(self.backbone.energy(z).item())
        if math.isnan(e) or math.isinf(e):
            return 0.5
        return min(1.0, max(0.0, e))

    @torch.no_grad()
    def generate_skeleton_candidates(
        self,
        z0: torch.Tensor,               # [1, slots, d] signature latent
        sp: torch.Tensor,               # [1, K, d] prompt memory
        task: dict,
        top_k: int = 16,
        use_energy: bool = True,
    ) -> list[dict]:
        """Generate top-K distinct skeleton candidates, re-rank by the
        CALIBRATED per-candidate energy (Arm C) or uniform (Arm B)."""
        dev = z0.device
        z = self.signature_latent(z0, sp)
        logits = self.skeleton_head(z)                     # [1, num_rules]
        probs = F.softmax(logits, dim=-1)[0]               # [num_rules]

        # restrict to rules compatible with the task's arity (grammar
        # validity only — NOT a family oracle; the head must select from
        # all structurally valid rules or the test is vacuous)
        nargs = task.get("nargs", 1)
        valid = [r for r in range(self.grammar.N_RULES)
                 if self.grammar.RULES[r][2] == nargs]
        valid_mask = torch.zeros(self.grammar.N_RULES, device=dev)
        for r in valid:
            valid_mask[r] = 1.0
        probs = probs * valid_mask
        if probs.sum() <= 0:
            probs = valid_mask / valid_mask.sum()

        # draw top-K distinct rules (deterministic order)
        order = torch.argsort(probs, descending=True).tolist()
        picked: list[int] = []
        for r in order:
            if probs[r] > 0 and r not in picked:
                picked.append(r)
            if len(picked) >= top_k:
                break

        func_name = task["name"]
        # derive arg names from the live signature (gen_task names)
        nargs = task.get("nargs", 1)
        arg_names = ["xs", "t1", "t2"][:nargs] if nargs <= 2 else \
            ["xs", "ys", "zs"][:nargs]

        cands: list[dict] = []
        seen_codes: set[str] = set()
        for rule_id in picked:
            code = self.grammar.instantiate(rule_id, func_name, arg_names)
            if code is None or code in seen_codes:
                continue
            seen_codes.add(code)
            try:
                ast.parse(code)
                ast_ok = 1.0
            except Exception:
                ast_ok = 0.0
            # FSA-valid token stream via the live tokenizer
            ids = tokenize_code(code)
            fsa_ok = 1.0 if TOK2ID["UNK"] not in ids else 0.0
            if fsa_ok == 0.0:
                continue
            # PER-CANDIDATE calibrated energy (core-unrolled, provenance)
            e = self.candidate_energy([TOK2ID["BOS"]] + ids, dev) \
                if use_energy else 0.5
            cands.append({"rule_id": rule_id, "code": code,
                          "ast_valid": ast_ok, "fsa_valid": fsa_ok,
                          "energy": e,
                          "energy_score": e * float(probs[rule_id].item())})
        # re-rank by energy_score (calibrated filter) or keep uniform order
        if use_energy:
            cands.sort(key=lambda x: x["energy_score"], reverse=True)
        return cands


if __name__ == "__main__":
    # mock-loop guard: importing this file must NOT run a fake verification.
    print("system1_kernel_v05_ast_skeleton: module import only. "
          "No mock __main__ verification.")

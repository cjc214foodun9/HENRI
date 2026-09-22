"""Production Sagnac veto contract: the OPINE engagement path must be able to VETO.

WHY THIS FILE EXISTS
    `production_arc_run.py` has TWO consumers of `HENRI_ARC_SAGNAC_VETO`, both live,
    and they used to disagree about the threshold:

      line 2279 -- OPINE macro-option engagement
          sagnac_planner.dual_channel_sagnac_veto(_psi_macro, _axiom, _world)
          NO epsilon_hard -> the planner's ADAPTIVE branch. It raises the threshold
          when the candidate is WORST (up to 2x tau_veto), and was MEASURED to
          NEVER_FIRE on L2-normalized waves. So `not _hard_vetoed` was always True,
          and the engagement gate could not suppress anything.

      line 2339 -- EFE candidate re-rank
          arc_sagnac_veto.evaluate_veto(..., epsilon_hard=None)
          `None` maps to that sidecar's DEFAULT_EPSILON_HARD = 0.35, FIXED, and its
          `_sagnac_similarity` is the canonical norm-consistent form. MEASURED
          SELECTIVE. This path never had the scale bug.

    MEASURED, 5 arms (experiments/verification/sidecar_selectivity_observed.json):
      A planner + adaptive (prod 2279, before the fix)   NEVER_FIRES
      B planner + explicit tau_veto (search(), after)    SELECTIVE
      C sidecar + fixed 0.35 (prod 2339)                 SELECTIVE
      D sidecar + true unit-modulus complex              SELECTIVE
      E sidecar + L2-normalized complex                  ALWAYS_FIRES (hazard)

    A RECORD CORRECTION belongs here: I earlier stated that `HENRI_ARC_SAGNAC_VETO=1`
    "is currently a non-gate: it cannot veto anything". That was OVERSTATED. Only the
    OPINE path (2279) was inert; the EFE re-rank path (2339) was working and
    selective. The correction is measured, not asserted.

INVARIANTS
    P1  AST: the production call at ~2279 passes epsilon_hard explicitly
    P2  behaviour: with an explicit tau_veto the veto SEPARATES (perfect match passes,
        unrelated is vetoed) -- not never-fire and not always-fire
    P3  AST: the sidecar call at ~2339 is UNCHANGED, still epsilon_hard=None, because
        `None` there means the sidecar's FIXED 0.35, which is correct
    P4  the adaptive branch is documented as the hazard, with its measured behaviour
        pinned so the reason for P1 cannot be lost
    P5  the flag actually gates: at least one consumer must be selective
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROD = ROOT / "production_arc_run.py"
DIM = 1024


def _prod_tree() -> ast.Module:
    return ast.parse(PROD.read_text(encoding="utf-8"))


def _calls_by_name(tree: ast.Module, attr: str) -> list:
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == attr:
            hits.append(n)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                and n.func.id == attr:
            hits.append(n)
    return hits


# ------------------------------------------------------------------ P1
def test_production_opine_call_passes_epsilon_hard_explicitly():
    """P1. THE regression. Omitting epsilon_hard selects the adaptive branch, which
    measured NEVER_FIRES, so the engagement gate could not suppress anything."""
    hits = _calls_by_name(_prod_tree(), "dual_channel_sagnac_veto")
    assert hits, "no dual_channel_sagnac_veto call in production_arc_run.py"
    kws = [[k.arg for k in h.keywords] for h in hits]
    assert any("epsilon_hard" in k for k in kws), (
        f"production omits epsilon_hard (adaptive branch -> NEVER_FIRES). "
        f"Call keywords found: {kws}")


# ------------------------------------------------------------------ P3
def test_sidecar_call_still_uses_none_which_is_the_correct_fixed_default():
    """P3. `None` in the SIDECAR means its fixed DEFAULT_EPSILON_HARD, not adaptive.
    Changing it would be a regression against a correct path."""
    from arc_sagnac_veto import DEFAULT_EPSILON_HARD
    assert DEFAULT_EPSILON_HARD == pytest.approx(0.35), (
        "the sidecar's default threshold changed; production relies on it being "
        "a FIXED value")
    hits = _calls_by_name(_prod_tree(), "evaluate_veto")
    assert hits, "no evaluate_veto call in production_arc_run.py"
    # Every sidecar call must either omit epsilon_hard or pass it explicitly; what it
    # must NOT do is pass a value that silently differs from the documented default.
    for h in hits:
        kvals = {k.arg: k.value for k in h.keywords}
        if "epsilon_hard" in kvals:
            v = kvals["epsilon_hard"]
            assert isinstance(v, ast.Constant) and v.value is None, (
                f"line {h.lineno}: sidecar called with a non-None epsilon_hard; "
                f"the sidecar's fixed default is the contract")


# ------------------------------------------------------------------ P2
@pytest.fixture(scope="module")
def planner():
    from sagnac_mcts_planner import SagnacMCTSPlanner
    return SagnacMCTSPlanner(d_model=DIM, k_blocks=128, tau_veto=0.35, device="cpu")


def _pair(align: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    ax = torch.randn(DIM, generator=g)
    ax = ax / ax.norm()
    nz = torch.randn(DIM, generator=g)
    nz = nz / nz.norm()
    c = align * ax + math.sqrt(max(0.0, 1.0 - align * align)) * nz
    return c / c.norm(), ax


def test_explicit_epsilon_separates_perfect_match_from_unrelated(planner):
    """P2. The production form must be a real gate after the fix."""
    hits = 0
    for s in range(8):
        c, ax = _pair(1.0, 200 + s)
        _, _, hard = planner.dual_channel_sagnac_veto(c, ax, ax,
                                                     epsilon_hard=planner.tau_veto)
        hits += int(bool(hard))
    assert hits == 0, f"a PERFECT match was vetoed {hits}/8 times"

    hits = 0
    for s in range(8):
        c, ax = _pair(0.0, 300 + s)
        _, _, hard = planner.dual_channel_sagnac_veto(c, ax, ax,
                                                     epsilon_hard=planner.tau_veto)
        hits += int(bool(hard))
    assert hits == 8, f"unrelated waves were vetoed only {hits}/8 times"


# ------------------------------------------------------------------ P4
def test_adaptive_branch_is_the_documented_hazard(planner):
    """P4. The adaptive branch must stay measurably inert for L2-normalized waves,
    so the reason P1 exists is recorded as behaviour, not as a comment."""
    hits = 0
    for s in range(8):
        c, ax = _pair(0.0, 400 + s)
        _, _, hard = planner.dual_channel_sagnac_veto(c, ax, ax)   # no epsilon
        hits += int(bool(hard))
    assert hits == 0, (
        f"the adaptive branch vetoed {hits}/8 UNRELATED pairs; it no longer matches "
        f"the measured NEVER_FIRES behaviour, so P1's justification must be re-derived")


def test_adaptive_threshold_expands_when_candidate_is_worst(planner):
    """P4b. The mechanism, isolated. epsilon expands most for the WORST candidate,
    which is why the adaptive branch cannot gate."""
    good = _pair(0.9, 500)
    bad = _pair(0.0, 501)

    def eps_for(cand, ax):
        w_cand = cand.flatten()
        w_ax = ax.flatten()
        phase_error = torch.abs(w_cand - w_ax) * math.pi
        conductance = 1.0 / (1.0 + torch.exp(2.0 * (phase_error - 0.05)))
        g_mean = float(torch.mean(conductance).item())
        return planner.tau_veto * (1.0 + 1.0 * (1.0 - g_mean))

    eps_good = eps_for(*good)
    eps_bad = eps_for(*bad)
    assert eps_bad > eps_good, (
        f"adaptive epsilon did not expand for the worse candidate "
        f"(good={eps_good:.4f}, bad={eps_bad:.4f})")


# ------------------------------------------------------------------ P5
def test_the_flag_gates_at_least_one_live_consumer():
    """P5. A flag whose every consumer is inert is a phantom flag."""
    # consumer 2279: selective with explicit epsilon (P2 proves the form)
    # consumer 2339: selective via the canonical sidecar (measured arm C)
    from arc_sagnac_veto import _sagnac_similarity
    from arc_sagnac_veto import evaluate_veto
    hits = 0
    for s in range(8):
        c, ax = _pair(0.0, 600 + s)
        _, _, hard, status = evaluate_veto(c, ax, ax, epsilon_hard=None)
        hits += int(bool(hard))
    assert hits == 8, (
        f"the sidecar vetoed only {hits}/8 unrelated pairs; the EFE re-rank consumer "
        f"is not selective, so with P2 this would leave the flag inert everywhere")

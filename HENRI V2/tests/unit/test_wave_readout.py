"""Unit tests for the two-stage wave readout. Synthetic, deterministic, no corpus.

TEST-DESIGN NOTE (earned the hard way this turn)
    Five earlier iterations asserted significance or separation that the fixture
    could not carry. The specific trap here: with gold drawn UNIFORMLY over the tie
    group, argmax (always column 0) scores 1/G and uniform scores 1/G -- they are
    statistically INDISTINGUISHABLE, and no permutation test can separate them. Slot
    bias is only detectable when gold composition correlates with column index. The
    fixtures below therefore encode that property explicitly, and each test asserts
    only what its sample size can support.
"""
from __future__ import annotations

import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# tests/unit/ -> tests/ -> "HENRI V2"
_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_V2, os.path.join(_V2, "henri_core"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_core.wave_readout import (  # noqa: E402
    max_groups, permutation_drift, select)


def _tied_sim():
    """[2, 6]. Row 0: 3-way exact tie at max among cols {0,1,2}.
    Row 1: 2-way exact tie among {3,4}."""
    return torch.tensor([[0.9, 0.9, 0.9, 0.1, 0.2, 0.3],
                         [0.1, 0.2, 0.3, 0.8, 0.8, 0.4]], dtype=torch.float32)


def _slot_biased_bank(B=600, K=12, G=4, seed=0):
    """A bank whose max group is G wide and whose gold is NEVER the first index.

    This is the fixture that expresses SLOT BIAS. On it:
        argmax (column 0)        -> 0.0
        uniform within the group -> 1/G
    """
    torch.manual_seed(seed)
    sim = torch.zeros(B, K)
    sim[:, :G] = 1.0                       # exact G-way tie at the max
    gold = [{int(torch.randint(1, G, (1,)))} for _ in range(B)]
    return sim, gold


def test_max_groups_finds_exact_ties():
    g = max_groups(_tied_sim())
    assert g[0] == [0, 1, 2], g[0]
    assert g[1] == [3, 4], g[1]


def test_argmax_picks_the_first_group_member():
    """Pure PICK property -- no significance claim on a 2-row fixture."""
    assert select(_tied_sim(), mode="argmax") == [0, 3]


def test_slot_biased_bank_separates_argmax_from_uniform():
    """On a slot-biased bank, argmax scores 0 where uniform scores ~1/G."""
    sim, gold = _slot_biased_bank()
    a_arg = permutation_drift(lambda X: select(X, mode="argmax"), sim, gold)
    a_uni = permutation_drift(
        lambda X: select(X, mode="uniform",
                         generator=torch.Generator().manual_seed(7)),
        sim, gold)
    assert a_arg["acc"] == 0.0, a_arg
    assert 0.18 < a_uni["acc"] < 0.32, a_uni         # ~1/4
    assert a_uni["acc"] > a_arg["acc"] + 0.15, (a_arg, a_uni)
    # uniform is content-neutral: permutation drift stays inside noise
    assert a_uni["sigma_basis"] == "observed_p", a_uni
    assert a_uni["drift_in_sigma"] < 3.0, a_uni


def test_uniform_gold_makes_argmax_and_uniform_indistinguishable():
    """HONEST LIMIT: with gold uniform over the group, NO test separates them.

    Both score 1/G. Recorded so a future reader does not think a drift test can
    detect slot bias on a fixture that does not encode one.
    """
    B, K, G = 600, 12, 4
    torch.manual_seed(1)
    sim = torch.zeros(B, K)
    sim[:, :G] = 1.0
    gold = [{int(torch.randint(0, G, (1,)))} for _ in range(B)]
    a_arg = permutation_drift(lambda X: select(X, mode="argmax"), sim, gold)
    a_uni = permutation_drift(
        lambda X: select(X, mode="uniform",
                         generator=torch.Generator().manual_seed(3)),
        sim, gold)
    assert abs(a_arg["acc"] - a_uni["acc"]) < 0.12, (a_arg, a_uni)


def test_degenerate_p_uses_null_rate_sigma():
    """At p in {0,1} the observed-p sigma is zero; the module must fall back."""
    sim, gold = _slot_biased_bank()
    d = permutation_drift(lambda X: select(X, mode="argmax"), sim, gold)
    assert d["acc"] == 0.0
    assert d["sigma_basis"] == "null_rate_p=0.125", d
    assert math.isfinite(d["drift_in_sigma"])


def test_two_stage_recovers_gold_with_a_real_channel():
    S = _tied_sim()
    ch = torch.tensor([[0.1, 0.2, 0.9, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.1, 0.9, 0.0]], dtype=torch.float32)
    picks = select(S, channel=ch, mode="two_stage",
                   generator=torch.Generator().manual_seed(0))
    assert picks == [2, 4], picks


def test_two_stage_does_not_beat_uniform_without_signal():
    """NEGATIVE CONTROL: a channel independent of gold must not beat stage 1."""
    B, K = 300, 24
    sim = torch.zeros(B, K)
    sim[:, 0] = 1.0
    sim[:, 1:] = 0.0                     # every row: a 23-way tie
    gold = [{int(torch.randint(0, K, (1,)))} for _ in range(B)]
    h1 = h2 = 0
    trials = 5
    for t in range(trials):
        g = torch.Generator().manual_seed(t)
        p1 = select(sim, mode="uniform", generator=g)
        noise = torch.rand(B, K, generator=torch.Generator().manual_seed(t + 99))
        p2 = select(sim, channel=noise, mode="two_stage", generator=g)
        h1 += sum(1 for b in range(B) if p1[b] in gold[b])
        h2 += sum(1 for b in range(B) if p2[b] in gold[b])
    a1, a2 = h1 / (B * trials), h2 / (B * trials)
    assert abs(a2 - a1) < 0.05, (a1, a2)


def test_two_stage_requires_channel():
    try:
        select(_tied_sim(), mode="two_stage")
    except ValueError:
        return
    raise AssertionError("two_stage must require a channel")


def test_error_paths():
    S = _tied_sim()
    try:
        select(S.unsqueeze(0), mode="argmax")
    except ValueError:
        pass
    else:
        raise AssertionError("3-D sim must be rejected")
    try:
        select(S, mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode must be rejected")


def test_purge_no_exponential_sharpening_in_module_ast():
    """DIRECTIVE-2 PURGE PIN: exp(s/tau) must not return as a tie-breaker.

    WHY AST AND NOT grep. A grep for "exp(" matches the module DOCSTRING, which
    legitimately discusses the falsified mechanism. The pin must inspect CODE only, so
    it walks the parsed tree and looks for actual Call nodes named exp. A comment or a
    docstring cannot satisfy or defeat this check.
    """
    import ast
    import pathlib
    src_path = pathlib.Path(_V2) / "henri_core" / "wave_readout.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in ("exp", "exp2", "expm1", "softmax", "log_softmax"):
                offenders.append((name, getattr(node, "lineno", -1)))
    assert offenders == [], (
        f"monotone sharpening reintroduced in {src_path.name}: {offenders}. "
        "exp(s/tau) is strictly monotone, so it preserves the maximizer set EXACTLY "
        "and can never split a tie.")
    print("  [purge-1] AST walk found no exp/softmax call in wave_readout.py")


def test_purge_select_signature_has_no_temperature_parameter():
    """STRUCTURAL PIN: the selector must offer no tau/temperature knob at all.

    A behavioural test can be satisfied by a tau that happens to be ignored. Inspecting
    the signature is the tighter contract: reintroducing a temperature parameter is a
    deliberate change that must also update this test.
    """
    import inspect
    params = set(inspect.signature(select).parameters)
    banned = {"tau", "temperature", "temp", "beta", "sharpness"}
    hit = params & banned
    assert hit == set(), f"selector exposes a sharpening parameter: {sorted(hit)}"
    assert params == {"sim", "channel", "mode", "atol", "generator"}, sorted(params)
    print(f"  [purge-2] select() params = {sorted(params)}; no sharpening knob")


def test_purge_negative_control_sharpening_can_only_grow_the_tie_group():
    """NEGATIVE CONTROL: the measured REASON the purge is correct.

    exp(s/tau) is strictly increasing, so a correctly-rounded float exp is monotone
    and the maximizer set is preserved exactly; rounding can only MERGE distinct scores,
    which GROWS the tie group. The tie group can therefore never shrink, and at small
    tau it overflows to inf and swallows the ENTIRE row.

    The assertion is one-directional and falsifiable: for every tau, the sharpened max
    group is a SUPERSET-EQUAL of the reference group, and at least one tau must strictly
    enlarge it. If a future edit made the module rely on this, the growth is the defect.
    """
    S = _tied_sim()
    ref = [set(g) for g in max_groups(S)]
    grew = []
    for tau in (0.038, 0.1, 1.0, 4.0, 0.001):
        sharp = torch.exp(S / tau)
        got = [set(g) for g in max_groups(sharp)]
        for b in range(S.shape[0]):
            assert ref[b] <= got[b], (
                f"tau={tau} row={b}: sharpening SHRANK the max group "
                f"{ref[b]} -> {got[b]}. A monotone map cannot do that.")
        if any(len(got[b]) > len(ref[b]) for b in range(S.shape[0])):
            grew.append(tau)
    assert grew, "no tau enlarged the group; the demonstration measured nothing"
    # the overflow case must swallow whole rows -- the extreme failure of the mechanism
    over = torch.exp(S / 0.001)
    assert any(len(g) == S.shape[1] for g in max_groups(over)), \
        "tau=0.001 should overflow to inf and tie the WHOLE row"
    print(f"  [purge-3] sharpening never shrank a group; grew at tau in {grew};"
          f" tau=0.001 ties all {S.shape[1]} columns")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as exc:                                  # noqa: BLE001
            bad += 1
            print("FAIL", fn.__name__, "->", exc)
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)

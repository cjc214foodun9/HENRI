"""Contract tests — HENRI Successor-Feature Action Scoring (Arm F, default OFF).

Pre-registration: docs/arm_f_sfas_pre_registration.md.
Mechanism: per-action successor features read from the LIVE transition
operator; candidate-specific goal scores; blended EFE re-rank (argmin).

Covers: horizon/gamma bounds, normalization, action-discrimination,
fail-closed identity, reorder semantics, zero-trainable, horizon-1 sanity.
"""

import copy
import math

import pytest
import torch

from henri_successor_feature_scorer import (
    compute_sfas_scores,
    rerank_efe_table,
    successor_feature,
)

D, R = 8, 4


def make_transition(seed: int = 0):
    from efe_planner import LowRankCoupledTransition
    torch.manual_seed(seed)
    return LowRankCoupledTransition(num_blocks=D, block_dim=8, rank=R)


def make_wave(seed: int = 0):
    torch.manual_seed(seed)
    w = torch.randn(D, 8)
    return w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)


class TestHorizonGammaBounds:
    def test_horizon_bounds(self):
        tr = make_transition()
        s, a = make_wave(1), make_wave(2)
        with pytest.raises(ValueError):
            successor_feature(tr, s, a, horizon=0)
        with pytest.raises(ValueError):
            successor_feature(tr, s, a, horizon=5)

    def test_gamma_bounds(self):
        tr = make_transition()
        s, a = make_wave(1), make_wave(2)
        with pytest.raises(ValueError):
            successor_feature(tr, s, a, gamma=1.0)
        with pytest.raises(ValueError):
            successor_feature(tr, s, a, gamma=-0.1)


class TestSuccessorFeature:
    def test_normalized_unit_norm(self):
        tr = make_transition()
        s, a = make_wave(1), make_wave(2)
        psi = successor_feature(tr, s, a, horizon=2, gamma=0.9)
        assert psi is not None
        assert torch.isfinite(psi).all()
        assert torch.allclose(torch.norm(psi, p=2), torch.tensor(1.0), atol=1e-5)

    def test_shape(self):
        tr = make_transition()
        psi = successor_feature(tr, make_wave(1), make_wave(2))
        assert psi.shape == (D, 8)

    def test_horizon_one_is_state(self):
        # horizon=1: psi = phi(s) normalized -> score = cos(s, goal).
        tr = make_transition()
        s, g = make_wave(1), make_wave(3)
        psi = successor_feature(tr, s, make_wave(2), horizon=1, gamma=0.9)
        assert psi is not None
        s_n = s.reshape(-1) / (torch.norm(s.reshape(-1)) + 1e-12)
        g_n = g.reshape(-1) / (torch.norm(g.reshape(-1)) + 1e-12)
        assert torch.allclose(psi.reshape(-1), s_n, atol=1e-5)
        scores = compute_sfas_scores(s, g, {0: make_wave(2)}, tr, horizon=1)
        assert scores is not None
        assert abs(scores[0] - float(torch.dot(s_n, g_n).item())) < 1e-5


class TestActionDiscrimination:
    def test_scores_differ_across_actions(self):
        # THE core contract: the goal score must be CANDIDATE-SPECIFIC.
        # The live operator's action enters through the FHRR bind, so two
        # different action waves must yield different successor features and
        # different goal scores. This is the property arms C/D/E lacked
        # (goal_dist ~ 1.0 for EVERY action).
        tr = make_transition()
        s, g = make_wave(1), make_wave(3)
        a1, a2 = make_wave(10), make_wave(11)
        pred1 = tr(s, a1)
        pred2 = tr(s, a2)
        assert not torch.allclose(pred1, pred2, atol=1e-4), (
            "transition must be action-sensitive for the test to be meaningful")
        scores = compute_sfas_scores(s, g, {1: a1, 2: a2}, tr, horizon=2)
        assert scores is not None
        assert abs(scores[1] - scores[2]) > 1e-6, (
            f"goal scores must discriminate actions, got {scores}")


class TestFailClosed:
    def test_scores_none_keeps_order_byte_identical(self):
        table = [
            {"action": 0, "efe": 0.1},
            {"action": 1, "efe": 0.2},
            {"action": 2, "efe": 0.3},
        ]
        new_table, info = rerank_efe_table(table, None, lambda_sfas=1.0)
        assert [r["action"] for r in new_table] == [0, 1, 2]
        assert info["reordered"] is False
        assert info["discordance"] == 0

    def test_none_inputs(self):
        tr = make_transition()
        assert compute_sfas_scores(None, make_wave(3), {0: make_wave(2)}, tr) is None
        assert compute_sfas_scores(make_wave(1), None, {0: make_wave(2)}, tr) is None
        assert compute_sfas_scores(make_wave(1), make_wave(3), {}, tr) is None


class TestRerank:
    def test_reorders_when_score_favors_lower_efe(self):
        table = [
            {"action": 0, "efe": 0.0},
            {"action": 1, "efe": 1.0},
            {"action": 2, "efe": 2.0},
        ]
        scores = {0: -0.5, 1: 0.9, 2: 0.0}
        new_table, info = rerank_efe_table(table, scores, lambda_sfas=1.0)
        # blended: A=1.5, B=1.1, C=3.0 -> B wins.
        assert [r["action"] for r in new_table] == [1, 0, 2]
        assert info["reordered"] is True
        assert info["discordance"] == 2

    def test_gamescoring_action_enum_key(self):
        # Rows carry GameAction values (int .value); scores keyed by int.
        # Contract: the enum .value must be extracted so the int-keyed
        # score is found. Row 1 (GA(2)) must receive score 0.9; row 0
        # (GA(1)) has no score and keeps its raw EFE (0.0 < 1.1), so the
        # order is preserved but the score lookup is proven.
        class GA:
            def __init__(self, v):
                self.value = v
        table = [
            {"action": GA(1), "efe": 0.0},
            {"action": GA(2), "efe": 1.0},
        ]
        new_table, info = rerank_efe_table(table, {2: 0.9}, lambda_sfas=1.0)
        assert [r["action"].value for r in new_table] == [1, 2]
        assert info["scores"] == [None, 0.9]

    def test_missing_score_keeps_raw_efe(self):
        table = [
            {"action": 0, "efe": 0.0},
            {"action": 1, "efe": 1.0},
        ]
        new_table, info = rerank_efe_table(table, {1: 0.9}, lambda_sfas=1.0)
        # row 0 has no score -> blended stays 0.0 -> still first.
        assert [r["action"] for r in new_table] == [0, 1]
        assert info["scores"] == [None, 0.9]


class TestZeroTrainable:
    def test_does_not_mutate_transition(self):
        tr = make_transition()
        s, g = make_wave(1), make_wave(3)
        a1, a2 = make_wave(10), make_wave(11)
        before = [p.detach().clone() for p in tr.parameters()]
        compute_sfas_scores(s, g, {1: a1, 2: a2}, tr, horizon=2)
        after = [p.detach().clone() for p in tr.parameters()]
        for b, a in zip(before, after):
            assert torch.equal(b, a)

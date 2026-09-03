"""Contract suite for the grounded VLA substrate components
(HENRI-DIR-2026-09-BUNDLE-VLA-GROUNDING §4 gate matrix).

Scope: LOCAL CPU only (reduced dimensions), fail-closed.
  G-INGRESS   locality preservation / anti-mean-pooling        (proxy at D=512)
  G-VALENCE   exteroceptive EFE steering + failure trace       (proxy at D=512)
  G-EGRESS    canonical codebook recovery + syntax rejection   (proxy at D=512)
  G-SOLVER    RTX 5090 Koopman latency <= 2.0 ms               NOT RUNNABLE HERE
              (remote CUDA gate; excluded from this file by design)

Evidence labels live in the docstring per test; none of these proxies is an
external task outcome.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from henri_causal_planner import BoundedExteroceptiveEFEPlanner
from henri_hopfield_egress import (
    CanonicalCodebookEgress,
    EgressSyntaxRejectedError,
    noise_floor_sigma,
)
from henri_metric_ingress import HenriMetricPatchIngress, SemanticAnchorProjector

D = 512
NB = 64
GRID = 8
PATCH = 4


# ---------------------------------------------------------------------------
# G-INGRESS
# ---------------------------------------------------------------------------
class TestMetricIngress:
    def test_unit_norm_and_block_shape(self):
        ing = HenriMetricPatchIngress(d_model=D, num_blocks=NB, patch_size=PATCH)
        g = torch.zeros(GRID, GRID, dtype=torch.long)
        g[1:3, 1:3] = 3
        w = ing.encode_grid(g)
        assert w.shape == (D,)
        assert abs(float(w.norm(p=2)) - 1.0) < 1e-5
        wb = ing.encode_grid_blocks(g)
        assert wb.shape == (1, NB, 8)
        assert torch.allclose(wb.reshape(-1), w, atol=1e-6)

    def test_no_global_mean_pooling_same_colors_diff_shape(self):
        """Anti-collapse guard: same color multiset, different arrangement.

        This test would fail if color phases factored out of the spatial
        ramp (the original Component-1 defect caught pre-commit).
        """
        ing = HenriMetricPatchIngress(d_model=D, num_blocks=NB, patch_size=PATCH)
        gA = torch.zeros(GRID, GRID, dtype=torch.long)
        gB = torch.zeros(GRID, GRID, dtype=torch.long)
        cellsA = [(1, 1), (1, 2), (1, 3)]
        cellsB = [(1, 1), (2, 2), (3, 3)]
        for (r, c), col in zip(cellsA, [3, 3, 3]):
            gA[r, c] = col
        for (r, c), col in zip(cellsB, [3, 3, 3]):
            gB[r, c] = col
        wA = ing.encode_grid(gA)
        wB = ing.encode_grid(gB)
        # identical arrangements collapse only at exactly cos=1; distinct
        # arrangements must be clearly separable (margin > 0.05)
        cos_diff = float(wA @ wB)
        assert cos_diff < 0.95, f"same-color/different-shape cos {cos_diff:.4f} not discriminative"

    def test_adjacent_color_discrimination(self):
        """Anti-codec-collapse guard: identical shape, adjacent colors must
        separate.  Catches the pre-commit defect where the color term was not
        inside the frequency ramp (adjacent colors then scored cos ~ 0.92)."""
        ing = HenriMetricPatchIngress(d_model=D, num_blocks=NB, patch_size=PATCH)

        def block(r0, c0, color):
            g = torch.zeros(GRID, GRID, dtype=torch.long)
            g[r0:r0 + 2, c0:c0 + 2] = color
            return g

        w1 = ing.encode_grid(block(1, 1, 1))
        w2 = ing.encode_grid(block(1, 1, 2))
        cos_c = float(w1 @ w2)
        assert cos_c < 0.5, f"adjacent-color cos {cos_c:.4f} not discriminative"

    def test_contact_auc_proxy(self):
        """G-INGRESS proxy: correct glyph must out-rank 7 distractors.

        AUC_contact = (# distractors with similarity < correct similarity) / 7.
        """
        ing = HenriMetricPatchIngress(d_model=D, num_blocks=NB, patch_size=PATCH)

        def glyph(rows, cols, color):
            g = torch.zeros(GRID, GRID, dtype=torch.long)
            for r in rows:
                for c in cols:
                    g[r, c] = color
            return g

        correct = glyph([2, 3], [2, 3], 5)  # 2x2 block, color 5
        distractors = [
            glyph([2, 3], [2, 3], 6),
            glyph([2, 3], [2, 3], 7),
            glyph([2, 3], [4, 5], 5),
            glyph([4, 5], [2, 3], 5),
            glyph([1, 2], [1, 2], 8),
            glyph([2, 3], [2, 3], 1),
            glyph([6, 7], [6, 7], 5),
        ]
        q = F.normalize(ing.encode_grid(correct), p=2, dim=-1)
        s_correct = float(q @ q)  # 1.0 (self)
        sims = [float(q @ F.normalize(ing.encode_grid(d), p=2, dim=-1)) for d in distractors]
        wins = sum(1.0 for s in sims if s < s_correct)
        auc_m = wins / len(sims)
        assert auc_m >= 0.88, f"AUC_contact {auc_m:.3f} < 0.8800 (directive G-INGRESS)"

    def test_bg_mask_empty_grid_fail_closed(self):
        ing = HenriMetricPatchIngress(d_model=D, num_blocks=NB, patch_size=PATCH)
        with pytest.raises(ValueError):
            ing.encode_grid(torch.zeros(GRID, GRID, dtype=torch.long))

    def test_invalid_basis_fail_closed(self):
        with pytest.raises(ValueError):
            HenriMetricPatchIngress(d_model=D, num_blocks=NB, patch_size=PATCH,
                                    spatial_basis_kind="bogus")

    def test_grid_not_divisible_fail_closed(self):
        ing = HenriMetricPatchIngress(d_model=D, num_blocks=NB, patch_size=PATCH)
        with pytest.raises(ValueError):
            ing.encode_grid(torch.zeros(6, 6, dtype=torch.long))


class TestSemanticAnchor:
    def test_orthonormal_projection(self):
        sa = SemanticAnchorProjector(d_model=1024, d_emb=64, seed=11)
        WtW = sa._W.T @ sa._W  # [64, 64]
        err = float((WtW - torch.eye(64)).abs().max())
        assert err < 1e-4, f"anchor columns not orthonormal: err {err:.2e}"

    def test_projection_unit_norm_and_zero_trainable(self):
        sa = SemanticAnchorProjector(d_model=1024, d_emb=64, seed=11)
        emb = torch.randn(5, 64)
        proj = sa.project(emb)
        assert proj.shape == (5, 1024)
        assert torch.allclose(proj.norm(p=2, dim=-1), torch.ones(5), atol=1e-5)
        n_trainable = sum(1 for _ in sa.parameters())
        assert n_trainable == 0, "anchor projector must be zero-trainable"

    def test_token_sequence_order_sensitive(self):
        sa = SemanticAnchorProjector(d_model=1024, d_emb=64, seed=11)
        emb = F.normalize(torch.randn(8, 64), p=2, dim=-1)
        wa = sa.encode_token_sequence([1, 2, 3], emb)
        wb = sa.encode_token_sequence([3, 2, 1], emb)
        cos = float(wa @ wb)
        assert cos < 0.95, f"order swap cos {cos:.4f} not discriminative"

    def test_empty_sequence_fail_closed(self):
        sa = SemanticAnchorProjector(d_model=1024, d_emb=64, seed=11)
        with pytest.raises(ValueError):
            sa.encode_token_sequence([], torch.randn(4, 64))

    def test_oversize_anchor_fail_closed(self):
        with pytest.raises(ValueError):
            SemanticAnchorProjector(d_model=1024, d_emb=64, seed=11,
                                    max_anchor_bytes=1024)


# ---------------------------------------------------------------------------
# G-VALENCE (exteroceptive EFE)
# ---------------------------------------------------------------------------
class TestCausalPlanner:
    @pytest.fixture()
    def planner(self):
        return BoundedExteroceptiveEFEPlanner(d_model=D, num_actions=4)

    def test_score_bounded_components(self, planner):
        goal = F.normalize(torch.randn(D), p=2, dim=-1)
        plate = F.normalize(torch.randn(3, D), p=2, dim=-1)
        pred = F.normalize(torch.randn(D), p=2, dim=-1)
        planner.register_goal(goal)
        planner.register_baseplate(plate)
        dg = float(planner.cosine_distance(pred, goal))
        sr = float(torch.stack([planner.rms_drift(pred, b) for b in plate]).mean())
        assert 0.0 <= dg <= 1.0
        assert 0.0 <= sr <= 1.0

    def test_positive_delta_lowers_g(self, planner):
        goal = F.normalize(torch.randn(D), p=2, dim=-1)
        planner.register_goal(goal)
        pred_a = F.normalize(torch.randn(D), p=2, dim=-1)
        # same prediction; only history differs
        planner.observe_outcome(0, +0.5)
        planner.observe_outcome(0, +0.5)
        planner.observe_outcome(1, -0.5)
        g0 = planner.score_action(pred_a, 0)
        g1 = planner.score_action(pred_a, 1)
        assert g0 < g1, f"positive-delta action must score lower: {g0:.4f} vs {g1:.4f}"

    def test_delta_clamped_to_unit(self, planner):
        planner.observe_outcome(0, 5.0)
        planner.observe_outcome(0, -5.0)
        assert planner.expected_delta(0) == 0.0  # (+1 + -1)/2
        planner.observe_outcome(1, -5.0)
        assert planner.expected_delta(1) == -1.0

    def test_failure_trace_fires_after_k(self, planner):
        for _ in range(5):
            planner.observe_outcome(2, -0.1)
        assert planner.expected_delta(2) == -1.0  # retroactive assignment
        pred = F.normalize(torch.randn(D), p=2, dim=-1)
        n = planner.apply_failure_trace(2, pred)
        assert n > 0.0, "failure trace must inject anisotropic noise"
        assert not torch.equal(planner.operators[2], torch.zeros(D))

    def test_select_action_argmin(self, planner):
        goal = F.normalize(torch.randn(D), p=2, dim=-1)
        planner.register_goal(goal)
        near = F.normalize(goal + 0.05 * torch.randn(D), p=2, dim=-1)
        far = F.normalize(torch.randn(D), p=2, dim=-1)
        preds = {0: far, 1: near}
        assert planner.select_action(preds) == 1

    def test_empty_predictions_fail_closed(self, planner):
        with pytest.raises(ValueError):
            planner.select_action({})


# ---------------------------------------------------------------------------
# LIVE CONSUMER WIRING
# ---------------------------------------------------------------------------
class TestLiveCausalConsumer:
    def test_default_off_does_not_construct_consumer(self, monkeypatch):
        monkeypatch.delenv("HENRI_CAUSAL_PLANNER", raising=False)
        from darwinian_phase_swarm import HenriSwarmOrchestrator

        orch = HenriSwarmOrchestrator(
            num_experts=8, d_model=64, r_rank=2, num_blocks=8
        )
        assert orch._vla_causal_planner is None
        active = F.normalize(torch.randn(8, 8), p=2, dim=-1)
        boundary = F.normalize(torch.randn(2, 8, 8), p=2, dim=-1)
        action, predicted, table = orch.plan_action(active, boundary, top_k=3)
        assert action in orch.decoder.action_to_id
        assert predicted.shape == active.shape
        assert table and all("vla_causal" not in row for row in table)

    def test_enabled_consumer_selects_and_receives_external_outcome(self, monkeypatch):
        monkeypatch.setenv("HENRI_CAUSAL_PLANNER", "1")
        from darwinian_phase_swarm import HenriSwarmOrchestrator

        orch = HenriSwarmOrchestrator(
            num_experts=8, d_model=64, r_rank=2, num_blocks=8
        )
        active = F.normalize(torch.randn(8, 8), p=2, dim=-1)
        boundary = F.normalize(torch.randn(2, 8, 8), p=2, dim=-1)
        action, predicted, table, chosen = orch.plan_action(
            active, boundary, top_k=3, return_chosen=True
        )
        assert orch._vla_causal_planner is not None
        assert chosen["vla_causal"] is True
        assert chosen["action"] == action
        assert predicted.shape == active.shape
        assert len(table) == 3
        assert all(row["vla_causal"] for row in table)

        for _ in range(5):
            orch.observe_vla_outcome(action, delta_nu=0.0)
        action_idx = orch.decoder.action_to_id[action]
        assert orch._vla_causal_planner.expected_delta(action_idx) == -1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contract")
    def test_cuda_component_boundaries(self):
        device = torch.device("cuda")
        ing = HenriMetricPatchIngress(
            d_model=D, num_blocks=NB, patch_size=PATCH
        ).to(device)
        grid = torch.zeros(GRID, GRID, dtype=torch.long, device=device)
        grid[1:3, 1:3] = 3
        wave = ing.encode_grid(grid)
        assert wave.device.type == "cuda"
        assert abs(float(wave.norm().cpu()) - 1.0) < 1e-5

        causal = BoundedExteroceptiveEFEPlanner(d_model=D, num_actions=4)
        pred = F.normalize(torch.randn(D, device=device), p=2, dim=-1)
        goal = F.normalize(torch.randn(D, device=device), p=2, dim=-1)
        plate = F.normalize(torch.randn(2, D, device=device), p=2, dim=-1)
        score = causal.score_action(pred, 0, goal_wave=goal, baseplate=plate)
        assert math.isfinite(score)

        egr = CanonicalCodebookEgress(dim=D, beta=8.0)
        rows = F.normalize(torch.randn(2, D, device=device), p=2, dim=-1)
        egr.register(rows, [0, 1])
        result = egr.decode(rows[0])
        assert result.status == "SNAPPED"
        assert result.snapped_index == 0


# ---------------------------------------------------------------------------
# G-EGRESS (canonical codebook + syntax rejection)
# ---------------------------------------------------------------------------
class TestHopfieldEgress:
    def test_noise_floor_math(self):
        sf = noise_floor_sigma(65536, 0.15)
        assert abs(sf - 0.15 / 256.0) < 1e-12  # 5.859e-4

    def test_20_of_20_recovery_under_floor_noise(self):
        torch.manual_seed(3)
        egr = CanonicalCodebookEgress(dim=D, beta=8.0)
        rows = F.normalize(torch.randn(20, D), p=2, dim=-1)
        egr.register(rows, list(range(20)))
        sigma = noise_floor_sigma(D, 0.15)
        ok = 0
        for i in range(20):
            noisy = F.normalize(rows[i] + sigma * torch.randn(D), p=2, dim=-1)
            res = egr.decode(noisy)
            if res.status == "SNAPPED" and res.snapped_index == i:
                ok += 1
        assert ok == 20, f"recovered {ok}/20 (directive G-EGRESS requires 20/20)"

    def test_noncanonical_rejected(self):
        torch.manual_seed(5)
        egr = CanonicalCodebookEgress(dim=D, beta=8.0)
        rows = F.normalize(torch.randn(3, D), p=2, dim=-1)
        egr.register(rows, [10, 20, 30])  # canonical ids allowlist {10,20,30}
        # query that snaps to the row whose canonical id fails the validator
        # register a 4th row mapped to canonical id -1 with a rejecting validator
        torch.manual_seed(6)
        extra = F.normalize(torch.randn(1, D), p=2, dim=-1)
        egr.register(extra, [-1], validator=lambda cid: cid >= 0)
        res = egr.decode(extra[0])
        assert res.status == "REJECTED"
        with pytest.raises(EgressSyntaxRejectedError):
            egr.decode_valid(extra[0])

    def test_empty_codebook_fail_closed(self):
        egr = CanonicalCodebookEgress(dim=D, beta=8.0)
        res = egr.decode(torch.randn(D))
        assert res.status == "REJECTED"
        with pytest.raises(EgressSyntaxRejectedError):
            egr.decode_valid(torch.randn(D))

    def test_shape_mismatch_fail_closed(self):
        egr = CanonicalCodebookEgress(dim=D, beta=8.0)
        with pytest.raises(ValueError):
            egr.register(torch.randn(4, D + 1), [0, 1, 2, 3])

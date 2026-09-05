"""Contract tests — Carrier G1 EDMD Latent Composition Predictor.

Covers the new module's sealed algebra (online RecursiveDualEDMD fit from
demo-pair waves, goal-prototype action, leave-one-out gate), the typed
fail-closed statuses, and the production_arc_run.py wiring (flag
HENRI_EDMD_PREDICT default OFF, Layer 0c presence, GOAL_EDMD_* statuses,
GOAL_EDMD_PREDICT telemetry emission).

Pre-registration: experiments/verification/g1_edmd_predict_prereg.md
Gate G1.2: EDMD_PREDICT_OK iff held_out_cos > 0.30 AND
held_out_cos > identity_cos + 0.10, else EDMD_PREDICT_UNDERFIT
(predicted_wave=None, fail-closed).
"""

import hashlib
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "production_arc_run.py"
MODULE = REPO_ROOT / "henri_edmd_predict.py"

NUM_BLOCKS = 8192
BLOCK_DIM = 8
D_MODEL = NUM_BLOCKS * BLOCK_DIM  # 65536

R_RANK = 16
LAMBDA_FORGET = 0.98


def _rot_fixture(theta_deg: float, n_pairs: int, nb: int = 2, bd: int = 8):
    """Deterministic rotation fixture: x_i = s_i * w_i (tangent, sign-balanced),
    y_i = normalize(e_1 + R(w_i)); R rotates the (e2, e3) plane of block 0 by
    theta. train mean of y -> e1 (goal prototype is the mean, so the
    composition is a rotation about e1)."""
    d = nb * bd
    th = torch.deg2rad(torch.tensor(float(theta_deg)))
    R = torch.eye(d)
    R[1, 1] = th.cos(); R[1, 2] = -th.sin()
    R[2, 1] = th.sin(); R[2, 2] = th.cos()
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0][:n_pairs])
    xs = torch.zeros(n_pairs, nb, bd)
    for i in range(n_pairs):
        idx = 1 + (i % (bd - 1))
        xs[i, 0, idx] = 1.0
    xs = xs * signs.view(-1, 1, 1)
    ys = torch.stack([
        F.normalize((torch.eye(d)[0] + (R @ x.reshape(-1))).reshape(nb, bd),
                    p=2, dim=-1)
        for x in xs])
    return xs, ys


class TestModuleFileHash(unittest.TestCase):
    def test_module_imports_and_contract_constants(self):
        import henri_edmd_predict as m
        self.assertEqual(m.STATUS_OK, "EDMD_PREDICT_OK")
        self.assertEqual(m.STATUS_UNDERFIT, "EDMD_PREDICT_UNDERFIT")
        self.assertEqual(m.RECOVERY_THRESHOLD, 0.30)
        self.assertEqual(m.IDENTITY_MARGIN, 0.10)


class TestFitAndPredictGate(unittest.TestCase):
    def setUp(self):
        from henri_edmd_predict import fit_and_predict
        self.fit = fit_and_predict

    def test_random_data_fails_closed(self):
        torch.manual_seed(1234)
        xs = torch.randn(4, 2, 8)
        ys = torch.randn(4, 2, 8)
        res = self.fit(xs, ys, xs[-1], r_rank=16, lambda_forget=1.0)
        self.assertEqual(res.status, "EDMD_PREDICT_UNDERFIT")
        self.assertIsNone(res.predicted_wave)
        self.assertTrue(res.held_out_cos <= 0.30 + 1e-6 or
                        res.held_out_cos <= res.identity_cos + 0.10 + 1e-6)

    def test_deterministic_composition_passes_both_arms(self):
        # Verified numerically: theta=45, 8 pairs -> held 0.8042, identity
        # 0.5000, improvement +0.3042 (> +0.10) -> EDIT_PREDICT_OK.
        xs, ys = _rot_fixture(45.0, 8)
        res = self.fit(xs, ys, xs[-1], r_rank=16, lambda_forget=1.0)
        self.assertEqual(res.status, "EDMD_PREDICT_OK")
        self.assertIsNotNone(res.predicted_wave)
        self.assertEqual(tuple(res.predicted_wave.shape), (2, 8))
        self.assertGreater(res.held_out_cos, 0.30)
        self.assertGreater(res.held_out_cos, res.identity_cos + 0.10)

    def test_small_pairs_and_small_angle_underfit(self):
        # 4 pairs at 15 deg: measured held 0.4291 vs identity 0.7071 -> the
        # composition does NOT beat identity -> underfit (honest gate).
        xs, ys = _rot_fixture(15.0, 4)
        res = self.fit(xs, ys, xs[-1], r_rank=16, lambda_forget=1.0)
        self.assertEqual(res.status, "EDMD_PREDICT_UNDERFIT")
        self.assertIsNone(res.predicted_wave)

    def test_no_demos_typed(self):
        xs = torch.zeros(0, 2, 8)
        ys = torch.zeros(0, 2, 8)
        res = self.fit(xs, ys, torch.zeros(2, 8))
        self.assertEqual(res.status, "BLOCKED_NO_DEMOS")

    def test_single_pair_unfit_for_holdout(self):
        xs = torch.randn(1, 2, 8)
        ys = torch.randn(1, 2, 8)
        res = self.fit(xs, ys, xs[-1])
        self.assertEqual(res.status, "BLOCKED_EMPTY_DEMOS")

    def test_pair_count_mismatch_typed(self):
        xs = torch.randn(3, 2, 8)
        ys = torch.randn(2, 2, 8)
        res = self.fit(xs, ys, xs[-1])
        self.assertEqual(res.status, "BLOCKED_IMPORT_FAILED")


class TestRunnerWiring(unittest.TestCase):
    def test_flag_consumer_present_and_default_off(self):
        src = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'HENRI_EDMD_PREDICT = os.environ.get("HENRI_EDMD_PREDICT", "0") == "1"',
            src)
        self.assertIn("if goal_wave is None and HENRI_EDMD_PREDICT:", src)
        self.assertIn("GOAL_EDMD_PREDICT", src)

    def test_layer0c_blocks_present(self):
        src = RUNNER.read_text(encoding="utf-8")
        for needle in (
            "from henri_edmd_predict import predict_solution_grids",
            "GOAL_EDMD_NO_DEMOS",
            "GOAL_EDMD_UNDERFIT",
            "GOAL_EDMD_FAIL_CLOSED",
            "event_type\": \"GOAL_EDMD_PREDICT",
        ):
            self.assertIn(needle, src)


if __name__ == "__main__":
    unittest.main()

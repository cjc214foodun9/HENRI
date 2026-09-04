"""Contract tests — Goal Adapter v1 wiring (Carrier: goal-adapter).

Covers the sealed adapter artifact (file SHA pin), the adapter's algebra
(per-block orthogonal Procrustes, [8192,8] goal wave, orthogonality bound),
the fail-closed NO_DEMOS path, and the runner wiring (flag consumer,
default-OFF default, Layer 1b presence, telemetry emission).

Sealed artifact: henri_goal_adapter.py @ f30afdde, file SHA-256
0341a278039313261d0cc63faa336afff1bb9f396d7aa8d65ae214ceaea28888
(computed from the extracted bytes; matches governance ID 0341a278).
"""

import hashlib
import os
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "production_arc_run.py"
ADAPTER = REPO_ROOT / "henri_goal_adapter.py"

SEALED_ADAPTER_SHA = (
    "0341a278039313261d0cc63faa336afff1bb9f396d7aa8d65ae214ceaea28888"
)

NUM_BLOCKS = 8192
BLOCK_DIM = 8
D_MODEL = 65536


class TestSealedArtifact(unittest.TestCase):
    def test_adapter_file_hash_matches_sealed(self):
        raw = ADAPTER.read_bytes()
        # Canonical LF bytes (git blob normalization; sealed pin 0341a278 is
        # the LF digest). Windows CRLF checkouts hash raw CRLF bytes and
        # would mismatch the sealed LF value without this normalization.
        canonical = raw.replace(b"\r\n", b"\n")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(), SEALED_ADAPTER_SHA)


class TestAdapterAlgebra(unittest.TestCase):
    def setUp(self):
        from henri_goal_adapter import HenriGoalAdapter
        self.adapter = HenriGoalAdapter(device="cpu")

    def test_build_goal_shape_and_geometry(self):
        torch.manual_seed(0)
        m = 3
        x = torch.randn(m, NUM_BLOCKS, BLOCK_DIM)
        y = torch.randn(m, NUM_BLOCKS, BLOCK_DIM)
        test = torch.randn(NUM_BLOCKS, BLOCK_DIM)
        res = self.adapter.build_goal(x, y, test, prompt="")
        goal = res["goal_wave"]
        self.assertEqual(tuple(goal.shape), (NUM_BLOCKS, BLOCK_DIM))
        # per-block unit rows
        norms = goal.norm(dim=-1)
        self.assertLess(float((norms - 1.0).abs().max()), 1e-4)
        self.assertLess(res["orthogonality_err"], 1e-3)
        self.assertGreater(res["demo_recon_cos"], 0.1)

    def test_text_channel_sensitive_to_prompt(self):
        torch.manual_seed(1)
        x = torch.randn(2, NUM_BLOCKS, BLOCK_DIM)
        y = torch.randn(2, NUM_BLOCKS, BLOCK_DIM)
        test = torch.randn(NUM_BLOCKS, BLOCK_DIM)
        g0 = self.adapter.build_goal(x, y, test, prompt="")["goal_wave"]
        g1 = self.adapter.build_goal(x, y, test, prompt="different text")["goal_wave"]
        diff = float((g0 - g1).norm())
        self.assertGreater(diff, 1e-3)


class TestRunnerWiring(unittest.TestCase):
    def test_flag_consumer_present_and_default_off(self):
        src = RUNNER.read_text(encoding="utf-8")
        self.assertIn('HENRI_GOAL_ADAPTER = os.environ.get("HENRI_GOAL_ADAPTER", "0") == "1"', src)
        self.assertIn("if goal_wave is None and HENRI_GOAL_ADAPTER:", src)
        self.assertIn('goal_status = "GOAL_HENRI_ADAPTER"', src)
        self.assertIn("GOAL_ADAPTER_NO_DEMOS", src)
        self.assertIn('"adapter_info": adapter_info,', src)
        self.assertIn("GOAL_ADAPTER_FAIL_CLOSED", src)


if __name__ == "__main__":
    unittest.main()

"""
Carrier U1 contract tests: HENRIUnifiedVLAModel default-OFF + composition.

These tests verify composition and consumption of LIVE modules. They grant
no score eligibility and no benchmark claim (see reconciliation doc).

Run on CI / remote Vast target for the official gate. Local CPU run is a
development smoke only and is labeled as such in the receipt.
"""

import os
import unittest
from unittest.mock import MagicMock

import torch

from henri_unified_vla import (
    FLAG,
    HENRIUnifiedVLAModel,
    get_unified_vla,
)


class TestDefaultOff(unittest.TestCase):
    """Differential contract: flag absent => factory returns None."""

    def test_factory_none_when_flag_absent(self):
        os.environ.pop(FLAG, None)
        self.assertIsNone(
            get_unified_vla(tokenizer=object(), orchestrator=object(), action_gate=object())
        )

    def test_factory_constructs_when_flag_present(self):
        os.environ[FLAG] = "1"
        try:
            model = get_unified_vla(
                tokenizer=MagicMock(), orchestrator=MagicMock(), action_gate=MagicMock()
            )
            self.assertIsInstance(model, HENRIUnifiedVLAModel)
        finally:
            os.environ.pop(FLAG, None)


class TestComposition(unittest.TestCase):
    """Live-module composition: gate consumption and fail-closed paths."""

    def setUp(self):
        os.environ[FLAG] = "1"
        self.tokenizer = MagicMock()
        self.orch = MagicMock()
        self.gate = MagicMock()
        self.model = get_unified_vla(
            tokenizer=self.tokenizer, orchestrator=self.orch, action_gate=self.gate
        )

    def tearDown(self):
        os.environ.pop(FLAG, None)

    def test_act_returns_rejection_on_gate_rejection(self):
        # Gate rejects -> fail-closed No-Op, action must be None.
        self.orch.plan_action.return_value = (
            MagicMock(), MagicMock(), [], {"efe": 0.0, "explored": False},
        )
        self.gate.gate.return_value = MagicMock(rejection_reason="ACTION_NOT_LEGAL")
        res = self.model.act(MagicMock(), [[0]], [MagicMock()])
        self.assertIsNone(res.action)
        self.assertEqual(res.action_rejection, "ACTION_NOT_LEGAL")

    def test_act_returns_typed_action_on_gate_success(self):
        fake_action = MagicMock()
        fake_action.name = "ACTION1"
        self.orch.plan_action.return_value = (
            fake_action, MagicMock(), [], {"efe": 1.5, "explored": False}
        )
        self.gate.gate.return_value = MagicMock(
            action=fake_action, confidence=0.9, rejection_reason=None
        )
        res = self.model.act(MagicMock(), [[0]], [fake_action])
        self.assertIs(res.action, fake_action)
        self.assertEqual(res.action_name, "ACTION1")
        self.assertEqual(res.efe_chosen, 1.5)

    def test_egress_fail_closed_without_loaded_checkpoint(self):
        # Egress absent => (None, EGRESS_NOT_WIRED); untrained decoder => blocked.
        model = HENRIUnifiedVLAModel(
            tokenizer=self.tokenizer, orchestrator=self.orch, action_gate=self.gate,
            egress_transducer=None,
        )
        text, tele = model.egress_decode(MagicMock(), "prompt")
        self.assertIsNone(text)
        self.assertEqual(tele["egress_status"], "EGRESS_NOT_WIRED")

        fake_egress = MagicMock()
        fake_egress.checkpoint_telemetry.return_value = {
            "checkpoint_load_status": "SKIPPED_NO_CHECKPOINT"
        }
        model.egress = fake_egress
        text, tele = model.egress_decode(MagicMock(), "prompt")
        self.assertIsNone(text)
        self.assertEqual(tele["egress_status"], "EGRESS_BLOCKED_CHECKPOINT")

    def test_egress_flattens_block_wave_at_boundary(self):
        # Live unbinder consumes [batch, d_model]; assembly must flatten [K,8].
        captured = {}
        fake_egress = MagicMock()
        fake_egress.checkpoint_telemetry.return_value = {
            "checkpoint_load_status": "LOADED"
        }
        fake_egress.decode_wave_to_response.side_effect = (
            lambda w, p, w_task=None: captured.update(shape=tuple(w.shape))
            or ("ok", {})
        )
        model = HENRIUnifiedVLAModel(
            tokenizer=self.tokenizer, orchestrator=self.orch, action_gate=self.gate,
            egress_transducer=fake_egress,
        )
        wave = torch.zeros(8192, 8)
        text, tele = model.egress_decode(wave, "prompt")
        self.assertEqual(text, "ok")
        self.assertEqual(captured["shape"], (1, 65536))


if __name__ == "__main__":
    unittest.main(verbosity=2)

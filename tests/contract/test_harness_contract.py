"""Contract tests for the HENRI dual-speed harness (h1, default-OFF).

Local CPU suite: reduced dimensions + disabled checkpoint policy to stay
off the D=65536 required-checkpoint path. These are SOFTWARE contracts only;
the 20 kHz / 50 us claim is verified by the K1 CUDA probe, never here.
"""

import os
import numpy as np
import pytest
import torch

os.environ.setdefault("HENRI_ARC_HARNESS", "0")

from henri_dual_speed_harness import (
    HARNESS_FLAG,
    harness_active,
    HENRIDualSpeedHarness,
    VETO_TAU,
)

D = 1024
NB = 128
RANK = 4


@pytest.fixture(scope="module")
def harness():
    return HENRIDualSpeedHarness(
        d_model=D,
        num_blocks=NB,
        r_rank=RANK,
        device="cpu",
        checkpoint_policy="disabled",
        zone_c_dsn="offline://surrogate",
        zone_c_required=True,
    )


def _grid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10, size=(5, 5)).astype(np.int64)


class TestActivation:
    def test_default_off(self):
        assert HARNESS_FLAG == "HENRI_ARC_HARNESS"
        assert harness_active() is False  # env unset / 0 in this suite

    def test_flag_turns_on(self, monkeypatch):
        monkeypatch.setenv(HARNESS_FLAG, "1")
        assert harness_active() is True
        monkeypatch.delenv(HARNESS_FLAG)


class TestZoneAInvariant:
    def test_encode_grid_unit_norm_flat_wave(self, harness):
        w = harness.vision_encoder.encode_grid(_grid(1))
        # live encoder contract: flat real wave of dim d_model (D = NB*8)
        assert w.shape == (D,)
        norm = torch.norm(w, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)

    def test_inner_step_typed_telemetry(self, harness):
        w = harness.vision_encoder.encode_grid(_grid(2))
        out = harness.inner_step(w, w, axiom_wave=w)
        for key in ("pred_wave", "delta_axiom", "delta_epistemic", "vetoed", "latency_ms"):
            assert key in out
        assert isinstance(out["vetoed"], bool)
        assert out["delta_axiom"] >= 0.0 and out["delta_axiom"] <= 2.0
        assert out["latency_ms"] >= 0.0


class TestVetoContract:
    def test_outer_step_rejects_failing_code(self, harness):
        out = harness.outer_step("raise ValueError('boom')", sagnac_delta=0.1)
        assert out["is_vetoed"] is True
        assert out["q_score"] == float("-inf")
        assert out["returncode"] != 0

    def test_outer_step_accepts_clean_code(self, harness):
        out = harness.outer_step("print(1)", sagnac_delta=0.1)
        assert out["is_vetoed"] is False
        assert out["returncode"] == 0

    def test_veto_threshold_constant(self):
        assert VETO_TAU == 0.35


class TestZoneC:
    def test_offline_surrogate_roundtrip(self, harness):
        assert harness.zone_c_connected is True
        w = harness.vision_encoder.encode_grid(_grid(3))
        eid = harness.checkpoint_engram(w, "test", 0.1)
        assert isinstance(eid, str) and len(eid) > 0
        res = harness.retrieve_conditioning(w)
        assert res is None or isinstance(res, torch.Tensor)

    def test_fail_closed_when_unavailable(self):
        h = HENRIDualSpeedHarness(
            d_model=D,
            num_blocks=NB,
            r_rank=RANK,
            device="cpu",
            checkpoint_policy="disabled",
            zone_c_dsn=None,
            zone_c_required=False,
        )
        assert h.zone_c_connected is False
        assert h.retrieve_conditioning(torch.zeros(NB, 8)) is None

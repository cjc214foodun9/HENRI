# -*- coding: utf-8 -*-
"""Unit tests: WaveJEPASwarm batched GEMM engine + qFHRR batch unbinding.

Small-scale CPU tests (software correctness). The CUDA gates (Gram FP16,
Triton equivalence, latency) live in
experiments/verification/svd_rank_pcie5_verify.py and run on the Vast
RTX 5090 target only.
"""

import math

import pytest
import torch

from qfhrr_kernels import (
    build_cos_lut,
    quantize_phase_flat,
    dequantize_phase_flat,
    qfhrr_batch_similarity,
    _pytorch_batch_similarity_fallback,
)
from wave_jepa_swarm import WaveJEPASwarm


# ---------------------------------------------------------------------------
# WaveJEPASwarm
# ---------------------------------------------------------------------------

class TestWaveJEPASwarm:
    def test_forward_shape_and_unit_norm(self):
        m = WaveJEPASwarm(dim=1024, rank=16, num_actions=4, fp16_factors=False)
        x = torch.randn(3, 1024, dtype=torch.complex64)
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        y = m(x, action_idx=torch.tensor([0, 1, 2]))
        assert y.shape == (3, 1024)
        assert torch.isfinite(y).all()
        assert (torch.linalg.vector_norm(y, dim=-1) - 1).abs().max() < 1e-4

    def test_forward_scalar_action(self):
        m = WaveJEPASwarm(dim=512, rank=8, num_actions=3, fp16_factors=False)
        x = torch.randn(2, 512, dtype=torch.complex64)
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        y = m(x, action_idx=1)
        assert y.shape == (2, 512)
        assert torch.isfinite(y).all()

    def test_dimension_mismatch_raises(self):
        m = WaveJEPASwarm(dim=1024, rank=16, fp16_factors=False)
        x = torch.randn(1, 256, dtype=torch.complex64)
        with pytest.raises(ValueError):
            m(x)

    def test_gram_error_within_bound(self):
        m = WaveJEPASwarm(dim=512, rank=8, fp16_factors=False)
        err = float(m.gram_error(packed=False).item())
        assert err <= 1e-4

    def test_gram_error_fp16_packed_within_bound(self):
        m = WaveJEPASwarm(dim=512, rank=8, fp16_factors=True)
        err = float(m.gram_error(packed=True).item())
        assert err <= 1e-3  # fp16 rounding; CUDA gate G2 enforces 1e-4

    def test_edmd_update_adapts_sigma(self):
        m = WaveJEPASwarm(dim=1024, rank=16, fp16_factors=False)
        x = torch.randn(4, 1024, dtype=torch.complex64)
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        y = m(x, action_idx=torch.tensor([0, 1, 2, 3]))
        before = m.sigma_r.clone()
        err = m.update_edmd_factors(x, y, forget_factor=0.5)
        assert math.isfinite(err)
        assert not torch.equal(before, m.sigma_r)

    def test_fp16_footprint(self):
        m = WaveJEPASwarm(dim=131072, rank=128, fp16_factors=True)
        n_bytes = m.packed_footprint_bytes()
        assert n_bytes == 2 * 2 * (131072 * 128) * 2  # 2 factors x r/i x fp16
        assert n_bytes <= 128 * 1024 * 1024  # GB202 L2 budget

    def test_input_device_derived_fallback(self):
        """Durable rule regression: module on any device accepts CPU tensors.

        On a CUDA host the module auto-selects cuda while the caller passes a
        CPU tensor; the fallback path must move factors to the INPUT device.
        On a CPU host this is trivially satisfied. Fails pre-fix on CUDA.
        """
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = WaveJEPASwarm(dim=256, rank=8, num_actions=3, fp16_factors=False, device=dev)
        x = torch.randn(2, 256, dtype=torch.complex64)
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        y = m(x, action_idx=torch.tensor([0, 1]))
        assert y.shape == (2, 256)
        assert torch.isfinite(y).all()
        err = m.update_edmd_factors(x, y, forget_factor=0.5)
        assert math.isfinite(err)


# ---------------------------------------------------------------------------
# qFHRR batch unbinding
# ---------------------------------------------------------------------------

class TestQfhrrBatchSimilarity:
    def test_flat_codec_roundtrip(self):
        angles = torch.linspace(-math.pi, math.pi, 64)
        w = torch.complex(torch.cos(angles), torch.sin(angles))
        q = quantize_phase_flat(w)
        assert q.dtype == torch.int32 and q.max() < 256
        w2 = dequantize_phase_flat(q)
        # bin-START decode (scaffold convention): max phase error <= bin width
        err = (torch.angle(w2) - angles + math.pi) % (2 * math.pi) - math.pi
        assert err.abs().max() <= 2 * math.pi / 256 + 1e-6

    def test_self_similarity_is_one(self):
        w = torch.randn(64, 8)
        w = w / torch.norm(w, dim=-1, keepdim=True)
        q = (torch.atan2(w[..., 4:], w[..., :4]).flatten() + math.pi) * (256 / (2 * math.pi))
        q = (q.to(torch.int64) % 256).to(torch.uint8)
        s = qfhrr_batch_similarity(q, q)
        assert abs(float(s) - 1.0) < 1e-4

    def test_batch_shape(self):
        torch.manual_seed(0)
        q = torch.randint(0, 256, (3, 4096), dtype=torch.uint8)
        c = torch.randint(0, 256, (5, 4096), dtype=torch.uint8)
        s = qfhrr_batch_similarity(q, c)
        assert s.shape == (3, 5)
        assert (s.abs() <= 1.0).all()

    def test_fallback_matches_reference(self):
        torch.manual_seed(1)
        q = torch.randint(0, 256, (2, 2048), dtype=torch.uint8)
        c = torch.randint(0, 256, (4, 2048), dtype=torch.uint8)
        lut = build_cos_lut("cpu")
        s = qfhrr_batch_similarity(q, c, use_triton=False)
        s_ref = _pytorch_batch_similarity_fallback(q, c, lut)
        assert torch.allclose(s, s_ref, atol=1e-5)

    def test_singleton_squeeze(self):
        torch.manual_seed(2)
        q = torch.randint(0, 256, (1, 2048), dtype=torch.uint8)
        c = torch.randint(0, 256, (2048,), dtype=torch.uint8)
        s = qfhrr_batch_similarity(q, c)
        assert s.shape == (1,)

"""CORTICAL INGRESS — multi-scale continuous-phase transduction (VCNet-derived).

SOURCE
    Hill et al., "The Geometry of Cortical Computation: Manifold Disentanglement
    and Predictive Dynamics in VCNet", NeurIPS 2025 WORKSHOP (NeurReps + CogInterp).
    arXiv 2508.02995. Basis: project_henri_cortical_computation_engineering_blueprint.

WHY THIS EXISTS (the measured defect it repairs)
    Measured this session, in order:
      1. The legacy text->wave map has NO differentiable path. `wave_of` is
         no_grad-wrapped; encode_text() called DIRECTLY still returns
         requires_grad=False; the path severs at argmax(routes)->int, at the frozen
         `angle` buffer lookup, and at torch.polar(torch.tensor(1.0), phase) whose
         magnitude is a fresh constant. `requires_grad_(True)` is therefore a DEAD
         FLAG. The phase_residual term was added as a minimal repair.
      2. The pooling stage repairs GEOMETRY but not ACCURACY (pooling swap:
         PR 2.24 -> 27.09 at 12.1x, pairwise |cos| 0.6528 -> 0.1448, ridge
         +0.0014, below floor). Verdict POOLING_NOT_THE_LEVER.
    Therefore the accuracy limit is UPSTREAM. This module replaces the discrete
    phase-address lookup with a CONTINUOUS, end-to-end differentiable map:
        x (token ids) -> embedding -> parallel Conv1d k in {3,5,7} -> lateral
        consistency -> phase map exp(i*(W phi + b)) -> unit-norm complex wave.
    A convolution-to-phase map is smooth in its input, so the steepest-descent
    path is connected by construction. That is the property the legacy ingress
    lacks.

ADAPTATION RULES (do NOT transplant the blueprint literally)
    * The blueprint is IMAGE code (Conv2d, in_channels=3, light fields). M4 is
      character-tokenized PROGRAM TEXT. This module uses Conv1d over the token
      axis, which is the faithful text analog.
    * The blueprint's ventral path ends in AdaptiveAvgPool2d((1,1)) -- GLOBAL
      AVERAGE POOLING, the same operator family measured to collapse participation
      ratio 26.21 -> 2.45. Order matters for text (cos(w(RC),w(CR)) = 0.0 is a
      measured property). So the order-sensitive path here uses ORDER-PRESERVING
      pooling (position-weighted), never global average pooling.
    * The blueprint says D = 65,536 and nn.Linear(65536, 65536). That is 4.3e9
      parameters and cannot exist inside a 447M budget. This module keeps
      D = 4096 complex to match every committed receipt.
    * Blueprint arithmetic (92.08%, 0.04 MB, 43% reduction) is UNSOURCED; the
      paper states 92.1% and 74.4%. Those are the authors' image benchmarks and
      establish nothing about HENRI.

CONTRACT
    Default-OFF. `CorticalIngress` is constructed only when explicitly requested.
    It does not touch the legacy path, so every committed receipt still
    reproduces byte-for-byte.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateralConsistency(nn.Module):
    """Local intra-channel gating (blueprint stage 1, 'horizontal connections').

    A 1x1 sigmoid gate over the concatenated multi-scale features. It MODULATES,
    it does not aggregate: no reduction across the token axis, so order survives.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1),
                                  nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class MultiScaleTransduction(nn.Module):
    """Parallel Conv1d streams (k = 3, 5, 7) -> phase map -> unit complex wave.

    Differentiable end to end. This is the property the legacy argmax/buffer/
    polar path does not have.
    """

    def __init__(self, vocab: int, dim: int = 4096, d_emb: int = 64,
                 base_channels: int = 32, kernels: tuple[int, ...] = (3, 5, 7),
                 phase_gain: float = 1.0, use_pos_embed: bool = False,
                 max_len: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.kernels = tuple(kernels)
        self.use_pos_embed = bool(use_pos_embed)
        self.max_len = int(max_len)
        # MEASURED ROOT CAUSE OF ORDER BLINDNESS. Default nn.Linear init on a
        # LayerNorm'd input emits phases with std ~0.577 rad (measured 0.595),
        # so exp(i*phase) ~ 1 + i*phase and EVERY wave collapses toward the
        # all-ones vector: measured cos(scene, ones) = 0.9999 and a content
        # control pair scored 0.99982. Order-blindness is a SYMPTOM of that
        # collapse, not a missing position code. phase_gain spreads the phases
        # across the circle. Default 1.0 preserves legacy bytes.
        self.phase_gain = float(phase_gain)
        self.embed = nn.Embedding(vocab, d_emb)
        # depthwise-separable: a depthwise conv over channels + a 1x1 mix
        self.branches = nn.ModuleList()
        for k in self.kernels:
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_emb, base_channels, k, padding=k // 2, groups=1),
                nn.GELU(),
                nn.Conv1d(base_channels, base_channels, 1),
            ))
        ch = base_channels * len(self.kernels)
        self.lateral = LateralConsistency(ch)
        self.norm = nn.LayerNorm(ch)          # real-valued; applied BEFORE phase map
        self.to_phase = nn.Linear(ch, self.dim)
        # OPTION 2 (user-specified): LEARNED POSITION EMBEDDINGS. Created ONLY
        # when enabled, so the default path draws no extra RNG and keeps its
        # bytes. pos_embed adds order as an OFFSET to the token embedding, which
        # needs NO phase spreading (gain stays 1.0) and therefore need not
        # destroy the graded token-overlap similarity the readout relies on.
        self.pos_embed = (nn.Embedding(self.max_len, d_emb)
                          if self.use_pos_embed else None)

    def phase_map(self, ids: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
        """ids [B, T] long -> phase angles [B, T, D]. SINGLE implementation.

        SELF-CAUGHT DEAD FLAG: the facade used to RE-IMPLEMENT this computation
        inline, so a phase_gain patch applied here was unreachable from
        CorticalIngress.forward. The gain sweep returned byte-identical values
        for gains 1..32, which exposed it. Both callers now share this method.
        """
        x = self.embed(ids)                     # [B, T, d_emb]
        if self.pos_embed is not None:
            t_ax = ids.shape[1]
            x = x + self.pos_embed(torch.arange(t_ax, device=ids.device))[None]
        h = x.transpose(1, 2)                   # [B, d_emb, T]
        feats = [br(h) for br in self.branches]
        t = min(f.shape[-1] for f in feats)
        feats = [f[..., :t] for f in feats]
        cat = self.lateral(torch.cat(feats, dim=1))
        cat = self.norm(cat.transpose(1, 2))    # [B, T, ch]
        if mask is not None:
            cat = cat * mask[..., None].to(cat.dtype)
        return self.phase_gain * self.to_phase(cat)

    def forward(self, ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """ids [B, T] long -> psi [B, D] complex, unit norm per row."""
        phase = self.phase_map(ids, mask)                     # [B, T, D]
        psi_t = torch.polar(torch.ones_like(phase), phase)   # exp(i*phase)
        # ORDER-PRESERVING aggregation. A plain position-mean is a BAG OF TOKENS
        # and cannot encode order: the legacy ingress measures cos(w(IR),w(RI))=1.0
        # and this module's FIRST DRAFT, which used a plain mean, measured 0.989 --
        # still order-blind. Each position now carries a DISTINCT FIXED phase code,
        # so swapping two tokens changes the sum. The code is fixed (not learned),
        # keeping the map deterministic and reproducible.
        b, t, d = psi_t.shape
        pos = torch.arange(t, dtype=phase.dtype)
        code = torch.arange(d, dtype=phase.dtype)
        ramp = torch.exp(1j * (2.0 * math.pi * pos[:, None] * code[None, :] / d))
        ramp = ramp[None]                                   # [1, T, D]
        if mask is not None:
            w = mask.to(phase.dtype).unsqueeze(-1)          # [B, T, 1]
            psi = (psi_t * ramp * w).sum(1) / w.sum(1).clamp_min(1.0)
        else:
            psi = (psi_t * ramp).mean(1)
        return F.normalize(psi, p=2, dim=-1)


class DualStreamDisentanglement(nn.Module):
    """Ventral (content, order-invariant) / dorsal (order, equivariant).

    Measured motivation: the legacy Zone A wave is a bag of BPE, so
    cos('ab','ba') = 1.0 -- order is DISCARDED. A ventral-only ingress repeats
    that defect. The dorsal stream exists to keep order as a separate factor.
    Disentangling them is the point, so the two streams are bound (default) or
    concatenated (ablation arm), never averaged together.
    """

    def __init__(self, dim: int = 4096, bind: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.bind = bool(bind)

    def forward(self, psi_t: torch.Tensor, mask: torch.Tensor | None = None):
        """psi_t [B, T, D] complex -> (psi_id [B,D], psi_pose [B,D])."""
        # ventral: order-INVARIANT content (mean over positions, phase-averaged)
        if mask is not None:
            w = mask.to(psi_t.real.dtype).unsqueeze(-1)
            psi_id = (psi_t * w).sum(1) / w.sum(1).clamp_min(1.0)
        else:
            psi_id = psi_t.mean(1)

        # dorsal: order-EQUIVARIANT. A position-weighted sum with a distinct
        # phase ramp per position, so swapping two tokens changes psi_pose.
        b, t, d = psi_t.shape
        pos = torch.arange(t, dtype=psi_t.real.dtype)
        ramp = torch.exp(1j * (2.0 * math.pi * pos[:, None]
                               * torch.arange(d, dtype=psi_t.real.dtype)[None, :] / d))
        psi_pose = (psi_t * ramp[None]).sum(1)

        psi_id = F.normalize(psi_id, p=2, dim=-1)
        psi_pose = F.normalize(psi_pose, p=2, dim=-1)
        return psi_id, psi_pose

    def combine(self, psi_id: torch.Tensor, psi_pose: torch.Tensor
                ) -> torch.Tensor:
        """Bind (circular convolution) by default, concatenate for the ablation."""
        if not self.bind:
            return torch.cat([psi_id, psi_pose], dim=-1)
        fid = torch.fft.fft(psi_id, dim=-1)
        fpo = torch.fft.fft(psi_pose, dim=-1)
        return torch.fft.ifft(fid * fpo, dim=-1)


class NeuromodulatoryGating(nn.Module):
    """gamma = Softplus(W*psi + b). Scales a real curvature proxy.

    Blueprint stage 4. Kept small and separable: it modulates, it does not
    replace the novelty score. The score s(q) stays owned by novelty_gate.py.
    """

    def __init__(self, dim: int = 4096):
        super().__init__()
        self.fc = nn.Linear(2 * dim, 1)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([psi.real, psi.imag], dim=-1)
        return F.softplus(self.fc(pair)) + 1e-3


class TangentPredictionError(nn.Module):
    """Top-down predictive coding as a tangent vector on the wave sphere.

    Blueprint stage 3 / Directive 3. The Hermitian projection matters: for unit
    waves the identity  ||eps||^2 + |<obs,pred>|^2 = 1  must hold to 1e-6. A real
    inner product here would silently mis-project and corrupted eps would then
    drive the Sagnac veto (Directive 3).
    """

    def __init__(self, dim: int = 4096):
        super().__init__()
        self.dim = int(dim)
        self.generator = nn.Linear(dim, dim, bias=False)

    def forward(self, psi_obs: torch.Tensor, psi_ait: torch.Tensor):
        phase = self.generator(psi_ait.real)
        psi_pred = F.normalize(torch.polar(torch.ones_like(phase), phase), p=2, dim=-1)
        inner = (psi_obs * psi_pred.conj()).sum(dim=-1, keepdim=True)
        eps = psi_obs - inner * psi_pred
        return eps, inner


class CorticalIngress(nn.Module):
    """Facade. DEFAULT-OFF: nothing imports this unless a caller asks for it."""

    def __init__(self, vocab: int, dim: int = 4096, bind: bool = True,
                 d_emb: int = 64, phase_gain: float = 1.0,
                 use_pos_embed: bool = False, max_len: int = 8):
        super().__init__()
        self.dim = int(dim)
        self.transduce = MultiScaleTransduction(vocab, dim, d_emb=d_emb,
                                                phase_gain=phase_gain,
                                                use_pos_embed=use_pos_embed,
                                                max_len=max_len)
        self.dual = DualStreamDisentanglement(dim, bind=bind)
        self.neuromod = NeuromodulatoryGating(dim)
        self.tangent = TangentPredictionError(dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None):
        """-> dict with psi_scene, psi_id, psi_pose, gamma."""
        # Shared phase map: no inline duplication (that made phase_gain a dead flag).
        phase = self.transduce.phase_map(ids, mask)      # [B, T, D]
        psi_t = torch.polar(torch.ones_like(phase), phase)          # [B,T,D]
        psi_id, psi_pose = self.dual(psi_t, mask)
        scene = F.normalize(self.dual.combine(psi_id, psi_pose), p=2, dim=-1)
        return {"psi_scene": scene, "psi_id": psi_id, "psi_pose": psi_pose,
                "gamma": self.neuromod(psi_id)}

"""HENRI Goal Adapter v1 — HENRI-native text+grid -> [8192,8] planner goal wave.

Ground-up design (no pretrained backbones; directive 2026-08-25):
  Channel G (grid): per-block orthogonal Procrustes operator W_task compiled from
    public demo pairs (X_i, Y_i) via the live HENRIVisionEncoder waves; applied to
    the test grid wave -> goal grid component. O(8) per Cl(3,0) block is the
    smallest algebra-native transformation carrier (no dense D^2 operator).
  Channel T (text): deterministic char-engram + fractional-position qFHRR codec
    (Komer 2020 / Frady 2022 protocol; BPE falsified) -> Z_256 ring ->
    representation-aware ring_to_real -> [8192,8] real wave.
  Fusion: Psi_goal = normalize(beta_g * Psi_grid + beta_t * Psi_prompt).

Default-OFF (HENRI_GOAL_ADAPTER=1). Zero trainable parameters.
Contract: HENRI-SPEC-2026-08-GOAL-ADAPTER-V1. Prereg sealed 2026-08-25.
"""

import hashlib
import json
import math
import os

import torch
import torch.nn as nn

D_MODEL = 65536
NUM_BLOCKS = 8192
BLOCK_DIM = 8
K_BINS = 256


def _hash_expand(key: str, n_bytes: int) -> bytes:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    out = bytearray()
    i = 0
    while len(out) < n_bytes:
        out += hashlib.sha256(h + i.to_bytes(4, "little")).digest()
        i += 1
    return bytes(out[:n_bytes])


def is_enabled() -> bool:
    return os.environ.get("HENRI_GOAL_ADAPTER", "0") == "1"


class HenriPromptCodec(nn.Module):
    """Deterministic structured text codec: char engrams + fractional position.

    Binding: q^<p> = round(p * q) mod K with p = (pos+1)/(len+1) in (0,1).
    Bundle: modular sum over chars. ring_to_real: cos phasor (run21-verified
    representation-aware conversion), per-block unit rows.
    """

    codec_name = "henri_prompt_qfhrr"
    codec_version = "v1"

    def __init__(self, num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, k_bins=K_BINS,
                 device=None):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.block_dim = int(block_dim)
        self.k_bins = int(k_bins)
        self.d_model = self.num_blocks * self.block_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _char_ring(self, ch: str) -> torch.Tensor:
        raw = _hash_expand(f"henri_prompt_char::{ch}", self.d_model)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int32)

    def _pos_ring(self, pos: int) -> torch.Tensor:
        raw = _hash_expand(f"henri_prompt_pos::{pos}", self.d_model)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int32)

    def encode_prompt(self, text: str) -> torch.Tensor:
        """Circular-mean bundle of position-scaled char phasors (run21 protocol).

        bound[d] = (q_char[d] * (2*pos+1)) mod 256: odd multiplier is coprime
        to 256 -> full per-position discrimination; length-independent -> a
        1-char edit changes only that term. Bundling accumulates real/imag
        phasors then atan2 (true circular mean), never a separable modular sum
        (a separable sum is permutation-invariant by construction).
        """
        if not text:
            return torch.zeros(self.num_blocks, self.block_dim,
                               device=self.device)
        re = torch.zeros(self.d_model, dtype=torch.float32, device=self.device)
        im = torch.zeros(self.d_model, dtype=torch.float32, device=self.device)
        for pos, ch in enumerate(text):
            q = self._char_ring(ch).to(device=self.device, dtype=torch.float32)
            bound = (q * (2 * pos + 1)) % self.k_bins
            ang = bound * (2.0 * math.pi / self.k_bins)
            re += torch.cos(ang)
            im += torch.sin(ang)
        phase = torch.atan2(im, re)
        ring = (torch.round(phase * (self.k_bins / (2.0 * math.pi)))
                % self.k_bins).to(torch.uint8)
        return self.ring_to_real(ring)

    def ring_to_real(self, ring: torch.Tensor) -> torch.Tensor:
        phase = ring.to(torch.float32) * (2.0 * math.pi / self.k_bins)
        real = torch.cos(phase).to(self.device)
        wave = real.view(self.num_blocks, self.block_dim)
        wave = wave / (wave.norm(dim=-1, keepdim=True) + 1e-12)
        return wave

    def sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.reshape(-1).to(torch.float32)
        b = b.reshape(-1).to(torch.float32)
        return float((a @ b) / (a.norm() + 1e-12) / (b.norm() + 1e-12))

    def manifest(self) -> dict:
        return {
            "codec_name": self.codec_name, "codec_version": self.codec_version,
            "num_blocks": self.num_blocks, "block_dim": self.block_dim,
            "k_bins": self.k_bins,
            "binding": "position_scaled_char_phasor_odd_key",
            "bundling": "circular_mean_atan2", "ring_to_real": "cos_phasor",
            "d_model": self.d_model,
        }

    def manifest_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.manifest(), sort_keys=True).encode()).hexdigest()


class HenriTaskOperator(nn.Module):
    """Per-block orthogonal Procrustes: W_task = blockdiag{O_k}, O_k = U_k V_k^T.

    M_k = sum_i Y_i[:,k]^T X_i[:,k] (8x8); SVD -> O_k. Norm-preserving, O(K*8^3).
    """

    def compile_from_demos(self, psi_x: torch.Tensor, psi_y: torch.Tensor) -> torch.Tensor:
        # psi_x, psi_y: [m, 8192, 8] per-block unit-row waves
        m = torch.einsum("mka,mkb->kab", psi_y, psi_x)  # [8192,8,8]
        u, _, vh = torch.linalg.svd(m)
        return torch.einsum("kab,kbc->kac", u, vh)  # [8192,8,8]

    def apply(self, w: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kab,kb->ka", w, psi)  # [8192,8]

    def orthogonality_error(self, w: torch.Tensor) -> float:
        eye = torch.eye(w.shape[-1], device=w.device, dtype=w.dtype)
        per_block = torch.einsum("kab,kac->kbc", w, w) - eye
        # Dimension-normalized: max over per-block Frobenius errors.
        return float(per_block.norm(dim=(-2, -1)).max().item())


class HenriGoalAdapter(nn.Module):
    """Fuses grid-transformed goal with prompt-codec goal -> [8192,8] goal wave."""

    def __init__(self, beta_grid=0.7, beta_text=0.3, num_blocks=NUM_BLOCKS,
                 block_dim=BLOCK_DIM, device=None):
        super().__init__()
        self.beta_grid = float(beta_grid)
        self.beta_text = float(beta_text)
        self.codec = HenriPromptCodec(num_blocks=num_blocks, block_dim=block_dim,
                                      device=device)
        self.operator = HenriTaskOperator()
        self.num_blocks = int(num_blocks)
        self.block_dim = int(block_dim)

    @torch.no_grad()
    def build_goal(self, demo_x_waves: torch.Tensor, demo_y_waves: torch.Tensor,
                   test_x_wave: torch.Tensor, prompt: str = "") -> dict:
        """demo waves: [m,8192,8]; test: [8192,8]; prompt: str (optional).

        Channel G normalizes every wave per-block row BEFORE compiling/applying
        (representation-agnostic: the live vision encoder emits real-part
        phasor rows with row norm ~2, not unit rows; per-row normalization
        makes the Procrustes operator scale-invariant per block).
        """
        dev = self.codec.device

        def _unit_rows(t: torch.Tensor) -> torch.Tensor:
            return t.to(dev) / (t.to(dev).norm(dim=-1, keepdim=True) + 1e-12)

        demo_x_waves = _unit_rows(demo_x_waves)
        demo_y_waves = _unit_rows(demo_y_waves)
        test_x_wave = _unit_rows(test_x_wave)
        w = self.operator.compile_from_demos(demo_x_waves, demo_y_waves)
        goal_grid = self.operator.apply(w, test_x_wave)
        # per-demo reconstruction cosine (row-wise then mean)
        recos = []
        for i in range(demo_y_waves.shape[0]):
            r = torch.einsum("kab,kb->ka", w, demo_x_waves[i])
            recos.append(float(torch.cosine_similarity(
                r.reshape(-1), demo_y_waves[i].reshape(-1), dim=0).item()))
        demo_cos = sum(recos) / len(recos)
        goal_text = torch.zeros(self.num_blocks, self.block_dim,
                                device=self.codec.device)
        if prompt:
            goal_text = self.codec.encode_prompt(prompt)
        goal = self.beta_grid * goal_grid + self.beta_text * goal_text
        goal = goal / (goal.norm(dim=-1, keepdim=True) + 1e-12)
        return {
            "goal_wave": goal, "operator": w, "demo_recon_cos": demo_cos,
            "orthogonality_err": self.operator.orthogonality_error(w),
            "prompt_used": bool(prompt),
        }

    def manifest(self) -> dict:
        return {
            "adapter": "henri_goal_adapter", "version": "v1",
            "beta_grid": self.beta_grid, "beta_text": self.beta_text,
            "codec": self.codec.manifest(),
        }

    def manifest_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.manifest(), sort_keys=True).encode()).hexdigest()

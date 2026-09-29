"""HENRI VLA vision ingress: image tensor -> D-dimensional phase vector.

WHY THIS EXISTS
===============
`henri_vision_encoder.encode_spatial_grid` accepts TOY INTEGER GRIDS
({0..9}^HxW). A vision-language-action model needs to consume real pixels
[B, 3, H, W]. Nothing in the repo does that. This is that layer.

METHOD (VSA binding + superposition; Plate 1995, Kanerva 2009)
==============================================================
     psi = normalize( sum_i  c_i  (*)  p_i )
  c_i = W_c  patch_i        content vector   (learned or frozen projection)
  p_i = position code       fixed random unit vector per patch slot
  (*) = circular convolution   (the VSA binding operator)

Circular convolution does NOT preserve norm (a known contract in this repo), so
the result is re-normalized to the unit sphere. Superposition of N bound pairs
suffers crosstalk scaling as ~sqrt(N/D), so N is capped and reported.

MEMORY / CONTRACT A
===================
The only matrices are [d_patch, D] and [D] position codes: O(D) and O(d_patch*D).
No [D, D] object is ever formed. At D=65536, d_patch=256 the content matrix is
64 MB -- and the position codes are generated on the fly by RNG rather than
stored, so they cost no persistent memory.

OPTIONAL BACKBONE
=================
If a frozen ViT is available (CLIP / SigLIP / any `transformers`
vision encoder), its patch features are used as `patch_i`; otherwise a
deterministic patch-mean projection is used so the layer works with no
dependencies. The choice is reported in telemetry -- never silently assumed.
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

D_DEFAULT = 65536
PATCH_DEFAULT = 16
BLOCK_DIM = 8                      # matches the planner boundary [num_blocks, 8]
VISION_FALLBACK_MODELS = ("openai/clip-vit-base-patch16",
                          "google/siglip-base-patch16-224")


class VisionIngressError(RuntimeError):
    """Typed failure for the VLA vision ingress."""


def circ_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular convolution along the last dim, via FFT: O(D log D), no [D,D]."""
    fa = torch.fft.rfft(a, dim=-1)
    fb = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(fa * fb, n=a.shape[-1], dim=-1)


def circ_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Circular CORRELATION = approximate unbinding for unitary-ish codes."""
    fa = torch.fft.rfft(a, dim=-1)
    fb = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(fa * torch.conj(fb), n=a.shape[-1], dim=-1)


class VisionPhaseIngress(nn.Module):
    """[B, 3, H, W] float image in [0,1] -> [B, D] unit-norm phase vector."""

    def __init__(self, d_model: int = D_DEFAULT, patch: int = PATCH_DEFAULT,
                 d_patch: int = 256, image_size: int = 64,
                 learned_projection: bool = False, seed: int = 0,
                 vision_model: str | None = None):
        super().__init__()
        if d_model % BLOCK_DIM != 0:
            raise VisionIngressError(
                f"d_model must be a multiple of {BLOCK_DIM}; got {d_model}")
        if image_size % patch != 0:
            raise VisionIngressError(
                f"image_size {image_size} not divisible by patch {patch}")
        self.d_model = int(d_model)
        self.patch = int(patch)
        self.d_patch = int(d_patch)
        self.image_size = int(image_size)
        self.grid = self.image_size // self.patch
        self.n_patches = self.grid * self.grid
        if self.n_patches * self.d_patch < self.d_model:
            # content vectors must be able to excite the full dimension; pad with
            # a repeated hashed mixture instead of raising, and report it.
            self.needs_expand = True
        else:
            self.needs_expand = False

        g = torch.Generator().manual_seed(seed)
        patch_dim = 3 * self.patch * self.patch
        # content projection [patch_dim, d_patch] -- small, not [D, D]
        Wc = torch.randn(patch_dim, self.d_patch, generator=g) / math.sqrt(patch_dim)
        if learned_projection:
            self.Wc = nn.Parameter(Wc)
        else:
            self.register_buffer("Wc", Wc, persistent=True)
        # expansion to D: [d_patch, D] only if needed (n_patches*d_patch < D)
        if self.needs_expand:
            Wx = torch.randn(self.d_patch, self.d_model, generator=g) / math.sqrt(self.d_patch)
            self.register_buffer("Wx", Wx, persistent=True)
        self.seed = int(seed)
        self._pos_cache: dict[int, torch.Tensor] = {}
        self.vision_model_name = vision_model or ""
        self._vision = None
        self._vision_kind = "fallback_deterministic"

    # ------------------------------------------------------------- position
    def position_code(self, i: int, device, dtype, n: int = 1) -> torch.Tensor:
        """Deterministic unit-norm position code for slot i, generated on demand.

        Generated rather than stored: n_patches * D floats would be, at D=65536
        and 16 patches, only 4 MB -- but the same generator scales to thousands of
        slots without any persistent cost.
        """
        key = i
        if key not in self._pos_cache:
            g = torch.Generator(device="cpu").manual_seed(self.seed * 1000003 + i)
            v = torch.randn(self.d_model, generator=g)
            v = v / v.norm()
            self._pos_cache[key] = v
        return self._pos_cache[key].to(device=device, dtype=dtype)

    # ------------------------------------------------------------ backbone
    def _try_load_vision(self) -> bool:
        """Load a frozen ViT if available. Reported, never assumed."""
        if self._vision is not None:
            return True
        try:
            from transformers import CLIPVisionModel          # noqa: N813
        except Exception:                                          # noqa: BLE001
            return False
        for name in ([self.vision_model_name] if self.vision_model_name
                     else list(VISION_FALLBACK_MODELS)):
            if not name:
                continue
            try:
                m = CLIPVisionModel.from_pretrained(name)
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)
                self._vision = m
                self.vision_model_name = name
                self._vision_kind = "clip_vit"
                return True
            except Exception:                                      # noqa: BLE001
                continue
        return False

    # -------------------------------------------------------------- forward
    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() != 4 or img.shape[1] != 3:
            raise VisionIngressError(f"expected [B,3,H,W]; got {tuple(img.shape)}")
        B, _, H, W = img.shape
        if H != self.image_size or W != self.image_size:
            img = F.interpolate(img, size=(self.image_size, self.image_size),
                                mode="bilinear", align_corners=False)
        p = self.patch
        x = img.unfold(2, p, p).unfold(3, p, p)          # [B,3,g,g,p,p]
        x = x.contiguous().view(B, 3, -1, p * p)         # [B,3,N,p*p]
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, 3 * p * p)  # [B,N,3p^2]
        return x

    def content(self, img: torch.Tensor) -> tuple[torch.Tensor, str]:
        """[B,3,H,W] -> [B,N,d_patch] content vectors."""
        if self._vision is None and self.vision_model_name:
            self._try_load_vision()
        if self._vision is not None:
            with torch.no_grad():
                out = self._vision(pixel_values=img)
                feats = out.last_hidden_state               # [B, 1+N, d_vit]
            feats = feats[:, 1:, :]
            B, N, dv = feats.shape
            if not hasattr(self, "Wv") or self.Wv.shape[0] != dv:
                g = torch.Generator().manual_seed(self.seed + 7)
                self.register_buffer(
                    "Wv", torch.randn(dv, self.d_patch, generator=g) / math.sqrt(dv),
                    persistent=True)
            return feats.to(torch.float32) @ self.Wv.to(feats.device), "clip_vit"
        patches = self.patchify(img).to(torch.float32)
        return patches @ self.Wc.to(torch.float32), "fallback_deterministic"

    def forward(self, img: torch.Tensor,
                return_telemetry: bool = False):
        c, kind = self.content(img)                          # [B,N,d_patch]
        B, N, dc = c.shape
        if self.needs_expand:
            c = c @ self.Wx.to(c.dtype)                      # [B,N,D]
            dc = self.d_model
        acc = torch.zeros(B, self.d_model, device=c.device, dtype=torch.float32)
        n_bind = min(N, self.n_patches)
        for i in range(n_bind):
            ci = c[:, i, :]
            if ci.shape[-1] != self.d_model:
                # broadcast a small content vector across blocks with a per-block
                # phase so different slots remain distinguishable
                rep = self.d_model // ci.shape[-1]
                ci = ci.repeat(1, rep + 1)[:, : self.d_model]
            pi = self.position_code(i, c.device, torch.float32)
            acc = acc + circ_conv(ci, pi)
        nrm = acc.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        psi = acc / nrm
        if not return_telemetry:
            return psi
        tel = {"ingress_kind": kind, "n_patches_bound": n_bind,
               "crosstalk_estimate": round(math.sqrt(n_bind / self.d_model), 5),
               "norm_before_normalize": float(nrm.mean().item()),
               "needs_expand": self.needs_expand}
        return psi, tel


# --------------------------------------------------------------- self test
def _self_test() -> int:
    out = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            out[device] = "SKIP (no cuda)"
            continue
        d, size = 8192, 64           # reduced D for speed; shape contract identical
        ing = VisionPhaseIngress(d_model=d, image_size=size).to(device)
        torch.manual_seed(0)
        a = torch.rand(4, 3, size, size, device=device)
        b = torch.rand(4, 3, size, size, device=device)
        pa = ing(a)
        pb = ing(b)
        # a localized change should dominate the global random change
        c = a.clone(); c[:, :, :16, :16] = 1.0
        pc = ing(c)
        out[device] = {
            "shape": list(pa.shape),
            "norm_mean": round(float(pa.norm(dim=-1).mean().item()), 6),
            "finite": bool(torch.isfinite(pa).all().item()),
            "distinct_inputs_differ": round(float(1 - F.cosine_similarity(pa, pb, dim=-1).mean().item()), 6),
            "localized_change_detectable": round(float(1 - F.cosine_similarity(pa, pc, dim=-1).mean().item()), 6),
            "n_params": sum(p.numel() for p in ing.parameters()),
        }
        if device == "cuda":
            out[device]["peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

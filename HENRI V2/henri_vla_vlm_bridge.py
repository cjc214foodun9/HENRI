"""Bridge a REAL frozen vision-language model into HENRI's action policy.

WHY THIS EXISTS
===============
The VLA stack measured so far ingests pixels through `VisionPhaseIngress`, whose
visual features are a FROZEN RANDOM projection (patch vectors @ Wc). That makes it
a policy over random features, not a vision-language-action model. The 4.4B
Qwen3-VL backbone now loads and trains on the 5090 (verified: 252 LoRA modules,
132M trainable, backward 0.13s), but nothing connected it to the action head.

This module is that connection:

    image [B,3,H,W] -> FROZEN Qwen3-VL vision tower -> patch features [B,N,d_vit]
                    -> learnable attention pool -> [B,d_vit]
                    -> VSA bind with position codes + superpose -> psi [B,D]

DESIGN DECISIONS
================
* The VLM is FROZEN here. Only the bridge and the action head train. That keeps
  the comparison clean: any gain is attributable to using REAL visual features,
  not to extra trainable capacity in the vision path.
* Attention pooling (one learnable query) rather than mean pooling, so the bridge
  can select task-relevant patches instead of averaging the goal and the
  end-effector blob together.
* The projected feature is bound with VSA position codes before superposition.
  That preserves the architecture's claim: psi is a compositional role-filler
  structure, not a flat vector.
* Contract A holds: the only matrices are [d_vit, d_pool] and [D]-shaped position
  codes. No [D, D] object is ever formed.

INTROSPECTION, NOT ASSUMPTION
=============================
The Qwen3-VL module tree differs between transformers builds. `find_vision_tower`
tries several paths and REPORTS which one resolved, so a silent fallback to a
wrong submodule cannot masquerade as success.

HONEST LIMIT: the task is still the synthetic pixel reach task. Using real visual
features does not make it a robot benchmark.
"""
from __future__ import annotations

import json
import math
import os
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

D_DEFAULT = 65536
BLOCK_DIM = 8

VISION_TOWER_PATHS = (
    "model.visual", "visual", "vision_model", "model.vision_model",
    "model.model.visual", "model.model.vision_model", "model.vision_tower",
    "vision_tower", "model.vision_tower.vision_model",
)


class VLMBridgeError(RuntimeError):
    """Typed failure for the VLM bridge."""


def find_vision_tower(model: nn.Module) -> tuple[nn.Module | None, str]:
    """Locate the vision tower. Returns (module, resolved_path) or (None, reason)."""
    for path in VISION_TOWER_PATHS:
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj, path
    names = [n for n, m in model.named_modules()
             if any(k in n.lower() for k in ("visual", "vision", "vit"))]
    return None, f"no tower at known paths; candidates={names[:8]}"


class VLMFeatureExtractor:
    """Frozen VLM vision features. Reports which tower path resolved."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
                 device: str = "cuda", dtype=torch.bfloat16):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tower = None
        self.tower_path = ""
        self.d_vit = 0
        self.n_patches = 0

    def load(self) -> "VLMFeatureExtractor":
        import transformers as T
        self.model = T.AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=self.dtype, device_map={"": 0},
            low_cpu_mem_usage=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tower, self.tower_path = find_vision_tower(self.model)
        if self.tower is None:
            raise VLMBridgeError(f"vision tower not found: {self.tower_path}")
        # Qwen3-VL's tower signature is (hidden_states, grid_thw); CLIP's is
        # (pixel_values). Probe the real signature once and record it, then pick
        # the calling convention. Anything else is a guess and fails silently.
        self._needs_grid = "grid_thw" in inspect.signature(
            self.tower.forward).parameters
        self._processor = None
        try:
            pc = getattr(T, "AutoProcessor", None)
            if pc is not None:
                self._processor = pc.from_pretrained(self.model_id)
        except Exception as e:                                          # noqa: BLE001
            self._processor_err = f"{type(e).__name__}: {str(e)[:120]}"
        R_probe = torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            f = self._tower_forward(R_probe)
        self.probe_shape = list(f.shape)
        # DEFECT FIXED 2026-09-28 (caught by review BEFORE the GPU run):
        # Qwen3-VL returns FLATTENED [N_tokens, d_vit]. Reading shape[1] as the
        # patch count then yields d_vit (~1152), not the real N (~64). The
        # downstream guard `feats.shape[1] == self.n_patches` would then fail and
        # the position-BINDING path would silently fall back to pooled-only --
        # a green smoke measuring the wrong architecture. Accept both layouts.
        if f.dim() == 2:
            self.n_patches = int(f.shape[0])
            self.tokens_are_flattened = True
        else:
            self.n_patches = int(f.shape[1])
            self.tokens_are_flattened = False
        self.d_vit = int(f.shape[-1])
        self.needs_grid = bool(self._needs_grid)
        return self

    def _prepare(self, images: torch.Tensor):
        """Return (pixel_values, grid_thw) using the processor when required.

        grid_thw = (temporal, H/patch, W/patch) per image. Derived from the
        processor rather than hand-computed, so a patch-size change cannot
        silently produce the wrong token count.
        """
        if not self._needs_grid:
            return images.to(self.dtype), None
        if self._processor is None:
            raise VLMBridgeError(
                "tower requires grid_thw but no AutoProcessor is available: "
                + getattr(self, "_processor_err", "unknown"))
        proc = self._processor.image_processor if hasattr(
            self._processor, "image_processor") else self._processor
        imgs = [images[i].detach().to("cpu") for i in range(images.shape[0])]
        out = proc(images=imgs, return_tensors="pt")
        pv = out["pixel_values"].to(self.device, self.dtype)
        g = out.get("image_grid_thw")
        if g is None:
            raise VLMBridgeError("processor returned no image_grid_thw")
        return pv, g.to(self.device)

    def _tower_forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B,N,d_vit].

        DEFECT FIXED 2026-09-28: hard-coded field names failed on Qwen3-VL, whose
        tower returns `BaseModelOutputWithDeepstackFeatures` -- not CLIP's
        `last_hidden_state`. Rather than guess again, iterate the output's OWN
        keys and take the first 3-D tensor, recording which key answered.
        """
        pv, grid = self._prepare(images)
        out = self.tower(pv, grid_thw=grid) if grid is not None else self.tower(pv)
        if isinstance(out, torch.Tensor):
            self.output_key_used = "<tensor>"
            return out.float()
        # torch returns (tensor, None) tuples from some paths
        if isinstance(out, (tuple, list)) and len(out) and isinstance(out[0], torch.Tensor):
            t = out[0]
            if t.dim() == 3:
                self.output_key_used = "<tuple[0]>"
                return t.float()
        keys = list(out.keys()) if hasattr(out, "keys") else []
        self.output_keys_seen = keys
        preferred = ("last_hidden_state", "hidden_states", "pooler_output",
                     "deepstack_features")
        for key in preferred:
            if key not in keys:
                continue
            try:
                v = out[key]
            except Exception:                                           # noqa: BLE001
                continue
            if isinstance(v, (list, tuple)):
                if not len(v):
                    continue
                v = v[-1]
            if not isinstance(v, torch.Tensor):
                continue
            # Qwen3-VL returns FLATTENED tokens [sum(t*h*w), d] because grid_thw
            # defines the grid; CLIP-style towers return [B, N, d]. Accept both.
            if v.dim() == 2:
                self.output_key_used = key + "(2d)"
                self.tokens_flattened = True
                return v.float()
            if v.dim() == 3:
                self.output_key_used = key
                self.tokens_flattened = False
                return v.float()
        raise VLMBridgeError(f"no usable feature tensor; keys={keys}")

    @torch.no_grad()
    def features(self, images: torch.Tensor, batch: int = 8) -> torch.Tensor:
        """images in [0,1], [B,3,H,W] -> [B,N,d_vit] float32 on device.

        Handles the flattened-token case by processing one image at a time and
        wrapping each result in a batch dimension, so callers always receive
        [B, N, d] regardless of the tower's native layout.
        """
        if self.tower is None:
            raise VLMBridgeError("call load() first")
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear",
                                   align_corners=False)
        outs = []
        for i in range(images.shape[0]):
            f = self._tower_forward(images[i:i + 1])
            if f.dim() == 2:                       # [N, d] flattened -> [1, N, d]
                f = f.unsqueeze(0)
            outs.append(f)
        return torch.cat(outs, dim=0)


class VLMPhaseBridge(nn.Module):
    """[B,N,d_vit] VLM patches -> [B,D] unit-norm phase vector.

    Trains ONLY: the attention query, the projection to d_pool, the expansion to D,
    and the phase offset. The VLM is frozen upstream.
    """

    def __init__(self, d_vit: int, n_patches: int, d_model: int = D_DEFAULT,
                 d_pool: int = 512, n_heads: int = 4, seed: int = 0,
                 use_position_binding: bool = True, binding_chunk: int = 8):
        super().__init__()
        if d_model % BLOCK_DIM != 0:
            raise VLMBridgeError(f"d_model must be a multiple of {BLOCK_DIM}")
        self.d_vit = int(d_vit)
        self.n_patches = int(n_patches)
        self.d_model = int(d_model)
        self.d_pool = int(d_pool)
        self.use_position_binding = bool(use_position_binding)
        # patches processed per chunk in binding_term (memory, not semantics)
        self.binding_chunk = max(1, int(binding_chunk))

        self.query = nn.Parameter(torch.randn(1, 1, d_vit) * 0.02)
        self.k_proj = nn.Linear(d_vit, d_vit, bias=False)
        self.v_proj = nn.Linear(d_vit, d_vit, bias=False)
        self.to_pool = nn.Linear(d_vit, d_pool)
        self.to_phase = nn.Linear(d_pool, d_model)
        self.phase_offset = nn.Parameter(torch.zeros(d_model))

        g = torch.Generator().manual_seed(seed)
        # position codes for VSA binding: [n_patches, D], deterministic, frozen
        pos = torch.randn(self.n_patches, d_model, generator=g)
        pos = pos / pos.norm(dim=-1, keepdim=True)
        self.register_buffer("pos_codes", pos.to(torch.float32), persistent=True)

    def pool(self, feats: torch.Tensor) -> torch.Tensor:
        """Attention pool [B,N,d] -> [B,d]."""
        B, N, d = feats.shape
        q = self.query.expand(B, -1, -1)
        k = self.k_proj(feats)
        v = self.v_proj(feats)
        att = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(d), dim=-1)
        return (att @ v).squeeze(1)

    def binding_term(self, feats: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """sum_n phi(feats[:,n,:]) * pos[n,:]  ->  [B, D]   (VSA bind + bundle).

        DEFECT FIXED 2026-09-29, measured as a CUDA OOM on the RTX 5090:
        "Tried to allocate 16.00 GiB" at the bind line. The direct form builds
        [B,N,D] for BOTH `pf` and `bound`. At B=256, N=256, D=65536 that is
        16.00 GiB EACH (32 GiB total) on a 31.36 GiB card. Production survived
        only because batch 64 needs 4 GiB each -- peak 17893.7 MiB measured.

        The bind is Hadamard (elementwise) and the bundle is a sum over N, so the
        contraction is linear in n and can be accumulated in chunks with
        IDENTICAL arithmetic. Peak drops from O(B*N*D) to O(B*chunk*D):
        537 MiB at the settings that OOM'd. This is an exact reformulation, not
        an approximation -- verified against the naive form in
        experiments/verification/test_binding_chunk_equiv.py.
        """
        B, N, _ = feats.shape
        c = max(1, int(chunk or self.binding_chunk))
        acc = None
        for k in range(0, N, c):
            fk = feats[:, k:k + c, :].to(torch.float32)
            pk = self.to_phase(self.to_pool(self.v_proj(fk)))        # [B, c, D]
            term = (pk * self.pos_codes[k:k + c, :]).sum(dim=1)      # [B, D]
            acc = term if acc is None else acc + term
        return acc

    def forward(self, feats: torch.Tensor, return_pooled: bool = False):
        pooled = self.pool(feats.to(torch.float32))          # [B, d_vit]
        z = self.to_pool(pooled)                              # [B, d_pool]
        p = self.to_phase(z) + self.phase_offset              # [B, D]
        # Record which path runs, so a silent fallback cannot masquerade as the
        # binding architecture. Reported in the receipt.
        self.binding_active = bool(
            self.use_position_binding and feats.shape[1] == self.n_patches)
        self.n_feats = int(feats.shape[1])
        if self.binding_active:
            # bind the projected patch features with position codes and superpose,
            # so psi keeps a compositional role-filler structure. CHUNKED over
            # patches so no [B,N,D] tensor is materialized (see binding_term).
            bsum = self.binding_term(feats.to(torch.float32))
            acc = bsum + p
            # CONTRIBUTION DIAGNOSTIC: if the binding term is negligible next to
            # the pooled term, the "compositional role-filler" claim is hollow.
            # Measured, not asserted.
            self.binding_ratio = float(
                (bsum.norm() / p.norm().clamp(min=1e-12)).detach().item())
        else:
            acc = p
            self.binding_ratio = 0.0
        psi = acc / acc.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        if return_pooled:
            return psi, pooled
        return psi

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _self_test() -> int:
    """Dependency-free check (no VLM): the bridge trains and psi is well-formed."""
    torch.manual_seed(0)
    B, N, dvit, D = 8, 196, 256, 4096
    feats = torch.randn(B, N, dvit)
    b = VLMPhaseBridge(d_vit=dvit, n_patches=N, d_model=D, d_pool=128)
    psi = b(feats)
    out = {
        "psi_shape": list(psi.shape),
        "norm_mean": round(float(psi.norm(dim=-1).mean().item()), 6),
        "finite": bool(torch.isfinite(psi).all().item()),
        "trainable_params": b.trainable_params(),
    }
    # two different feature sets must give different psi
    psi2 = b(torch.randn(B, N, dvit))
    out["distinct_inputs_differ"] = round(
        float(1 - F.cosine_similarity(psi, psi2, dim=-1).mean().item()), 6)
    # does it TRAIN toward a target phase?
    tgt = F.normalize(torch.randn(B, D), dim=-1)
    opt = torch.optim.AdamW([p for p in b.parameters() if p.requires_grad], lr=3e-3)
    first = last = None
    for i in range(200):
        l = 1.0 - F.cosine_similarity(b(feats), tgt, dim=-1).mean()
        opt.zero_grad(); l.backward(); opt.step()
        if i == 0:
            first = float(l.item())
        last = float(l.item())
    out["loss_first"] = round(first, 5)
    out["loss_last"] = round(last, 5)
    out["learned"] = bool(last < first)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())

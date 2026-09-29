"""HENRI adaptive backbone — TRAINABLE path (supersedes the CLASS51 freeze).

WHY THIS EXISTS
===============
`henri_backbone_adapter.py` loads a real Qwen3-VL-8B but then applies the
CLASS51 amendment, which states:

    "Frozen baseline: after load the model is in eval() mode with zero trainable
     parameters. Training/tuning is prohibited by the CLASS51 amendment; this
     adapter never calls .train() or a backward pass."

The same clause appears in `zone_c_world_knowledge_encoder_pin.py`:
    "Zero trainable parameters. Eval-only by policy (CLASS51 amendment)."

That clause forbids LEARNING. A vision-language-action model that cannot be
trained cannot reach state-of-the-art intelligence, so the clause is superseded
by explicit user directive (2026-09-28) and this module is the trainable path.

WHAT IS KEPT (these are soundness, not policy)
==============================================
  * pinned immutable revision + optional per-shard SHA-256 manifest
  * fail-closed typed errors on provenance/input/load failure
  * telemetry that reports trainable vs total parameters, truthfully
  * the FROZEN adapter is NOT deleted or modified -- it stays importable and
    its contract tests keep passing. This module is ADDITIVE.

WHAT CHANGED
============
  * parameters receive gradients; `.train()` is permitted
  * adaptation is LOW-RANK (LoRA) by default, so the base weights stay intact and
    the adaptation is separable, auditable and cheap to store
  * a `frozen` flag is still offered: `freeze_base=True` trains only the adapters,
    which is the recommended configuration at 32 GB

MEMORY (this is why LoRA, not full fine-tuning)
==============================================
Full fine-tuning of an 8B model needs weights+grads+Adam fp32 ~ 16+16+64 GB,
which does not fit 32 GB. LoRA at rank 64 on attention+MLP projections trains
~0.5-2% of parameters with optimizer state in the low hundreds of MB.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

FROZEN_ADAPTER_MODULE = "henri_backbone_adapter"
CLASS51_SUPERSEDED_BY = "user-directive-2026-09-28-trainable-backbone"


class AdaptiveBackboneError(RuntimeError):
    """Base class for adaptive-backbone failures."""


class AdaptiveDisabledError(AdaptiveBackboneError):
    """Raised when HENRI_ADAPTIVE_BACKBONE is not truthy."""


ENV_ENABLE = "HENRI_ADAPTIVE_BACKBONE"
ENV_MODEL_DIR = "HENRI_BACKBONE_MODEL_DIR"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

# Default LoRA targets: attention projections + MLP projections. These cover the
# capacity that matters for adapting a VLM and avoid touching embeddings/output
# head (which would change token semantics).
DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj")


def adaptive_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").strip() in {"1", "true", "True", "yes"}


@dataclass
class AdaptiveTelemetry:
    model_id: str
    revision: str = ""
    lora_rank: int = 0
    lora_alpha: float = 0.0
    lora_targets: list[str] = field(default_factory=list)
    n_lora_modules: int = 0
    total_params: int = 0
    trainable_params: int = 0
    base_frozen: bool = True
    dtype: str = ""
    device: str = ""
    class51_status: str = "SUPERSEDED"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank update: y = Wx + (alpha/r) B A x.

    B is zero-initialised, so the module is EXACTLY the base layer at step 0.
    That makes the adaptation a measurable delta from the pretrained function
    rather than a perturbation of it.
    """

    def __init__(self, base: nn.Linear, r: int = 64, alpha: float = 64.0,
                 dropout: float = 0.0):
        super().__init__()
        if r < 1:
            raise AdaptiveBackboneError("LoRA rank must be >= 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scale = float(alpha) / float(r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.drop(x) @ self.A.t()) @ self.B.t()
        return out + lora * self.scale

    def delta_norm(self) -> float:
        """||B A||_F * scale -- the size of the learned adaptation."""
        with torch.no_grad():
            return float((self.B @ self.A).norm() * self.scale)


def apply_lora(model: nn.Module, r: int = 64, alpha: float = 64.0,
               targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
               dropout: float = 0.0,
               freeze_base: bool = True) -> tuple[int, list[str]]:
    """Replace matching nn.Linear modules with LoRALinear. Returns (count, names)."""
    if freeze_base:
        for p in model.parameters():
            p.requires_grad_(False)
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in targets):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha,
                                                       dropout=dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    return len(replaced), replaced


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


class AdaptiveBackbone:
    """Trainable LoRA-adapted backbone. Additive to the frozen CLASS51 adapter."""

    def __init__(self, model_dir: str | None = None,
                 model_id: str = DEFAULT_MODEL_ID,
                 revision: str = DEFAULT_REVISION,
                 dtype: torch.dtype = torch.bfloat16,
                 device: str | None = None,
                 lora_rank: int = 64, lora_alpha: float = 64.0,
                 lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
                 freeze_base: bool = True,
                 enable: bool | None = None):
        if enable is None:
            enable = adaptive_enabled()
        if not enable:
            raise AdaptiveDisabledError(
                f"set {ENV_ENABLE}=1 to use the trainable backbone")
        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir or os.environ.get(ENV_MODEL_DIR, ""))
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_targets = tuple(lora_targets)
        self.freeze_base = freeze_base
        self.model: nn.Module | None = None
        self.telemetry = AdaptiveTelemetry(
            model_id=model_id, revision=revision, lora_rank=lora_rank,
            lora_alpha=lora_alpha, lora_targets=list(self.lora_targets),
            dtype=str(dtype), device=self.device)

    # ------------------------------------------------------------------ load
    def load(self) -> "AdaptiveBackbone":
        if not self.model_dir.is_dir():
            raise AdaptiveBackboneError(f"model directory not found: {self.model_dir}")
        try:
            from transformers import AutoModelForVision2Seq as _Auto   # noqa: N813
            loader = _Auto
        except Exception:                                              # noqa: BLE001
            try:
                from transformers import AutoModelForCausalLM as loader   # noqa: N813
            except Exception as exc:                                   # noqa: BLE001
                raise AdaptiveBackboneError(
                    f"transformers unavailable: {exc}") from exc
        cfg = self.model_dir / "config.json"
        if cfg.is_file():
            try:
                declared = json.loads(cfg.read_text(encoding="utf-8"))
                mid = declared.get("_name_or_path") or declared.get("model_type")
                if mid:
                    self.telemetry.model_id = str(mid)
            except Exception:                                          # noqa: BLE001
                pass
        self.model = loader.from_pretrained(
            str(self.model_dir), torch_dtype=self.dtype,
            trust_remote_code=False, local_files_only=True)
        n, names = apply_lora(self.model, r=self.lora_rank, alpha=self.lora_alpha,
                              targets=self.lora_targets, freeze_base=self.freeze_base)
        self.telemetry.n_lora_modules = n
        self._lora_names = names
        self.model.to(self.device)
        # TRAINING IS PERMITTED HERE. The frozen adapter's prohibition is
        # superseded; .train() is intentional, not accidental.
        self.model.train()
        total, train = count_params(self.model)
        self.telemetry.total_params = total
        self.telemetry.trainable_params = train
        self.telemetry.base_frozen = self.freeze_base
        return self

    # ------------------------------------------------------- trainable params
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def param_groups(self, lr: float = 1e-4, weight_decay: float = 0.0) -> list[dict]:
        return [{"params": self.trainable_parameters(), "lr": lr,
                 "weight_decay": weight_decay}]

    def optimizer(self, lr: float = 1e-4, weight_decay: float = 0.0):
        if self.model is None:
            raise AdaptiveBackboneError("call load() first")
        return torch.optim.AdamW(self.param_groups(lr, weight_decay), betas=(0.9, 0.999))

    # --------------------------------------------------------------- training
    def train_step(self, batch: dict, loss_fn) -> float:
        """One optimizer step. `loss_fn(model, batch) -> scalar tensor`."""
        if self.model is None:
            raise AdaptiveBackboneError("call load() first")
        self.model.train()
        out = loss_fn(self.model, batch)
        loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        return float(loss.detach().item())

    @torch.no_grad()
    def eval_step(self, batch: dict, loss_fn) -> float:
        if self.model is None:
            raise AdaptiveBackboneError("call load() first")
        self.model.eval()
        out = loss_fn(self.model, batch)
        loss = out[0] if isinstance(out, tuple) else out
        return float(loss.item())

    # -------------------------------------------------------------- adapters
    def lora_state_dict(self) -> dict:
        """The adaptation alone: separable and small, so it can be audited and
        versioned independently of the frozen base."""
        if self.model is None:
            raise AdaptiveBackboneError("call load() first")
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()
                if ".A" in k or ".B" in k}

    def lora_delta_report(self) -> dict:
        if self.model is None:
            raise AdaptiveBackboneError("call load() first")
        norms = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, LoRALinear):
                norms[name] = round(mod.delta_norm(), 6)
        vals = list(norms.values())
        return {"n_modules": len(norms),
                "min_delta": min(vals) if vals else 0.0,
                "max_delta": max(vals) if vals else 0.0,
                "mean_delta": sum(vals) / len(vals) if vals else 0.0,
                "per_module_head": dict(list(norms.items())[:8])}

    def freeze_lora(self) -> None:
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.A.requires_grad_(False)
                m.B.requires_grad_(False)

    def mergeable_state(self) -> dict:
        """Everything needed to rehydrate: base id + revision + LoRA tensors."""
        return {"model_id": self.telemetry.model_id,
                "revision": self.revision,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_targets": list(self.lora_targets),
                "class51_disposition": "SUPERSEDED: training permitted per user directive",
                "superseded_by": CLASS51_SUPERSEDED_BY}


def build_lora_only_self_test(dim: int = 64, r: int = 8, steps: int = 60,
                             device: str = "cpu") -> dict:
    """Dependency-free proof that the LoRA path TRAINS, without any HF model."""
    torch.manual_seed(0)
    base = nn.Linear(dim, dim)
    for p in base.parameters():
        p.requires_grad_(False)
    # the base is deliberately WRONG for the target task; LoRA must fix it
    W = torch.randn(dim, dim, device=device) * 0.3
    X = torch.randn(512, dim, device=device)
    Y = torch.tanh(X @ W)
    model = nn.Sequential(base).to(device)
    n_mods, names = apply_lora(model, r=r, alpha=float(r), targets=("", ),
                               freeze_base=True)
    total, train = count_params(model)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-2)
    first = last = None
    for i in range(steps):
        loss = nn.functional.mse_loss(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = float(loss.item())
        last = float(loss.item())
    base_only = float(nn.functional.mse_loss(base(X), Y).item())
    return {"loss_first": first, "loss_last": last, "frozen_base_only_loss": base_only,
            "loss_ratio_last_over_first": round(last / first, 4),
            "beat_frozen_base": bool(last < base_only),
            "total_params": total, "trainable_params": train,
            "lora_modules": n_mods}


if __name__ == "__main__":
    print(json.dumps({"lora_only_self_test": build_lora_only_self_test()}, indent=2))

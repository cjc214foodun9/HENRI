"""E3 — egress-head reform: probe-distribution readout + generative CE head.

Carrier: carrier/e3-egress-reform (base 5d4a399 = carrier/p1-pk-diagnostics tip).
Prereg: experiments/verification/e3_egress_reform_prereg.md (sealed).

Construct mismatch being reformed (Gate 3.1, #a0a35408): the E2 head was
trained contrastively (window -> teacher-centroid alignment) but Gate 3.1 read
its RAW feature point against the 151,936-token table — a distribution the
head never learned to produce (its own lm_head is untrained).

  Arm A (zero training): k-NN softmax probe (k=16, tau=0.07 — the E2 frozen
          constants) over frozen E2 features -> cluster centroid -> token
          scores cos(z_hat, normalize(E)).
  Arm B (generative): E1 architecture warm-started from the E2 checkpoint,
          trained with CE on the gold next token against a FROZEN TIED teacher
          readout (logits = normalize(feats) @ normalize(E)^T). The head's own
          lm_head is overwritten with the frozen table, excluded from
          trainables, and never called (disclosed dead parameter in the export).

No dense [D, D] operator is formed. No synthetic fallback. This module is never
imported by the production runner (enforced by the e1/e2 contract tests).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from e1_egress_calibration import (
    E1Config,
    E1EgressHead,
    isometry_penalty,
    qr_retraction,
)


def probe_token_distribution(feats: torch.Tensor, E: torch.Tensor,
                             k: int = 16, tau: float = 0.07
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Arm A readout: k-NN softmax cluster centroid -> token scores.

    feats: [B, d_target] head features; E: [V, d_target] frozen teacher table.
    Returns (scores [B, V], z_hat [B, d_target]) with z_hat row-normalized.
    """
    if feats.dim() == 1:
        feats = feats.unsqueeze(0)
    z = F.normalize(feats.float(), p=2, dim=-1)
    En = F.normalize(E.float(), p=2, dim=-1)
    kk = max(1, min(int(k), int(En.shape[0])))
    sims = z @ En.t()                                   # [B, V]
    vals, idx = torch.topk(sims, kk, dim=-1)            # [B, k]
    w = F.softmax(vals / max(float(tau), 1e-12), dim=-1)
    z_hat = (w.unsqueeze(-1) * En[idx]).sum(dim=1)      # [B, d_target]
    z_hat = F.normalize(z_hat, p=2, dim=-1)
    scores = z_hat @ En.t()                             # [B, V]
    return scores, z_hat


class E3GenerativeHead(torch.nn.Module):
    """Arm B: E1 architecture, E2 warm start, FROZEN tied teacher readout."""

    def __init__(self, config: E1Config, E_table: torch.Tensor,
                 device: str = None):
        super().__init__()
        self.config = config
        self.head = E1EgressHead(config)
        E = E_table.detach().clone().to(dtype=torch.float32)
        with torch.no_grad():
            self.head.lm_head.weight.data = E
        self.head.lm_head.weight.requires_grad_(False)
        self.register_buffer("E_table", E)
        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_e2_checkpoint(cls, ckpt_path, E_table: torch.Tensor,
                           device: str = "cpu") -> "E3GenerativeHead":
        """Load the E2 checkpoint ({config, model_state, telemetry}) and warm
        start; the frozen tied readout is re-applied after load."""
        payload = torch.load(str(ckpt_path), map_location="cpu",
                             weights_only=True)
        if not (isinstance(payload, dict) and "model_state" in payload):
            raise ValueError("E2_CKPT_FORMAT_MISMATCH")
        cfg = payload.get("config", {}) or {}
        head_cfg = E1Config(
            d_model=int(cfg.get("d_model", 65536)),
            num_blocks=int(cfg.get("num_blocks", 8192)),
            block_dim=int(cfg.get("block_dim", 8)),
            d_bottleneck=int(cfg.get("d_bottleneck", 256)),
            d_target=int(cfg.get("d_target", 896)),
            vocab_size=int(cfg.get("vocab_size", 151936)),
            seed=int(cfg.get("seed", 20260908)),
            device=str(device),
        )
        head = cls(head_cfg, E_table=E_table, device=device)
        head.head.load_state_dict(payload["model_state"])
        with torch.no_grad():
            head.head.lm_head.weight.data = (
                E_table.detach().clone().to(dtype=torch.float32).to(
                    head.head.lm_head.weight.device))
        head.head.lm_head.weight.requires_grad_(False)
        head.to(device)
        head.eval()
        return head

    # ------------------------------------------------------------------ #
    def forward(self, waves: torch.Tensor):
        """waves [B, num_blocks, block_dim] -> (feats [B, d_target], logits)."""
        return self.head(waves)

    def token_logits(self, waves: torch.Tensor) -> torch.Tensor:
        """Frozen tied readout: cosine scores over the teacher table."""
        feats, _ = self.head(waves)
        z = F.normalize(feats.float(), p=2, dim=-1)
        En = F.normalize(self.E_table.float(), p=2, dim=-1)
        return z @ En.t()

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Adapter params only (excludes the QR-retracted Stiefel factor and
        the frozen tied lm_head)."""
        return [p for n, p in self.head.named_parameters()
                if ("lm_head" not in n and "stiefel_down_proj" not in n)]

    # ------------------------------------------------------------------ #
    def fit_ce(self, waves: torch.Tensor, golds: torch.Tensor,
               epochs: int = 3, batch: int = 32, lr: float = 3e-4,
               wd: float = 1e-4, iso_weight: float = 0.02,
               stiefel_lr: float = 1e-4, clip: float = 1.0) -> List[float]:
        """CE training on gold next-token ids against the frozen tied readout.

        waves: [N, num_blocks, block_dim] (CPU-resident; batches move to GPU).
        golds: [N] int64 gold token ids. Returns the per-step loss list.
        """
        device = next(self.head.parameters()).device
        self.head.train()
        params = self.trainable_parameters()
        opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(wd))
        n = int(waves.shape[0])
        losses: List[float] = []
        for ep in range(int(epochs)):
            g = torch.Generator().manual_seed(int(self.config.seed) + ep)
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, int(batch)):
                idx = perm[i:i + int(batch)]
                opt.zero_grad(set_to_none=True)
                feats, _ = self.head(waves[idx].to(device))
                z = F.normalize(feats.float(), p=2, dim=-1)
                En = F.normalize(self.E_table.float(), p=2, dim=-1)
                logits = z @ En.t()
                ce = F.cross_entropy(logits, golds[idx].to(device))
                loss = ce + float(iso_weight) * isometry_penalty(
                    self.head.stiefel_down_proj)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip))
                opt.step()
                with torch.no_grad():
                    gg = self.head.stiefel_down_proj.grad
                    if gg is not None:
                        self.head.stiefel_down_proj.copy_(qr_retraction(
                            self.head.stiefel_down_proj
                            - float(stiefel_lr) * gg))
                        self.head.stiefel_down_proj.grad = None
                losses.append(float(loss.detach()))
        self.head.eval()
        return losses

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def isometry_error(self) -> float:
        w = self.head.stiefel_down_proj
        gram = w.t() @ w
        eye = torch.eye(gram.size(0), device=w.device, dtype=w.dtype)
        return float((gram - eye).norm(p="fro").item())

    @torch.no_grad()
    def rotation_changed(self, eval_pairs: Sequence, seed: int = 7,
                         probes: int = 16, device: str = "cuda") -> int:
        """G4: seeded per-block orthogonal rotation must change the token argmax
        on >= 8/16 probes (row-norm-preserving transform of the input wave)."""
        g = torch.Generator().manual_seed(int(seed))
        changed = 0
        for p in list(eval_pairs)[:int(probes)]:
            w0 = p.wave if hasattr(p, "wave") else p
            w0 = w0.to(device).unsqueeze(0)                     # [1, K, 8]
            i0 = int(self.token_logits(w0).argmax(-1).item())
            R = torch.randn(8, 8, generator=g, device="cpu")
            Q, _ = torch.linalg.qr(R)
            Q = Q.to(device)
            wrot = torch.einsum("ij,bkj->bki", Q, w0)
            i1 = int(self.token_logits(wrot).argmax(-1).item())
            if i1 != i0:
                changed += 1
        return changed

    # ------------------------------------------------------------------ #
    def export(self, out_dir, name: str = "e3_egress_production.pt"):
        """Fail-closed export (caller decides); returns (path, sha256)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        payload = {
            "config": {k: (v.item() if torch.is_tensor(v) else v)
                       for k, v in vars(self.config).items()},
            "model_state": self.head.state_dict(),
            "telemetry": [],
            "e3": {"arm": "B_generative", "tied_frozen_readout": True,
                   "lm_head_dead_param": True},
        }
        torch.save(payload, str(path))
        return path, hashlib.sha256(Path(path).read_bytes()).hexdigest()

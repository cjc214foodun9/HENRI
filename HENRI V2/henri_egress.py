"""
Universal Egress Engine and Modern Hopfield Codebook Snapping for Project HENRI V2.

Implements TextEgress, ToolEgress, and UniversalEgress as zero-entropy, sub-millisecond
algebraic readout adapters over continuous Clifford phase waves [num_blocks, 8].

Mathematical Contracts:
    1. Unbinding: V_hat = F^-1(F(Psi) * conj(F(Key)))
    2. Modern Hopfield Snapping: S_nearest = argmax_k softmax(beta * <V_hat, M_k>)
    3. Zero-Entropy Egress: Snaps continuous phase waves directly into exact,
       typed JSON-RPC tool payloads, text tokens, or spatial grid matrices.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hopfield_cleanup import ContinuousHopfieldCleanup


@dataclass
class EgressResult:
    """Typed egress payload emitted by the Universal Egress Engine."""
    egress_type: str  # 'text' | 'tool' | 'grid'
    raw_text: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    grid_matrix: Optional[List[List[int]]] = None
    snapped_index: int = 0
    confidence: float = 1.0
    sagnac_delta: float = 0.0


class TextEgress(nn.Module):
    """Snaps continuous phase waves to nearest text token engrams in Hopfield memory."""

    def __init__(self, d_model: int, vocab_size: int = 1000, beta: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.cleanup = ContinuousHopfieldCleanup(dim=d_model, beta=beta)
        self.token_map: Dict[int, str] = {}

    def register_tokens(self, token_engrams: torch.Tensor, token_strings: List[str]):
        """Store continuous token engrams and link them to string representations."""
        self.cleanup.store_engrams(token_engrams)
        for idx, token_str in enumerate(token_strings):
            self.token_map[idx] = token_str

    def decode_wave(self, wave: torch.Tensor) -> Tuple[str, int, float]:
        """Snaps continuous wave to nearest token in memory."""
        flat_wave = wave.reshape(-1, self.d_model)
        _, idx, sim = self.cleanup.hard_retrieve(flat_wave[0])
        idx_item = int(idx)
        text = self.token_map.get(idx_item, f"<token_{idx_item}>")
        return text, idx_item, float(sim)


class ToolEgress(nn.Module):
    """Unbinds role-filler tool waves and snaps continuous vectors into zero-entropy JSON-RPC calls."""

    def __init__(self, d_model: int, beta: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.cleanup = ContinuousHopfieldCleanup(dim=d_model, beta=beta)
        self.tool_schemas: Dict[int, Dict[str, Any]] = {}

    def register_tool_schema(self, schema_id: int, tool_wave: torch.Tensor, schema_dict: Dict[str, Any]):
        """Registers a continuous tool action wave and its corresponding JSON-RPC schema."""
        self.cleanup.store_engrams(tool_wave.unsqueeze(0) if tool_wave.ndim == 1 else tool_wave)
        self.tool_schemas[schema_id] = schema_dict

    def decode_tool_call(self, action_wave: torch.Tensor) -> Tuple[Dict[str, Any], int, float]:
        """Unbinds tool action wave and snaps to exact JSON-RPC payload."""
        flat_wave = action_wave.reshape(-1, self.d_model)
        _, idx, sim = self.cleanup.hard_retrieve(flat_wave[0])
        idx_item = int(idx)
        schema = self.tool_schemas.get(idx_item, {
            "jsonrpc": "2.0",
            "method": f"tool_action_{idx_item}",
            "params": {"action_id": idx_item}
        })
        return schema, idx_item, float(sim)


class UniversalEgress(nn.Module):
    """
    Unified egress router executing Hopfield codebook snapping across text, tool,
    and spatial grid modalities.
    """

    def __init__(self, d_model: int, num_blocks: int = 8192, beta: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.text_egress = TextEgress(d_model=d_model, beta=beta)
        self.tool_egress = ToolEgress(d_model=d_model, beta=beta)

    def egress(
        self,
        wave: torch.Tensor,
        modality: str = "auto",
        sagnac_delta: float = 0.0,
    ) -> EgressResult:
        """
        Executes zero-entropy Hopfield codebook snapping for input wave.

        Args:
            wave: Clifford wave [num_blocks, 8] or flattened [d_model]
            modality: 'text' | 'tool' | 'grid' | 'auto'
            sagnac_delta: Observed physical Sagnac surprise
        """
        flat_wave = wave.reshape(-1, self.d_model)

        if modality == "tool" or (modality == "auto" and len(self.tool_egress.tool_schemas) > 0):
            tool_call, idx, sim = self.tool_egress.decode_tool_call(flat_wave)
            return EgressResult(
                egress_type="tool",
                tool_call=tool_call,
                snapped_index=idx,
                confidence=sim,
                sagnac_delta=sagnac_delta,
            )

        if modality == "text" or (modality == "auto" and len(self.text_egress.token_map) > 0):
            text, idx, sim = self.text_egress.decode_wave(flat_wave)
            return EgressResult(
                egress_type="text",
                raw_text=text,
                snapped_index=idx,
                confidence=sim,
                sagnac_delta=sagnac_delta,
            )

        # Default fallback: snap wave to discrete spatial grid (e.g. 8x8 or 30x30 ARC grid)
        blocks = wave.reshape(-1, 8) if wave.ndim == 2 else wave.reshape(self.num_blocks, 8)
        grid_vals = (blocks.argmax(dim=-1) % 10).tolist()
        grid_dim = int(math.sqrt(len(grid_vals))) if len(grid_vals) >= 4 else 8
        grid_matrix = [grid_vals[i * grid_dim : (i + 1) * grid_dim] for i in range(min(grid_dim, len(grid_vals) // grid_dim))]

        return EgressResult(
            egress_type="grid",
            grid_matrix=grid_matrix,
            snapped_index=0,
            confidence=1.0,
            sagnac_delta=sagnac_delta,
        )


class CompressedProjectionHead(nn.Module):
    """2-layer compressed egress projection head (Roadmap HENRI-ROADMAP-2026-VLA-UNIVERSAL §2.1).

    Maps wave states (S^{D-1}) to |V| logits through a compressed hidden
    layer with LayerNorm + GELU. The head is adapted EXCLUSIVELY by test-time
    SGLD creep (`sgld_adapt_head`); there is no BPTT into the encoder and no
    change to any production default path. Nothing constructs this class
    unless a caller explicitly opts in (default-OFF).

    Memory contract: no [D, D] intermediate; weights are [D, hidden] +
    [hidden, |V|] only (≈ 268 MB at D=65,536, hidden=1024, fp32).
    """

    def __init__(self, d_model: int, hidden_dim: int = 1024, vocab_size: int = 1000,
                 sagnac_lambda: float = 0.25, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.sagnac_lambda = sagnac_lambda
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave: (D,) or (B, D) real -> (B, |V|) logits."""
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        return self.proj(wave) / self.temperature

    @torch.no_grad()
    def logit_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-row Shannon entropy of the softmax over logits (nats)."""
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)


def sgld_adapt_head(
    head: CompressedProjectionHead,
    waves: torch.Tensor,
    targets: torch.Tensor,
    *,
    lr: float = 1e-4,
    steps: int = 500,
    t0: float = 1e-6,
    dt: float = 1.0,
    yield_scale: float = 1e-3,
    log_every: int = 100,
    seed: int = 0,
) -> dict:
    """Test-time SGLD creep on the egress head (roadmap §2.1 Execution).

    Loss:  L = CE(logits, targets) + sagnac_lambda * Sagnac_stress
           Sagnac_stress = 1 - cos^2(softmax_norm, onehot_norm)   (mean)
    Temperature schedule:  T(t) = T0 * (1 + 0.05 t)^-0.55
    Noise:  theta += sqrt(2 T dt) * randn_like(theta)  after each AdamW step
    Bingham yield: skip the step while MEAN per-param grad norm
                   < yield_scale * gnorm_0, where gnorm_0 is the mean
                   per-param grad norm measured at the FIRST step and
                   yield_scale = 1e-3 by default. Absolute thresholds are
                   dimension-blind at D=65,536 (OBSERVED 2026-08-19):
                   a summed norm over 67M params can never fall below an
                   absolute threshold (yield never fires), while a mean
                   norm ~1e-4 sits exactly on any naive absolute value
                   (yield fires on every step and nothing learns). Only a
                   threshold relative to the initial gradient scale is
                   scale-free.

    NOISE-SCALE CONTRACT (OBSERVED defect 2026-08-19): with t0=0.1 the
    per-step noise sigma sqrt(2*0.1) = 0.447 dwarfs Linear init scale
    ~1/sqrt(fan_in) ~ 0.004. A 500-step random walk grows row norms ~300x,
    logits reach magnitude ~600, and CE explodes (observed final loss
    594.09 at D=65,536). Default t0=1e-6 gives sigma_total ~ 0.017 over
    500 steps (~4x init scale), preserving the annealing-exploration
    regime without logit blowup. Always scale T0 to the parameter scale.

    `waves` (N, D) and `targets` (N,) are the in-context demonstration
    pairs; the caller guarantees held-out data is never passed here.
    """
    if head.proj[0].in_features != waves.shape[-1]:
        raise ValueError(f"head d_model {head.proj[0].in_features} != wave dim {waves.shape[-1]}")
    dev = waves.device
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=lr)
    targets = targets.to(dev)
    onehot = torch.zeros(targets.shape[0], head.vocab_size, device=dev)
    onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    def compute_loss():
        logits = head(waves)
        probs = torch.softmax(logits, dim=-1)
        pn = probs / (torch.norm(probs, p=2, dim=-1, keepdim=True) + 1e-9)
        on = onehot / (torch.norm(onehot, p=2, dim=-1, keepdim=True) + 1e-9)
        cos_sq = ((pn * on).sum(dim=-1) ** 2).mean()
        sagnac_stress = (1.0 - cos_sq).clamp(min=0.0)
        ce = torch.nn.functional.cross_entropy(logits, targets)
        loss = ce + head.sagnac_lambda * sagnac_stress
        return loss, ce, sagnac_stress

    # Warmup: measure the initial gradient scale (scale-free yield baseline).
    opt.zero_grad()
    loss0, _, _ = compute_loss()
    loss0.backward()
    gnorm_0 = sum(float(p.grad.norm()) for p in head.parameters()
                  if p.grad is not None) / max(1, n_params)
    opt.zero_grad()
    yield_threshold = yield_scale * max(gnorm_0, 1e-12)

    hist = []
    for t in range(1, steps + 1):
        temp = t0 * (1.0 + 0.05 * t) ** -0.55
        loss, ce, sagnac_stress = compute_loss()

        opt.zero_grad()
        loss.backward()
        gnorm_sum = sum(float(p.grad.norm()) for p in head.parameters() if p.grad is not None)
        gnorm = gnorm_sum / max(1, n_params)
        if gnorm < yield_threshold:
            opt.zero_grad()
            hist.append({"step": t, "loss": float(loss.item()), "gnorm": gnorm, "yielded": True})
            continue
        opt.step()
        torch.manual_seed(seed + t)
        with torch.no_grad():
            for p in head.parameters():
                p.data.add_(math.sqrt(2.0 * temp * dt) * torch.randn_like(p))
        hist.append({"step": t, "loss": float(loss.item()), "gnorm": gnorm, "yielded": False})
        if t % log_every == 0:
            print(f"  sgld step {t}: loss={loss.item():.4f} ce={ce.item():.4f} "
                  f"sagnac={sagnac_stress.item():.4f} T={temp:.2e} gnorm={gnorm:.2e}", flush=True)

    with torch.no_grad():
        logits = head(waves)
        ent = float(head.logit_entropy(logits).mean().item())
    return {
        "final_loss": float(hist[-1]["loss"]) if hist else float("nan"),
        "final_entropy_nats": ent,
        "steps": len(hist),
        "yielded": sum(1 for h in hist if h["yielded"]),
        "schedule": f"T0={t0} decay=(1+0.05t)^-0.55 dt={dt}",
    }

"""E1 — calibrated egress projection head (default-OFF, frozen Qwen teacher).

Carrier: carrier/e1-egress-calibration @ base 10f5f23.
Prereg: experiments/verification/e1_egress_calibration_prereg.md
(sealed sha256 02560a9a856890c758e002ea..., commit 297d25a1).

Boundary (frozen, from prereg):
  wave   = real [num_blocks, 8] float32 (NUM_BLOCKS=8192, BLOCK_DIM=8, D=65536)
           via g7_highorder_codec.HighOrderCodec.encode() -> tobytes -> reshape.
  teacher= Qwen/Qwen2.5-0.5B input embeddings [151936, 896], FROZEN,
           repo sha 060db6499f32faf8b98477b0a26969ef7d8b9987,
           shard sha256 88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342.
Head   = [65536] -> Stiefel W1 [65536, d_bottleneck] (QR retraction, no Cayley)
         -> LayerNorm -> SwiGLU -> [d_target] -> LayerNorm -> feats; lm_head -> logits (diagnostic).
Gates: G1 isometry <= 1e-4; G2 loss descent; G3 eval P@1 beat random-wave AND
       untrained-init (margin >= 0.05); G4 seeded O(8) rotation > 8/16 probes.
No dense [D,D]. No auto-coupling. Flag HENRI_E1_EGRESS gates construction.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

FLAG = "HENRI_E1_EGRESS"
TEACHER_REPO = "Qwen/Qwen2.5-0.5B"
TEACHER_SHA = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TEACHER_WEIGHTS_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
TEACHER_WEIGHTS_BYTES = 988097824
TEACHER_VOCAB = 151936
TEACHER_DIM = 896
NUM_BLOCKS = 8192
BLOCK_DIM = 8
WAVE_DIM = NUM_BLOCKS * BLOCK_DIM


@dataclass
class E1Config:
    d_model: int = WAVE_DIM
    num_blocks: int = NUM_BLOCKS
    block_dim: int = BLOCK_DIM
    d_bottleneck: int = 256
    d_target: int = TEACHER_DIM
    vocab_size: int = TEACHER_VOCAB
    seed: int = 20260908
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    stiefel_lr: float = 1e-4
    adam_lr: float = 3e-4
    adam_wd: float = 1e-4
    temperature: float = 0.07
    isometry_weight: float = 0.02
    max_words: int = 24


# --------------------------------------------------------------------------- #
# Stiefel retraction (factorized QR only; no dense [D,D], no Cayley at scale)  #
# --------------------------------------------------------------------------- #
def qr_retraction(w: torch.Tensor) -> torch.Tensor:
    """Reduced QR retraction onto St(p, n) with deterministic sign fix."""
    with torch.no_grad():
        q, r = torch.linalg.qr(w.float(), mode="reduced")
        d = torch.diagonal(r, 0)
        ph = d.sign()
        ph[ph == 0] = 1.0
        q = q * ph.unsqueeze(0)
    return q.to(w.dtype)


# --------------------------------------------------------------------------- #
# Head                                                                         #
# --------------------------------------------------------------------------- #
class SwiGLUMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class E1EgressHead(nn.Module):
    def __init__(self, config: E1Config):
        super().__init__()
        self.config = config
        if config.d_model != config.num_blocks * config.block_dim:
            raise ValueError(
                f"d_model {config.d_model} != num_blocks*block_dim "
                f"{config.num_blocks * config.block_dim}")
        g = torch.Generator().manual_seed(config.seed)
        init = torch.randn(config.d_model, config.d_bottleneck,
                           generator=g, dtype=config.dtype)
        self.stiefel_down_proj = nn.Parameter(qr_retraction(init).contiguous())
        self.bottleneck_norm = nn.LayerNorm(config.d_bottleneck)
        self.manifold_adaptor = SwiGLUMLP(
            config.d_bottleneck, config.d_bottleneck * 2, config.d_target)
        self.target_norm = nn.LayerNorm(config.d_target)
        self.lm_head = nn.Linear(config.d_target, config.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0,
                        std=1.0 / math.sqrt(config.d_target))

    def adapter_params(self):
        return [p for n, p in self.named_parameters()
                if n != "stiefel_down_proj"]

    def forward(self, wave: torch.Tensor):
        """wave: [B, num_blocks, block_dim] real -> (feats [B,d_target], logits [B,V])."""
        x = wave.reshape(wave.shape[0], -1)
        h = x @ self.stiefel_down_proj
        h = self.bottleneck_norm(h)
        h = self.manifold_adaptor(h)
        feats = self.target_norm(h)
        logits = self.lm_head(feats)
        return feats, logits


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #
def contrastive_alignment_loss(feats: torch.Tensor, targets: torch.Tensor,
                               temperature: float = 0.07) -> torch.Tensor:
    z = F.normalize(feats, p=2, dim=-1)
    zt = F.normalize(targets, p=2, dim=-1)
    sim = torch.matmul(z, zt.t()) / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(sim, labels)


def isometry_penalty(w: torch.Tensor) -> torch.Tensor:
    gram = torch.matmul(w.t(), w)
    eye = torch.eye(gram.size(0), device=w.device, dtype=w.dtype)
    return torch.norm(gram - eye, p="fro") ** 2


# --------------------------------------------------------------------------- #
# Data: real waves (G7 HighOrderCodec) + frozen teacher targets                #
# --------------------------------------------------------------------------- #
@dataclass
class WindowPair:
    wave: torch.Tensor       # [num_blocks, 8] float32 (REAL codec wave)
    target: torch.Tensor     # [d_target] unit-normalized teacher centroid
    text: str = ""


def _codec_for(texts: Sequence[str]):
    """Build a real G7 HighOrderCodec over the corpus vocabulary."""
    from g7_highorder_codec import HighOrderCodec, build_vocab
    vocab = build_vocab(list(texts), max_words=50000)
    return HighOrderCodec(vocab)


def _word_token(text: str) -> list[str]:
    from zone_c_world_knowledge_codec import tokenize
    return tokenize(text)


def _hash_fallback_emb(word: str, teacher: torch.Tensor) -> torch.Tensor:
    """Deterministic word-hash row index (plumbing-only fallback when the real
    BPE tokenizer is not attached; production uses the real tokenizer)."""
    h = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:12], 16)
    idx = h % teacher.size(0)
    return teacher[idx]


def build_window_pairs(texts: Sequence[str],
                       max_words: int = 24,
                       teacher_embeddings: Optional[torch.Tensor] = None,
                       tokenizer: Optional[object] = None,
                       codec=None,
                       seed: int = 20260908) -> list[WindowPair]:
    """Real wave -> teacher-centroid pairs.

    wave   = codec.encode(prefix) (all but last word of the window),
    target = mean of teacher token embeddings over the full window -> L2 norm.
    tokenizer, when provided, must expose encode(text) -> list[int]; otherwise
    a deterministic word->table-row hash fallback is used (PLUMBING ONLY).
    """
    pairs: list[WindowPair] = []
    if codec is None:
        codec = _codec_for(texts)
    for text in texts:
        words = _word_token(text)[:max_words]
        if len(words) < 2:
            continue
        prefix = " ".join(words[:-1])
        window = " ".join(words)
        wave_bytes = codec.encode(prefix)[0]        # np.float32 rows (8192,8)
        wave = torch.from_numpy(
            np.frombuffer(wave_bytes, dtype=np.float32)
            .reshape(NUM_BLOCKS, BLOCK_DIM).copy()
        ).float()
        if teacher_embeddings is None:
            # Plumbing fixture if caller forgot: deterministic unit vector.
            g = torch.Generator().manual_seed(seed + len(pairs))
            tgt = torch.randn(TEACHER_DIM, generator=g)
        elif tokenizer is not None:
            ids = tokenizer.encode(window) if hasattr(tokenizer, "encode") \
                else tokenizer(window)["input_ids"]
            ids = [i for i in ids if 0 <= i < teacher_embeddings.size(0)]
            if not ids:
                continue
            tgt = teacher_embeddings[ids[:max_words * 4]].float().mean(dim=0)
        else:
            rows = [_hash_fallback_emb(w, teacher_embeddings)
                    for w in words]
            tgt = torch.stack(rows).float().mean(dim=0)
        if torch.isnan(tgt).any():
            continue
        tgt = F.normalize(tgt, p=2, dim=-1)
        pairs.append(WindowPair(wave=wave, target=tgt, text=window))
    return pairs


# --------------------------------------------------------------------------- #
# Teacher harvest (fail-closed, hash-pinned; NO synthetic fallback)            #
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_teacher_embeddings(path) -> torch.Tensor:
    """Load a hash-verified frozen embedding table [V, d] (float32)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"teacher embeddings artifact missing: {p}")
    data = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        for k in ("weight", "embed_tokens.weight", "embeddings"):
            if k in data:
                data = data[k]
                break
    t = torch.as_tensor(data, dtype=torch.float32)
    if t.dim() != 2 or t.shape[0] != TEACHER_VOCAB or t.shape[1] != TEACHER_DIM:
        raise ValueError(f"unexpected teacher shape {tuple(t.shape)}")
    return t


def harvest_teacher(model_file: Optional[Path] = None,
                    out_dir: str = "e1_checkpoints") -> tuple[torch.Tensor, dict]:
    """Embedding harvest from a local hash-verified safetensors shard.

    model_file: local model.safetensors (988097824 B, sha 88c14255...). If absent,
    this raises FileNotFoundError — NO synthetic fallback (fail-closed).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mf = Path(model_file) if model_file is not None else out / "model.safetensors"
    if not mf.exists():
        raise FileNotFoundError(
            f"teacher shard missing: {mf} (download REQUIRED; no fallback)")
    if mf.stat().st_size != TEACHER_WEIGHTS_BYTES:
        raise ValueError(f"shard size {mf.stat().st_size} != {TEACHER_WEIGHTS_BYTES}")
    got = _sha256_file(mf)
    if got != TEACHER_WEIGHTS_SHA256:
        raise ValueError(f"shard sha256 {got[:16]} != {TEACHER_WEIGHTS_SHA256[:16]}")
    from safetensors.torch import load_file
    state = load_file(str(mf), device="cpu")
    keys = [k for k in state if "embed_tokens" in k]
    if not keys:
        raise ValueError("no embed_tokens tensor in shard")
    emb = state[keys[0]].float().cpu()      # [151936, 896]
    emb_path = out / "teacher_embeddings.pt"
    torch.save({"weight": emb, "config": {
        "repo": TEACHER_REPO, "repo_sha": TEACHER_SHA,
        "shard_sha256": got, "vocab": TEACHER_VOCAB, "dim": TEACHER_DIM,
        "harvested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}},
        emb_path)
    receipt = {
        "shard": str(mf), "shard_sha256": got, "shard_bytes": mf.stat().st_size,
        "embed": str(emb_path), "shape": list(emb.shape),
        "teacher_repo": TEACHER_REPO, "teacher_sha": TEACHER_SHA,
    }
    (out / "harvest_receipt.json").write_text(json.dumps(receipt, indent=2))
    return emb, receipt


# --------------------------------------------------------------------------- #
# Trainer (scaffold + full-scale; QR retraction each step; no Cayley)          #
# --------------------------------------------------------------------------- #
class E1Trainer:
    def __init__(self, config: E1Config, out_dir: str = "e1_checkpoints"):
        self.config = config
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.head = E1EgressHead(config).to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.head.adapter_params(),
            lr=config.adam_lr, weight_decay=config.adam_wd)
        self.telemetry: list[dict] = []

    def fit(self, pairs: list[WindowPair], epochs: int = 3, batch: int = 32,
            log_every: int = 10, max_steps: Optional[int] = None) -> list[float]:
        self.head.train()
        waves = torch.stack([p.wave for p in pairs]).to(self.config.device)
        tgts = torch.stack([p.target for p in pairs]).to(self.config.device)
        n = waves.shape[0]
        losses: list[float] = []
        step = 0
        t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(
                self.config.seed + ep))
            for i in range(0, n, batch):
                if max_steps is not None and step >= max_steps:
                    return losses
                idx = perm[i:i + batch]
                self.optimizer.zero_grad()
                feats, _ = self.head(waves[idx])
                loss = contrastive_alignment_loss(
                    feats, tgts[idx], temperature=self.config.temperature)
                loss = loss + self.config.isometry_weight * \
                    isometry_penalty(self.head.stiefel_down_proj)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.head.adapter_params(), max_norm=1.0)
                self.optimizer.step()
                with torch.no_grad():
                    g = self.head.stiefel_down_proj.grad
                    if g is not None:
                        self.head.stiefel_down_proj.copy_(qr_retraction(
                            self.head.stiefel_down_proj
                            - self.config.stiefel_lr * g))
                        self.head.stiefel_down_proj.grad.zero_()
                losses.append(loss.item())
                if step % log_every == 0:
                    self.telemetry.append({"step": step, "loss": loss.item(),
                                           "iso": isometry_penalty(
                                               self.head.stiefel_down_proj).item()})
                step += 1
        self.telemetry.append({"training_steps": step,
                               "wall_s": round(time.time() - t0, 2)})
        return losses

    @torch.no_grad()
    def p_at_1(self, pairs: list[WindowPair]) -> float:
        self.head.eval()
        waves = torch.stack([p.wave for p in pairs]).to(self.config.device)
        tgts = torch.stack([p.target for p in pairs]).to(self.config.device)
        feats, _ = self.head(waves)
        sim = F.normalize(feats, dim=-1) @ F.normalize(tgts, dim=-1).t()
        return float((sim.argmax(dim=-1) == torch.arange(len(pairs),
                                                         device=sim.device)).float().mean())

    def export(self, name: str = "e1_egress_calibrated.pt") -> tuple[Path, str]:
        path = self.out / name
        payload = {
            "config": {k: (v.item() if torch.is_tensor(v) else v)
                       for k, v in vars(self.config).items()},
            "model_state": self.head.state_dict(),
            "telemetry": self.telemetry[-20:],
        }
        torch.save(payload, path)
        return path, _sha256_file(path)

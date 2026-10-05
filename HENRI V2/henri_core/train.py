"""Training for the HENRI-Dec-450M text head. Reproducible, minimal, honest.

Objection this file answers
    The prior trainer built labels as `hash(text) % V`. That is NOT reproducible:
    three processes gave 25859 / 13472 / 15667 for the same input. A target that
    moves between runs cannot be learned, and a moving target cannot be gated.

Design
    Labels come from the proprietary ByteBPE tokenizer. Same corpus, same ids, in
    every process. The objective is next-token cross-entropy over the wave of a
    prefix: the decoder learns to read a settled wave as a lexicon. That matches
    doc p29 (decoder = metric Rosetta Stone) and doc p24 (text head, T* = 0.038).

Baselines (both must be reported, neither may be omitted)
    uniform   ln(V)  the do-nothing floor
    unigram   the training token-frequency floor a real model must beat
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .tokenizer import ByteBPE


@dataclass
class TrainReport:
    step: int = 0
    loss_trace: list[float] = field(default_factory=list)
    loss_first: float = 0.0
    loss_last: float = 0.0
    uniform_ce: float = 0.0
    unigram_ce: float = 0.0
    n_examples: int = 0
    vocab_size: int = 0
    note: str = ""

    @property
    def beats_unigram(self) -> bool:
        return self.loss_last < self.unigram_ce

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "loss_first": self.loss_first,
            "loss_last": self.loss_last,
            "loss_trace": self.loss_trace,
            "uniform_ce": self.uniform_ce,
            "unigram_ce": self.unigram_ce,
            "improvement_vs_unigram": self.unigram_ce - self.loss_last,
            "n_examples": self.n_examples,
            "vocab_size": self.vocab_size,
            "beats_unigram": self.beats_unigram,
            "note": self.note,
        }


def build_corpus(texts, tokenizer: ByteBPE, min_prefix: int = 1) -> list[tuple[list[int], int]]:
    """(prefix ids, next id) pairs. Deterministic order, no shuffling by default."""
    pairs: list[tuple[list[int], int]] = []
    for t in texts:
        ids = tokenizer.encode(t)
        for i in range(min_prefix, len(ids)):
            pairs.append((ids[:i], ids[i]))
    return pairs


def unigram_ce(token_stream: list[int], vocab_size: int) -> float:
    """Cross-entropy of the training token frequencies. The honest floor."""
    if not token_stream:
        return math.log(max(vocab_size, 2))
    counts = torch.bincount(torch.tensor(token_stream), minlength=vocab_size).float()
    p = counts / counts.sum().clamp_min(1.0)
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def train_text_head(
    system,
    tokenizer: ByteBPE,
    texts,
    steps: int = 200,
    lr: float = 3e-3,
    batch_pairs: int = 16,
    seed: int = 0,
    freeze_body: bool = True,
    log_every: int = 20,
) -> TrainReport:
    """Train the decoder text head. Returns a measured report.

    freeze_body=True trains only the egress heads: the wave mechanics stay
    training-free, which preserves the zero-pretraining contract for Zones A/B/C.
    """
    pairs = build_corpus(texts, tokenizer)
    if not pairs:
        return TrainReport(note="empty corpus")
    V = max(system.vocab, tokenizer.vocab_size, 2)

    # ---- freeze the geometry
    if freeze_body:
        for p in system.parameters():
            p.requires_grad_(False)
        for p in system.decoder.head_text.parameters():
            p.requires_grad_(True)
        for p in system.decoder.head_action.parameters():
            p.requires_grad_(True)
        for p in system.decoder.head_vision.parameters():
            p.requires_grad_(True)

    params = [p for p in system.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    # ---- baselines, computed from the SAME pairs
    stream = [tok for _, tok in pairs]
    uniform = math.log(V)
    uni = unigram_ce(stream, V)

    # ---- cache the wave of each prefix (the ingress is frozen and deterministic)
    wave_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def wave_of(prefix):
        key = tuple(prefix)
        if key not in wave_cache:
            text = tokenizer.decode(prefix)
            with torch.no_grad():
                wave_cache[key] = system.wave_of(text, tokenizer)
        return wave_cache[key]

    rep = TrainReport(uniform_ce=uniform, unigram_ce=uni,
                      n_examples=len(pairs), vocab_size=V)
    g = torch.Generator().manual_seed(seed)
    system.train()
    for step in range(int(steps)):
        idx = torch.randint(0, len(pairs), (min(batch_pairs, len(pairs)),), generator=g)
        waves, targets = [], []
        for j in idx.tolist():
            prefix, nxt = pairs[j]
            waves.append(wave_of(prefix))
            targets.append(min(nxt, V - 1))
        psi = torch.stack(waves)
        y = torch.tensor(targets, dtype=torch.long)
        out = system.decoder(psi)
        loss = F.cross_entropy(out["text_logits"], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        rep.loss_trace.append(lv)
        rep.step = step + 1
        if log_every and (step % log_every == 0 or step == steps - 1):
            # D81 (self-caught): progress must not share stdout with the JSON
            # payload. The first run corrupted the receiving json.load at
            # line 1, char 2 -- exactly the `  step` prefix. Machine output on
            # stdout, human progress on stderr.
            print(f"  step {step:>4}  ce {lv:.4f}  (uniform {uniform:.4f}, "
                  f"unigram {uni:.4f})", file=sys.stderr)
    system.eval()
    if rep.loss_trace:
        rep.loss_first = rep.loss_trace[0]
        rep.loss_last = rep.loss_trace[-1]
    rep.note = ("head-only training; Zones A/B/C remain training-free"
                if freeze_body else "full-parameter training")
    return rep

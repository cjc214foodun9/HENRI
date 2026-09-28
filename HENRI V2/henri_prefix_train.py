"""Prefix-projection training (Directive 2), with the capability question MEASURED.

THE DIRECTIVE, AND WHAT IS ACTUALLY TRUE
========================================
The directive reads: "Train the prefix adapter on input-output program pairs so that
wave representations in Zone B translate into valid causal attention keys and values."

MEASURED FACTS THAT CONSTRAIN IT (my own probes, not assumptions):
  * `henri_decoder.py` has NO attention core: zero occurrences of q_proj, k_proj,
    v_proj, MultiheadAttention, scaled_dot_product, past_key_values, causal_mask. So
    there are no "keys and values" to produce. That half of the directive is NOT
    achievable by training and is not claimed here.
  * `PrefixConditioner` exposes exactly ONE trainable parameter:
        self.prefix_gain = nn.Parameter(torch.zeros(self.d_hidden))
    It pools a prefix to [B, d_hidden] and adds `pooled * gain` to the hidden state.
    A per-dimension GAIN carries no projection capacity: it can only rescale a pooled
    bias that already lives in the hidden space.

This module therefore does the honest version: it TRAINS the prefix path on program
pairs and MEASURES whether the available capacity is sufficient. Three arms:

    OFF        prefix conditioning disabled (the untrained default; a control)
    GAIN       only `prefix_gain` trainable      (the capacity the module has today)
    GAIN+PROJ  `prefix_gain` + `down_proj` trainable (the capacity the task may need)

PRE-REGISTERED BAR (fixed here, before the run)
==============================================
  A trained arm PASSES iff its HELD-OUT loss beats the OFF control by >= `tau`
  (default 1e-2), on pairs never seen in training.

ANTI-VACUITY CONTROL (the one that matters)
===========================================
  SHUFFLED-TARGET arm: identical training procedure with the targets DERANGED. If a
  trained arm improves held-out loss under deranged targets too, the gain is learning
  a generic bias rather than the pair relation, and NO arm may be credited.

HONEST BOUNDARIES
=================
* SCAFFOLD SCALE. The decoder is constructed small (d_model=1024, vocab_size=256) with
  `checkpoint_policy="disabled"`, because this measures the MECHANISM. It is not the
  799 MB production backbone, and no production claim follows from it.
* No attention, no KV cache, no ARC/SciCode score is claimed. The only claim is the
  measured held-out loss delta between arms on this scaffold.
* A NEGATIVE result is a valid, useful outcome: it bounds the module's capacity and
  names the change that would be required.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

TRAIN_MODE_OFF = "OFF"
TRAIN_MODE_GAIN = "GAIN"
TRAIN_MODE_GAIN_PROJ = "GAIN+PROJ"
TRAIN_MODES = (TRAIN_MODE_OFF, TRAIN_MODE_GAIN, TRAIN_MODE_GAIN_PROJ)


class PrefixTrainError(RuntimeError):
    """Fail-closed contract violation."""


@dataclass
class PrefixTrainConfig:
    d_model: int = 1024
    d_hidden: int = 128
    vocab_size: int = 256
    n_train: int = 256
    n_heldout: int = 128
    seq_len: int = 12
    epochs: int = 40
    lr: float = 3e-3
    weight_decay: float = 0.0
    tau: float = 1e-2
    seed: int = 20260927
    device: str = "cpu"

    def validate(self) -> "PrefixTrainConfig":
        if self.d_model < 8 or self.d_hidden < 2 or self.vocab_size < 4:
            raise PrefixTrainError("dims too small")
        if self.n_train < 8 or self.n_heldout < 4:
            raise PrefixTrainError("need enough pairs to split")
        if self.epochs < 1 or self.lr <= 0:
            raise PrefixTrainError("bad optimisation settings")
        if self.tau < 0:
            raise PrefixTrainError("tau must be non-negative")
        return self


# --------------------------------------------------------------------- task
def _pairs(n: int, seq_len: int, vocab: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """Deterministic program pairs: y = reverse(x) then +1 (mod vocab).

    A fixed, learnable input->output RELATION. Small enough to be learnable, non-trivial
    enough that a generic bias cannot satisfy it -- which is what the shuffled-target
    control verifies.
    """
    g = torch.Generator().manual_seed(seed)
    out: List[Tuple[List[int], List[int]]] = []
    for _ in range(n):
        x = torch.randint(1, vocab, (seq_len,), generator=g).tolist()
        y = [(v + 1) % vocab for v in reversed(x)]
        out.append((x, y))
    return out


def _wave_from_tokens(tokens: Sequence[int], d_model: int, vocab: int,
                      generator: torch.Generator) -> torch.Tensor:
    """A deterministic wave prefix for a token sequence (stand-in for Zone B ingress).

    Each token contributes a fixed random unit vector at its position; the sequence is
    the normalised sum. This is an ENCODER PLACEHOLDER: it makes the prefix carry the
    input, which is the property the trainer needs. It is not the production encoder.
    """
    half = d_model // 2
    acc = torch.zeros(d_model)
    for t, tok in enumerate(tokens):
        g = torch.Generator().manual_seed(int(tok) * 7919 + t * 104729 + 13)
        ph = torch.rand(half, generator=g) * 2.0 * math.pi
        acc = acc + torch.stack([torch.cos(ph), torch.sin(ph)], dim=-1).reshape(-1)
    n = float(torch.linalg.vector_norm(acc))
    return acc / (n + 1e-12)


def _build_batch(pairs, cfg: PrefixTrainConfig, enc_g: torch.Generator):
    """Return (prefix_waves [N,1,d_model], x_ids, y_ids) tensors."""
    d, s, v = cfg.d_model, cfg.seq_len, cfg.vocab_size
    px = torch.zeros(len(pairs), 1, d)
    xx = torch.zeros(len(pairs), s, dtype=torch.long)
    yy = torch.zeros(len(pairs), s, dtype=torch.long)
    for i, (x, y) in enumerate(pairs):
        px[i, 0] = _wave_from_tokens(x, d, v, enc_g)
        xx[i] = torch.tensor(x, dtype=torch.long)
        yy[i] = torch.tensor(y, dtype=torch.long)
    return px, xx, yy


def _hidden_from_input(unbinder, x_ids: torch.Tensor, d_model: int) -> torch.Tensor:
    """Embed the INPUT tokens into [B, T, d_hidden] via the SAME trained down_proj.

    The unbinder is pointwise, so the "sequence" is supplied here: each input token is
    encoded as a one-hot wave, projected by down_proj. This keeps the prefix path on the
    real weights rather than inventing a second projection.
    """
    B, T = x_ids.shape
    oh = torch.zeros(B, T, d_model)
    oh.scatter_(2, x_ids.unsqueeze(-1) % d_model, 1.0)
    return unbinder.down_proj(oh)


def _arm_trainable(unbinder, mode: str) -> List[nn.Parameter]:
    for p in unbinder.parameters():
        p.requires_grad_(False)
    params: List[nn.Parameter] = []
    if mode == TRAIN_MODE_OFF:
        if unbinder.prefix_cond is not None:
            unbinder.prefix_cond.use_prefix = False
        return params
    if unbinder.prefix_cond is None:
        raise PrefixTrainError("arm %s needs a prefix conditioner" % mode)
    unbinder.prefix_cond.use_prefix = True
    unbinder.prefix_cond.prefix_gain.requires_grad_(True)
    params.append(unbinder.prefix_cond.prefix_gain)
    if mode == TRAIN_MODE_GAIN_PROJ:
        unbinder.down_proj.weight.requires_grad_(True)
        params.append(unbinder.down_proj.weight)
    return params


def _loss(unbinder, prefix, x_ids, y_ids, mode, d_model) -> torch.Tensor:
    h = _hidden_from_input(unbinder, x_ids, d_model)
    if mode != TRAIN_MODE_OFF and unbinder.prefix_cond is not None:
        p = prefix                                   # [B, 1, d_model]
        p = p / (torch.linalg.vector_norm(p, dim=-1, keepdim=True) + 1e-8)
        p_hidden = unbinder.down_proj(p)             # [B, 1, d_hidden]
        h = unbinder.prefix_cond(h, p_hidden)
    logits = unbinder.lm_head(unbinder.act(unbinder.layer_norm(h)))
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y_ids.reshape(-1))


def run_arm(mode: str, cfg: PrefixTrainConfig,
            derange_targets: bool = False) -> Dict[str, object]:
    """Train one arm; return its held-out loss and the diagnostics that bound it."""
    if mode not in TRAIN_MODES:
        raise PrefixTrainError("unknown mode %r" % mode)
    cfg.validate()
    from henri_decoder import HENRINeuralEgressUnbinder

    g = torch.Generator().manual_seed(cfg.seed)
    tr = _pairs(cfg.n_train, cfg.seq_len, cfg.vocab_size, cfg.seed)
    ho = _pairs(cfg.n_heldout, cfg.seq_len, cfg.vocab_size, cfg.seed + 977)
    if derange_targets:
        ys = [y for _x, y in tr]
        tr = [(x, ys[(i + 1) % len(ys)]) for i, (x, _y) in enumerate(tr)]

    torch.manual_seed(cfg.seed)
    ub = HENRINeuralEgressUnbinder(d_model=cfg.d_model, d_hidden=cfg.d_hidden,
                                   vocab_size=cfg.vocab_size, device=cfg.device,
                                   use_prefix=(mode != TRAIN_MODE_OFF))
    params = _arm_trainable(ub, mode)

    px, xx, yy = _build_batch(tr, cfg, g)
    hx, hxx, hyy = _build_batch(ho, cfg, torch.Generator().manual_seed(cfg.seed + 1))

    gain_before = (None if ub.prefix_cond is None
                   else ub.prefix_cond.prefix_gain.detach().clone())
    with torch.no_grad():
        held_before = float(_loss(ub, hx, hxx, hyy, mode, cfg.d_model))

    if params:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        for _ep in range(cfg.epochs):
            opt.zero_grad()
            loss = _loss(ub, px, xx, yy, mode, cfg.d_model)
            loss.backward()
            opt.step()
    with torch.no_grad():
        held_after = float(_loss(ub, hx, hxx, hyy, mode, cfg.d_model))
        train_after = float(_loss(ub, px, xx, yy, mode, cfg.d_model))

    moved = None
    if gain_before is not None:
        moved = float((ub.prefix_cond.prefix_gain.detach() - gain_before).abs().max())
    return {
        "mode": mode, "deranged": bool(derange_targets),
        "n_params": sum(p.numel() for p in params),
        "heldout_before": held_before, "heldout_after": held_after,
        "heldout_delta": held_before - held_after,
        "train_after": train_after,
        "gain_moved": moved,
        "tau": cfg.tau,
    }


def evaluate(cfg: Optional[PrefixTrainConfig] = None) -> Dict[str, object]:
    """Run OFF, GAIN, GAIN+PROJ, plus the shuffled-target control, and adjudicate."""
    cfg = (cfg or PrefixTrainConfig()).validate()
    rows = {m: run_arm(m, cfg) for m in TRAIN_MODES}
    ctrl = {m: run_arm(m, cfg, derange_targets=True)
            for m in (TRAIN_MODE_GAIN, TRAIN_MODE_GAIN_PROJ)}

    off = rows[TRAIN_MODE_OFF]["heldout_after"]
    for m in (TRAIN_MODE_GAIN, TRAIN_MODE_GAIN_PROJ):
        rows[m]["gain_vs_off"] = off - rows[m]["heldout_after"]
        rows[m]["passes_bar"] = bool(rows[m]["gain_vs_off"] >= cfg.tau)
        c = ctrl[m]
        rows[m]["control_gain_vs_off"] = off - c["heldout_after"]
        rows[m]["control_also_improves"] = bool(rows[m]["control_gain_vs_off"] >= cfg.tau)

    credited = [m for m in (TRAIN_MODE_GAIN, TRAIN_MODE_GAIN_PROJ)
                if rows[m]["passes_bar"] and not rows[m]["control_also_improves"]]
    # DEFECT FIXED 2026-09-27: `passes_bar` exists only on the TRAINED arms (OFF is a
    # control and has no bar to pass). Iterating every arm for the verdict raised
    # KeyError after all the work was done. The verdict now consults `trained_only`.
    trained_only = (TRAIN_MODE_GAIN, TRAIN_MODE_GAIN_PROJ)
    if credited:
        verdict = "PREFIX_TRAINING_PASSES_BAR:%s" % credited
    elif any(rows[m].get("passes_bar") for m in trained_only):
        verdict = "FALSIFIED_GAIN_IS_GENERIC_BIAS_CONTROL_ALSO_IMPROVES"
    else:
        verdict = "FALSIFIED_AVAILABLE_CAPACITY_INSUFFICIENT"

    return {
        "schema": "henri.prefix-train.v1",
        "evidence_class": "OBSERVED",
        "scale": "SCAFFOLD (d_model=%d, d_hidden=%d, vocab=%d, checkpoint disabled)"
                 % (cfg.d_model, cfg.d_hidden, cfg.vocab_size),
        "bar": {"tau": cfg.tau, "rule": "heldout gain vs OFF >= tau AND the "
                                        "shuffled-target control must NOT also improve"},
        "arms": rows, "controls": ctrl, "verdict": verdict,
        "credited_arms": credited,
        "not_claimed": ["attention keys/values", "KV cache", "ARC or SciCode score",
                        "production-backbone behaviour"],
    }

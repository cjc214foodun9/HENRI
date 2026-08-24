"""
System-1 v0.4.2 CEGIS Beam-Priority Decoder (Option 1 operational efficacy)
===========================================================================
FAITHFUL integration of the audited upload
(System-1_Kernel_v0.4.2_CEGIS_Engine.py, sha256 cadf9788...) onto the live
v0.4.1 substrate (system1_kernel_v041_energy_refactored.py; calibrated
checkpoint ckpt_v041/checkpoint.pt sha 11d56121..., energy head
Spearman 0.4383 / AUROC 0.7531, label CALIBRATED_DIAGNOSTIC_NOT_PROMOTED).

Audit dispositions (see experiments/system1-v04/SYSTEM1_V04_VERDICT.md):
  - Brier head / dual-rate core / name-cond decoder : ALREADY_IMPLEMENTED
    in live v0.4.1. No new trainable parameters; inference-only.
  - Upload scores energy on RAW TOKEN EMBEDDINGS (encode_tokens(cand_ids)):
    FALSIFIED for this use. The v0.4.1 head was calibrated on CORE-UNROLLED
    latents (eval_calibration steps=8). Corpus consult (ca4bb787, INFERRED):
    probe on raw embeddings is out-of-distribution -> discrimination
    collapses to chance (AUROC ~ 0.50). This decoder unrolls the core per
    candidate (8 steps, the calibration family).
  - Upload's __main__ (torch.randint dummy): FALSIFIED as evidence (mock
    loop). Smoke-only.
  - "25.86M params" claim: FALSIFIED (32k-vocab artifact). Live vocab ~90
    -> 2.14M + 49.7K energy head.
  - beta_priority = 0.40: PRE-REGISTERED HYPOTHESIS value from the roadmap.
    Single value, no post-hoc tuning on any eval split.

Faithfulness requirements (reference system1-outcome-energy-calibration.md):
  - Per-candidate core-unrolled latent: encode_tokens(cand_ids) -> 8 core
    steps -> model.energy. Same latent family the head was calibrated on.
  - Token-FSA mask at every step (live beam_decode semantics).
  - EOS handling identical to live System1KernelV04.beam_decode.
  - beta=0.0 => selection byte-identical to model.beam_decode (same
    cumulative log-prob scores, same top-4 expansion, same tie order).
  - Energy computed for BOTH arms (matched compute; enables candidate-level
    energy/outcome association for the beta=0 control too).
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, System1KernelV04)

CORE_STEPS = 8          # calibration-family latent depth (eval_calibration)
ENERGY_EPS = 1e-5


class CEGISBeamPriorityDecoder:
    """Energy-prioritized constrained beam decode. Inference-only, frozen.

    priority(cand) = (1 - beta) * cum_log_prob(cand)
                   + beta * log(pass_prob(cand) + ENERGY_EPS)

    beta=0.0  -> standard beam search (byte-identical to beam_decode).
    beta=0.40 -> CEGIS Brier-prioritized search (pre-registered value).
    """

    def __init__(self, model: System1KernelV04, core_steps: int = CORE_STEPS):
        self.model = model
        self.core_steps = core_steps

    @torch.no_grad()
    def candidate_energy(self, seq_ids: list[int], dev) -> float:
        """Core-unrolled candidate latent energy in [0,1] (calibration family)."""
        ids = torch.tensor([seq_ids], dtype=torch.long, device=dev)
        z = self.model.encode_tokens(ids)
        slow_cache = None
        for t in range(self.core_steps):
            z, slow_cache = self.model.core(z, t, slow_cache)
        e = float(self.model.energy(z).item())
        if math.isnan(e) or math.isinf(e):
            return 0.5                       # fail-safe, never 1.0
        return min(1.0, max(0.0, e))

    @torch.no_grad()
    def decode_cegis_beam(
        self,
        z0: torch.Tensor,                    # [1, slots, d] signature latent
        s_prompt: torch.Tensor,              # [1, K, d] prompt memory
        beam_width: int = 16,
        max_len: int = 48,
        beta_priority: float = 0.0,
    ) -> tuple[list[int], dict]:
        """Return (best_seq_ids, record). record holds engagement data and the
        final top-4 candidates (ids, energy) for sandbox-level association."""
        model = self.model
        dev = z0.device
        h = model.decoder.init_proj(z0.mean(dim=1)).expand(beam_width, -1).clone()
        toks = torch.full((beam_width,), TOK2ID["BOS"], device=dev)
        depth = torch.zeros(beam_width, dtype=torch.long, device=dev)
        seqs: list[list[int]] = [[] for _ in range(beam_width)]
        scores = torch.zeros(beam_width, device=dev)
        ener: list[list[float]] = [[] for _ in range(beam_width)]

        for _ in range(max_len):
            raw_t, h_new = [], []
            for i in range(beam_width):
                # s_prompt[0:1]: identical memory for every beam particle
                # (live beam_decode broadcast guard; never slice [i:i+1]).
                r, hh = model.decoder.step(
                    model.token_emb(toks[i:i + 1]), h[i:i + 1], s_prompt[0:1])
                raw_t.append(r)
                h_new.append(hh)
            raw_t = torch.cat(raw_t)
            h = torch.cat(h_new)
            mask_t, _ = model.fsa.apply_mask(raw_t, toks, depth)
            logp = F.log_softmax(mask_t, -1)

            cand = []                       # (score, beam_i, token, energy)
            for i in range(beam_width):
                topv, topi = logp[i].topk(4)
                for k in range(4):
                    e = self.candidate_energy([TOK2ID["BOS"]] + seqs[i]
                                              + [topi[k].item()], dev)
                    sc = scores[i].item() + topv[k].item()
                    if beta_priority > 0.0:
                        sc = ((1.0 - beta_priority) * sc
                              + beta_priority * math.log(e + ENERGY_EPS))
                    cand.append((sc, i, topi[k].item(), e))
            cand.sort(reverse=True)
            cand = cand[:beam_width]

            new_toks, new_depth, new_seqs, new_scores, new_ener = \
                [], [], [], [], []
            for sc, i, t, e in cand:
                new_toks.append(t)
                new_depth.append(depth[i].item() + model.fsa.depth_delta(
                    torch.tensor(t, device=dev)).item())
                new_seqs.append(seqs[i] + [t])
                new_scores.append(sc)
                new_ener.append(ener[i] + [e])
            toks = torch.tensor(new_toks, device=dev)
            depth = torch.clamp(torch.tensor(new_depth, device=dev), min=0)
            seqs, scores = new_seqs, torch.tensor(new_scores, device=dev)
            ener = new_ener
            if all(s and s[-1] == TOK2ID["EOS"] for s in seqs):
                break

        order = sorted(range(beam_width),
                       key=lambda i: (scores[i].item(), -i), reverse=True)
        best = seqs[order[0]]
        finals = [(seqs[i], ener[i][-1] if ener[i] else None)
                  for i in order[:4]]
        return best, {
            "final_candidates": finals,
            "best_score": float(scores[order[0]].item()),
            "best_energy": ener[order[0]][-1] if ener[order[0]] else None,
            "energies_note": "per-candidate core-unrolled latent, 8 steps",
        }

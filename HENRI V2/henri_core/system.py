"""HENRI Tri-Model System: composition of the three micro-models.

Document anchors:
  doc p32-35 : the end-to-end cycle
    Zone A ingress -> Zone C prefetch -> Zone B swarm (48.5 us) -> Decoder
    -> Sagnac homodyne veto

Contract
  Zone A  CliffordVLASlotEncoder   text/image/action -> unit-norm wave
  Zone B  SwarmWaveResonator       saddle traversal, returns psi_converged
  Zone C  HenriMem65M              spectral ingress, Gram compaction, prefetch
  Decoder HenriDec450M             macro-tokens -> typed egress
  Veto    SagnacHomodyneVeto       fail-closed interference check

The ZERO-PRETRAINING contract (USER.md) and the operator's "hyper fine tuned"
directive resolve here: the wave mechanics stay training-free, and only the
decoder's typed heads carry learned parameters. That matches doc p31: "Stage 0
Tabula-Rasa Pretraining sets D_c = 0" while the decoder is "a coordinate
transformer" calibrated on a small grounding dictionary (doc p29).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import substrate as sub
from . import novelty_gate as NG
from .model1_swarm import SwarmConsensusVeto, SwarmWaveResonator
from .model2_memory import HenriMem65M
from .model3_decoder import DecoderConfig, HenriDec450M
from .sagnac_homodyne import SagnacHomodyneVeto
from .zone_a import CliffordVLASlotEncoder

# Model IDs, one term per meaning
MODEL_1_ID = "zone_b_viscoelastic_swarm"
MODEL_2_ID = "henri-mem-65m"
MODEL_3_ID = "henri-dec-450m"


class TriModelSystem(nn.Module):
    """The complete 100% proprietary HENRI model.

    Args:
        dim:        wave dimension D (65536 full; small in tests)
        vocab:      tokenizer vocabulary size
        n_workers:  Zone B probe count (doc: 256-512)
        decoder_cfg: HENRI-Dec-450M sizing
        small:      if True, shrink every component for a CPU smoke test
    """

    def __init__(self, dim: int = sub.DEFAULT_DIM, vocab: int = 512,
                 n_workers: int = 256, steps: int = 16, beta: float = 26.10,
                 n_axioms: int = 8, seed: int = 20261004,
                 positional: bool = True, pos_multifreq: bool = False,
                 pos_block: int = 16, pos_rope_theta: float = 5.0e5,
                 ingress_seed: int | None = None,
                 dk_target: int = 32,
                 beta_scale: float = 1.0,
                 decoder_cfg: DecoderConfig | None = None, small: bool = False):
        super().__init__()
        if small:
            dim, n_workers, steps = 4096, 8, 4
            decoder_cfg = DecoderConfig(
                dim=dim, d_model=128, n_layers=2, n_heads=4, n_kv_heads=1,
                d_ffn=256, n_macro=16, n_invariants=32, vocab=vocab,
                dk_target=dk_target)
        self.dim = int(dim)
        self.vocab = int(vocab)
        self.small = bool(small)

        # ---- MODEL 1: Zone B swarm
        self.swarm = SwarmWaveResonator(
            dim=dim, n_workers=n_workers, steps=steps, beta=beta, seed=seed,
            beta_scale=beta_scale)
        self.consensus = SwarmConsensusVeto(min_delta_h=0.15)
        # ---- MODEL 2: HENRI-Mem-65M
        self.memory = HenriMem65M(dim=dim) if not small else _small_memory(dim)
        # ---- MODEL 3: HENRI-Dec-450M
        self.decoder = HenriDec450M(decoder_cfg or DecoderConfig(dim=dim, vocab=vocab))
        # ---- shared ingress and veto
        self.ingress = CliffordVLASlotEncoder(dim=dim, vocab=vocab,
                                              positional=positional,
                                              pos_multifreq=pos_multifreq,
                                              pos_block=pos_block,
                                              pos_rope_theta=pos_rope_theta,
                                              ingress_seed=ingress_seed)
        self.veto = SagnacHomodyneVeto(dim=dim)

        # Axiomatic baseplate: seeded wave bank, pinned by hash of its ids
        g = torch.Generator().manual_seed(seed)
        raw = torch.randn(n_axioms, dim, generator=g).to(torch.complex64)
        self.register_buffer("axiom_bank", sub.unit_norm(raw))

    # ------------------------------------------------------------------ ingress
    def wave_of(self, text: str, tokenizer) -> torch.Tensor:
        """Zone A: text -> unit-norm wave [D]."""
        with torch.no_grad():
            return self.ingress.encode_text(text, tokenizer)

    @torch.no_grad()
    def build_axioms(self, texts, tokenizer):
        """Pin the axiomatic baseplate from texts. Returns the wave bank.

        D-BANK (measured): this used to update ONLY the veto. The swarm kept
        matching against `axiom_bank`, which is 8 RANDOM unit vectors seeded at
        __init__ (measured: identical before/after this call, bank-vs-corpus
        |cos| max 0.0418 ~ 4/sqrt(D)). The swarm therefore resolved every input
        against semantically arbitrary patterns. Store the corpus waves so the
        swarm can use them when asked; default behaviour is unchanged.
        """
        waves = torch.stack([self.wave_of(t, tokenizer) for t in texts])
        self.veto.load_axioms(waves)
        self.axiom_waves = waves
        return waves

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def solve(self, prompt: str, tokenizer, patterns: torch.Tensor | None = None,
              use_swarm: bool = True, temperature: float | None = None,
              swarm_bank: str = "corpus",
              abstain_below: float | None = None,
              novelty_lambda: float = 0.0) -> dict:
        """Run the full closed loop on one prompt.

        Returns the decoded tokens, the converged wave, the Sagnac verdict, and
        the memory diagnostics. Nothing is dispatched when the veto is dark.

        swarm_bank: "corpus" (DEFAULT as of the approved D2 flip) uses the seeded
                    random bank. "corpus" uses the waves stored by
                    build_axioms. Measured effect on 6 distinct queries:
                    random -> 1 distinct answer, corpus -> 3 distinct answers.

        abstain_below: DEFAULT None (OFF). When set, solve() reads the
                    PRE-projection membership score s = max_k |<psi_in, bank_k>|
                    and ABSTAINS if s < abstain_below. The signal is graded, not a
                    predicate; see henri_core/novelty_gate.py and
                    design/zone_a/evidence/henri_q4_discriminative_receipt.json.
                    Not enabled by default: abstention is a behaviour change.
        """
        psi_in = self.wave_of(prompt, tokenizer).unsqueeze(0)
        if patterns is not None:
            bank = patterns
        elif swarm_bank == "corpus":
            bank = getattr(self, "axiom_waves", None)
            if bank is None:
                raise RuntimeError("swarm_bank='corpus' requires build_axioms first")
        else:
            bank = self.axiom_bank

        # --- D-Q4: PRE-projection membership score (the discriminative signal)
        # Measured AFTER the swarm, this statistic is ~0.99 for every input (the
        # CCCP projects any wave onto the bank). Read BEFORE, it discriminates.
        nov_bank = getattr(self, "axiom_waves", None)
        novelty = (NG.membership_score(psi_in[0], nov_bank)
                   if nov_bank is not None and nov_bank.numel() > 0 else None)

        if abstain_below is not None and novelty is not None and novelty < abstain_below:
            return {
                "prompt": prompt,
                "abstained": True,
                "reason": (f"membership {novelty:.6f} < abstain_below "
                           f"{abstain_below}"),
                "novelty": {"score": round(float(novelty), 8),
                            "threshold": abstain_below,
                            "bank": "axiom_waves", "n": int(nov_bank.shape[0])},
                "psi_in": psi_in[0],
                "models_engaged": [],
            }

        # --- MODEL 2: Zone C maintenance + prefetch on the incoming wave
        mem = self.memory(psi_in)

        # --- MODEL 1: Zone B swarm traversal
        if use_swarm:
            swarm_out = self.swarm(psi_in[0], patterns=bank)
            winner = self.consensus(swarm_out)
            psi_conv = winner["psi"].unsqueeze(0)
            swarm_diag = {
                "energy": winner["energy"],
                "delta_h": winner["delta_h"],
                "distinct_seeds": swarm_out["distinct_seeds"],
                "monotone_frac": self.swarm.monotone_fraction(swarm_out["energies"]),
            }
        else:
            psi_conv = psi_in
            swarm_diag = {"energy": None, "delta_h": None,
                          "distinct_seeds": 0, "monotone_frac": None}

        # --- MODEL 3: decoder
        ids, probs = self.decoder.snap_text(psi_conv, temperature=temperature)
        decoded = tokenizer.decode(ids.tolist())

        # --- Sagnac veto
        # Phase 1: the Q4 membership score s(q) enters as a novelty penalty.
        # s(q) is computed against the VETO's own axiom bank, so one bank is used
        # consistently. novelty_lambda=0.0 (DEFAULT) leaves the verdict unchanged.
        veto_s = (NG.membership_score(psi_in[0], self.veto.axioms)
                  if self.veto.n_axioms > 0 else None)
        verdict = self.veto(psi_conv, novelty_score=veto_s,
                            novelty_lambda=novelty_lambda)
        return {
            "prompt": prompt,
            "psi_in": psi_in[0],
            "psi_converged": psi_conv[0],
            "token_ids": ids.tolist(),
            "probability": float(probs.max()),
            "decoded": decoded,
            "sagnac": {
                "allow": bool(verdict["allow"][0]),
                "delta": float(verdict["delta"][0]),
                "delta_sagnac": float(verdict["delta_sagnac"][0]),
                "novelty_penalty": float(verdict["novelty_penalty"]),
                "novelty_lambda": float(novelty_lambda),
                "flipped_by_novelty": bool(verdict["flipped_by_novelty"][0]),
                "veto_source": verdict["veto_source"],
                "threshold": float(self.veto.threshold),
                "reason": verdict["reason"],
            },
            "swarm": swarm_diag,
            "memory": {
                "n_axioms": self.veto.n_axioms,
                "offdiag_max": float(mem["offdiag_max"].mean()),
                "gamma": mem["gamma"][0].tolist(),
            },
            # D-Q4: additive, non-breaking. The PRE-projection membership score is
            # reported on the DEFAULT path so the signal is observable without
            # enabling abstention. Absent only when build_axioms has not been run.
            "novelty": ({
                "score": round(float(novelty), 8),
                "bank": "axiom_waves",
                "n": int(nov_bank.shape[0]),
                "abstain_below": abstain_below,
                "in_bank_at_threshold": bool(
                    novelty >= (abstain_below if abstain_below is not None else 0.5)),
            } if novelty is not None else None),
            "models_engaged": [MODEL_1_ID, MODEL_2_ID, MODEL_3_ID],
        }

    # ------------------------------------------------------------- provenance
    def model_registry(self) -> dict:
        """Names, sizes, and engage status for each micro-model."""
        return {
            MODEL_1_ID: {
                "class": type(self.swarm).__name__,
                "params": sum(p.numel() for p in self.swarm.parameters()),
                "training_free": True,
                "workers": self.swarm.B,
            },
            MODEL_2_ID: {
                "class": type(self.memory).__name__,
                "params": sum(p.numel() for p in self.memory.parameters()),
                "training_free": False,
            },
            MODEL_3_ID: {
                "class": type(self.decoder).__name__,
                "params": sum(p.numel() for p in self.decoder.parameters()),
                "training_free": False,
            },
        }


def _small_memory(dim: int) -> HenriMem65M:
    """Shrunken HENRI-Mem-65M for CPU smoke tests. Same architecture, less width."""
    return HenriMem65M(dim=dim, n_features=32, n_layers=1, d_mem=64,
                       n_heads=4, gram_size=16, d_ffn=128)

"""Autonomous Daydream Consolidation Engine.

Authority: the attached daydream document (sha256 cf45f0b1...), sections 4 and 5,
read as SUGGESTION per the operator. Deviations from the document are named
inline and listed at the end of this docstring. Where the document's pseudocode
is wrong, this module implements the corrected form and says so.

THE FOUR STAGES (doc section 4)
    1  fictive trajectory generation   swarm-style Langevin proposals
    2  epiplexity audit                preconditioned gradient alignment r_i
    3  viscoelastic consolidation      decay, epistemic pruning, Stiefel retraction
    4  universal subspace retraction   W <- U_k U_k^T (W - eta grad), Sagnac veto

NAMED DEVIATIONS -- the document is a suggestion, these are decisions
    D1  TimescaleDB/TigerData does not exist on this host. The engram store is
        in-memory with a JSONL receipt. The document's hypertable IS a database
        feature; we do not claim it.
    D2  G-DD7 (peak VRAM <= 2.20 GiB) is NOT TESTABLE on CPU. The GPU budget is
        shut. The gate is reported NOT_APPLICABLE with the CPU RSS labelled,
        never silently passed.
    D3  The document's pseudocode computes r_i from param.grad without calling
        backward() first, and defines delta_theta with the opposite sign to its
        own text. This module runs a real backward pass and uses
        delta_theta = theta_past - theta_now, matching the document's prose.
    D4  The document's U_k is a per-parameter basis. We extract ONE basis over
        the flattened text-head path, which is the trained readout. Scope is
        printed, not implied.
    D5  No GRPO policy gradient. The generator proposes programs by uniformly
        sampling the op grammar and filters them by the alignment reward. That is
        a rejection-sampled curriculum, not reinforcement learning.
    D6  The document's info_gain formula uses beta=26.1 on UNNORMALISED dot
        products. That saturates to one-hot, so every entropy is 0 and delta H is
        0 for any input pair. This module normalises the vectors, adds a small
        uniform floor, and ships a positive control. Without the floor a gate on
        this metric CANNOT FAIL.

GATE DEFECTS FOUND IN THE FIRST RECEIPT (all self-caught, D100-D107)
    D100  G-DD1's negative control was the literal constant 0.0 asserted in code.
          A control that is not measured is not a control.
    D101  G-DD1 let dream proposals reach length 4 while the M4 training set was
          length <= 3, so the dream corpus COVERED the held-out regime. Growth on
          that corpus confounds coverage with composition. A length-2 arm separates.
    D102  G-DD2 reported explained variance 1.0 with T=3 trajectory points and
          k=32. With T-1 < k the top-k directions span the trajectory trivially.
          A rank guard must report NOT_INFORMATIVE, not PASS.
    D103  G-DD3 was BLOCKED because the trajectory held one point per epoch.
          Observation now happens per OPTIMISER STEP.
    D104  G-DD4 called info_gain with a psi shaped [1, M, d] and engrams shaped
          [8, M, d]; the [1,8] logits softmax to 1.0, H=0, so delta H was 0.0 for
          every pair on earth. The gate was dead, not failing.
    D105  stage4_retract was called with a per-parameter loop against a basis of
          the whole concatenated vector. Corrected: project the concatenation.
    D106  retraction_moved was reported with NO post-retractionheld-out CE, so
          "consolidation" and "damage" were indistinguishable.
    D107  engrams_pruned and engrams_axiomatic were both 0 because D104 killed
          the gain, so stage-3 pruning and promotion never executed at all.

HONEST SCOPE ON THE KAUSHIK CITATION
    Kaushik et al. (arXiv:2512.05117, verified) report 1,100+ models. We have
    ONE decoder, and the subspace evidence covers the READOUT HEAD only
    (3 tensors, 45,440 dims), not a 24-layer backbone. We do NOT claim
    universality.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import torch

# D98 (self-caught): daydream.py is run as `python henri_core/daydream.py`, so
# the package root is not on sys.path. Every sibling module that imports
# henri_core absolutely inserts it here; this file omitted the bootstrap.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import (              # noqa: E402
    INPUTS, M4Config, WaveTextGenerator, build_corpus, build_system, run_program,
)
from henri_core.subspace import UniversalSubspace   # noqa: E402


@dataclass
class Engram:
    """One consolidated dream trace. doc section 4.3."""
    spec: str
    trace: str
    weight: float = 1.0
    info_gain: float = 0.0
    alignment: float = 0.0
    axiomatic: bool = False


@dataclass
class DaydreamConfig:
    epochs: int = 3
    proposals: int = 24
    steps_per_epoch: int = 24
    lr: float = 3e-3
    seed: int = 0
    k_sub: int = 8                    # D119: must satisfy rank(Dc) > k or the
                                      # explained-variance gate is vacuous (D102).
                                      # The doc allows r_hat <= 32; 8 is within it
                                      # and is honest for a short trajectory, and
                                      # the rank guard still fires if T is small.
    crosstalk_limit: float = 0.12
    sagnac_threshold: float = 0.35
    beta: float = 26.10
    tau: tuple = (1e-3, 1e-1, 1e1, 1e6)      # doc section 4.3 time constants
    gamma: tuple = (1.0, 1.0, 1.0, 1.0)
    prune_weight: float = 1e-4
    prune_info: float = 0.01
    promote_info: float = 0.15
    gain_floor: float = 0.05                  # legacy alias, see beta_entropy
    beta_entropy: float = 1.0                 # D110: entropy temperature
    entropy_floor: float = 0.05               # D110: uniform mixture floor


@dataclass
class DaydreamReport:
    epoch_ce: list = field(default_factory=list)
    epoch_heldout_ce: list = field(default_factory=list)
    engrams_created: int = 0
    engrams_pruned: int = 0
    engrams_axiomatic: int = 0
    crosstalk_trace: list = field(default_factory=list)
    alignment_trace: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    retraction_moved: float = 0.0
    retraction_shapes: list = field(default_factory=list)
    retraction_dim_now: int = 0
    retraction_dim_fit: int = 0
    retraction_ce_before: float = 0.0
    retraction_ce_after: float = 0.0
    subspace_explained: float = 0.0
    subspace: object = None
    model: object = None


class EngramStore:
    """In-memory engram store with viscoelastic weights. DEVIATION D1."""

    def __init__(self, cfg: DaydreamConfig):
        self.cfg = cfg
        self.items: list[Engram] = []

    def add(self, spec: str, trace: str, alignment: float,
            info_gain: float) -> Engram:
        e = Engram(spec=spec, trace=trace, alignment=alignment,
                   info_gain=info_gain)
        self.items.append(e)
        return e

    def age(self, t: float) -> None:
        """Fractional viscoelastic decay. doc section 4.3 weight w_i(t)."""
        tau, gam = self.cfg.tau, self.cfg.gamma
        for e in self.items:
            if e.axiomatic:
                continue
            w = sum(g / T * math.exp(-t / T) for g, T in zip(gam, tau))
            e.weight = float(w)

    def prune(self) -> int:
        """Epistemic garbage collection. doc section 4.3."""
        before = len(self.items)
        self.items = [e for e in self.items
                      if e.axiomatic
                      or e.weight >= self.cfg.prune_weight
                      or e.info_gain >= self.cfg.prune_info]
        return before - len(self.items)

    def promote(self) -> int:
        """Axiomatic baseplate consolidation. doc section 4.3."""
        n = 0
        for e in self.items:
            if not e.axiomatic and e.info_gain > self.cfg.promote_info:
                e.axiomatic = True
                n += 1
        return n


class DaydreamEngine:
    """The four-stage consolidation loop. doc section 4."""

    def __init__(self, system, tok, cfg: DaydreamConfig):
        self.system = system
        self.tok = tok
        self.cfg = cfg
        self.store = EngramStore(cfg)
        self.rep = DaydreamReport()

    # ------------------------------------------------------- STAGE 1 generate
    def stage1_fictive(self, n: int, gen: torch.Generator,
                       max_len: int = 4) -> list[tuple[str, str]]:
        """Propose computable programs and their execution traces.

        DEVIATION D5: rejection-sampled from the op grammar, not a GRPO policy.
        max_len=2 gives the D101 coverage-controlled arm; max_len=4 probes one
        step beyond the M4 training distribution.
        """
        ops = ["I", "R", "C"]
        out = []
        for _ in range(n):
            L = int(torch.randint(1, max_len + 1, (1,), generator=gen).item())
            prog = "".join(ops[int(torch.randint(0, 3, (1,),
                                                    generator=gen).item())]
                           for _ in range(L))
            inp = INPUTS[int(torch.randint(0, len(INPUTS), (1,),
                                           generator=gen).item())]
            out.append((f"apply {prog} to {inp}", run_program(prog, inp)))
        return out

    # --------------------------------------------------------- STAGE 2 score
    def stage2_alignment(self, model: WaveTextGenerator,
                         batch: list[tuple[str, str]],
                         prev_flat: torch.Tensor | None, cfg: M4Config,
                         preconditioner: torch.Tensor | None
                         ) -> tuple[float, torch.Tensor]:
        """Preconditioned gradient-alignment reward. doc section 4.2.

        r_i = |<grad_theta L(y_i; theta), P_e delta_theta>| with
        delta_theta = theta_past - theta_now  (doc prose; DEVIATION D3 fixes the
        document pseudocode, which omits the backward pass and flips the sign).
        """
        specs = [b[0] for b in batch]
        targets = [self.tok.encode(b[1]) for b in batch]
        model.zero_grad(set_to_none=True)
        psi = model.wave(specs)
        logits = model.logits_from_wave(psi)
        m = min(cfg.max_trace, logits.shape[1])
        tgt = torch.full((len(targets), m), cfg.pad_id, dtype=torch.long)
        for i, ids in enumerate(targets):
            ids = ids[:m]
            tgt[i, :len(ids)] = torch.tensor(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :m, :].reshape(-1, logits.shape[-1]),
            tgt.reshape(-1), ignore_index=cfg.pad_id)
        loss.backward()                                    # D3: real backward
        flat = torch.cat([p.detach().reshape(-1).float()
                          for p in self._readout_params(model)])
        if prev_flat is None:
            return 0.0, flat
        delta = prev_flat - flat                           # theta_past - theta_now
        reward = 0.0
        off = 0
        for p in self._readout_params(model):
            n = p.numel()
            if p.grad is None:
                off += n
                continue
            g = p.grad.detach().reshape(-1).float()
            d = delta[off:off + n]
            pe = (preconditioner[off:off + n] if preconditioner is not None
                  else torch.ones_like(d))
            reward += float((g * pe * d).abs().sum())
            off += n
        return reward, flat

    @staticmethod
    def _readout_params(model: WaveTextGenerator):
        """The trained readout path. DEVIATION D4: one basis over this path."""
        return [model.dec.head_text.weight, model.dec.norm.weight,
                model.dec.norm.bias]

    def _flat_readout(self, model) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1).float()
                          for p in self._readout_params(model)])

    # ------------------------------------------------------ STAGE 3 consolidate
    def stage3_consolidate(self, waves: torch.Tensor) -> dict:
        """Stiefel crosstalk check and retraction. doc section 4.3 step 4.

        D105: report both the PRE and the POST crosstalk, so a retraction that
        fired is visible as a change rather than implied.
        """
        # D113 (self-caught): the waves are complex64 and the first draft called
        # .float(), which DISCARDS the imaginary part, so crosstalk was measured
        # on half the signal. Keep the complex dtype and use the Hermitian form.
        x = waves
        if x.shape[0] < 2:
            return {"crosstalk": 0.0, "pre": 0.0, "retracted": False,
                    "n": int(x.shape[0]), "waves": x}

        def xtalk(t: torch.Tensor) -> float:
            tn = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            G = tn @ tn.conj().transpose(0, 1)
            n = G.shape[0]
            off = G - torch.eye(n, dtype=G.dtype)
            return float((off.abs() ** 2).sum() / max(n * (n - 1), 1))

        pre = xtalk(x)
        retracted = pre > self.cfg.crosstalk_limit
        if retracted:
            q, _ = torch.linalg.qr(x.transpose(0, 1))
            x = q.transpose(0, 1)
        post = xtalk(x)
        return {"crosstalk": post, "pre": pre, "retracted": retracted,
                "n": int(x.shape[0]), "waves": x}

    # --------------------------------------------------------- STAGE 4 retract
    def stage4_retract(self, model, sub_space: UniversalSubspace, eta: float,
                       ho_spec, ho_tgt, cfg: M4Config) -> tuple[float, float, float]:
        """Project the readout parameters onto the universal subspace.

        doc section 4.4: W <- U_k U_k^T (W - eta grad).

        D99: U is a basis of the WHOLE concatenated readout trajectory
        (shape [45440, k]). The projection is a coordinate transform on the
        CONCATENATED vector: concatenate, project once, then split back.
        D106: measure held-out CE before and after, so damage is visible.
        Returns (moved, ce_before, ce_after).
        """
        shapes = [tuple(p.shape) for p in self._readout_params(model)]
        n_now = sum(p.numel() for p in self._readout_params(model))
        before = float(full_ce(model, self.tok, ho_spec, ho_tgt, cfg).detach())
        self.rep.retraction_shapes = [list(s) for s in shapes]
        self.rep.retraction_dim_now = int(n_now)
        self.rep.retraction_ce_before = before
        if sub_space.U is None:
            self.rep.notes.append("retraction SKIPPED: no basis fitted")
            self.rep.retraction_ce_after = before
            return 0.0, before, before
        n_fit = int(sub_space.U.shape[0])
        self.rep.retraction_dim_fit = n_fit
        if n_now != n_fit:
            self.rep.notes.append(
                f"retraction SKIPPED: basis fitted at {n_fit} dims, readout is "
                f"{n_now} dims.")
            self.rep.retraction_ce_after = before
            return 0.0, before, before
        flat = self._flat_readout(model)
        m = sub_space.mean.squeeze(0)
        proj = sub_space.U @ (sub_space.U.T @ (flat - m)) + m
        moved = 0.0
        with torch.no_grad():
            off = 0
            for p, shp in zip(self._readout_params(model), shapes):
                n = p.numel()
                new = proj[off:off + n]
                moved += float((new - flat[off:off + n]).abs().sum())
                p.copy_(new.reshape(shp).to(p.dtype))
                off += n
        after = float(full_ce(model, self.tok, ho_spec, ho_tgt, cfg).detach())
        self.rep.retraction_ce_after = after
        return moved, before, after

    # ----------------------------------------------------------- entropy gain
    @torch.no_grad()
    def _engram_dist(self, psi: torch.Tensor, engrams: torch.Tensor,
                     beta: float | None = None,
                     floor: float | None = None) -> torch.Tensor:
        """Softmax over the engram bank at the ENTROPY temperature.

        DEVIATION D6 / defects D104 and D110. The document multiplies beta=26.1
        into unnormalised dot products. Over an 8-key bank that saturates to
        one-hot, so every entropy is ~0 and delta H was EXACTLY 0.0 for all
        input pairs. The gate could not fail. This version:
          1. normalises both sides, so the temperature acts on cosines in [-1,1];
          2. uses an explicit ENTROPY temperature (default 1.0) reported in the
             gate, kept separate from the retrieval beta = 26.10;
          3. mixes an explicit uniform floor, so H is bounded away from 0 and
             has a measurable dynamic range up to log(n_engrams).
        D110 (self-caught): the previous revision only did (1). On 8 keys,
        beta=26.1 makes the softmax one-hot whatever the normalisation, so the
        metric would still have read 0.0. Normalising alone was cosmetic.
        D111 (self-caught): this helper then referenced `engrams` without taking
        it as a parameter, so section C raised NameError.
        """
        def flat(t: torch.Tensor) -> torch.Tensor:
            return t.mean(dim=1) if t.dim() == 3 else t

        def norm(t: torch.Tensor) -> torch.Tensor:
            return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        b = self.cfg.beta_entropy if beta is None else float(beta)
        fl = self.cfg.entropy_floor if floor is None else float(floor)
        q = flat(psi)
        k = flat(engrams)
        logits = b * (norm(q) @ norm(k).conj().T).real
        p = torch.softmax(logits, dim=-1)
        n = p.shape[-1]
        if fl <= 0.0:
            return p
        return (1.0 - fl) * p + fl / n

    @torch.no_grad()
    def _entropy(self, psi: torch.Tensor, engrams: torch.Tensor,
                 beta: float | None = None,
                 floor: float | None = None) -> float:
        p = self._engram_dist(psi, engrams, beta=beta, floor=floor)
        return float((-(p * (p + 1e-12).log()).sum(dim=-1)).mean())

    @torch.no_grad()
    def information_gain(self, psi: torch.Tensor, engrams: torch.Tensor,
                         beta: float | None = None) -> float:
        """H(prior) - H(posterior). DEVIATION D6, defect D115.

        The document defines dH(pi) = H(P_t) - E[H(P_{t+1} | o, pi)]: a DIFFUSE
        prior and a CONCENTRATED posterior. The first implementation passed two
        arbitrary specs, and two concentrated posteriors differ by zero, so the
        gate could not distinguish anything. Here the prior is the uniform
        distribution over the engram bank, H = log(n), and the posterior is the
        Hopfield softmax over that bank at the RETRIEVAL temperature.

        This is the document's own quantity and it is falsifiable in both
        directions: a query that matches a stored engram concentrates the
        posterior and drives entropy DOWN; an orthogonal query (cosine 0 to every
        engram) leaves H at log(n) and gives exactly 0.
        """
        n = int(engrams.shape[0])
        b = self.cfg.beta if beta is None else float(beta)
        return math.log(n) - self._entropy(psi, engrams, beta=b, floor=0.0)

    @torch.no_grad()
    def info_gain(self, psi_t: torch.Tensor, psi_next: torch.Tensor,
                  engrams: torch.Tensor) -> float:
        """delta H = H(P_t) - H(P_next). Identical states return exactly 0."""
        return (self._entropy(psi_t, engrams)
                - self._entropy(psi_next, engrams))

    @torch.no_grad()
    def info_gain_uniform(self, psi: torch.Tensor,
                          engrams: torch.Tensor) -> float:
        """H(P) for one state, used for the flat-vs-peaked control."""
        return self._entropy(psi, engrams)

    @torch.no_grad()
    def hopfield_step(self, psi: torch.Tensor,
                      engrams: torch.Tensor) -> torch.Tensor:
        """One modern-Hopfield update at the RETRIEVAL beta (Ramsauer et al.).

        psi_next = softmax(beta * psi engrams^T) engrams. This is the document's
        P_{t+1}, so G-DD4 measures entropy reduction on CONVERGENCE instead of
        between two arbitrary specs.
        """
        def flat(t: torch.Tensor) -> torch.Tensor:
            return t.mean(dim=1) if t.dim() == 3 else t

        def norm(t: torch.Tensor) -> torch.Tensor:
            return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        q, k = flat(psi), flat(engrams)
        logits = self.cfg.beta * (norm(q) @ norm(k).conj().T).real
        w = torch.softmax(logits, dim=-1)
        return w @ engrams

    # -------------------------------------------------------------- the loop
    def run(self, corpus, cfg: M4Config,
            dream_max_len: int = 4) -> DaydreamReport:
        torch.manual_seed(self.cfg.seed)
        gen = torch.Generator().manual_seed(self.cfg.seed)
        model = WaveTextGenerator(self.system, self.tok, train_body=True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.cfg.lr)
        readout = self._readout_params(model)
        D = sum(p.numel() for p in readout)
        prev_flat = self._flat_readout(model)
        sub_space = UniversalSubspace(k=self.cfg.k_sub)
        precond = torch.ones(D)

        ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
        ho_tgt = [self.tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]

        for epoch in range(self.cfg.epochs):
            batch = self.stage1_fictive(self.cfg.proposals, gen,
                                        max_len=dream_max_len)
            # D103: observe the trajectory at EVERY optimiser step, so the
            # Davis-Kahan gate has more than one point per epoch.
            # D108 (self-caught): max(1, steps) gave a zero-step "frozen" arm one
            # real optimiser step, so the control could not be frozen. Use max(0).
            reward = 0.0
            for _step in range(max(0, self.cfg.steps_per_epoch)):
                reward, _flat_before = self.stage2_alignment(
                    model, batch, prev_flat, cfg, precond)
                opt.step()
                opt.zero_grad(set_to_none=True)
                now = self._flat_readout(model)
                sub_space.observe(now - prev_flat)
                prev_flat = now
                self.rep.alignment_trace.append(reward)
                if len(sub_space._rows) >= 2:
                    sub_space.fit()

            # STAGE 3: consolidate the dream traces with a LIVE information gain
            psi_new = model.wave([b[0] for b in batch])
            # D118 (self-caught): the engram's information gain must be the
            # document's dH(pi) -- the entropy reduction of this trace's wave
            # against the bank of traces, H(uniform) - H(Hopfield posterior).
            # The previous line passed psi_new against a ROLL of itself, which is
            # two arbitrary traces again: the D115 defect, re-appearing inside the
            # loop. With D104 it pinned every gain at 0, so stage-3 pruning and
            # promotion never executed (D107).
            gain = self.information_gain(psi_new, psi_new)
            cons = self.stage3_consolidate(psi_new)
            self.rep.crosstalk_trace.append(cons["crosstalk"])
            for spec, trace in batch:
                self.store.add(spec, trace, alignment=reward, info_gain=gain)
            self.rep.engrams_created = len(self.store.items)
            self.store.age(t=float(epoch))
            self.rep.engrams_pruned += self.store.prune()
            self.rep.engrams_axiomatic += self.store.promote()

            self.rep.epoch_ce.append(
                float(model_loss_on(model, self.tok,
                                    [b[0] for b in batch],
                                    [self.tok.encode(b[1]) for b in batch],
                                    cfg).detach()))
            self.rep.epoch_heldout_ce.append(
                float(full_ce(model, self.tok, ho_spec, ho_tgt, cfg).detach()))

        # STAGE 4: retraction, then the Sagnac veto
        self.rep.subspace_explained = (sub_space.explained_variance()
                                       if sub_space.U is not None else 0.0)
        self.rep.subspace = sub_space
        self.rep.model = model
        self.rep.retraction_moved, _b, _a = self.stage4_retract(
            model, sub_space, eta=self.cfg.lr, ho_spec=ho_spec, ho_tgt=ho_tgt,
            cfg=cfg)
        return self.rep


def model_loss_on(model, tok, specs, targets, cfg: M4Config) -> torch.Tensor:
    """CE over the first min(max_trace, M) positions, padded positions ignored."""
    psi = model.wave(specs)
    logits = model.logits_from_wave(psi)
    m = min(cfg.max_trace, logits.shape[1])
    tgt = torch.full((len(targets), m), cfg.pad_id, dtype=torch.long)
    for i, ids in enumerate(targets):
        ids = ids[:m]
        tgt[i, :len(ids)] = torch.tensor(ids)
    return torch.nn.functional.cross_entropy(
        logits[:, :m, :].reshape(-1, logits.shape[-1]),
        tgt.reshape(-1), ignore_index=cfg.pad_id)


def full_ce(model, tok, specs, targets, cfg: M4Config) -> torch.Tensor:
    """CE over ALL M positions. Pad positions ignored. Used for the D106 audit."""
    psi = model.wave(specs)
    logits = model.logits_from_wave(psi)
    M = logits.shape[1]
    tgt = torch.full((len(targets), M), cfg.pad_id, dtype=torch.long)
    for i, ids in enumerate(targets):
        ids = ids[:M]
        tgt[i, :len(ids)] = torch.tensor(ids)
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
        ignore_index=cfg.pad_id)


# ------------------------------------------------------------------- gates
def _verdict(ok: bool, ctl_ok: bool, why: str) -> tuple[str, str]:
    if not ctl_ok:
        return "VACUOUS", f"control did not behave ({why})"
    return ("PASS", "") if ok else ("FAIL", "")


def _gate(gid, metric, value, op, bound, status, **extra) -> dict:
    d = {"id": gid, "metric": metric, "value": value, "op": op,
         "bound": bound, "status": status}
    d.update(extra)
    return d


def section_a(cfg: DaydreamConfig) -> tuple[list, dict]:
    """A: the daydream loop, its gates, and its raw report."""
    corpus = build_corpus()
    system, tok = build_system(corpus)
    m4cfg = M4Config(steps=cfg.steps_per_epoch)
    eng = DaydreamEngine(system, tok, cfg)
    rep = eng.run(corpus, m4cfg, dream_max_len=4)
    gates = []

    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]

    # ---- G-DD1 epiplexity growth, with a MEASURED frozen control (D100) and a
    #      length-controlled arm that cannot cover held-out length 3 (D101)
    ce = rep.epoch_heldout_ce
    growth = (ce[0] - ce[-1]) if len(ce) >= 2 else 0.0
    frozen_eng = DaydreamEngine(system, tok, DaydreamConfig(
        epochs=cfg.epochs, proposals=cfg.proposals, steps_per_epoch=0,
        lr=0.0, seed=cfg.seed))
    frozen = frozen_eng.run(corpus, m4cfg, dream_max_len=4)
    frozen_growth = (frozen.epoch_heldout_ce[0] - frozen.epoch_heldout_ce[-1]
                     if len(frozen.epoch_heldout_ce) >= 2 else 0.0)
    cov_eng = DaydreamEngine(system, tok, DaydreamConfig(
        epochs=cfg.epochs, proposals=cfg.proposals,
        steps_per_epoch=cfg.steps_per_epoch, lr=cfg.lr, seed=cfg.seed))
    cov = cov_eng.run(corpus, m4cfg, dream_max_len=2)
    cov_growth = (cov.epoch_heldout_ce[0] - cov.epoch_heldout_ce[-1]
                  if len(cov.epoch_heldout_ce) >= 2 else 0.0)
    ok = growth > 0.0
    ctl_ok = abs(frozen_growth) < 1e-9
    st, why = _verdict(ok, ctl_ok, "the zero-step learner moved")
    gates.append(_gate(
        "G-DD1", "epiplexity_growth_heldout_ce", growth, ">", 0.0, st,
        control_value=frozen_growth, why=why, heldout_ce_trace=ce,
        frozen_ce_trace=frozen.epoch_heldout_ce,
        coverage_arm_growth=cov_growth,
        coverage_arm_note=("dreams of length <= 2 cannot cover held-out length 3; "
                           "a similar growth there is composition, not coverage"),
        control_kind="MEASURED zero-step learner (D100)",))

    # ---- G-DD2 explained variance, with the rank guard (D102)
    ev = rep.subspace_explained
    U = rep.subspace.U
    D_rows = torch.stack(rep.subspace._rows)
    Dc = D_rows - D_rows.mean(0, keepdim=True)
    rank_dc = int(torch.linalg.matrix_rank(Dc))
    k = int(U.shape[1])
    g = torch.Generator().manual_seed(11)
    rand_basis = torch.linalg.qr(torch.randn(U.shape[0], k, generator=g))[0]
    ev_rand = float(((Dc @ rand_basis) ** 2).sum()
                    / (Dc ** 2).sum().clamp_min(1e-30))
    if rank_dc <= k:
        gates.append(_gate(
            "G-DD2", "subspace_explained_variance_topk", ev, ">=", 0.90,
            "NOT_INFORMATIVE", control_value=ev_rand, why=(
                f"rank(Dc)={rank_dc} <= k={k}: the top-{k} directions span the "
                "trajectory trivially. Raise steps_per_epoch (D102)."),
            k=k, rank_Dc=rank_dc, n_points=len(rep.subspace._rows),
            scope="one decoder readout head, not the paper's 1100+ models"))
    else:
        ok = ev >= 0.90
        ctl_ok = ev_rand < ev
        st, why = _verdict(ok, ctl_ok, "random basis matched the subspace")
        gates.append(_gate(
            "G-DD2", "subspace_explained_variance_topk", ev, ">=", 0.90, st,
            control_value=ev_rand, why=why, k=k, rank_Dc=rank_dc,
            n_points=len(rep.subspace._rows),
            scope="one decoder readout head, not the paper's 1100+ models"))

    # ---- G-DD3 Davis-Kahan drift, now unblocked by per-step observation (D103)
    if len(rep.subspace._rows) >= 4:
        dk = rep.subspace.half_split_drift()
        value, bound = dk["op_drift"], dk["bound"]
        U1 = UniversalSubspace._basis(D_rows[:len(D_rows) // 2], k)
        rand_arm = UniversalSubspace._basis(
            torch.randn_like(D_rows[len(D_rows) // 2:]), k)
        ctl = UniversalSubspace.projector_op_dist(U1, rand_arm)
        # D126 (self-caught): op_drift is sin(theta_max), so it lives in [0, 1].
        # The document's bound 0.05/gamma_k evaluated to 207.3 here, because
        # gamma_k = 2.41e-4. No measurement can exceed 1.0 and the unrelated
        # control also sits at 1.0, so the gate was structurally incapable of
        # failing -- the "gate that cannot fail" class. Report NOT_INFORMATIVE
        # when the bound reaches the metric ceiling. The MEASUREMENT still stands:
        # drift 0.997 means the two half-trajectory subspaces are nearly
        # orthogonal, i.e. the subspace is NOT stable across the trajectory.
        ok = value <= bound
        ctl_ok = ctl > bound
        if bound >= 1.0:
            st, why = "NOT_INFORMATIVE", (
                f"bound {bound:.6g} >= the metric ceiling 1.0 (op_drift is "
                f"sin(theta_max) in [0,1]); gamma_k={dk['gamma_k']:.3g} makes the "
                "document's 0.05/gamma_k bound unfailable. The measurement still "
                f"stands: drift {value:.6g} means the two half-trajectory "
                "subspaces are nearly orthogonal.")
        else:
            st, why = _verdict(ok, ctl_ok, "unrelated basis did not drift more")
        gates.append(_gate(
            "G-DD3", "davis_kahan_op_drift", value, "<=", bound, st,
            control_value=ctl, why=why, gamma_k=dk["gamma_k"],
            n_points=dk["n_points"],
            bound_formula="2/gamma_k * (2*B*eta_bar + eta_bar^2)"))
    else:
        gates.append(_gate("G-DD3", "davis_kahan_op_drift", None, "<=", None,
                           "BLOCKED", why="fewer than 4 trajectory points"))

    # ---- G-DD4 information gain, with a real positive control (D104)
    # D124 (self-caught): the first arm built the bank from "axiom i" waves and
    # the query from an unrelated spec family. Slot-sparse waves with disjoint
    # support give cosine EXACTLY 0 to every bank vector, so the posterior is
    # uniform for ANY query and the dynamic range collapses to 0 -- the D120
    # defect, re-appearing in section A. Draw the bank from the SAME corpus so
    # the query can concentrate, and measure the flat-vs-peaked DYNAMIC RANGE
    # (the corrected C-G1 form), not the difference between two arbitrary specs.
    engrams = rep.model.wave([corpus.specs[i] for i in corpus.train_idx[:8]])
    peak_q = rep.model.wave([corpus.specs[corpus.train_idx[0]]])
    flat_q = torch.zeros_like(peak_q)
    h_flat = eng.info_gain_uniform(flat_q, engrams)
    h_peak = eng.info_gain_uniform(peak_q, engrams)
    spread = h_flat - h_peak
    gain_ctl = eng.info_gain(peak_q, peak_q.clone(), engrams)  # must be exact 0
    ok = spread > 0.15
    # D121/D115: the control shows the metric is ALIVE (identical states give
    # exactly 0 and the dynamic range is nonzero). The bound decides the verdict.
    ctl_ok = (abs(gain_ctl) <= 1e-9) and (abs(spread) > 1e-6)
    st, why = _verdict(ok, ctl_ok, (
        f"identical states gave {gain_ctl} (must be 0) and the flat-vs-peak "
        f"entropy range was {spread} (must be nonzero for the metric to live)"))
    gates.append(_gate(
        "G-DD4", "info_gain_dynamic_range_nats", spread, ">", 0.15, st,
        control_value=gain_ctl, why=why, flat_state_H=h_flat, peak_state_H=h_peak,
        entropy_range_control=spread, beta=cfg.beta, bank="corpus train specs",
        note="bound 0.15 nats is the document's; not moved",
        deviation="D6: normalised cosines; the doc formula saturates to one-hot"))

    # ---- G-DD5 crosstalk, with the measured pre/post pair (D105)
    waves = rep.model.wave([e.spec for e in eng.store.items[:32]])
    cons = eng.stage3_consolidate(waves)
    value, pre = cons["crosstalk"], cons["pre"]
    dup = waves[:1].repeat(8, 1)
    ctl = eng.stage3_consolidate(dup)["pre"]
    ok = value <= cfg.crosstalk_limit
    ctl_ok = ctl > cfg.crosstalk_limit
    st, why = _verdict(ok, ctl_ok, "the correlated control did not exceed the limit")
    gates.append(_gate(
        "G-DD5", "stiefel_crosstalk", value, "<=", cfg.crosstalk_limit, st,
        control_value=ctl, why=why, retracted=cons["retracted"],
        pre_retraction=pre, post_retraction=value))

    # ---- G-DD6 Sagnac dark-port integrity under phase noise
    n_ax = 8
    axioms = rep.model.wave([f"dream axiom {i}" for i in range(n_ax)])
    system.veto.load_axioms(axioms)
    clean = system.veto.axioms.clone()
    corrupt = -clean                                  # delta_phi = pi
    res = system.veto.veto_accuracy(clean, corrupt)
    value = res["accuracy"]
    ctl_ok = res["clean_pass_rate"] >= 0.999
    ok = value >= 0.999
    st, why = _verdict(ok, ctl_ok, "clean states were rejected")
    gates.append(_gate(
        "G-DD6", "sagnac_veto_accuracy_phase_noise", value, ">=", 0.999, st,
        control_value=res["clean_pass_rate"], why=why,
        corrupt_reject_rate=res["corrupt_reject_rate"]))

    # ---- G-DD7 VRAM ceiling: NOT_TESTABLE here, declared not faked
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:                                  # noqa: BLE001
        rss = float("nan")
    gates.append(_gate(
        "G-DD7", "peak_vram_gib", None, "<=", 2.20, "NOT_APPLICABLE",
        control_value=None, why=("CPU-only host; the GPU budget stays shut. "
                                 f"CPU peak RSS {rss:.3f} MiB is NOT a VRAM pass. "
                                 "Deferred to a GPU run under separate approval."),
        cpu_peak_rss_mib=(rss if rss == rss else None)))

    report = {
        "epochs": cfg.epochs, "proposals": cfg.proposals,
        "steps_per_epoch": cfg.steps_per_epoch,
        "engrams_created": rep.engrams_created,
        "engrams_pruned": rep.engrams_pruned,
        "engrams_axiomatic": rep.engrams_axiomatic,
        "epoch_ce": rep.epoch_ce,
        "epoch_heldout_ce": rep.epoch_heldout_ce,
        "alignment_trace_tail": rep.alignment_trace[-5:],
        "alignment_trace_len": len(rep.alignment_trace),
        "crosstalk_trace": rep.crosstalk_trace,
        "subspace_explained": rep.subspace_explained,
        "subspace_rank_Dc": rank_dc,
        "subspace_k": k,
        "subspace_points": len(rep.subspace._rows),
        "retraction_moved": rep.retraction_moved,
        "retraction_dim_now": rep.retraction_dim_now,
        "retraction_dim_fit": rep.retraction_dim_fit,
        "retraction_shapes": rep.retraction_shapes,
        "retraction_ce_before": rep.retraction_ce_before,
        "retraction_ce_after": rep.retraction_ce_after,
        "retraction_delta_ce": rep.retraction_ce_after - rep.retraction_ce_before,
        "notes": rep.notes,
    }
    return gates, report


def section_b(cfg: DaydreamConfig) -> tuple[list, dict]:
    """B: the universal subspace, tested against the document's exact claim.

    The document (section 2.3) claims hats{r}_n <= 32, and section 3.2 cites
    Kaushik Theorem 2.5 on finite effective rank. This tests OUR decoder's
    declared parameter formula -- it does not test the paper's 1,100-model result.
    D109 (self-caught): the first draft read corpus.cfg, which does not exist, and
    returned an undefined `report`. The formula is checked against a real small
    instantiation, and the full-size formula is checked against the doc envelope
    WITHOUT paying for a 440M instantiation on CPU.
    """
    from henri_core import substrate as sub
    from henri_core.model3_decoder import (
        DecoderConfig, HenriDec450M, decoder_param_formula, resolve_n_mem,
    )

    gates = []
    small = DecoderConfig(dim=4096, d_model=128, n_layers=2, n_heads=4,
                          n_kv_heads=1, d_ffn=256, n_macro=16, n_invariants=32,
                          vocab=512)
    # D116 (self-caught): decoder_param_formula returns a DICT
    # {"total", "parts", "doc_target"}, not an int. The first draft subtracted an
    # int from the dict and raised TypeError. Read "total".
    f_small = int(decoder_param_formula(small)["total"])
    i_small = sum(p.numel() for p in HenriDec450M(small).parameters())
    d_small = abs(f_small - i_small)
    gates.append(_gate(
        "B-G1", "decoder_param_formula_error_small", float(d_small), "<=", 0.0,
        "PASS" if d_small == 0 else "FAIL",
        why="the declared formula must equal the instantiation exactly",
        control_value=float(d_small) if d_small else 0.0,
        formula=f_small, instantiated=i_small))

    full = DecoderConfig(dim=sub.DEFAULT_DIM)
    f_full = int(decoder_param_formula(full)["total"])
    in_env = 400_000_000 <= f_full <= 500_000_000
    gates.append(_gate(
        "B-G2", "decoder_param_full_envelope", float(f_full), "in",
        [400_000_000, 500_000_000], "PASS" if in_env else "FAIL",
        control_value=None,
        why="doc p21 documents the 448.6M envelope",
        n_mem=resolve_n_mem(full.d_model, full.n_mem),
        note="formula only; no 440M instantiation on CPU"))
    return gates, {"full_formula": f_full, "small_formula": f_small,
                   "small_instantiated": i_small,
                   "n_mem_full": resolve_n_mem(full.d_model, full.n_mem)}


def section_c(cfg: DaydreamConfig) -> tuple[list, dict]:
    """C: stage-3 visibility. Pruning and promotion must actually EXECUTE.

    D107: in the first receipt both counters were 0 because D104 killed the
    information gain. These gates inject engrams with known properties and assert
    the store reacts. A stage that never fires is untested, not working.
    """
    corpus = build_corpus()
    system, tok = build_system(corpus)
    eng = DaydreamEngine(system, tok, cfg)
    model = WaveTextGenerator(system, tok, train_body=True)
    gates = []

    # a bank of DISTINCT reference engrams.
    # D109: model.wave() is the entry point (eng.model_wave does not exist).
    # D120 (self-caught): the first bank was 8 scalar multiples of ONE vector,
    # base * (1 + 0.05*randn). Parallel bank vectors give cosine +1 to every
    # query, so the softmax is uniform for ANY input and the entropy spread is
    # zero BY CONSTRUCTION -- the D85 degeneracy, re-created inside my own
    # control. Use eight DISTINCT spec waves, whose support overlap is partial.
    bank_specs = [corpus.specs[i] for i in corpus.train_idx[:8]]
    bank = torch.stack([model.wave([s])[0] for s in bank_specs])
    flat_q = torch.zeros_like(bank[0]).unsqueeze(0)
    peak_q = model.wave([corpus.specs[corpus.heldout_idx[0]]])
    h_flat = eng.info_gain_uniform(flat_q, bank)
    h_peak = eng.info_gain_uniform(peak_q, bank)
    spread = h_flat - h_peak
    gain_matched = eng.info_gain(flat_q, peak_q, bank)
    gain_identical = eng.info_gain(peak_q, peak_q.clone(), bank)

    # D121 (self-caught): the control conflated "the metric is ALIVE" with "the
    # metric CLEARS THE BOUND". A control must show the measurement CAN move; the
    # bound decides pass/fail. Requiring spread > 0.15 in ctl_ok made a moving
    # metric report VACUOUS, which hides the real verdict. Control = (identical
    # states give 0) AND (the metric moves off its dead value at all).
    ok = spread > 0.15
    ctl_ok = (abs(gain_identical) <= 1e-9) and (abs(spread) > 1e-6)
    st, why = _verdict(ok, ctl_ok, (
        f"identical-state gain {gain_identical} must be 0 and the entropy spread "
        f"{spread} must be nonzero for the metric to be alive at all"))
    gates.append(_gate(
        "C-G1", "info_gain_dynamic_range_nats", spread, ">", 0.15, st,
        control_value=gain_identical, why=why, uniform_H=h_flat, peaked_H=h_peak,
        entropy_spread=spread, metric_value=spread,
        note="spread = H(flat query) - H(peaked query); the metric MOVES but does "
             "not clear the document's 0.15 nats bound. Bound not moved."))

    # promotion must fire on an engram whose gain clears promote_info
    st_store = EngramStore(cfg)
    st_store.add("promote-me", "1234", alignment=0.0, info_gain=0.50)
    st_store.add("prune-me", "5678", alignment=0.0, info_gain=0.0)
    st_store.add("keep-me", "9012", alignment=0.0, info_gain=0.05)
    st_store.items[1].weight = 1e-9              # below prune_weight
    promoted = st_store.promote()
    pruned = st_store.prune()
    surviving = [e.spec for e in st_store.items]
    ok_p = promoted == 1 and "promote-me" in surviving
    ok_q = pruned == 1 and "prune-me" not in surviving and "keep-me" in surviving
    gates.append(_gate(
        "C-G2", "stage3_promotion_fires", float(promoted), "==", 1.0,
        "PASS" if ok_p else "FAIL", control_value=None,
        why="an engram with gain 0.50 > promote_info 0.15 must become axiomatic",
        survive=surviving))
    gates.append(_gate(
        "C-G3", "stage3_pruning_fires", float(pruned), "==", 1.0,
        "PASS" if ok_q else "FAIL", control_value=1.0,
        why=("a light engram with zero gain must be dropped while the heavier "
             "one is kept"),
        survive=surviving))
    # D117 (self-caught): section_c returned a bare list, but run_daydream_gates
    # unpacked two values (g, _ = section_c(cfg)), so the caller raised
    # "too many values to unpack (expected 2)". Return the same (gates, report)
    # pair the other two sections return.
    return gates, {"pruned": pruned, "promoted": promoted,
                   "surviving": surviving,
                   "info_gain_matched": gain_matched,
                   "info_gain_identical": gain_identical,
                   "uniform_H": h_flat, "peaked_H": h_peak,
                   "entropy_spread": spread}


def run_daydream_gates(cfg: DaydreamConfig | None = None,
                       section: str = "all") -> dict:
    cfg = cfg or DaydreamConfig()
    gates, report = [], {}
    if section in ("all", "a"):
        g, r = section_a(cfg)
        gates += g
        report = r
    if section in ("all", "b"):
        g, _ = section_b(cfg)
        gates += g
    if section in ("all", "c"):
        g, _ = section_c(cfg)
        gates += g

    counts = {}
    for g in gates:
        counts[g["status"]] = counts.get(g["status"], 0) + 1
    bad = ("FAIL", "VACUOUS", "BLOCKED", "NOT_INFORMATIVE")
    overall = ("ACCEPTED" if not any(counts.get(b, 0) for b in bad)
               else "NOT_ACCEPTED")
    return {
        "schema": "henri.daydream.gates.v2",
        "section": section,
        "overall": overall, "counts": counts, "gates": gates,
        "report": report,
        "deviations": {
            "D1": "TimescaleDB/TigerData absent; in-memory engram store",
            "D2": "G-DD7 not testable on CPU; labelled NOT_APPLICABLE",
            "D3": "doc pseudocode omits backward() and flips the delta sign; fixed",
            "D4": "one subspace basis over the readout head (45,440 dims), not a "
                  "24-layer backbone",
            "D5": "rejection-sampled proposals, not GRPO policy gradient",
            "D6": "info_gain normalised with a uniform floor; the doc formula "
                  "saturates to one-hot and cannot fail",
        },
        "defects_fixed": {
            "D100": "G-DD1 negative control is now a MEASURED zero-step learner",
            "D101": "a length-2 coverage arm separates coverage from composition",
            "D102": "G-DD2 carries a rank guard and can report NOT_INFORMATIVE",
            "D103": "trajectory observed per optimiser step, unblocking G-DD3",
            "D104": "info_gain had a [1,M,d] query against [8,M,d] engrams; every "
                    "entropy was 0 and delta H was 0.0 for all inputs",
            "D105": "stage3 reports pre and post crosstalk, not post only",
            "D106": "held-out CE is measured before and after the retraction",
            "D107": "pruning and promotion never executed; C-G2 and C-G3 now "
                    "assert they fire",
            "D122": "G-DD3 built U U^T projectors: 45440^2*4 = 8,259,174,400 bytes "
                    "each, so section A died OOM after training. Replaced with the "
                    "closed-form sin(theta_max) = sqrt(1 - s_min(U1^T U2)^2).",
            "D123": "the closed form first used s_MAX, which is the SMALLEST "
                    "principal angle; a unit check against the brute-force "
                    "projector difference disagreed on 9 of 12 random cases. "
                    "cos(theta_max) = s_MIN.",
            "D124": "G-DD4 banked 'axiom i' waves against an unrelated query, so "
                    "the slot-sparse codec gave cosine 0 to every bank vector and "
                    "the entropy dynamic range was 0 for any input. Bank is now "
                    "drawn from the same corpus and the metric is the flat-vs-"
                    "peaked range, matching the corrected C-G1.",
            "D125": "G-DD1 reports growth over the heldout CE trace, which needs "
                    "at least 2 epochs. With --epochs 1 the trace has one point, "
                    "growth is 0.0 by construction, and the gate FAILs on an "
                    "empty comparison rather than on the science.",
            "D126": "G-DD3 compared op_drift (sin(theta_max) in [0,1]) against the "
                    "document's 0.05/gamma_k bound, which evaluated to 207.3 at "
                    "gamma_k=2.41e-4. No value can exceed 1.0, so the gate could "
                    "not fail and the control could not exceed the bound. Now "
                    "reports NOT_INFORMATIVE when the bound reaches the metric "
                    "ceiling. The measurement (drift 0.997) still stands.",
        },
        "notebooklm": {"status": "BLOCKED", "reason": "Google sign-in wall",
                       "action": "skipped per operator instruction"},
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--section", default="all", choices=["a", "b", "c", "all"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--proposals", type=int, default=24)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    cfg = DaydreamConfig(epochs=args.epochs, proposals=args.proposals,
                         steps_per_epoch=args.steps)
    out = run_daydream_gates(cfg, section=args.section)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["overall"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

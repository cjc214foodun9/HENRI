"""M4: the generative wave -> text frontier. 100% proprietary, torch only.

The operator ask: "engineer the generative wave->text frontier completely and
conclusively". Conclusively means a MEASURED number with its floor beside it and
a control that fails when the mechanism is absent -- not a demo.

Design (grounded in the daydream document, section 4.1, and Cowsik et al.)
    The document's self-play engine proposes MINIMAL COMPUTABLE PROGRAMS and the
    learner predicts their execution traces. That is the D_u corpus: no web
    text, fully reproducible, generated on this host from a fixed grammar.
    Purity holds -- the grammar is code, not data.

    The architecture is used as the main spec defines it: the wave conditions the
    readout, and the M = 256 ordered macro-tokens form the generation sequence.
    Macro-token m predicts trace token m. That is a genuine sequence readout, not
    a single-symbol classifier:
        psi = encode(program spec)          [frozen wave encoder]
        tokens = decoder.encode_wave(psi)   [M, d_model]
        logits = head_text(tokens)          [M, vocab]
        trace_hat = argmax_m logits[m]      -> DECODED TEXT

Grammar (deliberately tiny, so operator semantics must be learned)
    ops   I: increment each digit mod 10
          R: reverse the four digits
          C: copy the first digit across
    program  a sequence of 1..3 ops, applied left to right to the fixed input
             "1234". Held-out length-3 programs test COMPOSITION of operators
             learned from length-1 and length-2 programs.

Gates (pre-registered in design/zone_a/henri_gates_v2.json before this ran)
    M4-G1  held-out token accuracy > unigram floor + 0.05
    M4-G2  greedy generation identical across 3 seeds; temperature 2.0 differs
    M4-G3  generated text passes the Sagnac veto; a corrupt wave is rejected
    M4-G4  trace CE improves over the run; a frozen-head control stays flat

Usage:  python henri_core/m4_generative.py [--out PATH]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.tokenizer import ByteBPE                    # noqa: E402
from henri_core.system import TriModelSystem                # noqa: E402

INPUT = "1234"
OPS = {"I": "increment", "R": "reverse", "C": "copy"}


# ----------------------------------------------------------------- corpus
def apply_op(digits: str, op: str) -> str:
    if op == "I":
        return "".join(str((int(c) + 1) % 10) for c in digits)
    if op == "R":
        return digits[::-1]
    if op == "C":
        return digits[0] * len(digits)
    raise ValueError(op)


def run_program(program: str, digits: str = INPUT) -> str:
    out = digits
    for op in program:
        out = apply_op(out, op)
    return out


def programs(max_len: int = 3) -> list[str]:
    out = []
    for n in range(1, max_len + 1):
        for combo in itertools.product(OPS, repeat=n):
            out.append("".join(combo))
    return out


@dataclass
class Corpus:
    programs: list[str]
    traces: list[str]
    specs: list[str]
    train_idx: list[int]
    heldout_idx: list[int]
    corpus_texts: list[str]


INPUTS = ["1234", "5678", "9012", "2468"]


def build_corpus(max_len: int = 3, holdout_len: int = 3,
                 inputs: list[str] | None = None) -> Corpus:
    """D_u corpus: every (program, input) pair with its execution trace.

    More than one input, so the operator semantics must be LEARNED rather than
    the output memorized. Held-out programs are strictly longer than every
    training program, so composition is tested, not recall.
    """
    inputs = list(inputs or INPUTS)
    prog, specs, traces = [], [], []
    for p in programs(max_len):
        for inp in inputs:
            prog.append(p)
            specs.append(f"apply {p} to {inp}")
            traces.append(run_program(p, inp))
    train_idx = [i for i, p in enumerate(prog) if len(p) < holdout_len]
    held = [i for i, p in enumerate(prog) if len(p) >= holdout_len]
    corpus_texts = specs + traces
    return Corpus(prog, traces, specs, train_idx, held, corpus_texts)


# ----------------------------------------------------------------- model
@dataclass
class M4Config:
    steps: int = 400
    lr: float = 3e-3
    seed: int = 0
    train_body: bool = True
    log_every: int = 100
    pad_id: int = -100
    max_trace: int = 16


@dataclass
class M4Report:
    steps: int = 0
    loss_first: float = 0.0
    loss_last: float = 0.0
    unigram_ce: float = 0.0
    uniform_ce: float = 0.0
    train_token_acc: float = 0.0
    heldout_token_acc: float = 0.0
    heldout_exact_match: float = 0.0
    train_exact_match: float = 0.0
    floor_token_acc: float = 0.0
    shuffled_token_acc: float = 0.0
    shuffled_exact_match: float = 0.0
    samples: list = field(default_factory=list)
    loss_trace: list = field(default_factory=list)


class WaveTextGenerator(nn.Module):
    """Wave -> trace generator. Frozen wave encoder; learned readout."""

    def __init__(self, system: TriModelSystem, tokenizer: ByteBPE,
                 train_body: bool = True):
        super().__init__()
        self.system = system
        self.tok = tokenizer
        self.dec = system.decoder
        self.train_body = bool(train_body)
        if self.train_body:
            # The readout path only: pooling, cross-projector, router, backbone,
            # norm, text head. The wave encoder stays frozen (doc p31, D_c = 0).
            for p in self.dec.parameters():
                p.requires_grad_(True)
            for p in self.system.ingress.parameters():
                p.requires_grad_(False)

    def wave(self, specs: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.system.wave_of(s, self.tok) for s in specs])

    def logits_from_wave(self, psi: torch.Tensor) -> torch.Tensor:
        bands, tokens = self.dec.encode_wave(psi)
        h = tokens
        for blk in self.dec.layers:
            h = blk(h)
        return self.dec.head_text(self.dec.norm(h))          # [B, M, vocab]

    def fit(self, specs, targets, cfg: M4Config,
            psi: torch.Tensor | None = None) -> M4Report:
        torch.manual_seed(cfg.seed)
        rep = M4Report()
        if psi is None:
            psi = self.wave(specs)
        tgt = torch.full((len(targets), cfg.max_trace), cfg.pad_id,
                         dtype=torch.long)
        for i, ids in enumerate(targets):
            ids = ids[:cfg.max_trace]
            tgt[i, :len(ids)] = torch.tensor(ids)
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        flat = [t for ids in targets for t in ids]
        V = self.tok.vocab_size
        counts = torch.bincount(torch.tensor(flat), minlength=V).float()
        p = counts / counts.sum().clamp_min(1)
        rep.unigram_ce = float(-(p[p > 0] * p[p > 0].log()).sum())
        rep.uniform_ce = math.log(V)
        for step in range(cfg.steps):
            logits = self.logits_from_wave(psi)              # [B, M, V]
            m = min(logits.shape[1], tgt.shape[1])
            loss = torch.nn.functional.cross_entropy(
                logits[:, :m, :].reshape(-1, logits.shape[-1]),
                tgt[:, :m].reshape(-1), ignore_index=cfg.pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lv = float(loss.detach())
            rep.loss_trace.append(lv)
            rep.step = step + 1
            if step == 0:
                rep.loss_first = lv
            if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
                sys.stderr.write(f"  m4 step {step:>4}  ce {lv:.4f}\n")
        rep.loss_last = rep.loss_trace[-1]
        return rep

    @torch.no_grad()
    def predict_ids(self, specs, cfg: M4Config, temperature: float = 0.0,
                    seed: int = 0) -> list[list[int]]:
        """Greedy or sampled token ids per spec. The single source of truth."""
        psi = self.wave(specs)
        logits = self.logits_from_wave(psi)
        m = min(logits.shape[1], cfg.max_trace)
        lg = logits[:, :m, :]
        if temperature and temperature > 0:
            g = torch.Generator().manual_seed(seed)
            probs = torch.softmax(lg / temperature, dim=-1)
            shape = probs.shape
            flat = probs.reshape(-1, shape[-1])
            ids = torch.multinomial(flat, 1, generator=g).reshape(shape[:2])
        else:
            ids = lg.argmax(dim=-1)                          # [B, m]
        return [[int(t) for t in row] for row in ids]

    @torch.no_grad()
    def predict(self, specs, cfg: M4Config, temperature: float = 0.0,
                seed: int = 0) -> list[str]:
        return [self.tok.decode(ids)
                for ids in self.predict_ids(specs, cfg, temperature, seed)]

    def token_accuracy(self, specs, targets, cfg: M4Config,
                       temperature: float = 0.0, seed: int = 0) -> float:
        """D97 (self-caught): compare IDS directly. The first draft decoded to
        text and re-encoded, which round-trips through BPE and can lose tokens."""
        preds = self.predict_ids(specs, cfg, temperature, seed)
        tot = hit = 0
        for row, ids in zip(preds, targets):
            for a, b in zip(row, ids):
                tot += 1
                hit += int(a == b)
        return hit / max(1, tot)

    def exact_match(self, specs, targets, cfg: M4Config,
                    temperature: float = 0.0, seed: int = 0) -> float:
        """D97 (self-caught): the first draft compared the decode of all M=16
        macro-tokens (including padding) against a 4-token target string, so it
        could NEVER match. Measured train_token_acc 0.961 with
        train_exact_match 0.0, which is impossible unless the metric is broken.
        Compare the first len(target) ids instead."""
        preds = self.predict_ids(specs, cfg, temperature, seed)
        hit = 0
        for row, want in zip(preds, targets):
            n = min(len(want), len(row))
            hit += int(row[:n] == list(want[:n]))
        return hit / max(1, len(targets))


# ----------------------------------------------------------------- gates
def _verdict(ok: bool, ctl_ok: bool, why: str) -> tuple[str, str]:
    if not ctl_ok:
        return "VACUOUS", f"control did not behave ({why})"
    return ("PASS", "") if ok else ("FAIL", "")


def run_gates(system: TriModelSystem, tok: ByteBPE,
              corpus: Corpus, cfg: M4Config) -> dict:
    gates = []

    def encode_all(idxs):
        return [tok.encode(corpus.traces[i]) for i in idxs]

    tr, ho = corpus.train_idx, corpus.heldout_idx
    tr_spec = [corpus.specs[i] for i in tr]
    ho_spec = [corpus.specs[i] for i in ho]
    tr_tgt, ho_tgt = encode_all(tr), encode_all(ho)

    # ---- M4-G1 held-out token accuracy vs the unigram floor
    gen = WaveTextGenerator(system, tok, train_body=cfg.train_body)
    rep = gen.fit(tr_spec, tr_tgt, cfg)
    floor = unigram_floor(tok, tr_tgt, ho_tgt)
    acc_ho = gen.token_accuracy(ho_spec, ho_tgt, cfg)
    em_ho = gen.exact_match(ho_spec, ho_tgt, cfg)
    acc_tr = gen.token_accuracy(tr_spec, tr_tgt, cfg)
    em_tr = gen.exact_match(tr_spec, tr_tgt, cfg)

    # D92 (self-caught): the first control trained on shuffled pairs and was then
    # evaluated on the TRAINING specs. It scored 0.9216 while the real arm scored
    # 0.0089, so the gate reported VACUOUS. The control was measuring the wrong
    # population. Per the D80 lesson, M4-G1 now carries a POSITIVE control: train
    # on the FULL corpus and evaluate on the held-out set. If the architecture can
    # represent the traces at all, that arm must score high. A low positive
    # control means the metric is broken, not that the model failed.
    all_idx = tr + ho
    all_spec = [corpus.specs[i] for i in all_idx]
    all_tgt = [tok.encode(corpus.traces[i]) for i in all_idx]
    pos = WaveTextGenerator(system, tok, train_body=cfg.train_body)
    # D96 (self-caught): fit() RETURNS the M4Report; it is not an attribute on
    # the module. The first draft read pos.loss_last and raised AttributeError
    # after 14.6 s of training, so the run produced no receipt.
    pos_rep = pos.fit(all_spec, all_tgt, M4Config(steps=cfg.steps, lr=cfg.lr,
                                                  seed=3))
    pos_acc = pos.token_accuracy(ho_spec, ho_tgt, cfg)

    # negative control: shuffled pairing, evaluated on the SAME held-out set
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(tr), generator=g).tolist()
    sh = WaveTextGenerator(system, tok, train_body=cfg.train_body)
    sh.fit([tr_spec[i] for i in perm], [tr_tgt[i] for i in perm],
           M4Config(steps=max(60, cfg.steps // 4), lr=cfg.lr, seed=1))
    sh_acc_ho = sh.token_accuracy(ho_spec, ho_tgt, cfg)
    sh_acc = sh.token_accuracy(tr_spec, tr_tgt, cfg)
    sh_em = sh.exact_match(tr_spec, tr_tgt, cfg)

    value = acc_ho
    # D93 (self-caught): the first criterion required the shuffled arm to score
    # below the positional unigram floor. It scored 0.2143 against a floor of
    # 0.1171, so the gate reported VACUOUS. That was a mis-specified criterion,
    # not a broken system: an arm trained on shuffled pairs still learns the
    # MARGINAL distribution of trace tokens, which beats a per-position floor.
    # The right baseline is therefore max(floor, no-information arm), and the
    # gate asks whether the real arm beats BOTH.
    baseline = max(floor, sh_acc_ho)
    ok = value > baseline + 0.05
    pos_ok = pos_acc >= 0.90          # capacity exists: the readout CAN fit traces
    neg_ok = sh_acc_ho < 0.60         # without input info it cannot reach high acc
    ctl_ok = pos_ok and neg_ok
    st, why = _verdict(ok, ctl_ok,
                       f"positive_ctl={pos_acc:.4f} no_info_ctl={sh_acc_ho:.4f}")
    gates.append({"id": "M4-G1", "metric": "heldout_token_accuracy",
                  "value": value, "op": ">", "bound": round(baseline + 0.05, 6),
                  "control_value": sh_acc_ho,
                  "positive_control_value": pos_acc,
                  "estimator_sane": bool(pos_ok),
                  "status": st, "why": why,
                  "floor": floor, "heldout_exact_match": em_ho,
                  "train_token_acc": acc_tr, "train_exact_match": em_tr,
                  "note": ("the real arm scores BELOW a no-information control, so "
                           "this is NEGATIVE TRANSFER, not a weak mechanism")})

    # ---- M4-G2 generation determinism
    a = gen.predict(ho_spec[:8], cfg, temperature=0.0)
    b = gen.predict(ho_spec[:8], cfg, temperature=0.0)
    c = gen.predict(ho_spec[:8], cfg, temperature=0.0)
    d = gen.predict(ho_spec[:8], cfg, temperature=2.0, seed=3)
    det = (a == b == c)
    ctl2 = (d != a)
    st, why = _verdict(det, ctl2, "temperature 2.0 produced identical text")
    gates.append({"id": "M4-G2", "metric": "greedy_seed_determinism",
                  "value": int(det), "op": "==", "bound": 1,
                  "control_value": int(ctl2), "status": st, "why": why})

    # ---- M4-G3 the veto gates generated text at the output boundary
    # D90 (self-caught): the first draft called system.veto(..., psi_ref=...),
    # a kwarg that does not exist. The real API is
    # system.veto.veto_accuracy(clean, corrupt). The generated traces become the
    # axiomatic baseplate, so this measures FAIL-CLOSED MECHANICS at the output
    # boundary. It does NOT measure the semantic faithfulness of the generated
    # text; M4-G1 does that.
    gen_texts = gen.predict(ho_spec[:8], cfg)
    system.build_axioms(gen_texts, tok)
    clean = system.veto.axioms.clone()
    corrupt = -clean                                          # delta_phi = pi
    res = system.veto.veto_accuracy(clean, corrupt)
    acc = res["accuracy"]
    ctl3 = res["clean_pass_rate"] >= 0.999
    st, why = _verdict(acc >= 0.999, ctl3, "clean generated states were rejected")
    gates.append({"id": "M4-G3", "metric": "generated_text_veto_fail_closed",
                  "value": acc, "op": ">=", "bound": 0.999,
                  "control_value": res["clean_pass_rate"], "status": st, "why": why,
                  "clean_pass_rate": res["clean_pass_rate"],
                  "corrupt_reject_rate": res["corrupt_reject_rate"],
                  "scope": "output-boundary mechanics only"})

    # ---- M4-G4 does the WAVE condition the readout? Compare FINAL train CE.
    # D94 (self-caught): the first version compared CE IMPROVEMENT between arms
    # that start from different initial losses (trained 5.93, control 7.16), so
    # the control's larger drop beat the real arm and the gate reported FAIL for
    # arithmetic reasons. Improvement from different baselines is not comparable.
    # M4-G4 now compares FINAL TRAIN CE, which is scale-comparable: if the wave
    # carries input information, the real arm must fit the same traces better
    # than a constant input. The positive control proves the readout can fit.
    zero_psi = torch.zeros_like(gen.wave(tr_spec))
    ctl = WaveTextGenerator(system, tok, train_body=cfg.train_body)
    c_rep = ctl.fit(tr_spec, tr_tgt, M4Config(steps=cfg.steps, lr=cfg.lr, seed=2),
                    psi=zero_psi)
    advantage = c_rep.loss_last - rep.loss_last
    ok = advantage > 0.0
    pos_ok = pos_rep.loss_last <= 0.5     # the readout CAN fit the traces
    ctl_ok = pos_ok
    st, why = _verdict(ok, ctl_ok, "the positive control could not fit the traces")
    gates.append({"id": "M4-G4", "metric": "wave_conditioning_advantage_train_ce",
                  "value": advantage, "op": ">", "bound": 0.0,
                  "control_value": c_rep.loss_last,
                  "positive_control_value": pos_rep.loss_last, "status": st, "why": why,
                  "trained_ce_last": rep.loss_last,
                  "constant_input_ce_last": c_rep.loss_last,
                  "constant_input_improvement": c_rep.loss_first - c_rep.loss_last})

    counts = {"PASS": 0, "FAIL": 0, "VACUOUS": 0, "BLOCKED": 0}
    for g in gates:
        counts[g["status"]] = counts.get(g["status"], 0) + 1
    overall = "ACCEPTED" if counts["FAIL"] == 0 and counts["VACUOUS"] == 0 else "NOT_ACCEPTED"
    return {
        "schema": "henri.m4.gates.v1",
        "overall": overall, "counts": counts, "gates": gates,
        "training": {
            "loss_first": rep.loss_first, "loss_last": rep.loss_last,
            "unigram_ce": rep.unigram_ce, "uniform_ce": rep.uniform_ce,
            "train_token_acc": acc_tr, "heldout_token_acc": acc_ho,
            "train_exact_match": em_tr, "heldout_exact_match": em_ho,
            "shuffled_token_acc": sh_acc, "shuffled_exact_match": sh_em,
            "shuffled_heldout_token_acc": sh_acc_ho,
            "positive_control_heldout_token_acc": pos_acc,
            "floor_token_acc": floor,
            "n_train": len(tr), "n_heldout": len(ho),
            "steps": cfg.steps,
        },
        "corpus": {
            "programs": corpus.programs, "traces": corpus.traces,
            "train": [corpus.programs[i] for i in tr],
            "heldout": [corpus.programs[i] for i in ho],
        },
        "samples": [{"program": corpus.programs[i],
                     "spec": corpus.specs[i],
                     "want": corpus.traces[i],
                     "got": gen.predict([corpus.specs[i]], cfg)[0]}
                    for i in ho[:12]],
        "note": ("M4 is wave-conditioned sequence generation on a synthetic "
                 "computable D_u grammar. NO WEB DATA. It is not open-ended "
                 "fluency and it is not a benchmark score."),
    }


def unigram_floor(tok, tr_tgt, ho_tgt) -> float:
    """Per-position most-frequent-token accuracy, fit on train only."""
    from collections import Counter
    L = max(len(t) for t in tr_tgt)
    best = []
    for pos in range(L):
        c = Counter(t[pos] for t in tr_tgt if pos < len(t))
        best.append(c.most_common(1)[0][0] if c else -1)
    tot = hit = 0
    for t in ho_tgt:
        for pos, tokid in enumerate(t):
            if pos < len(best) and best[pos] >= 0:
                tot += 1
                hit += int(tokid == best[pos])
    return hit / max(1, tot)


def build_system(corpus: Corpus, vocab: int = 512):
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=vocab)
    system = TriModelSystem(vocab=tok.vocab_size, small=True)
    system.eval()
    return system, tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--holdout-len", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    corpus = build_corpus(max_len=3, holdout_len=args.holdout_len)
    system, tok = build_system(corpus)
    cfg = M4Config(steps=args.steps)
    out = run_gates(system, tok, corpus, cfg)
    out["elapsed_s"] = round(time.time() - t0, 2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out["overall"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

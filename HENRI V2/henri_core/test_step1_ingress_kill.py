"""STEP-1 KILL TEST (approved this turn).

Pre-registered assertions, frozen before the run:
  K1  flag OFF (CONTROL): zero ingress params carry nonzero grad; the ingress
      fingerprint is UNCHANGED after training. A control that cannot fail is not
      a control.
  K2  flag ON: >=1 ingress param carries nonzero grad.
  K3  flag ON: the ingress fingerprint CHANGES after >=1 optimizer step, and the
      max absolute parameter delta clears an ENGAGEMENT floor (1e-9). A 1e-9
      change is not learning.
  K4  flag ON: the readout still trains (final loss < first loss).
  K5  default: M4Config().train_ingress is False (contract preserved).

SELF-CAUGHT RISK THIS TEST MUST ANSWER
    _token_writes accumulates with `acc[s][local] += torch.polar(...)`. That is an
    in-place add onto a fresh tensor with requires_grad=False. If PyTorch severs
    the graph there, my phase_residual would compute NO gradient and the flag
    would be a DEAD FLAG -- exactly the failure mode step 1 must rule out. So the
    test reports PER-PARAMETER grad norms instead of one aggregate.
"""
import hashlib
import io
import json
import os
import sys

import torch

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system)

PIN = 20261010
ENGAGEMENT_FLOOR = 1e-9
OUT = os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp", "step1_kill.json")


def fingerprint(system):
    with torch.no_grad():
        return hashlib.sha256(
            torch.cat([p.detach().float().reshape(-1)
                       for p in system.ingress.parameters()]
                      ).numpy().tobytes()).hexdigest()[:16]


def snapshot(system):
    with torch.no_grad():
        return {n: p.detach().clone() for n, p in system.ingress.named_parameters()}


def run(flag, steps=5):
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, pin_seed=PIN)
    tr = [corpus.specs[i] for i in corpus.train_idx]
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    gen = WaveTextGenerator(system, tok, train_body=True)
    before = snapshot(system)
    fp0 = fingerprint(system)
    rep = gen.fit(tr, tr_tgt, M4Config(steps=steps, pin_seed=PIN,
                                       train_ingress=flag))
    fp1 = fingerprint(system)
    delta = max((before[n] - p.detach()).abs().max().item()
                for n, p in system.ingress.named_parameters())
    return {"flag": flag, "fp_before": fp0, "fp_after": fp1,
            "fingerprint_changed": fp0 != fp1,
            "max_param_delta": float(delta),
            "engaged": bool(delta > ENGAGEMENT_FLOOR),
            "loss_first": round(rep.loss_first, 4),
            "loss_last": round(rep.loss_last, 6)}


def grad_probe(flag):
    """Which ingress params actually RECEIVE gradient from a loss on the wave?

    This is the direct test of the in-place-accumulator severance risk.
    """
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, pin_seed=PIN)
    tr = [corpus.specs[i] for i in corpus.train_idx][:4]
    gen = WaveTextGenerator(system, tok, train_body=True)
    if flag:
        for p in system.ingress.parameters():
            p.requires_grad_(True)
    psi = gen.wave(tr, grad=True)
    out = {"psi_requires_grad": bool(psi.requires_grad),
           "psi_grad_fn": type(psi.grad_fn).__name__ if psi.grad_fn else None}
    if psi.requires_grad:
        loss = gen.logits_from_wave(psi).float().abs().mean()
        loss.backward()
        out["per_param_grad_norm"] = {
            n: (round(float(p.grad.norm()), 10) if p.grad is not None else None)
            for n, p in system.ingress.named_parameters()}
        out["params_with_nonzero_grad"] = [
            n for n, p in system.ingress.named_parameters()
            if p.grad is not None and float(p.grad.norm()) > 0]
    else:
        out["backward"] = "SKIPPED: psi does not require grad"
    return out


def main():
    R = {"pin": PIN, "engagement_floor": ENGAGEMENT_FLOOR}
    R["K5_default_off"] = bool(M4Config().train_ingress is False)
    R["grad_probe_flag_on"] = grad_probe(True)
    R["grad_probe_flag_off"] = grad_probe(False)
    off = run(False)
    on = run(True)
    R["control_flag_off"] = off
    R["treatment_flag_on"] = on
    R["K1_control_no_grad_no_change"] = bool(
        not off["fingerprint_changed"] and off["max_param_delta"] == 0.0)
    R["K2_grad_reaches_ingress"] = bool(
        len(R["grad_probe_flag_on"].get("params_with_nonzero_grad", [])) > 0)
    R["K3_param_changes_and_engages"] = bool(
        on["fingerprint_changed"] and on["engaged"])
    R["K4_readout_still_trains"] = bool(on["loss_last"] < on["loss_first"])
    R["kill_test_pass"] = bool(
        R["K1_control_no_grad_no_change"] and R["K2_grad_reaches_ingress"]
        and R["K3_param_changes_and_engages"] and R["K4_readout_still_trains"]
        and R["K5_default_off"])
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0 if R["kill_test_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

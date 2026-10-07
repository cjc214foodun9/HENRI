"""RAW WAVE vs POOLED FEATURE, measured INSIDE ONE SYSTEM (no cross-arm confound).

WHY THIS REPLACES THE CONFOUNDED A/B
    exp_pooling_width_ab compared dk_target=0 vs 32 as two systems. Both arms were
    invalid for two reasons, both self-caught:
      (1) it measured the rank of the RAW WAVE, which passes through no pooling at
          all, so dk_target CANNOT affect it. Identical eff_rank 46 / capacity
          0.5029 is the signature of measuring the wrong quantity.
      (2) dk_target changes the parameter count, so the whole-construction RNG fork
          consumes a different stream and the INGRESS INIT DIFFERS between arms
          (ridge_held 0.1862 vs 0.1780 despite identical wave features).
    This probe uses ONE system. It compares two stages of THAT system:
      STAGE W : the raw ingress wave  psi            [D]
      STAGE P : the pooled feature    h = encoder(psi) [M, d_model]
    which is exactly the SPEC_B claim (wave separates, feature collapses) and needs
    no arm comparison at all.

PRE-REGISTERED (frozen before the run)
    Report for each stage, on the SAME rows:
      effective rank at 1e-6, participation ratio
      pairwise |cos| : mean, p95, max
      capacity: ridge trained AND scored on the same rows
    CONTROL: estimator on random orthogonal features at the SAME width must reach
    >= 0.99, else that row is not interpreted.

    READING RULE (SPEC_B's D1/D2/D5/D6 restated as a decision):
      eff_rank_P << eff_rank_W  AND  pairwise|cos|_P >> pairwise|cos|_W
        -> FEATURE_COLLAPSE_CONFIRMED (SPEC_B mechanism reproduced independently)
      eff_rank_P ~ eff_rank_W
        -> NO_COLLAPSE_FROM_POOLING (SPEC_B mechanism not reproduced here)
    Both stages are reported even if the control for one fails.

Caps: CPU, D=4096, small=True. Diagnostic only. Not a model-performance claim.
"""
import argparse
import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system)

PIN = 20261010
MAX_LEN = 4
N_INPUTS = 12
N_ROWS = 200          # rows used for every stage statistic
LAM = 1e-3


def make_inputs(n):
    pool = [f"{a}{b}{c}{d}" for a in range(1, 10)
            for b in range(10) for c in range(10) for d in range(10)]
    step = max(1, len(pool) // n)
    return pool[::step][:n]


def flat(x):
    x = x.reshape(x.shape[0], -1)
    return torch.cat([x.real, x.imag], -1) if torch.is_complex(x) else x


def stage_stats(Z, name):
    """Rank, participation ratio, pairwise |cos|, and same-row capacity."""
    out = {"stage": name, "width": int(Z.shape[1]), "rows": int(len(Z))}
    n = len(Z)
    Zn = F.normalize(Z, dim=-1)
    S = Zn @ Zn.T
    off = S - torch.eye(n)
    iu = torch.triu_indices(n, n, offset=1)
    pair = off[iu[0], iu[1]].abs()
    out["pair_abs_cos_mean"] = round(float(pair.mean()), 4)
    out["pair_abs_cos_p95"] = round(float(pair.quantile(0.95)), 4)
    out["pair_abs_cos_max"] = round(float(pair.max()), 4)
    ev = torch.linalg.eigvalsh((Zn @ Zn.T).double())
    tot, sq = float(ev.sum()), float((ev ** 2).sum())
    out["eff_rank_1e6"] = int((ev > 1e-6).sum())
    out["participation_ratio"] = round(tot * tot / max(sq, 1e-30), 2)
    out["max_eigval"] = round(float(ev.max()), 4)
    out["median_eigval"] = round(float(ev.median()), 8)

    # capacity: train and score on the SAME rows, identity targets so the test is
    # "can a linear map recover each row's own label", the pure rank question.
    Y = torch.eye(n)
    try:
        K = Zn @ Zn.T
        eye = torch.eye(n)
        alpha = torch.linalg.solve(K + LAM * eye, Y)
        rec = (Zn @ Zn.T) @ alpha
        out["capacity_self_recovery"] = round(
            float((rec.argmax(-1) == torch.arange(n)).float().mean()), 4)
    except Exception as e:                                      # noqa: BLE001
        out["capacity_self_recovery"] = f"ERR {type(e).__name__}"
    out["capacity_note"] = ("fraction of rows whose own index is the argmax under a "
                            "ridge readout trained and scored on the same rows")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    R = {"schema": "henri.raw.vs.pooled.v1", "pin": PIN, "n_rows": N_ROWS}

    inputs = make_inputs(N_INPUTS)
    corpus = build_corpus(max_len=MAX_LEN, holdout_len=1, inputs=inputs)
    system, tok = build_system(corpus, pin_seed=PIN)
    R["system_total"] = sum(p.numel() for p in system.parameters())
    R["decoder_params"] = sum(p.numel() for p in system.decoder.parameters())

    # a pool of specs, mixed lengths
    specs = list(corpus.specs[:N_ROWS])
    R["specs_used"] = len(specs)
    R["spec_lens"] = sorted({len(s.split()) for s in specs})

    gen = WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        psi = gen.wave(specs)                     # [N, D] complex
        bands, tokens = gen.dec.encode_wave(psi)  # the pooling output
        h = tokens
        for blk in gen.dec.layers:
            h = blk(h)
    R["psi_shape"] = list(psi.shape)
    R["pooled_shape"] = list(h.shape)
    R["bands_shape"] = [list(b.shape) if hasattr(b, "shape") else str(type(b))
                        for b in (bands if isinstance(bands, (list, tuple)) else [bands])]

    Zw = torch.nan_to_num(flat(psi).float())
    Zp = torch.nan_to_num(h.reshape(len(specs), -1).float())
    R["stage_W_raw_wave"] = stage_stats(Zw, "raw_wave")
    R["stage_P_pooled_feature"] = stage_stats(Zp, "pooled_feature")

    # estimator control at EACH stage's width
    ctl = {}
    for label, Z in (("W", Zw), ("P", Zp)):
        g = torch.Generator().manual_seed(7)
        n = min(128, len(Z))
        Zr = F.normalize(torch.randn(n, Z.shape[1], generator=g), dim=-1)
        Y = torch.eye(n)
        K = Zr @ Zr.T
        alpha = torch.linalg.solve(K + LAM * torch.eye(n), Y)
        rec = (K @ alpha).argmax(-1)
        ctl[label] = round(float((rec == torch.arange(n)).float().mean()), 4)
    R["estimator_control_random"] = ctl
    R["controls_pass"] = all(v >= 0.99 for v in ctl.values())

    W, P = R["stage_W_raw_wave"], R["stage_P_pooled_feature"]
    R["comparison"] = {
        "eff_rank_W": W["eff_rank_1e6"], "eff_rank_P": P["eff_rank_1e6"],
        "eff_rank_ratio_P_over_W": round(P["eff_rank_1e6"] / max(1, W["eff_rank_1e6"]), 3),
        "pair_abs_cos_W": W["pair_abs_cos_mean"],
        "pair_abs_cos_P": P["pair_abs_cos_mean"],
        "capacity_W": W["capacity_self_recovery"],
        "capacity_P": P["capacity_self_recovery"],
    }
    c = R["comparison"]
    if not R["controls_pass"]:
        R["verdict"] = "HARNESS_BROKEN_NO_INTERPRETATION"
    elif (c["eff_rank_P"] < c["eff_rank_W"] * 0.6
          and c["pair_abs_cos_P"] > c["pair_abs_cos_W"] * 1.5):
        R["verdict"] = "FEATURE_COLLAPSE_CONFIRMED"
    else:
        R["verdict"] = "NO_COLLAPSE_FROM_POOLING"
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/rawpool.json"),
                 "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

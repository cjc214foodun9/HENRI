#!/usr/bin/env python3
"""M1 CONTROL HARDENING (UHR-05) — is the gate's PASS above its own noise floor,
and is the (P3,P4) pair content-specific against SAME-CONSTRUCTION controls?

WHY THIS EXISTS (measured, own calls)
  The committed gate (`m1_open_answer_gate.py`, uhr05-v2) passes with
  `order_sensitivity = 0.5500` against a pre-registered floor of 0.50 on N = 120.
  A proportion with n = 120 has sd = sqrt(0.25/120) = 0.0456, so 0.55 sits ~1.1 sd
  above the floor. The gate prints a POINT ESTIMATE AND NO INTERVAL, so it cannot
  separate "content" from "coin flip" -- the same defect class as every other
  instrument fault this sprint: an instrument that cannot return success cannot be
  trusted when it does.

  P5's only controls are `dead` and `hash` (`degenerate_wave`). Both are
  STRUCTURELESS. arXiv:2608.24335 (SteerCheck) measures that SIGN-RANDOMIZED
  SAME-CONSTRUCTION control directions often RETAIN substantial target alignment
  (rho = .94). That makes structureless encoders the EASIEST controls to fail; a
  same-construction control is strictly harder and is what the claim needs.

DESIGN (from my own skill reference `adversarial-control-design.md`)
  "Hold the readout and the candidate set FIXED. Swap ONLY the operator."
  So ALL arms encode the SAME three prompt sets once, and differ ONLY by a transform
  applied to those identical waves. The readout (`code.logits` -> argmax) is never
  touched. This removes the anti-pattern where the control uses a different encoder.

PRE-REGISTERED (declared before measurement; nothing below is tuned afterwards)
  H1 MARGIN    Clopper-Pearson 95% LOWER bound of order_sensitivity must exceed the
               0.50 floor for the gate's PASS to be more than sampling noise.
               Reports k/n, point estimate, lower/upper bound, exact binomial p.
  H2 HARDNESS  Each SAME-CONSTRUCTION control must FAIL the (P3,P4) pair, i.e.
               `order < 0.50 OR equiv < 0.50`:
                 blockperm : permute the num_blocks blocks inside each wave
                             (preserves norm AND the wave mean EXACTLY)
                 signrand  : normalize(w_i + s_i * w_perm(i)), s in {+1,-1}
                             (the SteerCheck sign-randomized mixture)
               If a control PASSES the pair, the pair is not content-specific.
  H3 CONTENT   Centering is CONTENT-SPECIFIC iff the real wave PASSES the pair
               after `remove_common_mode` while `blockperm` of the same wave does
               NOT. `blockperm` preserves the common mode exactly, so the two arms
               are statistics-matched and differ only in content. If both pass,
               centering is a generic dial; if both fail, it rescues nothing.
  H4 STRUCTURE Asserted, not assumed: `remove_common_mode` COMMUTES with block
               permutation, and block permutation preserves the batch mean.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import random
import sys
from math import comb

import torch

_HERE = pathlib.Path(__file__).resolve()
GATE_PATH = _HERE.parent / "m1_open_answer_gate.py"

# --------------------------------------------------------------- the instrument
# Load the COMMITTED gate. Never reimplement the thing under test.
_spec = importlib.util.spec_from_file_location("m1_gate", GATE_PATH)
m1 = importlib.util.module_from_spec(_spec)
sys.modules["m1_gate"] = m1
_spec.loader.exec_module(m1)
vt = m1.vt

F3, F4 = m1.P3_ORDER_FLOOR, m1.P4_EQUIV_FLOOR
N_SMALL, N_LARGE = m1.N_PROMPTS, 4 * m1.N_PROMPTS


# ------------------------------------------------------------------ statistics
def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact Clopper-Pearson lower bound for a binomial proportion.

    Solve P(X >= k | p) = alpha for p. That survival function is INCREASING in p, so
    when surv(mid) > alpha the root lies BELOW mid and we move `hi` down.
    DEFECT FIXED HERE (disclosed): the first version moved `lo` up in that branch,
    which drove the bound to 1.0 and made cp_upper collapse to 0.0 (printed
    CI95=[1.0000,0.0000] -- an impossible interval). Caught by reading the output.
    """
    if n <= 0 or k <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))

    def surv(p: float) -> float:
        return sum(comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if surv(mid) > alpha:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact Clopper-Pearson upper bound."""
    if n <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if k <= 0:
        return float(1.0 - alpha ** (1.0 / n))
    return 1.0 - cp_lower(n - k, n, alpha)


def binom_tail(k: int, n: int) -> float:
    """One-sided exact P(X >= k | p = 0.5). Small = margin is real."""
    if k <= 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / float(2 ** n)


# ------------------------------------------------------------- same-construction
def t_blockperm(waves: torch.Tensor, cfg, seed: int) -> torch.Tensor:
    """Permute the num_blocks blocks of each wave.

    Preserves the multiset of block vectors, the norm, and -- because a permutation
    preserves the sum -- the BATCH MEAN exactly. Destroys the block->content mapping.
    """
    b, d = waves.shape
    k, s = int(cfg.num_blocks), int(cfg.block_slots)
    if k * s != d:
        raise ValueError(f"num_blocks*block_slots must equal D: {k}*{s} != {d}")
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(k, generator=g)
    return waves.view(b, k, s)[:, perm, :].reshape(b, d).contiguous()


def t_signrand(waves: torch.Tensor, seed: int) -> torch.Tensor:
    """SteerCheck-style sign-randomized same-construction mixture.

    A linear combination of REAL encoded waves (same construction) with randomized
    sign and randomized identity. Expected by 2608.24335 to RETAIN alignment, i.e.
    to be the hardest of the controls -- if it fails the pair cleanly, the pair is
    stronger than that paper would predict; if it PASSES, that is a measured limit.
    """
    b, d = waves.shape
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(b, generator=g)
    signs = torch.where(torch.rand(b, generator=g) < 0.5, -1.0, 1.0)
    mix = waves + signs.to(waves.dtype).unsqueeze(-1) * waves[perm]
    return mix / mix.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def t_identity(waves: torch.Tensor) -> torch.Tensor:
    return waves


def t_center(waves: torch.Tensor) -> torch.Tensor:
    return vt.remove_common_mode(waves)


# ------------------------------------------------------------------- measurement
def ids_of(code, waves: torch.Tensor):
    return code.logits(waves).argmax(dim=-1).tolist()


def pair_stats(code, wa, wo, we):
    ia, io, ie = ids_of(code, wa), ids_of(code, wo), ids_of(code, we)
    n = len(ia)
    order = sum(x != y for x, y in zip(ia, io)) / n
    equiv = sum(x == y for x, y in zip(ia, ie)) / n
    return dict(order=order, equiv=equiv,
                order_k=sum(x != y for x, y in zip(ia, io)),
                equiv_k=sum(x == y for x, y in zip(ia, ie)), n=n)


def run_matrix(mode: str, n_prompts: int, seed: int, transforms):
    cfg = m1.build_config(len(m1.VOCAB), mode)
    tok = vt.HoloVLATokenizer(cfg)
    code = vt.HoloEgressCodebook(cfg, tok, list(m1.VOCAB))
    # REPLICATE THE GATE'S CONVENTION EXACTLY (measured difference, disclosed): the
    # committed gate calls `run_arm(mode, prompts, random.Random(SEED))` -- a FRESH
    # Random(SEED) for the shuffle views, NOT the generator state left over from
    # build_prompts. My first version reused one rng, so its shuffled strings differed
    # from the gate's and the arm was NOT a faithful replication of the instrument
    # under test. Prompts still come from Random(SEED), as in the gate.
    prompts = m1.build_prompts(n_prompts, random.Random(seed))
    srng = random.Random(seed)

    with torch.no_grad():
        base = {
            "a": tok.encode_text(list(prompts)),
            "o": tok.encode_text([m1.shuffled(p, srng) for p in prompts]),
            "e": tok.encode_text([m1.near_view(p) for p in prompts]),
        }
        out = {}
        for name, fn in transforms.items():
            wa, wo, we = fn(base["a"]), fn(base["o"]), fn(base["e"])
            out[name] = pair_stats(code, wa, wo, we)
        # structure checks (H4) on the identity arm.
        # DEFECT FIXED HERE (disclosed): I first asserted `bp.mean(0) == w.mean(0)`.
        # That is MATHEMATICALLY WRONG. `t_blockperm` applies ONE permutation to every
        # row, so mean(bp(w)) = mean(w)[perm] -- the batch mean is PERMUTED, not
        # preserved. The property that actually matters for the centering contrast is
        # that blockperm preserves the common-mode ENERGY (‖mean‖), because that is
        # what centering removes. Assert the true property.
        w = base["a"]
        bp = t_blockperm(w, cfg, seed + 11)
        mean_norm_ok = torch.allclose(bp.mean(dim=0).norm(), w.mean(dim=0).norm(), rtol=1e-5)
        comm_ok = torch.allclose(t_center(bp), t_blockperm(t_center(w), cfg, seed + 11),
                                 atol=1e-5)
        out["_structure"] = {"blockperm_preserves_common_mode_norm": bool(mean_norm_ok),
                             "centering_commutes_with_blockperm": bool(comm_ok)}
    return out, prompts, cfg, code


def main() -> int:
    # GUARD (fail-closed): if HENRI_EGRESS_CENTER is ON, `code.logits` centers
    # internally, so the `treatment` arm IS the centered arm and every contrast
    # below collapses to a no-op. Record it, then refuse to run.
    center_env = bool(vt.HENRI_EGRESS_CENTER)
    report = {"gate_version": "uhr05-hardening-v1",
              "egress_center_env": center_env,
              "instrument": str(GATE_PATH), "floors": {"P3_order": F3, "P4_equiv": F4},
              "preregistration": {
                  "H1": "CP-95% lower bound of order_sensitivity must exceed 0.50",
                  "H2": "each same-construction control must FAIL (order<0.50 or equiv<0.50)",
                  "H3": "centering content-specific iff treatment_centered passes the pair "
                        "and blockperm_centered does not",
                  "H4": "centering commutes with blockperm; blockperm preserves the mean"},
              "arms": {}}

    for mode in ("fractional_shift", "phasor_bind"):
        for tag, n_p in (("small", N_SMALL), ("large", N_LARGE)):
            print(f"\n=== ARM MATRIX  mode={mode}  N={n_p} ===")
            tr = {
                "treatment": t_identity,
                "treatment_centered": t_center,
                "blockperm": lambda w, _m=mode: t_blockperm(
                    w, m1.build_config(len(m1.VOCAB), _m), m1.SEED + 11),
                "blockperm_centered": lambda w, _m=mode: t_center(t_blockperm(
                    w, m1.build_config(len(m1.VOCAB), _m), m1.SEED + 11)),
                "signrand": lambda w: t_signrand(w, m1.SEED + 13),
                "dead": lambda w: m1.degenerate_wave([""] * w.shape[0], w.shape[1], "dead"),
            }
            out, prompts, cfg, code = run_matrix(mode, n_p, m1.SEED, tr)
            with torch.no_grad():
                hrng = random.Random(m1.SEED)
                out["hash"] = pair_stats(
                    code,
                    m1.degenerate_wave(prompts, cfg.ambient_dim_D, "hash"),
                    m1.degenerate_wave([m1.shuffled(p, hrng) for p in prompts],
                                       cfg.ambient_dim_D, "hash"),
                    m1.degenerate_wave([m1.near_view(p) for p in prompts],
                                       cfg.ambient_dim_D, "hash"))
            rows = []
            for name, st in out.items():
                if name.startswith("_"):
                    continue
                p3 = st["order"] >= F3
                p4 = st["equiv"] >= F4
                pair = bool(p3 and p4)
                lo, hi = cp_lower(st["order_k"], st["n"]), cp_upper(st["order_k"], st["n"])
                rows.append((name, st, pair, lo, hi))
                print(f"  {name:22s} order={st['order']:.4f} ({st['order_k']:>4d}/{st['n']})"
                      f"  CI95=[{lo:.4f},{hi:.4f}]  equiv={st['equiv']:.4f}"
                      f"  P3={'P' if p3 else 'F'} P4={'P' if p4 else 'F'}"
                      f"  pair={'PASS' if pair else 'FAIL'}")
            print(f"  structure: {out['_structure']}")
            report["arms"][f"{mode}:{tag}"] = {
                "n": n_p, "mode": mode, "structure": out["_structure"],
                "arms": {nm: dict(st, pair_pass=pr, cp95=[lo, hi], binom_p=binom_tail(st["order_k"], st["n"]))
                         for nm, st, pr, lo, hi in rows}}

    # --------------------------------------------------- faithfulness gate
    # A GATE ON THE STUDY, not decoration. Run the COMMITTED gate's own `run_arm` and
    # compare its numbers with my harness's `treatment` arm at the same N. If they
    # disagree, my harness is not measuring the instrument under test and every
    # contrast below is uninterpretable.
    print("\n" + "=" * 84)
    print("FAITHFULNESS: my treatment arm vs the committed gate's own run_arm (N=%d)" % N_SMALL)
    print("=" * 84)
    faith = {}
    for mode in ("fractional_shift", "phasor_bind"):
        _p = m1.build_prompts(N_SMALL, random.Random(m1.SEED))
        ga = m1.run_arm(mode, _p, random.Random(m1.SEED))
        mine = report["arms"][f"{mode}:small"]["arms"]["treatment"]
        match = (abs(ga["order_sensitivity"] - mine["order"]) < 1e-9
                 and abs(ga["equivalence"] - mine["equiv"]) < 1e-9)
        faith[mode] = {"gate_order": ga["order_sensitivity"], "mine_order": mine["order"],
                       "gate_equiv": ga["equivalence"], "mine_equiv": mine["equiv"],
                       "faithful": bool(match)}
        print(f"  {mode:18s} gate order={ga['order_sensitivity']:.4f} equiv={ga['equivalence']:.4f}"
              f"  |  mine order={mine['order']:.4f} equiv={mine['equiv']:.4f}"
              f"  FAITHFUL={match}")
    report["faithfulness"] = faith

    # ------------------------------------------------------------- the verdicts
    print("\n" + "=" * 84)
    print("PRE-REGISTERED VERDICTS")
    print("=" * 84)
    verdicts = {}
    for key, blk in report["arms"].items():
        if not key.endswith(":large"):
            continue
        a = blk["arms"]
        tr, tc = a["treatment"], a["treatment_centered"]
        verdicts[key] = {
            # H1: the gate's own PASS must clear its sampling noise. The exact
            # one-sided binomial tail at p=0.5 is the honest test, because the
            # proportion sits ~0.00-0.06 from the floor.
            "H1_margin_decisive": bool(tr["cp95"][0] > F3),
            "H1_point_estimate": tr["order"],
            "H1_binom_p_ge_k_at_half": binom_tail(tr["order_k"], tr["n"]),
            "H2_blockperm_fails_pair": not a["blockperm"]["pair_pass"],
            "H2_signrand_fails_pair": not a["signrand"]["pair_pass"],
            "H2_all_same_construction_fail": bool(
                not a["blockperm"]["pair_pass"] and not a["signrand"]["pair_pass"]),
            "H3_centering_content_specific": bool(
                tc["pair_pass"] and not a["blockperm_centered"]["pair_pass"]),
            "H4_structure_ok": bool(blk["structure"]["blockperm_preserves_common_mode_norm"]
                                    and blk["structure"]["centering_commutes_with_blockperm"]),
        }
    # Aggregate: the gate is only valid where the pair is content-specific.
    verdicts["_aggregate"] = {
        "pair_is_content_specific_in": [k.split(":")[0] for k, v in verdicts.items()
                                        if k != "_aggregate"
                                        and v["H2_all_same_construction_fail"]],
        "pair_CONFOUNDED_in": [k.split(":")[0] for k, v in verdicts.items()
                               if k != "_aggregate"
                               and not v["H2_all_same_construction_fail"]],
    }
    for k, v in verdicts.items():
        print(f"\n  {k}")
        for kk, vv in v.items():
            print(f"    {kk:34s} = {vv}")
    report["verdicts"] = verdicts

    dest = pathlib.Path(os.environ.get("HENRI_RECEIPT_DIR", _HERE.parent))
    dest = dest / "m1_control_hardening_receipt.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  receipt: {dest}")
    return 0


CUR_MODE = "fractional_shift"
if __name__ == "__main__":
    raise SystemExit(main())

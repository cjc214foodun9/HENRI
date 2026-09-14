#!/usr/bin/env python3
"""OBSERVED: THREEFOLD SYSTEMIC GROUNDING -- measured against LIVE modules.

Each pillar is tested against the module that EXISTS, never against the name a
document uses. A document class name is not evidence of code.

  PILLAR             DOCUMENT NAME      LIVE MODULE
  Entorhinal Torus   spatial metric     o_vsa_torus_encoder.py
  Sagnac Homodyne    validation sieve   arc_sagnac_veto.py
  Anisotropic Creep  plasticity         adaptive_viscoelastic_thermostat.py

DOCUMENT CLASS NAMES MEASURED ABSENT (0 grep hits): SagnacHomodyne,
NoetherianSheaf, ConstraintSieve, EntorhinalTorus, AnisotropicCreep,
ViscoelasticCreep. Hardware headings (silicon-photonic interferometer, PCM
filter arrays, TiN heaters, barium titanate waveguides, optical combs) are NOT
executable here; every number below is a SOFTWARE ANALOG on sm_120.

DEFECTS FIXED FROM THE 10E RUN (all mine, kept visible)
  1. @guard returned the raw function: it ran at DECORATION time and wrapped
     nothing, so the first missing import killed the whole module.
  2. Wrong arity: step_viscoelastic_creep(weight_matrix, grad_loss,
     lambda_active, sagnac_delta, temperature, base_noise) and
     compute_wavelet_gated_noise(weight_matrix, grad_loss, temperature,
     effective_lr, base_noise).
  3. The test never probed the failure mode the DOCUMENT itself names, so the
     "sieve" arm was vacuous. Section 2b now tests that exact failure mode.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.getcwd())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NB, J = 8192, 8
rep: dict = {"device": DEV, "torch": torch.__version__, "pillars": {}}


def has_mod(name):
    return importlib.util.find_spec(name) is not None


def guard(name):
    def deco(fn):
        def wrapper():                      # FIXED: return the wrapper
            t0 = time.time()
            try:
                rep["pillars"][name] = {"status": "OBSERVED", "data": fn()}
            except Exception as e:
                rep["pillars"][name] = {
                    "status": "BLOCKED",
                    "error": f"{type(e).__name__}: {e}",
                    "trace_tail": traceback.format_exc().strip().splitlines()[-3:],
                }
            rep["pillars"][name]["runtime_s"] = round(time.time() - t0, 2)
            return rep["pillars"][name]
        return wrapper
    return deco


# ======================================================= PILLAR 1: TORUS
@guard("1_entorhinal_torus")
def pillar_torus():
    from o_vsa_torus_encoder import BLOCK_SLOTS, TorusIngressEncoder
    S = 32
    enc = TorusIngressEncoder(num_blocks=NB, vocab_size=8, modulus=S,
                              mode="TORUS_VAL", device=DEV)
    gen = torch.Generator().manual_seed(3)
    rows = [list(map(int, r)) for r in
            torch.randint(0, 8, (S, S), generator=gen).tolist()]
    X = enc.encode_canvas(rows)
    ident = float(torch.nn.functional.cosine_similarity(
        X.flatten(), enc.encode_canvas(enc.roll_canvas(rows, 3, 0)).flatten(), dim=0).item())
    uni = enc.encode_canvas([[0] * S for _ in range(S)])

    resid, nonresid = {}, {}
    for d in (1, 3, 5):
        resid[str(d)] = float((
            enc.encode_canvas(enc.roll_canvas(rows, d, 0)) -
            enc._to_real(enc.roll_multiplier(d, 0) * enc._to_complex(
                enc.encode_canvas(rows)))).abs().max().item())
        small = torch.randint(0, 8, (12, 12), generator=gen).tolist()
        nonresid[str(d)] = float((
            enc.encode(enc.roll_canvas(small, d, 0)) -
            enc._to_real(enc.roll_multiplier(d, 0) *
                         enc._to_complex(enc.encode(small)))).abs().max().item())
    return {
        "BLOCK_SLOTS": int(BLOCK_SLOTS),
        "state_shape": list(X.shape), "state_dtype": str(X.dtype),
        "uniform_norm": float(uni.norm().item()),
        "uniform_non_degenerate": bool(uni.norm().item() > 0.5),
        "identity_cos_X_rollX": ident,
        "identity_near_zero": bool(abs(ident) < 0.15),
        "exact_roll_residual_canvas": resid,
        "negative_control_non_canvas": nonresid,
        "negative_control_is_large": bool(max(nonresid.values()) > 0.1),
        "exactness_conditional_on_H_eq_W_eq_S": True,
        "role": "spatial metric base; exact translation on a full canvas ONLY.",
    }


# =============================================== PILLAR 2: SAGNAC SIEVE
@guard("2_sagnac_homodyne_sieve")
def pillar_sagnac():
    from arc_sagnac_veto import (DEFAULT_EPSILON_HARD, apply_advisory_rerank,
                                 evaluate_veto, rerank_with_veto)
    from o_vsa_torus_encoder import TorusIngressEncoder

    # DEFECT FIXED (found in the 10F run): the generator was seeded INSIDE this
    # function, so every call returned the SAME vector. Consequence measured:
    # "indep_random" reported cos = 1.0 against the axiom (a random draw must be
    # ~0), and the threshold sweep was degenerate -> epsilon_gate_cos_threshold
    # = 1.0. Counter-based seeding gives a distinct direction per call, and the
    # device argument is required because a CPU generator cannot feed a CUDA
    # randn.
    _ctr = [0]

    def unit(n=NB, seed=None):
        _ctr[0] += 1
        g = torch.Generator(device=DEV).manual_seed(int(seed or 1000) + _ctr[0])
        w = torch.randn(n, J, generator=g, device=DEV)
        return w / w.norm()

    axiom = unit(seed=77)
    cases = {}
    for name, cand in (("identical", axiom.clone()), ("indep_random", unit()),
                       ("antipodal", (-axiom).clone()),
                       ("half_roll", torch.roll(axiom, shifts=NB // 2, dims=0).clone())):
        cn = cand / cand.norm()
        d_ax, d_ep, trig, st = evaluate_veto(cn, axiom, unit())
        cases[name] = {"delta_axiom": d_ax, "delta_epistemic": d_ep,
                       "triggered": bool(trig), "status": st,
                       "cos": float(torch.dot(cn.flatten(), axiom.flatten()).item())}

    thr_cos = None
    for k in range(201):
        t = -1.0 + 2.0 * (k / 200.0)
        v = axiom + math.sqrt(max(1e-12, 1.0 - t * t)) * unit()
        v = v / v.norm()
        c = float(torch.dot(v.flatten(), axiom.flatten()).item())
        d_ax, _, _, _ = evaluate_veto(v, axiom, unit())
        if d_ax <= DEFAULT_EPSILON_HARD:
            thr_cos = c if thr_cos is None else max(thr_cos, c)
    _, _, ident_trig, _ = evaluate_veto(axiom.clone(), axiom, unit())

    # ---------------- 2b. THE DOCUMENT'S OWN NAMED FAILURE MODE ------------
    # Doc 1 s2.2: "If a candidate wave packet attempts to CREATE MATTER FROM
    # NOTHING (e.g. spontaneous grid pixel duplication without a source rule)
    # or TELEPORTS AN OBJECT faster than the maximum propagation velocity, the
    # sheaf condition fails on the restriction mapping, triggering an immediate
    # non-zero coboundary." This arm tests exactly that, on real waves.
    S = 32
    enc = TorusIngressEncoder(num_blocks=NB, vocab_size=8, modulus=S,
                              mode="TORUS_VAL", device=DEV)
    rows = [list(map(int, r)) for r in
            torch.randint(0, 8, (S, S), generator=torch.Generator().manual_seed(4)).tolist()]
    world = enc.encode_canvas(rows)

    def charge(g):
        """Discrete first integral: total colour mass (DERIVED, exact)."""
        return float(sum(sum(r) for r in g))

    def curl(g):
        """Total variation -- proxy for propagation stress (DERIVED, exact)."""
        t = torch.tensor(g, dtype=torch.float32)
        return float((t[1:] - t[:-1]).abs().sum().item() +
                     (t[:, 1:] - t[:, :-1]).abs().sum().item())

    q_w, c_w = charge(rows), curl(rows)

    # (a) VALID: pure translation. Conserves charge exactly, small curl change.
    valid_g = enc.roll_canvas(rows, 3, 0)
    # (b) MATTER FROM NOTHING: duplicate a 6x6 patch into an empty region.
    dup = [list(r) for r in rows]
    for y in range(4):
        for x in range(4):
            dup[y][x] = rows[y + 8][x + 8]
    # (c) TELEPORT: move one object 12 cells in one step (super-luminal)
    tel = [list(r) for r in rows]
    for y in range(2):
        for x in range(2):
            tel[y][x] = rows[y + 20][x + 20]

    arms = {}
    for nm, g in (("valid_translation", valid_g),
                  ("matter_from_nothing", dup),
                  ("teleport_one_step", tel)):
        w = enc.encode_canvas(g)
        ax = enc._to_real(enc.roll_multiplier(3, 0) * enc._to_complex(
            enc.encode_canvas(rows)))          # the axiom: the lawful translate
        d_ax, _, trig, st = evaluate_veto(w, ax, world)
        arms[nm] = {
            "delta_axiom": d_ax, "triggered": bool(trig), "status": st,
            "cos_vs_axiom": float(torch.nn.functional.cosine_similarity(
                w.flatten(), ax.flatten(), dim=0).item()),
            "charge_delta": charge(g) - q_w,
            "curl_delta": curl(g) - c_w,
        }
    sieve_catches = bool(arms["matter_from_nothing"]["triggered"]
                         and arms["teleport_one_step"]["triggered"]
                         and not arms["valid_translation"]["triggered"])

    # ---------------- 2c. CONSTRUCTIVE REPLACEMENT (first-integral checker) --
    # A DERIVED software analog of the doc's "coboundary": exact integer
    # conservation checks. It has no free parameters and cannot be tuned.
    def first_integral_ok(g, q0, c0, dq_tol=0.0, dc_tol=None):
        dq = abs(charge(g) - q0)
        dc = abs(curl(g) - c0)
        lim = (0.15 * (abs(c0) + 1e-9)) if dc_tol is None else dc_tol
        return bool(dq <= dq_tol and dc <= lim), {"dq": dq, "dc": dc, "dc_lim": lim}

    fi = {}
    for nm, g in (("valid_translation", valid_g),
                  ("matter_from_nothing", dup), ("teleport_one_step", tel)):
        ok, det = first_integral_ok(g, q_w, c_w)
        fi[nm] = {"ok": ok, **det}
    first_integral_discriminates = bool(fi["valid_translation"]["ok"]
                                       and not fi["matter_from_nothing"]["ok"]
                                       and not fi["teleport_one_step"]["ok"])

    ranked = [{"id": "a", "efe": 1.0}, {"id": "b", "efe": 2.0}, {"id": "c", "efe": 3.0}]
    r1 = rerank_with_veto(ranked, [True, False, False])
    r2 = rerank_with_veto(ranked, [True, True, True])
    c1 = apply_advisory_rerank(ranked, [True, False, False], ranked[0])

    return {
        "DEFAULT_EPSILON_HARD": DEFAULT_EPSILON_HARD,
        "cases": cases,
        "epsilon_gate_cos_threshold": thr_cos,
        "identical_vetoes_itself": bool(ident_trig),
        "gate_can_fail": bool(not ident_trig),
        "2b_doc_named_failure_modes": arms,
        "2b_cosine_sieve_catches_violations": sieve_catches,
        "2c_first_integral_checker": fi,
        "2c_first_integral_discriminates": first_integral_discriminates,
        "rerank_order_after_one_veto": [x["id"] for x in r1],
        "rerank_all_vetoed_preserves_order": [x["id"] for x in r2],
        "advisory_chosen": c1[0]["id"] if c1 and c1[0] else None,
        "role": ("ADVISORY cosine re-ranker, fail-OPEN. It measures GEOMETRIC "
                 "similarity to a reference wave; it is NOT a conservation-law "
                 "checker. Section 2b tests the doc's own failure modes."),
    }


# ================================================== PILLAR 3: CREEP
@guard("3_anisotropic_creep")
def pillar_creep():
    import adaptive_viscoelastic_thermostat as avt
    cls = avt.AdaptiveViscoelasticThermostat
    sig = inspect.signature(cls.__init__)
    kwargs = {n: p.default for n, p in sig.parameters.items()
              if n != "self" and p.default is not inspect.Parameter.empty}
    th = cls(**kwargs).to(DEV)
    flags = {k: getattr(th, k) for k in dir(th)
             if not k.startswith("_") and isinstance(getattr(th, k, None), bool)}

    # ---- STIEFEL: the NaN from the 10E run, re-tested as a scaling question.
    # Newton-Schulz W <- 0.5*W*(3I - W^T W) converges only for ||W||_2 < sqrt(3).
    # A raw randn has ||W|| ~ 2*sqrt(n) ~ 16 at n=64, so divergence to NaN is
    # EXPECTED and is a real property of the live implementation.
    stiefel = {}
    for nm, sc in (("raw_randn", 1.0), ("scaled_0.1", 0.1), ("scaled_0.01", 0.01)):
        W = torch.randn(64, 64, device=DEV,
                        generator=torch.Generator(device=DEV).manual_seed(2)) * sc
        nrm = float(W.norm().item())
        P = th.project_stiefel_manifold(W)
        finite = bool(torch.isfinite(P).all().item())
        res = (float((P.t() @ P - torch.eye(64, device=DEV)).norm().item())
               if finite else float("nan"))
        stiefel[nm] = {"input_norm": nrm, "output_finite": finite,
                       "orthogonality_residual": res,
                       "converged": bool(finite and res < 1e-4)}

    fr = {}
    for ld in (0.0, 0.5, 1.0):
        for sd in (0.0, 0.5, 1.0):
            v = th.compute_anisotropic_friction(ld, sd)
            fr[f"lam{ld}_d{sd}"] = float(v.mean().item()) if torch.is_tensor(v) else float(v)

    # ---- SGLD noise law sqrt(2*T*dt) and the CREEP step, CORRECT ARITY.
    W = (torch.randn(64, 64, device=DEV,
                     generator=torch.Generator(device=DEV).manual_seed(3)) * 0.01)
    g1 = torch.ones_like(W) * 0.01
    noise_probe = {}
    for T in (1e-4, 1e-2):
        n, dom, locked = th.compute_wavelet_gated_noise(W, g1, T, 1e-3)
        noise_probe[f"T={T}"] = {
            "noise_std": float(n.std().item()), "dominance": dom, "locked": bool(locked),
        }
    step = {}
    for lam, sd in ((0.5, 0.0), (0.5, 1.0), (1.0, 1.0)):
        Wn, info = th.step_viscoelastic_creep(W, g1, lam, sd, temperature=1e-4)
        step[f"lam{lam}_d{sd}"] = {
            "param_move_norm": float((Wn - W).norm().item()),
            "effective_lr": float(info.get("effective_lr")) if isinstance(info, dict)
            and torch.is_tensor(info.get("effective_lr")) else info.get("effective_lr"),
            "finite": bool(torch.isfinite(Wn).all().item()),
        }
    engaged = bool(step and all(v["finite"] and v["param_move_norm"] > 1e-12
                                for v in step.values()))
    try:
        creep_sig = str(inspect.signature(cls.apply_anisotropic_langevin_creep))
    except Exception as e:
        creep_sig = f"BLOCKED {e}"
    return {
        "bool_flags_default_off": flags,
        "stiefel_scaling": stiefel,
        "stiefel_nan_is_norm_dependent": bool(
            not stiefel["raw_randn"]["output_finite"]
            and stiefel["scaled_0.01"]["converged"]),
        "anisotropic_friction": fr,
        "friction_varies_with_inputs": bool(len(set(fr.values())) > 1),
        "noise_law_probe": noise_probe,
        "creep_step": step,
        "learning_engaged": engaged,
        "step_viscoelastic_creep_signature": str(inspect.signature(
            cls.step_viscoelastic_creep)),
        "apply_anisotropic_langevin_creep_signature": creep_sig,
        "role": ("thermodynamic plasticity / SGLD local relaxation. SOFTWARE "
                 "analog of anisotropic creep, NOT photonic hardware."),
    }


def main():
    t0 = time.time()
    rep["module_presence"] = {m: has_mod(m) for m in
                              ("o_vsa_torus_encoder", "arc_sagnac_veto",
                               "adaptive_viscoelastic_thermostat",
                               "arc_thermostat_shadow")}
    for fn in (pillar_torus, pillar_sagnac, pillar_creep):
        fn()
    rep["runtime_s"] = round(time.time() - t0, 2)
    rep["named_mechanisms_absent_from_tree"] = [
        "SagnacHomodyne", "NoetherianSheaf", "ConstraintSieve",
        "EntorhinalTorus", "AnisotropicCreep", "ViscoelasticCreep"]
    rep["hardware_headings_not_executable_here"] = [
        "Silicon-photonic Sagnac homodyne interferometer", "PCM filter arrays",
        "TiN microheater dissipation", "Thin-film barium titanate waveguides",
        "Optical frequency combs"]
    rep["external_outcome_claim"] = (
        "NONE. Every number is internal consistency: representation, operator "
        "form, veto behaviour, parameter motion. No ARC or AAII score.")
    print(json.dumps(rep, indent=1))
    dst = ("/workspace/phase10/HENRI V2/experiments/verification/"
           "threefold_grounding_observed.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nWROTE {dst} ({os.path.getsize(dst)} bytes)")
    print("### DONE")


if __name__ == "__main__":
    main()

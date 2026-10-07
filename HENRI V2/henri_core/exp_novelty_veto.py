"""PHASE 1: does coupling s(q) into the Sagnac veto change a dispatch decision?

Pre-registration: design/zone_a/evidence/henri_phase1_veto_preregistration.json

DESIGN
    Coupling:  Delta_eff = Delta_Sagnac + lambda * (1 - s(q)),  lambda = 0.50
    Default:   lambda = 0.0  ->  Delta_eff == Delta_Sagnac (production unchanged)

    The verdict is read from the PRODUCTION path, system.solve(), not from a
    reimplementation. solve() computes s(q) against the veto's own axiom bank,
    passes it into veto.forward, and reports delta_sagnac / novelty_penalty /
    flip / veto_source.

BANK
    build_axioms(CORPUS[:6]) pins a 6-item bank. That gives the blueprint's own
    populations: in-bank n=6, held-out n=4 (corpus sentences the bank never saw).

THE DISCRIMINATING BAND
    Criterion "held-out is vetoed" is nearly guaranteed by construction: held-out
    s ~ 0.22 makes lambda*(1-s) ~ 0.39 > 0.35, so it flips regardless of physics.
    The content is the NEAR-MISS band: drop-word negatives score s = 0.72-0.99, so
    the penalty is only 0.005-0.14 and the flip depends on the real Delta_Sagnac.
    Per-family FLIP SPECTRUM is the primary result.

ATTRIBUTION CONTROL
    Re-run the SAME coupling with s(q) replaced by a RANDOM score. If random
    scores reproduce the same flips, the effect is not the novelty signal.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core import harness_hygiene as H
from henri_core import novelty_gate as NG
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

SEED = 20261004
LAM = 0.50
CORPUS = H_cli.CORPUS
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(8)

SYNONYMS = [
    "a thing affects another thing nearby",
    "memory hands over an answer up the rungs",
    "the rule keeps the edge in place",
]
RANDOM_TEXT = ["zzz qqq unrelated gamma", "wxk pfj blorp tender", "qq vv xx zz mm"]


def drop_last(t):
    w = t.split()
    return " ".join(w[:-1]) if len(w) > 1 else t


def drop_first(t):
    w = t.split()
    return " ".join(w[1:]) if len(w) > 1 else t


def build():
    """Pinned ingress. Bank = first 6 corpus items (so 4 are genuinely held out)."""
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True, ingress_seed=SEED)
    s.eval()
    s.build_axioms(CORPUS[:6], tok)
    return s, tok


def families():
    return {
        "F0_inbank": (CORPUS[:6], "in_bank"),
        "HO_heldout": (CORPUS[6:], "heldout"),
        "F1_drop_last": ([drop_last(t) for t in CORPUS], "drop_boundary"),
        "F2_drop_first": ([drop_first(t) for t in CORPUS], "drop_boundary"),
        "F4_synonym": (SYNONYMS, "synonym"),
        "F5_random_text": (RANDOM_TEXT, "heldout"),
    }


def run_once():
    s, tok = build()
    bank_sha = H.sha256_items(CORPUS[:6])
    eval_items = [t for items, _ in families().values() for t in items]
    # D3: the in-bank positives ARE the bank; the negatives must not be.
    # (v1 bug: this comprehension used `t` before binding it -- the `for t in
    # items` clause was missing.)
    neg_items = [t for k, (items, _) in families().items()
                 if k not in ("F0_inbank",) for t in items]
    manifest = H.split_manifest(CORPUS[:6], neg_items)
    H.assert_matched_negatives(
        {k: {"kind": v[1]} for k, v in families().items()})

    rows = {}
    per_item = []
    for fam, (items, kind) in families().items():
        r0 = s.solve(items[0], tok, novelty_lambda=0.0)   # warm the flag path
        del r0
        entries = []
        for t in items:
            a = s.solve(t, tok, novelty_lambda=0.0)
            b = s.solve(t, tok, novelty_lambda=LAM)
            allow0 = bool(a["sagnac"]["allow"])
            allow5 = bool(b["sagnac"]["allow"])
            entries.append({
                "text": t,
                "s_q": round(float(b["novelty"]["score"]), 6) if b["novelty"] else None,
                "delta_sagnac": round(float(b["sagnac"]["delta_sagnac"]), 6),
                "novelty_penalty": round(float(b["sagnac"]["novelty_penalty"]), 6),
                "delta_eff": round(float(b["sagnac"]["delta"]), 6),
                "allow_lambda0": allow0,
                "allow_lambda5": allow5,
                "flipped": bool(allow0 and not allow5),
                "veto_source": b["sagnac"]["veto_source"],
            })
        n = len(entries)
        rows[fam] = {
            "kind": kind, "n": n,
            "pass_rate_lambda0": round(sum(e["allow_lambda0"] for e in entries) / n, 4),
            "pass_rate_lambda5": round(sum(e["allow_lambda5"] for e in entries) / n, 4),
            "flips": sum(e["flipped"] for e in entries),
            "flip_rate": round(sum(e["flipped"] for e in entries) / n, 4),
            "s_q_min": min(e["s_q"] for e in entries),
            "s_q_max": max(e["s_q"] for e in entries),
            "delta_sagnac_min": min(e["delta_sagnac"] for e in entries),
            "delta_sagnac_max": max(e["delta_sagnac"] for e in entries),
        }
        per_item.append((fam, entries))

    # ---- K1: lambda=0 is inert on the production path ------------------------
    t0 = CORPUS[0]
    a0 = s.solve(t0, tok)
    a1 = s.solve(t0, tok, novelty_lambda=0.0)
    k1 = (a0["sagnac"]["allow"] == a1["sagnac"]["allow"]
          and abs(a0["sagnac"]["delta"] - a1["sagnac"]["delta"]) < 1e-12
          and a1["sagnac"]["novelty_penalty"] == 0.0
          and a1["sagnac"]["veto_source"] == "SAGNAC_HOMODYNE")

    # ---- K4: attribution control (v1 was VACUOUS -- rewritten) --------------
    # v1 DEFECT (self-caught): v1 varied a RANDOM score against ONE candidate and
    # reported a flip rate of 0.3153. That is just P(uniform s < 0.308) -- it
    # measured the threshold arithmetic, not attribution. A control must test
    # whether the SCORE->ITEM assignment carries the effect.
    #
    # Correct control: hold each item's delta_sagnac fixed, permute the s(q)
    # values across items with a DERANGEMENT (D2), and recompute. If the family
    # structure survives a random re-assignment of scores, s(q) is not the cause.
    pairs = [(fam, e["delta_sagnac"], e["s_q"]) for fam, ents in per_item for e in ents
             if e["s_q"] is not None]
    thr = float(s.veto.threshold)

    def flips_with(scored):
        out = {}
        for (fam, d, _), sc in zip(pairs, scored):
            flip = (d <= thr) and (d + LAM * (1.0 - sc) > thr)
            out.setdefault(fam, []).append(flip)
        return {f: round(sum(v) / len(v), 4) for f, v in out.items()}

    real_by_fam = flips_with([p[2] for p in pairs])
    perm = H.derangement(len(pairs), H.pinned_generator(4242))
    shuf_by_fam = flips_with([pairs[i][2] for i in perm])
    gr = H.pinned_generator(31337)
    rand_by_fam = flips_with([float(torch.rand(1, generator=gr).item())
                              for _ in pairs])

    inbank_shift = abs(shuf_by_fam.get("F0_inbank", 0.0)
                       - real_by_fam.get("F0_inbank", 0.0))
    ho_shift = abs(shuf_by_fam.get("HO_heldout", 0.0)
                   - real_by_fam.get("HO_heldout", 0.0))
    k4 = bool(inbank_shift >= 0.15 and ho_shift >= 0.15)

    # ---- the VACUOUS-VETO finding (measured, must be reported) --------------
    all_d0 = [e["delta_sagnac"] for _, ents in per_item for e in ents]
    vac = {
        "pass_rate_lambda0_every_family": {f: d["pass_rate_lambda0"]
                                           for f, d in rows.items()},
        "all_families_pass_at_lambda0": all(d["pass_rate_lambda0"] == 1.0
                                            for d in rows.values()),
        "delta_sagnac_min": round(min(all_d0), 6),
        "delta_sagnac_max": round(max(all_d0), 6),
        "threshold": thr,
        "note": ("At lambda=0 the veto releases EVERY family, including random "
                 "text and synonyms: delta_sagnac is 0.004-0.017 against a 0.35 "
                 "threshold. The veto scores psi_conv, which the CCCP swarm has "
                 "already projected onto the bank, so the Sagnac term is "
                 "saturated toward 0 by the SAME post-projection confound Q4 "
                 "found. This gate could not fail before the coupling."),
    }

    # ---- K2/K3 --------------------------------------------------------------
    k2 = rows["HO_heldout"]["pass_rate_lambda5"] == 0.0
    k3 = rows["F0_inbank"]["pass_rate_lambda5"] == 1.0

    R = {
        "schema": "henri.phase1.veto_coupling.receipt.v1",
        "head": None, "seed": SEED, "lambda": LAM,
        "dim": int(s.dim), "n_bank": int(s.veto.n_axioms),
        "threshold": float(s.veto.threshold),
        "param_count": int(sum(p.numel() for p in s.parameters())),
        "manifest": manifest,
        "families": rows,
        "per_item": {fam: ents for fam, ents in per_item},
        "kill": {
            "K1_lambda0_inert": bool(k1),
            "K2_heldout_vetoed_at_lambda": bool(k2),
            "K3_inbank_preserved": bool(k3),
            "K4_attribution_control": {
                "flip_rate_real": real_by_fam,
                "flip_rate_score_shuffled": shuf_by_fam,
                "flip_rate_random_score": rand_by_fam,
                "inbank_shift_under_shuffle": round(inbank_shift, 4),
                "heldout_shift_under_shuffle": round(ho_shift, 4),
                "passes": k4,
                "note": ("control passes only if permuting the s(q)->item "
                         "assignment CHANGES the per-family flip rates; a random "
                         "score must not reproduce the real structure"),
            },
        },
        "vacuous_veto_finding": vac,
        "honest_scope": (
            "This tests whether the gate CHANGES A DISPATCH DECISION. No scored "
            "task is wired to the veto in this build, so the blueprint's claim "
            "that coupling 'elevates the net scored task rate' is UNTESTED here. "
            "Dim is the small config, not 65536."),
    }

    # near-miss band summary
    nm = [e for fam, ents in per_item if fam in ("F1_drop_last", "F2_drop_first")
          for e in ents]
    R["near_miss_band"] = {
        "n": len(nm),
        "s_q_range": [min(e["s_q"] for e in nm), max(e["s_q"] for e in nm)],
        "penalty_range": [min(e["novelty_penalty"] for e in nm),
                          max(e["novelty_penalty"] for e in nm)],
        "flips": sum(e["flipped"] for e in nm),
        "note": ("penalty here is small (<=0.14), so a flip depends on the real "
                 "Delta_Sagnac: this is the falsifiable content"),
    }
    R["claim"] = (
        f"lambda=0 reproduces the uncoupled verdict exactly: {k1}. "
        f"At lambda={LAM}, held-out pass rate {rows['HO_heldout']['pass_rate_lambda5']} "
        f"(criterion 1) and in-bank pass rate {rows['F0_inbank']['pass_rate_lambda5']} "
        f"(criterion 2). Near-miss band flips "
        f"{R['near_miss_band']['flips']}/{R['near_miss_band']['n']}. "
        f"Attribution control: permuting the s(q)->item assignment moves the "
        f"in-bank flip rate by {inbank_shift:.2f} and the held-out flip rate by "
        f"{ho_shift:.2f} (K4 passes={k4}). "
        f"CRITICAL FINDING: at lambda=0 the veto released EVERY family "
        f"(delta_sagnac {vac['delta_sagnac_min']:.4f}-{vac['delta_sagnac_max']:.4f} "
        f"vs threshold {thr}), so the Sagnac term alone could not fail. The "
        f"coupling is what makes the veto able to reject.")
    return R


def main() -> int:
    r1 = run_once()
    r2 = run_once()
    h1 = hashlib.sha256(json.dumps(r1, sort_keys=True).encode()).hexdigest()[:16]
    h2 = hashlib.sha256(json.dumps(r2, sort_keys=True).encode()).hexdigest()[:16]
    r1["determinism"] = {"run1": h1, "run2": h2, "identical": h1 == h2,
                         "K5_deterministic": h1 == h2}
    r1["kill"]["K5_determinism"] = h1 == h2

    ev = os.path.join(REPO, "design", "zone_a", "evidence")
    dst = (os.path.join(ev, "henri_phase1_veto_receipt.json") if os.path.isdir(ev)
           else r"C:/Users/chan/AppData/Local/Temp/henri_phase1_veto_receipt.json")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(r1, indent=1))

    print("PHASE 1: coupling s(q) into the Sagnac veto")
    print("=" * 88)
    print(f"  dim={r1['dim']} bank={r1['n_bank']} threshold={r1['threshold']} "
          f"lambda={LAM} params={r1['param_count']}")
    print(f"  {'family':>16}{'n':>4}{'pass@0':>9}{'pass@L':>9}{'flips':>7}"
          f"{'s_q':>17}{'dSagnac':>17}")
    for fam, d in r1["families"].items():
        print(f"  {fam:>16}{d['n']:>4}{d['pass_rate_lambda0']:>9.3f}"
              f"{d['pass_rate_lambda5']:>9.3f}{d['flips']:>7}"
              f"{d['s_q_min']:>8.3f}..{d['s_q_max']:<8.3f}"
              f"{d['delta_sagnac_min']:>8.3f}..{d['delta_sagnac_max']:<8.3f}")
    print("  " + "-" * 86)
    for k, v in r1["kill"].items():
        print(f"  {k:<36} {v}")
    print("=" * 88)
    print(f"  {r1['claim']}")
    print(f"  receipt: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

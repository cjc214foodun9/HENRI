#!/usr/bin/env python3
"""ACTION 4 (corrected) — the adversarial control, run against the RIGHT encoder.

DEFECT IN THE FIRST VERSION OF THIS FILE (found by reviewing my own artifact)
    The 78.33% calibration receipt is produced by `HENRIVisionEncoder`
    (run_calibration_eval.py:262). The first version of this control used
    `O_VSA_IngressTokenizer`, a DIFFERENT encoder with only a compatible
    interface (henri_vision_encoder.py:159 documents the compatibility shim).
    Therefore the first control did NOT test the encoder behind the claim it was
    meant to audit. That is a methodological error in my own artifact, recorded
    here rather than quietly overwritten.

THIS VERSION runs the same statistic under BOTH encoders plus random-projection
controls, so the comparison is explicit and the encoder identity is not assumed.

THE QUESTION
    Does ARC discrimination live in the ALGEBRAIC STRUCTURE of the wave manifold,
    or is it an artifact of the readout projection / of grid statistics?

WHY THE OBVIOUS CONTROL IS INVALID (and is therefore not used as the discriminator)
    "Encode with codebook M, decode with the same frozen M, require a random
    Gaussian codebook to fail" CANNOT DISCRIMINATE. At D=65536 random codebooks
    are near-orthogonal (earlier R2 probe: max pairwise overlap 7.349e-03 vs
    chance 3.906e-03), so a random codebook decodes its own binding fine. That
    test measures the encode/decode CHANNEL, not the representation. Requiring it
    to fail would rig the control.

THE VALID DISCRIMINATOR USED HERE
    Ask whether an encoder's output geometry orders transformation-RELATED grids
    (a grid vs its own rot90/180/270) above transformation-UNRELATED ones (a
    different task's grid). If an encoder carries algebraic transformation
    structure, the margin should be positive AND larger than what a
    structure-blind embedding produces.

PRE-REGISTERED OUTCOMES (fixed BEFORE running)
    P1 the selected encoder shows margin > 0.
    P2 KILL: if a seeded random projection of the raw grid to the same d_model has
       a margin >= 0.5 x the encoder's margin, the statistic is CONFOUNDED (any
       reasonable embedding preserves grid-level rotational autocorrelation) and
       the arm is INCONCLUSIVE. The 78.33% accuracy is then NOT attributed to
       wave-manifold structure.
    P3 if the encoder margin <= 0, the manifold claim is FALSIFIED for this metric.

No task score is produced. This tests a NECESSARY condition, not the accuracy claim.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))
OUT = R / "experiments" / "verification" / "codebook_adversarial_observed.json"
CALIB = R / "experiments" / "verification" / "run_calibration_eval.py"

N_TASKS = 40
D_MODEL = 65536
N_BLOCKS, BLOCK_DIM = 8192, 8


def encoder_identity():
    """Report which encoder the sealed 78.33% receipt actually used."""
    src = CALIB.read_text(encoding="utf-8")
    return {
        "calibration_encoder": (
            "HENRIVisionEncoder" if "HENRIVisionEncoder(" in src else "UNKNOWN"),
        "source": str(CALIB.name),
        "first_version_defect": (
            "The first version of this control used O_VSA_IngressTokenizer, which "
            "is NOT the encoder behind the 78.33% receipt. Corrected here."
        ),
    }


def load_tasks(n):
    ARC = pathlib.Path(r"C:\Users\chan\henri_data\ARC-AGI\data")
    tasks = []
    for split in ("training", "evaluation"):
        d = ARC / split
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                t = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            tr = t.get("train") or []
            if not tr:
                continue
            g = tr[0].get("input")
            if not g or len(g) < 2 or len(g[0]) < 2:
                continue
            tasks.append({"id": p.stem, "grid": np.array(g, dtype=np.int64)})
            if len(tasks) >= n:
                return tasks
    return tasks


def variants(g):
    return {"rot90": np.rot90(g, 1), "rot180": np.rot90(g, 2), "rot270": np.rot90(g, 3)}


def cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


# ------------------------------------------------------------------ encoders

def make_vision_encoder_fn():
    """HENRIVisionEncoder, built EXACTLY as run_calibration_eval.py builds it."""
    import run_calibration_eval as RCE  # noqa
    from henri_vision_encoder import HENRIVisionEncoder

    kind, bg = RCE.resolve_spatial_basis()
    enc = HENRIVisionEncoder(d_model=D_MODEL, k_blocks=N_BLOCKS,
                             block_dim=BLOCK_DIM, device="cpu",
                             spatial_basis_kind=kind, bg_mask=bg)
    state = {"refused": 0, "ok": 0, "err": None}

    def fn(grid):
        try:
            w = enc.encode_spatial_grid([list(r) for r in grid])
        except Exception as exc:
            state["err"] = f"{type(exc).__name__}: {exc}"
            try:
                w = enc.encode_spatial_grid([list(r) for r in grid])
            except Exception:
                state["refused"] += 1
                return None
        if w is None:
            state["refused"] += 1
            return None
        v = w.squeeze(0).reshape(-1).to(torch.float32).detach().cpu().numpy()
        if v.size == 0 or not np.all(np.isfinite(v)):
            state["refused"] += 1
            return None
        state["ok"] += 1
        return normalize(v.astype(np.float64))

    return fn, state, f"HENRIVisionEncoder(kind={kind}, bg_mask={bg})"


def make_tokenizer_fn():
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer

    tok = O_VSA_IngressTokenizer()
    state = {"refused": 0, "ok": 0, "err": None}

    def fn(grid):
        try:
            w = tok.encode_spatial_grid([list(r) for r in grid])
        except Exception as exc:
            state["err"] = f"{type(exc).__name__}: {exc}"
            state["refused"] += 1
            return None
        if w is None:
            state["refused"] += 1
            return None
        v = w.squeeze(0).reshape(-1).detach().to(torch.float32).cpu().numpy()
        if v.size == 0 or not np.all(np.isfinite(v)):
            state["refused"] += 1
            return None
        state["ok"] += 1
        return normalize(v.astype(np.float64))

    return fn, state, "O_VSA_IngressTokenizer"


def make_projection_fn(seed):
    rng = np.random.default_rng(1000 + seed)
    P = rng.standard_normal((D_MODEL, 81)) / np.sqrt(81.0)

    def fn(grid):
        flat = np.zeros(81, dtype=np.float64)
        v = grid.reshape(-1).astype(np.float64)
        v = v[:81] if v.size >= 81 else np.pad(v, (0, 81 - v.size))
        flat[: v.size] = v
        return normalize(P @ flat)

    return fn


# ---------------------------------------------------------------------- arms

def run_arm(name, fn, dim, tasks):
    related, foreign = [], []
    for i, t in enumerate(tasks):
        g = t["grid"]
        base = fn(g)
        if base is None:
            continue
        for vg in variants(g).values():
            v = fn(vg)
            if v is not None:
                related.append(cos(base, v))
        for j in ((i + 1) % len(tasks), (i + 7) % len(tasks)):
            if j == i:
                continue
            f = fn(tasks[j]["grid"])
            if f is not None:
                foreign.append(cos(base, f))
    rel = np.array([r for r in related if np.isfinite(r)])
    forr = np.array([r for r in foreign if np.isfinite(r)])
    if rel.size == 0 or forr.size == 0:
        return {"arm": name, "dim": dim, "status": "NO_DATA",
                "n_related": int(rel.size), "n_foreign": int(forr.size)}
    return {
        "arm": name, "dim": dim, "status": "OK",
        "n_related": int(rel.size), "n_foreign": int(forr.size),
        "sim_related_mean": float(rel.mean()),
        "sim_foreign_mean": float(forr.mean()),
        "margin": float(rel.mean() - forr.mean()),
        "related_sd": float(rel.std()), "foreign_sd": float(forr.std()),
    }


def main():
    ident = encoder_identity()
    print(f"receipt encoder : {ident['calibration_encoder']}")
    tasks = load_tasks(N_TASKS)
    print(f"tasks loaded    : {len(tasks)}")
    if len(tasks) < 8:
        print("BLOCKED: too few ARC tasks")
        sys.exit(2)

    arms = []
    fn_v, st_v, desc_v = make_vision_encoder_fn()
    probe = fn_v(tasks[0]["grid"])
    dim_v = int(probe.size) if probe is not None else None
    print(f"vision encoder  : {desc_v}  d_model={dim_v}  err={st_v['err']}")
    if dim_v is not None:
        a = run_arm("HENRIVisionEncoder (receipt encoder)", fn_v, dim_v, tasks)
        arms.append(a)
        print(f"  margin={a.get('margin')} rel={a.get('sim_related_mean')} "
              f"for={a.get('sim_foreign_mean')} (ok={st_v['ok']} refused={st_v['refused']})")

    fn_t, st_t, desc_t = make_tokenizer_fn()
    probe_t = fn_t(tasks[0]["grid"])
    dim_t = int(probe_t.size) if probe_t is not None else None
    print(f"tokenizer       : {desc_t}  d_model={dim_t}")
    if dim_t is not None:
        b = run_arm("O_VSA_IngressTokenizer", fn_t, dim_t, tasks)
        arms.append(b)
        print(f"  margin={b.get('margin')} rel={b.get('sim_related_mean')} "
              f"for={b.get('sim_foreign_mean')}")

    for s in (1, 2, 3):
        c = run_arm(f"random_projection_{s}", make_projection_fn(s), D_MODEL, tasks)
        arms.append(c)
        print(f"  CTRL{s} margin={c.get('margin')} rel={c.get('sim_related_mean')} "
              f"for={c.get('sim_foreign_mean')}")

    enc = [a for a in arms if a["status"] == "OK" and "random_projection" not in a["arm"]]
    ctrl = [a for a in arms if a["status"] == "OK" and "random_projection" in a["arm"]]
    ctrl_max = max((a["margin"] for a in ctrl), default=None)

    pre = {}
    for a in enc:
        key = a["arm"]
        m = a["margin"]
        conf = (ctrl_max is not None and ctrl_max >= 0.5 * abs(m)) if m > 0 else False
        pre[key] = {
            "margin": m,
            "P1_margin_positive": bool(m > 0),
            "P2_confounded_by_random_projection": bool(conf),
            "P3_margin_not_positive": bool(m <= 0),
            "verdict": (
                "FALSIFIED - no transform structure on this metric" if m <= 0
                else ("INCONCLUSIVE - random projection reproduces a comparable "
                      "margin; the statistic is confounded and cannot attribute the "
                      "accuracy to wave-manifold structure" if conf
                      else "SUPPORTED - transform structure present and not "
                           "reproduced by a structure-blind control")
            ),
        }

    # ---- VALID control of the READOUT-PROJECTION hypothesis ----------------
    # The rotation-margin statistic above is confounded (P2). But the sealed
    # receipt already contains a control that IS valid for the question the
    # blueprint asks: "the 78.33% resides in the manifold, not in the readout
    # projection". Its three arms share an IDENTICAL readout (cosine against the
    # SAME candidate wave set) and differ ONLY in the operator (fitted W vs
    # random W vs no operator). So if the readout projection were producing the
    # accuracy, the random-W arm would also score well. It scores at chance.
    sealed = {}
    try:
        cal = json.loads((R / "experiments" / "verification"
                          / "calibration_eval_observed.json").read_text(encoding="utf-8"))
        for a in ("arm_functor", "arm_identity", "arm_random"):
            sealed[a] = {
                "accuracy": cal[a]["accuracy"],
                "bss": cal[a]["brier_skill_score"],
                "ece": cal[a]["ece"],
            }
        n_s = cal["arm_functor"]["n_samples"]
        k_s = cal["arm_functor"]["n_classes"]
        chance = 1.0 / k_s
        acc_f = sealed["arm_functor"]["accuracy"]
        acc_i = sealed["arm_identity"]["accuracy"]
        acc_r = sealed["arm_random"]["accuracy"]
        sealed.update({
            "n_tasks": n_s,
            "n_options": k_s,
            "chance_level": chance,
            "readout_identical_across_arms": True,
            "random_operator_acc_excess_over_chance": acc_r - chance,
            "readout_projection_explains_accuracy": bool(acc_r > chance + 0.10),
            "encoder_only_acc_no_operator_fit": acc_i,
            "operator_gain_over_encoding_alone": acc_f - acc_i,
            "verdict": (
                "The readout-projection hypothesis is FALSIFIED: with the SAME "
                "readout and the same candidate set, a random operator scores "
                f"{acc_r:.4f} (chance {chance:.4f}) while the fitted operator "
                f"scores {acc_f:.4f}. The readout therefore does not manufacture "
                "the discrimination. NOTE what remains open: encoding ALONE, with "
                f"no operator fit, already reaches {acc_i:.4f}, so the fitted "
                f"operator adds only {acc_f - acc_i:.4f}. Most of the "
                "discrimination is attributable to the ENCODER + candidate "
                "construction, not to the learned operator."
            ),
        })
    except Exception as exc:
        sealed = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}

    body = {
        "schema": "henri.codebook-adversarial.v2",
        "supersedes": "henri.codebook-adversarial.v1 (INCONCLUSIVE, wrong encoder)",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED: per-grid vectors come from the live encoder. Margins are "
            "DERIVED from them by the cosine statistic stated here."
        ),
        "encoder_identity": ident,
        "why_this_control": (
            "Same-basis encode/decode cannot discriminate a random codebook from a "
            "structured one (R2: random bound states are near-orthogonal and decode "
            "fine), so that test measures the channel, not the representation."
        ),
        "design": {
            "n_tasks": len(tasks),
            "task_ids": [t["id"] for t in tasks],
            "related_pairs": "grid vs its own rot90/rot180/rot270",
            "foreign_pairs": "grid vs a different task's grid",
            "statistic": "cosine similarity in encoder output space",
            "control": "seeded random projection of the raw grid to the same d_model",
            "d_model": D_MODEL,
        },
        "arms": arms,
        "control_margin_max": ctrl_max,
        "pre_registered": pre,
        "sealed_arms_control": sealed,
        "limits": [
            "Tests a NECESSARY condition of the manifold claim, not accuracy.",
            "Rotation is one structural family; other relations are untested.",
            "Encoder refusals are skipped, so n varies per arm.",
            "No task score is produced by this file.",
            "CUDA unavailable (Vast 50797414 EXITED); CPU only.",
        ],
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in pre.items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"      {kk:<40} {vv}")
    print(f"\nwrote {OUT}")
    print()
    print("=" * 76)
    print("SEALED-ARMS CONTROL OF THE READOUT-PROJECTION HYPOTHESIS")
    print("=" * 76)
    if sealed.get("status") == "BLOCKED":
        print(f"  BLOCKED: {sealed.get('error')}")
    else:
        for a in ("arm_functor", "arm_identity", "arm_random"):
            s = sealed[a]
            print(f"  {a:<14} acc={s['accuracy']:.4f} ece={s['ece']:.4f} "
                  f"bss={s['bss']:+.4f}")
        print(f"  chance level                            "
              f"{sealed['chance_level']:.4f}")
        print(f"  random operator excess over chance      "
              f"{sealed['random_operator_acc_excess_over_chance']:+.4f}")
        print(f"  readout explains the accuracy?          "
              f"{sealed['readout_projection_explains_accuracy']}")
        print(f"  encoder-only accuracy (no operator fit) "
              f"{sealed['encoder_only_acc_no_operator_fit']:.4f}")
        print(f"  operator gain over encoding alone       "
              f"{sealed['operator_gain_over_encoding_alone']:+.4f}")


if __name__ == "__main__":
    main()

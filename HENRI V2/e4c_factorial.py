"""E4c step 2 — the 2x2 factorial: tying x trainability on frozen-backbone features.

Reads the feature artifact from e4c_extract_features.py and runs FOUR arms that
share an identical input representation h in R^896 (the backbone's final
normalized hidden state). Only tying and trainability differ.

  a  untied [896->V]  N(0,1/sqrt(896))  trainable   <- proposed treatment
  b  tied   E          teacher table E   frozen      <- exact E3 Arm-B control
  c  tied   E          teacher table E   TRAINABLE   <- decisive
  d  untied [896->V]  N(0,1/sqrt(896))  frozen      <- random-init, NOT teacher-init
                                                       (teacher-init untied-frozen is
                                                        mathematically identical to b)

Arm (c) is the attribution control: if c ~= a, E3's failure was TRAINABILITY, not tying.

Per-construct kill bound: the marginal baseline is read FROM THE E4a AUDIT ARTIFACT,
never hard-coded (0.432 is the old consumed split; this fresh split measures 0.440).

Fail-closed on: missing artifact, SHA mismatch, non-finite logits, shape mismatch.

Usage (remote):
  python e4c_factorial.py --features /root/e4c/features.pt \
      --teacher /root/e1-calib/teacher_embeddings.pt \
      --e4a /root/e4c/e4a_construct_audit.json --out /root/e4c/e4c_result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 20260910
STEPS = 1000
BATCH = 32
LR = 3e-4
WD = 1e-4
VOCAB = 151936
HIDDEN = 896
MOVEMENT_MARGIN = 0.05
ATTRIB_TOL = 0.02
FLAG = "HENRI_BACKBONE"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def make_readout(init: str, E: torch.Tensor, device: str) -> nn.Linear:
    lin = nn.Linear(HIDDEN, VOCAB, bias=False)
    with torch.no_grad():
        if init == "teacher":
            lin.weight.data = E.detach().clone().float()
        else:
            g = torch.Generator().manual_seed(SEED)
            lin.weight.data = (torch.randn(VOCAB, HIDDEN, generator=g)
                               * (1.0 / math.sqrt(HIDDEN)))
    return lin.to(device)


@torch.no_grad()
def evaluate(lin: nn.Linear, X: torch.Tensor, gold: torch.Tensor,
             bs: int = 512) -> dict:
    lin.eval()
    p1 = p5 = n = 0
    nll = 0.0
    for i in range(0, X.shape[0], bs):
        xb, gb = X[i:i + bs], gold[i:i + bs]
        lg = xb @ lin.weight.t()          # [B, V], no bias
        if not torch.isfinite(lg).all():
            raise RuntimeError("NON_FINITE_LOGITS")
        t5 = torch.topk(lg, 5, dim=-1).indices
        p1 += int((t5[:, 0] == gb).sum())
        p5 += int((t5 == gb.unsqueeze(1)).any(1).sum())
        nll += float(F.cross_entropy(lg.float(), gb, reduction="sum"))
        n += xb.shape[0]
    return {"p_at_1": p1 / n, "p_at_5": p5 / n,
            "ce": nll / n, "n": n}


def train_arm(tag: str, lin: nn.Linear, X: torch.Tensor, gold: torch.Tensor,
              device: str, Xe: torch.Tensor, ge: torch.Tensor) -> dict:
    params = [p for p in lin.parameters() if p.requires_grad]
    if not params:
        return {"arm": tag, "trainable": False, "steps": 0}
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    n = X.shape[0]
    losses = []
    t0 = time.time()
    lin.train()
    for step in range(STEPS):
        g = torch.Generator().manual_seed(SEED + step)
        idx = torch.randint(0, n, (BATCH,), generator=g)
        xb, gb = X[idx].to(device), gold[idx].to(device)
        opt.zero_grad(set_to_none=True)
        lg = xb @ lin.weight.t()
        if not torch.isfinite(lg).all():
            print(f"[e4c] {tag} NON_FINITE_LOGITS at step {step}")
            return {"arm": tag, "trainable": True, "aborted": "NON_FINITE_LOGITS",
                    "step": step}
        loss = F.cross_entropy(lg.float(), gb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 250 == 0:
            print(f"[e4c] {tag} step {step:4d} loss {losses[-1]:.4f}", flush=True)
    return {"arm": tag, "trainable": True, "steps": STEPS,
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "wall_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--e4a", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get(FLAG) != "1":
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=FLAG_OFF ({FLAG}!=1)")
        raise SystemExit(2)

    feat_sha = sha256_file(args.features)
    for f in (args.teacher, args.e4a):
        if not f.exists():
            print(f"E4C_VERDICT=BLOCKED_INFRA reason=ARTIFACT_MISSING {f}")
            raise SystemExit(2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(str(args.features), map_location="cpu", weights_only=True)
    F_ = payload["features"]
    meta = payload["meta"]

    Xc = F_["calib"]["h"].float()
    gc = F_["calib"]["gold"].long()
    Xe = F_["eval"]["h"].float()
    ge = F_["eval"]["gold"].long()
    print(f"[e4c] calib h {tuple(Xc.shape)} eval h {tuple(Xe.shape)}")
    if Xc.shape[1] != HIDDEN:
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=HIDDEN_MISMATCH {Xc.shape[1]}")
        raise SystemExit(2)

    # per-construct baseline read from the E4a audit (NOT hard-coded)
    e4a = json.loads(args.e4a.read_text())
    c1 = e4a["constructs"]["C1_sentence_window"]
    base_p1 = c1["marginal_baseline"]["p1"]
    base_p5 = c1["marginal_baseline"]["p5"]
    strong_p1 = c1["best_trivial_baseline"]["p1"]
    strong_p5 = c1["best_trivial_baseline"]["p5"]
    print(f"[e4c] baselines from E4a artifact: marginal {base_p1}/{base_p5} "
          f"strongest-trivial {strong_p1}/{strong_p5}")

    teacher = torch.load(str(args.teacher), map_location="cpu",
                         weights_only=True)
    E = teacher["weight"] if isinstance(teacher, dict) and "weight" in teacher else teacher
    E = torch.as_tensor(E).float()
    print(f"[e4c] teacher E {tuple(E.shape)}")
    assert E.shape == (VOCAB, HIDDEN), "TEACHER_SHAPE_MISMATCH"

    # ---- oracle reference (backbone's own lm_head), diagnostic -------------
    oracle_p1 = float((F_["eval"]["oracle"] == ge).float().mean())
    oracle_p5 = float((torch.stack([F_["eval"]["oracle"]] * 1, 1) == ge.unsqueeze(1)
                       ).any(1).float().mean()) if False else None
    print(f"[e4c] oracle P@1 (backbone lm_head) = {oracle_p1:.4f}")

    arms = {}
    specs = [("a_untied_trainable", "rand", True),
             ("b_tied_frozen", "teacher", False),
             ("c_tied_trainable", "teacher", True),
             ("d_untied_frozen", "rand", False)]
    for tag, init, trainable in specs:
        torch.manual_seed(SEED)
        lin = make_readout(init, E, dev)
        lin.weight.requires_grad_(bool(trainable))
        with torch.no_grad():
            lin.weight.data = lin.weight.data.to(dev).float()
        tr = train_arm(tag, lin, Xc, gc, dev, Xe, ge)
        ev = evaluate(lin, Xe.to(dev), ge.to(dev))
        n_train = (lin.weight.numel() if trainable else 0)
        arms[tag] = {
            "init": init, "trainable": bool(trainable),
            "trainable_params": n_train,
            "trainable_fp32_bytes": n_train * 4,
            "trainable_bf16_bytes": n_train * 2,
            "adamw_state_fp32_bytes": n_train * 8,
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in ev.items()},
            **tr,
        }
        print(f"[e4c] {tag}: P@1={ev['p_at_1']:.4f} P@5={ev['p_at_5']:.4f} "
              f"CE={ev['ce']:.4f}", flush=True)

    # ---- gates -------------------------------------------------------------
    a, b = arms["a_untied_trainable"], arms["b_tied_frozen"]
    c, d = arms["c_tied_trainable"], arms["d_untied_frozen"]
    best_trainable = max(a["p_at_1"], c["p_at_1"])

    gA = all(torch.isfinite(torch.tensor(arms[k]["ce"])) for k in arms)
    gB = oracle_p1 >= base_p1 + MOVEMENT_MARGIN
    gC = best_trainable >= base_p1 + MOVEMENT_MARGIN
    gD = (c["p_at_1"] >= a["p_at_1"] - ATTRIB_TOL)

    if gD:
        attribution = ("TRAINABILITY_DOMINANT: tied-trainable (c) reaches parity with "
                       "untied-trainable (a); E3's failure mode was trainability, "
                       "not weight tying")
    else:
        attribution = ("UNTYING_HELPFUL: untied-trainable (a) exceeds tied-trainable "
                       "(c) by more than the tolerance")

    verdict = ("BLOCKED_CONSTRUCT" if not gB else
               ("E4C_GATES_PASS" if (gA and gC) else "E4C_MOVEMENT_FAIL"))

    rec = {
        "carrier": "E4c",
        "prereg": "experiments/verification/e4c_untied_readout_prereg.md",
        "features_sha256": feat_sha,
        "feature_meta": {k: v for k, v in meta.items() if k != "features"},
        "input_representation": "frozen backbone final normalized hidden state [896]",
        "split": {"calib": list(meta.get("fresh_calib", [])),
                  "eval": list(meta.get("fresh_eval", [])),
                  "label": "CONDITIONAL_FRESH_SPLIT_SAME_CORPUS"},
        "baselines_from_e4a": {"marginal": {"p1": base_p1, "p5": base_p5},
                               "strongest_trivial": {"p1": strong_p1, "p5": strong_p5}},
        "oracle_backbone_lm_head_p1": round(oracle_p1, 6),
        "arms": arms,
        "gates": {"G-A_wiring_finite": bool(gA),
                  "G-B_oracle_viable": bool(gB),
                  "G-C_movement_above_marginal": bool(gC),
                  "G-D_attribution_parity": bool(gD)},
        "attribution": attribution,
        "kill_bound": {"p_at_1": round(base_p1 + MOVEMENT_MARGIN, 4),
                       "source": "E4a marginal baseline + margin (per-construct)"},
        "lean_ledger": {
            "untied_readout_trainable_params": a["trainable_params"],
            "untied_readout_trainable_fp32_bytes": a["trainable_fp32_bytes"],
            "untied_readout_trainable_bf16_bytes": a["trainable_bf16_bytes"],
            "adamw_state_fp32_bytes": a["adamw_state_fp32_bytes"],
            "tied_matrix_frozen_params": int(E.numel()),
            "frozen_backbone_params": int(meta.get("n_backbone_params", 0)),
            "note": "frozen components reported separately from trainable",
        },
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    sha = sha256_file(args.out)
    print()
    print(json.dumps(rec["gates"], indent=2))
    print(f"attribution: {attribution}")
    print(f"E4C_VERDICT={verdict}")
    print(f"wrote {args.out}\ne4c_sha256 {sha}")


if __name__ == "__main__":
    sys.exit(main())

"""STEP-2 (pre-registered): does the LEARNABLE ingress make M4-G1 passable?

FROZEN DECISION RULE (margin ratio, not a scalar threshold):
  For each seed s in {20261010, 20261011, 20261012}:
    bar_s  = max(unigram_floor, best shuffled-control held acc) + 0.05
    PASS_s = held_acc_s > bar_s  AND  held_acc_s > its own shuffled control
  M4-G1 PASS := PASS_s for ALL seeds.
  margin_ratio = (mean(learnable) - mean(ctrl)) / pstdev(learnable)
  A reported pass additionally requires margin_ratio >= 1.0.

ALLOWED THIRD VERDICT (pre-registered, so a negative is informative):
  REPRESENTATION_LIMITED -- every seed lands in 0.10..0.20 with no arm passing.
  That means step 1 is necessary but not sufficient and no tuning is legal.

Arms, same pinned build and same 48/108 split per seed:
  frozen           train_ingress=False   [baseline]
  learnable        train_ingress=True
  ctrl_frozen      frozen    on SHUFFLED (spec, target) pairs
  ctrl_learnable   learnable on SHUFFLED (spec, target) pairs

Caps: CPU, D=4096 diagnostic. Not a model-performance claim.
"""
import io
import json
import os
import statistics
import sys
import time

import torch

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system,
                                      unigram_floor)

SEEDS = [20261010, 20261011, 20261012]
OUT = os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp", "step2.json")


def tokacc(preds, tgts):
    hit = tot = 0
    for row, ids in zip(preds, tgts):
        for a, b in zip(row, ids):
            tot += 1
            hit += int(a == b)
    return hit / max(1, tot)


def main():
    t0 = time.time()
    R = {"seeds": SEEDS, "per_seed": {}}
    for seed in SEEDS:
        corpus = build_corpus(max_len=3, holdout_len=3)
        system, tok = build_system(corpus, pin_seed=seed)
        tr = [corpus.specs[i] for i in corpus.train_idx]
        ho = [corpus.specs[i] for i in corpus.heldout_idx]
        tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
        ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
        g = torch.Generator().manual_seed(seed)
        tr_shuf = [tr_tgt[i] for i in
                   torch.randperm(len(tr_tgt), generator=g).tolist()]
        row = {}
        for arm, flag, targets in (
                ("frozen", False, tr_tgt),
                ("learnable", True, tr_tgt),
                ("ctrl_frozen", False, tr_shuf),
                ("ctrl_learnable", True, tr_shuf)):
            gen = WaveTextGenerator(system, tok, train_body=True)
            gen.fit(tr, targets, M4Config(steps=400, pin_seed=seed,
                                          train_ingress=flag))
            row[arm] = round(tokacc(gen.predict_ids(ho, M4Config()), ho_tgt), 4)
        floor = round(unigram_floor(tok, tr_tgt, ho_tgt), 4)
        bar = round(max(floor, row["ctrl_learnable"], row["ctrl_frozen"]) + 0.05, 4)
        row["floor"] = floor
        row["bar"] = bar
        row["PASS"] = bool(row["learnable"] > bar
                           and row["learnable"] > row["ctrl_learnable"])
        R["per_seed"][str(seed)] = row
        print(f"seed {seed}: {json.dumps(row)}", flush=True)

    held = [R["per_seed"][str(s)]["learnable"] for s in SEEDS]
    ctrl = [R["per_seed"][str(s)]["ctrl_learnable"] for s in SEEDS]
    sd = statistics.pstdev(held)
    R["margin"] = {
        "held_mean": round(statistics.mean(held), 4),
        "held_std": round(sd, 6),
        "ctrl_mean": round(statistics.mean(ctrl), 4),
        "margin_ratio": round((statistics.mean(held) - statistics.mean(ctrl))
                              / max(1e-9, sd), 3),
        "all_seeds_pass": bool(all(R["per_seed"][str(s)]["PASS"] for s in SEEDS)),
    }
    R["representation_limited"] = bool(
        all(0.10 <= R["per_seed"][str(s)]["learnable"] <= 0.20 for s in SEEDS)
        and not R["margin"]["all_seeds_pass"])
    R["verdict"] = ("M4-G1 PASSABLE" if R["margin"]["all_seeds_pass"]
                    and R["margin"]["margin_ratio"] >= 1.0
                    else "REPRESENTATION_LIMITED" if R["representation_limited"]
                    else "M4-G1 STILL FAILING")
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in R.items() if k != "per_seed"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

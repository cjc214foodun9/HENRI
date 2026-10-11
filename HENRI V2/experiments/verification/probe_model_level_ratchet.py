"""probe_model_level_ratchet.py -- the MODEL-LEVEL ratchet (thesis item 2).

THESIS ITEM 2. "Model-level ratchet. Requires a trained readout. The store-level ratchet
measured here is a prerequisite, not the result."

A STORE-LEVEL ratchet was already measured (henri_ndl_ratchet_observed.json): the store
accumulates distinct verifiable (program, input) pairs, real CUM 138 vs shuffled 12, margin
11.5, 6/6 checks. That says consolidation accumulates STRUCTURE. It says nothing about
whether a TRAINED READOUT gets better. This probe measures that.

WHAT "RATCHET" MEANS AT THE MODEL LEVEL
    Fit a readout on the accumulated experience, then measure competence on a HELD-OUT set
    that is NEVER ingested. A ratchet means competence is MONOTONE NON-DECREASING across
    consolidation cycles and does not decay. A metric that merely fluctuates is not one.

THE PRE-REGISTERED DECISION RULE IS ALREADY IN THE TREE, SO I REUSE IT
    henri_core/exp_m4_readout_capacity.py froze this rule before its own run:
        bar = max(unigram_floor, shuffled-arm held accuracy) + 0.05
        constrained arm beats bar with its OWN shuffled-label control intact
            -> the readout was the blocker; a constrained head is the fix
        constrained arm does not beat its shuffled-label control
            -> the frozen wave carries insufficient information; learnable ingress is a
               prerequisite and NO readout fix can pass
    I do not invent a new bar. I apply that one.

ARMS
    CEILING  ridge fitted on the ENTIRE waking set at cycle 0. Shows what is reachable at
             all; a ratchet that never reaches a fitting arm is not a ratchet.
    R        cycles ingest real wake pairs; readout refitted each cycle on the accumulated
             store. The candidate ratchet.
    S        identical cycle count and identical ingestion VOLUME, but each ingested spec is
             paired with a SHUFFLED trace. Own shuffled-label control, as the rule requires.
    N        no ingestion at all; readout refitted each cycle on the synthetic replay pool
             only. Separates "refitting moves" from "experience moves".

MY OWN TRAPS, GUARDED
    1. TRACE-COLLISION NO-OP. If a shuffled pairing happens to reproduce the true trace the
       shuffle is a NO-OP. Idempotent ops collide (C, CC, CCC ALL give "1111"). The probe
       measures and reports the DISTINGUISHABLE FRACTION and refuses a verdict when low.
    2. LABEL ALPHABET. The readout decodes into the union output alphabet of the WHOLE
       corpus, fixed at design time. Held-out traces are therefore representable and the
       measurement cannot be vacuous by an alphabet gap.
    3. MONOTONE-BY-CONSTRUCTION. The metric is an ARGMAX accuracy, not a union, so
       monotonicity is a MEASURED result and not a property of the accumulator.

SCOPE, STATED PLAINLY
    Frozen encoder (train_body=False), ridge readout, single seed, CPU, small N. This is a
    readout-competence measurement on a synthetic digit-program substrate. It is NOT an
    AAII claim, NOT a capability claim, and NOT a CUDA result. No bar is moved.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
V = REPO / "HENRI V2"
if str(V) not in sys.path:
    sys.path.insert(0, str(V))

import henri_core.m4_generative as m                        # noqa: E402
import henri_core.henri_candidate_bank as cb                # noqa: E402

EV = REPO / "design" / "zone_a" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

PIN = 20261010
MAX_LEN = 3
HOLDOUT_LEN = 3
LAM = 10.0
SEED = 20261011

CYCLES = 6
WAKE_PER_CYCLE = 8
REPLAY_PER_CYCLE = 24
RATCHET_GROWTH_MIN = 0.05
CONTROL_MARGIN = 0.05
MIN_DISTINGUISHABLE = 0.60

R = {
    "schema": "henri.model.ratchet.observed.v1",
    "evidence_class": "OBSERVED",
    "what": "model-level ratchet: readout competence across consolidation cycles",
    "reuses_bar_from": "henri_core/exp_m4_readout_capacity.py (frozen before its own run)",
    "decision_rule": ("bar = max(unigram_floor, shuffled-arm held acc) + 0.05; the "
                      "constrained arm must beat that bar with its own shuffled control "
                      "intact, else the frozen wave carries insufficient information"),
    "preregistered": {
        "cycles": CYCLES, "wake_per_cycle": WAKE_PER_CYCLE,
        "replay_per_cycle": REPLAY_PER_CYCLE, "ridge_lambda": LAM, "seed": SEED,
        "ratchet_growth_min": RATCHET_GROWTH_MIN,
        "control_margin": CONTROL_MARGIN,
        "min_distinguishable_fraction": MIN_DISTINGUISHABLE,
    },
    "non_claims": [
        "frozen encoder (train_body=False); the wave is NOT trained here",
        "ridge readout on a synthetic digit-program substrate, single seed, CPU",
        "NOT an AAII claim, NOT a capability claim, NOT a CUDA result",
        "the store-level ratchet is a prerequisite and is already measured separately",
    ],
}

print("=" * 78)
print("STEP 1 -- corpus, frozen waves (cached per spec), fixed output alphabet")
print("=" * 78)
corpus = m.build_corpus(max_len=MAX_LEN, holdout_len=HOLDOUT_LEN)
tr, ho = corpus.train_idx, corpus.heldout_idx
system, tok = m.build_system(corpus, pin_seed=PIN)
gen = m.WaveTextGenerator(system, tok, train_body=False)

specs_all = list(corpus.specs)
traces_all = list(corpus.traces)
W = cb.build_waves(gen, specs_all)              # [N, D] L2-normalised
isfin = bool(torch.isfinite(W).all())
D = int(W.shape[1])

# fixed output alphabet over the WHOLE corpus, so held-out traces are representable
alphabet = sorted(set(traces_all))
cat = {t: i for i, t in enumerate(alphabet)}
Y = torch.zeros(len(alphabet), len(alphabet), dtype=torch.float64)
Y[torch.arange(len(alphabet)), torch.arange(len(alphabet))] = 1.0

R["setup"] = {"n_pairs": len(specs_all), "n_train": len(tr), "n_heldout": len(ho),
              "wave_dim": D, "waves_finite": isfin,
              "n_output_categories": len(alphabet), "pin": PIN, "lambda": LAM}
print(f"  pairs {len(specs_all)}  train {len(tr)}  held {len(ho)}  D {D}  "
      f"finite {isfin}  categories {len(alphabet)}")

# held-out (spec, trace) pairs -- NEVER ingested
ho_specs = [specs_all[i] for i in ho]
ho_idx = list(ho)
X_ho = W[ho_idx]
print(f"  held-out {X_ho.shape[0]} pairs, never ingested")


def held_metrics(store_rows):
    """Fit dual ridge on store_rows, evaluate on the held-out set. Returns exact + token acc."""
    if len(store_rows) < 4:
        return None
    idx = torch.tensor(store_rows, dtype=torch.long)
    Xs = W[idx].to(torch.float64)
    ys = torch.tensor([cat[traces_all[i]] for i in store_rows], dtype=torch.long)
    Ys = Y[ys]
    # bias column
    Xa = torch.cat([Xs, torch.ones(Xs.shape[0], 1, dtype=torch.float64)], dim=1)
    K = Xa @ Xa.T
    A = K + LAM * torch.eye(K.shape[0], dtype=torch.float64)
    alpha = torch.linalg.solve(A, Ys)
    Xh = torch.cat([X_ho.to(torch.float64),
                    torch.ones(X_ho.shape[0], 1, dtype=torch.float64)], dim=1)
    scores = (Xh @ Xa.T) @ alpha
    pred = scores.argmax(dim=1)
    pred_traces = [alphabet[int(c)] for c in pred]
    gold_traces = [traces_all[i] for i in ho]
    exact = sum(1 for a, b in zip(pred_traces, gold_traces) if a == b) / len(gold_traces)
    th = tt = 0
    for a, b in zip(pred_traces, gold_traces):
        for x, yv in zip(a, b):
            tt += 1
            th += int(x == yv)
    return {"exact_match": float(exact), "token_acc": float(th / max(tt, 1)),
            "n_store": len(store_rows), "n_pred_distinct": len(set(pred_traces))}


print()
print("=" * 78)
print("STEP 2 -- CEILING arm: ridge on the ENTIRE waking set at cycle 0")
print("=" * 78)
ceiling = held_metrics(list(tr))
R["arm_ceiling"] = ceiling
print(f"  CEILING  store {ceiling['n_store']}  exact {ceiling['exact_match']:.4f}  "
      f"token {ceiling['token_acc']:.4f}  distinct preds {ceiling['n_pred_distinct']}")

# unigram floor on the held set
from collections import Counter
uni = Counter(t for t in traces_all)
floor = sum(uni[traces_all[i]] for i in ho) / max(len(ho), 1) / len(alphabet)
uni_tok = 0.0
cnt = Counter()
for t in traces_all:
    cnt.update(t)
tot = sum(cnt.values())
uni_tok = sum(cnt[ch] for i in ho for ch in traces_all[i]) / max(sum(len(traces_all[i]) for i in ho), 1)
R["unigram_floor"] = {"trace_floor": float(floor), "token_floor": float(uni_tok)}

rng = torch.Generator().manual_seed(SEED)


def run_arm(name, shuffle_pairs: bool):
    """Ingest WAKE_PER_CYCLE pairs per cycle; refit each cycle on the accumulated store."""
    store: list[int] = []
    rows, distinguish = [], []
    for c in range(CYCLES):
        batch = [int(x) for x in torch.randperm(len(tr), generator=rng)[:WAKE_PER_CYCLE]]
        src = [tr[b] for b in batch]
        if shuffle_pairs and len(src) > 1:
            perm = torch.randperm(len(src), generator=rng).tolist()
            # count positions whose trace actually changed -- a no-op shuffle is vacuous
            changed = sum(1 for k in range(len(src))
                          if traces_all[src[perm[k]]] != traces_all[src[k]])
            distinguish.append(changed / len(src))
            # ingest the SPEC of row k with the TRACE of row perm[k]
            for k in range(len(src)):
                probe_row = src[k]
                # emulate a mismatched pair by storing a NEGATIVE marker is not possible in a
                # categorical readout; instead store the row whose trace is the shuffled one,
                # i.e. use perm[k]'s trace with k's wave. Encode as a synthetic index pair.
                store.append((-1, src[k], src[perm[k]]))
        else:
            store.extend(("ok", r, r) for r in src)
        rows.append({"cycle": c, "store": len(store),
                     **({"n/a": True} if False else {})})
    return store, rows, distinguish


def evaluate_arm(name, pairs):
    """pairs = list of (kind, wave_row, trace_row). Fit on wave_row -> trace_row."""
    rows = []
    for c in range(CYCLES):
        upto = (c + 1) * WAKE_PER_CYCLE
        seen = pairs[:upto]
        if len(seen) < 4:
            rows.append({"cycle": c, "n_store": len(seen), "exact_match": None})
            continue
        w_rows = [p[1] for p in seen]
        t_rows = [p[2] for p in seen]
        idx = torch.tensor(w_rows, dtype=torch.long)
        Xs = W[idx].to(torch.float64)
        ys = torch.tensor([cat[traces_all[t]] for t in t_rows], dtype=torch.long)
        Ys = Y[ys]
        Xa = torch.cat([Xs, torch.ones(Xs.shape[0], 1, dtype=torch.float64)], dim=1)
        A = (Xa @ Xa.T) + LAM * torch.eye(Xa.shape[0], dtype=torch.float64)
        alpha = torch.linalg.solve(A, Ys)
        Xh = torch.cat([X_ho.to(torch.float64),
                        torch.ones(X_ho.shape[0], 1, dtype=torch.float64)], dim=1)
        pred = ((Xh @ Xa.T) @ alpha).argmax(dim=1)
        pred_traces = [alphabet[int(cc)] for cc in pred]
        gold_traces = [traces_all[i] for i in ho]
        ex = sum(1 for a, b in zip(pred_traces, gold_traces) if a == b) / len(gold_traces)
        th = tt = 0
        for a, b in zip(pred_traces, gold_traces):
            for x, yv in zip(a, b):
                tt += 1
                th += int(x == yv)
        rows.append({"cycle": c, "n_store": len(seen), "exact_match": float(ex),
                     "token_acc": float(th / max(tt, 1))})
    return rows


print()
print("=" * 78)
print(f"STEP 3 -- arms over {CYCLES} cycles")
print("=" * 78)
# ARM R: real pairs
pairs_R = []
dist_R = []
for c in range(CYCLES):
    batch = [int(x) for x in torch.randperm(len(tr), generator=rng)[:WAKE_PER_CYCLE]]
    pairs_R.extend(("ok", r, r) for r in batch)
rows_R = evaluate_arm("R", pairs_R)

# ARM S: shuffled trace pairing, identical wave rows and identical VOLUME
pairs_S = []
dist_S = []
for c in range(CYCLES):
    batch = [int(x) for x in torch.randperm(len(tr), generator=rng)[:WAKE_PER_CYCLE]]
    src = [tr[b] for b in batch]
    perm = torch.randperm(len(src), generator=rng).tolist()
    changed = sum(1 for k in range(len(src))
                  if traces_all[src[perm[k]]] != traces_all[src[k]])
    dist_S.append(changed / len(src))
    pairs_S.extend(("shuf", src[k], src[perm[k]]) for k in range(len(src)))
rows_S = evaluate_arm("S", pairs_S)

# ARM N: refit on the synthetic replay pool only (same volume, NO real experience)
pairs_N = []
for c in range(CYCLES):
    pool = m.programs(MAX_LEN)
    pick = [int(x) for x in torch.randperm(len(pool), generator=rng)[:WAKE_PER_CYCLE]]
    for p in pick:
        try:
            row = traces_all.index(m.run_program(pool[p], m.INPUTS[0]))
            pairs_N.append(("syn", row, row))
        except ValueError:
            continue
rows_N = evaluate_arm("N", pairs_N) if len(pairs_N) >= 4 else [
    {"cycle": c, "n_store": len(pairs_N), "exact_match": None} for c in range(CYCLES)]

for tag, rows in (("R", rows_R), ("S", rows_S), ("N", rows_N)):
    print(f"  ARM {tag}:")
    for r in rows:
        ex = r.get("exact_match")
        print(f"    cyc {r['cycle']}  store {r['n_store']:3d}  "
              f"exact {('%.4f' % ex) if ex is not None else '  n/a'}")

R["arm_R_real"] = rows_R
R["arm_S_shuffled"] = rows_S
R["arm_N_synthetic_only"] = rows_N
R["shuffle_distinguishable_fraction"] = {
    "min": float(min(dist_S)) if dist_S else None,
    "mean": float(np.mean(dist_S)) if dist_S else None,
}

print()
print("=" * 78)
print("STEP 4 -- verdict against the pre-registered rule")
print("=" * 78)
ex_R = [r["exact_match"] for r in rows_R if r.get("exact_match") is not None]
ex_S = [r["exact_match"] for r in rows_S if r.get("exact_match") is not None]
mono = all(ex_R[i] <= ex_R[i + 1] + 1e-12 for i in range(len(ex_R) - 1)) if len(ex_R) > 1 else False
growth = (ex_R[-1] - ex_R[0]) if len(ex_R) > 1 else 0.0
shuf_final = ex_S[-1] if ex_S else float("nan")
bar = max(R["unigram_floor"]["trace_floor"], shuf_final) + CONTROL_MARGIN
dist_ok = (R["shuffle_distinguishable_fraction"]["min"] or 0.0) >= MIN_DISTINGUISHABLE

R["ratchet"] = {"exact_match_series_R": ex_R, "monotone": bool(mono),
                "growth_cycle0_to_final": float(growth)}
R["decision"] = {"bar": float(bar),
                 "shuffled_final": float(shuf_final) if ex_S else None,
                 "ceiling_exact": ceiling["exact_match"],
                 "ceiling_token": ceiling["token_acc"]}
checks = {
    "waves_finite": isfin,
    "control_is_not_a_noop": bool(dist_ok),
    "arm_R_growth_meets_min": bool(growth >= RATCHET_GROWTH_MIN),
    "arm_R_is_monotone": bool(mono),
    "arm_R_beats_shuffled_control": bool(len(ex_R) and len(ex_S) and ex_R[-1] > shuf_final),
    "arm_R_beats_preregistered_bar": bool(len(ex_R) and ex_R[-1] >= bar),
    "arm_R_reaches_ceiling_region": bool(ex_R and ex_R[-1] >= 0.5 * ceiling["exact_match"]),
}
R["checks"] = checks
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"  growth {growth:+.4f} (min {RATCHET_GROWTH_MIN})  bar {bar:.4f}  "
      f"R final {ex_R[-1] if ex_R else float('nan'):.4f}  ceiling {ceiling['exact_match']:.4f}")

if not dist_ok:
    R["verdict"] = "MODEL_RATCHET_INCONCLUSIVE_CONTROL_VACUOUS"
elif not checks["arm_R_beats_preregistered_bar"]:
    R["verdict"] = ("MODEL_RATCHET_NOT_CONFIRMED_READOUT_WEAK__FROZEN_WAVE_INSUFFICIENT")
elif all(checks.values()):
    R["verdict"] = "MODEL_RATCHET_CONFIRMED_AT_READOUT_LEVEL"
else:
    R["verdict"] = "MODEL_RATCHET_PARTIAL_SEE_CHECKS"
R["residual"] = ("The frozen wave carries weak signal and the wave-space geometry is "
                 "population-invariant (wave_readout docstring ASSUMED block), so this "
                 "readout measures whether ACCUMULATED EXPERIENCE improves decoding. It "
                 "does not measure an omni-modal capability.")
print(f"  VERDICT: {R['verdict']}")

out = EV / "henri_model_ratchet_observed.json"
out.write_text(json.dumps(R, indent=2, default=str) + "\n", encoding="utf-8")
print(f"\n  receipt -> {out.relative_to(REPO)}")
print("  OBSERVED by execution. No bar moved. No CUDA. No AAII claim.")

"""build_candidate_bank.py -- GENERATE the HENRI candidate bank (G-SJ3a/b unblock).

WHY THIS IS THE HIGHEST-LEVERAGE UNBLOCK
    G-SJ3a reported BLOCKED_BANK_ABSENT because no component turned waves into the
    candidate bank the selector consumes. The tree had a selector and a corpus
    generator, but no producer for the bank. This is that producer.

BANK-SCALE IS NOT COSMETIC -- the 10x-lift clause is scale-dependent
    G_SJ3A_BAR = 0.0450, control 0.0046, lift required 10x.
    A 10x lift over ROW-UNIFORM chance 1/K requires top1 >= 10/K:
        K=28   10/K = 0.3571 -> UNREACHABLE at a 0.0450 bar
        K=217  10/K = 0.0461 -> marginally above the bar (the ruling's own tension)
        K=300  10/K = 0.0333 -> reachable
    Measured control on a 217-column bank is 1/217 = 0.00461 ~= the recorded 0.0046.
    So the CALIBRATION bank has ~217 columns and ~1170 rows (5.4x redundancy).
    A verdict read off a 28-column bank is a FIXTURE MISMATCH, not a result.

TWO IDENTITIES, both produced and both reported
    full   one row per training (program, input) pair -- row chance inflated by
           trace multiplicity.
    dedup  one row per DISTINCT trace -- containment caps any within-group rule.

ARTIFACT
    Writes the bank to disk as a torch .pt OUTSIDE the git index (binaries stay
    untracked). NOTE: torch.save is a pickle/zip, NOT .npz -- the name says .pt so the
    artifact label matches the format.
    records its sha256 in the receipt, and writes the measurement receipt.

NO TRAINING. Frozen wave encoder. CPU. No model-performance claim.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

V = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, V)

import henri_core.m4_generative as m          # noqa: E402
import henri_core.henri_candidate_bank as cb  # noqa: E402
import henri_core.wave_readout as wr          # noqa: E402

PIN = 20261010


def inputs_for(n: int) -> list[str]:
    """Distinct 4-digit inputs. The corpus keys on (program, input) so more inputs
    means more ROWS but not more distinct traces once the op-semantics saturate --
    which is exactly what produces the bank multiplicity the selector consumes."""
    return [f"{1000 + i}" for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--n-inputs", type=int, default=30)
    ap.add_argument("--holdout-len", type=int, default=4)
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--out-dir", default=os.path.join(V, "..", "design", "zone_a",
                                                     "evidence", "banks"))
    a = ap.parse_args()

    t0 = time.time()
    R = {"pin": PIN, "training": "NONE (frozen waves)", "dtype": "cpu",
         "calibration": {"bar": wr.G_SJ3A_BAR, "control": wr.G_SJ3A_RANDOM_CONTROL,
                         "lift_required": wr.G_SJ3A_LIFT_REQUIRED,
                         "index_bar": wr.G_SJ3A_INDEX_BAR}}

    print("=" * 80)
    print(f"STEP 1  corpus  max_len={a.max_len} inputs={a.n_inputs} holdout_len={a.holdout_len}")
    print("=" * 80)
    corpus = m.build_corpus(max_len=a.max_len, holdout_len=a.holdout_len,
                            inputs=inputs_for(a.n_inputs))
    tr, ho = corpus.train_idx, corpus.heldout_idx
    R["corpus"] = {"n_programs": len(corpus.programs), "n_train_rows": len(tr),
                   "n_heldout_rows": len(ho), "n_inputs": a.n_inputs}
    print(f"  programs={len(corpus.programs)} train={len(tr)} heldout={len(ho)} "
          f"({time.time()-t0:.1f}s)")

    print()
    print("=" * 80)
    print("STEP 2  system (pinned construction)")
    print("=" * 80)
    system, tok = m.build_system(corpus, pin_seed=PIN)
    gen = m.WaveTextGenerator(system, tok, train_body=False)
    print(f"  built, pin_seed={PIN} ({time.time()-t0:.1f}s)")

    # cost control: subsample QUERIES only; the BANK keeps every train row, because
    # bank multiplicity is part of what is being measured.
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_idx = list(range(len(ho)))
    if a.n_queries and a.n_queries < len(ho_idx):
        gsel = torch.Generator().manual_seed(PIN)
        perm = torch.randperm(len(ho_idx), generator=gsel).tolist()
        ho_idx = sorted(perm[:a.n_queries])
    ho_sel = [ho[i] for i in ho_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho_sel]
    R["sampling"] = {"n_bank_rows": len(tr), "n_queries": len(ho_sel),
                     "queries_subsampled": len(ho_sel) < len(ho)}

    print()
    print("=" * 80)
    print(f"STEP 3  frozen waves  ({len(tr)} bank rows + {len(ho_sel)} queries)")
    print("=" * 80)
    tw = time.time()
    Ztr = cb.build_waves(gen, [corpus.specs[i] for i in tr])
    print(f"  bank waves {tuple(Ztr.shape)} ({time.time()-tw:.1f}s)")
    Zho = cb.build_waves(gen, [corpus.specs[i] for i in ho_sel])
    print(f"  query waves {tuple(Zho.shape)} ({time.time()-tw:.1f}s total)")
    R["wave_dim"] = int(Ztr.shape[1])
    R["waves_finite"] = bool(torch.isfinite(Ztr).all() and torch.isfinite(Zho).all())

    sim = Zho @ Ztr.T
    if torch.is_complex(sim):
        sim = sim.real
    gold = cb.gold_sets(tr_tgt, ho_tgt)
    R["gold_nonempty_frac"] = sum(1 for g in gold if g) / len(gold)

    print()
    print("=" * 80)
    print("STEP 4  BANK IDENTITY")
    print("=" * 80)
    rep_full = cb.containment_report(sim, gold)
    cols = cb.dedup_columns(tr_tgt)
    sim_d, gold_d = cb.restrict(sim, cols, gold)
    rep_dedup = cb.containment_report(sim_d, gold_d)
    R["bank_full"] = rep_full
    R["bank_dedup"] = rep_dedup
    R["n_distinct_traces"] = len(cols)
    R["redundancy"] = round(rep_full["bank_columns"] / max(len(cols), 1), 4)
    for tag, r in (("full", rep_full), ("dedup", rep_dedup)):
        print(f"  {tag:>5} cols={r['bank_columns']:>5} containment={r['containment']:.4f} "
              f"ctrl={r['size_matched_control']:.4f} grp={r['mean_group_size']:.2f} "
              f"prec={r['within_group_gold_precision']:.5f} rowchance={r['row_uniform_chance']:.5f}")
    print(f"  redundancy (rows/distinct) = {R['redundancy']}")

    print()
    print("=" * 80)
    print("STEP 5  SELECTOR on the DEDUP bank")
    print("=" * 80)
    ch = cb.frequency_channel(tr_tgt, cols, sim_d.shape)
    noop = cb.channel_noop_check(ch)
    R["channel_noop_check"] = noop
    print(f"  channel uniq={noop['n_unique_values']} const={noop['frac_constant']} "
          f"is_noop={noop['is_noop']}")
    arms = {}
    for mode, c in (("argmax", None), ("uniform", None), ("two_stage", ch)):
        arms[mode] = cb.score_selector(sim_d, gold_d, c, mode, tr_tgt)
        q = arms[mode]
        print(f"  {mode:>9} top1={q['top1']:.4f} perm={q['top1_under_permutation']:.4f} "
              f"drift={q['drift']:.4f}")
    R["selector_dedup"] = arms

    print()
    print("=" * 80)
    print("STEP 6  G-SJ3a  (SAME-BANK control; cross-bank comparison refused)")
    print("=" * 80)
    best = max(arms.items(), key=lambda kv: kv[1]["top1"])
    same_bank = rep_dedup["row_uniform_chance"]
    gate = wr.gate_sj3a(best[1]["top1"], rand_control=same_bank, slot_r=None)
    R["gate_sj3a"] = gate
    R["bank_matches_calibration"] = abs(len(cols) - 217) <= 40
    for k, v in gate.items():
        print(f"  {k:28} {v}")
    print(f"  same-bank row-uniform control : {same_bank:.5f}  (calibration 0.0046)")
    print(f"  K={len(cols)}  10/K={(10.0/max(len(cols),1)):.4f}  bar={wr.G_SJ3A_BAR}")

    R["verdict"] = ("G_SJ3A_PASSED" if gate.get("passes")
                    else "G_SJ3A_FAILED_" + best[0].upper())
    print(f"\n  VERDICT: {R['verdict']}  (best={best[0]} top1={best[1]['top1']:.4f})")

    # ---- ARTIFACT: persist the bank OUTSIDE the git index ----
    os.makedirs(a.out_dir, exist_ok=True)
    bank_path = os.path.join(a.out_dir, "candidate_bank_dedup.pt")
    torch.save({"Ztr": Ztr, "Zho": Zho, "sim_d": sim_d, "cols": torch.tensor(cols),
                "gold_d": [sorted(g) for g in gold_d],
                # persisted for the leak/slot_r probe. tr_tgt stays a PYTHON list:
                # held-out programs differ in length, so token ids are RAGGED and
                # torch.tensor(ragged) raises -- that was my rc=1 cause.
                "ch_dedup": ch, "tr_tgt": [list(map(int, t)) for t in tr_tgt]},
               bank_path)
    h = hashlib.sha256(open(bank_path, "rb").read()).hexdigest()
    R["bank_artifact"] = {"path": os.path.abspath(bank_path),
                          "sha256": h, "bytes": os.path.getsize(bank_path),
                          "tracked_in_git": False}
    print(f"\n[bank]    {bank_path}")
    print(f"[sha256]  {h[:32]}...  ({os.path.getsize(bank_path)} bytes)")

    # PATH FIX (self-caught): V is already "…/HENRI V2", so ONE ".." reaches the repo
    # root. The first version used V/"../.." and climbed past it, producing
    # FileNotFoundError on the receipt write after the bank had already been written.
    rp = os.path.join(V, "..", "design", "zone_a", "evidence",
                      "henri_candidate_bank_observed.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(f"[receipt] {os.path.abspath(rp)}")
    print(f"[elapsed] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

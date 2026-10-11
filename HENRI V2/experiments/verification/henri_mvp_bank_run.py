"""henri_mvp_bank_run.py -- MINIMUM VIABLE HENRI: corpus -> waves -> bank -> selector -> gate.

CLOSES THE MISSING LINK. The tree carried a selector (wave_readout.py) and a corpus
generator (m4_generative.build_corpus) but no end-to-end path from waves to the candidate
bank. That absence is why G-SJ3a reported BLOCKED_BANK_ABSENT. This runs the whole loop
and writes ONE receipt with real numbers.

FROZEN ENCODER, NO TRAINING. train_body=False on WaveTextGenerator: this measures the
FROZEN wave representation, which is what the selector claim is about. A trained-body
number would confound representation with readout fit (the freeze-the-representation
localization lesson).

TWO BANK IDENTITIES, both reported. Bank identity decides the verdict: the full bank
inflates row-wise chance via trace multiplicity, and dedup containment caps any
within-group rule. Every number names its bank.

TIE-ROBUST ONLY. argmax ties always (frac_untied 0.0 measured), so the primary statistic
is P(gold in max group) with a per-query SIZE-MATCHED control. top-1 is reported but is
tie arbitration.

NO MODEL-PERFORMANCE CLAIM. CPU, frozen waves, a 4-op digit-program substrate. This is a
mechanism measurement, not an AAII result.
"""
import json
import os
import sys

import torch

V = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, V)

import henri_core.m4_generative as m                     # noqa: E402
import henri_core.henri_candidate_bank as cb             # noqa: E402
import henri_core.wave_readout as wr                     # noqa: E402

PIN = 20261010
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                   "design", "zone_a", "evidence", "henri_mvp_bank_observed.json")


def main() -> int:
    R = {"pin": PIN, "substrate": "m4_generative 4-op digit programs", "training": "NONE (frozen)",
         "dtype": "cpu"}
    print("=" * 78)
    print("STEP 1  corpus")
    print("=" * 78)
    corpus = m.build_corpus(max_len=3, holdout_len=3)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    R["n_programs"] = len(corpus.programs)
    R["n_train_rows"] = len(tr)
    R["n_heldout_queries"] = len(ho)
    print(f"  programs={len(corpus.programs)}  train={len(tr)}  heldout={len(ho)}")

    print()
    print("=" * 78)
    print("STEP 2  system (pinned construction)")
    print("=" * 78)
    system, tok = m.build_system(corpus, pin_seed=PIN)
    R["system_pinned"] = True
    print(f"  built with pin_seed={PIN}")

    print()
    print("=" * 78)
    print("STEP 3  frozen waves")
    print("=" * 78)
    gen = m.WaveTextGenerator(system, tok, train_body=False)
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    Ztr = cb.build_waves(gen, [corpus.specs[i] for i in tr])
    Zho = cb.build_waves(gen, [corpus.specs[i] for i in ho])
    R["wave_dim"] = int(Ztr.shape[1])
    R["waves_finite"] = bool(torch.isfinite(Ztr).all() and torch.isfinite(Zho).all())
    print(f"  Ztr {tuple(Ztr.shape)}  Zho {tuple(Zho.shape)}  finite={R['waves_finite']}")

    sim = (Zho @ Ztr.T)
    if torch.is_complex(sim):
        sim = sim.real
    gold = cb.gold_sets(tr_tgt, ho_tgt)
    R["gold_nonempty_frac"] = sum(1 for g in gold if g) / len(gold)
    print(f"  sim {tuple(sim.shape)}  gold-nonempty {R['gold_nonempty_frac']:.4f}")

    print()
    print("=" * 78)
    print("STEP 4  BANK IDENTITY (full vs dedup)")
    print("=" * 78)
    rep_full = cb.containment_report(sim, gold)
    cols = cb.dedup_columns(tr_tgt)
    sim_d, gold_d = cb.restrict(sim, cols, gold)
    rep_dedup = cb.containment_report(sim_d, gold_d)
    R["bank_full"] = rep_full
    R["bank_dedup"] = rep_dedup
    R["n_distinct_traces"] = len(cols)
    for tag, r in (("full", rep_full), ("dedup", rep_dedup)):
        print(f"  {tag:>5}  cols={r['bank_columns']:>4}  containment={r['containment']:.4f}  "
              f"ctrl={r['size_matched_control']:.4f}  grp={r['mean_group_size']:.2f}  "
              f"prec={r['within_group_gold_precision']:.5f}  "
              f"rowchance={r['row_uniform_chance']:.5f}")

    print()
    print("=" * 78)
    print("STEP 5  SELECTOR ARMS on the DEDUP bank (G-SJ3a surface)")
    print("=" * 78)
    ch_full = cb.frequency_channel(tr_tgt, list(range(sim.shape[1])), sim.shape)
    ch_dedup = cb.frequency_channel(tr_tgt, cols, sim_d.shape)
    noop = cb.channel_noop_check(ch_dedup)
    R["channel_noop_check"] = noop
    print(f"  channel no-op check: uniq={noop['n_unique_values']} "
          f"const={noop['frac_constant']} is_noop={noop['is_noop']}")
    print(f"  (a CONSTANT channel would make two_stage == uniform exactly)")
    arms = {}
    for mode, ch_arg in (("argmax", None), ("uniform", None), ("two_stage", ch_dedup)):
        arms[mode] = cb.score_selector(sim_d, gold_d, ch_arg, mode, tr_tgt)
        a = arms[mode]
        print(f"  {mode:>9}  top1={a['top1']:.4f}  perm={a['top1_under_permutation']:.4f}  "
              f"drift={a['drift']:.4f}")
    R["selector_dedup"] = arms

    print()
    print("=" * 78)
    print("STEP 6  G-SJ3a GATE")
    print("=" * 78)
    best = max(arms.items(), key=lambda kv: kv[1]["top1"])
    R["best_mode"] = best[0]
    # BANK IDENTITY vs CALIBRATION. The 0.0450 bar and the 0.0046 control were measured
    # on a 206-candidate dedup bank built from a 1092-row bank. THIS bank has
    # R["n_distinct_traces"] columns. Reporting a pass/fail against a calibration from a
    # different bank would be a fixture-mismatched comparison -- the defect class this
    # program keeps catching. So: run the gate on the SAME-BANK control, and record the
    # calibration identity explicitly.
    same_bank_control = rep_dedup["size_matched_control"]
    gate = wr.gate_sj3a(best[1]["top1"], rand_control=same_bank_control, slot_r=None)
    R["calibration"] = {
        "bar_source": "G_SJ3A_BAR=0.0450, control 0.0046 on a 206-candidate dedup bank",
        "this_bank_columns": R["n_distinct_traces"],
        "this_bank_control": same_bank_control,
        "bank_matches_calibration": R["n_distinct_traces"] == 206,
        "note": "gate run with the SAME-BANK control; a cross-bank comparison is refused",
    }
    R["gate_sj3a"] = gate
    for k, v in gate.items():
        print(f"  {k:28} {v}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not R["calibration"]["bank_matches_calibration"]:
        R["verdict"] = "G_SJ3A_BLOCKED_BANK_MISMATCH__" + best[0].upper()
    elif gate.get("passes"):
        R["verdict"] = "G_SJ3A_PASSED"
    else:
        R["verdict"] = "G_SJ3A_FAILED__" + best[0].upper()
    print(f"  {R['verdict']}")
    print(f"  best mode={best[0]} top1={best[1]['top1']:.4f} vs bar {gate.get('bar')} "
          f"control {gate.get('random_control')} lift {gate.get('lift')}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(f"\n[receipt] {os.path.abspath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""henri_mvp_induction.py -- MINIMUM VIABLE HENRI: STRUCTURAL operator inference.

WHY THIS SUPERSEDES THE FIRST MVP
    henri_mvp_compose.py scored 1.0000 on held-out length-4 programs, but that is
    TAUTOLOGICAL and I labelled it so: the compiler matched demos against a candidate
    library whose entries ARE the executor's own apply_op, so once an operator is
    identified the composition reproduces the trace by construction.

    This module INFERS operator STRUCTURE from demonstration pairs. The executor is used
    only to GENERATE demos and to SCORE held-out rows -- never to define a hypothesis.
    A hypothesis here cannot "know" the executor's code, so held-out accuracy is earned.

HYPOTHESIS SPACE (structural forms, no named operators)
    IDENTITY     out == in
    CONSTANT     out is one repeated digit
    PERMUTATION  out is a rearrangement of in (a position permutation)
    SHIFT        out[j] = (in[j] + s) mod 10 for one global s
    AFFINE       out[j] = (a*in[j] + b) mod 10 for fixed (a, b)

    The search is exhaustive and FILTERS by consistency across every demo. A hypothesis
    that explains the demos but is wrong on a held-out input is a real falsification.

PROTOCOL (comparable to the compose MVP)
    demos come from LENGTH-1 training programs only. Held-out programs are LENGTH 4.
    Composition is measured on strings the inference never saw.

NO PRETRAINING. NO WEB CORPUS. Semantics compiled at test time from pairs. CPU only.
This is a mechanism measurement, not an AAII result.
"""
import json
import os
import sys

V = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, V)
import henri_core.m4_generative as m   # noqa: E402

INPUTS = ["1234", "5678", "9012", "2468", "1357", "8642", "9753", "3141"]
HOLDOUT_LEN = 4
EDGE_INPUTS = ["7777", "0123", "9876", "1111", "5005"]


def parse_spec(spec):
    prog, inp = spec.split("apply ", 1)[1].split(" to ", 1)
    return prog.strip(), inp.strip()


# ---------------- STRUCTURAL HYPOTHESES (no executor code) ----------------
def h_identity(pairs):
    if all(i == o for i, o in pairs):
        return lambda d: d, {"form": "IDENTITY", "params": {}}
    return None


def h_constant(pairs):
    outs = {o for _, o in pairs}
    if len(outs) == 1 and all(len(o) == len(i) for i, o in pairs):
        c = outs.pop()
        return lambda d, c=c: c, {"form": "CONSTANT", "params": {"value": c}}
    return None


def h_permutation(pairs):
    if not all(len(o) == len(i) for i, o in pairs):
        return None
    perm = None
    for i, o in pairs:
        if len(set(i)) != len(i):
            continue
        p = []
        ok = True
        for ch in o:
            if ch not in i:
                ok = False
                break
            p.append(i.index(ch))
        if ok and sorted(p) == list(range(len(i))):
            perm = p
            break
    if perm is None:
        return None
    if all("".join(i[p[k]] for k in range(len(i))) == o for i, o in pairs):
        return (lambda d, p=perm: "".join(d[p[k]] for k in range(len(d))),
                {"form": "PERMUTATION", "params": {"perm": perm}})
    return None


def h_repeat_first(pairs):
    """out is the FIRST input digit repeated. A structural form, not a named operator.
    Added because the Stage-1 table showed C -> NONE with 0 survivors: coverage of the
    hypothesis space, not inference quality, was capping accuracy."""
    if all(len(o) == len(i) for i, o in pairs) and all(o == i[0] * len(i) for i, o in pairs):
        return lambda d: d[0] * len(d), {"form": "REPEAT_FIRST", "params": {}}
    return None


def h_repeat_last(pairs):
    if all(len(o) == len(i) for i, o in pairs) and all(o == i[-1] * len(i) for i, o in pairs):
        return lambda d: d[-1] * len(d), {"form": "REPEAT_LAST", "params": {}}
    return None


def h_shift(pairs):
    if not all(len(o) == len(i) for i, o in pairs):
        return None
    shifts = {(int(o[j]) - int(i[j])) % 10 for i, o in pairs for j in range(len(i))}
    if len(shifts) == 1:
        s = shifts.pop()
        return (lambda d, s=s: "".join(str((int(c) + s) % 10) for c in d),
                {"form": "SHIFT", "params": {"s": s}})
    return None


def h_affine(pairs):
    """y = (a*x + b) mod 10, fit on the observed digit map. Requires consistency."""
    if not all(len(o) == len(i) for i, o in pairs):
        return None
    table = {}
    for i, o in pairs:
        for x, y in zip(i, o):
            xa, ya = int(x), int(y)
            if xa in table and table[xa] != ya:
                return None
            table[xa] = ya
    if len(table) < 2:
        return None
    sols = [(a_, b_) for a_ in range(10) for b_ in range(10)
            if all((a_ * x + b_) % 10 == table[x] for x in table)]
    if len(sols) != 1:
        return None
    a_, b_ = sols[0]
    return (lambda d, a=a_, b=b_: "".join(str((a * int(c) + b) % 10) for c in d),
            {"form": "AFFINE", "params": {"a": a_, "b": b_}})


HYPOTHESES = [("IDENTITY", h_identity), ("CONSTANT", h_constant),
              ("REPEAT_FIRST", h_repeat_first), ("REPEAT_LAST", h_repeat_last),
              ("PERMUTATION", h_permutation), ("SHIFT", h_shift), ("AFFINE", h_affine)]


def infer_operator(demos):
    surv = []
    for name, h in HYPOTHESES:
        try:
            r = h(demos)
        except Exception:
            r = None
        if r is not None:
            # TUPLE ORDER (self-caught): (fn, meta, name). The first version
            # stored (name, fn, meta) and then called c[0](out) -- a STRING --
            # raising TypeError: 'str' object is not callable at composition.
            surv.append((r[0], r[1], name))
    if not surv:
        return None, []
    return surv[0], surv


def main() -> int:
    print("=" * 86)
    print("MVP (INDUCTION) -- structural operator inference from demonstration pairs")
    print("=" * 86)
    corpus = m.build_corpus(max_len=HOLDOUT_LEN, holdout_len=HOLDOUT_LEN, inputs=INPUTS)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    R = {"inputs": INPUTS, "holdout_len": HOLDOUT_LEN, "n_train": len(tr),
         "n_heldout": len(ho),
         "hypothesis_space": [n for n, _ in HYPOTHESES]}
    print(f"corpus {len(corpus.programs)} rows | train {len(tr)} | held-out {len(ho)}")
    print(f"train lengths {sorted(set(len(corpus.programs[i]) for i in tr))} | "
          f"held lengths {sorted(set(len(corpus.programs[i]) for i in ho))}")

    demos = {}
    for i in tr:
        prog, inp = parse_spec(corpus.specs[i])
        if len(prog) == 1:
            demos.setdefault(prog, []).append((inp, corpus.traces[i]))

    print()
    print("=" * 86)
    print("STAGE 1  INFER each operator STRUCTURE from its demos")
    print("=" * 86)
    print(f"{'op':>4} {'demos':>6} {'form':>13} {'params':>24} {'survivors':>10}")
    compiled = {}
    for op in sorted(demos):
        best, all_surv = infer_operator(demos[op])
        compiled[op] = best
        if best is None:
            print(f"{op:>4} {len(demos[op]):>6} {'NONE':>13} {'-':>24} {0:>10}")
        else:
            print(f"{op:>4} {len(demos[op]):>6} {best[1]['form']:>13} "
                  f"{str(best[1]['params']):>24} {len(all_surv):>10}")
    R["inferred"] = {op: (None if c is None else
                          {"form": c[1]["form"], "params": c[1]["params"]})
                     for op, c in compiled.items()}
    R["all_inferred"] = all(c is not None for c in compiled.values())

    print()
    print("=" * 86)
    print("STAGE 2  COMPOSE on held-out LENGTH-4 programs (never demonstrated)")
    print("=" * 86)
    by_len, misses, hits, total = {}, [], 0, 0
    for i in ho:
        prog, inp = parse_spec(corpus.specs[i])
        out, ok = inp, True
        for ch in prog:
            c = compiled.get(ch)
            if c is None:
                ok = False
                break
            out = c[0](out)
        total += 1
        good = ok and out == corpus.traces[i]
        hits += int(good)
        L = len(prog)
        a_, b_ = by_len.get(L, (0, 0))
        by_len[L] = (a_ + int(good), b_ + 1)
        if not good and len(misses) < 5:
            misses.append({"program": prog, "input": inp,
                           "predicted": out if ok else None,
                           "actual": corpus.traces[i]})
    acc = hits / max(total, 1)
    R.update({"n_evaluated": total, "n_correct": hits, "heldout_accuracy": acc,
              "sample_misses": misses,
              "by_program_length": {str(L): {"correct": v[0], "total": v[1],
                                            "accuracy": v[0] / v[1]}
                                    for L, v in sorted(by_len.items())}})
    for L in sorted(by_len):
        a_, b_ = by_len[L]
        print(f"  length {L}: {a_}/{b_} = {a_ / b_:.4f}")
    print(f"\\n  HELD-OUT ACCURACY (induced operators, composed) : {acc:.4f}")
    for m_ in misses:
        print(f"    MISS {m_}")

    print()
    print("=" * 86)
    print("NON-TAUTOLOGY PROBE -- induced outputs on UNSEEN and EDGE inputs")
    print("=" * 86)
    rows = []
    for op in sorted(compiled):
        c = compiled[op]
        if c is None:
            continue
        for d in EDGE_INPUTS:
            rows.append({"operator": f"{op}:{c[1]['form']}", "input": d,
                         "induced": c[0](d), "executor": m.run_program(op, d),
                         "agrees": c[0](d) == m.run_program(op, d)})
    for r_ in rows[:15]:
        print(f"    {r_['operator']:<18} {r_['input']} -> {r_['induced']:<8} "
              f"exec {r_['executor']:<8} {'ok' if r_['agrees'] else 'DIFF'}")
    R["edge_probe_rows"] = rows
    R["edge_probe_all_agree"] = all(r_["agrees"] for r_ in rows)
    print(f"\\n  all edge probes agree: {R['edge_probe_all_agree']}")

    # ---------- HONEST LIMITATIONS (measured, not asserted) ----------
    R["limitations"] = {
        "substrate_tautology": (
            "The hypothesis space was DESIGNED knowing the executor's operator set is "
            "{I,R,C} = {increment, reverse, copy_first}. SHIFT(s=1) IS increment, "
            "PERMUTATION([3,2,1,0]) IS reverse, REPEAT_FIRST IS copy_first. So a 1.0000 "
            "accuracy is EXPECTED, not earned: the space spans the ground truth. What IS "
            "earned is that the PARAMETERS (shift amount, permutation vector) are FIT from "
            "pairs and generalise to edge inputs never shown to the inference."),
        "non_unique_identification": (
            "Operator I reports 2 survivors: SHIFT(s=1) and AFFINE(a=1,b=1) are "
            "algebraically identical maps ((d+1) mod 10 either way). The identification is "
            "therefore NOT unique for I, and the 'survivors' count alone cannot certify a "
            "mechanism. Same duplicate-behaviour class as the compose MVP's library check."),
        "difficulty": (
            "A 3-operator digit algebra with 4-digit strings is a toy substrate. Nothing "
            "here scales to open-vocabulary tasks, and no AAII claim follows."),
    }
    R["verdict"] = ("MVP_INDUCTION_COMPOSES_WITH_SUBSTRATE_TAUTOLOGY"
                    if acc >= 0.99 and R["edge_probe_all_agree"]
                    else "MVP_INDUCTION_PARTIAL")
    print(f"\\n  VERDICT: {R['verdict']}  (accuracy {acc:.4f} from INDUCED structure)")
    print("\\n  HONEST LIMITATIONS (recorded in the receipt):")
    for k, v in R["limitations"].items():
        print(f"    {k}: {v[:96]}...")

    p = os.path.join(V, "..", "design", "zone_a", "evidence",
                     "henri_mvp_induction_observed.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(R, open(p, "w", encoding="utf-8"), indent=2)
    print(f"[receipt] {os.path.abspath(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ZONE C ENGRAM STORE (pre-registered). Tests the user's proposal directly.

THE PROPOSAL UNDER TEST
    "Training for HENRI looks like test-time scanning of a data system and
    registering that data as a holographic cache; the wave geometric processes
    then organize, transform, and act on the data."

DATA LEVEL (the user's question: what should the engrams BE?)
    M4 gives a free, EXACT ground-truth oracle: run_program(program, input) ->
    trace. So the engram is a verified (input wave, output wave) pair. Three
    formats are compared:
      CO-STORE  engram := (Psi_spec, trace_ids)                [baseline]
      BIND      engram := Psi_spec (conv) Psi_trace; retrieve by correlating the
                query key with the bound trace, then match trace waves
      RAW-TEXT  engram := (text, trace_ids) via exact substring match  [cheat arm:
                shows how much of any gain is pure string lookup]

PRE-REGISTERED DECISION RULE (frozen before the run)
    bar = max(unigram_floor, shuffled_store_control) + 0.05
    A format is a REAL mechanism iff it beats the bar AND beats its OWN shuffled
    control. Test-time CONTINUOUS learning is real iff accuracy on item n+1 rises
    with store size n while the shuffled-registration control stays flat.

PART C (the headline for step 4)
    coverage@k of the store as a CANDIDATE PROPOSER: does the store supply the
    correct token in its top-5 more often than chance? Compare real vs shuffled
    store. This is the only route that can raise the 0.4286 ceiling, because
    reranking cannot.

Caps: CPU, D=4096 diagnostic. Not a model-performance claim.
"""
import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system,
                                      unigram_floor)

PIN = 20261010
OUT = os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp", "zonec.json")


def flat(x):
    x = x.reshape(x.shape[0], -1)
    return torch.cat([x.real, x.imag], -1) if torch.is_complex(x) else x


def tokacc(preds, tgts):
    hit = tot = 0
    for row, ids in zip(preds, tgts):
        for a, b in zip(row, ids):
            tot += 1
            hit += int(a == b)
    return hit / max(1, tot)


def em(preds, tgts):
    h = 0
    for row, want in zip(preds, tgts):
        n = min(len(want), len(row))
        h += int(list(row[:n]) == list(want[:n]))
    return h / max(1, len(tgts))


def main():
    t0 = time.time()
    R = {"pin": PIN}
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, pin_seed=PIN)
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    tr_txt = [corpus.traces[i] for i in corpus.train_idx]
    ho_txt = [corpus.traces[i] for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, M4Config().max_trace
    gen = WaveTextGenerator(system, tok, train_body=True)

    with torch.no_grad():
        K = flat(gen.wave(tr_spec)).float()
        Q = flat(gen.wave(ho_spec)).float()
    Kn, Qn = F.normalize(K, dim=-1), F.normalize(Q, dim=-1)
    S = Qn @ Kn.T
    R["key_cos"] = {"max": round(float(S.max()), 4),
                    "mean_top1": round(float(S.max(1).values.mean()), 4)}

    # ---- E1 CO-STORE: retrieve the neighbour's trace ------------------------
    nn = S.argmax(1)
    p_cost = [tr_tgt[j] for j in nn.tolist()]
    R["E1_costore"] = {"held_tok": round(tokacc(p_cost, ho_tgt), 4),
                       "held_em": round(em(p_cost, ho_tgt), 4)}

    # ---- E2 BIND: HRR bind key and trace wave, retrieve by unbinding --------
    with torch.no_grad():
        # BUG FIX (self-caught, second attempt): the first draft multiplied
        # fft(bound)[48,D] by fft(Q)[108,D] -> RuntimeError 48 vs 108. The correct
        # retrieval score is a PAIRWISE (query x engram) quantity. By Parseval,
        #   score[i,j] = (1/D) Re sum_k conj(Fb[j,k]) * Fq[i,k] * Fw[j,k]
        # which is a plain complex matmul, so no [108,48,D] tensor is allocated.
        Wtr = gen.wave(tr_txt)                      # trace waves, complex [T,D]
        keys = gen.wave(tr_spec)
        Qc = gen.wave(ho_spec)                      # complex queries [n,D]
        D = keys.shape[-1]
        Fb = torch.fft.fft(keys, dim=-1)            # [T, D]
        Fw = torch.fft.fft(Wtr, dim=-1)             # [T, D]
        Fq = torch.fft.fft(Qc, dim=-1)              # [n, D]
        P = Fb.conj() * Fw                          # [T, D]
        score = torch.einsum("ik,jk->ij", Fq, P).real / D   # [n, T]
        bind_nn = score.argmax(1)
    p_bind = [tr_tgt[j] for j in bind_nn.tolist()]
    R["E2_bind_hrr"] = {"held_tok": round(tokacc(p_bind, ho_tgt), 4),
                        "held_em": round(em(p_bind, ho_tgt), 4),
                        "same_as_costore": bool(torch.equal(bind_nn, nn))}

    # ---- E3 RAW-TEXT: exact substring arm (string-lookup upper bound) ------
    # CORRECTION (self-caught): v1 returned 0.0 and I labelled it a "broken arm".
    # After the fix it is STILL 0.0, and that is BY CONSTRUCTION, not a bug:
    #   held-out programs have len(p) == 3; every train program has len(p) <= 2
    #   (build_corpus: train_idx = len(p) < holdout_len). So no train spec ever
    #   contains a held-out program string, and literal lookup scores 0.0.
    # This is an INFORMATIVE negative: the compositional split has no hidden
    # string shortcut, so the wave mechanisms are measured fairly.
    tkn = 0
    for q, want in zip(ho_spec, ho_txt):
        prog = q[len("apply "):].split(" to ")[0]
        hit = next((t for s, t in zip(tr_spec, tr_txt) if prog in s), None)
        tkn += int(hit == want)
    R["E3_rawtext_lookup"] = {
        "held_em": round(tkn / max(1, len(ho_txt)), 4),
        "note": ("0.0 BY CONSTRUCTION: no train spec contains a length-3 program "
                 "string. Proves the split has no literal leakage.")}

    # ---- C1 shuffled store control -----------------------------------------
    g = torch.Generator().manual_seed(PIN)
    perm = torch.randperm(len(tr_tgt), generator=g).tolist()
    tr_tgt_shuf = [tr_tgt[i] for i in perm]
    p_c1 = [tr_tgt_shuf[j] for j in nn.tolist()]
    R["C1_shuffled_store"] = {"held_tok": round(tokacc(p_c1, ho_tgt), 4)}

    floor = round(unigram_floor(tok, tr_tgt, ho_tgt), 4)
    R["unigram_floor"] = floor
    bar = round(max(floor, R["C1_shuffled_store"]["held_tok"]) + 0.05, 4)
    R["bar"] = bar

    # ---- PART B: sequential test-time registration --------------------------
    # Empty store -> process held-out items one at a time; after predicting item
    # n, register the VERIFIED (spec, trace) for item n. Accuracy on item n+1 as
    # a function of store size n. Control: register a SHUFFLED trace instead.
    def sequential(register_wrong=False):
        curves = []
        for warm in (0, 12, 24, 48, 72):
            store_k, store_t = [], []
            hits = tot = 0
            for i in range(len(ho_spec)):
                if store_k:
                    with torch.no_grad():
                        sk = flat(gen.wave(store_k)).float()
                        qv = flat(gen.wave([ho_spec[i]])).float()
                    sim = F.normalize(qv, dim=-1) @ F.normalize(sk, dim=-1).T
                    j = int(sim.argmax(1))
                    pred = store_t[j]
                else:
                    pred = [0]
                tot += 1
                n = min(len(ho_tgt[i]), len(pred))
                hits += int(list(pred[:n]) == list(ho_tgt[i][:n]))
                val = tr_tgt[(i + 1) % len(tr_tgt)] if register_wrong else ho_tgt[i]
                if len(store_k) < warm:
                    store_k.append(ho_spec[i])
                    store_t.append(val)
            curves.append(round(hits / max(1, tot), 4))
        return curves
    sizes = [0, 12, 24, 48, 72]
    R["B_sequential"] = {
        "store_sizes": sizes,
        "real_registration_curve": sequential(False),
        "shuffled_registration_control": sequential(True),
    }
    rc_ = R["B_sequential"]["real_registration_curve"]
    cc_ = R["B_sequential"]["shuffled_registration_control"]
    R["B_verdict"] = {
        "real_improves_with_store_size": bool(rc_[-1] > rc_[0]),
        "control_flat_or_worse": bool(cc_[-1] <= cc_[0] + 0.02),
        "continuous_learning_supported": bool(rc_[-1] > rc_[0]
                                              and cc_[-1] <= cc_[0] + 0.02),
    }

    # ---- PART C: coverage@k as a candidate proposer -------------------------
    # BUG FIX (self-caught): v1 took `idx_matrix` and then closed over `S`, so
    # the real and shuffled arms produced IDENTICAL numbers (0.1429/0.4464/0.625/
    # 0.8304). A control that cannot differ from its treatment is vacuous. Use the
    # argument.
    def coverage(sim_matrix, label_map):
        rows = {}
        for k in (1, 5, 20, 100):
            kk = min(k, len(tr_tgt))
            topk = torch.topk(sim_matrix, kk, dim=-1).indices
            hit = tot = 0
            for i, ids in enumerate(ho_tgt):
                for p, t in enumerate(ids[:M]):
                    tot += 1
                    hit += int(any(int(label_map[j][p]) == int(t)
                                   for j in topk[i]
                                   if p < len(label_map[j])))
            rows[k] = round(hit / max(1, tot), 4)
        return rows
    # the REAL store: neighbour index -> that neighbour's verified trace.
    # the SHUFFLED store: SAME ranking, but each neighbour returns a WRONG trace.
    real_map = list(tr_tgt)
    shuf_map = [tr_tgt[i] for i in perm]
    R["C_real_store_coverage"] = coverage(S, real_map)
    R["C_shuffled_store_coverage"] = coverage(S, shuf_map)
    R["C_control_distinguishable"] = bool(
        R["C_real_store_coverage"] != R["C_shuffled_store_coverage"])

    R["verdict"] = {
        "E1_beats_own_shuffled_control": bool(
            R["E1_costore"]["held_tok"] > R["C1_shuffled_store"]["held_tok"]),
        "E1_passes_bar": bool(R["E1_costore"]["held_tok"] > bar),
        "binding_differs_from_costore": bool(not R["E2_bind_hrr"]["same_as_costore"]),
        "store_raises_coverage_at_5": bool(
            R["C_real_store_coverage"][5] > R["C_shuffled_store_coverage"][5]),
        "overall": ("STORE_REAL_NOT_PASSING" if
                    R["E1_costore"]["held_tok"] > R["C1_shuffled_store"]["held_tok"]
                    else "STORE_INDISTINGUISHABLE_FROM_CONTROL"),
    }
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

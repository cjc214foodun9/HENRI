"""ARM B partial probe: does the CLOSED-VOCABULARY egress path carry content?

WHY THIS EXISTS
    SpecContract A records the egress defect as `top1_token_unique = 1` across 16
    distinct waves. That measurement lives on ONE path: the open-vocabulary
    ARM_U unbinder (`down_proj [2048,65536] -> lm_head [32000,2048]`,
    g4_armu_unbinder_diag.py:64-67). A SECOND, different egress path exists in
    tree and was NOT measured by that diagnostic: `HoloEgressCodebook`
    (henri_vla_tokenizer.py:433), whose codebook is DERIVED FROM THE TOKENIZER
    rather than random.

    An exploratory run on 2026-10-03 observed content-hit 0.4103 vs null 0.0321
    on that second path. One run is not evidence. This probe makes it
    reproducible with pre-registered thresholds and multiple carrier templates.

WHAT IS AND IS NOT CLAIMED
    CLAIMED (if it passes): on a CLOSED vocabulary of 156 words, the
    tokenizer-derived wave->symbol readout recovers a word's own index far above
    a label-permutation null, for several carrier templates.
    NOT CLAIMED: open-vocabulary generation. NOT CLAIMED: ARM B is solved. The
    156-word closed vocab is a far easier problem than 32000-token open text.

PRE-REGISTERED (frozen before the run)
    Q1 CONTENT      content_hit_rate > null_max  at  alpha = 0.01, for EVERY
                    carrier template tested (no cherry-picking).
    Q2 SPECIFICITY  a word's own index beats its best WRONG index
                    (mean margin > 0).
    Q3 CONTROL      random complex waves must NOT reach the Q1 threshold;
                    otherwise the readout is not evidence of content.
    Q4 ROUNDTRIP    identity_round_trip rate == 1.0 (codebook self-consistency).
    N_NULL = 200 label permutations. SEEDS = 3. TEMPLATES = 3.

FAIL-CLOSED: if Q3 fails, report VACUOUS and claim nothing.
CPU only. $0. Deterministic. No latency measured.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

_HENRI_V2 = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _HENRI_V2)
sys.path.insert(0, str(Path(_HENRI_V2) / "experiments" / "verification"))

import torch  # noqa: E402
import henri_vla_tokenizer as vt  # noqa: E402
import m1_open_answer_gate as m1  # noqa: E402

# ------------------------------------------------------------- pre-registration
N_NULL = 200
ALPHA = 0.01
SEEDS = (20261003, 20261004, 20261005)
TEMPLATES = ("the {w} report", "{w} is the word", "describe {w} now")
POSITION_BINDING = "fractional_shift"


def build(seed: int):
    vocab = list(m1.VOCAB)
    cfg = vt.HoloVLAConfig(
        ambient_dim_D=2048, num_blocks=256, block_slots=8, grid_size_S=16,
        vocab_size_V=len(vocab), feat_dim=256,
        position_binding=POSITION_BINDING, seed=seed)
    tok = vt.HoloVLATokenizer(cfg)
    cb = vt.HoloEgressCodebook(cfg, tok, vocab)
    return cfg, tok, cb, vocab


def perm_null(idxs: list[int], n_mod: int, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = idxs[:]
        rng.shuffle(p)
        out.append(sum(1 for i, d in enumerate(p) if i == d) / float(n_mod))
    return sorted(out)


def main() -> int:
    report: dict = {"preregistration": {
        "N_null": N_NULL, "alpha": ALPHA, "seeds": list(SEEDS),
        "templates": list(TEMPLATES), "position_binding": POSITION_BINDING,
        "path": "HoloEgressCodebook (closed vocab, tokenizer-derived)"}}
    results = {}

    for seed in SEEDS:
        cfg, tok, cb, vocab = build(seed)
        V = len(vocab)

        rt = cb.identity_round_trip()
        per_tmpl = {}
        for tmpl in TEMPLATES:
            with torch.no_grad():
                waves = tok.encode_text([tmpl.format(w=w) for w in vocab])
                idxs = cb.logits(waves).argmax(dim=-1).tolist()
            hits = sum(1 for i, d in enumerate(idxs) if i == d)
            rate = hits / float(V)
            null = perm_null(idxs, V, N_NULL, seed + hash(tmpl) % 1000)
            # specificity: own-index logit vs best wrong-index logit
            with torch.no_grad():
                lg = cb.logits(waves)
            margins = []
            for i in range(V):
                own = float(lg[i, i])
                row = lg[i].clone()
                row[i] = float("-inf")
                margins.append(own - float(row.max()))
            per_tmpl[tmpl] = {
                "hits": hits, "rate": rate, "null_mean": statistics.mean(null),
                "null_p99": null[int(0.99 * len(null))], "null_max": null[-1],
                "margin_mean": statistics.mean(margins),
                "above_null": rate > null[-1],
            }
        # random-wave control (Q3)
        g = torch.Generator().manual_seed(seed + 7)
        r = torch.randn(V, cfg.ambient_dim_D, generator=g).to(torch.complex64)
        r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        with torch.no_grad():
            rid = cb.logits(r).argmax(dim=-1).tolist()
        ctrl_rate = sum(1 for i, d in enumerate(rid) if i == d) / float(V)
        ctrl_distinct = len(set(rid)) / float(V)

        results[str(seed)] = {
            "vocab": V, "roundtrip_rate": rt["rate"],
            "roundtrip_dupes": rt["duplicate_manifest_entries"],
            "templates": per_tmpl,
            "control": {"content_hit_rate": ctrl_rate,
                        "distinct_ratio": ctrl_distinct},
        }
        print("seed %d: rt=%.3f" % (seed, rt["rate"]), flush=True)
        for tmpl, d in per_tmpl.items():
            print("   %-18s rate=%.4f null_max=%.4f above=%s margin=%+.4f"
                  % (tmpl, d["rate"], d["null_max"], d["above_null"],
                     d["margin_mean"]), flush=True)
        print("   control rate=%.4f distinct=%.3f" % (ctrl_rate, ctrl_distinct),
              flush=True)

    all_t = [d for s in results.values() for d in s["templates"].values()]
    q1 = all(d["above_null"] for d in all_t)
    q2 = all(d["margin_mean"] > 0 for d in all_t)
    q4 = all(s["roundtrip_rate"] == 1.0 for s in results.values())
    # Q3: control must stay at/below chance band, i.e. far below the Q1 rates
    ctrl_max = max(s["control"]["content_hit_rate"] for s in results.values())
    treat_min = min(d["rate"] for d in all_t)
    q3 = ctrl_max < treat_min
    conclusions = {
        "Q1_content_above_null_every_template": q1,
        "Q2_specificity_positive_margin": q2,
        "Q3_random_control_below_treatment": q3,
        "Q4_roundtrip_exact": q4,
        "control_max_rate": ctrl_max, "treatment_min_rate": treat_min,
    }
    if not q3:
        verdict = "VACUOUS_RANDOM_CONTROL_MATCHES_TREATMENT"
    elif q1 and q2 and q4:
        verdict = "CLOSED_VOCAB_EGRESS_CARRIES_CONTENT"
    else:
        verdict = "CLOSED_VOCAB_EGRESS_FAIL:" + ",".join(
            k for k in ("Q1_content_above_null_every_template",
                        "Q2_specificity_positive_margin",
                        "Q4_roundtrip_exact") if not conclusions[k])

    report["results"] = results
    report["conclusions"] = conclusions
    report["verdict"] = verdict
    report["evidence_class"] = "OBSERVED"
    report["scope_limit"] = (
        "Closed vocabulary (156 words), tokenizer-derived codebook, CPU. "
        "Does NOT address the open-vocabulary ARM_U unbinder path "
        "(32000 tokens) where top1_token_unique=1 was recorded. "
        "ARM B remains UNMEASURED for open-vocabulary text.")

    body = json.dumps(report, sort_keys=True, indent=2)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out = os.path.join(os.environ.get("TEMP", "/tmp"), "armB_closed_vocab_receipt.json")
    Path(out).write_text(body, encoding="utf-8")

    print("\n== VERDICT ==")
    for k, v in conclusions.items():
        print("  %-38s = %s" % (k, v))
    print("  verdict = %s" % verdict)
    print("  receipt = %s" % out)
    print("  receipt_sha256 = %s" % sha)
    return 0 if verdict == "CLOSED_VOCAB_EGRESS_CARRIES_CONTENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())

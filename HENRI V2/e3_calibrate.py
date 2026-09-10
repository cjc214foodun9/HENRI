"""E3 calibrate runner — egress-head reform: Arm A (probe readout) + Arm B (generative CE).

Carrier: carrier/e3-egress-reform. Prereg (sealed):
experiments/verification/e3_egress_reform_prereg.md

Split: EXACT sealed E2 partition — e2_calibrate.build_pairs_ordered with NO seed
argument (builder default 20260909); calib 10,000 / eval 1,000. Gold token =
first BPE token of the window's last word (prefix-stability guarded).
Label: CONDITIONAL_SAME_CORPUS_HELDOUT.

Bounds (user-supplied; unchanged from Gate 3.1): P@1 >= 0.285, P@5 >= 0.640.
Single run, no retries. Export e3_egress_production.pt ONLY on Arm-B full pass.

Usage (remote layout defaults; override E2_DIR / TEACHER_DIR if needed):
  python e3_calibrate.py --mode scaffold --data CORPUS --teacher-root TDIR --out OUT
  python e3_calibrate.py --mode full     --data CORPUS --teacher-root TDIR --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- pre-registered bounds (module constants; never env-overridable) -------
P_AT_1_BOUND = 0.285
P_AT_5_BOUND = 0.640

E2_CKPT_SHA_PREFIX = "08747c70"
TE_TABLE_SHA_PREFIX = "48b174e9"
CORPUS_SHA_PREFIX = "e83889ba"
N_CALIB = 10000
N_EVAL = 1000
LABEL = "CONDITIONAL_SAME_CORPUS_HELDOUT"


def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond, msg, code):
    if not cond:
        print(f"E3_VERDICT=BLOCKED_INFRA reason={code} ({msg})")
        raise SystemExit(2)


def gold_next_token(text: str, tok) -> int:
    """First BPE token of the window's last word (prefix-stability guarded).

    The WindowPair wave encodes the PREFIX (all but the last word); the gold
    token is the next token after that prefix. If BPE is not prefix-stable the
    formula is ambiguous -> fail closed.
    """
    words = text.split()
    if len(words) < 2:
        raise ValueError("window must contain >= 2 words")
    prefix = " ".join(words[:-1])
    ids_all = tok(text)
    ids_pre = tok(prefix)
    _require(ids_all[:len(ids_pre)] == ids_pre and len(ids_all) > len(ids_pre),
             f"tokenization prefix instability for {text[:40]!r}",
             "GOLD_TOKEN_PREFIX_MISMATCH")
    return ids_all[len(ids_pre)]


def batch_metrics(scores: torch.Tensor, gold: torch.Tensor):
    """(p1_hits, p5_hits, nll_sum) for token scores [B, V] and gold [B]."""
    top5 = torch.topk(scores, 5, dim=-1).indices
    p1 = int((top5[:, 0] == gold).sum())
    p5 = int((top5 == gold.unsqueeze(1)).any(dim=1).sum())
    nll = -torch.log_softmax(scores.float(), dim=-1).gather(
        1, gold.unsqueeze(1)).sum()
    return p1, p5, float(nll)


def run(mode: str, data: Path, teacher_root: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    e2_dir = Path(os.environ.get("E2_DIR", "/root/e2-calib/out_full"))
    t_dir = Path(os.environ.get("TEACHER_DIR", str(teacher_root)))
    ckpt = e2_dir / "e2_egress_production.pt"
    te = t_dir / "teacher_embeddings.pt"
    tokj = t_dir / "tokenizer.json"

    for f in (ckpt, te, tokj, data):
        _require(Path(f).exists(), str(f), "ARTIFACT_MISSING")
    ck_sha = _sha256_file(ckpt)
    _require(ck_sha.startswith(E2_CKPT_SHA_PREFIX), ck_sha[:16],
             "E2_CKPT_SHA_MISMATCH")
    te_sha = _sha256_file(te)
    _require(te_sha.startswith(TE_TABLE_SHA_PREFIX), te_sha[:16],
             "TE_TABLE_SHA_MISMATCH")
    c_sha = _sha256_file(data)
    _require(c_sha.startswith(CORPUS_SHA_PREFIX), c_sha[:16],
             "CORPUS_SHA_MISMATCH")

    from e2_calibrate import (  # noqa: E402
        load_corpus, sentence_split, make_tokenizer, build_pairs_ordered)
    from e1_egress_calibration import (  # noqa: E402
        E1Config, E1EgressHead, load_teacher_embeddings)
    from e3_egress_reform import E3GenerativeHead, probe_token_distribution

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = load_teacher_embeddings(te)                 # CPU [V, d_target]
    tok = make_tokenizer(t_dir)

    sents = []
    for r in load_corpus(data):
        sents.extend(sentence_split(r))
    cfg_eval = E1Config(device="cpu")
    want = 16 if mode == "scaffold" else N_CALIB + N_EVAL
    # EXACT sealed E2 call: NO seed argument (builder default 20260909).
    pairs = build_pairs_ordered(sents, cfg_eval, emb, tok, want)
    if mode == "scaffold":
        calib_pairs, eval_pairs = pairs[:16], []
    else:
        calib_pairs = pairs[:N_CALIB]
        eval_pairs = pairs[N_CALIB:N_CALIB + N_EVAL]
        _require(len(eval_pairs) == N_EVAL, str(len(eval_pairs)),
                 "EVAL_SPLIT_MISMATCH")

    receipt = {
        "mode": mode,
        "label": LABEL,
        "bounds": {"p1": P_AT_1_BOUND, "p5": P_AT_5_BOUND},
        "ckpt_sha256": ck_sha,
        "te_sha256": te_sha,
        "corpus_sha256": c_sha,
        "n_calib": len(calib_pairs),
        "n_eval": len(eval_pairs),
    }

    # ---------------- Arm A: probe-distribution readout (zero training) ----
    if mode == "full":
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        _require(isinstance(payload, dict) and "model_state" in payload,
                 str(list(payload)[:6]), "E2_CKPT_FORMAT_MISMATCH")
        cfg = payload["config"]
        head_cfg = E1Config(
            d_model=int(cfg.get("d_model", 65536)),
            num_blocks=int(cfg.get("num_blocks", 8192)),
            block_dim=int(cfg.get("block_dim", 8)),
            d_bottleneck=int(cfg.get("d_bottleneck", 256)),
            d_target=int(cfg.get("d_target", 896)),
            vocab_size=int(cfg.get("vocab_size", 151936)),
            seed=int(cfg.get("seed", 20260908)),
            device="cpu",
        )
        e2_head = E1EgressHead(head_cfg)
        e2_head.load_state_dict(payload["model_state"])
        e2_head.to(device).eval()

        E_dev = emb.to(device=device, dtype=torch.float32)
        p1A = p5A = 0
        nllA = 0.0
        nA = 0
        B = 64
        with torch.no_grad():
            for i0 in range(0, len(eval_pairs), B):
                batch = eval_pairs[i0:i0 + B]
                gold = torch.tensor([gold_next_token(p.text, tok)
                                     for p in batch], device=device)
                waves = torch.stack([p.wave.to(device) for p in batch])
                feats, _ = e2_head(waves)
                scores, _ = probe_token_distribution(feats, E_dev, k=16,
                                                     tau=0.07)
                a, b, nl = batch_metrics(scores, gold)
                p1A += a
                p5A += b
                nllA += nl
                nA += len(batch)
        pA1 = p1A / max(nA, 1)
        pA5 = p5A / max(nA, 1)
        receipt["arm_A"] = {
            "p_at_1": round(pA1, 6), "p_at_5": round(pA5, 6),
            "perplexity": round(math.exp(nllA / max(nA, 1)), 4),
            "n_eval": nA,
            "verdict": ("E3A_PASS" if (pA1 >= P_AT_1_BOUND
                        and pA5 >= P_AT_5_BOUND) else "E3A_FAIL"),
        }
        print(f"[E3] Arm A (probe readout): P@1={pA1:.4f} P@5={pA5:.4f} "
              f"n={nA} -> {receipt['arm_A']['verdict']}")

    # ---------------- Arm B: generative CE head ---------------------------
    E_table_dev = emb.to(device=device, dtype=torch.float32)
    ghead = E3GenerativeHead.from_e2_checkpoint(ckpt, E_table=E_table_dev,
                                                device=device)

    if mode == "scaffold":
        waves = torch.stack([p.wave for p in calib_pairs])
        golds = torch.tensor([gold_next_token(p.text, tok)
                              for p in calib_pairs])
        losses = ghead.fit_ce(waves, golds, epochs=2, batch=8, lr=3e-4)
        ok = bool(losses) and all(math.isfinite(l) for l in losses) \
            and losses[-1] < losses[0]
        receipt["scaffold_ok"] = bool(ok)
        receipt["scaffold_ce"] = [round(l, 4) for l in losses[:8]]
        receipt["iso"] = ghead.isometry_error()
        (out / "scaffold_receipt.json").write_text(
            json.dumps(receipt, indent=2))
        print(f"[E3] scaffold CE {losses[0]:.4f} -> {losses[-1]:.4f} ok={ok}")
        print(f"E3_SCAFFOLD_RC={0 if ok else 1}")
        return 0 if ok else 1

    calib_waves = torch.stack([p.wave for p in calib_pairs])
    calib_golds = torch.tensor([gold_next_token(p.text, tok)
                                for p in calib_pairs])
    losses = ghead.fit_ce(calib_waves, calib_golds, epochs=3, batch=32, lr=3e-4)
    iso = ghead.isometry_error()
    g1 = iso <= 1e-4
    g2 = losses[-1] < 0.9 * losses[0]

    p1B = p5B = 0
    nllB = 0.0
    nB = 0
    B = 64
    with torch.no_grad():
        for i0 in range(0, len(eval_pairs), B):
            batch = eval_pairs[i0:i0 + B]
            gold = torch.tensor([gold_next_token(p.text, tok)
                                 for p in batch], device=device)
            waves = torch.stack([p.wave.to(device) for p in batch])
            scores = ghead.token_logits(waves)
            a, b, nl = batch_metrics(scores, gold)
            p1B += a
            p5B += b
            nllB += nl
            nB += len(batch)
    pB1 = p1B / max(nB, 1)
    pB5 = p5B / max(nB, 1)
    changed = ghead.rotation_changed(eval_pairs, device=device)
    g4 = changed >= 8
    verdictB = ("E3B_PASS" if (g1 and g2 and g4
                and pB1 >= P_AT_1_BOUND and pB5 >= P_AT_5_BOUND)
                else "E3B_FAIL")
    receipt["arm_B"] = {
        "p_at_1": round(pB1, 6), "p_at_5": round(pB5, 6),
        "perplexity": round(math.exp(nllB / max(nB, 1)), 4),
        "iso": iso, "g1": bool(g1), "g2": bool(g2),
        "ce_first": round(losses[0], 4), "ce_last": round(losses[-1], 4),
        "rotation_changed": int(changed), "g4": bool(g4),
        "n_eval": nB, "verdict": verdictB,
    }
    print(f"[E3] Arm B (generative CE): P@1={pB1:.4f} P@5={pB5:.4f} "
          f"iso={iso:.2e} ce={losses[0]:.2f}->{losses[-1]:.2f} "
          f"rot={changed}/16 -> {verdictB}")

    all_pass = (receipt.get("arm_A", {}).get("verdict") == "E3A_PASS"
                and verdictB == "E3B_PASS")
    if verdictB == "E3B_PASS":
        path, sha = ghead.export(out)
        receipt["export"] = {"path": str(path), "sha256": sha}
    (out / "e3_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"E3_VERDICT={'E3_PASS' if all_pass else 'E3_FAIL'}")
    return 0 if all_pass else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scaffold", "full"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rc = run(args.mode, args.data, args.teacher_root, args.out)
    print(f"E3_RC={rc}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

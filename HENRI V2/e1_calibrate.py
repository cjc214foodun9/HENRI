"""E1 calibrate runner — scaffold + full-scale, gates G1-G4, fail-closed.

Carrier: carrier/e1-egress-calibration. Prereg sha256 02560a9a856890c758e002ea...
Usage:
  python e1_calibrate.py --mode scaffold --data corberi_2609_04732.txt
        --teacher-root /root/e1-calib --out /root/e1-calib/out
  python e1_calibrate.py --mode full [same args] --epochs 3 --batch 32

Scaffold: <=16 windows, <=2 epochs, CPU or CUDA; asserts finite/descent.
Full: 80/20 ordered split, train, then G1-G4; export only if all pass.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from e1_egress_calibration import (
    E1Config, E1Trainer, build_window_pairs, load_teacher_embeddings,
    isometry_penalty,
)


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").replace("  ", " "))
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def make_tokenizer(teacher_root: Path):
    """Load Qwen BPE tokenizer.json via tokenizers lib; returns encode(text)->list[int]."""
    from tokenizers import Tokenizer
    tok_path = teacher_root / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer missing: {tok_path}")
    tok = Tokenizer.from_file(str(tok_path))
    return lambda text: tok.encode(text).ids


def gate_g3_retrieval(head, eval_pairs, teacher_emb, device):
    waves = torch.stack([p.wave for p in eval_pairs]).to(device)
    targets = torch.stack([p.target for p in eval_pairs]).to(device)
    emb = teacher_emb.to(device)

    @torch.no_grad()
    def p1(head_):
        feats, _ = head_(waves)
        z = F.normalize(feats, dim=-1)
        sim = z @ F.normalize(emb, dim=-1).t()
        pred = sim.argmax(dim=-1)
        # score: predicted row embedding vs target centroid cosine
        scored = F.cosine_similarity(F.normalize(emb[pred], dim=-1),
                                     F.normalize(targets, dim=-1))
        return float(scored.mean().item())

    trained = p1(head)

    # untrained init: same config, fresh seed draws
    head2 = head.__class__(head.config).to(device)
    untrained = p1(head2)

    # random-wave baseline: random unit waves through the untrained head
    torch.manual_seed(7)
    rand = torch.randn(waves.shape, device=device)
    rand = F.normalize(rand.reshape(rand.shape[0], -1), dim=-1).reshape(waves.shape)
    @torch.no_grad()
    def p1_rand():
        feats, _ = head2(rand)
        z = F.normalize(feats, dim=-1)
        sim = z @ F.normalize(emb, dim=-1).t()
        pred = sim.argmax(dim=-1)
        return float(F.cosine_similarity(F.normalize(emb[pred], dim=-1),
                                         F.normalize(targets, dim=-1)).mean().item())
    random_w = p1_rand()
    return trained, untrained, random_w


def gate_g4_rotation(head, eval_pairs, teacher_emb, device, seed=7, probes=16):
    """Seeded per-block O(8) rotation must change argmax teacher retrieval for >= 8/16."""
    g = torch.Generator().manual_seed(seed)
    emb = teacher_emb.to(device)
    changed = 0
    with torch.no_grad():
        for w in eval_pairs[:probes]:
            w0 = w.wave.to(device).unsqueeze(0)
            feats0, _ = head(w0)
            pred0 = F.normalize(feats0, dim=-1) @ F.normalize(emb, dim=-1).t()
            i0 = int(pred0.argmax(-1).item())
            # per-block random orthogonal rotation via QR of random 8x8
            # NOTE: CPU generator cannot draw CUDA tensors (torch pitfall);
            # draw on CPU first, then transfer.
            R = torch.randn(8, 8, generator=g, device="cpu").to(device)
            Q, _ = torch.linalg.qr(R)
            wrot = torch.einsum("ij,bjk->bik", Q, w0)
            feats1, _ = head(wrot)
            pred1 = F.normalize(feats1, dim=-1) @ F.normalize(emb, dim=-1).t()
            i1 = int(pred1.argmax(-1).item())
            if i1 != i0:
                changed += 1
    return changed


def run(mode: str, data: Path, teacher_root: Path, out: Path,
        epochs: int, batch: int, device: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    text = data.read_text(encoding="utf-8")
    sents = sentence_split(text)
    print(f"[E1] sentences={len(sents)} from {data.name}")

    emb = load_teacher_embeddings(teacher_root / "teacher_embeddings.pt")
    print(f"[E1] teacher embeddings {tuple(emb.shape)}")

    tok = make_tokenizer(teacher_root)
    config = E1Config(device=device)
    limit = 16 if mode == "scaffold" else None
    train_texts = sents[:limit] if limit else sents
    # ordered split by sentence order
    n = len(train_texts)
    if mode == "full":
        n_cal = int(0.8 * n)
        calib_sents, eval_sents = train_texts[:n_cal], train_texts[n_cal:]
    else:
        calib_sents, eval_sents = train_texts, []
    print(f"[E1] calib={len(calib_sents)} eval={len(eval_sents)}")

    calib_pairs = build_window_pairs(calib_sents, max_words=config.max_words,
                                     teacher_embeddings=emb, tokenizer=tok)
    eval_pairs = build_window_pairs(eval_sents, max_words=config.max_words,
                                    teacher_embeddings=emb, tokenizer=tok)
    print(f"[E1] pairs calib={len(calib_pairs)} eval={len(eval_pairs)}")
    if not calib_pairs:
        print("[E1] ABORT: no calibration pairs"); return 2

    trainer = E1Trainer(config, out_dir=str(out))
    losses = trainer.fit(calib_pairs, epochs=epochs, batch=batch,
                         log_every=max(1, len(calib_pairs) // max(1, batch)))
    rec: dict = {"mode": mode, "pairs_calib": len(calib_pairs),
                 "pairs_eval": len(eval_pairs), "losses": losses[:50],
                 "loss_first": losses[0] if losses else None,
                 "loss_last": losses[-1] if losses else None,
                 "steps": len(losses)}

    if mode == "scaffold":
        rec["scaffold_ok"] = bool(losses) and math.isfinite(losses[-1]) \
            and losses[-1] < losses[0]
        (out / "scaffold_receipt.json").write_text(json.dumps(rec, indent=2))
        print(f"[E1] scaffold ok={rec['scaffold_ok']} "
              f"loss {rec['loss_first']:.4f}->{rec['loss_last']:.4f}")
        return 0 if rec["scaffold_ok"] else 1

    # full: gates
    iso = isometry_penalty(trainer.head.stiefel_down_proj).item()
    g1 = iso <= 1e-4
    avg_first = float(np.mean(losses[:max(1, len(losses) // 5)]))
    avg_last = float(np.mean(losses[-max(1, len(losses) // 5):]))
    g2 = avg_last < 0.9 * avg_first
    tr, un, rw = gate_g3_retrieval(trainer.head, eval_pairs, emb, device)
    g3 = (tr > un + 0.05) and (tr > rw + 0.05)
    changed = gate_g4_rotation(trainer.head, eval_pairs, emb, device)
    g4 = changed >= 8

    rec.update({"iso": iso, "g1": g1,
                "avg_first": avg_first, "avg_last": avg_last, "g2": g2,
                "p1_trained": tr, "p1_untrained": un, "p1_random": rw, "g3": g3,
                "rotation_changed": changed, "g4": g4})
    all_pass = g1 and g2 and g3 and g4
    rec["verdict"] = "E1_GATES_PASS" if all_pass else "E1_GATES_FAIL"
    if all_pass:
        path, sha = trainer.export("e1_egress_calibrated.pt")
        rec["checkpoint"] = str(path)
        rec["checkpoint_sha256"] = sha
    (out / "full_receipt.json").write_text(json.dumps(rec, indent=2))
    print(f"[E1] G1={g1} G2={g2} G3={g3} G4={g4} verdict={rec['verdict']}")
    return 0 if all_pass else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scaffold", "full"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    rc = run(args.mode, args.data, args.teacher_root, args.out,
             args.epochs, args.batch, args.device)
    print(f"E1_RC={rc}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

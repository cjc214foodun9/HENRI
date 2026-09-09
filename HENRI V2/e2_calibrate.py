"""E2 calibrate runner — scaffold + full-scale, gates G1-G4 with k-NN probe.

Carrier: carrier/e2-egress-knn-scale. Prereg sha256 e58e6b41374c3a8e45b108d0... (sealed)
Usage:
  python e2_calibrate.py --mode scaffold --data /root/e2-corpus/wikitext2_train.parquet
        --teacher-root /root/e1-calib --out /root/e2-calib/out_scaffold
  python e2_calibrate.py --mode full [same args] --epochs 3 --batch 32

Scaffold: <=16 windows, <=2 epochs; asserts finite/descent (fail-closed).
Full: first 10,000 calib windows / next 1,000 eval windows (ordered by row,
sentence-disjoint); train on calib; G1-G4 with G3 = k-NN softmax alignment margin:
  margin = align_trained - max(align_untrained, align_random) >= +0.05
Export e2_egress_production.pt ONLY if all gates pass (fail-closed, no promotion).
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
    E1Config, E1EgressHead, E1Trainer, build_window_pairs, load_teacher_embeddings,
    isometry_penalty,
)
from e2_egress_knn import knn_softmax_probe


def load_corpus(parquet_path: Path) -> list[str]:
    """wikitext-2-raw-v1 parquet -> sentence list (ordered by row)."""
    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path))
    texts = table.column("text").to_pylist()
    return texts


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").replace("  ", " "))
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def make_tokenizer(teacher_root: Path):
    from tokenizers import Tokenizer
    tok_path = teacher_root / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer missing: {tok_path}")
    tok = Tokenizer.from_file(str(tok_path))
    return lambda text: tok.encode(text).ids


def build_pairs_ordered(sents: list[str], config, emb, tok, want: int,
                        seed: int = 20260909):
    """Deterministic ordered window pairs: first `want` valid windows, chunked
    with ONE shared codec built over the first 30k sentences (implied vocab)."""
    from e1_egress_calibration import _codec_for
    codesrc = sents[:30000] if len(sents) > 30000 else sents
    codec = _codec_for(codesrc)
    pairs = []
    for i in range(0, len(sents), 2000):
        chunk = sents[i:i + 2000]
        pairs.extend(build_window_pairs(chunk, max_words=config.max_words,
                                        teacher_embeddings=emb, tokenizer=tok,
                                        codec=codec))
        if len(pairs) >= want:
            break
    return pairs[:want]


@torch.no_grad()
def _probe_align(feats: torch.Tensor, emb: torch.Tensor,
                 targets: torch.Tensor, k: int, tau: float, chunk: int = 128) -> float:
    """Mean cosine(z_hat, y_target) over all rows, chunked to bound [B,V] sim."""
    sims = []
    for i in range(0, feats.shape[0], chunk):
        z_hat, _ = knn_softmax_probe(feats[i:i + chunk], emb, k=k, tau=tau)
        sims.append(F.cosine_similarity(F.normalize(z_hat, dim=-1),
                                        F.normalize(targets[i:i + chunk], dim=-1)))
    return float(torch.cat(sims).mean().item())


def gate_g3_knn(head, eval_pairs, emb, device, k=16, tau=0.07):
    """(align_trained, align_untrained, align_random) under the k-NN probe."""
    waves = torch.stack([p.wave for p in eval_pairs]).to(device)
    targets = torch.stack([p.target for p in eval_pairs]).to(device)
    t_emb = emb.to(device)

    @torch.no_grad()
    def align(head_):
        feats, _ = head_(waves)
        return _probe_align(feats, t_emb, targets, k, tau)

    trained = align(head)
    head2 = head.__class__(head.config).to(device)   # same config seed init = untrained
    untrained = align(head2)
    # random-wave baseline through the untrained head (E1 parity)
    torch.manual_seed(7)
    rand = torch.randn(waves.shape, device=device)
    rand = F.normalize(rand.reshape(rand.shape[0], -1), dim=-1).reshape(waves.shape)
    feats_r, _ = head2(rand)
    random_w = _probe_align(feats_r, t_emb, targets, k, tau)
    return trained, untrained, random_w


def gate_g4_rotation(head, eval_pairs, emb, device, seed=7, probes=16):
    """Seeded per-block O(8) rotation must change argmax teacher retrieval >= 8/16."""
    g = torch.Generator().manual_seed(seed)
    t_emb = emb.to(device)
    changed = 0
    with torch.no_grad():
        for w in eval_pairs[:probes]:
            w0 = w.wave.to(device).unsqueeze(0)
            feats0, _ = head(w0)
            pred0 = F.normalize(feats0, dim=-1) @ F.normalize(t_emb, dim=-1).t()
            i0 = int(pred0.argmax(-1).item())
            R = torch.randn(8, 8, generator=g, device="cpu").to(device)
            Q, _ = torch.linalg.qr(R)
            wrot = torch.einsum("ij,bkj->bki", Q, w0)
            feats1, _ = head(wrot)
            pred1 = F.normalize(feats1, dim=-1) @ F.normalize(t_emb, dim=-1).t()
            i1 = int(pred1.argmax(-1).item())
            if i1 != i0:
                changed += 1
    return changed


def run(mode: str, data: Path, teacher_root: Path, out: Path,
        epochs: int, batch: int, device: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rows = load_corpus(data)
    sents = []
    for r in rows:
        sents.extend(sentence_split(r))
    print(f"[E2] corpus_rows={len(rows)} sentences={len(sents)}")

    emb = load_teacher_embeddings(teacher_root / "teacher_embeddings.pt")
    print(f"[E2] teacher embeddings {tuple(emb.shape)}")
    tok = make_tokenizer(teacher_root)
    config = E1Config(device=device)
    want = 16 if mode == "scaffold" else 11000
    train_want = 0 if mode == "scaffold" else 10000
    pairs = build_pairs_ordered(sents, config, emb, tok, want)
    calib_pairs = pairs[:train_want] if train_want else pairs
    eval_pairs = pairs[train_want:train_want + (0 if mode == "scaffold" else 1000)]
    print(f"[E2] pairs calib={len(calib_pairs)} eval={len(eval_pairs)}")
    if not calib_pairs:
        print("[E2] ABORT: no calibration pairs")
        return 2

    trainer = E1Trainer(config, out_dir=str(out))
    ep = 2 if mode == "scaffold" else epochs
    losses = trainer.fit(calib_pairs, epochs=ep, batch=batch,
                         log_every=max(1, len(calib_pairs) // max(1, batch)))
    rec: dict = {"mode": mode, "pairs_calib": len(calib_pairs),
                 "pairs_eval": len(eval_pairs), "losses": losses[:50],
                 "loss_first": losses[0] if losses else None,
                 "loss_last": losses[-1] if losses else None,
                 "steps": len(losses), "k": 16, "tau": 0.07}

    if mode == "scaffold":
        rec["scaffold_ok"] = bool(losses) and math.isfinite(losses[-1]) \
            and losses[-1] < losses[0]
        (out / "scaffold_receipt.json").write_text(json.dumps(rec, indent=2))
        print(f"[E2] scaffold ok={rec['scaffold_ok']} "
              f"loss {rec['loss_first']:.4f}->{rec['loss_last']:.4f}")
        return 0 if rec["scaffold_ok"] else 1

    iso = isometry_penalty(trainer.head.stiefel_down_proj).item()
    g1 = iso <= 1e-4
    avg_first = float(np.mean(losses[:max(1, len(losses) // 5)]))
    avg_last = float(np.mean(losses[-max(1, len(losses) // 5):]))
    g2 = avg_last < 0.9 * avg_first
    tr, un, rw = gate_g3_knn(trainer.head, eval_pairs, emb, device)
    g3 = tr > un + 0.05 and tr > rw + 0.05
    changed = gate_g4_rotation(trainer.head, eval_pairs, emb, device)
    g4 = changed >= 8

    rec.update({"iso": iso, "g1": g1,
                "avg_first": avg_first, "avg_last": avg_last, "g2": g2,
                "knn_trained": tr, "knn_untrained": un, "knn_random": rw, "g3": g3,
                "rotation_changed": changed, "g4": g4})
    all_pass = g1 and g2 and g3 and g4
    rec["verdict"] = "E2_GATES_PASS" if all_pass else "E2_GATES_FAIL"
    if all_pass:
        path, sha = trainer.export("e2_egress_production.pt")
        rec["checkpoint"] = str(path)
        rec["checkpoint_sha256"] = sha
    (out / "full_receipt.json").write_text(json.dumps(rec, indent=2))
    print(f"[E2] G1={g1} G2={g2} G3={g3} G4={g4} verdict={rec['verdict']}")
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
    print(f"E2_RC={rc}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

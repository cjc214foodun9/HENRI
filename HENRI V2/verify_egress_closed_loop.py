"""Gate 3.1 — P@1 / P@5 / perplexity diagnostic (teacher-anchored single-shot).

Carrier: carrier/p1-pk-diagnostics. Prereg (sealed, verified against the real
E2 export format {config, model_state, telemetry} — NOT the fabricated W/b):
experiments/verification/gate31_p1pk_prereg.md

TEACHER-ANCHORED SINGLE-SHOT definitions (NOT autoregressive LM perplexity):
  z_i      = normalize(E1EgressHead(Ψ_i)[0])          # teacher space [896]
  logits_i = E_norm @ z_i                             # [V=151936]
  P@1      = mean I[argmax(logits_i) == y_i]
  P@5      = mean I[y_i in top5(logits_i)]
  PPL      = exp(mean -log softmax(logits_i)[y_i])

Split: the SEALED E2 evaluation partition reconstructed via the real
e2_calibrate.build_pairs_ordered (corpus rows ordered; calib 10,000 / eval
1,000; seed semantics = E1Config.seed 20260908 for head init, ordered corpus
split deterministic). Label: CONDITIONAL_SAME_CORPUS_HELDOUT.

Frozen artifacts (fail-closed on absence/mismatch):
  - E2 checkpoint e2_egress_production.pt (sha prefix 08747c70)
  - teacher shard model.safetensors @ rev 060db6499f32... (embed key
    model.embed_tokens.weight, [151936, 896])
  - corpus wikitext2_train.parquet (sha e83889ba...)
  - teacher_embeddings.pt (E1 harvest artifact)

Single run, no retries. Bounds are module constants; never env-overridable.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- pre-registered bounds (module constants; NOT env-overridable) --------
P_AT_1_BOUND = 0.285
P_AT_5_BOUND = 0.640

E2_CKPT_SHA_PREFIX = "08747c70"
TEACHER_REV = "060db6499f32"
CORPUS_SHA_PREFIX = "e83889ba"
SPLIT_SEED = 20260908
N_CALIB = 10000
N_EVAL = 1000


def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond, msg, code):
    if not cond:
        print(f"GATE31_VERDICT=BLOCKED_INFRA reason={code} ({msg})")
        raise SystemExit(2)


def load_token_embeddings(shard_path: Path, device: str) -> torch.Tensor:
    """Frozen Qwen2.5-0.5B token-embedding table [151936, 896]."""
    from safetensors import safe_open
    key = "model.embed_tokens.weight"
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        _require(key in keys, f"missing {key}", "TEACHER_EMBED_KEY_MISSING")
        E = f.get_tensor(key)
    return E.to(device=device, dtype=torch.float32)


def gold_next_token(text: str, tok) -> int:
    """First BPE token of the last word (the token after the prefix wave).

    The E2 WindowPair wave encodes the PREFIX (all but last word); this is the
    pre-registered teacher-anchored 'next token' (single-shot, non-autoregressive).
    """
    words = text.split()
    if len(words) < 2:
        raise ValueError("window must contain >= 2 words")
    prefix = " ".join(words[:-1])
    ids_all = tok(text)
    ids_pre = tok(prefix)
    _require(len(ids_all) > len(ids_pre),
             f"tokenization mismatch for {text[:40]!r}", "GOLD_TOKEN_UNRESOLVED")
    return ids_all[len(ids_pre)]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e2_dir = Path(os.environ.get("E2_DIR", "/root/e2-calib/out_full"))
    teacher_dir = Path(os.environ.get("TEACHER_DIR", "/root/e1-calib"))
    corpus_path = Path(os.environ.get("CORPUS",
                                      "/root/e2-corpus/wikitext2_train.parquet"))
    out_dir = Path(os.environ.get("OUT", "/root/gate31"))

    ckpt = e2_dir / "e2_egress_production.pt"
    shard = teacher_dir / "model.safetensors"
    _require(ckpt.exists(), str(ckpt), "E2_CKPT_MISSING")
    _require(shard.exists(), str(shard), "TEACHER_SHARD_MISSING")
    _require((teacher_dir / "tokenizer.json").exists(),
             "tokenizer.json missing", "TOKENIZER_MISSING")
    _require((teacher_dir / "teacher_embeddings.pt").exists(),
             "teacher_embeddings.pt missing", "TEACHER_EMBEDDINGS_MISSING")
    _require(corpus_path.exists(), str(corpus_path), "CORPUS_MISSING")

    ck_sha = _sha256_file(ckpt)
    _require(ck_sha.startswith(E2_CKPT_SHA_PREFIX), ck_sha[:16],
             "E2_CKPT_SHA_MISMATCH")
    c_sha = _sha256_file(corpus_path)
    _require(c_sha.startswith(CORPUS_SHA_PREFIX), c_sha[:16],
             "CORPUS_SHA_MISMATCH")

    # Real E2 split reconstruction (same code path as the sealed E2 run).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e2_calibrate import (  # noqa: E402
        load_corpus, sentence_split, make_tokenizer, build_pairs_ordered)
    from e1_egress_calibration import (  # noqa: E402
        E1Config, E1EgressHead, load_teacher_embeddings)

    payload = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and "model_state" in payload,
             f"unexpected checkpoint keys {list(payload)[:6]}",
             "E2_CKPT_FORMAT_MISMATCH")
    cfg = payload["config"]
    head_cfg = E1Config(
        d_model=cfg.get("d_model", 65536),
        num_blocks=cfg.get("num_blocks", 8192),
        block_dim=cfg.get("block_dim", 8),
        d_bottleneck=cfg.get("d_bottleneck", 256),
        d_target=cfg.get("d_target", 896),
        vocab_size=cfg.get("vocab_size", 151936),
        seed=SPLIT_SEED,
        device="cpu",
    )
    head = E1EgressHead(head_cfg)
    head.load_state_dict(payload["model_state"])
    head.to(device)
    head.eval()

    E = load_token_embeddings(shard, device)
    E_norm = F.normalize(E, p=2, dim=-1)
    del E
    if device == "cuda":
        torch.cuda.empty_cache()

    # Reconstruct eval pairs exactly as the sealed E2 full run did.
    sents = []
    for r in load_corpus(corpus_path):
        sents.extend(sentence_split(r))
    emb = load_teacher_embeddings(teacher_dir / "teacher_embeddings.pt")
    tok = make_tokenizer(teacher_dir)
    cfg_eval = E1Config(device="cpu")
    # Mirror e2_calibrate.run() EXACTLY (sealed E2 split): no seed argument,
    # builder default 20260909 (head-init seed is E1Config.seed=20260908).
    pairs = build_pairs_ordered(sents, cfg_eval, emb, tok,
                                want=N_CALIB + N_EVAL)
    eval_pairs = pairs[N_CALIB:N_CALIB + N_EVAL]
    _require(len(eval_pairs) == N_EVAL, str(len(eval_pairs)),
             "EVAL_SPLIT_MISMATCH")

    golds = [gold_next_token(p.text, tok) for p in eval_pairs]

    p1_total = 0
    p5_total = 0
    rnd_total = 0
    nll_sum = 0.0
    n = 0
    g = torch.Generator().manual_seed(SPLIT_SEED + 1)
    B = 64
    with torch.no_grad():
        for i0 in range(0, len(eval_pairs), B):
            batch = eval_pairs[i0:i0 + B]
            gold = torch.tensor(golds[i0:i0 + B], device=device)
            waves = torch.stack([p.wave.to(device) for p in batch])  # [B,8192,8]
            feats, _ = head(waves)                                   # [B,896]
            z = F.normalize(feats.float(), dim=-1)
            scores = z @ E_norm.t()                                  # [B,V]
            top5 = torch.topk(scores, 5, dim=-1).indices
            p1_hits = int((top5[:, 0] == gold).sum())
            p5_hits = int((top5 == gold.unsqueeze(1)).any(dim=1).sum())
            nll = -torch.log_softmax(scores.float(), dim=-1).gather(
                1, gold.unsqueeze(1)).sum()
            rnd = F.normalize(
                torch.randn(B, z.shape[1], generator=g).to(device), p=2, dim=-1)
            rnd_hits = int((rnd @ E_norm.t()).argmax(dim=-1).eq(gold).sum())
            p1_total += p1_hits
            p5_total += p5_hits
            rnd_total += rnd_hits
            nll_sum += float(nll)
            n += len(batch)

    p_at_1 = p1_total / n
    p_at_5 = p5_total / n
    random_p_at_1 = rnd_total / n
    perplexity = math.exp(nll_sum / n)
    verdict = ("GATE31_PASS" if (p_at_1 >= P_AT_1_BOUND and p_at_5 >= P_AT_5_BOUND)
               else "GATE31_FAIL")
    receipt = {
        "verdict": verdict,
        "p_at_1": round(p_at_1, 6),
        "p_at_5": round(p_at_5, 6),
        "perplexity": round(perplexity, 4),
        "random_p_at_1": round(random_p_at_1, 6),
        "n_eval": n,
        "bounds": {"p1": P_AT_1_BOUND, "p5": P_AT_5_BOUND},
        "label": "CONDITIONAL_SAME_CORPUS_HELDOUT",
        "definition": "teacher-anchored single-shot NLL (non-autoregressive); "
                      "head lm_head disclosed as UNTRAINED; readout = teacher "
                      "embedding table cosine",
        "ckpt_sha256": ck_sha,
        "corpus_sha256": c_sha,
        "teacher_rev": TEACHER_REV,
        "seed": SPLIT_SEED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate31_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(f"GATE31_VERDICT={verdict}")


if __name__ == "__main__":
    main()

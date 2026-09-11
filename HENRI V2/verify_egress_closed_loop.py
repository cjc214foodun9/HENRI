"""Gate 3.1 — P@1 / P@5 / perplexity diagnostic (teacher-anchored single-shot).

Carrier: carrier/p1-pk-diagnostics. Prereg (sealed):
experiments/verification/gate31_p1pk_prereg.md

TEACHER-ANCHORED SINGLE-SHOT definitions (NOT autoregressive LM perplexity):
  feats_i = normalize(E1EgressHead(Psi_i)[0])         # teacher space [896]
  logits_i = E_norm @ feats_i                         # [V]
  P@1      = mean I[argmax(logits_i) == y_i]
  P@5      = mean I[y_i in top5(logits_i)]
  PPL      = exp(mean -log softmax(logits_i)[y_i])

Split: the SEALED E2 evaluation partition reconstructed EXACTLY as the E2 full
run did — e2_calibrate.build_pairs_ordered (ordered, no seed arg → builder
default 20260909; head-init seed = E1Config.seed 20260908). windows: first
10,000 calibration, next 1,000 evaluation. Label: CONDITIONAL_SAME_CORPUS_HELDOUT.

Frozen artifacts, fail-closed on absence/mismatch:
  - E2 checkpoint e2_egress_production.pt (sha256 prefix 08747c70; verified
    keys {'config','model_state','telemetry'})
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
# ---- legacy bounds RETIRED 2026-09-11 (see bounds_retirement_receipt.json) --
# The E3 legacy constants (historically 0.285 / 0.640) are RETIRED:
# E4a measured the context-free marginal at 0.440 and the strongest trivial
# baseline at 0.483 on a fresh split, so 0.285 sits BELOW the trivial baseline
# and cannot separate a mechanism from a constant predictor. Bounds are now
# loaded per construct from the sealed E4a receipt and are never hard-coded.
LEGACY_BOUNDS_RETIRED = {"p_at_1": 0.285, "p_at_5": 0.640}
E4A_RECEIPT = Path(os.environ.get(
    "HENRI_E4A_RECEIPT",
    r"C:\Users\chan\henri-telemetry\e3\e4a_construct_audit.json"))


def load_registered_bounds(construct: str = "C1_sentence_window") -> dict:
    """Load per-construct bounds from the sealed E4a audit; fail closed."""
    if not E4A_RECEIPT.exists():
        print("BOUNDS_VERDICT=BLOCKED_INFRA reason=E4A_RECEIPT_MISSING")
        raise SystemExit(2)
    rec = json.loads(E4A_RECEIPT.read_text(encoding="utf-8"))
    b = rec["registered_bounds"][construct]
    base = rec["constructs"][construct]["best_trivial_baseline"]["p1"]
    if not (b["p1"] > base and 0 < b["ce_max"]):
        print("BOUNDS_VERDICT=BLOCKED_INFRA reason=BOUND_NOT_ABOVE_TRIVIAL")
        raise SystemExit(2)
    return b


_REG = None  # resolved lazily; import must never fail closed


def __getattr__(name: str):
    """Resolve retired bounds lazily (PEP 562).

    Importing this module always succeeds; the bound is loaded from the
    sealed E4a receipt on FIRST ACCESS and fails closed only then. This keeps
    importers such as e3_diag_target_gap.py (gold_next_token) working on a
    host that has no receipt, while the bound itself can never be
    hard-coded again.
    """
    global _REG
    if name in ("P_AT_1_BOUND", "P_AT_5_BOUND"):
        if _REG is None:
            _REG = load_registered_bounds()
        return _REG["p1"] if name == "P_AT_1_BOUND" else _REG["p5"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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


def gold_next_token(text: str, tok) -> int:
    """First BPE token of the window's last word.

    The E2 WindowPair wave encodes the PREFIX (all but last word); the gold
    token is the next token after that prefix — single-shot, non-autoregressive.
    BPE must be prefix-stable for `y = ids_all[len(ids_pre)]` to be
    well-defined; otherwise fail closed.
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e2_dir = Path(os.environ.get("E2_DIR", "/root/e2-calib/out_full"))
    teacher_dir = Path(os.environ.get("TEACHER_DIR", "/root/e1-calib"))
    corpus_path = Path(os.environ.get("CORPUS",
                                      "/root/e2-corpus/wikitext2_train.parquet"))
    out_dir = Path(os.environ.get("OUT", "/root/gate31"))

    ckpt = e2_dir / "e2_egress_production.pt"
    shard = teacher_dir / "model.safetensors"
    tokj = teacher_dir / "tokenizer.json"
    _require(ckpt.exists(), str(ckpt), "E2_CKPT_MISSING")
    _require(shard.exists(), str(shard), "TEACHER_SHARD_MISSING")
    _require(tokj.exists(), str(tokj), "TOKENIZER_MISSING")
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

    # Frozen teacher token-embedding table [151936, 896] from the shard.
    from safetensors import safe_open
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        key = "model.embed_tokens.weight"
        _require(key in keys, key, "TEACHER_EMBED_KEY_MISSING")
        E = f.get_tensor(key).to(device=device, dtype=torch.float32)
    E_norm = F.normalize(E, p=2, dim=-1)
    del E
    if device == "cuda":
        torch.cuda.empty_cache()

    # Reconstruct the eval split EXACTLY as e2_calibrate.run() (no seed arg).
    sents = []
    for r in load_corpus(corpus_path):
        sents.extend(sentence_split(r))
    emb = load_teacher_embeddings(teacher_dir / "teacher_embeddings.pt")
    tok = make_tokenizer(teacher_dir)
    cfg_eval = E1Config(device="cpu")
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
                torch.randn(len(batch), z.shape[1], generator=g).to(device),
                p=2, dim=-1)
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

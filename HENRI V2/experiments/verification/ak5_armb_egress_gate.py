"""A-K5 / ARM B EGRESS GATE -- the wave->text path the operator named.

WHAT THIS ANSWERS
    The operator's words: "wave-to-text egress currently yields
    top1_token_unique = 1 across 16 distinct waves". The recorded remedy file
    (g4_armu_unbinder_diag.py) reproduces that count. It does NOT answer the
    question that matters: is the count a DEFECT IN THE MECHANISM, or a
    property of an UNTRAINED artefact?

    Those are different findings with different fixes. The first needs a new
    egress design. The second needs a training run. Reporting either as the
    other would be the surrogate failure this project rejects.

THE DISCRIMINATOR (D0)
    Compare the loaded checkpoint tensors against a freshly constructed module
    at the same shapes. If the checkpoint's down_proj / lm_head statistics are
    indistinguishable from fresh init, no training signal reached the head.
    Then top1_token_unique is trivially expected and proves nothing about the
    mechanism. That must be established BEFORE any egress verdict.

TAUGHT BY M1 (measured 2026-10-03)
    Distinct-top-1 count is VACUOUS as a semantic metric. The pre-registered
    M1 gate returned VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE: the random-wave
    arm scored 0.59-0.70 distinct while the treatment scored 0.12-0.23.
    THE NEGATIVE CONTROL BEAT THE TREATMENT.
    So this gate does NOT score distinctness. It scores logit-space SEPARATION
    against a random-vector control, and it reports the top-1 margin above the
    runner-up (the argmax is beta-invariant; the MARGIN carries the signal).

SCOPE
    Read-only. SELECT only. No optimizer step. No save. Local CPU. $0.
    Checkpoint: 799034119 B, sha256 75572389083455a371546b40500b6614
    (identical in aaii-v43 and zone-a-selfplay).

CONTROL AMENDED (2026-10-03) -- evidence-based, disclosed.
    The first run used ONLY a random-Gaussian control. That is not
    structure-matched: Gaussian vectors are not in the codec wave family, so
    "real waves differ from noise" would be a weak claim either way. The
    PRIMARY control is now the SAME-FAMILY CONTENT-DESTROYED wave: permute the
    8192 blocks per sample (destroy_content). That preserves every block vector
    and the unit norm exactly, and destroys only the arrangement that carries
    content. Same defect class as the A-K4 contaminated-A2 finding.

PRE-REGISTERED (frozen before the first run)
    D0  ARTIFACT TRAINED?   ||W_ckpt - W_fresh|| / ||W_ckpt|| must exceed 0.5
                            for BOTH down_proj and lm_head.
                            If not -> verdict UNTRAINED_ARTEFACT and STOP:
                            no egress claim is admissible either way.
    Q1  SEPARATION          mean pairwise logit cosine distance between
                            DISTINCT real waves must exceed BOTH controls by
                            >= 0.10 absolute. PRIMARY control = same-family
                            content-destroyed (block-shuffled) waves.
    Q2  MARGIN              top-1 minus runner-up logit, mean over real waves,
                            must exceed BOTH controls by >= 0.10 absolute.
    Q3  DETERMINISM         repeat forward is bit-exact.
    Q4  SENSITIVITY         seeded per-block rotation must move the top-1
                            margin by a nonzero amount for some seed
                            (proves the wave reaches the head).
    Verdict PASS only if D0 and Q1 and Q2 and Q3 and Q4 pass.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

CKPT_REL = "HENRI V2/models/henri_decoder_checkpoint.pt"
D_MODEL, D_HIDDEN, VOCAB = 65536, 2048, 32000
N_RANDOM = 12
SEEDS = (20261002, 20261003, 20261004)


def repo_root() -> str:
    return os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def git_short_head(root: str) -> str:
    """Record the checkout SHA in the receipt (self-describing artifact)."""
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def wave_bytes_to_tensor(b: bytes) -> torch.Tensor:
    """Stored payload -> unit [1, 65536]. Layout [8192, 8] float32."""
    w = np.frombuffer(b, dtype=np.float32).copy()
    flat = torch.from_numpy(w).reshape(-1, D_MODEL).to(torch.float32)
    n = flat.norm(dim=-1, keepdim=True) + 1e-12
    return flat / n


def rotor(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(8192, 8, 8, generator=g)
    Q, _ = torch.linalg.qr(Q)
    return Q


def apply_rot(wave: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    q = Q.to(wave.dtype)
    out = torch.einsum("nkj,kij->nki", wave.reshape(wave.shape[0], 8192, 8), q)
    return out.reshape(wave.shape[0], D_MODEL)


def fetch_waves(dsn: str):
    """Distinct real Zone C waves. corpus_chunks first, then engrams."""
    import psycopg
    conn = psycopg.connect(dsn, connect_timeout=15)
    cur = conn.cursor()
    ids, waves = [], []
    try:
        cur.execute(
            "SELECT chunk_id, wave_payload FROM corpus_chunks "
            "ORDER BY chunk_id LIMIT 16")
        for cid, payload in cur.fetchall():
            if payload is None:
                continue
            if len(payload) != D_MODEL * 4:
                continue
            ids.append(cid)
            waves.append(wave_bytes_to_tensor(payload))
    except Exception as exc:  # noqa: BLE001
        print("corpus_chunks query failed: %s" % type(exc).__name__)
    try:
        cur.execute(
            "SELECT id::text, engram_wave_bytes FROM phylogenetic_engrams_65536 "
            "WHERE octet_length(engram_wave_bytes) = %s ORDER BY id LIMIT 16",
            (D_MODEL * 4,))
        for eid, payload in cur.fetchall():
            ids.append("engram:" + str(eid))
            waves.append(wave_bytes_to_tensor(payload))
    except Exception as exc:  # noqa: BLE001
        print("engram query failed: %s" % type(exc).__name__)
    cur.close()
    conn.close()
    return ids, waves


def destroy_content(waves: torch.Tensor, seed: int) -> torch.Tensor:
    """Same-family content-destroyed control: permute the 8192 blocks.

    Preserves every block vector and the unit norm exactly. Destroys only the
    arrangement that carries content. This is the structure-matched control --
    a random Gaussian vector is NOT in the codec wave family.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = waves.reshape(waves.shape[0], 8192, 8)
    out = torch.empty_like(B)
    for i in range(B.shape[0]):
        out[i] = B[i][torch.randperm(8192, generator=g)]
    return out.reshape(waves.shape[0], D_MODEL)


def pairwise_cos_dist(logits: torch.Tensor) -> float:
    """Mean pairwise cosine DISTANCE between rows. 0 => identical directions."""
    z = logits - logits.mean(dim=-1, keepdim=True)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
    gram = z @ z.t()
    n = gram.shape[0]
    off = (gram.sum() - gram.diagonal().sum()) / (n * (n - 1))
    return float(1.0 - off)


def margin(top2: torch.Tensor) -> torch.Tensor:
    """top1 minus runner-up, per row."""
    v, _ = torch.topk(top2, k=2, dim=-1)
    return v[:, 0] - v[:, 1]


def main() -> int:
    t0 = time.time()
    root = repo_root()
    ckpt_path = os.path.join(root, CKPT_REL)
    out = {
        "gate": "A-K5_ARM_B_EGRESS",
        "commit": git_short_head(root),
        "ckpt_path": CKPT_REL,
        "ckpt_sha256_prefix": None,
        "device": "cpu",
        "strip_flag": os.environ.get("HENRI_STRIP_DISCRETE_EGRESS", "UNSET"),
        "seeds": list(SEEDS),
    }

    if not os.path.exists(ckpt_path):
        print("MISSING CHECKPOINT %s" % ckpt_path)
        return 2
    out["ckpt_bytes"] = os.path.getsize(ckpt_path)
    import hashlib as _hl
    _h = _hl.sha256()
    with open(ckpt_path, "rb") as _f:
        for _b in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_b)
    out["ckpt_sha256_prefix"] = _h.hexdigest()[:32]

    dsn = (os.environ.get("ZONE_C_PROD_DSN")
           or os.environ.get("K5_TZCSM_TEST_DSN") or "")
    if not dsn:
        print("NO_DSN")
        return 1
    out["dsn_set"] = True

    ids, waves = fetch_waves(dsn)
    out["n_real_waves"] = len(ids)
    out["wave_ids"] = ids[:16]
    if len(waves) < 4:
        print("INSUFFICIENT_WAVES %d (need >=4)" % len(waves))
        print(json.dumps(out, indent=2))
        return 1
    real = torch.cat(waves, dim=0)  # [N, 65536]

    # ---- load the artifact, and a FRESH module at identical shapes --------
    from henri_decoder import HENRINeuralEgressUnbinder
    torch.manual_seed(0)
    fresh = HENRINeuralEgressUnbinder(d_model=D_MODEL, d_hidden=D_HIDDEN,
                                      vocab_size=VOCAB, device="cpu")
    fresh_sd = {k: v.detach().clone() for k, v in fresh.state_dict().items()}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    out["ckpt_keys"] = sorted(ckpt.keys())
    unb = HENRINeuralEgressUnbinder(d_model=D_MODEL, d_hidden=D_HIDDEN,
                                    vocab_size=VOCAB, device="cpu")
    missing, unexpected = unb.load_state_dict(ckpt, strict=False)
    out["missing_keys"] = list(missing)
    out["unexpected_keys"] = list(unexpected)
    unb.eval()

    # ---- D0: was any training signal written into the head? --------------
    d0 = {}
    for name in ("down_proj.weight", "lm_head.weight", "layer_norm.weight"):
        if name not in ckpt:
            d0[name] = {"present": False}
            continue
        w = ckpt[name].to(torch.float32)
        f = fresh_sd.get(name)
        if f is None or f.shape != w.shape:
            d0[name] = {"present": True, "fresh_comparable": False,
                        "ckpt_std": round(float(w.std()), 6)}
            continue
        f = f.to(torch.float32)
        denom = float(w.norm()) + 1e-12
        d0[name] = {
            "present": True,
            "shape": list(w.shape),
            "ckpt_std": round(float(w.std()), 8),
            "fresh_std": round(float(f.std()), 8),
            "rel_delta": round(float((w - f).norm()) / denom, 8),
            "near_identical_to_fresh": bool(
                float((w - f).norm()) / denom < 1.0e-3),
        }
    out["D0_artifact"] = d0

    trained = []
    for name in ("down_proj.weight", "lm_head.weight"):
        rec = d0.get(name, {})
        if rec.get("present") and "rel_delta" in rec:
            trained.append(rec["rel_delta"] > 0.5)
    d0_pass = bool(trained) and all(trained)
    out["D0_trained"] = d0_pass
    if not d0_pass:
        out["verdict"] = "UNINFORMATIVE_UNTRAINED_ARTEFACT:D0"
        out["note"] = ("Checkpoint tensors are indistinguishable from fresh "
                       "init. top1_token_unique=1 is then TRIVIALLY expected. "
                       "No egress-mechanism claim is admissible. The remedy is "
                       "a training run, not an egress redesign.")
        out["elapsed_s"] = round(time.time() - t0, 2)
        print(json.dumps(out, indent=2))
        return 0

    # ---- Q1/Q2: separation and margin, real vs BOTH controls -------------
    # PRIMARY control = same-family content-destroyed (block-shuffled).
    # SECONDARY control = random Gaussian (NOT same-family; weaker evidence).
    g = torch.Generator(device="cpu").manual_seed(SEEDS[0])
    rand = torch.randn(N_RANDOM, D_MODEL, generator=g)
    rand = rand / (rand.norm(dim=-1, keepdim=True) + 1e-12)
    shuf = destroy_content(real, SEEDS[0])

    with torch.no_grad():
        lg_real = unb(real)
        lg_rand = unb(rand)
        lg_shuf = unb(shuf)
        lg_real_2 = unb(real)               # Q3 determinism
        idx = lg_real.argmax(dim=-1)
        out["reproduced_top1_token_unique"] = int(len(set(idx.tolist())))
        out["reproduced_top1_tokens"] = [int(v) for v in idx.tolist()]

        q1_real = pairwise_cos_dist(lg_real)
        q1_shuf = pairwise_cos_dist(lg_shuf)
        q1_rand = pairwise_cos_dist(lg_rand)
        q1_ok = (q1_real - q1_shuf) >= 0.10 and (q1_real - q1_rand) >= 0.10
        out["Q1_separation"] = {
            "real": round(q1_real, 6),
            "shuffled_same_family_PRIMARY": round(q1_shuf, 6),
            "random_gaussian_secondary": round(q1_rand, 6),
            "delta_vs_shuffled": round(q1_real - q1_shuf, 6),
            "delta_vs_random": round(q1_real - q1_rand, 6),
            "threshold": 0.10, "pass": bool(q1_ok),
        }
        m_real = float(margin(lg_real).mean())
        m_shuf = float(margin(lg_shuf).mean())
        m_rand = float(margin(lg_rand).mean())
        q2_ok = (m_real - m_shuf) >= 0.10 and (m_real - m_rand) >= 0.10
        out["Q2_margin"] = {
            "real": round(m_real, 6),
            "shuffled_same_family_PRIMARY": round(m_shuf, 6),
            "random_gaussian_secondary": round(m_rand, 6),
            "delta_vs_shuffled": round(m_real - m_shuf, 6),
            "delta_vs_random": round(m_real - m_rand, 6),
            "threshold": 0.10, "pass": bool(q2_ok),
        }
        out["Q3_determinism"] = {
            "bit_exact": bool(torch.equal(lg_real, lg_real_2)),
        }

        sens = []
        for s in SEEDS:
            Q = rotor(s)
            lg_r = unb(apply_rot(real, Q))
            sens.append(round(float((margin(lg_r) - margin(lg_real)).abs().mean()), 6))
        out["Q4_rotation_margin_shift"] = sens
        out["Q4_pass"] = bool(any(v > 0.0 for v in sens))

    q1p = out["Q1_separation"]["pass"]
    q2p = out["Q2_margin"]["pass"]
    q3p = out["Q3_determinism"]["bit_exact"]
    q4p = out["Q4_pass"]
    if q1p and q2p and q3p and q4p:
        out["verdict"] = "AK5_EGRESS_PASS:DISTINCT_WAVES_SEPARATE_ABOVE_CONTROL"
    else:
        fails = []
        if not q1p:
            fails.append("Q1_separation")
        if not q2p:
            fails.append("Q2_margin")
        if not q3p:
            fails.append("Q3_determinism")
        if not q4p:
            fails.append("Q4_rotation_insensitive")
        out["verdict"] = "AK5_EGRESS_FAIL:" + "+".join(fails)

    out["elapsed_s"] = round(time.time() - t0, 2)
    print(json.dumps(out, indent=2))

    outp = os.path.join(os.environ.get("TEMP", "/tmp"), "ak5_egress_receipt.json")
    out["receipt_sha256"] = __import__("hashlib").sha256(
        json.dumps(out, sort_keys=True).encode("utf-8")).hexdigest()
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("RECEIPT=%s" % outp)
    print("RECEIPT_SHA256=%s" % out["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

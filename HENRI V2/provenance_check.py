"""Teacher-artifact provenance: is teacher_embeddings.pt the backbone's own
embed_tokens.weight? Decisive, garble-immune (file output + canonical hashing).

Context: config tie_word_embeddings=True and the shard has NO lm_head.weight, so
the output head IS embed_tokens. If teacher_embeddings.pt equals embed_tokens,
then the E1/E2/E3 "frozen teacher readout" was the backbone's own output head,
not an independent artifact.

Tests
  1. fp32-canonical bitwise equality (exact)
  2. teacher -> bf16 vs embed bitwise  (is the teacher the bf16 roundtrip?)
  3. embed  -> fp32 vs teacher bitwise (is the teacher the fp32 original?)
  4. max/mean abs diff + fraction of elements differing above thresholds
  5. NEGATIVE CONTROL: same tensor vs a seeded random tensor of equal shape
     (a test that cannot distinguish must be rejected)
  6. canonical SHA-256 of both tensors in fp32 bytes
  7. file-level SHA-256 of teacher_embeddings.pt and of the shard

Writes JSON to --out; nothing is printed that must be parsed by eye.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def sha_bytes(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    shard = "/root/e4c/qwen05b/model.safetensors"
    tpath = "/root/e1-calib/teacher_embeddings.pt"
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/e4c/prov.json")

    st = load_file(shard)
    keys = [k for k in st if "embed" in k or "lm_head" in k or "wte" in k]
    emb = st["model.embed_tokens.weight"]

    pay = torch.load(tpath, map_location="cpu", weights_only=True)
    if isinstance(pay, dict):
        E = pay.get("weight", pay)
    else:
        E = pay
    E = torch.as_tensor(E)

    rec: dict = {
        "shard": shard,
        "teacher_path": tpath,
        "shard_embed_like_keys": keys,
        "lm_head_present": any("lm_head" in k for k in st),
        "teacher_dtype": str(E.dtype),
        "embed_dtype": str(emb.dtype),
        "teacher_shape": list(E.shape),
        "embed_shape": list(emb.shape),
        "shape_equal": list(E.shape) == list(emb.shape),
    }
    if list(E.shape) != list(emb.shape):
        rec["VERDICT"] = "SHAPE_MISMATCH"
        out.write_text(json.dumps(rec, indent=2))
        print("PROV_VERDICT=SHAPE_MISMATCH")
        return

    Ef = E.detach().float().contiguous()
    ef = emb.detach().float().contiguous()

    # --- 1. fp32-canonical exact equality --------------------------------
    rec["fp32_bitwise_equal"] = bool(torch.equal(Ef, ef))
    rec["fp32_n_diff"] = int((Ef != ef).sum())
    # --- 2/3. roundtrip tests -------------------------------------------
    rec["teacher_to_bf16_equals_embed"] = bool(torch.equal(E.to(torch.bfloat16), emb))
    rec["teacher_to_bf16_n_diff"] = int((E.to(torch.bfloat16) != emb).sum())
    rec["embed_to_fp32_equals_teacher"] = bool(torch.equal(ef, Ef))
    # --- 4. diffs --------------------------------------------------------
    d = (Ef - ef).abs()
    rec["max_abs_diff"] = float(d.max())
    rec["mean_abs_diff"] = float(d.mean())
    rec["n_total"] = int(d.numel())
    rec["n_gt_1e-8"] = int((d > 1e-8).sum())
    rec["n_gt_1e-6"] = int((d > 1e-6).sum())
    rec["frac_gt_1e-6"] = round(float((d > 1e-6).float().mean()), 6)
    # --- 5. NEGATIVE CONTROL --------------------------------------------
    g = torch.Generator().manual_seed(7)
    R = torch.randn(Ef.shape, generator=g)
    dr = (Ef - R).abs()
    rec["control_random_max_abs_diff"] = float(dr.max())
    rec["control_random_mean_abs_diff"] = float(dr.mean())
    rec["control_discriminates"] = bool(dr.mean() > d.mean() * 10)
    # --- 6. canonical hashes --------------------------------------------
    rec["sha_teacher_fp32"] = sha_bytes(Ef)
    rec["sha_embed_fp32"] = sha_bytes(ef)
    rec["sha_teacher_fp32_eq_embed"] = rec["sha_teacher_fp32"] == rec["sha_embed_fp32"]
    rec["sha_teacher_as_bf16_bytes"] = hashlib.sha256(
        E.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    rec["sha_embed_as_bf16_bytes"] = hashlib.sha256(
        emb.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    # --- 7. file hashes --------------------------------------------------
    rec["file_sha_teacher_pt"] = sha_file(tpath)
    rec["file_sha_shard"] = sha_file(shard)
    # magnitude context
    rec["teacher_absmax"] = float(Ef.abs().max())
    rec["embed_absmax"] = float(ef.abs().max())

    # --- verdict ---------------------------------------------------------
    if rec["fp32_bitwise_equal"]:
        v = "BITWISE_IDENTICAL_FP32"
    elif rec["teacher_to_bf16_equals_embed"] or rec["embed_to_fp32_equals_teacher"]:
        v = "SAME_TENSOR_UP_TO_BF16_ROUNDING"
    elif rec["frac_gt_1e-6"] > 0.999 and rec["max_abs_diff"] < 0.02:
        v = "NEAR_IDENTICAL_BF16_SCALE_DIFF"
    elif rec["control_discriminates"] and rec["max_abs_diff"] > 0.1:
        v = "INDEPENDENT_TENSOR"
    else:
        v = "INCONCLUSIVE"
    rec["VERDICT"] = v
    rec["VERDICT_BASIS"] = (
        "fp32_bitwise_equal=%s teacher_to_bf16_equals_embed=%s "
        "max_abs_diff=%.6f frac_gt_1e-6=%.4f control_mean=%.4f"
        % (rec["fp32_bitwise_equal"], rec["teacher_to_bf16_equals_embed"],
           rec["max_abs_diff"], rec["frac_gt_1e-6"],
           rec["control_random_mean_abs_diff"]))

    out.write_text(json.dumps(rec, indent=2))
    print(f"PROV_VERDICT={v}")
    print(f"PROV_WROTE={out}")


if __name__ == "__main__":
    main()

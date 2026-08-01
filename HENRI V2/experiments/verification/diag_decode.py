"""Diagnostic: inspect checkpoint state dict + decode first 8 MBPP items, print RAW model output."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/mbpp-pilot/HENRI V2")

import mbpp_heldout_pilot as P
from henri_decoder import HENRIUnifiedEgressTransducer
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

CKPT = "/workspace/HENRI V2/HENRI V2/models/henri_decoder_checkpoint.pt"


def flat(v):
    if torch.is_tensor(v):
        return float(v.item()) if v.numel() == 1 else list(v.flatten()[:4].tolist())
    return v


# --- checkpoint state dict ---
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
print("CKPT_TYPE:", type(ckpt).__name__)
if isinstance(ckpt, dict):
    print("TOP_KEYS:", list(ckpt.keys())[:20])
    sd = ckpt.get("model_state_dict", ckpt)
    if isinstance(sd, dict):
        sk = list(sd.keys())
        print("STATE_KEYS:", len(sk))
        for k in sk[:40]:
            t = sd[k]
            print("  ", k, tuple(t.shape) if hasattr(t, "shape") else type(t).__name__)
        n = sum(sd[k].numel() for k in sk if hasattr(sd[k], "numel"))
        print("TOTAL_PARAMS:", n)

# --- decode real items ---
manifest, items = P.validate_static_bundle()
print("N_ITEMS:", len(items), "FIRST_IDS:", [int(i["task_id"]) for i in items[:8]])

provenance = P.load_json(P.CHECKPOINT_PROVENANCE_PATH)
ckpt_sha = P.checkpoint_preflight(Path(CKPT), provenance)
print("CKPT_SHA:", ckpt_sha)

t = HENRIUnifiedEgressTransducer(d_model=65536, device="cuda", checkpoint_path=CKPT)
codec = qFHRREpistemicCodec(d_model=65536, device="cuda")

for item in items[:8]:
    tid = int(item["task_id"])
    prompt = P.render_prompt(item)
    pw = codec.encode_text(prompt)
    to = codec.encode_text("MBPP_CODING_OPERATOR")
    gw = codec.bind_hadamard(to, pw)
    try:
        resp, tele = t.decode_wave_to_response(gw, prompt)
        print(f"--- ITEM {tid} ---")
        print("RESPONSE_HEAD:", repr(resp[:400]))
        print("TELE:", json.dumps({k: flat(v) for k, v in (tele or {}).items()})[:600])
    except Exception as exc:
        print(f"--- ITEM {tid} ERROR: {type(exc).__name__}: {exc}")

print("DIAG_DONE")

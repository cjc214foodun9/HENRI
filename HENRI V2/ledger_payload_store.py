"""K0 carrier: content-addressed payload sidecar for the T0 transition ledger.

T0 rows persist digests only. K0 adds canonical payload persistence so the
ledger corpus can feed Koopman dynamics (K1/K2). Payloads are stored once
per digest under <root>/<digest>.bin with <digest>.json metadata, keyed by
the SAME canonical digest functions as the ledger (wave_digest /
action_digest). Contract: payload bytes reproduce the recorded digest
(sha256(raw) == digest for every kind); missing or corrupt references
fail closed.

Default-OFF: HENRI_LEDGER_PAYLOADS=1 must be set. A ledger used WITHOUT a
store emits rows byte-identical to the T0 format (differential contract).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from temporal_transition_ledger import action_digest, wave_digest

FLAG = "HENRI_LEDGER_PAYLOADS"
SCHEMA = "payload.v1"


class PayloadStoreDisabledError(RuntimeError):
    pass


class PayloadReferenceError(RuntimeError):
    pass


def encode_payload(obj: Any) -> Dict[str, Any]:
    """Canonical (kind, raw_bytes, meta, digest) with digest == sha256(raw)."""
    import torch
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous().to(torch.float32)
        return {"kind": "tensor", "raw": t.numpy().tobytes(),
                "meta": {"shape": list(t.shape), "dtype": "float32"},
                "digest": wave_digest(obj)}
    if isinstance(obj, (list, dict)):
        raw = json.dumps(obj, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
        return {"kind": "grid", "raw": raw, "meta": {"schema": "grid-json"},
                "digest": wave_digest(obj)}
    if hasattr(obj, "name"):
        data = getattr(obj, "data", None)
        base = f"{type(obj).__name__}:{obj.name}"
        if data is not None:
            base += f":{json.dumps(data, sort_keys=True, default=str)}"
        return {"kind": "action", "raw": base.encode("utf-8"),
                "meta": {"schema": "gameaction-string", "name": obj.name},
                "digest": action_digest(obj)}
    raw = str(obj).encode("utf-8")
    return {"kind": "text", "raw": raw, "meta": {"schema": "text"},
            "digest": action_digest(obj)}


class LedgerPayloadStore:
    """Content-addressed, once-per-digest payload store (fail-closed)."""

    def __init__(self, root_dir: str | Path, *, flag: str = FLAG):
        if os.environ.get(flag, "0") != "1":
            raise PayloadStoreDisabledError(
                f"{flag} is not set; payload persistence is default-OFF")
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, obj: Any) -> Dict[str, Any]:
        enc = encode_payload(obj)
        digest = enc["digest"]
        bin_path = self.root / f"{digest}.bin"
        meta_path = self.root / f"{digest}.json"
        if not bin_path.exists():
            bin_path.write_bytes(enc["raw"])
        if not meta_path.exists():
            meta_path.write_text(json.dumps(
                {"schema": SCHEMA, "kind": enc["kind"], "digest": digest,
                 "bytes": len(enc["raw"]), **enc["meta"]},
                sort_keys=True), encoding="utf-8")
        return {"digest": digest, "kind": enc["kind"], "ref": f"{digest}.bin"}

    def get(self, digest: str) -> bytes:
        bin_path = self.root / f"{digest}.bin"
        if not bin_path.exists():
            raise PayloadReferenceError(
                f"payload reference {digest[:12]} is missing")
        data = bin_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise PayloadReferenceError(
                f"payload {digest[:12]} is corrupt (digest mismatch)")
        return data

    def ref_exists(self, digest: str) -> bool:
        return (self.root / f"{digest}.bin").exists()

    def get_decoded(self, digest: str):
        """Decode a stored payload to (kind, object) using its sidecar meta."""
        meta_path = self.root / f"{digest}.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kind = meta.get("kind", "text")
        raw = self.get(digest)
        if kind == "grid":
            return kind, json.loads(raw.decode("utf-8"))
        if kind == "tensor":
            import numpy as np
            import torch
            arr = np.frombuffer(raw, dtype=np.float32).reshape(meta["shape"])
            return kind, torch.from_numpy(arr.copy())
        if kind == "action":
            # raw format: "<type>:<name>:<json data>"
            text = raw.decode("utf-8")
            head, _, rest = text.partition(":")
            name, _, data_s = rest.partition(":")
            data = json.loads(data_s) if data_s else None
            return kind, {"type": head, "name": name, "data": data}
        return kind, raw.decode("utf-8")

    def count(self) -> int:
        return len(list(self.root.glob("*.bin")))

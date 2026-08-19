# -*- coding: utf-8 -*-
"""
Phase 8.32 — Authorized ARC trajectory bank (default-OFF).

Captures live (o_t, a_t, o_t+1) tuples from production ARC runs when
HENRI_ARC_TRAJECTORY_BANK=1. Tuples originate ONLY from the public ARC
arcade environment during a live run -> data_source="authorized" by
construction. This bank is the legitimate calibration input for
henri_calibrated_action_head.py. It NEVER reads evaluation caches,
solution files, or reconstructed labels.

Storage: float16 on CPU (memory-safe staging; D=65,536 -> 128 KB/wave),
npz payload + human-readable jsonl meta + sealed manifest (digests).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

BANK_SCHEMA_ID = "henri.arc-trajectory-bank.v1"
DEFAULT_ACTION_NAMES = [f"ACTION{i}" for i in range(1, 7)]
MAX_DEFAULT_RECORDS = 50_000


class TrajectoryBankError(RuntimeError):
    """Typed failure for the trajectory bank (fail-closed callers)."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bank_enabled_from_env(env: Optional[Dict[str, str]] = None) -> bool:
    """Default-OFF gate: only the exact '1' enables the bank."""
    env = os.environ if env is None else env
    return env.get("HENRI_ARC_TRAJECTORY_BANK", "0") == "1"


def filter_onehot_to_vocab(
    onehot: np.ndarray,
    bank_vocab: List[str],
    target_vocab: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only rows whose action belongs to target_vocab, reordered to it.

    onehot [M, len(bank_vocab)] uint8. Returns (filtered [M', len(target)],
    kept_mask [M] bool). Raises if no record survives.
    """
    target_idx = [bank_vocab.index(n) for n in target_vocab]
    kept = onehot[:, target_idx].sum(axis=1) > 0
    if not bool(kept.any()):
        raise TrajectoryBankError(
            f"no records in target vocab {target_vocab} "
            f"(bank vocab {bank_vocab})")
    return onehot[kept][:, target_idx].astype(np.float32), kept


class TrajectoryBank:
    """In-memory accumulator flushed to npz + jsonl + manifest on run end."""

    def __init__(
        self,
        log_dir: str,
        run_id: str = "",
        provenance: str = "",
        action_names: Optional[List[str]] = None,
        max_records: int = MAX_DEFAULT_RECORDS,
        store_next_wave: bool = True,
    ) -> None:
        if not os.path.isdir(log_dir):
            raise TrajectoryBankError(f"log_dir missing: {log_dir}")
        self.log_dir = log_dir
        self.run_id = run_id or f"run-{int(time.time())}"
        self.provenance = (
            provenance or f"arc-live-run {self.run_id} (authorized)")
        self.action_names: List[str] = list(action_names or DEFAULT_ACTION_NAMES)
        self.max_records = max_records
        self.store_next_wave = store_next_wave
        self._waves: List[np.ndarray] = []
        self._next_waves: List[np.ndarray] = []
        self._action_idx: List[int] = []
        self._meta: List[Dict[str, Any]] = []
        self._truncated = False

    # -- capture ---------------------------------------------------------
    def record(
        self,
        state_wave: torch.Tensor,
        action_name: Optional[str],
        meta: Optional[Dict[str, Any]] = None,
        next_wave: Optional[torch.Tensor] = None,
    ) -> None:
        """Append one authorized (o_t, a_t, o_t+1) tuple.

        Unknown action names auto-extend the bank vocab (append-only);
        calibration-time filtering decides which names are eligible.
        """
        if len(self._waves) >= self.max_records:
            self._truncated = True
            return
        if not action_name:
            raise TrajectoryBankError("action_name required")
        if action_name not in self.action_names:
            self.action_names.append(action_name)
        w = (
            state_wave.detach().to("cpu", dtype=torch.float16)
            .numpy().astype(np.float16)
        )
        if w.ndim != 1:
            w = w.reshape(-1)
        self._waves.append(w)
        if next_wave is not None and self.store_next_wave:
            nw = (
                next_wave.detach().to("cpu", dtype=torch.float16)
                .numpy().astype(np.float16)
            )
            if nw.ndim != 1:
                nw = nw.reshape(-1)
            self._next_waves.append(nw)
        self._action_idx.append(self.action_names.index(action_name))
        self._meta.append(dict(meta or {}, action_name=action_name,
                               t=time.time()))

    # -- flush -----------------------------------------------------------
    def flush(self) -> Dict[str, Any]:
        """Write npz + jsonl + manifest. Returns summary dict."""
        M = len(self._waves)
        if M == 0:
            raise TrajectoryBankError("bank is empty; nothing to flush")
        psi = np.stack(self._waves).astype(np.float16)          # [M, D]
        nxt = (
            np.stack(self._next_waves).astype(np.float16)
            if self._next_waves else None
        )
        A = len(self.action_names)
        onehot = np.zeros((M, A), dtype=np.uint8)
        for i, idx in enumerate(self._action_idx):
            onehot[i, idx] = 1

        stem = f"trajectories_{self.run_id}"
        npy_path = os.path.join(self.log_dir, f"{stem}.npz")
        jsonl_path = os.path.join(self.log_dir, f"{stem}.jsonl")
        manifest_path = os.path.join(self.log_dir, f"{stem}_manifest.json")

        # Explicit empty [0, D] array when no next-wave was captured;
        # np.savez stores Python None as a 0-d OBJECT array, which breaks
        # load() casts and corrupts the digest (observed in unit tests).
        next_arr = (
            nxt if nxt is not None
            else np.zeros((0, psi.shape[1]), dtype=np.float16)
        )
        np.savez(npy_path, psi=psi, next_wave=next_arr,
                 actions_onehot=onehot,
                 action_names=np.array(self.action_names))
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in self._meta:
                f.write(json.dumps(rec, default=str) + "\n")

        envs = sorted({str(m.get("env", "?")) for m in self._meta})
        digest_bytes = psi.tobytes() + onehot.tobytes()
        if nxt is not None:
            digest_bytes += nxt.tobytes()
        dataset_digest = _sha256_bytes(digest_bytes)
        with open(npy_path, "rb") as f:
            npz_sha = _sha256_bytes(f.read())
        with open(jsonl_path, "rb") as f:
            jsonl_sha = _sha256_bytes(f.read())

        manifest = {
            "schema_id": BANK_SCHEMA_ID,
            "version": "1",
            "run_id": self.run_id,
            "provenance": self.provenance,
            "data_source": "authorized",
            "record_count": int(M),
            "wave_dim": int(psi.shape[1]),
            "action_vocab": list(self.action_names),
            "envs": envs,
            "store_next_wave": bool(self.store_next_wave),
            "truncated": bool(self._truncated),
            "dataset_digest": dataset_digest,
            "npz_sha256": npz_sha,
            "jsonl_sha256": jsonl_sha,
            "timestamp": time.time(),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, default=str)

        return {
            "records": int(M),
            "npz_path": npy_path,
            "jsonl_path": jsonl_path,
            "manifest_path": manifest_path,
            "dataset_digest": dataset_digest,
            "npz_sha256": npz_sha,
            "jsonl_sha256": jsonl_sha,
        }

    # -- load ------------------------------------------------------------
    @classmethod
    def load(
        cls,
        npz_path: str,
        manifest_path: Optional[str] = None,
        verify_digest: bool = True,
    ) -> Dict[str, Any]:
        """Load bank; verify dataset digest when manifest present."""
        if not os.path.isfile(npz_path):
            raise TrajectoryBankError(f"npz missing: {npz_path}")
        z = np.load(npz_path, allow_pickle=False)
        try:
            psi = z["psi"].astype(np.float32)
            onehot = z["actions_onehot"].astype(np.float32)
            bank_vocab = [str(n) for n in z["action_names"].tolist()]
            nxt = z["next_wave"]
            if nxt.ndim != 2:
                raise TrajectoryBankError("next_wave must be [M, D] or empty [0, D]")
            if nxt.shape[0] == 0:
                nxt = None
            else:
                nxt = nxt.astype(np.float32)

            manifest: Dict[str, Any] = {}
            if manifest_path and os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if verify_digest and manifest.get("dataset_digest"):
                    digest_bytes = psi.astype(np.float16).tobytes() + onehot.astype(np.uint8).tobytes()
                    if nxt is not None:
                        digest_bytes += nxt.astype(np.float16).tobytes()
                    if _sha256_bytes(digest_bytes) != manifest["dataset_digest"]:
                        raise TrajectoryBankError("dataset digest mismatch")
            return {
                "psi": psi,            # [M, D] float32
                "next_wave": nxt,      # [M, D] float32 | None
                "actions_onehot": onehot,  # [M, A] float32
                "action_vocab": bank_vocab,
                "manifest": manifest,
            }
        finally:
            z.close()  # release the npz memmap on ALL paths (Windows file-lock)

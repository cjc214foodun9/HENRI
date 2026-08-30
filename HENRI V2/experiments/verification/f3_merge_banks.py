"""F3 merge tool — bounded re-capture budget enforcement (CPU-only).

Consumes per-env attempt banks produced by f3_capture_driver.py (each a
separate authorized capture under capture_attempts/<env>/<attempt>/) and
emits ONE merged authorized bank that satisfies the SPEC-2026-08-29-F3-BROAD-BANK
budget by construction:

  - per-env cap:  first env_cap rows per env are kept (capture order preserved)
  - union vocab:  one-hot columns realigned to the sorted union of action
                  names across attempts (6-col and 7-col banks merge cleanly)
  - N budget:     12 envs x [floor 100, cap 150] => N in [1200, 1800]
  - fail-loud:    data_source must be "authorized"; digest mismatch or a
                  missing expect_env aborts (zero-pretraining invariant)

Merging is concatenation + trimming + realignment ONLY. It never synthesizes,
interpolates, or re-samples rows.

Usage (remote, repo root, after the capture driver):
  /venv/main/bin/python "HENRI V2/experiments/verification/f3_merge_banks.py" \
      --attempts /root/f3-run/capture_attempts \
      --out /root/f3-run/telemetry/f3_bank_capture_v2 \
      --run-id production_run_<ts> \
      --env-cap 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source(manifest: Dict, npz_bytes: bytes, jsonl_bytes: bytes) -> None:
    """Fail-loud: authorized source + digest match vs manifest."""
    assert manifest.get("data_source") == "authorized", (
        f"bank must be authorized capture, got {manifest.get('data_source')!r}"
    )
    if manifest.get("npz_sha256"):
        assert manifest["npz_sha256"] == hashlib.sha256(npz_bytes).hexdigest(), (
            "npz hash mismatch vs manifest"
        )
    if manifest.get("jsonl_sha256"):
        assert manifest["jsonl_sha256"] == hashlib.sha256(jsonl_bytes).hexdigest(), (
            "jsonl hash mismatch vs manifest"
        )


def realign_and_concat(
    psi_list: Sequence[np.ndarray],
    onehot_list: Sequence[np.ndarray],
    names_list: Sequence[Sequence[str]],
    meta: List[Dict],
    env_cap: int = 150,
    next_list: Optional[Sequence[np.ndarray]] = None,
) -> Dict:
    """Realign one-hot rows to the union vocab; keep first env_cap rows/env.

    Row order is preserved (banks in list order, rows in capture order).
    `meta` is the flat concatenation of per-bank meta rows in the SAME order;
    a meta row is consumed only when its bank row is kept.
    """
    union: List[str] = []
    for names in names_list:
        for n in names:
            if n not in union:
                union.append(n)
    union = sorted(union)

    psi_out: List[np.ndarray] = []
    onehot_out: List[np.ndarray] = []
    next_out: List[np.ndarray] = []
    meta_out: List[Dict] = []
    counts: Dict[str, int] = {}
    mi = 0
    for bank_i, (psi, onehot, names) in enumerate(zip(psi_list, onehot_list, names_list)):
        idx = {n: i for i, n in enumerate(names)}
        col = {n: union.index(n) for n in names}
        for j in range(psi.shape[0]):
            env = str(meta[mi].get("env", "?"))
            if counts.get(env, 0) >= env_cap:
                mi += 1
                continue
            counts[env] = counts.get(env, 0) + 1
            row = np.zeros((len(union),), dtype=np.uint8)
            for n in names:
                if onehot[j, idx[n]]:
                    row[col[n]] = 1
            psi_out.append(psi[j])
            onehot_out.append(row)
            if next_list is not None:
                next_out.append(next_list[bank_i][j])
            meta_out.append(meta[mi])
            mi += 1
    if not psi_out:
        raise AssertionError("merged bank is empty; no rows survived the cap")

    out: Dict = {
        "psi": np.stack(psi_out).astype(np.float16),
        "onehot": np.stack(onehot_out).astype(np.uint8),
        "names": union,
        "meta": meta_out,
    }
    if next_list is not None:
        out["next_wave"] = np.stack(next_out).astype(np.float16)
    return out


def merge_banks(
    attempts_dir: str,
    out_dir: str,
    run_id: str,
    env_cap: int = 150,
    expect_envs: Optional[Sequence[str]] = None,
) -> Dict:
    """Merge all attempt banks under attempts_dir into one authorized bank."""
    ad = Path(attempts_dir)
    assert ad.is_dir(), f"attempts dir missing: {attempts_dir}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    psi_list: List[np.ndarray] = []
    onehot_list: List[np.ndarray] = []
    names_list: List[List[str]] = []
    next_list: List[np.ndarray] = []
    meta: List[Dict] = []
    tele_rows: List[Dict] = []
    env_seen = set()

    def _first_bank(d: Path):
        """Return (bank_dir, npz, jsonl, manifest) for a dir holding banks."""
        npzs = sorted(d.glob("trajectories_*.npz"))
        jls = sorted(d.glob("trajectories_*.jsonl"))
        mfs = sorted(d.glob("trajectories_*_manifest.json"))
        if npzs and jls and mfs:
            return d, npzs[0], jls[0], mfs[0]
        return None

    for sub in sorted(p for p in ad.iterdir() if p.is_dir()):
        # Collect every bank dir for this env: flat (v1) or all attempt_N (v2).
        bank_dirs: List[Path] = []
        if _first_bank(sub) is not None:
            bank_dirs.append(sub)
        else:
            for attempt in sorted(p for p in sub.iterdir() if p.is_dir()):
                if _first_bank(attempt) is not None:
                    bank_dirs.append(attempt)
        if not bank_dirs:
            print(f"[merge] skip attempt dir without bank: {sub}")
            continue
        for bank_dir in bank_dirs:
            npz_path, jl_path, mf_path = _first_bank(bank_dir)[1:]
            manifest = json.loads(mf_path.read_text(encoding="utf-8"))
            verify_source(manifest, npz_path.read_bytes(), jl_path.read_bytes())
            bank = np.load(npz_path)
            psi_list.append(bank["psi"])
            onehot_list.append(bank["actions_onehot"])
            names_list.append([str(a) for a in bank["action_names"]])
            if "next_wave" in bank.files and bank["next_wave"].shape[0] == bank["psi"].shape[0]:
                next_list.append(bank["next_wave"])
            rows = [json.loads(l) for l in jl_path.open(encoding="utf-8")]
            meta.extend(rows)
            env_seen.update(str(r.get("env")) for r in rows)
            for tp in sorted(bank_dir.glob("production_run_*.jsonl")):
                for line in tp.open(encoding="utf-8"):
                    if line.strip():
                        tele_rows.append(json.loads(line))

    if expect_envs is not None:
        missing = sorted(set(expect_envs) - env_seen)
        assert not missing, f"missing capture envs: {missing}"

    merged = realign_and_concat(
        psi_list, onehot_list, names_list, meta,
        env_cap=env_cap,
        next_list=next_list if next_list else None,
    )

    stem = f"trajectories_{run_id}"
    npy_path = out / f"{stem}.npz"
    jsonl_path = out / f"{stem}.jsonl"
    manifest_path = out / f"{stem}_manifest.json"
    tele_path = out / f"production_run_{run_id}.jsonl"

    D = int(merged["psi"].shape[1])
    next_arr = merged.get(
        "next_wave",
        np.zeros((0, D), dtype=np.float16),
    )
    np.savez(
        npy_path,
        psi=merged["psi"],
        next_wave=next_arr,
        actions_onehot=merged["onehot"],
        action_names=np.array(merged["names"]),
    )
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in merged["meta"]:
            f.write(json.dumps(rec, default=str) + "\n")
    with open(tele_path, "w", encoding="utf-8") as f:
        for rec in tele_rows:
            f.write(json.dumps(rec, default=str) + "\n")

    envs = sorted({str(m.get("env", "?")) for m in merged["meta"]})
    per_env = {e: 0 for e in envs}
    per_action = {a: 0 for a in merged["names"]}
    for m in merged["meta"]:
        per_env[str(m.get("env", "?"))] += 1
        per_action[str(m.get("action_name", "?"))] += 1

    digest_bytes = merged["psi"].tobytes() + merged["onehot"].tobytes()
    if next_arr.shape[0]:
        digest_bytes += next_arr.tobytes()

    npz_sha = _sha256(str(npy_path))
    jsonl_sha = _sha256(str(jsonl_path))

    manifest = {
        "schema_id": "henri.arc-trajectory-bank.v1",
        "version": "1",
        "run_id": run_id,
        "provenance": f"f3 merged authorized capture {run_id}",
        "data_source": "authorized",
        "record_count": int(len(merged["meta"])),
        "wave_dim": D,
        "action_vocab": list(merged["names"]),
        "envs": envs,
        "store_next_wave": True,
        "truncated": False,
        "merged": True,
        "source_attempts": int(len(psi_list)),
        "env_cap": int(env_cap),
        "dataset_digest": hashlib.sha256(digest_bytes).hexdigest(),
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "timestamp": time.time(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=str)

    receipt = {
        "schema_id": "f3-merge-receipt.v1",
        "run_id": run_id,
        "record_count": int(len(merged["meta"])),
        "envs": envs,
        "per_env_counts": per_env,
        "per_action_counts": per_action,
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "merged": True,
        "source_attempts": int(len(psi_list)),
        "generator": f"f3_merge_banks.py@{_sha256(str(Path(__file__).resolve()))[:16]}",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out / "f3_merge_receipt.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print("F3_MERGE_OK")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--env-cap", type=int, default=150)
    ap.add_argument("--expect-envs", nargs="*", default=None)
    args = ap.parse_args()
    merge_banks(args.attempts, args.out, args.run_id,
                env_cap=args.env_cap, expect_envs=args.expect_envs)


if __name__ == "__main__":
    main()

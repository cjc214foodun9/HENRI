"""Default-OFF live-planner action-wave manifest exporter.

Purpose
-------
Capture the exact action waves produced at the live planner selection
boundary (HenriSwarmOrchestrator.plan_action -> candidate_action_waves)
and emit a provenance manifest consumable by
koopman_corpus_runner.validate_action_wave_manifest
(origin == LIVE_ORIGIN == "live_planner_boundary").

This is the fail-closed provenance source for K1/K2 real-corpus verdicts:
absent or placeholder waves -> BLOCKED_MISSING_PRODUCTION_ACTION_WAVES
in the corpus runner. The exporter itself never constructs a fit.

Default-OFF
-----------
HENRI_ACTION_WAVE_EXPORT=1 is required. When off, the module is never
imported by the live loop and the production path is byte-identical.

Environment
-----------
HENRI_ACTION_WAVE_EXPORT=1
HENRI_ACTION_WAVE_EXPORT_DIR=<out dir for .npy waves + action_waves.json>
HENRI_RUN_ID=<run id>          (fallback "unknown")
HENRI_COMMIT_SHA=<commit sha>  (fallback: git rev-parse HEAD at call time)

Frozen-learning invariant
-------------------------
Under HENRI_FREEZE_LEARNING=1 an action's wave must not change between
recordings. A digest change for the same action name raises RuntimeError
(fail loud) so a corpus built from a mutating planner is never exported.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

FLAG = "HENRI_ACTION_WAVE_EXPORT"
DIR_ENV = "HENRI_ACTION_WAVE_EXPORT_DIR"
RUN_ID_ENV = "HENRI_RUN_ID"
COMMIT_ENV = "HENRI_COMMIT_SHA"
ORIGIN = "live_planner_boundary"
BLOCK_DIM = 8
REQUIRED_PROVENANCE = frozenset({
    "path", "source", "commit", "run_id", "episode", "step",
    "shape", "dtype", "normalization", "encoder", "basis",
    "digest", "origin",
})


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


class ActionWaveExporter:
    """Collect (action_name -> wave) with full provenance; write manifest."""

    _instance: Optional["ActionWaveExporter"] = None

    def __init__(self, out_dir: str, run_id: str, commit: str,
                 episode: str = "", step: Optional[int] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.commit = commit
        self.episode = episode
        self.step = step
        self._waves: Dict[str, Path] = {}
        self._entries: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get(cls) -> Optional["ActionWaveExporter"]:
        if os.environ.get(FLAG, "0") != "1":
            return None
        if cls._instance is None:
            out_dir = os.environ.get(DIR_ENV)
            if not out_dir:
                raise RuntimeError(f"{FLAG}=1 requires {DIR_ENV}=<dir>")
            cls._instance = cls(
                out_dir=out_dir,
                run_id=os.environ.get(RUN_ID_ENV, "unknown"),
                commit=os.environ.get(COMMIT_ENV, _git_head()),
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def set_context(self, episode: str, step: int) -> None:
        """Optional per-step context from the harness (audit only)."""
        self.episode = episode
        self.step = step

    def record(self, action_name: str, wave: torch.Tensor, *, encoder: str,
               source: str, normalization: str = "per_block_l2",
               basis: str = "cl3x0_blocks") -> None:
        w = wave.detach().cpu().to(torch.float32)
        if w.dim() != 2 or w.shape[-1] != BLOCK_DIM:
            raise ValueError(
                f"action wave must be [*, {BLOCK_DIM}] real, "
                f"got {tuple(w.shape)}")
        if action_name in self._waves:
            old = np.load(self._waves[action_name])
            if not np.array_equal(old, w.numpy()):
                raise RuntimeError(
                    f"action {action_name!r} wave changed across recordings; "
                    "learning must be frozen during corpus export")
            return
        rel = f"{action_name}.npy"
        path = self.out_dir / rel
        np.save(path, w.numpy())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self._waves[action_name] = path
        self._entries[action_name] = {
            "path": str(path), "source": source, "commit": self.commit,
            "run_id": self.run_id, "episode": self.episode,
            "step": self.step, "shape": list(w.shape), "dtype": "float32",
            "normalization": normalization, "encoder": encoder,
            "basis": basis, "digest": digest, "origin": ORIGIN,
        }
        # Crash-safe sidecar: persists the full entry so a later process
        # (the corpus launcher) can rebuild the manifest from disk.
        (self.out_dir / f"{action_name}.json").write_text(
            json.dumps(self._entries[action_name], indent=2, sort_keys=True),
            encoding="utf-8")

    def write_manifest(self, path: Optional[str] = None) -> str:
        target = Path(path) if path else self.out_dir / "action_waves.json"
        target.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True),
            encoding="utf-8")
        return str(target)

    @classmethod
    def finalize_manifest(cls, out_dir: str,
                          path: Optional[str] = None) -> str:
        """Rebuild action_waves.json from on-disk per-action sidecars.

        The exporter records in one process (the replay); the corpus
        launcher runs in another process, so the in-memory manifest does
        not survive. finalize_manifest reads the {action}.json sidecars
        written by record(), verifies each .npy still matches its digest,
        and writes the merged manifest. Raises on missing wave files or
        digest mismatches (fail-loud).
        """
        d = Path(out_dir)
        entries: Dict[str, Dict[str, Any]] = {}
        for sidecar in sorted(d.glob("*.json")):
            if sidecar.name == "action_waves.json":
                continue
            name = sidecar.stem
            entry = json.loads(sidecar.read_text(encoding="utf-8"))
            wave_path = Path(entry["path"])
            if not wave_path.exists():
                raise FileNotFoundError(
                    f"{name}: wave file missing: {wave_path}")
            if hashlib.sha256(wave_path.read_bytes()).hexdigest() != \
                    entry["digest"]:
                raise RuntimeError(
                    f"{name}: digest mismatch on disk for {wave_path}")
            entries[name] = entry
        if not entries:
            raise RuntimeError(f"no action sidecars found in {d}")
        target = Path(path) if path else d / "action_waves.json"
        target.write_text(
            json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
        return str(target)

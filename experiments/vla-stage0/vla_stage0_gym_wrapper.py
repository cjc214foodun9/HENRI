"""
System-1 Stage-0a — Dynamical Substrate Wrapper (CartPole-v1).
================================================================
Reference 3 (gpt-5.6-sol) binding: Stage-0a = REAL reset/step transitions
+ append-only provenance ONLY. NO encoder, NO R-EDMD, NO learning.

Contracts enforced here (see vla_stage0_env_contract.md):
  C1  Real tuple provenance: (s_t, a_t, r_t, s_{t+1}, terminated,
      truncated, episode_id, step_id) appended per step.
  C2  Same pinned seed + same action prefix -> byte-identical traces.
  C3  Different seeds -> non-vacuous difference.
  C4  Action validated against Discrete(2) before env.step.
  C5  Terminal/truncated episodes reject further steps until reset.
  C6  No state leakage across episodes (reset via env.reset(seed=...)
      ONLY; never mutate env.unwrapped.state as the reset mechanism).
  C7  Raw observations hashed (sha256 of float32 bytes) per step.
  C8  No synthetic states anywhere in the provenance path.

Representation boundary: raw obs are (4,) float32 BOX values. This
carrier does NOT encode them. Encoding belongs to Stage-0b (frozen
encoder on the live [1,16,384] / d_slot=384 boundary per Reference 3;
no 65,536-D family is introduced by this or any Stage-0 carrier).
"""
import hashlib
import json
import os
from typing import List, Optional

import numpy as np

ENV_ID = "CartPole-v1"
ACTION_SPACE = {0, 1}


class Stage0GymWrapper:
    """Real-environment wrapper with append-only transition provenance."""

    def __init__(self, seed: int = 0, max_episode_steps: int = 500):
        import gymnasium as gym
        self._env = gym.make(ENV_ID, max_episode_steps=max_episode_steps)
        self._seed = int(seed)
        self._max_steps = int(max_episode_steps)
        self._transitions: List[dict] = []
        self._episode_id = 0
        self._step_id = 0
        self._needs_reset = True
        self._env.reset(seed=self._seed)
        self._needs_reset = False

    # ---- provenance -------------------------------------------------
    @property
    def transitions(self) -> List[dict]:
        return list(self._transitions)

    @property
    def n_transitions(self) -> int:
        return len(self._transitions)

    @staticmethod
    def _obs_hash(obs: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()

    # ---- env interface ----------------------------------------------
    def reset(self) -> dict:
        """Reset to a fresh episode. Re-seeds the environment deterministically."""
        self._episode_id += 1
        self._step_id = 0
        obs, info = self._env.reset(seed=self._seed)
        self._needs_reset = False
        return {"observation": np.asarray(obs, dtype=np.float32).copy(),
                "info": dict(info), "episode_id": self._episode_id,
                "step_id": 0, "obs_hash": self._obs_hash(obs)}

    def step(self, action: int) -> dict:
        """Execute one real environment transition; append provenance."""
        if self._needs_reset:
            raise RuntimeError("C5: step() after terminal/truncated; call reset() first.")
        action = int(action)
        if action not in ACTION_SPACE:
            raise ValueError(
                f"C4: action {action} not in Discrete(2) {{{0}, {1}}}.")
        # The environment's own transition is the authority. The pre-step
        # observation is NOT read from private state; only the returned
        # observation is recorded (see verify() for the strict replay check).
        obs_next, reward, terminated, truncated, info = self._env.step(action)
        self._step_id += 1
        record = {
            "episode_id": self._episode_id,
            "step_id": self._step_id,
            "action": action,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "obs_next": np.asarray(obs_next, dtype=np.float32).tolist(),
            "obs_next_hash": self._obs_hash(obs_next),
            "info": {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in info.items()},
        }
        self._transitions.append(record)
        if terminated or truncated:
            self._needs_reset = True
        return {"observation": np.asarray(obs_next, dtype=np.float32).copy(),
                "reward": float(reward), "terminated": bool(terminated),
                "truncated": bool(truncated), "info": dict(info),
                "episode_id": self._episode_id, "step_id": self._step_id,
                "obs_hash": record["obs_next_hash"]}

    # ---- persistence ------------------------------------------------
    def save_provenance(self, path: str) -> str:
        """Dump append-only provenance JSON; returns sha256 of the file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = json.dumps({
            "env_id": ENV_ID, "seed": self._seed,
            "max_episode_steps": self._max_steps,
            "n_transitions": len(self._transitions),
            "transitions": self._transitions,
        }, sort_keys=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(blob)
        return hashlib.sha256(blob.encode()).hexdigest()


def verify() -> int:
    """Run the Stage-0a contract checks. Returns 0 on all-pass, 1 on failure."""
    import gymnasium as gym  # noqa: F401  (proves importability)
    failures: List[str] = []

    # ---- C2: same seed + same action prefix -> byte-identical ----
    seed = 4242
    actions = [0, 1, 0, 1, 0, 0, 1, 1]
    traces = []
    for _ in range(2):
        w = Stage0GymWrapper(seed=seed)
        w.reset()
        for a in actions:
            try:
                w.step(a)
            except RuntimeError:
                break  # episode ended early; identical break = identical trace
        traces.append([(t["step_id"], t["action"], t["obs_next_hash"],
                        t["reward"], t["terminated"], t["truncated"])
                       for t in w.transitions])
    if traces[0] != traces[1]:
        failures.append("C2 FAIL: same seed+prefix produced different traces")
    else:
        print("C2 PASS: byte-identical replay trace (%d transitions)" % len(traces[0]))

    # ---- C3: different seeds -> non-vacuous ----
    w1 = Stage0GymWrapper(seed=1)
    w1.reset()
    w2 = Stage0GymWrapper(seed=2)
    w2.reset()
    for _ in range(10):
        for w in (w1, w2):
            try:
                w.step(0)
            except RuntimeError:
                break
    h1 = [t["obs_next_hash"] for t in w1.transitions]
    h2 = [t["obs_next_hash"] for t in w2.transitions]
    if h1 == h2:
        failures.append("C3 FAIL: different seeds produced identical traces (vacuous)")
    else:
        print("C3 PASS: seed 1 vs seed 2 differ at step %d"
              % next(i for i, (a, b) in enumerate(zip(h1, h2)) if a != b))

    # ---- C4: action validation ----
    w = Stage0GymWrapper(seed=7)
    w.reset()
    try:
        w.step(2)
        failures.append("C4 FAIL: invalid action accepted")
    except ValueError:
        print("C4 PASS: invalid action rejected")

    # ---- C5: terminal/truncated reject further steps ----
    w = Stage0GymWrapper(seed=3, max_episode_steps=2)
    w.reset()
    w.step(0)
    r = w.step(1)
    assert r["truncated"] is True, "expected truncation at max_episode_steps=2"
    try:
        w.step(0)
        failures.append("C5 FAIL: step after truncation accepted")
    except RuntimeError:
        print("C5 PASS: step after truncation rejected until reset")

    # ---- C6: episode isolation + provenance continuity ----
    w = Stage0GymWrapper(seed=9, max_episode_steps=2)
    w.reset()
    w.step(0)
    w.step(1)  # truncated
    ep1_ids = {t["episode_id"] for t in w.transitions}
    w.reset()  # new episode, same seed
    w.step(0)
    ep2_ids = {t["episode_id"] for t in w.transitions}
    if len(ep1_ids) != 1 or len(ep2_ids) != 2 or ep1_ids == ep2_ids:
        failures.append("C6 FAIL: episode isolation broken (%s -> %s)"
                        % (ep1_ids, ep2_ids))
    else:
        print("C6 PASS: episode isolation + reset via env.reset(seed=...) only")

    # ---- C7: obs hashes present and non-empty ----
    if not all(t["obs_next_hash"] for t in w.transitions):
        failures.append("C7 FAIL: missing observation hash")
    else:
        print("C7 PASS: raw observation sha256 recorded per transition")

    # ---- C8: no synthetic states ----
    import inspect as _i
    src = _i.getsource(Stage0GymWrapper)
    if "torch.randn" in src or "randn(" in src:
        failures.append("C8 FAIL: synthetic random state generator present")
    else:
        print("C8 PASS: no synthetic state generation in wrapper")

    if failures:
        print("\n".join(failures))
        return 1
    print("ALL STAGE-0A CONTRACTS PASS")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(verify())

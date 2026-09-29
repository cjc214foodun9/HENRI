"""HENRI VLA task: pixel-observation continuous-control environment.

WHY THIS EXISTS
===============
The project needs a LEARNING TASK with an EXTERNAL OUTCOME to measure. ARC-AGI
supplies 0 demonstrations through its public API (measured this session), so
Path B is starved of data. Until a real robot corpus is attached, this gives the
whole stack a measurable closed-loop number: SUCCESS RATE.

WHAT THIS IS, HONESTLY
======================
A synthetic 2-D planar reaching/pushing task rendered to PIXELS, with continuous
7-DoF actions. It is an INFRASTRUCTURE SCAFFOLD. A success rate here proves the
learning loop closes -- vision -> wave -> policy -> continuous action -> world ->
reward -> gradient. It is NOT a claim about robot manipulation, and no result
from this environment is transferable evidence about ARC, LIBERO or any
benchmark. That limit is stated, not implied away.

DESIGN
======
  Observation : [3, H, W] float in [0,1] -- a rendered scene, RGB channels used
                for (end-effector, target, goal-mask) so a single Conv/ingress
                stack sees the full state.
  Action      : 7 DoF (dx, dy, dz, droll, dpitch, dyaw, gripper) in [-1, 1].
                Only dx, dy affect the 2-D plant; the other 5 are accepted and
                carried so the interface matches the VLA action space.
  Reward      : dense negative distance + terminal success bonus.
  Success     : ||ee - goal|| < tol at any step.
  Episodes    : fixed horizon, deterministic seeds, so runs are COMPARABLE.

The expert is analytic (move toward the goal). Behaviour cloning on expert
trajectories gives a supervised signal; closed-loop rollout then measures
SUCCESS RATE, which is the external outcome.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

N_DOF = 7


@dataclass
class EnvConfig:
    size: int = 64
    horizon: int = 48
    step_size: float = 0.08
    tol: float = 0.06
    dot_radius: int = 3
    # Task regions for CONTINUAL LEARNING: Task A restricts the goal to the LEFT
    # half, Task B to the RIGHT half. Disjoint supports are what make a forgetting
    # measurement meaningful -- a shared support would let a policy score on both
    # without retaining anything.
    goal_region: str = "any"          # "any" | "left" | "right"


class PixelReachEnv:
    """Deterministic pixel-observation reaching task."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.n_dof = N_DOF
        self.reset(0)

    # ------------------------------------------------------------------ core
    def reset(self, seed: int = 0):
        g = torch.Generator().manual_seed(int(seed))
        s = self.cfg.size
        # GOAL sampling respects the task region (continual learning).
        lo, hi = 0.25, 0.75
        if self.cfg.goal_region == "left":
            hi = 0.5
        elif self.cfg.goal_region == "right":
            lo = 0.5

        def draw():
            return lo + (hi - lo) * float(torch.rand(1, generator=g))

        self.ee = torch.tensor([0.25 + 0.5 * float(torch.rand(1, generator=g)),
                                0.25 + 0.5 * float(torch.rand(1, generator=g))])
        self.goal = torch.tensor([draw(), draw()])
        # start far enough apart that random play does NOT succeed trivially
        tries = 0
        while float((self.ee - self.goal).norm()) < 0.30 and tries < 64:
            tries += 1
            self.ee = torch.tensor([0.25 + 0.5 * float(torch.rand(1, generator=g)),
                                    0.25 + 0.5 * float(torch.rand(1, generator=g))])
            self.goal = torch.tensor([draw(), draw()])
        self.t = 0
        self.done = False
        self.success = False
        _ = s
        return self.observe()

    def observe(self) -> torch.Tensor:
        """[3, H, W] float in [0,1]."""
        s = self.cfg.size
        dev = self.ee.device

        def blob(px: float, py: float, r: int) -> torch.Tensor:
            yy, xx = torch.meshgrid(
                torch.arange(s, device=dev, dtype=torch.float32),
                torch.arange(s, device=dev, dtype=torch.float32), indexing="ij")
            cx, cy = px * (s - 1), py * (s - 1)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            return torch.exp(-d2 / (2.0 * float(r) ** 2))

        img = torch.zeros(3, s, s, device=dev)
        img[0] = blob(float(self.ee[0]), float(self.ee[1]), self.cfg.dot_radius)
        img[1] = blob(float(self.goal[0]), float(self.goal[1]), self.cfg.dot_radius)
        # channel 2: normalized distance field -- gives a global gradient cue so
        # the policy is not required to solve credit assignment from two dots
        yy, xx = torch.meshgrid(
            torch.arange(s, device=dev, dtype=torch.float32),
            torch.arange(s, device=dev, dtype=torch.float32), indexing="ij")
        d = torch.sqrt(((xx / (s - 1)) - float(self.goal[0])) ** 2 +
                       ((yy / (s - 1)) - float(self.goal[1])) ** 2)
        img[2] = 1.0 - d.clamp(0, 1)
        return img.clamp(0, 1)

    def expert_action(self) -> torch.Tensor:
        """Analytic expert: unit step toward the goal, other 5 DoF zero."""
        v = self.goal - self.ee
        n = v.norm()
        if float(n) > 1e-9:
            v = v / n
        a = torch.zeros(self.n_dof, device=self.ee.device)
        a[0], a[1] = v[0], v[1]
        return a

    def step(self, action: torch.Tensor):
        """Returns (obs, reward, done, info). action: [n_dof] in [-1,1]."""
        if self.done:
            return self.observe(), 0.0, True, {"success": self.success}
        a = torch.as_tensor(action, dtype=torch.float32, device=self.ee.device).reshape(-1)
        if a.numel() < self.n_dof:
            a = torch.cat([a, torch.zeros(self.n_dof - a.numel(), device=a.device)])
        delta = a[:2].clamp(-1.0, 1.0) * self.cfg.step_size
        self.ee = (self.ee + delta).clamp(0.0, 1.0)
        self.t += 1
        dist = float((self.ee - self.goal).norm())
        self.success = dist < self.cfg.tol
        self.done = self.success or self.t >= self.cfg.horizon
        reward = -dist + (5.0 if self.success else 0.0)
        return self.observe(), reward, self.done, {"success": self.success, "dist": dist}

    # ------------------------------------------------------------- utilities
    def expert_episode(self, seed: int):
        """One expert (demonstration) episode: list of (obs, action)."""
        self.reset(seed)
        traj = []
        while not self.done:
            obs = self.observe()
            act = self.expert_action()
            traj.append((obs, act))
            self.step(act)
        return traj


def random_policy_success(n_episodes: int = 64, seed: int = 0) -> float:
    """Baseline: success rate of uniformly random actions. The number the
    learner must beat for 'learning happened' to mean anything."""
    env = PixelReachEnv()
    wins = 0
    g = torch.Generator().manual_seed(seed + 991)
    for i in range(n_episodes):
        env.reset(seed + i)
        done = False
        while not done:
            a = (torch.rand(env.n_dof, generator=g) * 2 - 1)
            _, _, done, info = env.step(a)
        wins += int(info["success"])
    return wins / n_episodes


def expert_success(n_episodes: int = 64, seed: int = 0) -> float:
    env = PixelReachEnv()
    wins = 0
    for i in range(n_episodes):
        env.reset(seed + i)
        done = False
        while not done:
            _, _, done, info = env.step(env.expert_action())
        wins += int(info["success"])
    return wins / n_episodes


if __name__ == "__main__":
    out = {
        "random_policy_success_rate": round(random_policy_success(), 4),
        "analytic_expert_success_rate": round(expert_success(), 4),
        "obs_shape": list(PixelReachEnv().observe().shape),
        "n_dof": N_DOF,
        "horizon": PixelReachEnv().cfg.horizon,
    }
    print(json.dumps(out, indent=2))

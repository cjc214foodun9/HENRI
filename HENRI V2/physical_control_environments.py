"""
Multi-Modal Closed-Loop Physical Control Environments for Project HENRI V2.

Implements continuous dynamical control benchmarks coupled directly to
UniversalEgress and Clifford phase multivectors:
  1. InvertedPendulumEnvironment: Continuous torque control u(t) in [-1, 1]
  2. CartPolePhysicsEnvironment: Continuous force control F(t) in [-10, 10]
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


class InvertedPendulumEnvironment:
    """
    Continuous Non-Linear Inverted Pendulum ODE environment:
      theta'' = (g/l) * sin(theta) + (1/(m*l^2)) * u(t) - b * theta'
    """

    def __init__(self, dt: float = 0.02, g: float = 9.81, l: float = 1.0, m: float = 1.0, b: float = 0.1):
        self.dt = dt
        self.g = g
        self.l = l
        self.m = m
        self.b = b
        self.reset()

    def reset(self, theta_init: float = 0.1, dtheta_init: float = 0.0):
        self.theta = theta_init
        self.dtheta = dtheta_init
        return self._get_state_array()

    def step(self, torque: float):
        u = float(np.clip(torque, -2.0, 2.0))
        ddtheta = (self.g / self.l) * math.sin(self.theta) + (1.0 / (self.m * self.l**2)) * u - self.b * self.dtheta
        self.dtheta += ddtheta * self.dt
        self.theta += self.dtheta * self.dt
        # Keep theta in [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        state = self._get_state_array()
        # Homeostatic viability cost (upright position theta=0, low velocity)
        cost = self.theta**2 + 0.1 * self.dtheta**2 + 0.001 * u**2
        done = abs(self.theta) > math.pi / 2.0
        return state, cost, done

    def _get_state_array(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta), self.dtheta], dtype=np.float32)

    def state_to_wave(self, num_blocks: int, device: torch.device) -> torch.Tensor:
        """Encodes state array into [num_blocks, 8] Clifford wave."""
        st = self._get_state_array()
        seed_val = int(abs(st[0] * 1000 + st[1] * 2000 + st[2] * 3000)) % 100000
        g = torch.Generator(device="cpu").manual_seed(seed_val)
        w = torch.randn(num_blocks, 8, generator=g).to(device)
        return F.normalize(w, p=2, dim=-1)


class CartPolePhysicsEnvironment:
    """
    Continuous CartPole ODE Environment:
      x'' = (F + m_p*l*(theta'^2 * sin(theta) - theta''*cos(theta))) / (m_c + m_p)
    """

    def __init__(self, dt: float = 0.02, g: float = 9.81, m_c: float = 1.0, m_p: float = 0.1, l: float = 0.5):
        self.dt = dt
        self.g = g
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.total_mass = m_c + m_p
        self.polemass_length = m_p * l
        self.reset()

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        return self.state

    def step(self, force: float):
        x, x_dot, theta, theta_dot = self.state
        f = float(np.clip(force, -10.0, 10.0))

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (f + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.g * sintheta - costheta * temp) / (self.l * (4.0 / 3.0 - self.m_p * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        done = bool(x < -2.4 or x > 2.4 or theta < -12 * (math.pi / 180.0) or theta > 12 * (math.pi / 180.0))
        cost = theta**2 + 0.1 * theta_dot**2 + 0.01 * x**2
        return self.state, cost, done

    def state_to_wave(self, num_blocks: int, device: torch.device) -> torch.Tensor:
        st = self.state
        seed_val = int(abs(sum(st) * 10000.0)) % 100000
        g = torch.Generator(device="cpu").manual_seed(seed_val)
        w = torch.randn(num_blocks, 8, generator=g).to(device)
        return F.normalize(w, p=2, dim=-1)

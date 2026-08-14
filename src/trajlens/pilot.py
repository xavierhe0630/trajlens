"""
pilot.py — generates demonstrations for ReachAvoidEnv.

Honesty note (see README/AI_USAGE): I do not have a DualSense controller or
a robot in this sandbox, so the "teleoperator" here is a scripted
potential-field controller with injected human-like imperfection (reaction
lag, aim noise, occasional overcorrection) rather than a live human or a
learned policy. This is explicitly called out as a substitution, per the
assignment's allowance for "a simulator ... [when] you do not have
hardware". The RECORDER, SYNCHRONIZATION, SCHEMA, QUALITY-CONTROL and
ANNOTATION code around it is what's being demonstrated, and all of that is
agnostic to where the actions came from — a real DualSense trace would
plug into exactly the same `recorder.Recorder.step()` call.

Two pilot skill levels are provided so collected data has realistic
variance in quality (a real demo-collection process has this too):
  - "expert":   low noise, short reaction lag  -> mostly clean successes
  - "novice":   higher noise, longer lag, more overcorrection -> more
                failures / meandering trajectories / lower-value demos
"""
from __future__ import annotations
import numpy as np

from .env import ReachAvoidEnv, OBSTACLE_RADIUS, AGENT_RADIUS


class PotentialFieldPilot:
    """Attractive force toward goal, repulsive force away from obstacles,
    plus a first-order lag filter and Gaussian noise to emulate a human
    operator rather than a perfect controller."""

    def __init__(self, skill: str = "expert", seed: int = 0):
        assert skill in ("expert", "novice")
        self.skill = skill
        self.rng = np.random.default_rng(seed)
        self._last_action = np.zeros(2, dtype=np.float32)
        if skill == "expert":
            self.noise_std = 0.06
            self.lag = 0.15          # fraction of new command applied per step (higher = snappier)
            self.overcorrect_p = 0.01
        else:
            self.noise_std = 0.22
            self.lag = 0.45
            self.overcorrect_p = 0.08

    def act(self, env: ReachAvoidEnv) -> np.ndarray:
        to_goal = env.goal - env.pos
        d = np.linalg.norm(to_goal) + 1e-6
        force = to_goal / d

        for o in env.obstacles:
            away = env.pos - o
            dist = np.linalg.norm(away) + 1e-6
            margin = OBSTACLE_RADIUS + AGENT_RADIUS + 0.8
            if dist < margin:
                force += (away / dist) * (margin - dist) * 2.0

        force = force / (np.linalg.norm(force) + 1e-6)
        force += self.rng.normal(0, self.noise_std, size=2)

        if self.rng.random() < self.overcorrect_p:
            force = -force * self.rng.uniform(0.5, 1.5)

        cmd = self.lag * force + (1 - self.lag) * self._last_action
        cmd = np.clip(cmd, -1.0, 1.0).astype(np.float32)
        self._last_action = cmd
        return cmd

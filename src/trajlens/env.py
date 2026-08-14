"""
env.py — a small, deterministic 2D "reach-avoid" interactive task.

Why a custom env instead of e.g. gym's CarRacing / a real robot?
- The assignment is about the DATA PIPELINE (capture, sync, QC, annotation,
  export), not about environment design or control. A custom env keeps the
  observation/action/render surface completely transparent and lets us
  render frames with plain PIL (no GPU, no display server, fully headless),
  which matters a lot for a sandboxed take-home.
- It still has everything a real interactive-data-collection problem has:
  a continuous action space, a camera-like RGB observation stream, a
  proprioceptive vector stream, contact/failure events, and a notion of
  demonstration quality (efficient vs. wandering vs. failed).

State (proprioception):  [x, y, vx, vy, goal_x, goal_y]      (float32, 6)
Action (control):        [ax, ay]  thrust in x/y, clipped    (float32, 2)
Observation (vision):    96x96x3 RGB frame (agent, goal, obstacles)
Reward:                  dense shaping: -dist_to_goal - 0.01*|a| ; +10 on success
Episode ends:            success (reached goal), collision (failure), or timeout
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from PIL import Image, ImageDraw

FRAME_SIZE = 96
WORLD = 10.0          # world is a WORLD x WORLD square, origin at center
DT = 1.0 / 20.0        # 20 Hz control loop -> matches a plausible teleop rate
MAX_STEPS = 200
GOAL_RADIUS = 0.5
OBSTACLE_RADIUS = 0.6
AGENT_RADIUS = 0.25
MAX_ACCEL = 6.0
DAMPING = 0.90


def _world_to_px(pt: np.ndarray) -> tuple[float, float]:
    """map world coords in [-WORLD/2, WORLD/2] to pixel coords in [0, FRAME_SIZE]."""
    x, y = pt
    px = (x + WORLD / 2) / WORLD * FRAME_SIZE
    py = (y + WORLD / 2) / WORLD * FRAME_SIZE
    return px, FRAME_SIZE - py  # flip y for image coords


@dataclass
class ReachAvoidEnv:
    """Minimal, dependency-free (no gym) sequential decision task."""
    seed: int = 0
    n_obstacles: int = 3
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.pos = np.zeros(2, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles = np.zeros((self.n_obstacles, 2), dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------ core
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        lo, hi = -WORLD / 2 + 1, WORLD / 2 - 1
        self.pos = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        self.goal = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        while np.linalg.norm(self.goal - self.pos) < 3.0:
            self.goal = self.rng.uniform(lo, hi, size=2).astype(np.float32)
        obs = []
        for _ in range(self.n_obstacles):
            for _try in range(20):
                cand = self.rng.uniform(lo, hi, size=2).astype(np.float32)
                if (np.linalg.norm(cand - self.pos) > 1.5
                        and np.linalg.norm(cand - self.goal) > 1.5):
                    obs.append(cand)
                    break
            else:
                obs.append(cand)
        self.obstacles = np.array(obs, dtype=np.float32)
        self.vel[:] = 0.0
        self.t = 0
        return self.state()

    def state(self) -> np.ndarray:
        return np.concatenate([self.pos, self.vel, self.goal]).astype(np.float32)

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) * MAX_ACCEL
        self.vel = (self.vel + a * DT) * DAMPING
        self.pos = self.pos + self.vel * DT
        self.pos = np.clip(self.pos, -WORLD / 2, WORLD / 2)
        self.t += 1

        dist_goal = float(np.linalg.norm(self.pos - self.goal))
        success = dist_goal < GOAL_RADIUS
        collided = any(
            np.linalg.norm(self.pos - o) < (OBSTACLE_RADIUS + AGENT_RADIUS)
            for o in self.obstacles
        )
        timeout = self.t >= MAX_STEPS
        done = bool(success or collided or timeout)

        reward = -dist_goal * 0.1 - 0.01 * float(np.linalg.norm(a))
        if success:
            reward += 10.0
        if collided:
            reward -= 5.0

        info = {
            "success": success,
            "collided": collided,
            "timeout": timeout,
            "dist_goal": dist_goal,
        }
        return self.state(), reward, done, info

    # ---------------------------------------------------------------- render
    def render(self) -> np.ndarray:
        img = Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (24, 26, 32))
        draw = ImageDraw.Draw(img)

        gx, gy = _world_to_px(self.goal)
        r = GOAL_RADIUS / WORLD * FRAME_SIZE
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=(60, 200, 120))

        for o in self.obstacles:
            ox, oy = _world_to_px(o)
            r = OBSTACLE_RADIUS / WORLD * FRAME_SIZE
            draw.ellipse([ox - r, oy - r, ox + r, oy + r], fill=(210, 70, 70))

        ax, ay = _world_to_px(self.pos)
        r = AGENT_RADIUS / WORLD * FRAME_SIZE
        draw.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(80, 150, 240))
        # heading tick so replay video shows motion direction, not just position
        if np.linalg.norm(self.vel) > 1e-3:
            hd = self.vel / (np.linalg.norm(self.vel) + 1e-6)
            draw.line([ax, ay, ax + hd[0] * 8, ay - hd[1] * 8], fill=(255, 255, 255), width=2)

        return np.asarray(img, dtype=np.uint8)

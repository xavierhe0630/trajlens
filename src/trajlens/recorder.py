"""
recorder.py — synchronized capture of one demonstration episode.

Design decisions (the "what real learning loop would use this data, and how
do you preserve provenance / synchronize streams" question the assignment
asks):

1. SINGLE CONTROL-LOOP TIMESTAMP PER STEP.
   Real teleop/robotics stacks often have observation, action, and reward
   arriving from *different* processes with different clocks (camera driver,
   controller poll, reward/step function). Here I collapse them onto one
   authoritative step clock (the recorder's own monotonic step counter),
   which is the only thing that makes "synchronized" a meaningful, checkable
   property instead of an assumption. Every stream is indexed by the same
   integer step id AND a wall-clock timestamp, so downstream QC can detect
   drift (timestamps not evenly spaced) independently of frame indexing
   (steps skipped or duplicated).

2. RAW CAPTURE IS APPEND-ONLY AND UNTRUSTED.
   `Recorder` never filters, cleans, or judges data quality. It writes
   exactly what happened, including corrupted/duplicated/dropped frames if
   `corruption.py` injects them upstream of it (simulating a flaky camera
   or a controller USB hiccup). Quality control is a separate, auditable
   pass (see quality.py) over this raw log -- mixing capture and cleaning
   is a common real-world bug (you silently lose the ability to audit what
   your sensors actually did).

3. SCHEMA VERSIONING + PROVENANCE.
   Every episode file carries a `schema_version`, the git-describe-style
   `code_version` (a hash of this package's source at record time), the RNG
   seed, and the pilot/skill config used to generate it. This is what makes
   two sessions collected weeks apart, by different code, comparable or
   flaggable as incomparable.

On-disk layout per episode:
  raw/{session_id}/ep_{idx:04d}.npz     -- arrays: state, action, reward,
                                            done, step_id, t_wall, frame (uint8 stack)
  raw/{session_id}/ep_{idx:04d}.meta.json -- provenance + env/pilot config
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

SCHEMA_VERSION = "trajlens.raw.v1"


def _code_version(pkg_dir: Path) -> str:
    """Cheap provenance hash: content hash of all .py files in the package.
    Stands in for a git commit hash so raw sessions can be tied to the exact
    code that produced them even outside a git repo."""
    h = hashlib.sha256()
    for f in sorted(pkg_dir.glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


@dataclass
class EpisodeMeta:
    schema_version: str
    session_id: str
    episode_index: int
    code_version: str
    env_seed: int
    pilot_skill: str
    pilot_seed: int
    dt_nominal: float
    n_steps: int
    outcome: str          # "success" | "collision" | "timeout"
    wall_start: float
    wall_end: float
    corruption_profile: dict[str, Any]  # what synthetic capture noise was injected, if any


class Recorder:
    """Records one episode's synchronized streams into an in-memory buffer,
    then flushes to disk. Call reset_episode() -> record_step() * N -> save()."""

    def __init__(self, out_dir: str | Path, session_id: str):
        self.out_dir = Path(out_dir) / session_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._pkg_dir = Path(__file__).parent
        self._code_version = _code_version(self._pkg_dir)
        self._buf: dict[str, list] = {}
        self._episode_index = -1
        self._wall_start = None

    def reset_episode(self, episode_index: int):
        self._episode_index = episode_index
        self._buf = {"state": [], "action": [], "reward": [], "done": [],
                     "step_id": [], "t_wall": [], "frame": []}
        self._wall_start = time.time()

    def record_step(self, step_id: int, state: np.ndarray, action: np.ndarray,
                     reward: float, done: bool, frame: np.ndarray | None):
        """Append one synchronized tuple. `frame` may be None to simulate a
        dropped camera frame (a real capture-quality issue); `step_id` may
        skip or repeat to simulate dropped/duplicated control-loop ticks."""
        self._buf["state"].append(np.asarray(state, dtype=np.float32))
        self._buf["action"].append(np.asarray(action, dtype=np.float32))
        self._buf["reward"].append(float(reward))
        self._buf["done"].append(bool(done))
        self._buf["step_id"].append(int(step_id))
        self._buf["t_wall"].append(time.time())
        if frame is None:
            self._buf["frame"].append(np.zeros((1,), dtype=np.uint8))  # sentinel: dropped
        else:
            self._buf["frame"].append(np.asarray(frame, dtype=np.uint8))

    def save(self, outcome: str, env_seed: int, pilot_skill: str, pilot_seed: int,
              dt_nominal: float, corruption_profile: dict[str, Any] | None = None) -> Path:
        n = len(self._buf["step_id"])
        frames_shapes = {f.shape for f in self._buf["frame"]}
        real_frame_shape = next((s for s in frames_shapes if s != (1,)), (1,))

        stacked_frames = np.zeros((n, *real_frame_shape), dtype=np.uint8)
        frame_present = np.zeros(n, dtype=bool)
        for i, f in enumerate(self._buf["frame"]):
            if f.shape == real_frame_shape:
                stacked_frames[i] = f
                frame_present[i] = True

        path = self.out_dir / f"ep_{self._episode_index:04d}.npz"
        np.savez_compressed(
            path,
            state=np.stack(self._buf["state"]),
            action=np.stack(self._buf["action"]),
            reward=np.array(self._buf["reward"], dtype=np.float32),
            done=np.array(self._buf["done"], dtype=bool),
            step_id=np.array(self._buf["step_id"], dtype=np.int64),
            t_wall=np.array(self._buf["t_wall"], dtype=np.float64),
            frame=stacked_frames,
            frame_present=frame_present,
        )

        meta = EpisodeMeta(
            schema_version=SCHEMA_VERSION,
            session_id=self.session_id,
            episode_index=self._episode_index,
            code_version=self._code_version,
            env_seed=env_seed,
            pilot_skill=pilot_skill,
            pilot_seed=pilot_seed,
            dt_nominal=dt_nominal,
            n_steps=n,
            outcome=outcome,
            wall_start=self._wall_start,
            wall_end=time.time(),
            corruption_profile=corruption_profile or {},
        )
        meta_path = self.out_dir / f"ep_{self._episode_index:04d}.meta.json"
        meta_path.write_text(json.dumps(asdict(meta), indent=2))
        return path

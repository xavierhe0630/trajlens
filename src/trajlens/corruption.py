"""
corruption.py — simulates realistic capture-time faults so the quality
pipeline (quality.py) has real problems to find, and so this is honestly
demonstrated rather than asserted.

A real teleop/robotics rig commonly exhibits:
  - dropped frames        (camera driver hiccup -> frame missing this step)
  - duplicated frames      (camera outputs same buffer twice -> stale frame)
  - timestamp jitter/drift (USB scheduling, GC pause, OS scheduling)
  - duplicated step ids    (control loop double-fires)
  - NaN/inf in proprioception (sensor glitch)

`CorruptionProfile` is sampled once per episode and applied step-by-step
inside the collection loop (see scripts/run_demo.py), then written verbatim
into the episode's meta.json so QC results can be checked against ground
truth -- this is what lets me report QC precision/recall in the analysis
report instead of just asserting "it works".
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class CorruptionProfile:
    drop_frame_p: float = 0.0
    duplicate_frame_p: float = 0.0
    jitter_std_s: float = 0.0
    duplicate_step_p: float = 0.0
    sensor_glitch_p: float = 0.0

    def as_dict(self):
        return asdict(self)


CLEAN = CorruptionProfile()
FLAKY_CAMERA = CorruptionProfile(drop_frame_p=0.04, duplicate_frame_p=0.03, jitter_std_s=0.003)
FLAKY_CONTROLLER = CorruptionProfile(duplicate_step_p=0.03, jitter_std_s=0.02)
SENSOR_FAULT = CorruptionProfile(sensor_glitch_p=0.02, jitter_std_s=0.001)

PROFILES = {
    "clean": CLEAN,
    "flaky_camera": FLAKY_CAMERA,
    "flaky_controller": FLAKY_CONTROLLER,
    "sensor_fault": SENSOR_FAULT,
}


def sample_profile(rng: np.random.Generator) -> tuple[str, CorruptionProfile]:
    name = rng.choice(list(PROFILES.keys()), p=[0.55, 0.2, 0.15, 0.1])
    return name, PROFILES[name]


def maybe_corrupt_frame(frame: np.ndarray, prev_frame: np.ndarray | None,
                         profile: CorruptionProfile, rng: np.random.Generator):
    """Returns (frame_or_None, was_duplicate: bool)."""
    if rng.random() < profile.drop_frame_p:
        return None, False
    if prev_frame is not None and rng.random() < profile.duplicate_frame_p:
        return prev_frame.copy(), True
    return frame, False


def maybe_corrupt_state(state: np.ndarray, profile: CorruptionProfile, rng: np.random.Generator):
    if rng.random() < profile.sensor_glitch_p:
        s = state.copy()
        idx = rng.integers(0, len(s))
        s[idx] = np.nan
        return s
    return state


def jitter_sleep_amount(profile: CorruptionProfile, rng: np.random.Generator) -> float:
    if profile.jitter_std_s <= 0:
        return 0.0
    return float(abs(rng.normal(0, profile.jitter_std_s)))

"""
quality.py — automated quality checks over raw recorded episodes.

These run BEFORE any human annotation and are meant to catch mechanical
capture problems (the assignment's "detect missing, delayed, corrupted,
duplicated, or low-value demonstrations" question), as distinct from
semantic/behavioral judgments (which belong to human annotation, see
annotate.py).

Each check returns a list of QCFlag(step_range, kind, detail) so flags are
localized in time, not just a per-episode boolean -- you often want to keep
90% of an episode and drop a 12-step corrupted window, not discard the
whole demo.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class QCFlag:
    kind: str            # "dropped_frame" | "duplicate_frame" | "timestamp_drift" |
                          # "duplicate_step_id" | "sensor_glitch" | "low_value_episode"
    start: int
    end: int
    detail: str

    def as_dict(self):
        return {"kind": self.kind, "start": self.start, "end": self.end, "detail": self.detail}


def load_episode(npz_path: Path):
    data = np.load(npz_path)
    meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    return data, meta


def check_dropped_frames(data) -> list[QCFlag]:
    present = data["frame_present"]
    flags = []
    in_run = False
    start = 0
    for i, p in enumerate(present):
        if not p and not in_run:
            in_run, start = True, i
        elif p and in_run:
            flags.append(QCFlag("dropped_frame", start, i - 1, f"{i - start} missing frame(s)"))
            in_run = False
    if in_run:
        flags.append(QCFlag("dropped_frame", start, len(present) - 1, f"{len(present) - start} missing frame(s)"))
    return flags


def check_duplicate_frames(data) -> list[QCFlag]:
    """Flag consecutive frames that are byte-identical (beyond the trivial
    case of a genuinely static scene, which we approximate via near-zero
    proprioceptive velocity so we don't flag "agent stopped" as corruption)."""
    frames = data["frame"]
    present = data["frame_present"]
    state = data["state"]
    flags = []
    for i in range(1, len(frames)):
        if not (present[i] and present[i - 1]):
            continue
        if np.array_equal(frames[i], frames[i - 1]):
            vel = state[i, 2:4]
            if np.linalg.norm(vel) > 0.05:  # agent was moving -> frame SHOULD have changed
                flags.append(QCFlag("duplicate_frame", i - 1, i, "identical consecutive frames while agent in motion"))
    return flags


def check_timestamp_drift(data, dt_nominal: float, tol_factor: float = 4.0) -> list[QCFlag]:
    t = data["t_wall"]
    flags = []
    if len(t) < 2:
        return flags
    dt = np.diff(t)
    med = np.median(dt) if len(dt) else dt_nominal
    thresh = max(dt_nominal * tol_factor, med * tol_factor)
    bad = np.where(dt > thresh)[0]
    for b in bad:
        flags.append(QCFlag("timestamp_drift", int(b), int(b + 1),
                             f"gap {dt[b]*1000:.1f}ms vs nominal {dt_nominal*1000:.1f}ms"))
    return flags


def check_duplicate_step_ids(data) -> list[QCFlag]:
    ids = data["step_id"]
    flags = []
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            flags.append(QCFlag("duplicate_step_id", i - 1, i, f"step_id {ids[i]} repeated"))
    return flags


def check_sensor_glitches(data) -> list[QCFlag]:
    state = data["state"]
    flags = []
    bad_rows = np.where(~np.isfinite(state).all(axis=1))[0]
    for r in bad_rows:
        flags.append(QCFlag("sensor_glitch", int(r), int(r), "NaN/inf in proprioceptive state"))
    return flags


def check_low_value_episode(data, meta) -> list[QCFlag]:
    """Heuristics for demonstrations unlikely to be useful for imitation
    learning: near-zero displacement (agent barely moved), or failure in the
    first handful of steps (likely a setup glitch, not a real attempt)."""
    state = data["state"]
    flags = []
    pos = state[:, 0:2]
    displacement = float(np.linalg.norm(pos[-1] - pos[0])) if len(pos) else 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else 0.0
    if displacement < 0.3 and path_length < 1.0:
        flags.append(QCFlag("low_value_episode", 0, len(state) - 1,
                             f"near-zero net motion (disp={displacement:.2f}, path_len={path_length:.2f})"))
    if meta["outcome"] != "success" and meta["n_steps"] < 8:
        flags.append(QCFlag("low_value_episode", 0, meta["n_steps"] - 1,
                             "failure within first 8 steps -- likely bad start, not a real attempt"))
    return flags


def run_all_checks(npz_path: Path) -> dict:
    data, meta = load_episode(npz_path)
    flags: list[QCFlag] = []
    flags += check_dropped_frames(data)
    flags += check_duplicate_frames(data)
    flags += check_timestamp_drift(data, meta["dt_nominal"])
    flags += check_duplicate_step_ids(data)
    flags += check_sensor_glitches(data)
    flags += check_low_value_episode(data, meta)
    return {
        "episode_file": str(npz_path),
        "session_id": meta["session_id"],
        "episode_index": meta["episode_index"],
        "n_flags": len(flags),
        "flags": [f.as_dict() for f in flags],
        "recommend_discard": any(f.kind == "low_value_episode" for f in flags),
    }

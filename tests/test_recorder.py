import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajlens.env import ReachAvoidEnv
from trajlens.recorder import Recorder
from trajlens import quality as qc


def test_recorder_streams_are_same_length(tmp_path):
    env = ReachAvoidEnv(seed=1)
    rec = Recorder(tmp_path, "session_x")
    rec.reset_episode(0)
    for i in range(10):
        a = np.zeros(2, dtype=np.float32)
        frame = env.render()
        state, r, done, info = env.step(a)
        rec.record_step(i, state, a, r, done, frame)
    path = rec.save(outcome="timeout", env_seed=1, pilot_skill="expert", pilot_seed=0,
                     dt_nominal=0.05, corruption_profile={})
    data, meta = qc.load_episode(path)
    n = 10
    assert data["state"].shape[0] == n
    assert data["action"].shape[0] == n
    assert data["reward"].shape[0] == n
    assert data["frame"].shape[0] == n
    assert data["frame_present"].all()
    assert meta["n_steps"] == n
    assert meta["schema_version"].startswith("trajlens")


def test_recorder_handles_dropped_frame(tmp_path):
    env = ReachAvoidEnv(seed=2)
    rec = Recorder(tmp_path, "session_y")
    rec.reset_episode(0)
    for i in range(5):
        a = np.zeros(2, dtype=np.float32)
        frame = env.render() if i != 2 else None   # simulate a dropped frame at step 2
        state, r, done, info = env.step(a)
        rec.record_step(i, state, a, r, done, frame)
    path = rec.save(outcome="timeout", env_seed=2, pilot_skill="expert", pilot_seed=0,
                     dt_nominal=0.05, corruption_profile={})
    data, meta = qc.load_episode(path)
    assert data["frame_present"][2] == False
    assert data["frame_present"].sum() == 4


def test_code_version_is_deterministic(tmp_path):
    r1 = Recorder(tmp_path, "a")
    r2 = Recorder(tmp_path, "b")
    assert r1._code_version == r2._code_version
    assert len(r1._code_version) == 12

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajlens.env import ReachAvoidEnv
from trajlens.recorder import Recorder
from trajlens import quality as qc


def _record_episode(tmp_path, session_id, drop_at=None, dup_step_at=None,
                     nan_at=None, big_gap_at=None):
    import time
    env = ReachAvoidEnv(seed=3)
    rec = Recorder(tmp_path, session_id)
    rec.reset_episode(0)
    n = 20
    for i in range(n):
        a = np.array([0.5, 0.5], dtype=np.float32)  # nonzero action -> agent moves
        frame = env.render()
        state, r, done, info = env.step(a)
        if nan_at is not None and i == nan_at:
            state = state.copy()
            state[0] = np.nan
        if big_gap_at is not None and i == big_gap_at:
            time.sleep(0.05)  # dt_nominal in test is 0.001, so this is a huge relative gap
        rec.record_step(i, state, a, r, done, frame if i != drop_at else None)
        if dup_step_at is not None and i == dup_step_at:
            rec.record_step(i, state, a, r, done, frame)
    return rec.save(outcome="timeout", env_seed=3, pilot_skill="expert", pilot_seed=0,
                     dt_nominal=0.001, corruption_profile={})


def test_detects_dropped_frame(tmp_path):
    path = _record_episode(tmp_path, "s1", drop_at=5)
    result = qc.run_all_checks(path)
    kinds = [f["kind"] for f in result["flags"]]
    assert "dropped_frame" in kinds


def test_detects_duplicate_step_id(tmp_path):
    path = _record_episode(tmp_path, "s2", dup_step_at=7)
    result = qc.run_all_checks(path)
    kinds = [f["kind"] for f in result["flags"]]
    assert "duplicate_step_id" in kinds


def test_detects_sensor_glitch(tmp_path):
    path = _record_episode(tmp_path, "s3", nan_at=4)
    result = qc.run_all_checks(path)
    kinds = [f["kind"] for f in result["flags"]]
    assert "sensor_glitch" in kinds


def test_detects_timestamp_drift(tmp_path):
    path = _record_episode(tmp_path, "s4", big_gap_at=10)
    result = qc.run_all_checks(path)
    kinds = [f["kind"] for f in result["flags"]]
    assert "timestamp_drift" in kinds


def test_clean_episode_has_few_flags(tmp_path):
    path = _record_episode(tmp_path, "s5")
    result = qc.run_all_checks(path)
    # a clean, moving episode should have zero dropped/duplicate/glitch flags
    kinds = [f["kind"] for f in result["flags"]]
    assert "dropped_frame" not in kinds
    assert "duplicate_step_id" not in kinds
    assert "sensor_glitch" not in kinds


def test_low_value_episode_detected_for_stationary_agent(tmp_path):
    env = ReachAvoidEnv(seed=9)
    rec = Recorder(tmp_path, "s6")
    rec.reset_episode(0)
    for i in range(15):
        a = np.zeros(2, dtype=np.float32)  # never moves
        frame = env.render()
        state, r, done, info = env.step(a)
        rec.record_step(i, state, a, r, done, frame)
    path = rec.save(outcome="timeout", env_seed=9, pilot_skill="expert", pilot_seed=0,
                     dt_nominal=0.001, corruption_profile={})
    result = qc.run_all_checks(path)
    kinds = [f["kind"] for f in result["flags"]]
    assert "low_value_episode" in kinds
    assert result["recommend_discard"] is True

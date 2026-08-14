import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajlens.env import ReachAvoidEnv
from trajlens.recorder import Recorder
from trajlens import export as export_mod


def _make_episode(raw_dir, session_id, ep_idx, n=10, moving=True):
    env = ReachAvoidEnv(seed=ep_idx)
    rec = Recorder(raw_dir, session_id)
    rec.reset_episode(ep_idx)
    for i in range(n):
        a = np.array([0.5, 0.5], dtype=np.float32) if moving else np.zeros(2, dtype=np.float32)
        frame = env.render()
        state, r, done, info = env.step(a)
        rec.record_step(i, state, a, r, done, frame)
    rec.save(outcome="success" if moving else "timeout", env_seed=ep_idx,
              pilot_skill="expert", pilot_seed=0, dt_nominal=0.05, corruption_profile={})


def test_export_excludes_low_value_and_splits_by_episode(tmp_path):
    raw_dir = tmp_path / "raw"
    ann_dir = tmp_path / "annotations"   # deliberately empty -> majority_keep defaults True
    out_dir = tmp_path / "processed"

    for i in range(6):
        _make_episode(raw_dir, "session_a", i, moving=True)
    _make_episode(raw_dir, "session_a", 100, moving=False)  # low-value, should be excluded

    summary = export_mod.build_dataset(raw_dir, ann_dir, out_dir, val_fraction=0.34, split_seed=0)

    assert summary["n_episodes_total"] == 7
    assert summary["n_episodes_included"] == 6  # the stationary one is excluded
    ep_df = pd.read_csv(out_dir / "episodes.csv")
    frame_df = pd.read_csv(out_dir / "dataset.csv")

    excluded_row = ep_df[ep_df.episode_index == 100].iloc[0]
    assert excluded_row["included_in_dataset"] == False

    # no episode should have frames split across both train AND val
    for ep_idx in frame_df.episode_index.unique():
        splits = frame_df[frame_df.episode_index == ep_idx]["split"].unique()
        assert len(splits) == 1

    # frame count matches sum of n_steps for included episodes only
    included_steps = ep_df[ep_df.included_in_dataset]["n_steps"].sum()
    assert len(frame_df) == included_steps

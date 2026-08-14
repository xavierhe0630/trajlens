"""
export.py — turn raw + annotated episodes into a training-ready dataset.

Schema loosely follows the LeRobot / RLDS convention (episode_index,
frame_index, next global frame_index range per episode) because that's
close to what an actual imitation-learning loader expects, rather than
inventing something bespoke.

Two concrete design choices worth calling out (asked for in the prompt):

- SPLIT BY EPISODE, NOT BY FRAME. Frames within an episode are highly
  correlated (same trajectory); a frame-level random split leaks
  near-duplicate states between train/val and silently inflates validation
  performance. Splitting by session+episode id avoids that.

- DISCARD DECISION IS EXPLICIT AND AUDITABLE. An episode is excluded from
  the exported dataset if EITHER automated QC recommends discard OR the
  majority of annotators marked keep=False. Both reasons are recorded per
  excluded episode in `exclusions.json` rather than silently dropped, so a
  reviewer can second-guess the pipeline later.

Output: processed/dataset.csv (one row per frame, flat columns) +
        processed/episodes.csv (one row per episode, summary) +
        processed/exclusions.json
No parquet/pyarrow dependency (not available in this sandbox) — CSV is the
right choice at this scale (a few thousand rows) and is trivially loadable
by pandas, numpy, or any ML framework without an extra dependency.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from . import quality as qc


def _episode_files(raw_dir: Path):
    return sorted(raw_dir.glob("*/ep_*.npz"))


def _load_annotations(ann_dir: Path, session_id: str, episode_index: int) -> list[dict]:
    out = []
    if not ann_dir.exists():
        return out
    for annotator_dir in ann_dir.iterdir():
        if not annotator_dir.is_dir():
            continue
        p = annotator_dir / session_id / f"ep_{episode_index:04d}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def build_dataset(raw_dir: Path, ann_dir: Path, out_dir: Path, val_fraction: float = 0.2,
                   split_seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _episode_files(raw_dir)

    ep_rows = []
    frame_rows = []
    exclusions = []
    global_frame_idx = 0

    rng = np.random.default_rng(split_seed)
    ep_ids = [(f.parent.name, int(f.stem.split("_")[1])) for f in files]
    unique_eps = sorted(set(ep_ids))
    shuffled = unique_eps.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])

    for f in files:
        data, meta = qc.load_episode(f)
        qc_result = qc.run_all_checks(f)
        anns = _load_annotations(ann_dir, meta["session_id"], meta["episode_index"])

        keep_votes = [a["keep"] for a in anns]
        majority_keep = (sum(keep_votes) > len(keep_votes) / 2) if keep_votes else True
        qc_discard = qc_result["recommend_discard"]
        included = (not qc_discard) and majority_keep

        split = "val" if (meta["session_id"], meta["episode_index"]) in val_set else "train"

        ep_row = {
            "session_id": meta["session_id"], "episode_index": meta["episode_index"],
            "outcome": meta["outcome"], "n_steps": meta["n_steps"],
            "n_qc_flags": qc_result["n_flags"], "qc_recommend_discard": qc_discard,
            "n_annotators": len(anns),
            "annotator_keep_votes": sum(keep_votes) if keep_votes else None,
            "mean_quality_score": float(np.mean([a["quality_score"] for a in anns])) if anns else None,
            "included_in_dataset": included, "split": split,
            "code_version": meta["code_version"], "pilot_skill": meta["pilot_skill"],
            "corruption_profile": json.dumps(meta["corruption_profile"]),
            "total_reward": float(np.sum(data["reward"])),
        }
        ep_rows.append(ep_row)

        if not included:
            exclusions.append({
                "session_id": meta["session_id"], "episode_index": meta["episode_index"],
                "qc_discard": qc_discard, "majority_keep": majority_keep,
                "reasons": [fl["kind"] for fl in qc_result["flags"]],
            })
            continue

        state, action, reward, done = data["state"], data["action"], data["reward"], data["done"]
        for i in range(len(state)):
            frame_rows.append({
                "episode_index": meta["episode_index"], "session_id": meta["session_id"],
                "frame_index": i, "global_index": global_frame_idx,
                "x": state[i, 0], "y": state[i, 1], "vx": state[i, 2], "vy": state[i, 3],
                "goal_x": state[i, 4], "goal_y": state[i, 5],
                "action_x": action[i, 0], "action_y": action[i, 1],
                "reward": reward[i], "done": bool(done[i]), "split": split,
            })
            global_frame_idx += 1

    ep_df = pd.DataFrame(ep_rows)
    frame_df = pd.DataFrame(frame_rows)
    ep_df.to_csv(out_dir / "episodes.csv", index=False)
    frame_df.to_csv(out_dir / "dataset.csv", index=False)
    (out_dir / "exclusions.json").write_text(json.dumps(exclusions, indent=2))

    summary = {
        "n_episodes_total": len(ep_rows),
        "n_episodes_included": int(ep_df["included_in_dataset"].sum()) if len(ep_df) else 0,
        "n_frames_total": len(frame_rows),
        "n_train_episodes": int(((ep_df.split == "train") & ep_df.included_in_dataset).sum()) if len(ep_df) else 0,
        "n_val_episodes": int(((ep_df.split == "val") & ep_df.included_in_dataset).sum()) if len(ep_df) else 0,
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

#!/usr/bin/env python3
"""
run_demo.py — end-to-end demonstration of the full TrajLens pipeline.

    collect (env + pilot + recorder + corruption)
        -> automated quality control (quality.py)
        -> simulated dual-annotator labeling (annotate.py)
        -> export to training-ready dataset (export.py)
        -> analysis report with plots (analysis.py)
        -> manifest for the static HTML annotation/replay viewer

Run:  python scripts/run_demo.py --n-sessions 6 --episodes-per-session 8
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trajlens.env import ReachAvoidEnv, MAX_STEPS
from trajlens.pilot import PotentialFieldPilot
from trajlens.recorder import Recorder
from trajlens import corruption as corr
from trajlens import quality as qc
from trajlens import annotate as ann_mod
from trajlens import export as export_mod
from trajlens import analysis as analysis_mod

DT_NOMINAL = 1.0 / 20.0


def collect_session(data_root: Path, session_id: str, n_episodes: int,
                     env_seed_base: int, pilot_skill: str, corruption_name: str | None,
                     rng: np.random.Generator):
    profile = corr.PROFILES[corruption_name] if corruption_name else corr.CLEAN
    recorder = Recorder(data_root / "raw", session_id)
    pilot = PotentialFieldPilot(skill=pilot_skill, seed=int(rng.integers(0, 1 << 30)))

    for ep_idx in range(n_episodes):
        env_seed = env_seed_base + ep_idx
        env = ReachAvoidEnv(seed=env_seed)
        recorder.reset_episode(ep_idx)
        prev_frame = None
        outcome = "timeout"
        for step_id in range(MAX_STEPS):
            action = pilot.act(env)
            state = corr.maybe_corrupt_state(env.state(), profile, rng)
            frame = env.render()
            frame_or_none, _dup = corr.maybe_corrupt_frame(frame, prev_frame, profile, rng)
            if frame_or_none is not None:
                prev_frame = frame_or_none

            next_state, reward, done, info = env.step(action)

            this_step_id = step_id
            recorder.record_step(this_step_id, state, action, reward, done, frame_or_none)
            if profile.duplicate_step_p > 0 and rng.random() < profile.duplicate_step_p:
                # emulate a control loop double-fire: log the same step_id again
                recorder.record_step(this_step_id, state, action, reward, done, frame_or_none)

            jitter = corr.jitter_sleep_amount(profile, rng)
            if jitter:
                time.sleep(min(jitter, 0.02))  # capped so demo stays fast

            if done:
                if info["success"]:
                    outcome = "success"
                elif info["collided"]:
                    outcome = "collision"
                else:
                    outcome = "timeout"
                break

        recorder.save(
            outcome=outcome, env_seed=env_seed, pilot_skill=pilot_skill,
            pilot_seed=pilot.rng.bit_generator.seed_seq.entropy if hasattr(pilot.rng.bit_generator, "seed_seq") else 0,
            dt_nominal=DT_NOMINAL, corruption_profile=profile.as_dict(),
        )


def run_qc_and_annotate(data_root: Path):
    raw_dir = data_root / "raw"
    ann_dir = data_root / "annotations"
    qc_dir = data_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    all_qc = []
    for npz_path in sorted(raw_dir.glob("*/ep_*.npz")):
        result = qc.run_all_checks(npz_path)
        all_qc.append(result)
        out_p = qc_dir / npz_path.relative_to(raw_dir).with_suffix("").with_suffix(".qc.json")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(result, indent=2))

        ann_a = ann_mod.simulate_annotator(npz_path, result, "sim_annotator_a", noise_seed=hash((result["session_id"], result["episode_index"], "a")) % (2**31))
        ann_b = ann_mod.simulate_annotator(npz_path, result, "sim_annotator_b", noise_seed=hash((result["session_id"], result["episode_index"], "b")) % (2**31))
        ann_mod.save_annotation(ann_a, ann_dir)
        ann_mod.save_annotation(ann_b, ann_dir)

    return all_qc


def build_viewer_manifest(data_root: Path, out_path: Path, max_episodes: int = 12, max_frames_per_ep: int = 40):
    """Samples a handful of episodes and embeds low-res base64 thumbnails so
    the standalone HTML viewer (viewer/index.html) needs zero server and zero
    network access to browse real captured trajectories."""
    raw_dir = data_root / "raw"
    files = sorted(raw_dir.glob("*/ep_*.npz"))[:max_episodes]
    episodes = []
    for f in files:
        data, meta = qc.load_episode(f)
        n = meta["n_steps"]
        idxs = np.linspace(0, n - 1, min(max_frames_per_ep, n)).astype(int)
        frames_b64 = []
        for i in idxs:
            if not data["frame_present"][i]:
                frames_b64.append(None)
                continue
            img = Image.fromarray(data["frame"][i]).resize((64, 64))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        episodes.append({
            "session_id": meta["session_id"], "episode_index": meta["episode_index"],
            "outcome": meta["outcome"], "n_steps": n,
            "frame_indices": idxs.tolist(), "frames_b64": frames_b64,
            "actions": data["action"][idxs].tolist(),
            "rewards": data["reward"][idxs].tolist(),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"episodes": episodes}, indent=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-sessions", type=int, default=6)
    ap.add_argument("--episodes-per-session", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    rng = np.random.default_rng(args.seed)

    session_configs = []
    skills = ["expert", "expert", "novice", "expert", "novice", "expert"]
    corruptions = ["clean", "flaky_camera", "clean", "flaky_controller", "sensor_fault", "clean"]
    for i in range(args.n_sessions):
        session_configs.append((f"session_{i:02d}", skills[i % len(skills)], corruptions[i % len(corruptions)]))

    print(f"[1/5] Collecting {args.n_sessions} sessions x {args.episodes_per_session} episodes ...")
    for session_id, skill, corruption_name in session_configs:
        collect_session(
            data_root, session_id, args.episodes_per_session,
            env_seed_base=hash(session_id) % 100000, pilot_skill=skill,
            corruption_name=corruption_name, rng=rng,
        )
    print("      done.")

    print("[2/5] Running automated QC + simulated dual-annotator labeling ...")
    run_qc_and_annotate(data_root)
    print("      done.")

    print("[3/5] Exporting training-ready dataset (episode-level train/val split) ...")
    summary = export_mod.build_dataset(data_root / "raw", data_root / "annotations", data_root / "processed")
    print(f"      {summary}")

    print("[4/5] Running analysis + generating report ...")
    result = analysis_mod.run_analysis(data_root / "processed", data_root / "annotations", data_root / "reports")
    print(f"      {result}")

    print("[5/5] Building viewer manifest for the standalone HTML annotation/replay tool ...")
    build_viewer_manifest(data_root, Path("viewer") / "manifest.json")
    print("      done. Open viewer/index.html in a browser to page through real episodes.")


if __name__ == "__main__":
    main()

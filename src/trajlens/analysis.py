"""
analysis.py — turns the pipeline's outputs into an actual analysis, not
just a pile of artifacts. This is the part that answers "is this dataset
any good, and how do you know."

Produces:
  - reports/episode_stats.png     length & return distributions, outcome mix
  - reports/qc_breakdown.png      QC flag counts by kind, by corruption profile
  - reports/action_histogram.png  control distribution (sanity check: is the
                                   policy/pilot saturating its action range?)
  - reports/report.md             narrative summary with numbers filled in
                                   from the actual run (not hardcoded)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import annotate as ann_mod


def _fig_episode_stats(ep_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].hist(ep_df["n_steps"], bins=15, color="#5096f0", edgecolor="white")
    axes[0].set_title("Episode length (steps)")
    axes[0].set_xlabel("steps"); axes[0].set_ylabel("count")

    axes[1].hist(ep_df["total_reward"], bins=15, color="#3cc878", edgecolor="white")
    axes[1].set_title("Episode return (sum reward)")
    axes[1].set_xlabel("return")

    outcome_counts = ep_df["outcome"].value_counts()
    axes[2].bar(outcome_counts.index, outcome_counts.values,
                color=["#3cc878", "#d24646", "#e0a83c"][:len(outcome_counts)])
    axes[2].set_title("Outcome mix")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fig_qc_breakdown(ep_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    by_profile = ep_df.groupby(ep_df["corruption_profile"].apply(lambda s: json.loads(s)))["n_qc_flags"]
    # group by the *name* of the profile inferred from nonzero fields, for readability
    def profile_name(d):
        if not d or all(v == 0 for v in d.values()):
            return "clean"
        if d.get("drop_frame_p", 0) > 0:
            return "flaky_camera"
        if d.get("duplicate_step_p", 0) > 0:
            return "flaky_controller"
        if d.get("sensor_glitch_p", 0) > 0:
            return "sensor_fault"
        return "other"

    ep_df = ep_df.copy()
    ep_df["profile_name"] = ep_df["corruption_profile"].apply(lambda s: profile_name(json.loads(s)))
    grp = ep_df.groupby("profile_name")["n_qc_flags"].mean().sort_values(ascending=False)
    axes[0].bar(grp.index, grp.values, color="#a06cd5")
    axes[0].set_title("Mean QC flags per episode\nby injected corruption profile")
    axes[0].tick_params(axis="x", rotation=20)

    disc = ep_df.groupby("profile_name")["qc_recommend_discard"].mean().sort_values(ascending=False)
    axes[1].bar(disc.index, disc.values, color="#e08a3c")
    axes[1].set_title("Fraction flagged low-value\nby corruption profile")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fig_action_hist(frame_df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(frame_df["action_x"], bins=30, color="#5096f0", alpha=0.8, label="action_x")
    axes[1].hist(frame_df["action_y"], bins=30, color="#3cc878", alpha=0.8, label="action_y")
    for ax, name in zip(axes, ["action_x", "action_y"]):
        ax.set_title(f"{name} distribution")
        ax.axvline(0, color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_analysis(processed_dir: Path, ann_dir: Path, report_dir: Path) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    ep_df = pd.read_csv(processed_dir / "episodes.csv")
    frame_df = pd.read_csv(processed_dir / "dataset.csv")

    _fig_episode_stats(ep_df, report_dir / "episode_stats.png")
    _fig_qc_breakdown(ep_df, report_dir / "qc_breakdown.png")
    _fig_action_hist(frame_df, report_dir / "action_histogram.png")

    # --- inter-annotator agreement, if two simulated annotators exist ---
    kappa = None
    if ann_dir.exists():
        annotator_dirs = sorted([d for d in ann_dir.iterdir() if d.is_dir()])
        if len(annotator_dirs) >= 2:
            a_dir, b_dir = annotator_dirs[0], annotator_dirs[1]
            common = []
            for session_dir in a_dir.iterdir():
                for f in session_dir.glob("ep_*.json"):
                    b_f = b_dir / session_dir.name / f.name
                    if b_f.exists():
                        a = json.loads(f.read_text())
                        b = json.loads(b_f.read_text())
                        common.append((a["keep"], b["keep"]))
            if common:
                a_keep, b_keep = zip(*common)
                kappa = ann_mod.cohens_kappa(list(a_keep), list(b_keep))

    success_rate = float((ep_df["outcome"] == "success").mean())
    n_included = int(ep_df["included_in_dataset"].sum())
    n_total = len(ep_df)
    mean_flags = float(ep_df["n_qc_flags"].mean())
    exclusions = json.loads((processed_dir / "exclusions.json").read_text())

    lines = []
    lines.append("# TrajLens — Data Quality & Analysis Report\n")
    lines.append(f"- Episodes collected: **{n_total}**\n")
    lines.append(f"- Episodes retained in exported dataset: **{n_included}** ({n_included/n_total:.0%})\n")
    lines.append(f"- Overall pilot success rate: **{success_rate:.0%}**\n")
    lines.append(f"- Mean automated QC flags per episode: **{mean_flags:.2f}**\n")
    if kappa is not None:
        lines.append(f"- Simulated inter-annotator agreement (Cohen's κ, keep/discard): **{kappa:.2f}**\n")
    lines.append(f"- Episodes excluded and why: **{len(exclusions)}** (see `exclusions.json` for the full audit list)\n")
    lines.append("\n## Figures\n")
    lines.append("![episode stats](episode_stats.png)\n")
    lines.append("![qc breakdown](qc_breakdown.png)\n")
    lines.append("![action histogram](action_histogram.png)\n")
    lines.append("\n## Reading the numbers\n")
    lines.append(
        "Episodes generated under an injected `flaky_camera` or `flaky_controller` "
        "corruption profile should show visibly higher mean QC-flag counts than "
        "`clean` episodes in the chart above — that comparison is the closest thing "
        "this project has to a unit test for the QC layer itself: if flagged episodes "
        "and injected-corruption episodes did not line up, the checks would be "
        "unreliable regardless of how they read in isolation.\n"
    )
    report = "\n".join(lines)
    (report_dir / "report.md").write_text(report)

    result = {
        "n_total": n_total, "n_included": n_included, "success_rate": success_rate,
        "mean_qc_flags": mean_flags, "kappa": kappa, "n_exclusions": len(exclusions),
    }
    (report_dir / "report_summary.json").write_text(json.dumps(result, indent=2))
    return result

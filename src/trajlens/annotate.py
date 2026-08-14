"""
annotate.py — the human-judgment layer on top of automated QC.

Automated checks (quality.py) catch mechanical capture faults. They cannot
tell you "the agent technically avoided the obstacle but grazed it in a way
that would be dangerous on real hardware" or "this success was lucky, not
skillful." That's what annotation is for, and it's why the two are kept as
separate passes with separate outputs.

Two ways to produce annotations, both writing the same schema:

1. `viewer/` (static HTML+JS, see that directory) — a human pages through
   episodes rendered from the exported manifest and clicks labels. Zero
   server, zero extra dependency, works offline; exports a JSON the human
   downloads and drops back into annotations/.

2. `simulate_annotator()` below — for *this* submission I don't have a
   second human available in the time box, so to still demonstrate the
   inter-annotator-agreement workflow (a real data-quality practice) I
   simulate two annotators as noisy perturbations of a ground-truth-ish
   rule, and report Cohen's kappa on their agreement in the analysis
   report. This is explicitly disclosed here and in AI_USAGE.md /
   TECHNICAL_NOTE.md — it is a stand-in for real human annotators, not a
   claim that real agreement was measured.

Annotation schema (per episode), written to annotations/{session}/ep_{i}.json:
{
  "session_id": ..., "episode_index": ...,
  "annotator_id": "sim_a" | "sim_b" | "<human id>",
  "keep": bool,                      # would you include this in a training set?
  "quality_score": 1-5,
  "event_windows": [{"start": int, "end": int, "label": "contact"|"correction"|"stall"}],
  "notes": str
}
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from . import quality as qc


@dataclass
class Annotation:
    session_id: str
    episode_index: int
    annotator_id: str
    keep: bool
    quality_score: int
    event_windows: list[dict] = field(default_factory=list)
    notes: str = ""

    def as_dict(self):
        return {
            "session_id": self.session_id, "episode_index": self.episode_index,
            "annotator_id": self.annotator_id, "keep": self.keep,
            "quality_score": self.quality_score, "event_windows": self.event_windows,
            "notes": self.notes,
        }


def _rule_based_quality(npz_path: Path, qc_result: dict) -> tuple[bool, int, list[dict]]:
    """A simple, transparent heuristic used as the 'ground truth' that both
    simulated annotators perturb. Real annotation guidelines would be
    written up for humans; this stands in for that rulebook."""
    data, meta = qc.load_episode(npz_path)
    n_flags = qc_result["n_flags"]
    outcome = meta["outcome"]

    base_score = {"success": 5, "timeout": 2, "collision": 2}.get(outcome, 2)
    score = max(1, base_score - min(n_flags, 3))
    keep = outcome == "success" and n_flags <= 2

    windows = []
    for f in qc_result["flags"]:
        if f["kind"] in ("dropped_frame", "duplicate_frame", "duplicate_step_id"):
            windows.append({"start": f["start"], "end": f["end"], "label": "capture_glitch"})
    if outcome == "collision":
        windows.append({"start": meta["n_steps"] - 1, "end": meta["n_steps"] - 1, "label": "contact"})

    return keep, score, windows


def simulate_annotator(npz_path: Path, qc_result: dict, annotator_id: str,
                        noise_seed: int, disagreement_rate: float = 0.15) -> Annotation:
    rng = np.random.default_rng(noise_seed)
    keep, score, windows = _rule_based_quality(npz_path, qc_result)
    data, meta = qc.load_episode(npz_path)

    # perturb to emulate genuine but bounded human disagreement
    if rng.random() < disagreement_rate:
        keep = not keep
    score = int(np.clip(score + rng.integers(-1, 2), 1, 5))

    return Annotation(
        session_id=meta["session_id"], episode_index=meta["episode_index"],
        annotator_id=annotator_id, keep=keep, quality_score=score,
        event_windows=windows,
        notes="simulated annotation (see AI_USAGE.md / TECHNICAL_NOTE.md)",
    )


def save_annotation(ann: Annotation, out_dir: Path):
    d = out_dir / ann.annotator_id / ann.session_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"ep_{ann.episode_index:04d}.json").write_text(json.dumps(ann.as_dict(), indent=2))


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Cohen's kappa for binary keep/discard agreement between two annotators."""
    a = np.array(labels_a, dtype=bool)
    b = np.array(labels_b, dtype=bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = np.mean(a == b)
    p_a1 = np.mean(a)
    p_b1 = np.mean(b)
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))

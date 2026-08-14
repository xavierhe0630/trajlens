# TrajLens

A small end-to-end pipeline for **collecting, quality-checking, annotating,
and exporting synchronized multimodal demonstration data** from an
interactive simulated task — built for the Iacon Autonomics take-home
(Problem Statement 2: *Build a Data Collection or Annotation Tool*).

Rather than a single script, this is the full path raw interaction data has
to take before it's trustworthy training data:

```
   ┌────────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────┐
   │ ReachAvoid │──▶│ Recorder │──▶│ Automated QC  │──▶│ Annotation   │──▶│ Exporter │──▶ training-ready
   │ env+pilot  │   │ (sync'd  │   │ (drift, drops,│   │ (keep/discard│   │ (episode-│    dataset.csv
   │ (data src) │   │  capture)│   │ dupes, glitch,│   │  quality,    │   │  level    │
   │            │   │          │   │  low-value)   │   │  events)     │   │  split)   │
   └────────────┘   └──────────┘   └───────────────┘   └──────────────┘   └──────────┘
                                                              ▲
                                                     viewer/index.html
                                                (standalone HTML annotator,
                                                 zero server, zero deps)
```

## Why a simulated task instead of a real DualSense/robot?

The assignment explicitly allows this ("If you do not have hardware, a
simulator, game, or recorded multimodal dataset is acceptable"). I built a
tiny custom 2D "reach-avoid" task (`src/trajlens/env.py`) — an agent must
reach a goal while avoiding obstacles — rendered headlessly with PIL (no
GPU, no display server needed, which matters in a sandboxed environment).
Demonstrations come from a scripted potential-field "pilot"
(`src/trajlens/pilot.py`) with injected human-like imperfection (reaction
lag, aim noise, occasional overcorrection), standing in for a human
teleoperator. **This substitution is disclosed, not hidden** — see
`AI_USAGE.md` and the technical note. Everything downstream of action
generation (synchronization, schema, corruption, QC, annotation, export,
analysis) is exactly what would be needed for a real DualSense/robot
teleoperation pipeline; only the source of the actions differs.

## What's actually being demonstrated

1. **Synchronized capture** (`recorder.py`) — every step is indexed by both
   an integer `step_id` and a wall-clock timestamp, and every stream
   (proprioception, RGB frame, action, reward, done) is captured through one
   authoritative call so "synchronized" is a checkable property, not an
   assumption.
2. **Realistic capture faults, injected on purpose** (`corruption.py`) —
   dropped frames, duplicated frames, timestamp jitter, duplicated control
   ticks, NaN sensor glitches — so the QC layer has real problems to catch
   and I can verify it actually catches them (see `tests/test_quality.py`
   and the QC-vs-corruption-profile chart in the analysis report).
3. **Automated quality control** (`quality.py`) — localizes each fault to a
   step range rather than just flagging whole episodes, and separately
   flags "low-value" demonstrations (near-zero motion, instant failure).
4. **Annotation layer** (`annotate.py` + `viewer/index.html`) — a
   zero-server, zero-dependency HTML/JS tool for a human to page through
   real captured episodes frame-by-frame and record keep/discard, a 1–5
   quality score, and notes; plus a programmatic/simulated-annotator path
   (used for this submission's demo run, disclosed in AI_USAGE.md) that
   still lets me report a real inter-annotator-agreement number.
5. **Training-ready export** (`export.py`) — episode-level (not frame-level)
   train/val split to avoid leakage, explicit and auditable exclusion
   reasons (`exclusions.json`), LeRobot/RLDS-style episode/frame indexing.
6. **Analysis** (`analysis.py`) — episode length/return distributions,
   outcome mix, QC-flag breakdown by corruption profile, action-saturation
   histograms, Cohen's κ, all computed from the actual run, written into a
   generated `report.md`.

## Quickstart

```bash
pip install -r requirements.txt        # numpy, pandas, matplotlib, pillow
python scripts/run_demo.py --n-sessions 6 --episodes-per-session 8
```

This collects 6 simulated sessions (mixed pilot skill + mixed injected
corruption profiles), runs QC, simulates two annotators, exports the
dataset, and writes the analysis report + viewer manifest. Takes well
under a minute.

Outputs land in `data/`:
```
data/raw/<session>/ep_XXXX.npz + .meta.json   # raw synchronized capture
data/qc/<session>/ep_XXXX.qc.json             # automated QC flags per episode
data/annotations/<annotator>/<session>/...    # per-episode annotations
data/processed/dataset.csv, episodes.csv,     # training-ready export
              exclusions.json, export_summary.json
data/reports/report.md, *.png                 # analysis report
```

Then open `viewer/index.html` directly in a browser (no server needed) to
page through real captured episodes frame-by-frame and try the manual
annotation tool yourself.

## Running tests

```bash
python tests/run_tests.py
```
(pytest isn't installable in this offline sandbox; `run_tests.py` runs the
exact same `test_*` functions with a tiny dependency-free harness. The test
files themselves are ordinary pytest-style tests and will run unmodified
under `pytest tests/` if you have it installed.)

10/10 tests currently pass, covering: recorder stream-length/sync
invariants, QC detection of each injected fault type on a controlled
episode, and exporter split/exclusion correctness.

## Project layout

```
src/trajlens/
  env.py         reach-avoid simulated task (state, action, render)
  pilot.py       scripted demonstration generator (teleop stand-in)
  recorder.py    synchronized capture + provenance metadata
  corruption.py  injected realistic capture faults
  quality.py     automated QC checks
  annotate.py    annotation schema + simulated dual-annotator + Cohen's kappa
  export.py      training-ready dataset builder (episode-level split)
  analysis.py    stats, plots, generated report
scripts/run_demo.py     end-to-end orchestration
viewer/index.html       standalone HTML/JS episode viewer & annotator
tests/                  unit tests + dependency-free runner
demo_outputs/           sample generated artifacts (for quick viewing without re-running)
```

## Limitations (see TECHNICAL_NOTE.md for the full discussion)

- The "teleoperator" is scripted, not a live human or real controller — the
  synchronization/QC/export machinery is agnostic to this, but the resulting
  demonstrations are cleaner than what a real novice human would produce.
- Inter-annotator agreement is computed on two *simulated* annotators
  perturbing a shared heuristic, not two independent humans — it validates
  the *workflow*, not real annotator reliability.
- The task/domain is intentionally simple (2D, low-dimensional) so the whole
  pipeline could be built, tested, and evaluated within the suggested time
  box, rather than spending the budget on a more visually impressive but
  less-scrutinized environment.

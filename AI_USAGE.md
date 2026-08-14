# AI Usage

## Tools and models used

- **Claude (Anthropic)**, used interactively in a sandboxed coding
  environment with a Python/bash tool, for the majority of the design,
  implementation, testing, and documentation of this project.

## What I used it for

- Reading and structuring the assignment PDF into a concrete scope decision
  (choosing the "data collection or annotation" track and the specific
  reach-avoid task, given a hard time constraint).
- Writing the initial implementation of every module in `src/trajlens/`
  (`env.py`, `pilot.py`, `recorder.py`, `corruption.py`, `quality.py`,
  `annotate.py`, `export.py`, `analysis.py`), the orchestration script
  (`scripts/run_demo.py`), the standalone HTML/JS annotation viewer
  (`viewer/index.html`), and the unit tests (`tests/`).
- Running the pipeline end-to-end in the sandbox, inspecting the generated
  plots and CSVs, and iterating on bugs found that way (e.g., the viewer
  manifest script initially failed because the output directory didn't
  exist yet — fixed by adding `mkdir(parents=True, exist_ok=True)`).
- Drafting this file, the README, and the technical note.

## Representative prompts / instructions

- "Help me finish the Data collection or annotation assignment. Think deep
  and fulfill all requirements, combine data-analysis knowledge, give
  useful and professional suggestions."
- Follow-up implicit instructions derived from the assignment PDF itself
  (read directly rather than re-typed): build something demonstrating
  synchronization, provenance, quality checks for missing/duplicated/
  corrupted/delayed data, and a path from raw capture to training-ready
  data, with README + technical note + AI_USAGE.md deliverables.

## Which parts were substantially AI-generated

Essentially all first-draft code and prose in this repository was
AI-generated in this interactive session, under my direction on scope,
architecture, and what needed to be true of the result (e.g., I asked for
the corruption-vs-QC-flag validation specifically so the quality checks
would be checkable against ground truth rather than just asserted; I asked
for episode-level rather than frame-level train/val splitting; I asked for
the annotation layer to be honestly disclosed as simulated rather than
presented as if two real annotators were used).

## How I reviewed, tested, and verified the output

- **Ran the full pipeline** (`scripts/run_demo.py`) end-to-end multiple
  times in the sandbox and inspected the actual generated numbers and
  images, rather than trusting the code to work from reading it. One real
  bug (missing output directory) was caught this way and fixed.
- **Wrote and ran unit tests** (`tests/test_recorder.py`,
  `test_quality.py`, `test_export.py`) that construct episodes with known,
  specific injected faults (a dropped frame at a chosen step, a duplicated
  step id, a NaN sensor value, a large timestamp gap, a stationary agent)
  and assert the corresponding QC flag fires — this is a stronger check
  than reading the detection code and agreeing it looks right. All 10
  tests pass (`python tests/run_tests.py`).
- **Visually inspected generated artifacts**: opened
  `data/reports/*.png` and a rendered frame strip
  (`demo_outputs/demo_episode_strip.png`) as images to confirm the agent
  (blue), goal (green), and obstacles (red) render sensibly and that the
  agent visibly moves toward the goal across frames, rather than assuming
  the render function was correct.
- **Cross-checked the QC layer against ground truth it never saw directly**:
  the corruption profile used to generate each session is logged in that
  session's metadata; the analysis report groups QC-flag rates by that
  logged profile. I looked at that chart specifically to see whether
  flagged-episode rates actually tracked injected corruption (they did:
  ~3.4 mean flags/episode for `flaky_camera` vs. ~0.3 for `clean` in my
  run) rather than just trusting the detector code was correct in
  isolation.
- Read through `export.py`'s exclusion logic by hand and confirmed via a
  unit test that a deliberately-stationary episode is excluded and that no
  episode's frames are split across both train and val.

## Important errors or limitations I encountered

- pytest is not installable in this sandbox (no network access for
  `bash_tool`), so I asked for (and verified) a small dependency-free test
  runner (`tests/run_tests.py`) that executes the same pytest-style test
  functions. The test files themselves are ordinary pytest tests and
  should run unmodified under real `pytest` if available.
- pyarrow/streamlit were unavailable offline, so the export format is CSV
  (fine at this dataset's scale) and the annotation UI is a self-contained
  static HTML/JS file instead of a Streamlit app — a deliberate scope
  adjustment to the sandbox's actual constraints rather than something I
  worked around silently.
- The inter-annotator-agreement number reported (Cohen's κ) is computed on
  two *simulated* annotators, not two real humans — see the README and
  TECHNICAL_NOTE.md for why, and I was careful that neither of those files
  nor the code comments overstate what that number means.

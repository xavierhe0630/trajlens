# Technical Note — TrajLens

## Why this problem

I chose the data-collection/annotation track because the interesting failure
modes in real robot-learning and agent-training pipelines are rarely in the
model — they're in the data: silent timestamp drift, a camera that drops
one frame in forty, an annotator who disagrees with themselves a week
later, a "successful" demonstration that only succeeded because the agent
got lucky. Those problems are invisible unless the pipeline is built to
surface them, and I wanted to build something where I could *prove* the
surfacing works, not just assert it.

## Decisions and tradeoffs

**Simulated task over real hardware.** I don't have a DualSense or a robot
in this environment, and the assignment explicitly allows a simulator. I
built a small custom 2D "reach-avoid" task rather than reaching for
`gymnasium`, mainly because a from-scratch environment let me render frames
with plain PIL with zero display-server dependency — a real constraint in a
sandboxed take-home — and because owning the whole state/render pipeline
made it trivial to inject and later verify realistic capture faults.

**Scripted pilot over a learned policy or live human.** With a fixed time
box, a live human data-collection session wasn't reproducible enough to
build QC/tests against, and training an RL policy would have spent the
whole budget on the wrong part of the assignment. I used a potential-field
controller with injected noise, lag, and occasional overcorrection to
produce demonstrations with realistic *quality variance* (not every episode
succeeds cleanly) rather than a suspiciously perfect dataset. This is
disclosed in the README and AI_USAGE.md — the honest framing is that the
recorder, schema, corruption model, QC, annotation, and export code are the
deliverable, and all of it is agnostic to where the actions came from.

**Injecting corruption on purpose.** Rather than claim the QC layer "checks
for drift and drops," I sample one of four corruption profiles per session
(clean / flaky camera / flaky controller / sensor fault) and log which
profile produced each episode. This turned the QC step from something I
had to trust into something I could check: the analysis report's
QC-flags-by-profile chart shows flagged-episode rates tracking the injected
profile almost exactly (flaky-camera episodes average ~3.4 QC flags vs.
~0.3 for clean episodes in my demo run), which is the closest thing this
project has to ground-truth validation of the detectors themselves.

**Episode-level, not frame-level, train/val split.** Frames within one
episode are highly correlated; a random frame split would leak
near-duplicate states across train/val and overstate how well a model
generalizes. The exporter splits by `(session_id, episode_index)` instead.

**Simulated dual annotation.** I didn't have a second human annotator
available inside the time box, so I built a real annotation *schema* and a
standalone, zero-server HTML viewer a human can use, but for this
submission's demo run I generated two simulated annotators as noisy
perturbations of a shared heuristic rulebook, and reported Cohen's κ (0.45
in my run) on their keep/discard agreement. I want to be explicit that this
validates the *pipeline* — that annotations flow correctly into export
decisions and that agreement is computed correctly — and does **not**
constitute a real inter-annotator-reliability study.

## What worked

The pieces I most trust are the QC detectors and the exporter's exclusion
logic, both because they're covered by unit tests that construct an episode
with a specific injected fault and assert the corresponding flag fires
(`tests/test_quality.py`), and because the corruption-profile validation
above corroborates them on data the tests never touched. The
episode-vs-frame split logic is also directly tested.

## What didn't work as well

My pilot is good enough (~90%+ success in a typical run) that "timeout" and
genuinely marginal episodes are underrepresented — a real teleoperation
session, especially from novice operators, would produce more ambiguous
cases for the annotation layer to actually earn its keep on. The duplicate
step-id injection (simulating a control loop double-firing) is also a
slightly artificial mechanism compared to real controller-driver behavior;
I'd want to look at actual USB HID polling traces before trusting that
model much further.

## How I evaluated the result

Three ways: (1) unit tests asserting each fault type is detected on a
controlled synthetic episode, (2) the corruption-profile-vs-flag-rate
comparison as an indirect validation on the full, un-inspected demo run,
and (3) manually reviewing sample episodes through `viewer/index.html` to
confirm the rendered frames and action overlays actually correspond to
what the recorded arrays say happened.

## What I'd build or test next

Swap the scripted pilot for real keyboard-driven teleoperation (the render
loop already produces a live frame per step, so a Pygame input capture loop
is a small addition) to get genuinely noisy human data, then re-run the
same QC/export pipeline unchanged and see whether the corruption-profile
validation still holds on real human timing jitter. I'd also want a
held-out real second annotator to replace the simulated one and check
whether κ on real humans looks anything like 0.45.

## If I joined Iacon Autonomics

I'd want to work on the boundary between data collection and evaluation —
specifically, using the same synchronized-trajectory infrastructure to
build *model-based* evaluators (does a policy's rollout distribution match
the demonstration distribution on the dimensions that matter, like
action-saturation or contact-event rate, rather than just success rate) so
that data quality and model quality are assessed with the same tooling
instead of two disconnected pipelines. Given the gym's multi-domain scope
(LLM agents, robotics, games, science), I'd also be interested in whether
the same schema/QC/export approach generalizes across those domains, or
whether each needs its own notion of "low-value" and "drift" — that seems
like a genuinely open question worth a focused experiment.

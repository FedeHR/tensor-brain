# DROID as a Tensor Brain testbed: what it enables, and what it does not

Written 2026-08-21. This is an experiment-scoping document, in the genre of
`docs/pvsg_perception.md`: it defines candidate experiments and their protocols before it
defines any implementation. Nothing here has been run.

**Scheduling note.** The thesis is due 2026-09-03. Nothing in this document is thesis
material and nothing here should displace the 14-day plan in `thesis_roadmap/01_THESIS_PLAN.md`.
This is the *next* programme — the one that turns Chapters 7 and 8 into papers.

---

## 0. Provenance: what was actually released, and by whom

This matters, because it determines what is novel and what is two years mature.

| Artifact | Who | When | What it is |
|---|---|---|---|
| **DROID** | 13-institution academic consortium (Khazatsky, Pertsch, … Levine, Finn), RSS 2024 | Mar 2024 | The dataset itself. CC-BY 4.0. |
| Language annotations | DROID team | Dec 2024 | 3 crowd annotations for 75k episodes (95% of successes) |
| Improved camera calibration | DROID team | Apr 2025 | Higher-accuracy extrinsics for ~36k episodes |
| **PointWorld-DROID** | NVIDIA | 15 Apr 2026 | 3D point trajectories + optimized cameras + depth for **42,935 episodes**; 4.65 TB restored |
| **Cosmos3-DROID** | NVIDIA | **19 Aug 2026** | DROID repackaged in LeRobotDataset v3.0, 640×360 |
| **Cosmos 3 Edge Policy (DROID)** | NVIDIA | **19 Aug 2026** | 4B on-device VLA post-trained on DROID, + post-training recipe |

So DROID is not an NVIDIA dataset and it is not new. What is two days old is NVIDIA's
repackaging plus a **latency-instrumented, on-device baseline policy**. For our purposes that
is *better* than a brand-new dataset: the data is well-characterised and stable, and there is
now a published, hardware-measured latency number to argue against.

**The number that matters most to this project:**

> Cosmos 3 Edge generates an action chunk of 32 steps in **≈1.53 s**, and that chunk covers
> **≈2.13 s** of robot motion at 15 Hz. It achieves 22.9% success on closed-loop RoboLab
> across 120 tasks.

A duty ratio of 0.72 — the robot spends 72% of its motion budget acting on a belief formed
before the motion started. Hold that thought until §5.1.

---

## 1. What DROID contains

All numbers from the RSS 2024 paper unless marked.

**Scale and spread.** 76,000 successful teleoperated trajectories, 350 hours, **564 scenes**,
**52 buildings**, **86 tasks/verbs**, 13 institutions, 18 robots, 50 collectors, 12 months.
An additional **~16,000 trajectories marked "not successful"** are shipped in the release but
excluded from the headline count. 1,417 distinct camera viewpoints.

Derived: 350 h × 15 Hz ≈ **18.9M timesteps**; ÷ 76k episodes ≈ **249 timesteps ≈ 16.6 s per
episode**; ÷ 564 scenes ≈ **135 successful episodes per scene** (≈163 including failures);
≈10.8 scenes per building.

**Hardware, identical at all 13 sites.** Franka Panda 7-DoF + Robotiq 2F-85 gripper; two
adjustable Zed 2 stereo cameras (third-person) + one wrist-mounted Zed Mini; Polymetis
controller at **15 Hz**; Oculus Quest 2 teleoperation.

**Per-timestep fields (RLDS).**

- `observation`: `joint_position` (7), `cartesian_position` (6), `gripper_position` (1),
  `wrist_image_left`, `exterior_image_1_left`, `exterior_image_2_left` (each 180×320×3)
- `action_dict`: `joint_position`/`joint_velocity` (7), `cartesian_position`/`cartesian_velocity`
  (6), `gripper_position`/`gripper_velocity` (1)
- `action` (7): 6 joint velocities + 1 gripper position
- `is_first`, `is_last`, `is_terminal`, `reward`, `discount`
- `language_instruction`, `language_instruction_2`, `language_instruction_3`

**Per-episode metadata (raw release).** `uuid`, `lab`, `user`, `user_id`, `date`, `timestamp`,
**`building`**, **`scene_id`**, **`success`**, `current_task`, `trajectory_length`,
`robot_serial`, `wrist_cam_serial`, `ext1_cam_serial`, `ext2_cam_serial`, and the three camera
extrinsics lists.

**Collection protocol facts that turn out to be the scientifically load-bearing ones.**

1. A collector sets the robot down in a scene, calibrates, enters a task list, and collects
   **~100 trajectories over ~20 minutes** before moving to a new scene.
2. The GUI **randomly samples which task** to attempt next, so tasks are not ordered by ease.
3. The GUI **periodically prompts a "scene augmentation"**: move and re-calibrate the exterior
   cameras, change the room lighting, add or remove objects from the scene.
4. The collector marks each episode success or failure by hand.

Points 1 and 3 together are the whole reason this dataset is interesting to us. See §3.

**Auxiliary resources.** `KarlP/droid` on HuggingFace ships `cam2base_extrinsics.json` (~36k),
`cam2cam_extrinsics.json` (~90k), `cam2base_extrinsic_superset.json` (~24k), `intrinsics.json`
(~72k, `episode → serial → [fx, cx, fy, cy]`), **`keep_ranges_1_0_1.json`** (idle-frame
filtering ranges per episode), and `episode_id_to_path.json`.

**Sizes.** RLDS full ≈ **1.7 TB**; raw stereo ≈ 8.7 TB (`gs://gresearch/robotics/droid_raw`);
**`droid_100` ≈ 2 GB**, 100 trajectories, identical schema — the smoke-test subset.

---

## 2. What the corpus has established, restated as capabilities we can bet on

Condensed from `thesis_roadmap/02_ASSET_INVENTORY.md`, `03_PROBE_FINDINGS.md`, and
`docs/index_feedback_evidence.md`. Only the properties a DROID experiment could exploit.

| # | Property | Evidence | Status |
|---|---|---|---|
| P1 | **Order-invariance at machine precision** (~5e-17); every competing correction loses it | `section3` | Proved + measured |
| P2 | **Exact retraction** — un-observe a symbol without replaying history | `experiments/crowd/model.py:retract` | Implemented, never exploited |
| P3 | **Cost**: `O(nK)` per symbol, no enumeration; crossover law `λ*·T_exact ≈ 0.06` | probe E3 | Measured, synthetic |
| P4 | **Evolution caps the `M²` blow-up**: static 0.099→2.484 vs temporal 0.114→**0.100 flat** at M=8 | alt repo | Measured, synthetic, **under-advertised** |
| P5 | **Small-M downstream win**: additive beats exact Bayes to ~5 absorbed symbols on COCO | COCO run | Measured, real data |
| P6 | **The gauge is free** and improves calibration (ECE 0.060→0.021); MC-estimable in `O(SnK)` | COCO + probe E5 | Measured |
| P7 | **Gated emission makes the rule exact**, not approximate (1e-16); τ-family estimable | `section3` | Proved + verified |
| P8 | **Index feedback works when the index is recognizable**; null when the injected vector is a near-constant mixture | pair runs | Measured both ways |
| N1 | ✗ Redundant evidence is double-counted: ×8 redundancy, Bayes 11.88 vs additive **17.87** | alt repo | Measured |
| N2 | ✗ Degrades *worse* under misspecification: noise 0→0.8, Bayes 5.08→6.76, additive 5.61→**7.52** | alt repo | Measured |
| N3 | ✗ **Uncertainty sampling is an anti-gate** — it selects low-`Z` points where the rule is worse | alt repo | Measured |
| N4 | ✗ Overconfident at large `M` in the static regime | Ch 3 | Measured |

The honest reading: **P3, P4, P5 and P7 are the reasons to be on a robot. N1 and N3 are the
reasons a robot could embarrass us.** Both are testable on DROID, and a design that only tests
the first four is not worth running.

---

## 3. The structural match — and the structural mismatch

### 3.1 What DROID gives that PVSG and COCO cannot

**(a) Time, with a price on it.** `01_THESIS_PLAN.md` names this as the corpus's biggest hole:
*"The whole corpus contains no task, utility, action or regret."* DROID is the first dataset in
the programme where the state changes while you compute and the consequence is physical.
Probe E3's crossover law is currently a synthetic Poisson-drift statement measured against
laptop wall-clock. DROID makes both halves real: real state dynamics at 15 Hz, and a published
1.53 s inference latency for a deployed policy on the same data.

**(b) `M` in the hundreds.** A DROID episode absorbs ~249 observations. PVSG's pair schedule
has ~4 concept windows; the COCO run tops out at `M=8`, exactly where the static `M²` blow-up
becomes fatal (recovery 70.4%). P4 says evolution stops that growth — but P4 has only ever been
shown to `M=8` on synthetic data. **DROID offers a 30× longer horizon on a real sensor stream.**
If the additive filter is still flat at `M=249`, P4 stops being an under-advertised footnote and
becomes the headline.

**(c) A genuine known-entity regime, at scale, with nuisance variation.** This is the one
condition `docs/index_feedback_evidence.md` §7 names as decisive-and-untested:

> *"Known-identity pair evaluation (VRD-EX analogue) … the only condition in which identity
> feedback can carry information. If feedback is null here too, that is a real negative result
> about the mechanism. Until then, it is untested."*

DROID's collection protocol builds this for free. Within one `scene_id`: ~135 episodes over
~20 minutes, same physical objects, **with the cameras deliberately moved and re-calibrated and
the lighting deliberately changed mid-session**. That is a re-identification test with real
nuisance variation across episodes — strictly harder and more meaningful than VRD-EX (which
distorted copies of training images) and harder than PVSG's `blocked` protocol (later frames of
the same continuous video, same camera, same lighting).

**(d) A labelled failure set.** ~16k `success=false` episodes. Real distributional failure, not
CIFAR-vs-SVHN. This is what the `heisenberg-frontier` energy-OOD campaign has been missing.

**(e) Genuinely multi-source, genuinely asynchronous, genuinely droppable input.** Three camera
streams + proprioception + language, at different information rates, with real occlusion and
real mid-episode camera motion. The README already frames repeated `integrate_input` calls as
QTB Eq. 46's gated sum `g(ν) = Σ_k μ_k g(ν_k)`. DROID is the first dataset where that equation
is doing work rather than being a notational convenience.

**(f) Redundancy you can dial.** `exterior_image_1` and `exterior_image_2` view the same scene
from two angles. That is *literally* the redundant-evidence condition of N1, available as a
controlled variable rather than a synthetic ×8 replication.

### 3.2 What DROID takes away — read this before getting excited

- **No object identities, no masks, no relation labels.** The entire PVSG evidence pipeline
  (mask-pooled DINO over stable per-video object IDs) has no analogue. Identity must be
  *constructed*: from `scene_id` (scene identity, which is not object identity), from
  PointWorld-DROID's 3D point tracks, or from an off-the-shelf tracker. Every object-level
  proposal below carries this cost, and it is the single largest engineering risk.
- **`scene_id` is a noisy identity by construction.** Scene augmentations add and remove objects
  mid-session. "Same scene" means "same room and same table", not "same object set".
- **Labels are thin.** One collector-judged binary per episode, plus a free-text instruction and
  three crowd paraphrases. No dense reward, no per-step annotation, no ontology.
- **No closed-loop evaluation without a Franka.** Everything we can do is offline replay.
  DROID's own headline claim was validated by 10 real rollouts per task per method. We cannot
  make policy-success claims, and should not try.
- **Scale is a real cost.** 1.7 TB RLDS. Vision-feature extraction over the full set is a
  serious cluster job, not a Mac job.

---

## 4. Candidate experiments, ranked

Ranked by (scientific value to the TB/Heisenberg programme) × (feasibility) ÷ cost. Each entry
states what would *falsify* it, because a design that cannot fail is not an experiment.

Baselines throughout follow the project rule: **strong and capacity-comparable, with known-good
hyperparameters, no equal-budget sweeps.** Arms are kept to the minimum the question needs.

---

### E1 — The staleness of a real robot's belief · ⭐ start here

**Question.** Probe E3 proved *"exact inference loses once the state's half-life falls below
roughly ten times its own latency"* (`λ*·T_exact ≈ 0.06`, equivalently half-life ≈ 11.6·T).
That law has never met a real state. **What is the actual half-life of task-relevant state on a
real manipulation robot, and where does that put real inference latencies on the crossover
curve?**

**Why the TB specifically.** This is the whole normative case for the additive update
(Mechanism C), and it is currently the weakest-anchored one. It is also the case that no
competing method can make, because the argument is *about* the cost of the competitor.

**Protocol.** Derive a set of binary task facts *exactly and for free* from proprioception at
15 Hz — no vision, no learning, no annotation:

```
gripper_closed      gripper_position below threshold
in_contact          gripper commanded closed but position stalls above the closed stop
lifted              cartesian z above the episode's resting z by a margin
moving              cartesian velocity norm above threshold
near_base           cartesian xy radius below threshold
approaching         d/dt of radius negative
wrist_down          orientation within a cone of vertical
... (target n ≈ 12–16, so exact enumeration over 2^n stays tractable)
```

Then measure three things:

1. **Empirical half-life per fact and per scene type** — the distribution of dwell times before
   a flip. This is a descriptive statistic about robot manipulation that, as far as I can tell,
   nobody has published.
2. **Placement on the crossover curve.** For each `n`, probe E3 already gives `T_exact(n)`;
   crossover half-life is `11.6·T_exact(n)`. Read off the `n` above which exact enumeration is
   already losing on DROID's measured dynamics.
3. **The staleness statistic for a deployed VLA.** During a 1.53 s Cosmos 3 Edge inference
   window, how many of the `n` facts flip? Expressed as a fraction of the fact set, per task
   type, this is *the belief error a deployed on-device policy accepts by construction*.

**Motivating arithmetic, to be checked not asserted.** Inverting the law at `T = 1.53 s` gives a
crossover half-life of ≈17.7 s. The mean DROID episode is **16.6 s**. If the whole task
completes in less time than the crossover half-life, then essentially every task-relevant fact
turns over at least once per inference window. I expect the measurement to confirm this
comfortably, but *the extrapolation is an analogy, not an instantiation* — the 1.53 s is a
neural forward pass, not enumeration latency, and the law was derived where both arms solve the
same inference problem. E1 must measure fact half-lives directly and report the law's
instantiation separately from the VLA analogy.

**Baselines.** Exact enumerated filter over the fact set; the additive rule; the gauge-corrected
additive rule. All three charged their own measured wall-clock, all propagating to their own
completion time, scored against the true fact vector at that instant — the E3 protocol,
transplanted onto measured dynamics instead of synthetic Poisson drift.

**Endpoint.** Crossover `n` on real dynamics; median fact half-life; VLA staleness fraction.

**What falsifies it.** Fact half-lives of tens of seconds, i.e. manipulation state that is far
more inert than the synthetic drift model assumed. That would be a genuine negative and would
weaken Mechanism C rather than strengthen it — which is exactly why it is worth measuring.

**Cost. Very low.** Proprioception only. `droid_100` for the smoke test, a few thousand episodes
on a laptop for the result, no GPU, no vision, no training. **This is a days-not-weeks
experiment and it converts the corpus's most speculative mechanism into a measurement.**

---

### E2 — Does the additive filter stay flat at `M ≈ 249`?

**Question.** P4 says the evolution operator caps the `M²` error growth (25× at `M=8`,
synthetic). **Does that hold on a real sensor stream at thirty times the horizon?**

**Why the TB specifically.** P4 is the direct answer to Chapter 3's worst result and, per the
inventory, *"one of the most under-advertised results in the corpus."* Its weakness is that it
lives at `M ≤ 8` in simulation. A robot episode is the natural habitat for the claim.

**Protocol.** Reuse E1's fact set (so `n ≤ 16` and the exact filter remains enumerable — this is
the design constraint that makes the experiment possible at all). Fit a small index layer `A`
over symbols emitted from the stream; run the filter over an entire episode. Three arms, which
are the minimum the question needs:

1. static additive (no evolution between windows) — the blow-up control
2. additive + evolution — the claim
3. exact enumerated filter — the reference

Plot marginal KL to exact against `M` from 1 to ~249. Report the steady-state error budget
(factorization / dropped `log Z` / transition linearization) as the alt repo does, but at
`M=249` on real data.

**Endpoint.** Error vs `M` curve; steady-state budget; the `M` at which the static arm exceeds
the temporal arm by 10×.

**What falsifies it.** Error growing past a few dozen symbols. Plausible mechanism if it does:
DROID's facts are strongly autocorrelated at 15 Hz, so consecutive observations are *redundant*
— and N1 says redundancy is precisely what the additive rule handles worst. **E2 is therefore a
real contest between P4 and N1, not a victory lap.** Subsampling the stream (1 Hz vs 15 Hz)
should be reported as a sweep over redundancy, since it directly trades M against correlation.

**Cost. Low.** Proprioception + a small fitted `A`. Laptop-feasible; cluster only if the fact
set is widened.

---

### E3 — Known-entity index feedback, in the regime the corpus says is decisive

**Question.** `docs/index_feedback_evidence.md` established that the PVSG identity-feedback null
is *scoped*, not general: it was measured where no identity candidate could be correct, and the
injected vector was ~92% constant across examples. **Does index feedback help when the index is
recognizable and the candidate set is small enough not to wash out?**

**Why the TB specifically.** This is the architectural core — the same `A` serving bottom-up
scoring and top-down injection — and settling it is the single highest-stakes open question in
the TB line. DROID supplies the missing regime.

**Protocol.** The scene index is the enrollable entity. Within a `scene_id`:

- **enroll** on the earliest episodes,
- **embargo** a block of episodes,
- **evaluate** on the latest episodes,

so the same physical scene recurs across the split while camera extrinsics and lighting change
under the deliberate scene-augmentation prompts. This is the `blocked` protocol of
`docs/pair_known_entity_protocol.md` transplanted to a harder, more realistic nuisance regime.
Task: predict
the instruction's verb and object (86 verbs; objects via the paper's spaCy semantic-parsing
pipeline) from visual evidence at a held-out frame.

Arms — again the minimum, and chosen to test the corpus's own stated prediction:

1. **no feedback** (the control the paper never ran cleanly)
2. **P-SA** — expected feedback over scene columns
3. **P-Samp** — winner-take-all injection of the top-scoring scene embedding

**The pre-registered prediction, taken verbatim from the corpus:** *"P-Samp (winner-take-all)
over category candidates should be the strongest condition, because it injects one specific
embedding rather than a mixture that re-averages toward the common direction."* DROID is where
that prediction can be tested, because there are 564 scenes rather than 2,474 near-collinear
identities, and because the correct index is *in* the candidate set by construction.

**Diagnostics are mandatory, not optional.** The `55b34b0` instrumentation
(`direction_cosine_mean`, `unsupervised_attention_mass_mean`, `l2_over_pre_feedback_q`) must be
on from the first run. The PVSG round wasted two cycles because the mechanism of the null was
only diagnosed afterwards. Report the identity↔identity collinearity of the scene bank up front:
if mean pairwise cosine is as high as PVSG's 0.479, expect the same washing-out and say so
before running.

**Two confounds inherited from PVSG that must be controlled from the start**, both recorded in
the `docs/fidelity.md` ledger:

- **Injection magnitude.** The ledger's *"Bottom-up versus top-down scale"* row is explicit:
  visual drive to index-embedding norm runs `27.7 : 1.0` at initialization, so a single injected
  embedding moves the state by **4–7%**, and *"any claim that top-down feedback is too weak to
  matter must report this ratio rather than treating it as a property of the architecture."*
  Magnitude is named as the last surviving mechanical explanation for the PVSG null. On DROID the
  input normalization is ours to choose, so `μ` must be an explicit reported variable — a null at
  4% of the state norm would settle nothing.
- **Which schedule counts as the Tensor Brain.** The ledger carries an unresolved *suspected
  discrepancy*: the original paper treats attention as an **alternative** to sampled injection
  (§5.3 "replaces line 20"), while QTB treats Algorithms 2 and 3 as **sequential**, which applies
  roughly 3× the top-down drive and under which the `integral-none` control *"is not a Tensor
  Brain at all"*. The QTB sequential schedule has never been run. DROID is a reasonable place to
  run it for the first time, but it must be a named condition, not a silent default.

**Why the scene index should carry information.** This is the paper's own stated benefit
(§6.4): a recognized index biases the label. A kitchen scene affords *pour*, *toast*, *open
fridge*; a desk scene affords *open drawer*, *stack*. Scene → verb is a real, strong channel —
DROID's own Fig. 4 shows 10 scene types with sharply different task distributions. If top-down
biasing works anywhere, it works here.

**Baseline.** A capacity-matched bottom-up classifier over the same features, with the scene
identity supplied as a one-hot input rather than as index feedback. That separates *"knowing
which scene"* from *"the TB's mechanism for using it"*, and it is the comparison a reviewer will
demand.

**What falsifies it.** Feedback null again, with `direction_cosine_mean` high (i.e. the injected
vector *is* example-specific) and feedback magnitude at a workable fraction of `‖q‖`. That
combination would be a real negative result about the mechanism — the one the corpus explicitly
says is still owed.

**Cost. Medium-high.** DINOv3 features over a scene-dense subset. Suggested scoping: retain
scenes with ≥80 episodes, sample ~8 frames per episode. At ~200 scenes × 100 episodes × 8 frames
≈ 160k images, this is comparable to the PVSG extraction already done — a cluster job of hours,
not days.

---

### E4 — Additive multi-source integration and graceful degradation under sensor loss

**Question.** QTB Eq. 46 makes input a gated additive sum over sources. **Does that give a
measurable robustness advantage when a source drops out, and how much does redundancy between
the two exterior cameras cost us (N1)?**

**Why the TB specifically.** This is the clearest *architectural* advantage available on this
dataset, and it is a robotics-facing claim rather than an inference-theory one. The DROID
policy baseline concatenates ResNet-50 features from both exterior cameras with a frozen
DistilBERT language embedding and proprioception, then feeds an MLP. **A concatenation model has
no principled behaviour when a camera is missing** — it must impute, or be retrained. Additive
gated integration makes a missing source contribute exactly zero, which is the correct Bayesian
default for "no evidence", and it costs nothing to implement because it is the input equation.

**Protocol.** Predict the fact set (E1's, or the verb/object set) at frame `t`. Sources:
`exterior_1`, `exterior_2`, `wrist`, proprioception, language. Conditions:

1. **all sources, learned gates `μ_k`** — the model
2. **redundancy stress** — both exterior cameras present vs one; measure the double-counting
   penalty directly, then measure how much of it the free gauge fix (P6) removes
3. **dropout at evaluation** — drop each source in turn, at rates matched to DROID's real
   occlusion and camera-motion statistics

**Baseline.** The DROID-style concatenation encoder at matched capacity with the published
hyperparameters, with missing sources zero-imputed (standard practice). No equal-budget sweep;
both models get their known-good settings.

**Endpoint.** Held-out NLL and calibration (ECE) as a function of which sources are present.
The claim to be tested is *the shape of the degradation curve*, not the peak number.

**What falsifies it.** The concatenation baseline degrading just as gracefully under
zero-imputation — quite possible, since zero-imputation of a normalized feature is not obviously
worse than dropping an additive term. If so, the interesting result becomes N1: **how badly does
the second exterior camera hurt?** That is a publishable negative either way.

**Cost. Medium-high.** Shares the feature extraction with E3; run them off one extraction job.

---

### E5 — Is embodied perception state-gated? Estimating τ on a real sensor stream

**Question.** Mechanism B says the `log Z` term appears only because you condition on *how many*
symbols were emitted; if emission is state-gated, it cancels exactly and the additive rule is
**exact, not approximate** (verified to 1e-16). The τ-family `π(x) = Z(x)^τ/C` interpolates, and
τ is recoverable unbiased by Poisson regression. **Is real robot perception closer to the gated
protocol or the fixed-`M` protocol?**

**Why the TB specifically.** This is the highest-conceptual-payoff item in the document. It
would be the first measurement of the emission protocol on a real embodied stream, and it puts
an empirical anchor under the corpus's best reviewer-facing sentence: *"Your exact-Bayes
baseline is itself an approximation — it assumes a protocol."*

**Why a robot is the natural place to ask.** The number of symbols emitted at time `t` is not a
free parameter — it is set by the state. Occlusion, distance, clutter, and gripper contact
determine what is recognizable. That is a state-dependent gate arising from physics, not from a
modelling convenience.

**Protocol.** Define symbol emission operationally and *before* looking at the fit: a symbol is
emitted at `t` when its index score exceeds a fixed threshold under the fitted `A`. Regress the
emitted-symbol count `M_t` on `log Z(x_t)` by Poisson regression to recover τ. Report τ per
scene type and per building.

**Falsification is built in and must be pre-registered.** `output/tau/dart.json` is a warning:
it ships fitted τ values that `SYNTHESIS.md` disowns as uninterpretable. The pre-registered
interpretability criterion here is **cross-scene consistency**: if τ is a property of embodied
perception it must be stable across the 564 scenes and 52 buildings; if it varies with scene
type as much as it varies from zero, the estimate is measuring the threshold, not the protocol,
and the result should be reported as uninterpretable and closed — as the diffusion probe was.

**Cost. Medium, risk high.** Requires a fitted index layer (so it follows E3/E4) plus a defended
symbol definition. **Do not run this before E1 and E2.** But if τ comes back near zero and
stable, it is the best result in this document.

---

### E6 — Retraction under mid-episode recalibration

**Question.** P2 — exact retraction — is implemented and has never been exploited. **Is there a
real setting where un-observing evidence beats never having a way to un-observe it?**

**Why the TB specifically.** Additivity gives retraction for free: `q ← q − a_k` exactly undoes
`q ← q + a_k`, with no history replay and no re-running the filter. **No recurrent network, no
transformer, and no diffusion policy can do this.** It is the cleanest "new capability"
available, not merely a performance claim.

**Why DROID supplies the setting.** The collection protocol *deliberately* moves and
re-calibrates the exterior cameras mid-session. Evidence integrated under a stale calibration is
retrospectively known to be wrong, and the episode metadata plus the improved-extrinsics files
tell you exactly when. The task: maintain a running belief over the fact set across a
multi-episode scene session; on a recalibration event, retract the affected evidence and
continue.

**Baselines.** (a) never retract — carry the corrupted evidence; (b) full replay from the
session start with the corrupted evidence removed, which is the *correct* answer and gives the
accuracy ceiling; (c) sliding window — discard everything older than the event, the standard
practical fix. The TB claim is to **match (b)'s accuracy at (a)'s cost**, and to beat (c), which
throws away good evidence along with bad.

**Endpoint.** Belief error after a recalibration event, against wall-clock and against evidence
retained.

**What falsifies it.** Full replay being cheap enough that retraction buys nothing — likely if
sessions are short. Mitigation: report the crossover in session length, which is the honest form
of the claim anyway.

**Cost. Low-medium**, and it composes with E1/E2's proprioceptive fact set.

---

### E7 — Crowd aggregation over the three language annotations · not recommended

The `experiments/crowd` line fits worker confusion matrices as index embeddings and aggregates
with the categorical additive update, with confidence-based stopping. DROID's three independent
crowd annotations per episode look like a match.

**They are not, and the reason should be recorded so this is not proposed again.** Crowd-Kit
`relevance-5` has 104,111 annotations over 5,080 fitted tasks with gold labels; the aggregator's
advantage there shows up in calibration (Brier 0.520 → 0.307, ECE 0.279 → 0.029 at 10
annotations). DROID gives **exactly 3 annotations per item, no gold labels, and no worker
identity in the released annotation file**, so worker confusion matrices cannot be estimated at
all and the stopping rule has nothing to stop. Adding it would be a condition the question does
not need.

**What is worth extracting instead, cheaply:** inter-annotator disagreement over the three
paraphrases is a free per-episode *ambiguity* measure. Use it as a covariate in E3 — the
prediction being that top-down scene feedback helps most where the instruction is most ambiguous,
which is precisely the paper's label-biasing claim. That is one column in an existing table, not
a new experiment.

---

## 5. The simple additive Heisenberg update specifically

The user's second question. Three of the six experiments above (E1, E2, E5) are
Heisenberg-first rather than TB-first. Beyond those, DROID opens two things that are hard to get
any other way.

### 5.1 A cheap belief filter running underneath a slow VLA · the strongest application

This is the system proposal, and it is where the additive rule's cost profile is the entire
point rather than a footnote.

Cosmos 3 Edge emits a 32-step action chunk every ≈1.53 s, covering ≈2.13 s of motion. For 72% of
its motion budget the robot is executing a plan formed against a belief that is already stale.
The proposal:

> Run an additive Heisenberg filter over task-relevant facts at the full 15 Hz control rate,
> *underneath* the chunked policy. Between chunks it costs `O(nK)` per observation — nanoseconds
> against the policy's 1.53 s. Use it to (a) detect when the belief has diverged from what the
> executing chunk assumed and trigger early re-planning, or (b) gate execution of the remaining
> chunk steps.

Every piece of the correctness argument already exists in the corpus: P3 gives the cost, E1
gives the staleness measurement that motivates it, P4 gives the guarantee that the filter stays
accurate over hundreds of absorbed observations, and P7 gives the conditions under which it is
exact rather than approximate.

**This is a genuine robotics contribution, not a re-run of an inference benchmark**, and it is
the answer to "where does the TB offer something difficult to achieve through other methods."
The honest caveat is stated once and prominently: **without a Franka, the trigger rule can be
evaluated offline (does it fire before the recorded failure?) but not closed-loop.** Offline,
DROID's 16k failures give the labels; the endpoint is lead time to failure versus false-alarm
rate, which is a legitimate and complete offline result. Closing the loop needs hardware and
should be scoped as a collaboration, not smuggled in as a claim.

### 5.2 Gauge-dependence of failure detection, with real failures

The `heisenberg-frontier` campaign found that energy-based OOD detection is **gauge-dependent**:
`E(h) = −logsumexp(logits)` is exactly what the gauge moves, and on GPT-2 the AUROC ranges
0.054 → 0.784 across gauges of a bit-identical model, with the canonical flat gauge — chosen with
no OOD labels — moving it 0.153 → 0.439. That is a claim against a heavily-cited paper, and the
confirmation run (CIFAR-10 vs SVHN on a ResNet) is still not done.

DROID offers a **better confirmation set than CIFAR-vs-SVHN**: 16k real failure episodes against
76k successes, on a model (Cosmos 3 Edge, with its Nemotron reasoner readout) that is actually
deployed. Real distributional failure with real consequences beats a synthetic OOD pair, and the
"is your OOD detector measuring the model or the coordinate system?" question lands harder when
the OOD set is *robots dropping things*.

This composes with the frontier paper rather than competing with the thesis, and it needs no new
theory — only the existing gauge instrument pointed at a VLA readout.

### 5.3 A falsifiable prediction we should want to test

N3 says uncertainty sampling is an **anti-gate**: it selects low-`Z` points, i.e. `τ < 0`,
exactly where the additive rule is worse than on unselected data. Transplanted to a robot this
predicts that *"absorb evidence when uncertain"* — an entirely standard active-perception
heuristic — should **underperform** uniform absorption for an additive filter. That is
counterintuitive, cheap to test inside E2 as a sampling arm, and interesting whichever way it
falls.

---

## 6. What not to do

Recording these so they are not re-proposed.

1. **Do not compete with diffusion policies or VLAs on action prediction.** The TB has no action
   head, closed-loop evaluation is impossible without hardware, and the comparison would be lost
   on engineering rather than on science. Wrong fight, and it would discredit the rest.
2. **Do not make policy success-rate claims from offline replay.** DROID's own claims rest on
   10 real rollouts per task per method. Ours cannot.
3. **Do not rebuild the PVSG relation pipeline on DROID.** No relation annotations exist;
   constructing them is a dataset-paper's worth of work and tests nothing about the TB.
4. **Do not lead with crowd aggregation** (§E7).
5. **Do not treat `scene_id` as object identity.** Scene augmentations change the object set
   mid-session. Any object-level claim needs PointWorld-DROID tracks or a real tracker, and
   should be scoped separately.

---

## 7. Staging and compute

Consistent with the project's compute split: **the Mac gets smoke tests, training goes to the
cluster.**

| Stage | What | Data | Where |
|---|---|---|---|
| 0 | Schema smoke test, fact extractors, plots | `droid_100` (2 GB) | Mac |
| 1 | **E1** fact half-lives + crossover placement | proprio only, ~5k episodes | Mac |
| 2 | **E2** long-horizon filter, `n ≤ 16` exact reference | proprio only | Mac, cluster if `n` grows |
| 3 | DINOv3 extraction, scene-dense subset (~160k frames) | scenes with ≥80 episodes | Cluster |
| 4 | **E3** + **E4** off the same extraction | as above | Cluster |
| 5 | **E6** retraction | proprio + episode metadata | Mac |
| 6 | **E5** τ estimation | needs stage 4's fitted `A` | Cluster |

Stages 1 and 2 need no GPU, no vision, and no download beyond a few GB of low-dimensional
fields. **They can start immediately after the thesis ships and produce results in days.**

One practical note: the RLDS `episode_metadata` exposes only `recording_folderpath` and
`file_path` — `success`, `scene_id`, `building` and `lab` live in the *raw* release's
`metadata_*.json`. E1/E2 need only the low-dimensional trajectory fields, but **E3 and E6 depend
on the raw metadata**, so budget for the join through `episode_id_to_path.json` early rather
than discovering it at stage 3. Also apply `keep_ranges_1_0_1.json`: idle frames would otherwise
inflate every dwell-time statistic in E1 and every `M` in E2.

---

## 8. Summary judgement

**DROID is a good fit for this programme, but not for the reason it looks like it should be.**
It is not a better PVSG — it is strictly worse as a perception dataset, having no identities, no
masks and no relations. It is valuable because it supplies the two things the corpus most
conspicuously lacks: **a clock with a price on it, and a hundreds-long observation horizon on a
real sensor stream.** Those map onto Mechanism C (timeliness) and P4 (evolution caps the `M²`
growth) — respectively the corpus's most speculative claim and its most under-advertised one.

The recommended entry point is **E1**, because it is days of laptop work, needs no vision, no
training and no GPU, and converts a synthetic law into a measurement about real robots. **E2**
follows immediately on the same infrastructure and is a genuine contest between P4 and N1.
**E3** carries the highest scientific stakes for the Tensor Brain proper, since it is the one
regime the index-feedback question has never been asked in. **§5.1** is the strongest
application-facing story and the clearest answer to *"where does the TB do something hard to
achieve otherwise"* — with the hardware limitation stated honestly and up front.

---

## Sources

- [DROID project page](https://droid-dataset.github.io/)
- [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (RSS 2024)](https://www.roboticsproceedings.org/rss20/p120.pdf)
- [DROID documentation](https://droid-dataset.github.io/droid/the-droid-dataset)
- [KarlP/droid — language annotations, calibration, keep-ranges](https://huggingface.co/KarlP/droid)
- [nvidia/PointWorld-DROID](https://huggingface.co/datasets/nvidia/PointWorld-DROID)
- [Post-Train NVIDIA Cosmos 3 Edge for On-Device Robot Control (19 Aug 2026)](https://developer.nvidia.com/blog/post-train-nvidia-cosmos-3-edge-for-on-device-robot-control/)
- [DROID policy learning code](https://github.com/droid-dataset/droid_policy_learning)
- [DROID schema in Daft documentation](https://docs.daft.ai/en/stable/datasets/droid/)

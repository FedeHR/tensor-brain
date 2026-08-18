# Five candidate applications for the Heisenberg update

Companion to `docs/heisenberg_experiment_design.md`, which lives on branch
`worktree-heisenberg-experiment-design` (2026-08-17) and already ranks six
opportunities in its §3.3. This document does **not** restate that list. It adds
the two application shapes that list does not cover, and gives a sharpened,
independently-checked take on three that it does.

Read §3.3 of the scoping document first. Everything here assumes it.

---

## 0. The selection rule I used

The theory forecloses the obvious pitches — no robustness gain under
misspecification, worse under redundant evidence, learning does not fix the
approximation (`bayes_approximation.md` §6). So "the approximation is good
enough and cheap" is not a thesis. The one claim that is both sharp and
apparently unoccupied in the literature is the cancellation identity:

> If a window opens with probability `π(x) = Z(x)/C` and only opened windows are
> recorded, then `p(O=1, k | x) = exp(s_k(x))/C`. The normalizer cancels, and the
> plain additive rule is the **exact** posterior at every `M`
> (`tb_update_generalized.md` §6, §13d).

So the search is for **pipelines whose evidence passed a score-proportional
gate**, not for tasks where the numbers happen to be better. Three filters,
applied to every candidate below:

1. **Does the number of recorded items vary with score mass?** If a fixed count
   is always returned, the gate is a conditioning-on-total-count, the normalizer
   does *not* cancel, and the whole story inverts. This is Trap 1 of the scoping
   doc and it is what disqualifies fixed top-`k` RAG.
2. **Does the gate depend on the latent, not on observables?** If it depends on
   observables, standard propensity/IPS methods already handle it and the
   contribution is nil. The interesting case is precisely the one the MNAR
   literature calls *nonignorable* — and the claim becomes **ignorability by
   cancellation**.
3. **Is there a strong incumbent that "corrects" the normalizer?** If so, the
   theory predicts the correction *hurts*, which is contrarian and falsifiable
   in both directions. That is worth more than a small win.

A fourth, practical filter: an unverified gate is a dead experiment. The
τ-family and its Poisson-regression estimator (scoping doc §1.1–1.2) should be
run as a **cheap screening instrument on any candidate corpus before committing
to it**. τ=0 is the unconditional model, τ=1 is the gated one; a few thousand
instances separate them. Every proposal below should be τ-screened first.

---

## 1. Clinical event streams — informative missingness in MIMIC-IV *(new)*

**Why it is not in the existing list.** Nothing in §3.3 is a *temporal* task.
Yet the evolution operator is the single largest hole in the thesis: it is
absent from ch.03 (scoped out), ch.04 (deferred), and every experiment run so
far. The theory for it already exists and is strong —
`tb_update_generalized.md` §13 gives a bounded steady state instead of `M²`
growth (§13a), an exactly-sufficient factorized state under gating (§13d), and
the silence correction (§13e) — and none of it has ever met data.

**Why the pipeline shape matches.** A lab test is ordered *because a clinician
suspects something*. The gate depends on the patient's latent acuity, not on
recorded observables — filter 2, the hard nonignorable case. The number of
tests ordered varies with suspicion — filter 1. And a test *not* ordered is
evidence of a well patient, which is exactly §13e's silent window.

**The incumbent is the interesting part.** GRU-D handles this with a learned
mask plus an exponential decay toward the empirical mean, and transformers match
it once given missingness indicators. Both are hand-engineered. §13e *derives*
the correction — a constant drift `q ← q + c_s` per silent window, computed once
per model, which recovered 80–90% of the closable gap synthetically. The claim
is not "we beat GRU-D"; it is "GRU-D's decay is the affine part of
`log(1 − Z/C)`, and here is the derivation it never had."

**Design.** Latent `x` = a small set of binary physiological states; symbols
`k` = ordered-and-resulted lab/vital events over a fixed vocabulary; windows =
hourly bins; task = in-hospital mortality or decompensation. Arms: additive
update; additive + gauge fix; additive + silence drift; GRU-D; transformer with
missingness indicators; and a no-silence ablation. Report AUROC/AUPRC **and
calibration** — the crowd-aggregation experiment already found calibration is
where this update family wins, and that is consistent with the theory rather
than a coincidence.

**Scope the novelty carefully.** "Silence is evidence" is not new: event-triggered
state estimation with *negative information* is a developed subfield that
computes the non-transmission likelihood properly, and the scoping doc's §3.2
already flags it. The defensible claim is narrower and better: under a `Z`-gate
the *positive* channel is exactly conjugate and needs no correction at all,
while the *negative* channel needs exactly one constant vector. That
decomposition is the contribution.

**Risk.** PhysioNet credentialing (CITI training) is real calendar time. Smoke-test
the pipeline on the open MIMIC-IV demo subset first, and treat credentialing as a
blocking prerequisite to be started immediately or not at all.

---

## 2. Self-initiated measurement in RL with costly observations *(new)*

**First, the honest negative on the obvious framing.** Vanilla POMDP RL — an
agent maintaining a belief while the environment emits an observation every
step — is a *bad* fit, and the repo's own numbers say so. Unconditional
measurement is the τ=0 regime, exactly where the additive rule is the worst
update rule available and degrades as `M²` (ch.03's `M`-sweep: recovery
97.0% → 72.6% from `M`=1 to 8; `tb_update_generalized.md` §10 table A). Adding a
Heisenberg belief to a standard POMDP benchmark should be expected to lose. That
is the reason to *not* do the experiment as first conceived.

**The framing that inverts it.** The advantage appears only when the agent
controls *when* it observes. If the agent opens a concept window with
probability proportional to `Z(x)` — the total drive its state delivers to the
index layer, i.e. how strongly the situation matches anything nameable — then by
§6 its own cheap `O(n)` additive update is *exactly* Bayesian, and by §13d the
whole filter is exact and the factorized state is sufficient rather than lossy.

> The agent is not approximating a belief. It is choosing the measurement policy
> that makes its own cheap inference exact.

That sentence is the chapter, and it is a direct answer to QTB §12.1.3 ("what
initiates a measurement?"), which the paper leaves open and which your unwritten
self-initiated-measurement chapter is already aimed at.

**Why now.** There is live 2026 work on agents that decide when to observe —
action-triggered observations, cost-sensitive selective measurement, active
tactile perception — so the setting has a community and baselines, and none of
it has a conjugacy argument for the belief update.

**Design.** A POMDP with an explicit per-observation cost. Cross two factors:
*trigger* (always-observe / learned trigger / `Z`-gated trigger) × *belief*
(additive `O(n)` / recurrent GRU encoder / exact filter where enumerable). Keep
the state small enough to enumerate the exact filter so belief KL is measurable,
not just return. Predictions: the `Z`-gated × additive cell matches the exact
filter to numerical precision while the always-observe × additive cell degrades
with the observation budget; and the additive belief reaches the recurrent
encoder's return at a fraction of the training steps, because it has no belief
representation to learn.

**Reuse.** The `tb-agency` worktree already has gridworld/MiniGrid/MemoryMaze
infrastructure and 138 REINFORCE runs — the trigger and cost machinery is the
only genuinely new code.

**Risk.** RL variance demands ~6 seeds; this is the most cluster-hungry proposal
here, and the weakest one to attempt under deadline pressure.

---

## 3. Recommenders with latent-dependent exposure *(sharpening of §3.3 ④)*

I agree this is the strongest *systems* fit and would not change the ranking.
Three additions.

**Add a within-dataset paired contrast.** The COCO design's real strength is that
captions and instance annotations give two annotation channels over the *same*
images and the *same* latent. KuaiRec supplies the same trick for
recommendation: a **fully-observed** user–item matrix (7,176 users × 10,728
items) alongside ordinary MNAR logs. That is a gated arm and an exhaustive arm
over one population — the paired test the simulated §10 scene A/B could not
provide, and which §1.4 of the scoping doc showed was confounded there by prior
misspecification. Open Bandit Dataset remains the best fit for *logged
propensities*; KuaiRec is the better fit for *exhaustive ground truth*. Use both;
they fail differently.

**Yahoo!R3 and Coat as the cheap third arm.** Both pair an MNAR training set
with a genuinely MAR test set (Yahoo!R3: 311,704 MNAR ratings, 54,000 MAR;
Coat: 290 users × 300 items, 24 self-selected vs 16 randomly exposed). Small
enough to run on the laptop, and they are the standard debiasing benchmarks, so
the comparison is legible to that community.

**State the disagreement precisely.** The contrarian claim is not "debiasing is
wrong." It is that for the specific gate `π ∝ Z(x)`, the correction is already
paid by cancellation, so applying IPS or the two-tower `logQ` correction on top
*double-corrects* and should measurably hurt. This is the same shape as the
verified synthetic result that applying unconditional corrections to a gated
corpus damages it (affine −0.138, gradient −0.424 nats). Run the τ estimator
first: if a corpus screens as τ≈0, the prediction is the opposite, and reporting
that is still a result.

---

## 4. The delta rule as the categorical branch *(endorsement of §3.3 ①)*

Highest ceiling of anything on either list, and the only candidate that touches a
genuinely SOTA-2026 object. I checked the load-bearing claim independently and it
holds up: modern linear-attention states update
`S ← S + βₜ(v − Sk)kᵀ`, structurally identical to `q ← q + A(e_k − p)`, and the
Bayesian reading of that family is already taken **in the Gaussian branch** —
Gated KalmaNet (arXiv:2511.21016) states that DeltaNet, Gated DeltaNet and Kimi
Delta Attention are approximations to the Kalman recurrence under an identity
error-covariance assumption.

What is unoccupied is *when the assumption is true*. §8b answers it: the
covariance stays isotropic, and the delta rule is exact rather than approximate,
exactly when the accumulated keys form a **tight frame** — with a closed-form
optimal gain `β* = 1/(1 + cτ²/σ²)` that is the architecture's own gate, not a
fitted hyperparameter, and a cheap diagnostic `‖KKᵀ − cI‖_F` predicting where
full-covariance machinery buys nothing.

Two cautions I would keep in front of this one. Respect Trap 2 — the 2024–2026
arc is explicitly about *destroying* commutativity to get state tracking, so
order invariance must never be pitched as a token-mixer advantage. And this is
the proposal most exposed to being scooped or pre-empted between now and
submission; it is a paper, not a thesis chapter.

---

## 5. Neural collapse ⟺ exactness over training *(endorsement of §3.3 ③)*

Best effort-to-reward ratio on either list, and the one I would actually run
under a deadline. All ingredients are published and the composition is not made:
neural collapse says trained classifier readouts converge to a simplex ETF; §8b
says a tight-frame readout makes the additive update exactly Bayesian. The
prediction is a single curve — Heisenberg-vs-Bayes fidelity should improve
monotonically over training, tracking the neural-collapse metric.

**The reason to prefer it is that it resolves a contradiction you already own.**
The synthetic experiment found that training *raised* `Var[log Z]` (0.025 →
0.118), and concluded that learning does not drive the model into the exact
regime. Neural collapse predicts the opposite for a classification objective
trained into the terminal phase. Those cannot both be general. Whichever way it
resolves is a result, and it retires an outstanding negative in the thesis
either way.

**Design.** Take classifier checkpoints across training (CIFAR-100 locally,
ImageNet on the cluster), and at each checkpoint measure `Var[log Z]`, the
affine fraction, the simplex-ETF/neural-collapse metric, and the belief error of
the additive update against exact Bayes on a small enumerable latent. No new
model, no new data pipeline, no RL variance, no credentialing. Hours of cluster
time, not days.

---

## What I would drop

- **Fixed top-`k` RAG.** Trap 1 is correct and decisive: a constant-count gate is
  the multinomial side of the Poisson trick and the normalizer does not cancel.
  The `Z`-gate lives only where the recorded *count* varies with score mass —
  threshold retrieval, abstention gates, agentic loops that decide whether to
  retrieve again. Stating that boundary is itself worth a paragraph in the
  thesis; pitching fixed-`k` RAG as the application would be an error.
- **Vanilla POMDP belief updating.** See §2 — the wrong half of the τ-family.
- **Model merging / task arithmetic.** Saturated, and the gauge freedom has no
  weight-space analogue. The scoping doc is right to drop it.

---

## Feasibility, stated plainly

`thesis/new_version/dbstmpl.tex` carries `\abgabetermin{03.09.2026}` — 16 days
from this document. Against that: chapters 01, 02 and the appendix do not exist,
ch.04 has four rival drafts with the choice unmade, and `new_version` ch.04
(line 266) still gives the exactness condition as "$Z(x)$ does not depend on the
state" — sufficient, but weaker than the sharp affine criterion established in
`tb_update_generalized.md` §4.

None of the five above fits in that window as a *new* experimental chapter. If
the date is real, the honest ordering is:

1. Finish the written chapters and pick a ch.04 draft.
2. Use the **already-built COCO experiment** as the experimental chapter — it is
   run, tested to 2.2e-15 against reference rules, and contains the first
   implementation of ch.04's derived gauge fix, which wins at every `M`. Its
   designed-but-unrun instance-channel arm is the paired gated-vs-exhaustive
   test and is the single highest-value remaining unit of work.
3. Add §5 (neural collapse) only if time genuinely remains.

Treat §1 (MIMIC) and §2 (self-initiated RL) as the post-thesis programme. They
are the two that fill the evolution-operator hole, which is the most defensible
reason to keep working on this after submission.

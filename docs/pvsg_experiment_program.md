# PVSG experiment program: what the original Tensor Brain paper could not test

## Purpose

This document evaluates the original Tensor Brain experiments (Sections 6, 7, 9 and 10 of
*The Tensor Brain*, extended version) against what the PVSG snapshot, the reviewed object
hierarchy, and persistent `(video_id, object_id)` identities now make measurable. It then
recommends a concrete experimental program.

It is a research-design document. It does not restate the data contract, which lives in
[PVSG perception and memory experiments](pvsg_perception.md), and it does not repeat the
paper-mechanics program in [the QTB WIP review](qtb_wip_review_2026-07-22.md). Those two
documents answer *how the data is built* and *which equations need small exact tests*. This one
answers a different question: **given that we now have real video with real individuals, which
claims of the Tensor Brain line are currently unsupported, and which experiments would settle
them.**

The short version of the assessment:

- The original paper's *conceptual* contribution is large and mostly untested. Almost everything
  it says about time — recent versus remote episodic memory, recency-triggered recall, object
  permanence, state change, forecasting, memory-supported decision — is illustrated with figures
  and stories, never measured. VRD is a static image dataset; it structurally could not measure
  any of it.
- Several *quantitative* claims that were measured are confounded in ways PVSG removes. The two
  most important are the identity-recognition result (measured on geometric distortions of the
  training images) and the "the dynamic context layer is essential" result (measured against a
  baseline that also had less information).
- The most valuable experiments are therefore not reproductions. They are (a) three or four
  cheap repairs that make the original claims either survive or fail honestly, and (b) a set of
  temporal, capacity, and continual-learning experiments that were simply not possible before.

---

## 1. What the original paper actually tested

For reference, the measured results, stripped of narrative:

| Table | Claim | Evidence |
|---|---|---|
| 5 | Index feedback improves unary labels | VRD-E: P-SA 78.09 vs P-Direct 77.97 average. VRD-EX: P-Samp 95.93 vs P-Direct 89.56 |
| 4 | The dynamic context layer is essential for binary labels | VRD-E @1: P-Direct 31.68, P-Samp 45.09, P-SA 46.84 |
| 6 | Generalized statements generalize; superior zero-shot | z-s-rl: P-SA 81.61 vs BFM 76.05 |
| 7, 8 | Semantic memory recalls entity and class knowledge | SM-givenEntity 100.0 across all unary columns |
| 9 | Semantic memory enriches perception with nonvisual labels | `Dangerous`: P-SA 52.01, P-enriched 98.24 |
| 10 | Multimodal/social facts integrate into memory | VRD-S friend retrieval @10 97.39 vs RESCAL 95.76 |
| 12 | Entity indices matter for episodic recall | EM 100.0 unary; P-noI (no entity indices) 12.97–71.05 |
| 13 | Self-supervised learning does not damage old knowledge and slightly helps new | +SSL vs SL on generalization: 75.08 vs 74.78 |

Everything else in Sections 9 and 10 — the lurking bear, the broken coffee machine, remote
episodic memory guiding action, future episodic memory, replay-based consolidation, forgetting —
is theory plus an illustrative figure.

---

## 2. Where the original evidence is weak, and what PVSG changes

### 2.1 Identity recognition was measured on distorted duplicates

VRD-EX was constructed by applying translation, rotation, shear and horizontal flip to each
training image, and then a *second* distortion of the same images to make the test set. The paper
states plainly that every test entity "has already occurred in the training set twice", and that
the VRD-E to VRD-EX improvement is "likely due to some form of memorization by overfitting".

So the headline entity accuracy of 92.81% measures whether a network can match an affine-distorted
copy of a bounding box it has already seen. It does not measure re-identification of an individual
across genuine change in pose, scale, viewpoint, illumination, deformation or partial occlusion.
Nothing in VRD could have measured that: an "entity index" in VRD-E is one bounding box
(26,430 boxes, 26,430 entity indices), so most individuals had exactly one observation.

**PVSG changes this categorically.** Identities persist across time with hundreds of genuinely
different observations, and the `blocked` protocol (train on the first 45% of a video, embargo
10%, evaluate on the last 45%) makes recognition causal and temporally separated. The audited
snapshot contains 394 videos, 147,795 frames at 5 FPS, 1,495,227 object observations and
7,143 retained tracks under the reviewed hierarchy. This is the single most important upgrade
available, and it is available today with no new annotation.

### 2.2 "The dynamic context layer is essential" is not a matched-information comparison

The Table 4 gap (31.68 to 46.84 @1) is attributed to the dynamic context layer. But P-Direct
predicts the binary label from the predicate bounding box alone, while the BTN transports scene,
subject and object evidence into the same decision. The comparison therefore conflates three
different things: access to more evidence, serialized recurrence, and index feedback. The paper
even acknowledges the mechanism informally ("information on the subject and the object bounding
boxes is required"), which is a statement about *information*, not about *recurrence*.

`docs/pvsg_perception.md` already flags this. It should be promoted from a caveat to the first
experiment, because every later claim about TB mechanisms inherits this confound.

### 2.3 `Dangerous` / `Harmless` is a relabeling of animacy

The nonvisual-enrichment result (Table 9) uses a label defined as: `Dangerous` for all living
things, `Harmless` for all non-living things. But the model already predicts the G-Class
`LivingBeing` at 98.2–98.47% accuracy. `P-enriched` reaching 98.24% on `Dangerous` is therefore
approximately the same number, obtained from approximately the same information. The experiment
does not demonstrate transport of knowledge that perception cannot supply; it demonstrates that
a deterministic function of a well-predicted label is also well predicted.

The semantic inventory already declines to reproduce this and instead defines
`can_cause_physical_harm` as a positive, non-derivable fact with unknown-by-omission semantics.
That is the right correction, and it makes a real version of the experiment possible.

### 2.4 The `Young` / `Old` labels were randomly assigned

By construction, `Y/O` is unpredictable for novel entities (VRD-E: 47.88–50.60, i.e. chance) and
predictable only for known ones (VRD-EX: 93.10–94.58). This is a clean probe of "is there an
identity-bound memory at all", and it is fine as such — but it is a synthetic lookup task, and it
is the *only* evidence in the paper for identity-bound non-visual facts. With PVSG, the same
question can be asked with facts that are real, identity-stable and not visually derivable at
query time (an object's colour or material recalled while it is occluded, its owner, its typical
affordance).

### 2.5 The class hierarchy had a catch-all at every level

The VRD-E ontology assigns the default class `Other` at any level where WordNet mapping failed;
15% of all labels are `Other`. Levels were also reported as three independent top-1 accuracies
(B-Class, P-Class, G-Class). Two consequences:

- accuracy is inflated by a large, semantically empty class;
- there is no measure of whether the three predictions are *mutually coherent*. Three separate
  accuracies say nothing about whether the model's fine label is actually a child of its
  predicted coarse label.

Your hierarchy is built precisely to remove both problems: 121 fine, 78 basic, 22 coarse and 5
domain labels, pairwise disjoint, strictly decreasing in size, no `object`/`thing`/`other` root,
with strict "is a kind of" paths for the three category levels and `domain` explicitly a broad
partition rather than a hypernymy claim. This makes hierarchical *consistency* measurable for the
first time in this line of work, and it makes the ordered readout experiments in Section 4.2
below well posed.

### 2.6 Statistical reporting

Single runs, no seeds, no confidence intervals, and best-in-column bolding on differences as small
as 0.12 points (Table 5, VRD-E average: P-SA 78.09 vs P-Direct 77.97). Several of the paper's
qualitative conclusions rest on gaps well inside plausible seed noise. This is cheap to fix and it
is the first thing a 2026 reviewer will check.

### 2.7 Everything temporal was asserted, not measured

Sections 9.6 to 9.9 and 10.4 to 10.6 are the conceptually strongest part of the paper and contain
no experiments:

- recent episodic memory as a substitute for current perception (the lurking bear);
- recall triggered by *recency* and relevance;
- semantic memory as a "sluggish state estimator" that lags real state changes;
- remote episodic memory retrieved by scene similarity and used for decisions;
- forecasting and future episodic memory;
- consolidation by replay, forgetting, and the claim that "catastrophic forgetting does not show
  up in our preliminary experiments".

Each of these becomes a measurable curve on PVSG. This is where the thesis contribution is.

---

## 3. What your PVSG assets specifically unlock

| Asset | Already have | Unlocks |
|---|---|---|
| Persistent `(video_id, object_id)` identities, ~7,143 retained tracks | yes | real re-identification, per-individual learning curves, index capacity, few-shot enrollment |
| Four-level disjoint hierarchy, no catch-alls | yes | hierarchy consistency metrics, ordered readouts, level-graded compositional generalization |
| 5 FPS time base, 147,795 frames, relation spans with onset and cessation | yes | recency curves, state-change lag, anticipation, episode formation |
| Naturally missing evidence: 133,698 records missing subject, 99,323 missing object, 184,650 missing union (26.7% of positive-pair records do not join completely) | yes | occlusion and object permanence, with no synthetic corruption needed |
| 1,114 training category triples; 33,576 of 127,895 validation predicate assignments involve a category triple unseen in training | yes | a compositional generalization test roughly two orders of magnitude larger than VRD's zero-shot split |
| Exact panoptic masks | yes | compositing interventions (unexpected-object experiments), region ablation |
| 19,859 multi-predicate records | yes | sequential multi-label measurement (underpowered on its own; see 6.3) |
| Identity-stable colour/material/shape, affordance, risk, ownership | inventory defined, facts not yet asserted | a non-degenerate replacement for `Dangerous` and for `Young`/`Old` |

Two properties of the label distribution constrain everything below and should be fixed as
reporting policy now: `on` accounts for 257,431 of 712,132 predicate assignments (36%), while
`jumping over` has 55 and `grabbing` 80. Micro-averaged predicate numbers are close to
meaningless on this dataset. Every predicate result should lead with macro-AP over the
train-supported labels, carry per-predicate support, and be compared against a
predict-the-marginal-prior baseline.

---

## 4. Recommended program

Four tiers. Tier A repairs the original claims and is a prerequisite for believing anything else.
Tier B is the genuinely new science. Tier C positions the work in 2026 deep learning. Tier D is
cheap data enrichment that unlocks specific Tier B/C items.

### Tier A — repairs (do these first; all cheap, all on cached features)

#### A1. Decompose "the dynamic context layer is essential"

Train a ladder in which every model sees the *same four evidence sources* (scene CLS, subject
mask-pooled, object mask-pooled, union region) and only the mechanism changes:

| Model | Mechanism | Isolates |
|---|---|---|
| M0 | union feature only, MLP head | the original P-Direct information condition |
| M1 | concatenate all four features, MLP head, no TB operations | **information**, the missing control |
| M2 | TB schedule with evolution between concept windows, `feedback_gate = 0` | **serialized recurrence** |
| M3 | TB with expected identity feedback (P-SA) | **index feedback** |
| M4 | M3's checkpoint evaluated with argmax feedback (P-Samp) | sampling vs attention approximation |

Attribution: `M1 − M0` is information, `M2 − M1` is recurrence, `M3 − M2` is index feedback,
`M4 − M3` is the approximation gap. Report macro-AP, per-example Recall@K, and per-predicate
support, plus a prior-only baseline.

This experiment can produce a negative result — the honest and quite likely outcome is that most
of the paper's 15-point gap is `M1 − M0`, i.e. information rather than mechanism. That is worth
publishing and it is the correct foundation for every later claim. Matched parameter count,
matched optimizer, matched schedule, ≥5 seeds.

#### A2. Identity recognition without the distortion confound

Use `blocked`, not a random frame split. Report identity Hits@1/5/10 and MRR stratified by frames
since the last training exposure. Add a second stratification by *appearance drift*: cosine
distance between the query object feature and the mean of that identity's prefix features. This
directly separates "the model remembers the individual" from "the query looks like a stored
template", which is exactly the distinction VRD-EX could not draw.

Two mandatory baselines:

- **frozen-DINO nearest centroid** over the same prefix observations. If the TB identity columns
  do not beat this, there is no identity memory claim to make; there is a feature-space claim.
- **random-frame split**, reported only as a leakage diagnostic, to quantify how much the
  original-style protocol inflates the number.

#### A3. Hierarchy consistency

The original reported three independent accuracies over a vocabulary containing a 15% catch-all.
Report instead, on the disjoint four-level vocabulary:

- per-level top-1 and macro-F1;
- **path consistency rate**: fraction of examples where the argmax fine, basic, coarse and domain
  predictions form a legal path in the reviewed hierarchy;
- hierarchical precision / recall / F1;
- accuracy at each level conditioned on the parent level being correct;
- per-level calibration (ECE, NLL).

Conditions: independent readouts (the paper's setup) versus sequential readouts with index
feedback. The interesting claim to test is not that feedback raises accuracy — the paper's own
numbers suggest that effect is around a point — but that **feedback raises coherence**. The paper
argues qualitatively that a detected `Sparky` biases the unary labels toward what is known about
Sparky. Path consistency is the metric that would actually show this, and it has never been
reported.

Prerequisite: run the frozen-DINO level-wise probe diagnostic already specified in the hierarchy
document first. If DINO cannot linearly separate the fine level, hierarchy results measure the
encoder, not the Tensor Brain.

#### A4. Statistical hygiene

≥5 seeds, bootstrap confidence intervals over evaluation records, paired per-example tests for
P-SA vs P-Samp vs P-Direct (they share evaluation records, so paired tests are both valid and much
more powerful), and a prior-only baseline in every predicate table. Cheap, and it converts several
of the original paper's sub-point differences into either real effects or acknowledged noise.

### Tier B — new capability (the thesis contribution)

#### B1. Order effects in perception — flagship

QTB Sections 7.3 and 12.2 make sharp, falsifiable predictions that no Tensor Brain paper has
tested on real perceptual data:

- neural PVM (`retain_gate = 0`, `feedback_gate = 1`) exhibits an order effect within a concept
  interval, because the CBS is dominated by the most recently activated index;
- neural HB-POVM (`1, 1`) does not exhibit a posterior order effect;
- causal postselection — restricting candidates conditioned on earlier outcomes — induces a
  *likelihood* order effect even for HB-POVM;
- Section 12.2.4 claims empirically that identifying specific concepts first and general ones
  after works better, and offers this as an argument for symbolic indices.

Your four-level hierarchy is the ideal substrate, and it is strictly better than the static
ImageNet-hierarchy version referenced in the WIP: the levels are disjoint, catch-all-free, human
reviewed, and attached to individuals observed repeatedly over time.

Design: one object concept window, crossed factors.

- **readout order**: fine → basic → coarse → domain; reversed; random permutation.
- **gates**: `(1,1)` HB-POVM, `(0,1)` PVM, `(1,0)` no feedback.
- **candidate policy**: full level vocabulary versus restriction to the descendants of the
  previous outcome (causal postselection).

Measures: KL divergence between the final joint label distributions under different orders;
accuracy asymmetry per level; and, for the restricted-candidate condition, the error-propagation
rate — how often a wrong coarse outcome makes the correct fine label unreachable. That last number
is the practical cost of causal postselection and it is a genuinely useful result for anyone
building hierarchical or constrained decoders.

Extension with no extra data: subject → object → predicate versus object → subject → predicate.
Order effects on *relational* readout appear nowhere in the TB line.

#### B2. Occlusion and object permanence — the lurking bear, measured

The resource already exists in your audit and needs no synthetic corruption: 26.7% of positive-pair
frame records do not join to complete evidence, with 133,698 missing subjects and 99,323 missing
objects. These are naturally occurring disappearances of a tracked individual that reappears.

Task: maintain identity and relation state across a gap in visual evidence, then be evaluated on
reappearance. Conditions:

1. P-Direct — must fail; it has no state. This is the one place where P-Direct *is* a fair
   baseline, because the claim is about state, not information.
2. TB with dynamic context, no index feedback.
3. TB with dynamic context and identity index feedback.
4. TB with an episodic index written at the last visible frame.

Report accuracy as a function of gap length in frames. The lurking-bear claim becomes a decay
curve, and the difference between conditions 2 and 3 is a direct measurement of whether symbolic
index feedback — not merely recurrence — is what carries the individual across the gap.

Design constraints: your fidelity ledger forbids silently padding absent evidence, which is
correct. Make absence explicit as a named condition (zero drive plus an availability flag, or a
learned "absent" drive), and report it. Also ablate the scene CLS token, since a model can cheat
by inferring "the dog is probably still in this kitchen" from the scene alone; that is an
interesting result but it is not object permanence.

#### B3. Relation state change and the lag of the semantic prior

The perception document already warns that persistence can dominate the headline anticipation
number. Turn the caveat into the experiment.

Stratify every predicate prediction into: onset frame, mid-span persistence, cessation frame, and
the window immediately after cessation. Then measure the quantity the paper describes narratively
in Section 9.9 and never quantifies: **how many frames does the model take to stop asserting a
relation after it ends?** That is the state-estimator lag of the model, and it is exactly the
"semantic memory is a sluggish state estimator, recent episodic memory carries the change"
hypothesis in numeric form.

Cross with gate regimes. QTB predicts that PVM `(0,1)`, whose CBS is dominated by the most recent
index, should update faster after a change but retain less; HB-POVM `(1,1)` should be more stable
but laggier. If that trade-off appears on real video, it is a clean empirical validation of the
gate theory, obtained with no new data and no new model.

#### B4. Level-graded compositional generalization

The original reports one zero-shot number on unseen `(subject-class, predicate, object-class)`
triples. You have 1,114 training category triples, and 33,576 of 127,895 validation predicate
assignments involve a triple unseen in training — about 26% of the validation signal.

With four levels, novelty can be *graded* rather than binary:

- fine triple unseen, basic triple seen;
- basic triple unseen, coarse triple seen;
- coarse triple unseen, domain triple seen;
- domain triple unseen.

Hypothesis: a model whose knowledge lives in a shared bidirectional `A`, with generalized
statements over class indices, should degrade gracefully along this ladder, while a flat fusion
model should fall off a cliff. Whether or not the hypothesis holds, a graded abstraction-level
generalization curve is a result of independent interest to the compositional-generalization
literature, and it is only possible because your hierarchy is strict and disjoint.

Check per-stratum support before committing; the deepest strata may be too small to report.

#### B5. Per-individual learning curves and few-shot index recruitment

In VRD-E each entity had one observation, and in VRD-EX about three near-duplicates. The question
"how much evidence does an individual index need" was therefore unanswerable. On PVSG each
identity has tens to hundreds of genuinely varied observations.

Using the `fewshot` protocol (enroll a new identity column from its first k mask-visible
observations, adapt only that column, embargo 25 frames before queries):

- sweep `k ∈ {1, 2, 5, 10, 25, 100}` and plot identity Hits@k against exposure count;
- measure **interference**: does enrolling 500 new identities degrade previously learned ones?
  Evaluate the old identities before and after enrollment;
- baselines: DINO nearest-centroid and k-NN over cached features with the same k support frames.

This is the Tensor Brain instance of a question the field cares about in 2026 — weight-based fast
memory versus non-parametric retrieval — and it operationalizes the complementary-learning-systems
story that Section 10.1 tells but does not test.

#### B6. Index-layer capacity

Section 7.6 and 10.1 claim that high dimensionality, sparsity and modularity of the embedding
vector give robustness, locality of updates, and freedom from catastrophic forgetting. The paper
fixes `r = 4096` and measures none of this.

Sweep `state_dim ∈ {128, 256, 512, 1024, 2048, 4096}` against the number of enrolled identities
(100 up to the full 7,143 retained tracks) and plot the interference surface. Add the sparsity
condition the paper mentions in passing (Lasso on `A`, reported to reach 70% sparsity) and test
whether sparsity actually raises capacity as claimed.

This is very cheap — it reuses the cached features and touches only the index layer — and it
produces an empirical capacity scaling law for a bidirectional index memory. It connects directly
to current work on memory layers, product-key memories and superposition, which makes it the
easiest result in this program to position outside the Tensor Brain community.

#### B7. Continual learning on the video stream

The paper's forgetting claim rests on one row of Table 13 and the sentence "catastrophic
forgetting does not show up in our preliminary experiments". PVSG provides a natural non-i.i.d.
stream: 394 videos, each with its own identities, scenes and relation distribution. Train
video-by-video in order, without reshuffling.

Report standard continual-learning metrics: average accuracy, backward transfer (forgetting),
forward transfer, per-video retention curves. Conditions:

- plain streaming SGD;
- **replay of episodic index activations** rather than raw data. This is what Section 10.4
  actually proposes — an episodic index is activated, the representation layer is populated with
  `a_t`, and a neocortical index learns from that activation — and it is a distinctive, testable
  and modern claim (replaying embeddings, not stored inputs);
- freeze the static prefix of `A` and grow only identity columns.

Either the paper's no-forgetting claim survives a real stream, or it does not. Both outcomes are
publishable and neither is currently known.

#### B8. Self-supervised index recruitment, verified against ground-truth tracks

The original SSL experiment (Table 13) pseudo-labels the second half of the training images and
measures label accuracy. It cannot answer the more interesting question: when the agent decides
an entity is novel and recruits a new index, is the recruited index *right*?

On PVSG you can answer it. Run the model over unlabeled videos, apply the paper's own novelty
criterion (all identity activations below a threshold, footnote 6), recruit a new identity index,
self-train on winner-take-all outcomes, and then evaluate the recruited indices against the
ground-truth `object_id` tracks that PVSG supplies: cluster purity, completeness, V-measure,
over- and under-segmentation of individuals. VRD had no such ground truth for invented indices.

This is the strongest available version of "to perceive is to learn" and, as far as I can tell,
nothing equivalent exists in the literature for this model family.

### Tier C — modern positioning

#### C1. One contemporary reference frame

The paper compares against RESCAL and 2016–2017 VRD methods. To be legible in 2026 you need at
least one modern reference point — not to win, but to characterize where the TB inductive bias
pays. Minimum set, all on identical cached features:

- frozen DINO + linear probe (feature-quality lower bound);
- a small transformer over the same four evidence tokens at matched parameter count (the modern
  form of "flat fusion with attention");
- optionally a frozen VLM prompted on the union crop, as a zero-shot reference.

My expectation is that the transformer matches or beats TB on headline predicate macro-AP, and
that TB's advantages, if they exist, appear in few-shot identity binding (B5), stability under
occlusion (B2), continual learning (B7), and calibration (C2). Stating that division of labour
explicitly is far more credible than claiming a uniform win, and it is the framing that makes the
work interesting rather than defensive.

#### C2. Calibration and probability quality

The Tensor Brain is presented as a sampling engine whose CBS is a factorized Bernoulli
approximation to a posterior, yet the original paper reports only accuracy and Hits@k. Report ECE,
NLL and Brier per level and per predicate; compare P-SA and P-Samp on entropy, hard/soft agreement
and calibration (your metrics list already anticipates this); and evaluate calibration under the
distribution shift induced by `heldout_video`.

This is also where the WIP review's Priority-0 exact-Bayes question gets a real-data grounding:
for small candidate sets the exact enumerated posterior is computable, so the TB logit-addition
update can be compared against exact Bayes, factorized Bayes and mean-field on actual perceptual
evidence rather than only on a synthetic Bernoulli problem.

#### C3. Sequential multi-label predicate measurement

Instead of one multi-hot readout, measure predicates one at a time with feedback and a stop index.
This tests the serialization hypothesis of Section 8.5 — that a single global representation layer
forces sequential symbolic readout — and asks whether feeding back the first predicate improves
the second.

Caveat up front: only 19,859 records (2.9%) carry multiple predicates, and the co-occurrence
structure is dominated by a few pairs involving `on` and `holding`. Compute the co-occurrence
matrix before investing; this experiment may be underpowered, and if so it should be reported as
such rather than stretched.

#### C4. Top-down candidate restriction as inattentional blindness

Causal postselection has an appealing empirical face: restricting the candidate set by scene
context should improve detection of expected objects and *cause misses* of unexpected ones. With
exact panoptic masks you can composite an unexpected object into a video, or simply manipulate the
candidate group, and measure the trade-off curve directly.

This makes a QTB-specific mechanism visible as a phenomenon that a general audience already knows
from cognitive science, at low implementation cost. It is a good final chapter or demo; it is not
a prerequisite for anything else.

### Tier D — cheap enrichments, ranked by unlock-per-cost

1. **Category-typical affordance and risk facts.** About 121 fine classes times five facts, one
   review pass. Unlocks a non-degenerate replacement for the `Dangerous`/`Harmless` experiment:
   test whether `P-enriched` really transports a fact that perception cannot supply, using a fact
   that is *not* a deterministic function of the predicted class. This directly repairs 2.3.
2. **Identity-stable colour, material and shape** via the VLM-over-temporal-mosaic procedure
   already specified in the semantic inventory. Unlocks the real version of the `Young`/`Old`
   probe (2.4): recall a property of a known individual while that individual is occluded — which
   composes with B2 into one of the more striking experiments available here.
3. **Caption-derived ownership and kinship triples.** Unlocks a non-synthetic version of the VRD-S
   social-network experiment. Medium cost, medium payoff; the original's version was fully
   synthetic, so even a small real subset is an improvement.
4. **Reward or valence tags on a small set of episodes.** Unlocks memory-supported decision
   (Section 9.7), which is conceptually the most valuable untested claim in the paper. Also the
   highest annotation risk and the least well-defined. Defer until Tier B is running.

---

## 5. Suggested order

**Phase 1 — credibility.** A1 decomposition, A2 identity under `blocked`, A4 statistics. Outcome:
you know how much of the original effect is mechanism rather than information, and every later
number has error bars. This phase can already be written up as a rigorous replication-and-repair
study.

**Phase 2 — the temporal contribution.** B2 occlusion, B3 state-change lag. Both reuse the same
runners as Phase 1 with different record strata. Outcome: the first quantitative evidence for the
paper's central temporal claims.

**Phase 3 — the hierarchy contribution.** A3 consistency, then B1 order effects, then B4 graded
compositional generalization. These share the hierarchy vocabulary and readout code. Outcome: the
flagship result and the one most tightly coupled to the QTB gate theory.

**Phase 4 — memory as a system.** B6 capacity (cheapest), B5 few-shot recruitment, B7 continual
learning, B8 self-supervised recruitment. Outcome: the "index layer as a memory system" story,
which is the most legible to a general deep-learning audience.

**Phase 5 — positioning and extras.** C1 baselines and C2 calibration should be folded into
Phases 1–4 as they go rather than done at the end. C3 and C4 are optional.

Tier D item 1 should start in parallel with Phase 1 because it is a review task, not a compute
task, and item 2 should start once Phase 2 shows the occlusion protocol works.

---

## 6. Risks that could invalidate results

1. **Predicate long tail.** `on` is 36% of all assignments; fifteen predicates have fewer than 500.
   Micro-averaged numbers will look impressive and mean nothing. Commit to macro-AP plus
   per-predicate support tables as the headline format before running anything.
2. **Positive-unlabeled negatives.** Already flagged in the perception document: it is not
   established that every unannotated visible pair is a true negative. Do not build headline
   all-pair relationship-prediction or anticipation numbers before that audit. Positive-pair
   predicate recognition, B1–B4 and B6–B8 are unaffected.
3. **Frozen-encoder ceiling.** If DINOv3 features cannot separate the fine level linearly, then
   A3, B1 and B4 measure DINO rather than the Tensor Brain. Run the level-wise probe diagnostic
   first and report it alongside every hierarchy result.
4. **Identity metrics are protocol-dependent.** `heldout_video` has no closed-set identity metric
   by construction — every evaluation identity is novel. All identity claims must come from
   `blocked` and `fewshot`, and this must be stated wherever identity numbers appear, or the
   comparison across protocols will be misread.
5. **Extraction provenance is heterogeneous.** The snapshot spans 18 provenance groups with
   different processed sizes (240×448, 256×448, 272×448, …) from aspect-preserving resizing. This
   is by design, but keep the group as a covariate: if a per-video effect appears in any result,
   check it against provenance before interpreting it scientifically.
6. **Underpowered strata.** B4's deepest novelty stratum and C3's multi-predicate records may both
   be too small. Compute support before designing the table, and report "insufficient support"
   rather than a number with an invisible denominator.

---

## 7. What not to do

- **Do not chase PVSG leaderboard numbers.** The published PVSG relation pipeline solves a
  different problem (it pairs feature tubes using ground-truth relations). Competing with it is
  neither the contribution nor a fair comparison, and the oracle-mask setting makes any such
  comparison misleading in both directions.
- **Do not reproduce `Dangerous` / `Harmless`.** It is animacy under another name; reproducing it
  would import the original paper's weakest result.
- **Do not report random frame splits as headline results.** Only as leakage diagnostics, as the
  perception document already requires.
- **Do not introduce loss weights, extra capacity or architectural additions before A1.** Until
  the information-versus-mechanism decomposition exists, there is no way to tell whether a tweak
  improved the model or merely gave it more evidence.
- **Do not treat P-Direct as a general-purpose baseline.** It is a fair control only for
  state-dependent claims (B2). Everywhere else, M1 flat fusion is the honest comparison.

---

## 8. One-paragraph summary of the contribution this program supports

The Tensor Brain proposes that perception, episodic memory and semantic memory are operating modes
of one oscillation between a symbolic index layer and a representation layer. That proposal was
evaluated on a static image dataset with synthetically duplicated entities, randomly assigned
identity-bound labels, a catch-all-padded hierarchy, a non-matched baseline, and no temporal
structure at all. PVSG with persistent identities and a strict four-level hierarchy allows the
same claims to be tested where they are actually meant to apply: individuals recognized across
real time and real appearance change, relations that begin and end, evidence that disappears and
returns, and a stream of experience long enough for forgetting and consolidation to be measured
rather than asserted. The most valuable outcome is not a better number on a scene-graph benchmark.
It is a set of curves — recognition against temporal distance, relation assertion against state
change, capacity against dimensionality, retention against stream position, order effects against
gate configuration — that either support the theory or bound it. Both are worth having, and
neither exists today.

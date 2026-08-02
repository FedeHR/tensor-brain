# Core-capability experiments: an object-first thesis program

## Purpose and headline recommendation

This document defines the experimental program for testing the Tensor Brain as a model of
perception and memory, under a real time constraint and after substantial data-curation effort has
already been spent.

**The headline recommendation is a scope pivot: build the thesis on the object-observation stream,
and demote subject/object/predicate relations to a single late chapter.**

That is not a concession to time pressure. It is the better scientific choice, and the rest of this
document argues it. Nine of the original paper's ten core claims about perception and memory need
no relations at all, several of them are *better* tested without relations, and the single claim
that does need them — that the dynamic context layer matters — has a cleaner, information-matched
test on objects than the one the paper ran on predicates.

It also happens to discard, at a stroke, every unresolved data problem: the positive-unlabeled
question, the relation-span convention, span censoring, the `on`-dominated predicate long tail, and
the 26.7% incomplete pair joins. None of them touch the object stream.

Sections 1 and 2 establish the pivot. Section 3 is the prioritized experiment list. Section 4 says
what to cut. Section 5 is the schedule. Section 6 keeps the scale verification, which has become
*more* important under the pivot, not less.

---

## 1. Why object observations carry the thesis

### 1.1 The claim-coverage table

The original paper's core claims about perception and memory, and what each actually requires:

| # | Claim (original paper) | Needs relations? |
|---|---|---|
| 1 | Perception binds evidence to **individuals**, not only classes (§1, §6.5) | No |
| 2 | Top-down index feedback informs and biases perception (§6.4) | No |
| 3 | The dynamic context layer transports information between perceptual acts (§6.7) | **No — and better without** |
| 4 | Semantic decoding: an index's embedding *is* its engram, decodable into labels (§7.5, §7.7) | No |
| 5 | Episodic memory: an index per instance, recalled and decoded (§9.3, §9.4) | No |
| 6 | Entity indices are required for recall — the `P-noI` ablation (Table 12) | No |
| 7 | Semantic memory enriches perception with information perception cannot supply (§7.8, Table 9) | No |
| 8 | Embeddings organize into a conceptual space / cognitive map (§7.5, Figs 7–8) | No |
| 9 | Generalized statements: querying a *class* index decodes what the class is about (Table 7) | No |
| 10 | To perceive is to learn: new indices, self-supervised refinement (§10.3) | No |
| — | Binary-label prediction requires dynamic context (Table 4) | Yes |

Only the last line needs pairs. Everything the paper says about perception, individuals, semantic
memory, episodic memory and the cognitive map is unary.

This is not an accident of the paper's design. The Tensor Brain's central object is a *concept
index and its embedding*; relations are one kind of index among several. The relation machinery was
prominent in the original experiments because VRD is a visual-relationship dataset, not because the
theory demands it.

### 1.2 The object stream is larger, cleaner, and richer

| | Object stream | Pair stream |
|---|---:|---:|
| Observations | **1,495,227** | 8,330,261 raw / **506,501** complete joins |
| Completeness | complete by construction | 73.3% join rate |
| Mean per frame | **10.12 visible objects** | 56.4 pairs |
| Label balance | 121/78/22/5 disjoint hierarchy levels | `on` = 36% of assignments |
| Known data hazards | none outstanding | PU negatives, span convention, 29.6% right-censored spans |

The object stream is three times larger than the usable pair stream, and it arrives without a
single unresolved curation question.

**The 10.12 visible objects per frame is the important number**, and it is the one that changes the
scientific opportunity rather than merely the logistics. It means every frame is a natural
*sequence of perceptual acts* — roughly ten of them — over a shared scene. That is exactly the
substrate the dynamic context layer was theorized for, and it is what VRD could not provide.

### 1.3 Why this makes the dynamic-context test better, not worse

The paper's evidence that the dynamic context layer matters is Table 4: P-Direct scores 31.68 @1 on
binary labels, P-SA scores 46.84. The repository already flags the confound — P-Direct sees only
the predicate box while the BTN transports scene, subject and object evidence into the same
decision. The comparison varies information and mechanism together, so it cannot support the causal
claim it is used for.

**The object-scan version has no information confound at all.** Scan the objects of one frame in
sequence. Every model sees exactly one object feature per concept window. The only thing that
differs is whether context from previously scanned objects is available. Recognition of the *n*-th
object can then be measured as a function of *n*, giving a graded accumulation curve rather than a
single contrast, and a same-frame versus shuffled-frame control separates genuine scene-level
transport from a warm-up effect.

That is a strictly stronger test of the paper's own claim, it requires no predicates, and it is
only possible because PVSG frames contain ten tracked objects instead of one annotated pair.

---

## 2. What the pivot costs

Honesty about what is given up:

- **The direct numerical parallel to Table 4 is demoted**, not lost. Experiment 11 keeps it as one
  chapter on the already-materialized pair manifests.
- **Relation onset, cessation and anticipation leave the core.** These were attractive, but they
  are also where the censoring and PU problems live, and they are the most annotation-dependent
  part of the dataset. They become future work with a clear statement of why.
- **The thesis is about unary perception and memory rather than scene graphs.** Given that the
  paper's title is *Perception, Memory and Semantic Decoding*, this is closer to the source
  material than a relation-centric program would have been.

What is *not* given up: the curation work already done is not wasted. The hierarchy, identities,
protocols, exclusions and audit all serve the object stream directly. The pair materialization
serves Experiment 11.

---

## 3. The prioritized experiment list

Ordered by value per unit of remaining time. Every experiment in Tiers 1–3 runs on object
observations only.

### Tier 1 — the core (no relations, no new components, no new data)

#### E1. Within-category individuation
*Paper parallel: Table 5, "Entity" column (92.81% on VRD-EX).*

Can the index layer separate `person_3` from `person_5` in the same kitchen? Report identity
Hits@1/5/10 and MRR **stratified by the number of same-category competitors present in the video**,
which is the axis VRD could not vary — one entity per box, mostly distinct categories.

Conditions: full model; `P-noI-a` with identity indices removed, categories kept; `P-noI-b` with
identity kept, hierarchy removed. The pair isolates whether the benefit is identity-specific rather
than "more supervision helps."

Mandatory baseline: frozen-DINO nearest centroid over the same observations. If the learned index
columns do not beat it, the claim is about DINO, not about the Tensor Brain.

**Why it matters:** this is the paper's foundational hypothesis — encode individuals, not just
classes — and the original evidence for it was measured on affine-distorted copies of the training
images. This is the experiment that either establishes or refutes the premise of everything else.

#### E2. Identity persistence across time and occlusion
*Paper parallel: none. Untestable in VRD; the "lurking bear" of §9.6 made quantitative.*

Using the already-materialized `blocked` protocol (with `boundaries.jsonl` and
`blocked_last_observation`), report recognition stratified by:

- Δt since the last training exposure;
- **occlusion gap length** — object tracks have visible and invisible frames, so gaps are computed
  directly from the object observation table, with no pair-join proxy;
- appearance drift, as cosine distance between the query feature and the prefix-mean feature.

The third stratifier is what separates "the model remembers the individual" from "the query looks
like a stored template," which is precisely the distinction VRD-EX could not draw.

Fold the existing `fewshot` manifests in here as the low-exposure end of the same curve.

#### E3. The object-scan test of dynamic context — **centerpiece**
*Paper parallel: Table 4's claim, information-matched.*

Scan the objects of a frame in sequence: scene → object₁ → object₂ → … → objectₙ, with identity and
hierarchy readouts at each window and evolution between them.

Measure recognition accuracy of the *n*-th object as a function of *n*. Conditions:

1. no dynamic context and no feedback (independent per-object decoding);
2. dynamic context, no index feedback;
3. dynamic context and index feedback;
4. **shuffled control** — same sequence length, but objects drawn from different frames.

The prediction is that accuracy rises with *n* in conditions 2–3 and not in 1, and that the rise
disappears in condition 4. Also report whether *scan order* matters, which connects directly to the
measurement experiments in E8.

**Why it matters:** this repairs the paper's weakest headline claim with a design its authors could
not run. Every model sees identical evidence per window, so any gain is attributable to transport
rather than to information. Ten objects per frame gives a real curve rather than a two-point
contrast.

#### E4. Semantic enrichment under degraded evidence
*Paper parallel: Table 9 (`Dangerous`, P-SA 52.01 → P-enriched 98.24) — made non-degenerate.*

The paper's enrichment result used a label defined as animacy, which the model already predicted at
~98%. The honest version is available for free: **stratify object observations by `mask_area`,
which is already cached in the object table.**

For small, heavily occluded or motion-blurred masks the visual evidence is genuinely insufficient.
Ask whether identity recognition plus semantic decoding recovers the hierarchy labels that the
pixels cannot support. Compare, on the same low-area stratum: independent decoding, TB with
identity feedback, and the same identities at high mask area as the ceiling.

**Why it matters:** this is the paper's real claim — memory supplements perception when perception
is insufficient — tested with a genuine information deficit rather than a relabeled class, at zero
annotation cost. It is the strongest available parallel to the original Table 9.

#### E5. Is an index embedding a prototype or a classifier row?
*Paper parallel: §7.5's claim that the embedding is "a prototypical vector for that concept."*

Track, over training, the cosine between `a_k` and the centroid of the (RMS-normalized) features of
class or identity `k`. If the embedding converges toward the centroid it is a prototype, as the
paper claims; if it becomes discriminative while drifting away from the centroid it is a classifier
row, and much of the semantic-memory narrative needs rewording.

Also: does the identity embedding outperform its own best single observation and its running
feature mean at recognition? VRD could not ask this — entities had one to three views; PVSG
identities have hundreds.

**Why it matters:** it settles a central interpretive claim, it costs about twenty lines of
analysis, and it answers without a decoder the question the decoder was going to be built for.
Under a time constraint this is the highest insight-per-hour item in the program.

#### E6. Semantic decoding of generalized statements
*Paper parallel: Table 7 — qualitative illustration made quantitative.*

Activate a class index with no visual input and decode. For each of the 121 fine labels, is the
correct basic label the argmax over the basic group? The correct coarse label over the coarse
group? Which identities score highest, and do they belong to that class?

Report as accuracies over the whole hierarchy rather than as four hand-picked rows.

**Constraint:** this experiment runs a pure index activation `q = a_k` and is therefore directly
exposed to the scale issue in Section 6. Run the scale diagnostic first and report the
scale-matched condition alongside the raw one.

#### E7. The cognitive map, quantified
*Paper parallel: Figures 7 and 8 — t-SNE pictures made into metrics.*

Correlation between embedding distance and hierarchy tree distance; k-NN purity at each of the four
levels; whether same-fine-class identity embeddings cluster; whether the five domains separate.
Cheap analysis on artifacts E1–E3 already produce.

### Tier 2 — measurement theory (objects plus hierarchy)

#### E8. Order effects, chain-length scaling, and identity as level zero
*Parallel: QTB §7.3, §12.2; strongest bridge to the second paper.*

Readout order over the hierarchy levels, crossed with gate regimes `(1,1)` HB-POVM, `(0,1)` PVM and
`(1,0)` no feedback.

Three design choices that make this stronger than the existing QTB evidence:

- **Chain-length scaling.** Predict that PVM order-sensitivity grows with chain length while
  HB-POVM stays flat. A scaling curve resists confounds that a single contrast does not.
- **Tree consistency, not reversal rate.** Reversal cannot distinguish "the answer changed" from
  "the answer became incoherent"; the disjoint, catch-all-free hierarchy makes consistency well
  defined.
- **Identity as level zero.** Test whether measuring identity first collapses semantic uncertainty
  and suppresses order effects *even for PVM*. This ties the index layer to the measurement
  formalism rather than leaving them as separate stories, and it is a genuinely novel hypothesis.

**Reduce the design.** Four semantic levels plus identity is 120 orders; interpretation collapses
long before that. Run forward, reverse, identity-first, identity-last and about eight random
permutations, with exhaustive enumeration reserved for a three-level subset where six orders is
complete and legible. Include the scrambled-hierarchy falsification control.

#### E9. P-SA versus P-Samp, gated and calibrated
*Paper parallel: Table 5 rows; the discrete-bottleneck question in modern terms.*

Report ECE, NLL and Brier alongside accuracy, as a function of chain length. Run with the feedback
gate fixed at several values so that the approximation question is not confounded with feedback
magnitude — see Section 6. Folds into E3 and E8 rather than needing its own runs.

### Tier 3 — episodic memory (objects only; needs pre-allocated episodic indices)

#### E10. Episodic recall, recency, remote retrieval, interference
*Paper parallel: Table 12 (EM row), Figures 10–12 — illustrations made into measurements.*

Take a frame, or a short segment, as the episodic instance — the direct analogue of VRD's
image-as-instance, but now embedded in real time. Pre-allocate one episodic index per instance for
a subsampled set (of order 5,000, not 147,795). **This needs no allocator and no core change:** the
episodic group is simply another named group in the experiment-side vocabulary builder.

Four measurements the paper asserts and never makes:

- **Recall.** Activate episodic index `t*`, decode which objects were present and their labels.
- **Recency.** Is recall better for recent instances? Plot accuracy against age. The paper claims
  recency-triggered recall throughout §9.6 and never measures it.
- **Remote retrieval by similarity.** Given the current frame's state, which stored episodic index
  scores highest, and are the retrieved frames actually similar — same video, same scene type?
- **Interference.** Does recall degrade as the number of stored episodes grows? A capacity curve.

**Why it matters:** this is the whole of Section 9 turned from figures into numbers, and it is the
part of the paper with the largest gap between conceptual ambition and evidence.

### Tier 4 — relations (one chapter, existing manifests)

#### E11. Information-matched decomposition on positive-pair predicates
*Paper parallel: Table 4, decomposed.*

The ladder M0 (union only) → M1 (flat fusion of all four sources, no TB operations) → M2 (dynamic
context, no feedback) → M3 (P-SA) → M4 (P-Samp), so that `M1 − M0` isolates information,
`M2 − M1` recurrence and `M3 − M2` feedback. Macro-AP with per-predicate support and a prior-only
baseline.

Keep it to positive-pair predicate recognition, which is unaffected by the PU question. This is one
chapter, not the spine of the thesis, and it is the natural place to state plainly that PVSG masks
and tracks are oracle: the claim is about binding and memory *given* grouping, not about detection.

### Tier 5 — contributions beyond the original paper

Tiers 1–4 test claims the paper made. This tier makes claims it did not. Ordered by value per unit
of remaining time, and the first two are close to free.

#### X1. The scale conflict in shared bidirectional weights
*New. Detailed in [scale and normalization](tb_scale_and_normalization.md).*

`A` serves as a logit-scale readout and a state-scale write simultaneously, and the required norms
for the two roles diverge **linearly in the representation dimension** — 26× at `D = 128`, 159× at
the current `D = 768`, and 853× at the original paper's `r = 4096`. The original never hit this
because its trainable VGG encoder was a free scale parameter; freezing DINO with an identity input
map removed that degree of freedom.

Deliverables: the `D`-scaling law; the three concrete failures (input scale, feedback magnitude,
score offset); an empirical comparison of the five candidate resolutions; and the connection to
tied input/output embeddings in language models, where the same conflict is handled by the
`√d_model` multiplier.

**Why it counts:** it is a structural critique of the architecture with a quantitative scaling law,
and the work has to be done anyway to run Tiers 1–4. Highest value per effort in the program.

#### X2. Superposition capacity of the representation layer
*New. Directly quantifies the global-workspace claim of §6.6 and §8.5.*

Set `q = Σ_{k∈S} a_k` for a random index set `S` of size `m`, then ask whether the members of `S`
are the top-`m` scored indices. With the score offset absorbed (see X1), random columns at
`D = 768`, `N = 4000` give perfect recall to `m = 16`, 0.94 at `m = 32`, and smooth degradation
after.

**A 768-dimensional global workspace holds roughly 16–32 simultaneously active concepts.**

Run it as a real experiment with the controlled variables the preliminary numbers lack: random
versus *trained* columns (does learning improve packing?), `D`, vocabulary size `N`, and
hierarchically related versus unrelated index sets (are semantically close concepts harder or easier
to superpose?).

**Why it counts:** the paper asserts a single global workspace and argues in §8.5 that it forces
serialization of parallel computing. This measures the bottleneck. It costs a script, needs no
training, connects to global-workspace theory and to current superposition/feature-capacity work,
and no number like it exists for this model.

#### X3. The decoder `g⁺` and compositional embodiment
*Extends QTB §10.8, which specifies `ν̂_k ← g⁺(sig(a_k))` but was never built. Full staged design in
[the decoder plan](tb_feature_decoder_plan.md).*

Beyond grounding and reconstruction, one genuinely novel test: **compositional superposition
decoding.** The paper explicitly describes superposition — activating `Sparky` then `Friendly`
"leads to a superposition of both embeddings in `q̈_S`". So decode `σ(a_i + a_j)` and ask whether it
lands where the conjunction should: `fine:dog + coarse:living_being` on dogs, an identity plus its
class on that individual, two unrelated indices nowhere coherent.

That tests a stated mechanism nobody has verified, and it composes with X2: superposition capacity
measures how many indices survive *readout*, compositional decoding measures whether the superposed
state is *meaningful*.

Keep E5 (prototype versus classifier row) as the cheap prerequisite — if embeddings turn out to be
classifier rows, the decoder's grounding result is predictable and the chapter should be scoped
accordingly.

#### X4. An extensible index layer
*Extends §10.1–10.2 (complementary learning systems), which the repository cannot currently
express.*

Three components, each small: a reserve-pool allocator; a one-shot Hebbian write `a_new ← σ(q)`,
which is the direct reading of §10.2's "copying the episodic memory trace"; and a slow
consolidation pass. Together they partition `A` into a gradient-trained slow region and a
one-shot-written fast region — complementary learning systems as a property of one matrix.

Novel results available:

- **one-shot versus gradient-trained enrollment** at `k = 1, 5, 25` exposures, under `blocked`;
- **an index-capacity scaling law** — sweep `state_dim` against enrolled identity count and find
  where interference begins, which pairs directly with X2;
- **verified self-supervised recruitment** — recruit indices for unlabeled tracks by the paper's own
  novelty criterion, then score the recruited indices against ground-truth `object_id` tracks with
  cluster purity and completeness. VRD had no ground truth for invented indices; PVSG does. This is
  the strongest available version of "to perceive is to learn."

#### X5. Causal intervention on symbolic indices
*New framing; falsification test for the framework's central mechanism.*

Inject a wrong identity index and measure the downstream change in hierarchy and predicate
readouts. Compare against the same intervention on a flat-fusion model's hidden state, where the
intervention cannot even be *specified*.

**Why it counts:** the Tensor Brain's intermediate variables are named, discrete and human-readable
*by construction* — the property a large body of interpretability work exists to recover from
transformers post hoc. Demonstrating that they are causally load-bearing is a contribution to that
discussion. It is equally a falsification test: if injecting a wrong identity barely moves the
downstream distribution, index feedback is decorative and the result reframes the thesis.

Run it *after* X1, since a `10⁻³`-relative feedback perturbation would produce a null result for
scale reasons rather than architectural ones.

#### X6. Object-level anticipation
*New. §9.8 (forecasting, future episodic memory) is entirely prose in the original.*

Anticipation without relations, so none of the span or PU hazards apply:

- **presence prediction** — given frames up to `t`, which identities are visible at `t + Δ`?
  A set-prediction task straight from the object table.
- **permanence as prediction** — during an occlusion gap, does the model keep the identity index
  active and correctly predict reappearance? This is the predictive form of E2 and turns the
  lurking-bear illustration into a forecast.
- **feature-level rollout** — with X3's decoder, unroll evolution and score decoded predictions
  against actual future features. **Mandatory baseline: persistence.** At 5 FPS consecutive DINO
  features are near-identical, so any claim that does not beat "predict no change" is empty.

#### X7. Evolution backend as a controlled variable
*Extends the ledger's stated future direction; only meaningful on real temporal data.*

Original TB recurrence versus QTB feed-forward versus GRU versus Mamba/xLSTM, with measurement,
candidate sets and the shared `A` path held fixed. The object-scan schedule of E3 and the temporal
schedule of E2 are the two natural test beds. Lowest priority here — include only if time remains,
since it is an architecture comparison rather than a claim about the theory.

---

## 4. What to cut, and why

- **Relation anticipation and onset/cessation strata.** They depend on the censoring fix and on the
  span convention, and they are the most annotation-fragile part of the dataset. With relations
  demoted, the cost is no longer justified. *(If they return, note that 1,787 of 6,035 spans —
  29.6% — end at the final valid frame and are right-censored rather than observed cessations;
  building cessation strata without marking those would corrupt the result.)*
- **All-pair relationship prediction.** The positive-unlabeled question is unresolved and would
  need its own audit before the task is even well posed.
- **The synthetic environment, action indices, planning.** Correctly out of scope for this thesis:
  PVSG is passive, so decision support and planning are structurally untestable on it.
- **Relation-level anticipation.** Superseded by X6, which asks the same question on objects without
  the span and PU hazards.

The decoder and index recruitment are **no longer cut** — they return as X3 and X4. E5 remains the
cheap prerequisite for X3 rather than a replacement for it.
- **VLM zero-shot baseline.** Optional; include only if a reviewer-facing reference point is wanted
  and time remains.

---

## 5. Schedule

Assumes runners and metrics are the bottleneck, not data — which the readiness table in Section 7
supports.

| Phase | Work | Output |
|---|---|---|
| Days 1–2 | `a0` offset fix (**F0**); scale diagnostic; object-scan and gap-detection record builders | unblocked; **X1** and **X2** largely fall out of this |
| Week 1 | **E1** individuation, **E5** prototype analysis | the premise of the thesis, plus the cheapest result |
| Week 2 | **E3** object-scan dynamic context | the centerpiece |
| Week 3 | **E2** persistence and occlusion, absorbing `fewshot`; **X5** intervention | the temporal chapter plus the falsification test |
| Week 4 | **E4** enrichment; **E6**, **E7** decoding and map; **X2** written up properly | the semantic-memory chapter |
| Weeks 5–6 | **E8** order effects (reduced design), **E9** folded in; **X1** written up | the QTB bridge and the scale chapter |
| Week 7 | **E10** episodic memory | Section 9 made quantitative |
| Weeks 8–9 | **X4** extensible index layer | the first extension chapter |
| Week 10 | **X3** decoder and compositional embodiment | the second extension chapter |
| If time | **E11** relations, **X6** anticipation, **X7** evolution backends | additive |

**Risk profile.** E1, E3 and E5 alone constitute a defensible thesis. X1 and X2 are close to free
and are the two contributions most likely to survive contact with limited time — both largely fall
out of work that has to happen in the first two days regardless. X4 and X3 are the substantial
extension chapters and are correctly placed late, where they can be cut without damaging the core.

If the schedule slips, cut from the bottom: X7, then X6, then E11, then X3. Do not cut X1 or X2 —
they cost days and carry the novelty.

---

## 6. Scale verification (retained, and now more important)

> Full treatment, including the `D`-scaling law, the newly identified `a0` offset problem, the
> asymmetric-transformation variants, and the ranked menu of fixes, is in
> [scale and normalization](tb_scale_and_normalization.md). This section keeps the summary that
> gates the experiments below.

### 6.1 The working-tree fix is correct

Replacing per-vector L2 normalization with `sqrt(D) * L2-normalize` gives every input component RMS
one. Measured at `D = 768`:

| | `‖drive‖` | `σ(q)` sd | `‖a_k‖` | `‖a_k‖ / ‖drive‖` |
|---|---:|---:|---:|---:|
| Previous (plain L2) | 1.00 | **0.0090** | 1.03 | **1.03** |
| Current (RMS) | 27.71 | **0.2080** | 1.03 | **0.037** |

The CBS dynamic range improves 23-fold. Previously `σ(q)` was pinned at `0.5 ± 0.009` and the
representation layer was effectively constant. This was a real bug and the fix is right.

### 6.2 It addresses the input side only, and widens the other asymmetry

`A` still initializes at `std = state_dim^{-1/2}`, so `‖a_k‖ ≈ 1` and `σ(a_k)` has sd `0.0093`.
Before the change, drive and index column were matched at norm ≈ 1 — both too small, but
symmetric. After it, the drive is correct and the index column is 27× behind. **Index feedback is
now a 3.7% perturbation of the state it is supposed to inform.**

Expected feedback is worse: `Σ_k π_k a_k` averages near-orthogonal unit columns and shrinks toward
`1/√K_eff`.

| Identity candidates `K` (near-uniform) | `‖feedback‖` | relative to `‖q‖ ≈ 27.7` |
|---:|---:|---:|
| 100 | 0.096 | `3.5 × 10⁻³` |
| 1,000 | 0.030 | `1.1 × 10⁻³` |
| 4,000 | 0.015 | `5.5 × 10⁻⁴` |
| **P-Samp (single column)** | **1.03** | **`3.7 × 10⁻²`** |

So P-SA and P-Samp differ in feedback *magnitude* by up to 70×, and a difference between them
cannot presently be attributed to expected-versus-sampled rather than to scale.

### 6.3 Why the pivot raises the stakes

Under the object-first program this stops being one experiment's problem. **E6 activates a bare
index (`q = a_k`) and decodes it — that is the entire semantic-memory chapter, and it lives
precisely in the regime where `σ(a_k)` is nearly flat.** With `‖a_k‖ ≈ 1` the sigmoid is in its
linear range, so decoding reduces to a scaled inner product `a_j·a_k / 4` plus a per-index offset
that `a0` must absorb. That may work, but it is a different operating regime from the one the model
trains in, and it must be checked rather than assumed.

E4 and E10 have the same exposure: both depend on the state being meaningfully moved by an index
rather than by pixels.

### 6.4 Recommendation

**Measure first — one hour, no model change.** Log `‖a_k‖` by index group over training,
`‖feedback‖ / ‖q‖` per window for both P-SA and P-Samp, and the component histograms of `σ(q)`
versus `σ(a_k)` at convergence. If `‖a_k‖` grows to within a factor of about three of `‖q‖`, nothing
needs changing.

**If it does not, prefer a gate over re-initializing `A`.** Setting `A`'s init to `std = 1` would
give `‖a_k‖ ≈ 27.7` against `‖σ(q)‖ ≈ 15`, pushing logits toward ±400. The shared matrix serves as a
logit-scale readout and a state-scale write simultaneously, and the two roles want incompatible
norms.

That tension is a structural property of the paper's shared bidirectional `A`, not a bug to patch
away, and it deserves a paragraph in the thesis. The fidelity-preserving resolution is the gate the
formalism already provides:

- add `retain_gate` / `feedback_gate` to `attend`, mirroring `measure` — the ledger already
  establishes that gates are the QTB generalization and that the core does not constrain their
  values, so this is an API completion rather than a model change;
- run `β` fixed at several values as a controlled condition;
- then run `β` **learned** through a sigmoid parameterization, at which point *how strongly the
  model wants index feedback* becomes a reportable measurement.

Note the API asymmetry that makes this necessary: `measure` exposes both gates, `attend` exposes
neither, and `IntegralTB` trains through `attend`. The weaker path is currently the one that cannot
be corrected.

---

## 7. Readiness

| Experiment | Data ready? | Still needed |
|---|---|---|
| E1 individuation | **Yes** | same-category competitor counts from manifests + hierarchy |
| E2 persistence | **Yes** | `blocked` boundaries already materialized; gap detection from object table |
| E3 object scan | **Yes** | frame-grouped scan records; a scan runner |
| E4 enrichment | **Yes** | `mask_area` already cached; stratification only |
| E5 prototype analysis | **Yes** | analysis script only |
| E6 generalized statements | **Yes** | index-activation evaluation; scale-matched condition |
| E7 cognitive map | **Yes** | analysis on E1–E3 artifacts |
| E8 order effects | **Yes** | sequential readout schedules; `vocabulary.py` already wires the levels |
| E9 P-SA/P-Samp | **Yes** | calibration metrics; gate on `attend` |
| E10 episodic memory | **Yes** | episodic group in the experiment-side vocabulary builder |
| E11 relations | **Yes** | flat-fusion model class |
| X1 scale study | **Yes** | logging plus the fix comparison; no new data |
| X2 superposition capacity | **Yes** | a script; no training required |
| X3 decoder | **Yes** | decoder head outside `src/tb` |
| X4 extensible index layer | **Yes** | reserve-pool allocator and one-shot write — the only `src/tb` addition |
| X5 intervention | **Yes** | small harness |
| X6 anticipation | **Yes** | Δ-shifted presence targets from the object table |
| X7 evolution backends | **Yes** | backend implementations behind the existing contract |

No experiment in this program requires new feature extraction. Only X4 requires an addition to
`src/tb` (the allocator); everything else needs at most the optional `attend` gate and the `a0`
initialization fix. The curation is finished; what remains is runners,
metrics and analysis.

---

## 8. The thesis in one paragraph

The Tensor Brain claims that perception, episodic memory and semantic memory are operating modes of
one oscillation between a symbolic index layer and a distributed representation layer, and that the
key commitment is representing *individuals* rather than only classes. That claim was evaluated on
static images in which individuals recurred only as affine distortions of themselves, attributes
were partly synthetic, hierarchy levels were padded with a catch-all, the decisive baseline saw
less information than the model, and every temporal claim was illustrated rather than measured.
PVSG supplies what was missing: ten tracked individuals per frame, recurring across real viewpoint,
pose and occlusion change, under a strict four-level hierarchy with no catch-alls. The program
above tests the claim where it is meant to apply — individuation among same-category competitors,
transport across a genuine sequence of perceptual acts, memory that survives occlusion, and
semantic decoding that supplies what the pixels cannot — and it does so without needing relations,
which is why it can be finished in the time available.

# Experiment plan for the remaining three working weeks

Budget: ~4 weeks total, ~1 week reserved for writing, so **~15–21 working days of experiments**.
Everything below is sized against that, and against the fact that the experimental apparatus is
currently at zero (issue **S12**).

The organizing decision: **build the thesis on the object-observation stream, and keep relations as
one late chapter on the manifests you already have.** The prior planning document
(`pvsg_core_capability_plan.md`) argued this on scientific grounds and it is right. Two further
arguments it does not make, which settle it:

- **It fits in memory.** 1 495 227 object observations × 768 × fp16 = **2.3 GB**, plus 0.23 GB of
  scene features. Load once, shuffle globally, epochs in tens of seconds. The pair stream is
  **12.8 GB** and is not. This removes the streaming loader, the block-sampling correlation problem
  (**S13**), and most of the engineering risk from the critical path.
- **It discards every unresolved data question at once** — positive-unlabeled negatives, the
  relation-span convention, 29.6 % right-censored spans, the `on`-dominated long tail, and the
  26.7 % incomplete joins. None of them touch objects.

---

## 1. Days 1–3: unblock. Nothing here is a result; everything here is a prerequisite

### 1.1 Model and core fixes (half a day)

| Fix | Issue | Change |
|---|---|---|
| Learned `nn.Linear(768, state_dim)` per input source as `g(·)` | **S1** | experiment-side, outside `src/tb` |
| `retain_gate` / `feedback_gate` on `attend` | **S11**, **S1** | mirrors `measure`; API completion |
| Post-feedback semantic readouts at subject/object windows | **S3** | the readout the paper's claim is made of |
| Model ladder replaces the two-model comparison | **S2** | see E-A |
| Standing diagnostics: `‖a_k‖` by group, `‖feedback‖/‖q‖`, `γ` histogram, `a0` magnitude | **S1**, **M17** | logged every epoch of every run, forever |

### 1.2 Manifest regeneration — the only data work I recommend (half a day, minutes of compute)

Three changes, all manifest-level, all reusing the cached features. `materialize.py` refuses to
overwrite an existing snapshot, so this produces `section6-v2` beside `v1`; that is correct
practice, not a problem.

1. **Dev split** (**S6**) — deterministic hash partition of official-train video IDs, ~15 %.
2. **Spread few-shot support frames** (**S7**) — five frames spaced across the track rather than
   five consecutive frames, plus a recorded support-set diversity statistic.
3. **Frame-grouped object-scan records** — for each frame, the ordered list of visible object rows.
   This is what E-A consumes and it is a `groupby`, not an extraction.

### 1.3 Apparatus (two days)

One runner, one metrics module, one diagnostics module, one artifact writer
(`config.json` / `split.json` / `checkpoint.pt` / `results.json` / `predictions.pt`, as
`docs/pvsg_perception.md:376-385` already specifies). Keep schedules explicit per `AGENTS.md`; do
not build a generic composer.

### 1.4 Two sanity gates before any real run

- **Overfit gate.** 200 object rows to near-zero loss. Step 5 of your own implementation order.
- **Scale gate.** With the trained model, `‖feedback‖ / ‖q‖` must be in a reportable range (order
  10⁻¹, not 10⁻³) for both P-SA and P-Samp, and the two must be **matched** before they are
  compared. If they are not, the S1 fix did not take and nothing downstream means anything.

### 1.5 Reserve columns instead of an allocator

You do **not** need a growable vocabulary component. Size `A` with a reserve suffix — build the
vocabulary with `N` real labels plus `R` unused `reserve:####` columns, keep them out of every
candidate group until enrollment, then write into them under `no_grad` and add them to a candidate
list. Unused columns receive no gradient (indexing `A[:, indices]` only touches selected columns),
and with plain Adam a permanently zero gradient produces a zero step, so there is no mid-run
parameter surgery and no checkpoint incompatibility. This turns the prior plan's "keystone A-tier
component" into a one-line change to the vocabulary builder, and it is what makes E-E affordable.

**One caveat that will bite if missed:** decoupled weight decay (`AdamW`) shrinks parameters
regardless of gradient, so reserve columns would decay toward zero before you ever write to them.
Either use `Adam`, or exclude `A` from weight decay, or re-initialize a reserve column at enrollment
time. Assert `‖a_reserve‖` is unchanged at enrollment; it is a one-line check that turns a silent
failure into a loud one.

---

## 2. The experiments

Five, ordered by value per day. **E-A + E-C + E-D is a defensible thesis on its own.** Everything
after is additive, which is the right risk profile.

Compute is not the constraint: a full 7 143-way readout over 1.5 M rows is ~50 TFLOP per epoch
including backward, i.e. tens of seconds on a 2080 Ti. You can afford the full ladder at three
seeds.

---

### E-A. The mechanism ladder on the object scan — **the centerpiece** (5 days)

**Claim under test.** The dynamic context transports information between perceptual acts, and
top-down index feedback adds something beyond that transport.
**Paper parallel.** Table 4 (`P-Direct` 31.68 → `P-SA` 46.84 @1), repaired.

Every frame contains **10.12 tracked objects on average**. Scan them in sequence:
`scene → object₁ → object₂ → … → objectₙ`, with identity and hierarchy readouts at each window and
an evolution step between windows.

**Rungs — each adds exactly one mechanism, and every rung sees identical evidence per window:**

| | Model | Adds |
|---|---|---|
| M0 | independent per-object decoding | — |
| M1 | + scene evidence, flat fusion, no TB operations | information |
| M2 | + evolution between windows, no index feedback | transport |
| M3 | + P-SA feedback, `β` fixed | soft symbolic feedback |
| M4 | M3's checkpoint evaluated with P-Samp (argmax) | hard symbolic commitment |
| M5 | M3 with `β` learned | how much feedback the model wants |

`M1 − M0` isolates information, `M2 − M1` isolates recurrence, `M3 − M2` isolates feedback,
`M4 − M3` isolates the discrete bottleneck, and `M5` reports the model's own preference. **This is
the decomposition the paper's Table 4 cannot support**, because there `P-Direct` varied information
and mechanism together — the confound your own `docs/pvsg_perception.md:266-274` already names.

**Measurements.**
- Accuracy of the *n*-th object as a function of *n* — a graded accumulation curve, not a two-point
  contrast. This is only possible because PVSG gives ten individuals per frame where VRD gave one
  annotated pair.
- **Within-video** identity accuracy as the primary identity number (**S4**), global as secondary.
- Fine / basic / coarse / domain accuracy, read *after* feedback (**S3**).
- **Shuffled-frame control**: same sequence length, objects drawn from different frames. If the
  accumulation curve survives shuffling, it is a warm-up effect and not scene-level transport.
- Scan-order sensitivity (feeds E-F).

**Falsification.** If M2 = M1 and M3 = M2 with matched feedback magnitude, the dynamic context and
the index feedback are both decorative on real video. That is a substantial negative result about a
published framework, and it is worth reporting.

**Mandatory floor.** Frozen-DINO nearest-centroid over the same observations. If the learned index
columns do not beat it, the result is about DINOv3, not about the Tensor Brain.

---

### E-B. Memory supplements perception under a *measured* information deficit (3 days)

**Claim under test.** Memory supplies what perception cannot — the paper's strongest and most
poorly evidenced claim.
**Paper parallel.** Table 5's `Y/O` column (76.54 → 94.58, P-Direct → P-Samp on VRD-EX) and
Table 9's `Dangerous` enrichment (52.01 → 98.24). Both are weak evidence: `Y/O` is synthetic and
`Dangerous` was effectively animacy, which the model already predicted at ~98 %.

PVSG gives two **genuine, already-cached, zero-annotation-cost** deficit axes:

1. **Mask area.** `object_mask_areas` is in every object row. Below roughly one patch cell, the
   mask-pooled feature is dominated by background — the visual evidence is *genuinely* insufficient,
   not relabeled.
2. **Annotated-but-unobservable participants** (**S8**). 135 440 records have an annotated relation
   while the subject's mask is absent. The relation persists through occlusion; the pixels do not.

**Design.** `blocked` protocol (train on the first 45 %, embargo 10 %, evaluate on the last 45 %).
Stratify evaluation by mask-area decile × Δt since last exposure. Conditions: M2 (transport only),
M3 (identity feedback), and — the important addition — **oracle identity**, where the correct
identity is teacher-forced into the feedback slot.

The oracle condition is cheap and it is what makes the experiment interpretable: it upper-bounds how
much identity knowledge *could* contribute, separating "memory does not help" from "identity
recognition failed". Without it a flat result is uninterpretable.

**Prediction.** Memory's contribution grows monotonically as visual evidence degrades. A crossing
point where feedback overtakes pixels is the headline figure of this chapter.

**Also report** appearance drift — cosine between the query feature and the prefix-mean feature for
that identity. This separates "the model remembers the individual" from "the query looks like a
stored template", the distinction VRD-EX structurally could not draw.

---

### E-C. Are index embeddings prototypes or classifier rows — and embodiment for free (2 days)

**Claim under test.** §7.5's claim that an index embedding is "a prototypical vector for that
concept", and QTB §10.8's embodiment claim that `ν̂_k ← g⁺(sig(a_k))` reinstates something
perceptual.

**The free decoder.** Once `g` is a learned linear map (the S1 fix), its pseudo-inverse **is**
`g⁺` — QTB literally specifies `g⁺` as "the (approximate) inverse mapping". You get the decoder at
zero training cost as a side effect of a fix you had to make anyway. A two-layer learned decoder is
a 30-minute upgrade if the linear one is too weak.

**Measurements.**
- Cosine between `a_k` and the centroid of RMS-normalized features of class/identity `k`, tracked
  *over training*. Converging toward the centroid ⇒ prototype, as claimed. Becoming discriminative
  while drifting away ⇒ classifier row, and a good deal of the semantic-memory narrative needs
  rewording.
- Does `a_k` beat its own running feature mean, and its single best observation, at recognition?
  VRD entities had one to three views; PVSG identities have hundreds, so this is answerable here for
  the first time.
- **Embodiment**: activate index `k` with no visual input, decode `ν̂_k = g⁺(sig(a_k))`, and measure
  **nearest-neighbour retrieval accuracy** against held-out observations — a far more meaningful
  metric than MSE. Run it in a **scale-matched** condition as well as raw, because a bare `q = a_k`
  with `‖a_k‖ ≈ 1` sits in the linear range of the sigmoid and is a different operating regime from
  the one the model trains in (**M17**).
- Semantic decoding of generalized statements (paper Table 7, made quantitative): activate a fine
  label, is the correct basic label the argmax over the basic group? The correct coarse label over
  the coarse group? Report over all 121 fine labels, not four hand-picked rows.

**Why this is the best hours-to-insight ratio in the program.** It settles a central interpretive
claim, it converts embodiment from unfalsifiable to a number, and most of it is analysis on
artifacts E-A already produces.

---

### E-D. Causal intervention on the symbolic bottleneck — **the contribution to DL generally** (2 days)

**Claim under test.** That the measured index is causally load-bearing rather than decorative.

The Tensor Brain's structural peculiarity is that its intermediate variables are **named, discrete,
low-dimensional and human-readable by construction**. A very large amount of 2026 interpretability
work — activation patching, causal tracing, sparse autoencoders, circuit analysis — exists to
*recover* variables with those properties from transformers. This model has them for free and the
corresponding experiments have never been run.

**Interventions** (all on trained checkpoints; no new training):
- substitute the measured identity with a wrong one, stratified by semantic distance in the reviewed
  hierarchy (wrong individual, same fine class → wrong domain);
- ablate the feedback entirely at one window;
- shuffle the window order;
- clamp the feedback vector to a fixed value.

Measure downstream degradation, and run the *same* intervention on M1's hidden vector, where it
cannot even be specified. That contrast is the point.

**The sharpest single result available in this project.** M3 vs M4 in E-A is the same checkpoint,
the same parameters, one line different — soft mixture versus hard symbolic commitment, with
feedback magnitude held matched. **That is a clean measurement of the cost of a discrete
bottleneck**, and it is the question behind VQ, discrete latent variables, and
chain-of-thought-as-tokens. In a transformer you cannot hold the mechanism fixed and swap hard for
soft. Here you can. Report it as such.

**A second bridge worth one paragraph.** The shared bidirectional `A` — one matrix serving both
bottom-up scoring and top-down feedback — is structurally identical to **tied input/output
embeddings** in language models. "Is a tied embedding a prototype or a classifier row, and does
feeding it back into the residual stream help?" is E-C and E-A asked about LLMs. This is the
cleanest available route from a cognitive-architecture thesis to a claim a deep-learning audience
cares about.

---

### E-E. One-shot write and the extensible index layer (3 days)

**Claim under test.** §10.2's complementary-learning-systems story: a fast system that establishes a
new index by "copying the episodic memory trace", followed by slow consolidation. The repository
currently has one learning timescale, so this whole class of claims is unfalsifiable rather than
untested.

**The write rule is named by the framework's own mathematics.** Scoring is `a_k^T sig(q) + a0_k`, so
the write that makes a new index maximally responsive to the state that created it is
`a_new ← sig(q)`, up to normalization. Normalize to the trained `‖a_k‖` statistics — an unnormalized
write will dominate the softmax.

**Design.** `fewshot` protocol with the *repaired* support sets (**S7**). Enroll a validation-video
identity into a reserve column from 1 / 5 / 25 exposures, in one shot, no backprop.

**Three questions never asked of this model:**
1. How does one-shot Hits@1 compare to gradient-trained Hits@1 at each exposure count?
2. Does slow gradient consolidation move the fast-written column *toward* where gradient descent
   would have put it from scratch, or somewhere else?
3. Is the one-shot write a better initialization for consolidation than random?

**Risk to name in the write-up.** A one-shot write is trivially good at recognizing the exact state
that wrote it. The 25-frame embargo makes this a temporally separated test rather than template
matching — that is why **S7** must be fixed first, or the experiment measures the wrong thing.

**Why this is the strongest *extension* claim.** "One matrix, two timescales, symbolic addressing"
is complementary learning systems expressed as a property of a single parameter, and it lands
directly in the fast-weights / modern-Hopfield / continual-learning conversation.

---

### E-F. Stretch, only if week 3 has slack — pick one

- **Order effects and gate regimes** (QTB bridge). Readout order over hierarchy levels crossed with
  `(1,1)` HB-POVM, `(0,1)` PVM, `(1,0)` no feedback. Reduce hard: forward, reverse, identity-first,
  identity-last and ~8 random permutations, with exhaustive enumeration only on a three-level
  subset. Score **tree consistency**, not reversal rate — reversal cannot distinguish "the answer
  changed" from "the answer became incoherent", and the disjoint catch-all-free hierarchy makes
  consistency well defined. Include a scrambled-hierarchy falsification control. The novel
  hypothesis worth stating: measuring **identity first** may collapse semantic uncertainty and
  suppress order effects even under PVM.
- **The relation chapter.** Run the same E-A ladder on the complete-evidence positive-pair manifests
  you already have. One chapter, the direct numerical parallel to Table 4, and the natural place to
  state plainly that PVSG masks and tracks are oracle: the claim is about binding and memory *given*
  grouping, not about detection.

---

## 3. Schedule

| Days | Work | Deliverable |
|---|---|---|
| 1–3 | Fixes, manifest regeneration, runner/metrics/diagnostics, overfit + scale gates | apparatus exists; S1 verified fixed |
| 4–8 | **E-A** ladder, 3 seeds | the centerpiece |
| 9–11 | **E-B** degradation and memory | the memory chapter |
| 12–13 | **E-C** prototype, decoder, embodiment | the interpretive chapter |
| 14–15 | **E-D** interventions | the DL-general chapter |
| 16–18 | **E-E** one-shot enrollment | the extension chapter |
| 19–21 | Buffer, or **E-F** | — |
| Week 4 | Writing | thesis |

**Cut rule.** If E-A is not producing curves by end of day 10, drop E-E and E-F and finish with
E-A + E-B + E-C + E-D. If you are behind by day 14, drop E-B's occlusion axis (keep mask area) and
E-C's learned decoder (keep the pseudo-inverse). Do not cut E-D — it is two days and it is the part
a deep-learning reader will quote.

---

## 4. What to cut, and why

- **Relations as the spine.** Nine of the paper's ten core perception/memory claims need no
  relations, and the one that does has a *better*, information-matched test on objects. Demote to
  one chapter on existing manifests.
- **Relation anticipation, onset/cessation strata, all-pair prediction.** Depend on the censoring
  fix and the unresolved positive-unlabeled question. Not worth the days.
- **The semantic property/relation inventory** (49 unary values, 9 relations). Already built,
  probably unused by this thesis. Ship it as an appendix with a clear statement of what it is for.
  It is not wasted, it is early.
- **Episodic memory as its own chapter, consolidation by activation replay, forgetting/eviction,
  time-aware indices.** All genuinely interesting; all need components you will not have time to
  validate. E-E gives you the write path; the rest is future work.
- **Distributed indices.** Localized indices *are* the theory — they are what makes the one-shot
  write well defined and intervention meaningful. There is no measured capacity curve for a
  distributed variant to beat, so the trade cannot be assessed. Defer with that reason, not with
  "hard".
- **A synthetic environment, action indices, reward, planning.** Correct out of scope. Worth one
  honest paragraph in future work: PVSG is passive, third-person, fully annotated video, so agency
  claims are *structurally* untestable on it regardless of experiment quality.
- **Any further extraction or annotation.** None of the above needs it.

---

## 5. The thesis in one paragraph

The Tensor Brain claims that perception, episodic memory and semantic memory are operating modes of
one oscillation between a symbolic index layer and a distributed representation layer, and that its
key commitment is representing *individuals* rather than only classes. That claim was evaluated on
static images in which individuals recurred only as affine distortions of themselves, attributes
were partly synthetic, hierarchy levels were padded with a catch-all, the decisive baseline saw less
information than the model, and every temporal claim was illustrated rather than measured. PVSG
supplies what was missing: ten tracked individuals per frame, recurring across real viewpoint, pose
and occlusion change, under a reviewed four-level hierarchy with no catch-alls. This thesis tests
the claim where it is meant to apply — transport across a genuine sequence of perceptual acts,
individuation among same-category competitors, memory that survives occlusion and measured evidence
degradation, and index embeddings decoded back toward perception — and it reports, as a result of
independent interest, what it costs a model to commit to a discrete symbol when the alternative is a
soft mixture over the same learned embeddings.

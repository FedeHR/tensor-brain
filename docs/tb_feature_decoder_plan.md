# Concrete plan: the feature decoder `g⁺` and the embodiment experiments

## 1. What the papers actually specify

This component is not an invention. QTB Section 10.8 defines it:

> We are also considering, that an index can map back to the processing pipeline. The
> (approximate) inverse mapping from index `k` to CBS `q = a_k` and back to `ν_k` is an embodiment
> process. It could be realized by a top-down network implementing a function where
> `ν̂_k ← g⁺(sig(a_k))` is the embodiment of the index `k`. The map `ν_k → a_k → ν̂_k` forms an
> autoencoder structure that visually explains to the brain what an index is all about and that
> might be useful for self-supervised learning.

The extended original paper independently describes the BTN observation model as having "the
characteristics of an autoencoder" (Sections 4.1 and 4.6), and QTB Section 15.9 lists "Autoencoder
Learning" as a named thesis project. Three things follow directly from the quoted definition and
should be treated as fixed constraints, not choices:

1. **The domain of `g⁺` is the CBS `γ`, not the pre-CBS `q`.** The paper writes `g⁺(sig(a_k))`.
   This is also internally consistent: bottom-up scoring reads `a_k^T σ(q)`, so the top-down path
   should read from the same broadcast state. Keep the symmetry.
2. **Its codomain is input space `ν`,** i.e. the same space the perceptual drive comes from.
3. **`a_k` is fed in as if it were a state.** Activating index `k` alone means `q = a_k`. This is
   the embodiment operation, and — see Section 2 — it is where the practical difficulty lives.

Like the encoder, `g⁺` belongs outside `src/tb`. It is part of `g`, not part of the Tensor Brain.

What the definition does *not* fix is how deep `g` and `g⁺` should reach — the paper's own `f(·)` is
the entire fine-tuned DCNN, not a head on frozen features, and the embodiment claim is explicitly
about reaching *earlier* perceptual layers. §3.4 treats that question and turns it into
**Capability 5**, which measures how deep embodiment actually reaches instead of assuming a
boundary.

---

## 2. Prerequisite: run the scale diagnostic before building anything

There is a quantitative problem sitting between the definition and a working experiment, and it
would silently produce a null result that looks like "embodiment doesn't work."

**The current numbers.** `experiments/pvsg/models.py` uses `state_dim = feature_dim = 768` with an
identity input mapping, and `normalize_dino` scales each vector to component RMS one. So:

| Quantity | Value |
|---|---|
| Perceptual drive norm `‖g(ν)‖` | `√768 ≈ 27.7`, component RMS 1 |
| `σ(q)` after scene integration | mean `0.498`, **sd `0.208`** |
| Index column norm `‖a_k‖` at init | `≈ 1.00` by construction (`std = state_dim^{-1/2}`) |
| Component sd of `a_k` | `1/√768 ≈ 0.0361` |
| `σ(a_k)` | mean `0.500`, **sd `0.0093`** — near-constant in every component |

(Measured, not just derived: sampling 4,000 columns at the repository's initialization reproduces
these to three digits.)

The spread of `σ(a_k)` around `0.5` is **22× smaller** than the spread of a perceptual `σ(q)`. A
decoder trained on perceptual states and then asked to decode `σ(a_k)` is handed a signal far
outside the range it was fit on, compressed into a nearly flat vector. It will decode approximately
the dataset mean, and the natural but wrong conclusion would be "index embeddings are not
grounded."

**A second consequence, which matters independently of the decoder.** Index feedback adds a
norm-≈1 vector to a state of norm ≈27.7 — a 3.7% perturbation for P-Samp. It is far smaller for
P-SA, because expected feedback `Σ_k π_k a_k` is a convex combination of near-orthogonal unit
columns and its norm shrinks toward `1/√K_eff`:

| Identity candidates `K` (near-uniform) | `‖Σ π_k a_k‖` | relative to `‖q‖ ≈ 27.7` |
|---:|---:|---:|
| 10 | 0.311 | `1.1 × 10⁻²` |
| 100 | 0.096 | `3.5 × 10⁻³` |
| 1,000 | 0.030 | `1.1 × 10⁻³` |
| 4,000 | 0.015 | `5.5 × 10⁻⁴` |
| P-Samp (single column) | 1.03 | `3.7 × 10⁻²` |

Early in training, with a near-uniform distribution over thousands of identity candidates, P-SA
feedback is a relative perturbation of order `10⁻⁴`. P-Samp, which injects one full column, does
not have this shrinkage — so the two conditions differ in feedback *magnitude* by up to two orders
of magnitude, not only in whether feedback is expected or sampled.

This is a prediction from initialization, not a measured fact: `‖a_k‖` may grow substantially
during training, and if it does the problem resolves itself. But it is cheap to check and it
matters for the *existing* P-SA/P-Direct comparison, not only for the decoder.

**Diagnostic to run first (half a day, no new code beyond logging):**

- log the distribution of `‖a_k‖` over training, split by index group (predicate, category,
  identity);
- log `‖q‖` at each concept window, and `‖feedback‖ / ‖q‖` for both P-SA and P-Samp;
- log the component histogram of `σ(q)` versus `σ(a_k)` at convergence.

> **Update (working tree).** The input side of this has since been fixed: `normalize_dino` now
> applies `sqrt(D) * L2-normalize`, raising `σ(q)` sd from 0.0090 to 0.208. That fix is correct and
> necessary, but it does not touch `A`'s initialization, so it *widens* the drive-to-feedback
> asymmetry from 1.03 to 0.037. See
> [the core-capability plan](pvsg_core_capability_plan.md) Section 1 for the verification and the
> recommended gate-based response.

**Decision rule.** If at convergence `‖a_k‖` is within a factor of ~3 of the typical `‖q‖`, proceed
as specified. If the gap remains an order of magnitude, then the embodiment evaluation must include
an explicit, named **scale-matched condition**: decode `σ(c · a_k)` with `c` calibrated so that
`‖c · a_k‖` matches the median `‖q‖` at the corresponding window. Report matched and unmatched side
by side and never silently apply the correction — it is an experimental condition, not a fix.

---

## 3. Design

### 3.1 Target

Reconstruct the **RMS-normalized** feature, not the raw DINO vector. That is what `g` actually
injected; the raw norm was deliberately discarded by `normalize_dino` and reconstructing it would
measure something the model was never given.

A useful consequence: the target lies on a sphere of radius `√D`, so cosine similarity and MSE are
monotonically related. **Report cosine.** It is interpretable, comparable across sources, and it
avoids MSE numbers whose scale means nothing to a reader.

### 3.2 Architecture ladder

Start at the bottom and stop as soon as the scientific question is answered.

| Level | Form | Purpose |
|---|---|---|
| L0 | `ν̂ = logit(γ)` — **zero parameters** | mandatory baseline (see below) |
| L1 | `ν̂ = W γ + b`, `W ∈ ℝ^{768×768}` | the interpretable workhorse |
| L2 | 2-layer MLP, shared trunk, four role heads | if L1 underfits |
| L3 | role-conditioned, larger | only with evidence L2 is the bottleneck |

**L0 deserves emphasis, because it is a genuinely sharp control.** In the current identity-mapping
configuration `q` is literally the sum of the drives integrated so far, so `logit(σ(q)) = q`
recovers that sum exactly. For the *autoencoding* capability this makes L0 very strong, and any
learned decoder must beat it to have shown anything. But for *index grounding* L0 returns
`logit(σ(a_k)) = a_k`, the raw embedding — dimensionally in feature space but with no reason to
land near real features. So the parameter-free baseline cleanly separates the two capabilities:
it should win on reconstruction and fail on grounding. If a learned decoder cannot beat L0 on
reconstruction, the decoder is not learning; if L0 unexpectedly succeeds at grounding, then `A`
columns have simply become feature prototypes, which is itself the most interesting possible result
and should be reported as such.

### 3.3 Role conditioning

The four evidence sources — scene CLS, mask-pooled object, union region — have different
statistics, and the same `γ` should decode differently depending on what is being asked for.
Condition the decoder on the role: a shared trunk with four output heads (`scene`, `subject`,
`object`, `union`). This mirrors the role structure already present in the schedule and keeps the
parameter count low.

### 3.4 How deep should `g` and `g⁺` reach?

The plan so far treats `g` as "frozen DINO, then identity" and `g⁺` as a head onto the final pooled
vector. That is a scoping choice, not a principled boundary, and it is worth making the boundary
explicit because the papers point the other way.

**The papers place the whole encoder inside `g`.** In the original, `f(·)` is the complete
VGG-19 / Faster R-CNN backbone, fine-tuned end to end — not a projection head on frozen features.
And the embodiment claim in §7.5 and §10.8 is specifically that top-down index activation reaches
*earlier perceptual layers*: "if this affects earlier processing layers, this process is referred
to as embodiment." Decoding only to the final CLS or mask-pooled vector therefore demonstrates
embodiment down to a frozen encoder's **output**, which is a strictly weaker statement than the one
the paper makes.

So the honest framing is not "one layer versus the whole model." It is: **how deep does embodiment
reach?** That converts a scoping compromise into the experiment of Capability 5.

#### The target ladder

| Target | What it tests | Cached? | Cost |
|---|---|---|---|
| **T0** pooled vector (CLS / mask-pooled) | embodiment to the encoder's output | **yes** | free |
| **T1** intermediate block features, layers ℓ | **how deep embodiment reaches** | no — subset re-extraction | hours |
| **T2** patch-token grid at the final block | spatial structure, not just a summary | no — only pooled features were cached | hours |
| **T3** pixels / masked crops | qualitative illustration | no | GPU-day, plus a trained decoder |

T0 is the plan as written. T1 is the scientifically interesting extension and the direct answer to
the depth question. T2 is worth noting because inverting a *mask-pooled* quantity is many-to-one and
inherently lossy — the token grid is the tractable spatial target, not pixels.

#### Keep the measurement at feature level

A more powerful decoder is a liability for evidence. If `g⁺` reaches pixels and the reconstruction
looks like a dog, that is consistent with the index embedding carrying dog-information *and* with a
generative prior hallucinating a plausible dog from very little. This is the standard failure mode
of generative interpretability, and it is why Capability 1 scores grounding by **retrieval** rather
than reconstruction error: retrieval against held-out observations is much harder to fake.

**Rule: T0 and T1 produce the evidence. T3 produces at most one clearly labelled illustrative
figure.**

#### Do not make DINO trainable

Folding the encoder into a *trainable* `g` would change the project rather than extend it:

- the entire pipeline is built on cached float16 tables precisely so experiments are cheap;
  unfreezing the encoder discards that and makes every run a GPU job;
- it collides directly with the scale analysis. A trainable encoder is exactly the free scale
  parameter the original had (see [scale and normalization](tb_scale_and_normalization.md) §1.2),
  so it would quietly absorb the readout/write conflict — you would stop *observing* the phenomenon
  you are trying to characterize;
- the oracle-mask framing depends on the encoder being fixed: the claim is about binding and memory
  given grouping, not about learning better features.

If a learnable component is wanted in `g`, use the projection head already discussed — a small map
after frozen DINO — and run it as a named condition, not as the default.

### 3.5 Visualization without a generative model

There is a free way to *show* what an index has become, and it is more paper-faithful than a
generative decoder.

Decode `σ(a_k)`, retrieve the nearest cached DINO features among held-out observations, and display
those mask crops. No decoder to train, no re-extraction, no generative prior to confound the result.

This is exactly how the original paper visualizes recall: Figures 9 and 12 do not show generated
images, they show **retrieved bounding-box contents** for the sampled entities. The equivalent panel
here — "activate the index for this identity, and here are the crops it retrieves" — comes for free
from the Capability 1 retrieval metric, and it is the qualitative figure most likely to end up in
the thesis.

---

## 4. Three training regimes, in order

### R1 — post-hoc probe on a frozen checkpoint *(do this first, always)*

Freeze a trained TB checkpoint. Train only `g⁺` on `(γ, ν)` pairs collected by running the
schedule. Gradients never touch `A` or the evolution operator.

This is the clean scientific question — *is this information present in `γ` and in `A`?* — and it
cannot damage the model or contaminate any existing result. R1 alone is a complete, publishable
probe study, and it answers capabilities 1, 2 and part of 3 below.

### R2 — joint training with a reconstruction term

Add `λ · reconstruction_loss` to the training objective; gradients flow into `A` and the TB. This
is QTB Section 15.9's "autoencoder learning" and the self-supervised use the paper anticipates.

The question changes from *is it there* to *does requiring it help*: does reconstruction improve
downstream perception, few-shot identity grounding, or robustness to missing evidence? Sweep `λ`,
and include `λ = 0` as the control. Note that R2 changes the checkpoint, so it must be evaluated
against the full metric suite, not only reconstruction.

A likely and reportable outcome is that R2 improves grounding and few-shot behaviour while
slightly hurting headline predicate accuracy. That trade-off is a result, not a failure.

### R3 — decoder in the loop (true embodiment)

Feed the decoded `ν̂` back as an input drive, so top-down reconstruction re-enters perception. This
is the strongest reading of the paper's claim that index activation "might even activate earlier
perceptual layers."

Treat this as genuinely risky: it is a positive feedback loop and it can collapse to a fixed point
that ignores input. Gate it explicitly (`q ← q + μ_top · g⁺(γ)` with `μ_top` small and swept from
zero), cap the number of loop iterations, and monitor whether the state stops tracking the actual
input. Do not attempt R3 before R1 has shown that `g⁺` decodes anything meaningful.

---

## 5. What it unlocks, with metrics and mandatory baselines

### Capability 1 — index grounding (embodiment proper)

**Setup.** No perceptual input. Feed `σ(a_k)` for a single index. Decode.

**Metric — use retrieval, not MSE.** Take `ν̂_k`, retrieve nearest neighbours among held-out object
observations, and report precision@K for observations whose true label is `k`, plus the rank of the
true class centroid. Retrieval is interpretable and robust to the scale issues in Section 2 in a
way that raw reconstruction error is not.

**Baselines.** (a) the mean held-out feature of class `k` — an oracle prototype and effective upper
bound; (b) a random index column; (c) a label-shuffled control; (d) L0.

**The hierarchy extension, which is the novel part.** Ask whether the decoded space is
*compositional*: is `decode(a_coarse)` near the centroid of `decode(a_basic)` over its children,
and likewise down to `fine`? Your four disjoint levels make this well posed. A model whose
hierarchy is only a set of independent classifier rows will fail this; one whose hierarchy is
represented compositionally in `A` will pass. Nothing in either paper tests anything like it, and
it costs one extra evaluation loop.

### Capability 2 — autoencoding and workspace retention

**Setup.** Decode `σ(q)` at each point in the schedule and compare to the feature that was
integrated.

**The interesting version is not reconstruction, it is the retention curve.** After the full
schedule — scene → evolve → subject → feedback → evolve → object → feedback → evolve → union — how
much of the *subject's* appearance is still decodable at the predicate window? Plot decodable
information about each source against schedule position.

This is a direct empirical probe of the one-brain hypothesis of Section 8.6: a single global
representation layer, reused across windows, must lose earlier content. Nobody has measured that
loss. It also gives a principled diagnostic for when a longer schedule needs a larger `state_dim`.

**Baseline.** L0, which will be strong here by construction.

### Capability 3 — the surprise signal

**Setup.** `q → evolve → decode`, compared against the feature actually observed at the next
window or frame. The residual is per-step prediction error.

**Validation, which PVSG uniquely enables.** Do not just produce the signal — check it against
ground truth. Correlate prediction error with annotated **relation onset and cessation frames**
(the strata defined in experiment B3 of the experiment program). If prediction error spikes at real
relational state changes, the signal is a legitimate basis for the episode-boundary policy;
if it tracks only camera motion or object scale, it is not, and the boundary policy needs a
different signal. This is a cheap, decisive check that turns a hand-wavy component into a validated
one.

### Capability 4 — imagination and rollout

**Setup.** Observe a prefix, stop external input, unroll evolution `n` steps, decode each step, and
compare against the actual future features PVSG contains.

**Mandatory baseline: persistence.** At 5 FPS, consecutive DINO features are extremely similar, so
"predict the next feature to be identical to the current one" is a very strong baseline. Any
rollout claim that does not beat persistence at each horizon is meaningless. Report cosine against
horizon `n`, with persistence and a per-video mean-feature baseline on the same axes.

### Capability 5 — how deep does embodiment reach?

*This is the experiment the "should `g⁺` cover the whole encoder" question turns into, and it is the
most novel item in the plan.*

**Setup.** Re-extract intermediate block features for a subset of frames — a few thousand is ample,
not the full 147,795 — at a spread of depths (say blocks 3, 6, 9, 12 of the ViT-B/16). Train a
**separate small `g⁺_ℓ` per layer**, each with the same architecture and the same training design as
T0, then run Capability 1's retrieval-based grounding metric at every depth.

**Report grounding quality as a function of ℓ.** That curve is the result.

**Why it matters.** The paper's embodiment claim is directional, so the curve is a genuine test
rather than a description:

- if grounding decodes well to late layers and **non-trivially to early ones**, index feedback is
  embodiment in the paper's sense — a top-down signal that reinstates perceptual structure, not only
  a semantic label;
- if it works *only* at the final layer, embodiment in this model is a late-stage semantic
  phenomenon and the neuroscience framing of §7.5 overstates what the architecture does. That is a
  substantive negative result about a published claim, and it is worth as much as the positive one.

**Controls.** Each layer has different dimensionality and feature statistics, so grounding scores
are not comparable across ℓ without normalization. Use per-layer controls: the oracle class-centroid
at that layer as the ceiling, and a random-index column as the floor, then report grounding as a
*fraction of the achievable range at that depth*. Without this the curve measures layer statistics
rather than embodiment.

**Cost.** One subset re-extraction plus four small heads. This is the only item in the plan that
needs new features, and it is bounded — hours, not a re-run of the pipeline.

---

## 6. The memorization confound, and the one design that avoids it

This is the most important methodological point in the plan.

A decoder trained on `(γ, ν)` pairs *including* index-only states would trivially memorize a
lookup table from index embeddings to class-mean features. It would then score well on grounding
while having demonstrated nothing about the Tensor Brain — only that a 768×768 matrix can store 226
class prototypes, which it obviously can.

**The design that makes the test sharp:** train `g⁺` **exclusively on perceptual states** `σ(q)`
arising from real observations, and never show it `σ(a_k)` during training. Then, at evaluation
time, feed `σ(a_k)` — a state the decoder has never seen and which is out of distribution for it.

If decoding still lands in the right region of feature space, the conclusion is strong and
non-trivial: *the index embedding lives on the same manifold that the decoder learned from
perception*, which is precisely what grounding means. If it does not, the honest conclusion is that
index embeddings are classifier rows rather than prototypes — also a real finding, and one that
would materially affect how the rest of the framework should be interpreted.

Additional controls:

- evaluate class-index grounding on **held-out videos** (`heldout_video`), so the retrieval targets
  are not observations the TB was trained on;
- for identity indices, use `blocked` so query observations are temporally separated from the
  exposures that shaped the column;
- report the scale-matched and unmatched conditions from Section 2 separately.

---

## 7. Staged plan and cost

| Stage | Work | Cost | Exit criterion |
|---|---|---|---|
| 0 | Scale diagnostic: log `‖a_k‖`, `‖q‖`, feedback ratios, `σ` histograms | ~0.5 day | know whether scale matching is needed, and whether P-SA feedback is negligible — **run this regardless of the decoder** |
| 1 | L0 baseline + L1 linear decoder, R1 on a frozen checkpoint, role heads | ~1 day | L1 beats L0 on reconstruction |
| 2 | Capability 1: grounding by retrieval, with all four baselines and the OOD design of Section 6 | ~1 day | a defensible yes/no on grounding |
| 3 | Capability 2: retention curve across schedule positions | ~0.5 day | a decay curve with error bars |
| 4 | Capability 3: prediction error, validated against onset/cessation strata | ~1 day | correlation with real boundaries, or a clear negative |
| 5 | Hierarchy compositionality in decoded space | ~0.5 day | parent-centroid relation holds or fails |
| 6 | Capability 4: rollout versus persistence | ~1 day | beats persistence at some horizon, or does not |
| 6b | **Retrieval figure** (§3.5): decode `σ(a_k)`, show the crops it retrieves | ~2 hours | the qualitative panel, free from stage 2 |
| 7 | **Capability 5: depth of embodiment.** Subset re-extraction at 4 depths, one `g⁺_ℓ` each, per-layer controls | ~3 days | a grounding-versus-depth curve |
| 8 | R2 joint training, `λ` sweep, full metric suite | ~1 week | trade-off characterized |
| 9 | R3 in-the-loop, gated, small `μ_top` | ~1 week | stable, or documented as unstable |
| — | *Optional* T3 pixel decoder, one illustrative figure only | ~1 GPU-day | explicitly labelled illustration, never evidence |

**Stages 0–6b are about a week and answer the scientific questions.** Stage 7 is the depth
extension — the one item that justifies new extraction, and the strongest addition if a fuller
embodiment chapter is wanted. R2 and R3 are further extensions and should only follow a positive R1.

Stages 0–6b run entirely on the cached float16 feature tables: no re-extraction, no GPU cluster time
beyond what training the base checkpoints already costs. A 768×768 linear head over 1.5M object
observations trains in minutes. **Stage 7 is the only stage needing new features, and it is bounded
to a few thousand frames at four depths.** DINO itself stays frozen throughout — see §3.4 for why
unfreezing it would both discard the cached-feature economics and mask the scale phenomenon under
study.

---

## 8. What would falsify this, and what each outcome means

The plan is worth running because every outcome is informative:

- **`g⁺` decodes perceptual states well but index-only states fail, even scale-matched.** Index
  embeddings are classifier rows, not grounded prototypes. Embodiment as stated does not hold in
  this configuration. This is the most likely outcome and it is a substantive negative result about
  a published claim.
- **Index grounding works.** The autoencoder framing is validated, and R2/R3 become well motivated.
  Grounding also becomes a usable diagnostic for every other index in the vocabulary.
- **L0 matches the learned decoder on grounding.** `A` columns have become feature prototypes
  directly. Surprising, and the most interesting result available here — it would say the shared
  bidirectional matrix is doing something stronger than either paper claims.
- **Prediction error does not correlate with annotated boundaries.** The episode-boundary policy
  (component 3.9) needs a different signal, and knowing this before building it saves the larger
  effort.
- **Rollout never beats persistence.** Imagination and future episodic memory are not supported at
  this timescale on this data; either the timescale is wrong (see multi-timescale evolution,
  component 3.13) or the claim does not transfer to passive video.
- **Grounding decodes to late layers only (Capability 5).** Embodiment in this model is a
  late-stage semantic phenomenon rather than a top-down reinstatement of perceptual structure, and
  the neuroscience framing of §7.5 overstates what the architecture does.
- **Grounding decodes non-trivially to early layers.** Embodiment holds in the paper's stronger
  sense. This would be the most striking positive result available in the whole plan, because it is
  the claim the original could not test at all.

Stages 0–6b cost about a week, need no new data, and are the cheapest way to convert three
unfalsifiable claims — embodiment, autoencoding, imagination — into measurements. Stage 7 adds the
depth dimension for a bounded subset re-extraction, and converts the scoping question "which layer
should `g⁺` target" into the substantive question "how deep does embodiment reach."

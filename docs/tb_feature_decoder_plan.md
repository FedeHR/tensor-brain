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
| 7 | R2 joint training, `λ` sweep, full metric suite | ~1 week | trade-off characterized |
| 8 | R3 in-the-loop, gated, small `μ_top` | ~1 week | stable, or documented as unstable |

**Stages 0–6 are about a week and answer the scientific questions.** R2 and R3 are the extensions
and should only follow a positive R1.

Everything runs on the cached float16 feature tables; no re-extraction, no GPU cluster time beyond
what training the base checkpoints already costs. A 768×768 linear head over 1.5M object
observations trains in minutes.

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

The component costs about a week for the parts that matter, needs no new data, and it is the
single cheapest way to convert three separate unfalsifiable claims — embodiment, autoencoding,
imagination — into measurements.

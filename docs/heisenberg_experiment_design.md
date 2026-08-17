# Designing the experimental chapter for the Heisenberg update

Scoping document, 2026-08-17. Covers two questions:

1. **A real task with a learned `A`** — how good is the Heisenberg approximation
   when the index layer is fitted to data rather than sampled, and does the
   fidelity gap matter for the decisions the state feeds?
2. **Where the Heisenberg update could matter in modern deep learning** — which
   properties fill a real gap, and which are reframings of solved problems.

Everything numeric below was run during scoping and is reproducible from the
scripts named in each section. Nothing here has been written into the thesis.

---

## 0. Recommendations, up front

**Task 1.** Build it on **MS-COCO**: latent `x` = the 12 supercategories present
in an image (`2^12 = 4096` states, exact enumeration is a 4096×12 matmul), symbols
`k` = content words drawn from the five human captions, `M` swept 2–10. COCO
supplies two annotation channels over *the same* images and *the same* latent —
captions (saliency-gated) and instance annotations (exhaustive) — which turns the
theory's central claim into a paired, within-dataset test instead of a simulated
one. No feature extraction, no GPU for the data, CC BY 4.0, instant download.

**Task 2.** Do not go looking for a leaderboard win. The existing analysis
already rules out the advantages that would produce one — no robustness gain
under misspecification, worse under redundant evidence, learning does not fix the
approximation. The rule's genuine edges are *structural*: exactness under
selection-gated evidence, `O(n)` cost independent of state-space size, exact
order invariance, and being the provable zeroth-order term of CAVI. So the
target must be a setting whose **pipeline shape** matches, not a task where the
numbers happen to be better. Ranked candidates are in §3.

**One framing change that affects both.** Posterior fidelity and downstream
decision quality genuinely dissociate, and they dissociate in the direction that
favours the additive rule. §1.4 shows plain Heisenberg beating exact Bayes on
downstream NLL in 10/10 seeds while being, by construction, further from the
exact posterior. Fidelity belongs in the chapter as a *diagnostic*; the decision
is the objective.

---

## 1. Results established while scoping

These are new. They change what the experiment should measure, and two of them
correct existing material. Every table below is reproduced by a script in
[`experiments/heisenberg_scoping/`](../experiments/heisenberg_scoping/); see its
README for how to run them.

### 1.1 A one-parameter family unifies the two measurement models

Let a concept window open with probability `π(x) = Z(x)^τ / C`, then draw
`k ~ softmax(a0 + Aᵀx)`. Over `M` fired windows,

```
P(x | k_1..k_M)  ∝  p(x) · exp(Σ_m a_{k_m}ᵀx) · Z(x)^{M(τ−1)}
```

so the additive rule needs a correction of `(1−τ)·M·log Z(x)`. `τ = 0` is the
unconditional model of chapter 3; `τ = 1` is the saliency-gated model where plain
Heisenberg is exact. Verified over 12 random models (`n=8, K=16, M=3`):

| τ | KL(exact ‖ τ-corrected) | KL(exact ‖ plain additive) |
|---|---|---|
| 0.00 | −1.0e-16 | 0.4694 |
| 0.25 | −3.7e-17 | 0.2623 |
| 0.50 | 5.0e-18 | 0.1142 |
| 0.75 | −7.1e-17 | 0.0275 |
| 1.00 | 1.6e-17 | 0.0000 |

Because only the affine part of `log Z` is needed in practice, the usable rule
stays a pure sum and keeps exact order invariance:

```
q ← q + a_k − (1−τ)·c
```

**Why this matters.** On real data you cannot *assume* a measurement model. But
τ is estimable. Simulating a corpus where the count of recorded observations is
gated (`count | x ~ Poisson(λ·Z(x)^τ)`) and fitting τ by Poisson regression of
counts on `log Z`:

| true τ | n=500 | n=2000 | n=10000 | n=50000 |
|---|---|---|---|---|
| 0.00 | 0.016±0.053 | 0.019±0.050 | −0.004±0.016 | 0.000±0.010 |
| 0.25 | 0.278±0.079 | 0.236±0.040 | 0.248±0.021 | 0.252±0.009 |
| 0.50 | 0.536±0.074 | 0.510±0.032 | 0.495±0.020 | 0.500±0.008 |
| 0.75 | 0.780±0.069 | 0.756±0.044 | 0.742±0.016 | 0.750±0.006 |
| 1.00 | 1.014±0.081 | 0.996±0.043 | 0.995±0.019 | 1.000±0.006 |

Unbiased, and a few thousand labelled instances separate τ=0 from τ=1
decisively. **τ becomes a measurable property of a data pipeline, and it tells
you exactly how much normalizer correction that pipeline needs.**

⚠️ **The naive version of this test is confounded.** `log Z(x)` is large exactly
when many categories are present, and images with many categories present have
more objects to annotate for reasons unrelated to saliency gating. A plain
regression of annotation count on `log Z` reports τ > 0 trivially. The test is
only meaningful against a corpus whose annotation is known to be **exhaustive**,
so the two processes can be contrasted on the same images with the same latent.
This is the single strongest argument for the COCO design in §2.

*Scripts: `tau_check.py`, `tau_identify.py`.*

### 1.2 The general gate law

For *any* trigger of the form `π(x) = g(Z(x))/C`, the exact posterior needs

```
correction  =  M · [ log Z(x) − log g(Z(x)) ]
```

Verified to machine precision for four gate shapes (12 models, `M=3`):

| gate | KL(exact ‖ corrected) | KL(exact ‖ plain additive) |
|---|---|---|
| unconditional, `g(Z)=1` | −1.0e-16 | 0.4694 |
| full gate, `g(Z)=Z` | 1.6e-17 | 0.0000 |
| partial, `g(Z)=Z^0.5` | 5.0e-18 | 0.1142 |
| logistic, `g(Z)=Z/(1+Z)` | 1.4e-17 | 0.0021 |

Exact cancellation requires a gate **linear in `Z`** (Bernoulli/Poisson
thinning). A discrete-choice modeller reaching for the logit-with-outside-option
form `Z/(1+Z)` would *not* get cancellation in general.

### 1.3 …but the QTB offset widens which gates qualify

The logistic row above looks nearly harmless (0.0021). That is not general — it
is a consequence of the scale of `Z`. Sweeping an offset shift:

| E[Z] | sd log Z | sd log(1+Z) | plain \| uncond | plain \| logistic | plain \| full gate |
|---|---|---|---|---|---|
| 0.063 | 0.363 | 0.021 | 0.4694 | 0.0021 | 0.0000 |
| 1.256 | 0.363 | 0.196 | 0.4694 | 0.1424 | 0.0000 |
| 25.2 | 0.363 | 0.347 | 0.4694 | 0.4274 | 0.0000 |
| 507 | 0.363 | 0.362 | 0.4694 | 0.4671 | 0.0000 |

A saturating gate cancels only where it is locally linear, i.e. `Z ≪ 1`. And the
paper's own offset `a0_k = −Σᵢ softplus(A_ik)` forces `E[Z] = K·2^{−n}`, which
stays far below 1 at any realistic `K/n` (at `n=12, K=500`: 0.12). **The QTB
normalization convention places the model in the regime where a broad class of
saliency gates — not only the exactly-linear one — make the additive update
nearly exact.** That widens the trigger theorem's reach considerably, and it is
a second, independent reason the offset is load-bearing rather than cosmetic.

*Scripts: `gate_family.py`, `gate_scale.py`.*

### 1.4 The scene A/B result is confounded — correction to `tb_update_generalized.md` §10

§10 reports a ranking flip between an unconditionally-sampled corpus and a
saliency-gated one, justified by: *"Scenes are drawn from the model's own prior
in both, so 'exact Bayes' is genuinely exact for A."*

**That does not match the code.** `scene.scene_dataset` builds scenes with
`sample_scene`, which draws from a **5-situation mixture** with correlated
categories, while the inference model assumes an independent Bernoulli prior. The
prior is misspecified in *both* arms. Rebuilding the corpus from the model's own
factorized prior separates the two causes (10 corpus seeds, 300 scenes, 4 named
objects, paired per seed):

| corpus | measurement | paired TB − ExactBayes (NLL) | TB better |
|---|---|---|---|
| situation mixture (as shipped) | unconditional | **−0.186 ± 0.058** | 10/10 |
| situation mixture (as shipped) | gated | −0.889 ± 0.044 | 10/10 |
| model's own prior | unconditional | **+0.450 ± 0.060** | **0/10** |
| model's own prior | gated | −0.464 ± 0.044 | 10/10 |

**The flip is real — but only with a correctly specified prior.** As shipped,
plain TB wins in *both* arms, so that experiment does not demonstrate a flip; it
shows TB winning everywhere for two confounded reasons. Prior misspecification
and measurement-process misspecification both favour the additive rule.

This is a design requirement, not a footnote. **On real data the factorized prior
is always misspecified**, because real category co-occurrence is correlated. The
chapter must decompose the additive rule's advantage into a prior component and a
gating component, or the headline is uninterpretable.

*Scripts: `seed_sensitivity.py`, `confound_test.py`.*

### 1.5 Realistic structure makes the free gauge fix worth more

With ontology-structured `A` (PVSG hierarchy, 12 coarse categories, K=95
objects), `Var[log Z] = 0.092`, giving measurable KL of 0.16 / 0.51 / 0.93 nats
at `M = 2 / 4 / 6`. So the experiment is not at risk of being a null.

More interestingly the **affine fraction is 0.735**, against ~0.60 in the
synthetic study. The gauge fix `A ← A − c1ᵀ` — free, order-invariant, a
re-normalization of trained weights rather than a change to the update rule —
removes 66–70% of the error here. **Prediction to test on learned `A`: the
affine fraction rises further with realistic structure.**

*Script: `gap_probe.py`.*

---

## 2. Task 1 — the experiment

### 2.1 What is already covered

Chapter 3 is entirely synthetic. `A` is sampled i.i.d. Gaussian and never
learned; the inference model exactly matches the data generator; `n ≤ 18` by
enumeration; the only task metric is per-bit accuracy against the generating
state. Its own scope statement names the gaps verbatim:

> "The index matrix is sampled rather than learned; the inference model matches
> the data generator; […] Candidate restriction, perceptual uncertainty, state
> evolution, measurement selection, and prior mismatch are absent."

There is **no mention of any real dataset anywhere in the thesis**. Two further
openings: the corrected update `q ← q + a_k − c` is derived in ch.4 (~line 1054)
but never implemented or measured, and the `D_local` diagnostic is defined and
never evaluated.

### 2.2 Dataset: MS-COCO, supercategory latent + caption-word symbols

| | |
|---|---|
| Access | `annotations_trainval2017.zip` (~241 MB) + `captions_train2017.json`; images not needed. CC BY 4.0, direct download. |
| Scale | 118,287 train / 5,000 val images, 5 captions each |
| `x` | the 12 COCO supercategories present → **n = 12, 4096 states** |
| `k` | lemmatized content words from captions, top **K ≈ 500–2000** |
| `M` | 20–40 available symbols/image; sweep **M = 2–10** |
| Downstream | predict the 12-dim presence vector from revealed words |

Four properties make it the right choice:

1. **Ground-truth `x`.** Derived from the exhaustive instance annotations — a
   *different* channel from the captions that supply the symbols. So posterior
   fidelity *and* task accuracy are both measurable against truth.
2. **Two annotation channels over the same latent.** Captions are
   saliency-gated; instance annotations are exhaustive by protocol. The A/B
   contrast that §1.4 showed is confounded in simulation becomes a paired,
   within-image, same-latent comparison.
3. **`n = 12` exactly in the enumerable range**, with `K/n ≈ 40–170` — past the
   `K/n ≤ 16` range chapter 3 swept, in the direction chapter 3 found favourable.
4. **No feature extraction, no GPU for the data.** The first exact-vs-additive
   numbers are reachable on the Mac inside the smoke-test budget, before anything
   goes to Slurm.

It also permits something chapter 3 structurally cannot: with 118k images at
`n=12` you can estimate the **full empirical joint prior over all 4096 states**,
and so decompose the error into "non-factorized prior" and "non-factorized
likelihood" — which is exactly the decomposition §1.4 proved is necessary.

⚠️ **The degeneracy trap.** Any framing where `x` is a lookup from `k` voids the
experiment: `A` becomes an indicator matrix and both rules are trivially perfect.
COCO instance→supercategory is deterministic metadata, so the instance channel
must hold out half the 80 fine categories as symbols and define `x` from the
other half. This same trap **eliminates PVSG** — its reviewed hierarchy maps each
fine label to its coarse parent deterministically — and Instacart
(product→department). Worth stating in the thesis so a reader does not ask why
the existing PVSG pipeline was not reused.

**Other pitfalls.** `person` appears in ~half of all images while
`accessory`/`indoor` are rare — report per-supercategory metrics and consider an
`n=10` variant. Supercategories are strongly correlated (`kitchen`+`food`+
`appliance`), which is a feature: it is precisely what a factorized belief gets
wrong.

**Backups.** *eBird* complete checklists — the ecological occupancy model *is*
latent presence + gated detection, with a 20-year literature and genuine
certified non-detections for the silence rider; costs a 7-day access request and
has **no ground truth for `x`**. *Coat / Yahoo R3 / Open Bandit Dataset* — the
only designs where gated vs ungated evidence is **randomized** rather than
argued; small, but they convert the gate from an observational claim into an
intervention. Suggested shape if time allows: COCO as the main experiment
(measured gate), eBird as ecological validation (structural gate), Coat as a
randomized control — three genuinely different gate arguments rather than three
variations of one.

### 2.3 Protocol

**Fit.** Learn `(A, a0)` by maximum likelihood of `P(k|x)` on ground-truth `x`.
This keeps the likelihood component well specified so the comparison is about the
*update rule*, not model quality. The prior is fitted separately — both a
factorized Bernoulli (matching the theory) and the full empirical joint (the
ceiling), so the prior-misspecification component of §1.4 is measured rather than
assumed.

**Compare.** All rules already exist in `experiments/bayes_approximation/` and
operate on a plain `GenerativeModel(q_prior, A, a0)` dataclass, so a learned `A`
drops straight in: `prior`, `tb`, `tb_affine` (gauge-fixed), `tb_predictive_error`
(gradient / secant / CAVI), `assumed_density_filter`, `exact`, `exact_marginals`.
Add one non-TB baseline of comparable capacity — a DeepSets/MLP multi-label
classifier over the same symbol set — to answer "does the probabilistic machinery
buy anything over just training a classifier on the same inputs?"

**Conditions.** Keep the set minimal:

| # | condition | question |
|---|---|---|
| C1 | ML fit, factorized prior | the baseline comparison |
| C2 | C1 + gauge fix | is the free correction free on real data? |
| C3 | C1 + empirical joint prior | how much of the gap is the prior? |
| C4 | caption channel vs instance channel | does the ranking flip? |

End-to-end training through the update, and the `Var[log Z]` penalty, are a
*different* question (does learning fix the approximation?) — the synthetic study
answered it negatively, and replicating that on real data is a strong result but
a separate arm. Hold it back unless C1–C4 land.

**Metrics, in priority order.** Downstream first: mAP and macro-F1 on
supercategory presence, Bernoulli NLL, ECE. Then fidelity as diagnostic: joint
KL, marginal KL, recovery `R`. Then the structural advantages that are real and
cheap to measure: cost per symbol, and **order invariance** — permute the naming
order and measure how far each rule's belief moves. Measured on the scene model:

```
TB          1.1e-16      TB-affine   5.6e-17
TB-secant   9.1e-02      Exact Bayes 4.4e-16
```

Note what this does and does not show: order invariance is an advantage over
*state-dependent corrections*, not over exact Bayes, which is order-invariant
because it is exact. Say so.

### 2.4 The three questions, and what each outcome would mean

1. **How good is the approximation with a learned `A`?** Measured by `Var[log Z]`
   and joint KL. §1.5 predicts a measurable but not catastrophic gap, with a
   higher affine fraction than synthetic — i.e. more of it removable for free.
2. **Does the fidelity gap matter downstream?** §1.4 already shows the answer can
   be *no*, and can even invert. This is the chapter's most valuable result
   either way: a clean demonstration that posterior fidelity is the wrong
   objective would be more interesting than confirming it is the right one.
3. **Does the measurement process determine which rule wins?** The C4 contrast,
   with the τ estimate of §1.1 as the quantitative bridge. This is the first test
   of the theory's boldest claim on real data.

**Pre-registered risks.** (a) If learned `A` turns out nearly flat, questions 1–2
become null — mitigate by measuring `Var[log Z]` in a pilot *before* committing.
(b) Effect sizes in the scene pilot were ~0.2–0.9 nats against ~5 nats of NLL, so
paired comparisons with bootstrap CIs over instances are required, not optional.
(c) The τ confound of §1.1 — only the exhaustive channel makes the estimate
interpretable.

### 2.5 What to reuse, what to build

Reuse: the entire inference ladder, `general.gauge_fix`, `general.affine_correction`,
`general.triggered_posterior`, `temporal.silence_correction`, `diagnostics.*`, the
`scene.evaluate_corpus` / `scene.order_invariance` harness, the cross-worktree
import bridge already working in `thesis/section2/run_experiments.py`, and
`cluster/pvsg/submit.sh` as a Slurm template.

Build: (i) a COCO adapter producing `(x, [k], channel)` triples; (ii) an ML fit
for `(A, a0)`; (iii) a Monte-Carlo estimator for `Var[log Z]` and the affine
coefficients — ~20 lines, since `affine_correction` already uses the closed form
`c_i = E[log Z | x_i=1] − E[log Z | x_i=0]`, which samples fine; (iv) a
`docs/fidelity.md` ledger row, since this is an experimental extension.

Note `all_states` refuses `n > 20` and every `Var[log Z]`/gauge-fix routine
currently enumerates `2^n`. At `n=12` that is irrelevant; the MC estimator is
needed for task 2, not task 1.

### 2.6 Results — the experiment is built and run

Implemented in [`experiments/coco_heisenberg/`](../experiments/coco_heisenberg/),
tested in `tests/test_coco_heisenberg.py` (10 tests), and cross-checked against
the reference rules in the analysis worktree to **≤ 2.2e-15** on every rule.

117,266 images, 93,813 train / 23,453 held out, `n = 12`, `K = 1000`, 6,000
evaluation images per point. The fit takes 14 s on CPU. Held-out symbol NLL is
**5.639** against 6.908 for a uniform vocabulary, so the learned layer carries
1.27 nats about the named word — `A` is doing real work, not collapsing to an
indicator matrix.

`Var[log Z] = 0.0139`, **affine fraction 0.690**.

**Downstream (NLL of the true supercategory set) and fidelity (joint KL), by M:**

| rule | M=1 | M=2 | M=4 | M=6 | M=8 |
|---|---|---|---|---|---|
| prior | 5.453 / 0.222 | 5.453 / 0.518 | 5.453 / 1.180 | 5.453 / 1.778 | 5.445 / 2.299 |
| **heisenberg** | 4.773 / 0.007 | 4.240 / 0.026 | 3.503 / 0.096 | 3.123 / 0.191 | 2.952 / 0.298 |
| **heisenberg-gauge** | 4.766 / 0.002 | 4.215 / 0.009 | 3.420 / 0.040 | 2.969 / 0.090 | 2.718 / 0.152 |
| heisenberg-pe | 4.769 / 0.005 | 4.256 / 0.017 | 3.555 / 0.051 | 3.169 / 0.090 | 2.926 / 0.127 |
| adf | 4.783 / 0.002 | 4.254 / 0.007 | 3.486 / 0.020 | 3.030 / 0.035 | 2.735 / 0.048 |
| exact | 4.783 / 0.002 | 4.266 / 0.006 | 3.528 / 0.017 | 3.089 / 0.027 | 2.794 / 0.034 |
| *exact-empirical-prior* | 4.540 / 0.111 | 3.907 / 0.247 | 3.150 / 0.459 | 2.766 / 0.558 | 2.547 / 0.592 |

**Paired per-image differences, 95% bootstrap CI, negative favours the first rule:**

| contrast | M=1 | M=2 | M=4 | M=6 | M=8 |
|---|---|---|---|---|---|
| heisenberg − exact | −0.0097 | −0.0263 | **−0.0252** | **+0.0343** | +0.1580 |
| heisenberg-gauge − heisenberg | −0.0070 | −0.0244 | −0.0828 | −0.1539 | **−0.2345** |
| heisenberg-pe − heisenberg | −0.0040 | +0.0159 | +0.0528 | +0.0456 | −0.0264 |
| heisenberg − exact-empirical-prior | +0.2325 | +0.3326 | +0.3526 | +0.3574 | +0.4051 |

Every CI above excludes zero.

**Four findings.**

1. **Fidelity and downstream performance dissociate, and there is a crossover.**
   At `M=4` the Heisenberg posterior is 5.6× further from the exact posterior
   (0.096 vs 0.017 nats) yet **wins** on downstream NLL by 0.025 nats. That
   reverses by `M=6` and the loss grows to +0.158 by `M=8`. The `M²` error law
   shows up as a *crossover in a decision metric*: the additive rule's
   misspecification advantage is real but finite, and the accumulating normalizer
   error eventually overwhelms it. This directly answers the question of whether
   fidelity is the right objective — it is not, and now there is a measured
   regime boundary rather than an argument.
2. **The gauge fix wins at every `M`, and the margin grows.** −0.007 → −0.235
   nats, better on 68–76% of individual images, and it cuts ECE at `M=8` from
   0.060 to **0.021**. It is free, `O(n)`, exactly order-invariant, and a
   re-normalization of trained weights rather than a change to the inference
   loop. Chapter 4 derives `q ← q + a_k − c` and never implements it; this is the
   first measurement of it, and it is the chapter's clearest practical
   recommendation.
3. **The `O(nK)` prediction-error correction is not worth its cost here.** It
   improves fidelity but *hurts* downstream NLL at `M = 2, 4, 6`. The free
   `O(n)` gauge fix dominates it on the decision metric at every `M`.
4. **Most of the achievable gain is in the prior, not the update rule.** The gap
   to `exact-empirical-prior` stays at +0.23…+0.41 nats across the whole sweep,
   while the gap to `exact` moves from −0.01 to +0.16. So roughly 70% of what is
   left on the table belongs to the factorized state, which *no* product-of-
   Bernoullis agent can recover regardless of its update rule. This is §1.4's
   required decomposition, measured.

5. **The `M²` error law survives the move to a learned index layer.** Derived and
   tested on synthetic i.i.d. Gaussian `A`, it holds on `A` fitted to COCO
   captions with a measured exponent of **1.99 over `M ≤ 4`** and 1.92 over the
   full range. The coefficient needs the **posterior**-weighted variance, as the
   theory states: `Var_prior[log Z]` over-predicts by 29% at `M=1` and 41% at
   `M=8`, while `Var_posterior[log Z]` gives ratios of 0.75 → **0.94** across the
   sweep. This is the strongest available validation of chapter 4's central law
   outside the synthetic setting it was derived in.

6. **The standard classification metrics disagree with each other**, and this is
   the sharpest practical finding. Dropping `log Z` makes the belief
   overconfident, so at a 0.5 threshold it predicts more positives: macro recall
   rises to 0.716 against exact Bayes' 0.616, macro precision falls to 0.754
   against 0.853. Macro-F1 rewards that on rare categories and puts the additive
   rule **ahead at every `M`** (0.726 vs 0.699 at `M=8`); Hamming and subset
   accuracy are dominated by true negatives and **cross over near `M=6`**; mAP
   is essentially tied. A chapter reporting only macro-F1 would conclude the
   additive rule beats exact inference outright; one reporting only subset
   accuracy would conclude the opposite. Report several.

   A corollary worth stating in the thesis: **mAP is identical for the plain rule
   and the gauge fix to five decimals at every `M`**, because a constant shift
   cannot change within-category ranking. The gauge fix changes calibration, not
   ordering, and a rank-based metric is blind to it by construction.

Full description of the dataset, task, training and metrics, with seven figures,
in [`experiments/coco_heisenberg/README.md`](../experiments/coco_heisenberg/README.md).

**A caveat on the mechanism.** The likely reason the additive rule wins at low
`M` is that two misspecifications partly cancel: the factorized prior misses the
strong positive correlations between supercategories (`kitchen`+`food`+
`appliance`), which makes exact Bayes under that prior *under*-confident relative
to truth, while dropping `log Z` makes the additive rule *over*-confident. The
`exact-empirical-prior` row supports this — with the correct joint prior the NLL
falls sharply — but the chapter should present it as a hypothesis with this
evidence, not as an established mechanism.

---

## 3. Task 2 — where the Heisenberg update could matter

### 3.1 The honest win condition

The existing analysis already forecloses the obvious pitches: **no** robustness
advantage under model misspecification, **worse** under redundant evidence,
learning **does not** drive the model toward the exact regime. So a
leaderboard-win framing is likely to fail.

What survives is structural. The rule wins where the *pipeline shape* matches:
evidence that passed a selection gate, evidence that is a set rather than a
sequence, per-symbol cost that dominates, or a setting where being the provable
zeroth-order term of CAVI supplies theory an incumbent method lacks.

Your point about `n` is right and it matters here: the `n ≤ 20` ceiling is an
artifact of enumerating the *reference*, so it binds only task 1. In task 2 the
comparator is other ML methods, so `n` is unconstrained — at the cost of needing
the Monte-Carlo `Var[log Z]` estimator instead of enumeration.

### 3.2 Prior art that bounds every claim

The single most important scoping finding: **the additive-conjugate mechanism is
not new**, and it has several established homes. Claims must be pitched at the
*specific* deleted term, exactness condition and error law — never at the general
observation.

| what | where it already lives |
|---|---|
| additive natural-parameter updates as approximate Bayes | Khan & Rue, *The Bayesian Learning Rule*, JMLR 2023; Conjugate-Computation VI (2017) |
| contraction ⇒ bounded steady-state error (§13a) | Boyen & Koller, UAI 1998 — this *is* their theorem in a new family |
| the `Z`-gate cancellation | weighted distributions / selection models: Rao (1965), Patil & Rao (1978), Bayarri & DeGroot (1992) |
| softmax denominator = conditioning on a total count | the multinomial–Poisson transformation ("Poisson trick"), Birch 1963 / Baker 1994 / Lang 1996 |
| additive accumulation **plus** the compensator | **Zhang et al., J. Neurophysiol. 79:1017 (1998)** — Bayesian place-cell decoding is `P(x) · Π f_i(x)^{n_i} · exp(−τ Σ f_i(x))`. The additive part is the update; the exponential is exactly the term that cancels under gating. Given this is a brain model, expect this citation in review. |
| "silence is evidence" | event-triggered state estimation with *negative information* — a developed subfield that computes the non-transmission likelihood properly; the constant drift `c_s` is a crude version |
| a constant intercept correction under response-based sampling | Prentice & Pyke, Biometrika 66:403 (1979) |
| the `Var[log Z]` regularizer of §12b | the **self-normalization** penalty `α(log Z)²`, Andreas & Klein (2015) — same objective, invented for unrelated reasons |
| trained readouts converging to a tight frame | **neural collapse** (simplex ETF) |

The neuroscience and self-normalization entries are the two that most change the
thesis. The first *grounds* the update in established neural decoding rather than
undermining it. The second means the flatness regularizer already has a name and
a literature.

### 3.3 Ranked opportunities

**① The delta rule as the first-order correction.** Two independent research
sweeps converged on this. Modern linear-attention states update
`S ← S + βₜ(v − Sk)kᵀ` — "write what happened minus what you predicted" — which is
structurally identical to `q ← q + A(e_k − p)`. The difference is the likelihood:
DeltaNet's correction is one SGD step on `½‖v − Sk‖²`, a **Gaussian** likelihood;
yours is the gradient of categorical cross-entropy. So *the delta rule as
deployed is the Gaussian branch of a conjugate-update family, and the Heisenberg
correction is the categorical branch.*

The Bayesian framing is taken in the Gaussian branch — I verified Gated KalmaNet
(arXiv:2511.21016, Peng, Chattopadhyay, Zancato, Nunez, Xia, Soatto) states in
its abstract that DeltaNet, Gated DeltaNet and Kimi Delta Attention "are
approximations to the KF recurrence under an identity error covariance
assumption." **What is not taken: when that assumption is true.** Your tight-frame
criterion answers it — the covariance stays isotropic, and the delta rule is
exact rather than approximate, exactly when the accumulated keys form a tight
frame. That is a sharp, cheap theory note with a measurable diagnostic
(`‖KKᵀ − cI‖_F`) predicting where full-covariance machinery buys nothing.
Reportedly MIRAS enumerated the objective space (ℓ₂, ℓ_p, Huber, KL retention…)
and never reached a categorical likelihood — so the exponential-family branch
appears unoccupied.

*Test:* MQAR with controlled key geometry, swept tight-frame → anisotropic at
fixed capacity. Small (d_h=32, T≤2048).

**② Mechanistic overconfidence from the dropped `log Z`.** LogitNorm documents
the symptom ("the norm of the logit keeps increasing during training, leading to
overconfident output") with an empirical fix and no derivation. Confidence-
regulation neurons document a mechanism — entropy neurons modulating confidence
through the final LayerNorm's rescaling, token-frequency neurons pulling output
toward the unigram distribution. Nobody has the closed-form derivation. You do,
plus two predictions: overconfidence grows as `½M²Var[log Z]`, and the tilt
`Z(x)^M` favours states well covered by the vocabulary — for which token-frequency
neurons look like the model's *learned counter-measure*.

**③ Tight frame ⟺ exactness, bridged by neural collapse and self-normalization.**
All ingredients published, composition not made: neural collapse says trained
readouts converge to a simplex ETF; your criterion says a tight-frame readout
makes the additive update exactly Bayesian. **Prediction: Heisenberg-vs-Bayes
fidelity should improve monotonically over training, tracking the neural-collapse
metric.** Note this sits in tension with the synthetic finding that training
*raised* `Var[log Z]` — which is itself the interesting question, since that
experiment used a recognition objective on a small model rather than a
classification objective trained to the terminal phase. Cheap to test on a
CIFAR-100/ImageNet readout, which suits the local/cluster split.

**④ Selection-gated evidence in recommenders.** The strongest systems fit. Standard
IPS/propensity methods condition on *observables*; your gate depends on the
**latent**, which the MNAR literature classifies as the hard nonignorable case.
Your result says: for this one gate, no correction is needed — *ignorability by
cancellation*. That is a contrarian, testable disagreement with deployed
practice (the logQ correction in two-tower retrieval adds a normalizer term
back). **Open Bandit Dataset** is the best fit anywhere: ~26M impressions with a
uniform-random logging arm *and* a Thompson-sampling arm, true propensities
logged — the same population sampled both ways, which almost nothing in ML
provides.

**⑤ LLM agent memory.** Best novelty-per-effort. 2026 systems are converging on
probabilistic memory without a justified update rule — one maintains "an
auditable log-odds stance per proposition" with no argument for why addition is
right; another does noisy-OR belief updates with no selective-write gate
modelled. You can supply the justification theorem, plus the prescription that
sessions with *no* write must be counted and drifted — one counter, a few lines.
Benchmarks: LoCoMo, ALFWorld, OAKS. Hard dependency: the rule needs a fixed
symbol vocabulary, i.e. a closed ontology. Scope the claim and say so; the repo's
`IndexVocabulary` work is what makes it defensible.

**⑥ An `O(n)` alternative to mean-field parallel decoding.** Reportedly a June
2026 paper attacks the known failure of masked-diffusion parallel decoding (the
factorized reverse policy commits tokens from independent marginals, producing
"New City" instead of "New York") with sigmoid fixed-point iterations at
`O(m²|V|)` per step. Your rule is the `O(n)` zeroth-order term of the same CAVI,
with an exactness condition and an error bound the incumbent lacks. Direct,
benchmarkable competitor — but verify the paper exists and says this before
building on it.

### 3.4 What to drop, and two framing traps

**Drop:** model merging / task arithmetic (saturated — Fisher merging *is* the
second-order correction, and for an exponential family the Fisher is the Hessian
of the log-partition; the gauge freedom has no weight-space analogue);
associative memory (reframing); the cognitive DDM bridge (no benchmark, no
community); ADF/EP "revival" (correct but cold — use ADF as the correctness
argument, not the framing).

**Trap 1 — RAG is the counter-example, not the application.** Fixed top-k
retrieval is a hard, deterministic, constant-count gate: exactly k passages
return regardless of score. That is *conditioning on the total count*, i.e. the
multinomial side of the Poisson trick, where the normalizer does **not** cancel.
The `Z`-gate appears only where the *number* of recorded items varies with score
mass: threshold retrieval, abstention gates, agentic loops that decide whether to
retrieve again. Stating that distinction is itself a contribution; pitching
fixed-k RAG as the flagship application would be wrong.

**Trap 2 — order invariance is a liability in sequence modelling.** The 2024–2026
arc is explicitly about *destroying* commutativity to get state tracking, because
commutative updates provably cannot do it. Frame exact order invariance as a
property of a **global belief state**, where it is correct and desirable, and
never as an advantage of a token mixer.

**Do not lead with the SVI speedup.** It is the most impressive-sounding and
least defensible number in the package — the obvious objection is "you beat a
generic tool on a model where you know the conjugate structure." Lead with the
ADF-equality and the CAVI-order decomposition; those survive an expert reviewer.

### 3.5 Citation health warning

Of the ~120 references surfaced during scoping, I independently verified exactly
one: **arXiv:2511.21016 (Gated KalmaNet)**, whose title, authors and
identity-covariance claim are confirmed. Everything else — particularly the 2026
arXiv IDs, which are numerous and which I could not check — must be verified
before it enters the thesis. Two were explicitly flagged by the researching agent
as reaching it through a summarising fetch with unverified details.

---

## 4. Parallel decoding for discrete diffusion — a full assessment

The paper is real: **"Mean-Field Parallel Decoding for Discrete Diffusion
Language Models"**, Zoabi, Ali, Ringel & Wolf (arXiv:2606.15805). I read it
directly; what follows is checked against the text, not the earlier second-hand
summary.

### 4.1 What it actually does

Masked diffusion LMs can unmask several positions per forward pass, but tokens
chosen independently from their marginals form incompatible configurations —
"New City" instead of "New York". The paper picks *which* positions are safe to
commit together, by structured inference over binary indicators `S ∈ {0,1}^m`:

```
P(S) ∝ exp( Σ_i c_i s_i  −  Σ_{i≠j} D_ij s_i s_j )
c_i  = log π_i(v_i⁽¹⁾) − log π_i(v_i⁽²⁾)        top-2 log margin (confidence)
D_ij = 1 − JSD(π_i, π_j)/ln 2                    normalized, zero diagonal
```

solved by a mean-field fixed point `q_i = σ(c_i − Σ_j D_ij q_j)`, `R = 2`
iterations from `q⁽⁰⁾ = σ(c)`, at `O(m²|V|)` per denoising step. Average speedup
5.12× over entropy-based decoding on GSM8K / MATH / HumanEval / MBPP with
LLaDA-8B, LLaDA-1.5 and Dream-7B.

### 4.2 The obvious mapping is a re-description, not a contribution

Their state is a product of Bernoullis; their initialization `q⁽⁰⁾ = σ(c)` is the
belief with the interaction term **dropped**; each iteration adds the
state-dependent coupling. That is exactly the thesis's ladder — plain additive
rule, first-order correction, CAVI fixed point — and the correspondence is
structural, not analogical.

But it cuts the wrong way. Their `q⁽⁰⁾` *is* the zeroth-order additive rule, and
their whole contribution is that you need the correction on top of it. Applying
the thesis here would recommend the thing they improved on. Worse, their energy
is **constructed heuristically**, not derived from a likelihood, so there is no
`Z` and the `½M²Var[log Z]` law has nothing to attach to.

Say this plainly rather than dressing it up: on the selection problem the thesis
explains their method, it does not beat it.

### 4.3 The real opening, which their limitations section names

Two facts from the paper decide this, both verified in the text:

1. **They never update the other positions.** After committing, the method
   "proceeds to the next denoising step with a fresh forward pass" — the tokens
   just committed do not correct the still-masked positions' distributions.
2. **`D` is built from marginals alone.** In their own words, the JSD score *"is
   a lightweight proxy for unsafe simultaneous commitment; it captures pairwise
   predictive overlap but does not explicitly model higher-order structure or
   recover the true joint conditional distribution."*

And they discuss no pointwise mutual information, no conditional dependence
measure, and no additive logit correction anywhere.

So the unoccupied move is not *which positions to commit* but **what to do to the
rest once you have committed**. Exactly:

```
log P(x_j = v | ctx, x_i = k)  =  log π_j(v)  +  PMI(x_j = v ; x_i = k)
```

The correction to position `j`'s logits is a PMI vector. Approximating it by a
vector that depends only on the committed token,

```
logits_j  ←  logits_j + a_k          for every still-masked j
```

**is the Heisenberg update**, with `q` the logit vector at a masked position and
`a_k` the column of a shared index matrix. Its exactness condition is the
thesis's own: the update is exact exactly when the relevant log-partition is
affine, i.e. when the PMI is separable across positions and context.

This is complementary to their method, not competing. They commit fewer, safer
positions; this commits and *corrects*, inside one forward pass. The two stack.

Three further things transfer, and none of them exist in that literature:

- **A parameter-free `A`.** Taking `A = W_U W_E^ᵀ` — unembedding times embedding —
  gives a rank-`d_model` additive correction with **no new parameters**, and it is
  precisely the Tensor Brain's own "shared bidirectionally" structure. A full
  `|V|×|V|` table would be ~1B entries and is not an option, so the low-rank
  route is a requirement, not an elegance.
- **The gauge freedom.** `a_k` is identified only up to a per-position constant,
  because softmax is shift-invariant. So the free, order-invariant gauge fix
  applies unchanged — and it is the correction that just won at every `M` on
  COCO (§2.6).
- **A quantitative prediction the field lacks.** The `M²` law says degradation
  from committing `M` tokens in one pass grows quadratically. The field currently
  chooses how many tokens to commit with thresholds and heuristics. A predicted
  *shape* for the accuracy-versus-parallelism curve is a real contribution even
  if it never beats their throughput.

A last argument in favour: JSD similarity is not statistical dependence — two
positions can have identical marginals and be independent. PMI measures
dependence directly, so the proposed quantity is better founded than the
heuristic it would supplement. Their own limitation quote concedes the point.

### 4.4 Exact experimental procedure

> ## ⛔ Stage 0 has run, and it kills this direction
>
> Implemented in
> [`experiments/diffusion_heisenberg/`](../experiments/diffusion_heisenberg/),
> measured on **38,808 (commit, target) pairs** from a 0.6 B masked diffusion LM
> on GSM8K, leave-one-out throughout.
>
> | rule | mean KL, nats | captured |
> |---|---|---|
> | do nothing (what decoders do now) | 0.2008 | 0.0% |
> | additive `q += a_k` (gain 1) | 0.2028 | **−1.0%** |
> | additive `q += 0.40·a_k` (global gain) | 0.1991 | **0.9%** |
> | free `λ·E Eᵀ e_k` (global gain **0.00**) | 0.2008 | −0.0% |
> | additive, per-event gain (**oracle**) | 0.1852 | 7.8% |
>
> The interaction is real and local — a commit moves the adjacent position by
> 0.32 nats, decaying to 0.07 beyond ten tokens — so the rule is not failing for
> want of something to find. It fails because the movement is **not a fixed
> function of the committed token**. The tell is that the global-gain rule works
> only adjacent to the commit (7.9%, collapsing to <1% by three tokens) while the
> *oracle* stays flat at 6–12% everywhere: the direction carries a little signal,
> but the right scale is per-event. That is context dependence, which is exactly
> the assumption §4.3 makes and exactly the risk §4.6 flagged.
>
> **A rescue would require state dependence, which forfeits the `O(n)` cost and
> exact order invariance that motivated the proposal in the first place** — a
> state-dependent correction is just a cheap approximation to the forward pass it
> was trying to avoid. Stages 1–3 below are therefore not worth running. Cost of
> finding out: about forty minutes of laptop compute, which is what stage 0 was
> for.
>
> The contrast with §2.6 is the part worth keeping: on COCO a *fixed* correction
> (the gauge fix) won at every evidence count, and here it does nothing. The
> additive update is useful exactly where the dropped log-partition is close to
> affine in the carried statistics, and a token committed into a sentence is
> nowhere near that regime.
>
> The remaining sections of §4 are kept as written, because the reasoning that
> led to the measurement is what made the measurement worth doing.

**Stage 0 — the de-risking measurement. Do this first; it is cheap and it can
kill the idea.** Nothing is built until this passes.

1. Take a masked diffusion LM (LLaDA-8B-Instruct or Dream-v0-7B) and a few
   hundred prompts from GSM8K.
2. At a denoising step with `m` masked positions, record `π_j` for all `j` — one
   forward pass.
3. Commit the argmax token `k` at one position `i`. Run a **second** forward pass.
   Record `π'_j` for all `j ≠ i`.
4. The ground-truth correction is `Δ_j = log π'_j − log π_j`. This is the object
   an additive rule must approximate.
5. Fit and compare, on held-out steps:
   - **do nothing** (the current behaviour within a pass): `KL(π'_j ‖ π_j)`
   - **additive, token-only**: `KL(π'_j ‖ softmax(log π_j + a_k))`, `a_k` fitted
     across all positions and contexts
   - **additive, gauge-fixed**: the same with the free per-position shift removed
   - **additive, low-rank free**: `a_k = λ · W_U W_E^ᵀ e_k`, one scalar `λ`
   - **distance-modulated**: `a_k` scaled by a function of `|i − j|`
6. Report the fraction of the available correction each captures,
   `1 − KL(π'_j ‖ approx) / KL(π'_j ‖ π_j)`, and `Var` of the residual — the
   direct analogue of `Var[log Z]`, and the quantity the error law needs.

**Decision rule.** If the additive rule captures a large fraction, proceed to
stage 1. If the residual is dominated by context-dependent structure, stop — and
publish that, because it is a clean measurement of *why* parallel decoding is
hard and nobody has made it.

**Stage 1 — intra-pass sequential commitment.** Within one forward pass: commit
the highest-confidence position, apply `logits_j ← logits_j + a_k` to the rest,
re-rank, commit the next, repeat `M` times. Measure accuracy and tokens/second
against `M ∈ {1,2,4,8}`, with and without the gauge fix, and against the paper's
own selection rule as the baseline at matched throughput.

**Stage 2 — the law.** Plot degradation against `M` and test the predicted
quadratic. Estimate `Var[residual]` by sampling (never by enumeration — `|V|` is
32k) and check whether it predicts the coefficient across models and datasets.

**Stage 3 — composition.** Stack the additive correction *on top of* their
mean-field selection and ask whether the commit set can be enlarged at equal
accuracy. This is the only stage that claims a speedup over the incumbent, and it
is deliberately last.

### 4.5 What results are available, ranked by value

1. **The error law predicts the accuracy/parallelism curve.** The most valuable
   outcome and it does not require beating anyone. The field manages parallelism
   with heuristics; a derived `M²` shape with a measurable coefficient explains
   them. This is also the result that most directly validates the thesis's
   central law in a current setting.
2. **A free, parameter-free correction improves parallel decoding.** If
   `A = W_U W_E^ᵀ` with one fitted scalar buys extra tokens per pass, that is a
   strong, cheap, training-free result with an immediately reusable artifact.
3. **A clean negative with a mechanism.** If the additive approximation fails,
   the measured residual structure says *what kind* of dependence parallel
   decoding must model — a sharper statement than the paper's own admission that
   its proxy "does not recover the true joint conditional distribution".
4. **The gauge fix transfers.** Cheap to test at every stage, and it already
   replicated on COCO.

### 4.6 Risks, stated honestly

- **The core assumption is strong.** A state-independent `a_k` says the effect of
  committing a token is the same regardless of context. Language dependence is
  famously contextual — "New" → "York" depends on the rest of the sentence — so
  stage 0 exists precisely to measure this before anything is built.
- **Compute.** Stages 1–3 need GPU inference on 7–8B models. Inference-only and
  cluster-appropriate, but not a laptop experiment. Stage 0 is the cheapest
  possible probe and needs only two forward passes per step.
- **A moving target.** This is a June 2026 paper in a fast area; the baseline may
  shift. The `M²` result (outcome 1) is robust to that, the speedup claim
  (outcome 2) is not.
- **Scope.** This is a *decoding* contribution, not a Tensor Brain contribution.
  It borrows the update rule and its analysis; it says nothing about perception,
  evolution or the index layer. Worth stating so the thesis does not overclaim
  continuity.

---

## 5. Next steps

1. ~~Pilot the COCO fit~~ — **done**; see §2.6. The gap is measurable, the
   dissociation is real, and the crossover is located between `M=4` and `M=6`.
2. **Write the ledger row** in `docs/fidelity.md` classifying the extension.
3. **Remaining task-1 arms**, in order of value:
   - the **instance channel** (C4), giving the paired gated-vs-exhaustive
     contrast on the same images — the only arm that tests the measurement-process
     claim on real data;
   - the **τ estimate** (§1.1) on that contrast, which is what makes the claim
     quantitative rather than binary;
   - a **capacity-matched non-TB baseline** (DeepSets/MLP over the same symbol
     set), to answer whether the probabilistic machinery earns its place;
   - **seed replication** of the fit itself — §2.6 varies evaluation images but
     fits `A` once.
4. ~~Task 2: run stage 0 of §4.4~~ — **done, and it returned a decisive negative**;
   see the box in §4.4. The diffusion-decoding direction is closed. If a task-2
   thread is still wanted, §3.3's remaining candidates are unaffected: ① the
   exactness condition for DeltaNet's identity-covariance assumption and ③ the
   tight-frame/neural-collapse link are both theory contributions testable at
   small scale, and neither depends on what stage 0 measured.
5. **In parallel, cheap and non-committal:** submit the eBird data-access request
   (7-day turnaround), so it is off the critical path if the ecological arm is
   wanted.

Two corrections to fold back into existing material regardless of what happens
next: the §10 confound (§1.4 here) and the `log Z` **affine** criterion, which
the thesis still states as "constant" in chapter 4.

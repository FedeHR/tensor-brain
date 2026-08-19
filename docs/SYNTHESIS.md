# Where the Heisenberg update actually has purchase

Synthesis of a five-front scoping pass plus one new measurement, 2026-08-19.
Companion documents: `front_foundational.md`, `front_gated.md`,
`front_frames.md`, `NOTE_normalization_is_a_gauge_fix.md`. Code:
`experiments/logz_geometry/`, `experiments/tau_instrument/`.

---

## 0. The short version

Three things happened in this pass.

1. **A whole family of candidate applications is dead, for one clean reason I
   had wrong at the start.** Adding *logits* and softmaxing is exactly a product
   of experts — the normalizers are constants in the token index and cancel in
   the final softmax. So classifier-free guidance, contrastive decoding, DoLa,
   proxy tuning, logit ensembling and task arithmetic on logits are **not**
   Heisenberg updates and have no dropped-normalizer problem. The Heisenberg
   update lives **only** where a vector is accumulated in the *latent/state*
   space and read out through a softmax whose partition function depends on that
   state.

2. **The delta-rule / tight-frame direction is dead too, and worse than dead.**
   It has been derived independently at least four times (Preconditioned
   DeltaNet arXiv:2604.21100 Thm 3.1 is claim 5 with `P = I/c`; Longhorn already
   publishes the closed-form gain; Gated KalmaNet states the identity-covariance
   assumption in prose; test-time regression frames the family). And the pitch
   is backwards: under a tight frame the posterior mean is a *pure accumulation*
   with a global scalar, which is plain linear attention — the delta rule's
   `−βSkkᵀ` term is exactly the sequential whitening that tightness makes
   unnecessary. The honest claim is "tight frame ⟹ you don't need the delta
   rule", which is not a thesis.

3. **Two independent fronts converged on the same answer** — selection, restated
   as conditionality (§1) — and that convergence is the recommendation for the
   thesis chapter.

And one result came out of the measurements that I did not expect and that is
sharper than any of the proposals:

> **Energy-based OOD detection measures the coordinates, not the model.** The
> score `-logsumexp(logits)` is exactly the object the gauge freedom moves.
> Reparameterising `W ← W − 1cᵀ` leaves every predicted probability
> bit-identical and drives the detector's AUROC from 0.054 to 0.784 on GPT-2.
> The canonical flat gauge — chosen with **no OOD labels**, purely as the
> coordinates in which the additive update is exactly Bayesian — takes it from
> 0.153 to 0.439, i.e. to chance. (§3)

This is the answer to "can the Heisenberg update say something about a SOTA
task": it does not beat a benchmark, it says a widely-used benchmark statistic
is not identified.

---

## 1. The recommendation: selection, stated as conditionality

Both surviving fronts point at the cancellation identity, but neither pitches it
the way the existing docs do ("find a corpus that happened to be gated"). The
sharper statement, which the foundational front established and which I think is
the thesis:

> The softmax likelihood of a latent-state readout is a **conditional**
> likelihood — it conditions on how many symbols were emitted. The additive
> (Heisenberg) update is the corresponding **unconditional** posterior. They
> agree exactly iff the emitted count is *ancillary* for the state, equivalently
> iff `log Z` is affine in the carried statistic in some gauge. Otherwise they
> differ by the tilt `Z(x)^M`, whose cost `½M² Var[log Z]` is precisely the
> state-information carried by the event count.
>
> **Which of the two is correct is a property of the observation protocol —
> whether non-events were instrumented — not of the inference algorithm.**

Why this is much better than the current framing:

- **It flips the sign on the baseline, not on a method.** The additive rule uses
  *more* of the data; the "principled" normalizer correction throws the count
  channel away. Reviewer-facing version: *your exact-Bayes baseline is the
  approximation.* That is the surprise, and it costs nothing to claim because
  the repo already verified it (§10 of `tb_update_generalized.md`: on gated data
  the correction is the worst rule).
- **It has a fifty-year pedigree that nothing in the repo cites.** Cox's partial
  likelihood has a risk-set softmax denominator obtained by conditioning away
  event times, and **Efron (1977, JASA 72:557)** computed exactly that
  information loss. `Var[log Z]` is Efron's calculation, for a latent-state
  readout. Add Rubin, Dawid & Dickey (1977), Manski & Lerman (1977),
  Prentice & Pyke, Fithian & Hastie.
- **It derives the repo's most surprising number.** The gauge is the
  *transpose* of the case-control intercept shift: case-control moves *column*
  constants (identified, design-confounded); the gauge moves *row* constants
  (wholly unidentified in the multinomial branch, identified in the Poisson
  branch). That is *why* gauge-fixing must be free in fit and large in belief —
  the 53% result becomes a corollary of identification theory rather than an
  empirical curiosity.

### Two corrections to the existing docs, both load-bearing

- **The multinomial–Poisson mapping in `BRIEF.md` is wrong.** The Poisson branch
  does not delete the normalizer; it replaces `−M log Z` with `−(T/C)Z(x)`.
  Neither branch gives the additive rule. Worse, marginalising exposure under a
  Jeffreys prior returns `−M log Z` *exactly*, so "unknown exposure" is not the
  Heisenberg regime either. The correct statement is narrower and still new: the
  MP transformation is always stated for *emission* parameters, where the
  branches agree; for a *latent posterior* they differ by exactly `Z^M`.
- **Do not pitch "silence is evidence".** Han et al. (2015, IEEE TAC 60:2661)
  already design a stochastic trigger that keeps exact inference closed-form.
  The genuine delta is a *duality*: their trigger makes silence conjugate, the
  `Z`-gate makes the report conjugate and leaves silence expensive. One
  defensive page, not a chapter.

---

## 2. The 2026 target: DART-Math as a deliberate anti-gate

The gated front's candidate ① is the one modern hook that survives both filters
and has public data.

Rejection-sampling fine-tuning keeps the rollouts a verifier accepts. Per
accepted trace the acceptance probability cancels against the
conditional-on-acceptance emission — claim 3 exactly, with `Z₊(x)` the policy's
mass on accepted traces. The count varies (Binomial over ~2 orders of magnitude
across a math corpus) and the gate depends on the model's latent competence, not
on surface features of the problem. Both filters pass.

**DART-Math (arXiv:2407.13690, NeurIPS'24) diagnoses exactly this dependence as
a bias and removes it on purpose**, in two flavours:

| corpus | count rule | implied `tau` | regime |
|---|---|---|---|
| vanilla rejection tuning | `c_i ∝ p_i` | `tau ≈ 1` | gate; additive rule exact |
| DART-Math-Uniform | `c_i = const` | `tau ≈ 0` | conditioning on a total |
| DART-Math-Hard | `c_i` rises with difficulty | `tau < 0` | deliberate anti-gate |

The theory says the "bias" being corrected is not a bias: under the verifier
gate the count *is* the sufficient statistic, and flattening it destroys the one
thing that made additive accumulation exact.

The prediction does **not** dispute DART's accuracy gains — fidelity and
decision quality are already known to dissociate. It is on *calibration of the
model's belief about its own competence*, and it is cheap to test on released
checkpoints. Falsifier: if VRT and DART-Uniform have indistinguishable
competence calibration at matched token budget, candidate ① dies.

### Measured (`experiments/tau_instrument/`)

Mean retained responses per query, by MATH difficulty level, over the 7,457
queries of `dart-math-pool-math` that carry a level annotation. No regression,
no modelling — just counts:

| corpus | L1 | L2 | L3 | L4 | L5 | L5/L1 |
|---|---|---|---|---|---|---|
| the DART pool itself | 208.2 | 219.6 | 229.2 | 232.4 | 196.4 | 0.94 |
| **DART-Math-Uniform** | 40.0 | 40.0 | 39.8 | 39.7 | 37.7 | **0.94** |
| **DART-Math-Hard** | 14.3 | 33.5 | 55.0 | 79.9 | 108.7 | **7.61** |

DART-Uniform is flat to within 6% across five difficulty levels — a
constant-count rule, `tau = 0`, the multinomial branch. DART-Hard rises
**7.6×** from easiest to hardest — a deliberate anti-gate, `tau < 0`. Both
confirmed model-free, on public data, in one laptop run.

**Two corrections to how this should be reported.** First, a regression of
retained count on *pool mass* is not interpretable here and I dropped it: the
public pool is itself already balanced (≈216 responses per query at every
level), so pool mass does not proxy the pass rate. Second, and more
interestingly, **the public pool retains only accepted responses** — every one
of its 1.6M rows has `ans_correct = True`, so the rejected attempts are
unrecoverable. That is not a data limitation to apologise for; it is filter F4
confirmed empirically. The failures were never published, which is precisely the
regime in which the additive rule is exact. What the public data cannot support
is a measurement of vanilla rejection tuning's own `tau`, which is 1 by
construction and is reported as a reference line, not as evidence.

---

## 3. New measurement: what `log Z` looks like in a real network

Nothing in this project had ever measured the theory's central quantity on a
trained model. `experiments/logz_geometry/probe.py` does, over 20,000 hidden
states from Pile documents, with the gauge **fit on one half and scored on the
other** (`d` is 576–896, so an in-sample R² is an overfitting artifact — my
first run reported 0.9996 on 512 states purely because the regression was
underdetermined).

| model | `Var[log Z]` | affine fraction (held out) | residual `Var[r]` | gauge removes |
|---|---|---|---|---|
| GPT-2 124M | 4450.8 | 0.9998 | **1.05** | 99.6% |
| SmolLM2-135M | 56.9 | 0.968 | **1.83** | 99.3% |
| Qwen2.5-0.5B-Instruct | 7.9 | 0.838 | **1.29** | 65% |

**The invariant is the last column but one.** Raw `Var[log Z]` spans 560× across
these three models; after the free gauge fix all three land at 1–2 nats². The
gauge fix is verified free: applying `W_U ← W_U − 1cᵀ` changes the softmax by
`≤ 1e-5` (float32 noise) while removing 65–99.6% of the state dependence.

Operationally, `Var[r] ≈ 1.4` puts the error law at `KL ≈ 0.7 M²` nats. So
additive accumulation in a real LLM's state space is nearly exact for a single
observation and badly wrong by `M = 4` — the update rule is not free, and the
gauge fix is the difference between "hopeless" and "borderline".

**Controls, which matter more than the headline.** Holding the state
distribution fixed and swapping in moment-matched random readouts reproduces the
affineness *as well or better* (GPT-2: Gaussian-matched 0.9999, column-shuffled
0.99998). So near-affine `log Z` is **not a signature of training the readout**.
An isotropic readout destroys it (GPT-2 residual `Var` 193), so it is a property
of *anisotropy*. Holding the readout fixed and destroying the state
correlations, however, collapses it for the two modern models (SmolLM2
0.968 → 0.457, Qwen 0.838 → 0.225) but not for GPT-2 — so in modern models the
near-affineness does live in the trained state manifold, just not in the
readout.

### The sharpest thing in this document: the energy score is gauge-dependent

Energy-based OOD detection (Liu et al., arXiv:2010.03759, thousands of
citations) scores an input by `E(h) = -logsumexp(logits) = -log Z(h)`. But
`log Z` is exactly the object the gauge moves. Applying `W ← W − 1cᵀ` leaves
every predicted probability, the argmax, the loss and the accuracy
**bit-identical**, while shifting the score by `cᵀh` — which varies across
inputs. So the detector's score is not a function of the model.

Measured on GPT-2, ID = Pile text, OOD = uniformly random token sequences, LDA
gauge direction fit on a train half and scored on a held-out half
(`experiments/logz_geometry/gauge_ood.py`):

| gauge | energy-score AUROC | max &#124;Δsoftmax&#124; |
|---|---|---|
| baseline (`c = 0`) | 0.153 | 0 |
| **flat gauge (LS slope, uses no OOD labels)** | **0.439** | 3.4e-05 |
| chosen, α = −30 | 0.054 | 4.0e-05 |
| chosen, α = +30 | 0.784 | 3.8e-05 |

The AUROC ranges over **0.054 – 0.784** across gauges of the same model, with
the model bit-identical throughout (`3e-5` is float32 noise).

The load-bearing row is the second one, because it uses **no OOD labels**: the
canonical flat gauge — the coordinates in which the additive update is exactly
Bayesian — moves the AUROC from 0.153 to 0.439, i.e. to essentially chance
(0.5). Energy-based OOD detection works only in the accidental coordinates that
training happened to land in; in the canonical gauge the signal is gone, and no
prediction has changed.

Two honest limits. The α-sweep uses OOD labels, so it demonstrates
non-identifiability, not a deployable improvement. And this is one model with a
crude OOD channel — random tokens — at token level; the claim deserves a proper
vision benchmark (CIFAR-10 vs SVHN/LSUN) where the published AUROCs live before
it is asserted against that literature. Note also that GPT-2's baseline is
*anti*-predictive in the conventional orientation (random tokens give *higher*
`log Z` than natural text); flipping the sign gives 0.847.

### The gauge is not any of the named embedding-post-processing directions

Worth stating because it was the obvious guess and it is wrong. Stein's identity
gives `E[∇ log Z] = A p̄` — the mean unembedding row weighted by the model's own
predicted marginal, which is exactly "Zipfian Whitening" (Yokoi et al., NeurIPS
2024). That makes it look like the gauge must be a known object. It is not.
Residual `Var[log Z]` on held-out GPT-2 states after applying each candidate
(`experiments/logz_geometry/which_gauge.py`, reproduced independently by two
implementations):

| gauge direction | residual `Var[log Z]` |
|---|---|
| none (`c = 0`) | 4450.8 |
| uniform mean row — "all-but-the-top" (Mu & Viswanath 2018) | 11.78 |
| marginal-weighted row — Stein / Zipfian Whitening | 8.44 |
| **least-squares slope (what the theory asks for)** | **1.06** |

The least-squares gauge leaves **8× less** residual than the Stein direction and
**11× less** than all-but-the-top. So the gauge fix is a genuinely new direction,
not a Bayesian re-derivation of a known embedding trick — which is a cleaner
claim than the rediscovery would have been.

*Caveat I will not paper over:* the state covariance is ill-conditioned, so the
gauge vector's **norm and its cosines to other directions are not identified**
(two implementations agreed on every residual variance to three digits but
disagreed on `‖c‖` by four orders of magnitude). Only its action on the state
manifold — the residual column above — is stable, and only that is claimed here.

One theoretical note this produced (`NOTE_normalization_is_a_gauge_fix.md`):
every modern readout applies LayerNorm/RMSNorm first, so the state is on a
sphere, where the *isotropic* part of the second-order term is constant. The
tight-frame condition is therefore stronger than necessary — what is required is
isotropy **after projecting out the trace**, which is a much weaker and more
plausibly-satisfied condition.

---

## 4. What I would do next, in order

1. **Take the gauge/OOD result to a real benchmark.** A ResNet or ViT on
   CIFAR-10 with SVHN/LSUN/Places as OOD, which is where the energy score's
   published AUROCs live. If the flat gauge collapses a *published* number while
   leaving accuracy bit-identical, that is a short, self-contained paper aimed
   at a different community than the thesis, and it is the single most striking
   thing this pass produced. Half a day of cluster time.
2. **Write the conditionality chapter** around §1. It needs no new compute: the
   COCO two-channel experiment is already run, and the DART table in §2 is its
   modern hook. Read Dawid & Dickey (1977) in full first — six pages — because
   if their no-modification condition already covers likelihood-derived weights,
   the novelty narrows to the specific weight `w = Z`.
3. **Fold the `log Z` measurement in as the empirical section.** It is the first
   time the criterion has met a real network, it has proper controls, and it
   ends with a number (1–2 nats²) rather than a vibe.
4. **Drop** the delta-rule direction entirely, and drop logit-arithmetic before
   writing a word about it.

## 5. What is still open

- `Var[log Z]` measured over a *corpus* of hidden states is a wider distribution
  than the belief the error law integrates against, so 1–2 nats² is an upper
  bound on the operationally relevant quantity. Sharpening it needs a task where
  a belief actually exists.
- The gauge/OOD result is one model, one crude OOD channel, token level. It is a
  demonstration, not a benchmark result. See item 1.
- Three of five research fronts stopped early on an account spend limit: the
  guidance front was mid-pivot, the gated front had written its document but not
  its summary, and the `log Z` literature front had not reported. Two loose ends
  from their scratch files are worth chasing and are **unverified**:
  - **Token-level CFG is a Heisenberg update at the sequence level** (below).
  - **Token-level CFG is a Heisenberg update at the sequence level.** Within a
    step, adding logits and softmaxing is exact. But CFG renormalises at *every*
    step, so `P̃(y) = ∏_t [p_u^{1-w} p_c^w / Z_t]`, whereas the sequence-level
    geometric mixture has one global `Z`. Token-level CFG therefore equals the
    intended target tilted by `∏_t Z_t`, with sequence length in the role of
    `M`. This survives my correction in §0 and was not chased.

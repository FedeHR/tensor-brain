# Where the cancellation identity bites in 2026 ML pipelines

Scoping pass, 2026-08-18. Target: claim 3 of `BRIEF.md` — if items are recorded
through a gate that opens with probability proportional to the total score mass
`Z(x)`, naive additive accumulation of log-evidence *is* the exact posterior, and
applying a selection/propensity correction on top makes it strictly worse.

Sources read first: `docs/BRIEF.md`, and
`heisenberg_experiment_design.md` §1.1–1.3 (the `tau` family and the general gate
law), §3.3–3.4 (six opportunities, Trap 1), §6.1 (the three-regime taxonomy),
§6.3–6.5 (candidates ⑦–⑪). Nothing below restates those; every candidate here is
new, and where a new candidate touches an old one (⑨ weak supervision, ⑪
threshold retrieval) the relationship is stated explicitly.

---

## 0. The filters, restated so they can be applied strictly

**(F1) Variable count.** The *number* of recorded items must vary with score
mass. Fixed top-k, fixed group size `G`, fixed sample budget `n`, and
fixed-retained-mass (top-p) rules all condition on a total and reinstate
`-N log Z(x)` (§6.1 regime (c)).

**(F2) Latent-dependent gate.** The gate must depend on `x`, not on observables.
If it depends only on observables, IPS/propensity weighting is consistent
already and there is no contribution.

Two more that §1.2–1.3 make necessary and that I applied throughout:

**(F3) Linear-in-`Z` gate.** Cancellation needs `pi(x) = g(Z)/C` with `g` linear.
Saturating gates (`Z/(1+Z)`, "at least one of `n` accepted") cancel only in the
`Z << 1` regime. This is *good news* for the top candidates, because frontier
verifier pass rates are small — the union gate `1-(1-p)^n ≈ n·p` is linear
exactly where modern reasoning pipelines operate.

**(F4) Exposure unobserved.** §6.1(b): if you know how many opportunities passed,
the silent ones are data and carry a `-(T/C)Z(x)` compensator. Regime (a) — the
exact-cancellation regime — requires that the failures were *never recorded*.
This is the filter that separates "you ran the sampler yourself" from "you
downloaded the corpus", and it is the crux of candidate ①.

---

## 1. Ranked shortlist

### ① Verifier-gated rollout curation for reasoning SFT — and DART-Math's rebalancing as the double correction

**Rank 1. Highest profile, cleanest paired public data, sign-flipped against a
NeurIPS'24 method that the whole field copies.**

**The gate.** RFT / STaR / rejection-sampling SFT: draw rollouts for query `i`,
keep the ones the verifier accepts. Per accepted rollout the joint density is

```
P(accept and record trace k | x) = [Z_+(x)/C] · [exp(s_k(x))/Z_+(x)] = exp(s_k(x))/C
```

where `Z_+(x)` is the policy's mass on verifier-accepted traces. The acceptance
probability cancels against the conditional-on-acceptance emission. This is claim
3 exactly, with `Z_+` in the role of `Z`. At the query level, "this query
contributes at all" is the union gate `1-(1-p_i)^n ≈ n·p_i`, linear in the score
mass whenever pass rates are low (F3 ✓ — and low pass rates are the regime
everyone trains in).

- **F1 ✓** The number of accepted traces per query is `c_i ~ Binomial(n, p_i)`;
  it varies over roughly two orders of magnitude across a math corpus.
- **F2 ✓** `p_i` is the model's latent competence on that query. It is not an
  observable feature of the problem text — this is precisely what
  *Hard or Just Unreached? Diagnosing the Sampling Blind Spot in Math-Reasoning
  Difficulty Estimation* (arXiv:2606.19636, Jun 2026) argues, that pass-rate
  difficulty is not recoverable from surface features.
- **F3 ✓** in the low-pass-rate regime; degrades toward `Z/(1+Z)` saturation on
  easy corpora (GSM8K), which is itself a testable moderator.
- **F4 — the crux.** If *you* ran the sampler you know `n`, so the failures are
  data (regime (b)) and the additive rule is not exact. If you *consume a
  released corpus* — OpenR1, OpenThoughts3, DART-Math, Nemotron SFT sets — the
  failures were never released and you are in regime (a), where the additive rule
  is exact. **Almost the entire open post-training ecosystem is in regime (a) and
  does not know it.**

**Incumbent, and why it is a double correction.** *DART-Math: Difficulty-Aware
Rejection Tuning for Mathematical Problem-Solving* (arXiv:2407.13690, Tong,
Zhang, Wang, Wu, He; v1 Jun 2024, NeurIPS'24) diagnoses vanilla rejection tuning
(VRT) as "severely biased towards easy queries" *because* the number of retained
samples tracks the pass rate, and fixes it two ways: DARS-Uniform (equal count
per query) and DARS-Prop2Diff (count increasing in difficulty). In the `tau`
family of §1.1:

| corpus | recorded count rule | implied `tau` | regime |
|---|---|---|---|
| VRT (vanilla rejection tuning) | `c_i ∝ p_i` | `tau ≈ 1` | (a) — additive exact |
| DART-Math-Uniform | `c_i = const` | `tau ≈ 0` | (c) — normalizer reinstated |
| DART-Math-Hard (Prop2Diff) | `c_i` decreasing in `p_i` | `tau < 0` | anti-gate, Trap 3 |

DART-Uniform is *literally* Trap 1 applied deliberately, and DART-Hard is
*literally* Trap 3 (§6.4) applied deliberately. The theory says the "bias" being
corrected is not a bias: under the verifier gate the count is the sufficient
statistic, and flattening it destroys the only thing that made the additive
accumulation exact. Related incumbents in the same direction: GRPO's per-group
`std` division (a per-prompt normalizer; *Understanding R1-Zero-Like Training*,
arXiv:2503.20783, already removed it and got a win — a *confirming* datapoint,
not a novel prediction), DAPO's dynamic sampling (fixes the effective count),
and the RFT convention of deduplicating to at most `k` distinct correct paths per
question (Yuan et al., arXiv:2308.01825).

**The sign-flipped prediction.** DART reports accuracy gains, and the theory does
**not** dispute them — §0/§1.4 of the design doc already establish that fidelity
and decision quality dissociate. The prediction is on the *belief* metrics, and
it is the opposite of the incumbent's framing:

> A model trained on VRT is better calibrated about its own competence than a
> model trained on DART-Uniform or DART-Hard from the same generator and the same
> queries, and the calibration gap grows as `(M^2/2)·Var[log p_i]` across the
> corpus. Rebalancing buys accuracy on hard items by paying for it in the
> model's posterior over its own success probability.

Concretely: bin held-out MATH/GSM8K queries by reference pass rate; measure the
slope of (model's empirical pass@1) against (reference pass rate). Prediction:
VRT ≈ 1, DART-Uniform flattened, DART-Hard flattened further or inverted; and ECE
of the model's own confidence rises monotonically as `tau` falls.

**Decisive cheap experiment (three stages, increasing cost).**

*Stage 0 — the `tau` instrument, laptop, hours, no GPU.* §1.1's Poisson
regression, run on public corpora. Per query, regress recorded trace count on an
independently estimated `log Z`, with `log(attempts)` as **offset** — which is
what removes the confound §1.1 warns about (`log Z` correlates with how much
there was to record):

```
c_i ~ Poisson( exp( alpha + tau · log p̂_i + log n_i ) )
```

`c_i` and `n_i` come from the corpus; `p̂_i` must come from an *independent*
channel or the regression is tautological. Public independent channels:
`SynthLabsAI/Big-Math-RL-Verified` (arXiv:2502.17387) publishes
`llama8b_solve_rate` over 64 rollouts for ~250k problems, and
`ScalingIntelligence/monkey_business` (arXiv:2407.21787, MIT) publishes **10,000
labelled samples per problem** for 127 GSM8K and 128 MATH test problems — an
essentially noiseless `p_i`. Corpora to score: `open-r1/OpenR1-Math-220k`
config **`all`** (Apache 2.0; fields `generations`, `correctness_math_verify`,
`correctness_llama`, `correctness_count`, 2–4 traces per problem, 225k rows — it
ships *both* channels, exhaustive and gated, on identical problems, which is the
COCO paired design handed over ready-made), `hkust-nlp/dart-math-uniform`
(~591k), `hkust-nlp/dart-math-hard` (~585k). Predicted `taû`: OpenR1-`all`
gated subset ≈ 1, DART-Uniform ≈ 0, DART-Hard < 0. **This is the single highest
value-per-hour item in this document**: it demonstrates on three public,
famous corpora that `tau` is a measurable pipeline property, and that the field
has been building `tau = 0` and `tau < 0` corpora on purpose.

*Stage 1 — inference-only, one GPU.* DART released checkpoints per base model and
per variant plus VRT baselines (`hkust-nlp/dart-math-*`, github
`hkust-nlp/dart-math`). No training: evaluate calibration-of-competence as above
on the released checkpoints. Confounds are weak because the generator, the query
set and the base models are shared across variants.

*Stage 2 — small-cluster SFT.* Build three corpora from a *single* sampling run
on a 7B model over MATH+GSM8K (`n = 64`): keep-all-accepted (`tau=1`), subsample
to constant count (`tau=0`), subsample proportional-to-difficulty (`tau<0`),
matched on total token budget. One SFT run each. This is the within-run version
that removes the last confound, and it is the arm to run on the cluster.

*Falsifier.* If VRT and DART-Uniform have indistinguishable competence
calibration at matched token budget, the identity has no purchase on SFT
gradients and candidate ① dies.

**Regime (b) control, which protects the claim.** Process-reward-model labelling
(Math-Shepherd/OmegaPRM-style: label a step by whether any of `N` fixed
continuations succeeds) knows `N`, so it is regime (b), and the soft label `k/N`
*is* the correct object there. Reporting that contrast — "normalize when you
instrumented the exposure, don't when you didn't" — is what makes the claim look
like a theorem rather than a preference.

---

### ② Verifier/confidence-gated test-time aggregation — best-of-n and self-consistency

**Rank 2. The same identity one pipeline stage later, at a tenth the cost, with a
2025–2026 incumbent that normalizes explicitly.**

**The gate.** Sample `n` candidates, keep those a verifier accepts (or those
whose confidence clears a bar), aggregate the survivors into an answer
distribution. Accepted count per problem varies with the model's mass on correct
answers (F1 ✓), and that mass is the latent competence (F2 ✓). Union-gate
linearity holds on hard benchmarks (F3 ✓). Exposure: if you aggregate only over
survivors and discard the count of attempts, regime (a) (F4 ✓); if you keep `n`
you are in regime (b) and the compensator is available — which the theory says
should buy little when pass rates are low, and which is exactly what
*A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce*
(arXiv:2504.11343, Xiong, Yao, Xu, Pang, Wang, Sahoo, Li, Jiang, Zhang, Xiong,
Dong; Apr 2025, rev. Jun 2025) found empirically: RAFT — positives only, no
compensator — is competitive with GRPO and PPO, and GRPO's advantage comes mostly
from *discarding all-wrong prompts*, not from the negative gradient.

**Incumbent that corrects the normalizer.** Confidence-Informed Self-Consistency
(CISC) and DINCO-style aggregators explicitly **normalize by the total confidence
over candidate answers** to "relieve saturation and improve calibration". That is
the `-N log Z` correction applied to a channel where `Z` already cancelled.
Plain unweighted majority vote is the degenerate fixed-count case (§6.3 ⑨ makes
the same point for crowd labels, and the in-house `experiments/crowd/` result —
additive matching majority-vote accuracy at ~10x better ECE — is the same
phenomenon at a different scale).

**Sign-flipped prediction.** On a verifier- or threshold-gated survivor set with
*variable* count, the **unnormalized** sum of accepted votes/log-confidences is
better calibrated than the normalized one, and the gap grows with
`Var[log p_i]`. Under an artificially fixed survivor count (keep exactly `m`
accepted), the ordering flips and normalization wins. Same data, same model, one
line of difference — the paired within-dataset contrast pattern that made COCO
work.

**Decisive cheap experiment: laptop, CPU, hours, no model inference at all.**
`monkey_business` gives 10,000 labelled samples per problem. For each problem,
simulate the two arms from the same sample pool: (i) draw a Poisson-ish budget
and keep all accepted → variable count; (ii) keep exactly `m` accepted →
fixed count. Aggregate additively vs normalized; score NLL/ECE of the answer
posterior against ground truth. `M` sweeps for free. The `M^2 Var[log Z]` error
law is directly measurable because `Var[log p_i]` is known exactly from the pool.
This is the cheapest falsification available anywhere in this document and it
should be run first regardless of what else survives.

**Caveat to state.** *When Self-Consistency Backfires* (arXiv:2608.11403, Aug
2026) reports that on hard science problems agreement does not track correctness
for small models. That is a statement about the *readout* being miscalibrated,
not about the aggregation rule; but it means the experiment must report
calibration conditional on difficulty bin, not marginally.

---

### ③ Adaptive retrieval: the confidence gate and the uncertainty gate have opposite signs

**Rank 3. Turns §6.3 ⑪ from a defensive measurement into an offensive,
sign-flipped claim, at the same cost.**

§3.4's Trap 1 says fixed top-k is the counter-example. The new observation is
that the two dominant *adaptive* retrieval policies of 2025–2026 sit on
**opposite sides of `tau = 0`**, and nobody has noticed that they are the same
knob:

| policy | count varies with | `tau` | additive fusion is |
|---|---|---|---|
| score-threshold retrieval (keep every passage above `theta`) | total score mass | `≈ 1` | **exact** |
| fixed top-k | nothing | `0` | wrong by `M log Z` |
| top-p / cumulative-mass retrieval | nothing (mass fixed by construction) | `0` | wrong |
| retrieve-when-uncertain (Self-RAG, Adaptive-RAG, FLARE, agentic "search again?") | **inverse** score mass | `< 0` | **worse than ungated** |

F1 ✓ for rows 1 and 4, ✗ for 2 and 3. F2: partial — the gate reads the model's
own confidence, which is a function of the latent state but computable at run
time; this is the weakest F2 in the shortlist and must be conceded. F3 ✓ for a
hard threshold; ✗ for softmax-normalized similarity, where `Z ≡ 1` by
construction, so the experiment **must** use unnormalized similarity scores.

**Sign-flipped prediction.** Confidence-*triggered* retrieval (retrieve more when
the model already has mass on the topic) makes cheap additive evidence fusion
*exact*, while uncertainty-*triggered* retrieval — the near-universal design,
and the one *When Should Active RAG Retrieve? A Budget-Aware Evaluation of
Utility, Calibration, and Cost* (arXiv:2607.24010, Jul 2026) evaluates — puts the
pipeline at `tau < 0`, where additive fusion is *worse than it would have been
with no gate at all*. So the incumbent's trigger polarity is exactly backwards
for anyone fusing evidence additively, and the recommended design is the one the
field calls wasteful.

**Decisive cheap experiment.** One retrieval benchmark with a small enumerable
topic latent (the design-doc pattern: `n ≤ 12` topics, exact posterior by
enumeration). Same corpus, same encoder, same queries; four retrieval policies as
above; fuse retrieved evidence additively; score `KL(exact || additive)` and
downstream answer NLL. Predicted ordering, threshold < top-k ≈ top-p <
uncertainty-triggered, with the gap between rows 1 and 4 the headline. Cost: CPU
plus one embedding pass. Fits the "Mac gets smoke tests, cluster gets training"
split with no training at all.

---

### ④ Threshold-gated sparse computation: the partial-softmax denominator

**Rank 4. Highest profile of all (sparse attention is in every 2026 serving
stack), cheapest decisive test after ②, but the weakest theoretical mapping —
report it as a diagnostic note, not a chapter.**

Threshold sparse attention selects a variable number of KV units per head per
query and then **renormalizes the softmax over the selected set** — the partial
denominator. That renormalization is a normalizer correction applied to a
score-gated channel. The taxonomy splits the field cleanly and the split is not
one the field draws:

- **absolute-threshold selection** (keep every unit with score `> theta`): the
  retained mass fraction varies with the query's total score mass. F1 ✓.
- **cumulative-mass / top-p selection** (FlexPrefill, arXiv:2502.20766, ICLR'25
  oral; Twilight; Tactic; SampleAttention — surveyed in *The Sparse Frontier*,
  arXiv:2504.17768, Nawrot, Li, Huang, Ruder, Marchisio, Ponti, Apr 2025, final
  Jun 2026, code public): the retained *mass* is fixed at `gamma` by
  construction. **This is F1 failure in disguise** — it is conditioning on the
  total, regime (c) — and it is what every adaptive method in the taxonomy does.
- fixed-budget top-k: F1 ✗, and renormalization is correct there.

**F2 fails literally**: attention scores are observable to the mechanism. But
F2's *purpose* — "otherwise IPS already handles it, so there is no contribution"
— is satisfied, because no propensity machinery exists in this literature; the
incumbent just divides by the partial denominator and hopes. State this
explicitly rather than pretending F2 passes.

**Sign-flipped prediction.** For absolute-threshold selection the correct
combination is the **unnormalized** accumulation `sum_{j in S} exp(s_j) v_j`
divided by an estimate of the *full* `Z`, not by `Z_S`; renormalizing over `S`
over-corrects, and the damage is largest on exactly the tokens where the retained
mass fraction deviates most from its mean (measurable per token, so the
prediction is per-token, not aggregate). Corollary the field would find
surprising: cumulative-mass (top-p) selection is *safe* precisely because it
holds the retained mass constant, i.e. its apparent adaptivity is what makes it
theoretically boring.

**Decisive cheap experiment.** One 1B model, WikiText/LongBench, 1 GPU-hour: for
each layer/head, absolute-threshold selection at matched average sparsity,
scored with (a) partial-denominator renormalization, (b) unnormalized with a
cheap full-`Z` estimate. Report perplexity delta *conditioned on retained-mass
fraction*. Falsifier: if the delta is flat in retained-mass fraction, the
mechanism is not the one claimed. The same experiment structure transfers
verbatim to variable-`k` MoE routers with sigmoid (unnormalized) gates, where
DeepSeek-V3-style renormalization of selected affinities is the same operation —
but that arm needs training and should be skipped.

*(Citation health: I fetched a 2026 confidence-adaptive MoE-LoRA router,
arXiv:2607.26052, whose author list did not look credible on inspection. Do not
cite it without checking. FlexPrefill, The Sparse Frontier, DART-Math, Large
Language Monkeys, Big-Math, Dr. GRPO and the Minimalist/RAFT paper I consider
solid; everything with a 26xx ID that I reached through a single summarising
fetch is flagged in §5.)*

---

### ⑤ Abstention-gated preference and judge data — only one sub-case survives

**Rank 5. Included because it is the family most people would propose, and
because the strict analysis kills most of it. The surviving sub-case is real but
is mechanically candidate ①.**

- **Ties/abstentions discarded before a Bradley–Terry fit (Chatbot Arena,
  RewardBench-style pipelines): FAILS.** The judge abstains when the *contrast*
  `|s_A - s_B|` is small, not when the *total mass* `exp(s_A)+exp(s_B)` is small.
  A gate on the contrast is not a gate on `Z`, so nothing cancels, and the
  tie-aware models (Rao–Kupper, Davidson, Arena's half-weight convention) are
  doing something legitimate. Kill it and say why — this is the one a reviewer
  will raise.
- **"Discard the pair if *all* candidates fail the rubric": PASSES.** With
  independent per-candidate acceptance the keep probability is
  `1 - prod_k (1 - exp(s_k)/C) ≈ Z(x)/C` for rare acceptance — the union gate,
  linear in `Z` in exactly the regime §1.3 identifies, and the QTB offset
  convention `E[Z] = K·2^-n` puts the model there by construction. F1 ✓ (number
  of surviving pairs per prompt varies), F2 ✓ (rubric pass depends on latent
  quality). Prediction: reweighting the retained pairs by an estimated
  "both-bad" discard rate — the obvious debiasing move — makes the reward model
  worse, not better.
- Programmatic weak supervision with abstaining labelling functions is the same
  structure and is already ⑨ in the design doc; nothing here supersedes it.

**Decisive cheap experiment.** Fold into ①'s stage 0: run the `tau` instrument on
a public preference corpus that logs discarded comparisons. Only worth doing if a
corpus that logs its own discards can be found; without the exhaustive channel
the regression is unidentified (§1.1's warning, §6.1's presence-only warning).

---

## 2. Killed, with reasons

| candidate | verdict | why |
|---|---|---|
| **Speculative decoding acceptance** | **Dead on both filters and on motive.** | The gate `min(1, p/q)` is a function of two *observable* distributions (F2 ✗). Worse, the incumbent is already exact: the residual-resampling step makes the accepted-token stream distributed exactly as the target, so the "correction" is necessary, not a double correction, and there is no downstream accumulator treating accepted tokens as unconditional. The only live variant — lossy/typical acceptance, and SpecKD-style distillation on accepted tokens only — reintroduces a gate, but it is a gate on the *teacher's* observable probability, so IPS applies. No contribution. |
| **Active learning / uncertainty-triggered acquisition, event-triggered sensing, wake-word channels** | **Dead — anti-gate.** | This is Trap 3 of §6.4, and it generalizes: uncertainty sampling selects *low*-`Z` points, so `tau < 0` and the residual correction `(1-tau)·M·log Z` is *larger* than in the ungated case. The rule is worse on actively-acquired pools than on random ones. Keep the falsifiable by-product (a model fitted on uncertainty-sampled data should show higher `Var[log Z]` than one fitted on a random sample of equal size) as a one-line remark, not an experiment. |
| **Classifier-threshold pretraining curation (DCLM, FineWeb-Edu, perplexity filtering)** | **Dead on F2, and the algebra says so twice.** | The gate is a deterministic function of the *observed document* (F2 ✗), so it is covariate shift, and DSIR-style importance resampling is consistent. Structurally: per-item Bernoulli thinning with rate `h(k)` gives `p(record k | x) ∝ h(k) exp(s_k(x))`, so the filter shifts only the intercepts `a_{0,k}` — absorbable into the gauge, invisible to the update rule. The theory's contribution here is a *null*: "quality filtering needs no belief-level correction because it only moves the intercepts", which is true but unexciting and untestable cheaply. |
| **Fixed top-k retrieval; fixed-`G` GRPO groups; top-p / cumulative-mass selection (attention or retrieval); DAPO dynamic sampling to a fixed effective batch** | **Dead on F1.** | All condition on a total — count or mass. Regime (c), normalizer reinstated. Listing top-p here is the non-obvious part and is worth stating in the thesis: *fixing the retained mass is the same failure as fixing the retained count*. |
| **Ties-discarded Bradley–Terry / Arena leaderboards** | **Dead on gate shape.** | Gate on the contrast, not on `Z`. See ⑤. |
| **PRM step labelling by `N` fixed Monte-Carlo continuations** | **Not dead, but regime (b), not (a).** | Exposure is instrumented, so the compensator `-(T/C)Z(x)` is available and correct; soft labels `k/N` are right. Keep as ①'s control, not as a candidate. |

---

## 3. The `tau` instrument: yes, it runs cheaply, and here is the exact recipe

§1.1 fits `tau` by Poisson regression of recorded counts on `log Z`, and reports
it unbiased at `n = 2000` (`0.996 ± 0.043` at `tau = 1`) — a few thousand
instances suffice. §1.1's own warning is that a naive regression is confounded,
because `log Z` is large exactly when there was more to record. Two things fix
that here, and both are available in public data:

1. **An exposure offset.** Fit
   `c_i ~ Poisson(exp(alpha + tau·log p̂_i + log n_i))`. The offset `log n_i`
   absorbs the "more opportunities" confound directly. This is available wherever
   the corpus reports attempts — `OpenR1-Math-220k/all` reports
   `len(generations)` alongside `correctness_count`.
2. **An independent `Z` channel.** `p̂_i` must not come from the same counts.
   `Big-Math-RL-Verified`'s `llama8b_solve_rate` (64 rollouts, ~250k problems,
   arXiv:2502.17387) and `ScalingIntelligence/monkey_business` (10k labelled
   samples/problem, arXiv:2407.21787, MIT) both provide it, and both overlap the
   NuminaMath problem pool that OpenR1 and DART-Math draw from.

**Cost:** one afternoon, CPU only, `statsmodels.GLM(family=Poisson)` on a few
tens of thousands of rows. **Deliverable:** a table of measured `tau` for named,
famous corpora. Predicted values — pre-register these before looking:

| corpus | predicted `taû` | why |
|---|---|---|
| `OpenR1-Math-220k` (`all`, gated subset) | `0.8 – 1.0` | vanilla verifier gate |
| `dart-math-uniform` | `≈ 0` | count equalized by construction |
| `dart-math-hard` (Prop2Diff) | `< 0` | count anti-correlated with pass rate |
| `monkey_business`, threshold-gated resample | `1.00 ± 0.02` | positive control, gate imposed by us |
| `monkey_business`, fixed-`m` resample | `0.00 ± 0.02` | negative control |

The two `monkey_business` controls matter as much as the real corpora: they show
the instrument reads the gate and not the data, on the same items, which is the
paired within-dataset design pattern that the COCO experiment established.

---

## 4. What to run, in order

1. **②'s simulation on `monkey_business`** — hours, laptop, no GPU, no training.
   It is a complete falsification of the identity on real model outputs if it
   fails, and it produces the `M^2 Var[log Z]` error-law curve on real data.
2. **①'s stage 0 `tau` table** — one afternoon, CPU. Produces the headline
   artifact: measured `tau` for OpenR1, DART-Uniform, DART-Hard, with controls.
3. **①'s stage 1** — inference-only on released DART/VRT checkpoints, one GPU.
   First real sign-flip test against a published method.
4. **③** — CPU plus one embedding pass; converts §3.4's Trap 1 assertion into a
   measurement *and* adds the `tau < 0` anti-gate arm that makes it offensive.
5. **④** — 1 GPU-hour, highest profile, weakest mapping. Do it only if 1–3 land,
   and scope it as a diagnostic note.

Do not run ⑤ standalone; fold it into 2.

## 5. Verification status

Fetched and considered reliable: arXiv:2407.13690 (DART-Math), arXiv:2504.11343
(Minimalist/RAFT), arXiv:2504.17768 (Sparse Frontier), arXiv:2502.20766
(FlexPrefill, ICLR'25 oral), arXiv:2407.21787 + `ScalingIntelligence/monkey_business`,
arXiv:2502.17387 (Big-Math), arXiv:2503.20783 (Dr. GRPO), arXiv:2308.01825 (RFT),
`open-r1/OpenR1-Math-220k` field schema (fetched from the dataset card).

Reached through a single summarising fetch or search snippet and **not**
independently verified — check before citing: arXiv:2607.00152 (group-std
identity), arXiv:2607.24010 (active-RAG budget evaluation), arXiv:2605.05112
(rollout pass-rate control), arXiv:2606.19636 (difficulty estimation),
arXiv:2608.11403 (self-consistency backfires), arXiv:2607.26052 (**author list
looks fabricated — treat as nonexistent until verified**), CISC/DINCO (reached
only via secondary summaries; find the primary papers before citing).

# The Heisenberg update with a learned index layer, on MS-COCO

The first test of the additive update where `A` is **fitted to real data** rather
than sampled, and where the question is **what the belief is good for**, not only
how close it sits to the exact posterior.

```
latent   x in {0,1}^12   which COCO supercategories the image contains
index    k in 1..K       content words drawn from the five human captions
update   q <- q + a_k    one addition per revealed word
```

`x` comes from the *instance* annotations and `k` from the *caption*
annotations — two independent passes over the same image — so `x` is a genuine
ground truth rather than a relabelling of the evidence. A caption word is not a
deterministic function of the supercategory vector, which is what keeps `A`
non-degenerate. `n = 12` puts exact enumeration at 4096 states, so exact Bayes is
available as a reference at every step.

## Running it

```sh
# once: annotations only, no images, no GPU (~241 MB)
mkdir -p data/coco && cd data/coco
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip && cd -

# build the corpus (~8 s)
PYTHONPATH=".:src" python -c "
from pathlib import Path
from experiments.coco_heisenberg import data as D
c = D.build_corpus(Path('data/coco/annotations'), split='train2017', vocabulary_size=1000)
D.save(c, Path('data/coco/corpus_train2017_k1000.npz'))"

# fit and evaluate (~90 s on a laptop)
TB_BAYES_ROOT=../tensor-brain-bayes-approximation PYTHONPATH=".:src" \
  python -m experiments.coco_heisenberg.run_experiment \
    --corpus data/coco/corpus_train2017_k1000.npz --symbol-counts 1 2 4 6 8
```

`TB_BAYES_ROOT` is needed only for `--cross-check`, which verifies this module's
fast path against the reference rules in the `bayes-approximation` worktree. Both
worktrees ship a top-level `experiments` package, so `load_reference` binds the
analysis package under its own name rather than relying on `PYTHONPATH` order.

## Layout

| file | what it does |
|---|---|
| `data.py` | builds `(presence, symbols)` records from the two annotation channels; dependency-free tokenizer |
| `model.py` | maximum-likelihood fit of `(A, a0)` — softmax regression of the named word on the presence vector |
| `evaluation.py` | the update rules, downstream and fidelity metrics, paired bootstrap |
| `run_experiment.py` | CLI; sweeps `M` and writes `results.json` |

## Why the fit is cheap

`x` takes only `2^12` values, so the log likelihood depends on the corpus solely
through a `[4096, K]` table of how often symbol `k` was named for pattern `x`.
`model.sufficient_statistics` builds it, which makes the fit exact and roughly
two orders of magnitude cheaper than iterating over all ~1.1 M `(x, k)` pairs —
14 s rather than many minutes, entirely on CPU.

## What is measured

**Downstream, reported first.** NLL, accuracy, macro-F1, mean average precision,
exact-set accuracy and ECE of the belief against the true supercategory set.

**Fidelity, reported second.** Joint and marginal KL to the exact posterior under
the *same* learned model, which isolates the update rule from model quality.

**Paired contrasts.** Every rule sees identical evidence on identical images in
identical order, so differences are bootstrapped per image. The effect sizes here
are a few hundredths of a nat against ~3 nats of NLL, so unpaired means are not
enough to separate the rules.

## Rules

| rule | update | cost |
|---|---|---|
| `prior` | ignores the evidence | — |
| `heisenberg` | `q <- q + a_k` | `O(n)` |
| `heisenberg-gauge` | `q <- q + a_k - c`, `c` the least-squares slope of `log Z` | `O(n)` |
| `heisenberg-pe` | `q <- q + A(e_k - p)` | `O(nK)` |
| `adf` | exact update, then project back to a product | `O(2^n K)` |
| `exact` | exact posterior, factorized prior | `O(2^n K)` |
| `exact-empirical-prior` | exact posterior, full empirical joint prior | `O(2^n K)` |

`exact-empirical-prior` is deliberately *not* achievable by any agent whose state
is a product of Bernoullis. It is included as the ceiling that separates "the
update rule is wrong" from "the factorized prior is wrong", which the scoping
work showed are confounded and must be reported apart.

## The takeaway

Run: 117,266 images, 93,813 train / 23,453 held out, `n = 12`, `K = 1000`, 6,000
evaluation images per point, `A` fitted in 14 s on CPU. Held-out symbol NLL
**5.639** against 6.908 for a uniform vocabulary, so the index layer carries
1.27 nats about the named word — it is not collapsing to an indicator matrix.
`Var[log Z] = 0.0139`, affine fraction **0.690**. Every rule agrees with the
reference implementation to ≤ 2.2e-15.

**In one sentence: with a genuinely learned index layer, the Heisenberg update is
an excellent approximation, and — up to a measurable amount of evidence — a
*better decision rule* than the exact Bayesian posterior it approximates.**

Four findings, in the order they matter.

### 1. Fidelity and downstream quality disagree, and the crossover is locatable

![dissociation](../../output/coco_heisenberg/figures/01_dissociation.png)

At `M = 4` the Heisenberg belief is the **furthest of the four rules** from the
exact posterior (0.078 nats of marginal KL, against ADF's 0.002) and
simultaneously **predicts the true supercategories better than exact Bayes
does** (−0.025 nats of NLL, 95% CI [−0.033, −0.017]). By `M = 6` that reverses,
and at `M = 8` exact Bayes is ahead by 0.158 nats.

![contrasts](../../output/coco_heisenberg/figures/02_contrasts.png)

So the honest answer to "is the additive rule good enough?" is **not a single
number but a budget**: it is better than exact inference while fewer than about
five symbols have been absorbed, and worse after. The quadratic error law is what
sets the boundary, which makes the boundary predictable rather than empirical.

The likely mechanism — and it should be presented as a hypothesis, because this
experiment does not isolate it — is that two misspecifications partly cancel. The
factorized prior misses the strong positive correlations between supercategories
(`kitchen`+`food`+`appliance`), which makes exact Bayes under that prior
*under*-confident about co-occurring categories; dropping `log Z` makes the
additive rule *over*-confident. The `exact-empirical-prior` row supports this:
given the correct joint prior, NLL falls sharply at every `M`.

**Consequence for the thesis:** posterior fidelity is a diagnostic, not the
objective. A chapter that reported only KL would have concluded the additive rule
is uniformly worse, which is false for the decision the state actually feeds.

### 2. The gauge fix is free and wins everywhere

`q ← q + a_k − c`, with `c` the least-squares slope of `log Z` under the prior,
is `O(n)`, exactly order-invariant, and a **re-normalization of trained weights**
rather than a change to the inference loop. It improves downstream NLL at every
`M` — −0.007, −0.024, −0.083, −0.154, **−0.235** nats — on 68–76% of individual
images, and cuts ECE at `M = 8` from 0.060 to **0.021**.

Chapter 4 derives this correction (around line 1054) and never implements it.
This is its first measurement, and it is the clearest practical recommendation
the experiment produces: **always gauge-fix a trained index layer.**

### 3. The `M²` error law survives the move to a learned layer

![error law](../../output/coco_heisenberg/figures/03_error_law.png)

The law was derived and tested on synthetic i.i.d. Gaussian `A`. On `A` fitted to
COCO captions the measured exponent is **1.99 over `M ≤ 4`** and 1.92 over the
full range — the shape transfers intact.

The coefficient needs care, and this is worth stating precisely because it is
easy to get wrong: the law is `KL ≈ ½M²·Var_{P(·|k)}[log Z]`, weighted by the
**posterior**. Using the prior-weighted variance over-predicts by 29% at `M = 1`
and 41% at `M = 8`; using the posterior-weighted variance the ratio is 0.75 at
`M = 1` and **0.94 at `M = 8`**. The posterior concentrates as evidence
accumulates, so the prior-weighted variance drifts further wrong exactly where
the error is largest.

### 4. Most of the remaining error is the prior, not the update rule

![error budget](../../output/coco_heisenberg/figures/04_error_budget.png)

Splitting the gap to the best belief reachable with the correct joint prior: the
update rule costs between −0.03 and +0.16 nats, the **factorized prior costs
0.24–0.38 nats at every `M`** — two to twenty times more. No agent whose state is
a product of Bernoullis can recover that, whatever its update rule.

**Consequence:** effort spent on better update rules is capped. The `O(nK)`
prediction-error correction demonstrates this directly — it improves fidelity
but *hurts* downstream NLL at `M = 2, 4, 6`, and is beaten at every `M` by the
free `O(n)` gauge fix.

### What this does not show

The measurement-process question is untouched. Only the caption channel is used
here, so the gated-versus-exhaustive contrast — the one that would test the
saliency-gating claim on real data — is still open, and with it the τ estimate.
`A` is also fitted once; the sweep varies evaluation images, not fit seeds.

## Caveats worth stating in the thesis

- The five captions per image are **not** conditionally independent given `x`;
  a caption mentioning `pizza` is likelier to also mention `plate`. Every rule
  inherits that misspecification equally, so the comparison between rules stays
  fair, but no rule's absolute NLL should be read as well specified.
- Symbols are deduplicated per image, so evidence is a *set*. This matches the
  order-invariance framing and avoids double-counting the same word across
  captions; keeping multiplicities would make the redundant-evidence regime the
  default instead.
- `person` occurs in ~54% of images and `appliance` in ~7%; per-category metrics
  are reported for that reason.
- The tokenizer is a stopword list plus a crude plural fold, not a lemmatizer.
  It is deliberate: no NLP dependency, and the mapping is inspectable.

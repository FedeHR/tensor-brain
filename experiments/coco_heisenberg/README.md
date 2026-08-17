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

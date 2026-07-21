# Tensor Brain

A minimal PyTorch implementation of the core operations from:

- [The Tensor Brain: A Unified Theory of Perception, Memory and Semantic Decoding](https://arxiv.org/abs/2109.13392)
- [Bayes or Heisenberg: Who(se) Rules?](https://arxiv.org/abs/2510.13894)

The project is intentionally small. Its first milestone contains only the shared
representation/index machinery, interchangeable evolution equations, and parameter-free
symbolic index metadata. Experiment schedules will remain explicit so that their order can be
read directly against the papers.

The project's definition of paper fidelity, modernization, and experimental extension is
documented in [Research Fidelity and Project Scope](docs/fidelity.md).

The repository and distribution are named `tensor-brain`; the import package is the shorter
`tb` and lives under `src/tb`:

```python
from tb import IndexVocabulary, OriginalTBDynamicContext, TensorBrain
```

## State and index notation

The code preserves the paper symbols:

- `q`: representation-layer preactivation, or pre-CBS;
- `gamma = q.sigmoid()`: cognitive brain state (CBS);
- `context`: optional state of the chosen evolution backend; for the original TB recurrence,
  this is the dynamic-context preactivation `h`;
- `A[:, k]`: embedding `a_k` of global symbolic index `k`;
- `a0[k]`: bias of index `k`.

The four core operations are:

```python
scores = tb.index_scores(q, indices)
q, probabilities = tb.attend(q, indices)
q, outcome, probabilities = tb.measure(q, indices)
q, outcome, probabilities = tb.measure(q, indices, selection="argmax")
q, outcome, probabilities = tb.measure(
    q, indices, selection="teacher", outcome=known_index
)
q, context = tb.evolve(q, context)
```

External perception stays outside the core. With DINO features of the same dimension as `q`,
input integration is simply visible experiment code:

```python
q = q + dino_features
```

## Measurement

For candidate index `k`, the score is

```text
a0[k] + A[:, k] @ sigmoid(q)
```

After sampling or supplying an outcome, the generalized QTB update is

```text
q <- retain_gate * q + feedback_gate * A[:, outcome]
```

The probability vector is local to the supplied candidate list. Passing `candidates=None`
explicitly means that the complete global index layer competes. For example, with candidates
`[4, 9, 12]`, probability position `1` refers to global index `9`. `measure` performs this
translation internally and returns the global index, which can be resolved through the
`IndexVocabulary` and used directly as a column of `A`.

The default `(retain_gate=1, feedback_gate=1)` is the original Tensor Brain / neural
HB-POVM update. `(0, 1)` is the neural PVM update and `(1, 0)` removes index feedback.
`selection="sample"` performs generative sampling, `selection="argmax"` implements
winner-take-all decoding, and `selection="teacher"` uses the supplied global `outcome` for
teacher-forced measurement. Gates may also be tensors or learned parameters; the experiment is
responsible for any range-constraining parameterization.

## Index initialization

`A` is initialized as a zero-mean Gaussian embedding bank:

```text
A[i, k] ~ Normal(mean=0, variance=1 / state_dim)
# equivalently: standard deviation = 1 / sqrt(state_dim)
```

Each index column therefore has expected squared norm one, regardless of the number of
indices. This is more natural for `A` than applying Kaiming initialization in its stored
`[state_dim, num_indices]` orientation: `A` is used both as the effective readout `A.T` and as
generative feedback, rather than as a one-way ReLU layer.

## Evolution implementations

`QTBEvolution` implements the feed-forward transition

```text
h = sigmoid(v0 + V @ sigmoid(q))
q_next = W @ h
```

`QTBEvolution` is the paper-facing sigmoid variant and uses Xavier initialization for its
sigmoid hidden layer and linear output. `ReLUEvolution` is an explicit experimental variant:
it keeps `gamma = sigmoid(q)` and the unrestricted pre-CBS output, but uses ReLU in the hidden
layer and Kaiming initialization for the matrix feeding that ReLU. Keeping these as named
evolution classes makes activation and initialization a controlled experimental variable
without changing the meaning of `gamma` or the direct `A` feedback path.

`OriginalTBDynamicContext` implements the original recurrence

```text
h_next = B @ sigmoid(sigmoid(h) + V @ sigmoid(q))
q_next = W @ sigmoid(h_next)
```

`VanillaRNNDynamicContext` is included only as a conventional control. It is not an exact
implementation of the original TB recurrence.

### Future evolution backends

xLSTM and Mamba2 are well-motivated experimental alternatives behind the evolution boundary:
both provide persistent state mechanisms that can be compared with the original TB dynamic
context on long concept-window sequences. They are not paper-faithful TB implementations and
should preserve the same public contract, `q_next, context_next = evolution(q, context)`, while
leaving `gamma = sigmoid(q)`, measurement, attention, and direct index feedback unchanged.
Their value is a controlled question about memory capacity and temporal credit assignment, not
an assumption that newer sequence models automatically improve Tensor Brain.

### Evolution overfitting diagnostic

`experiments/evolution_overfit.py` contains a four-example XOR problem. Direct index scoring is
linear in `sigmoid(q)` and cannot solve XOR; the evolution operator must create the nonlinear
decision boundary before the index layer decodes the label. The original recurrent, sigmoid
feed-forward, and ReLU feed-forward variants all serve as controlled overfitting baselines. This
diagnostic establishes representational and optimization viability, not generalization or a
claim that one activation is universally better.

## Bottom-up index scoring extensions

The default score path remains the paper equation

```text
scores = a0[candidates] + A[:, candidates].T @ sigmoid(q)
```

It is scientifically reasonable to add a learned bottom-up adapter between `gamma` and the
index scores, for example `scores = a0 + A.T @ phi(gamma)`, as an explicit extension. The
top-down path should remain direct: the selected index still injects `A[:, outcome]` into `q`.
This creates a deliberate asymmetry (learned processing may enrich scoring, while symbolic
feedback retains the original index embedding) and should be evaluated against the direct-score
baseline.

## Index vocabulary

`IndexVocabulary` contains names and candidate groups but no learned parameters. Its global
index `k` refers to column `A[:, k]` and bias `a0[k]` in `TensorBrain`. Save the vocabulary
next to model checkpoints because changing label order changes the meaning of the learned
columns.

For realistic datasets, numeric indices should not be maintained as handwritten constants.
The dataset adapter should construct the vocabulary from a versioned ontology manifest, use a
deterministic label order, and serialize `vocabulary.to_dict()` beside every checkpoint. Code can
then request named groups such as `vocabulary.indices("predicate")` without depending on specific
numbers. Constants remain reasonable for tiny, fixed diagnostic experiments only.

## Tensor shapes

Public tensor boundaries use Jaxtyping annotations as executable shape and dtype
documentation. Axis names describe what an axis contains:

```python
q: Float[Tensor, "*batch state"]
indices: Int[Tensor, " indices"]
scores: Float[Tensor, "*batch indices"]
context: Float[Tensor, "*batch context"]
```

The semantic names `batch`, `state`, `indices`, and `context` identify tensor axes. Python
integers that store their sizes use ordinary configuration names such as `batch_size`,
`state_dim`, `num_indices`, and `hidden_dim`. Thus `indices` is an axis, while `num_indices` is
the integer length of that axis.

Runtime enforcement is enabled by the Jaxtyping pytest import hook with Beartype. It therefore
catches inconsistent shapes and dtypes in tests without adding type-checking overhead to normal
training or notebook execution. The leading space in a one-axis shape such as `" indices"` is the
Jaxtyping-recommended Ruff compatibility spelling and has no semantic effect.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

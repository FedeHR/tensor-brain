# Research Fidelity and Project Scope

## Purpose

This repository is not intended to reproduce the Tensor Brain paper's reported numbers,
historical software stack, or VRD label design exactly. It aims to provide a small and solid
Tensor Brain core that can support three kinds of research:

1. Transfer the paper's experiments and tested capabilities to new data, especially video.
2. Extend the experiments with new questions about perception, memory, reasoning,
   continual learning, and cognition that may also produce broader insights for deep learning.
3. Extend the Tensor Brain itself, for example by comparing alternative dynamic-context
   mechanisms while retaining a paper-faithful reference implementation.

Fidelity therefore means preserving the model's concepts, state semantics, mathematical
operations, and experimentally meaningful execution order. It does not mean preserving
every historical dataset choice, label group, visual backbone, hyperparameter, or incidental
implementation detail.

## Evidence hierarchy

The papers are the primary source of truth, but they are not assumed to be infallible. Use
the following evidence hierarchy when implementing or reviewing a component:

1. The mathematical definition and conceptual role stated in the papers.
2. The paper algorithms, figures, appendices, and experimental descriptions considered
   together rather than in isolation.
3. The authors' original implementation as supporting evidence about intent.
4. A minimal implementation that makes the chosen interpretation explicit and testable.

If these sources conflict, or if an apparently paper-faithful choice is mathematically or
experimentally suspicious, do not silently choose an interpretation. Record the evidence,
explain the consequences, and ask for review before changing behavior.

## Decision categories

Every non-obvious decision should be described using one of these categories:

- **Paper-faithful:** directly implements an unambiguous paper equation or algorithm.
- **Reasonable interpretation:** resolves an ambiguity without changing the intended role.
- **Deliberate modernization:** replaces a historical component while preserving its boundary.
- **Experimental extension:** adds a new hypothesis, task, or model variant.
- **Suspected discrepancy:** the paper, appendix, equations, or reference code appear to
  conflict; implementation requires review before proceeding.

This vocabulary prevents a convenient engineering choice from being mistaken for a claim
about the original Tensor Brain.

## Stable scientific core

The following concepts should remain recognizable in code and in experiment schedules:

- `q` is the representation-layer preactivation, or pre-CBS.
- `gamma = sigmoid(q)` is the cognitive brain state (CBS).
- `h` is the original Tensor Brain's dynamic-context preactivation, not the CBS.
- `A[:, k] = a_k` is the learned embedding of symbolic index `k`.
- The same `A` is used for bottom-up index scoring and top-down index feedback - for now. Later we could eventually introduce additional layers in the bottom-up index scoring, i.e., introduce layers between q and A before we produce the index scores. Even, we could get scores without A. However, for the top-down index feeback, it is imperative that A is directly used, as it contains the index embeddings which need to be directly injected.
- Input integration, measurement, attention, and evolution are distinct paper operations.
- Evolution happens when processing moves between concept windows.
- The order of scene, subject, object, unary-label, and predicate operations is part of the
  scientific model and should remain explicit in experiment code.
- The original Tensor Brain recurrence remains available as the reference recurrent
  dynamic-context implementation.

The core should remain small. Abstract trainable components that are genuine experimental
variables; do not hide experimental schedules behind generic protocol runners or composers.

## Replaceable research boundaries

The following are deliberately replaceable without claiming to alter the core Tensor Brain
concepts:

- Perception: precomputed DINO features initially replace VGG and bounding-box CNN features.
- Dataset and ontology: PVSG or other video data may replace VRD, and label groups should be
  defined by the scientific question rather than copied from the original eight groups.
- Dynamic context: QTB feed-forward evolution, the original recurrence, and later xLSTM or
  Mamba variants may be compared behind the evolution boundary.
- Experiment design: original capabilities provide a starting map, but new cognitive and
  deep-learning hypotheses are first-class goals.
- Training infrastructure: dependencies may be modernized as long as they do not obscure the
  model or its execution order.

## Current decision ledger

| Decision | Category | Rationale and status |
|---|---|---|
| Shared bidirectional matrix `A` | Paper-faithful | One learned column per global index is used for both scoring and feedback. |
| Scaled Gaussian initialization of `A` | Deliberate modernization | The paper reports Kaiming initialization for the non-VGG network. Here `A[i, k]` has variance `1 / state_dim`, giving each embedding expected squared norm one independently of vocabulary size. This reviewed deviation must not be described as the paper's initialization. |
| Activation-matched evolution initialization | Paper-faithful plus experimental extension | The sigmoid `QTBEvolution` and original recurrent reference use Xavier initialization for sigmoid/linear maps; `ReLUEvolution` uses Kaiming initialization only for the matrix feeding its ReLU hidden layer and Xavier for its linear output. The historical implementation's Kaiming/ReLU combination remains a reproducibility reference, not the default paper-faithful path. |
| Direct `selection="argmax"` | Reasonable interpretation | It selects the same winner as the paper's inverse-temperature limit and exposes winner-take-all behavior without retaining a temperature parameter. Returned probabilities remain the ordinary model distribution. |
| No inverse-temperature argument initially | Deliberate simplification | Revisit if calibrated sampling, temperature sweeps, or exact historical experiment protocols require it. |
| `retain_gate` and `feedback_gate` | Paper-faithful QTB generalization | They expose neural PVM `(0, 1)`, TB/HB-POVM `(1, 1)`, and no-feedback generative-RNN `(1, 0)` regimes without separate measurement classes. The core does not constrain their values: experiments may supply tensors or learned parameters and own any sigmoid or other parameterization. |
| `IndexVocabulary` owns no parameters | Implementation boundary | It maps symbolic names and candidate groups to stable global IDs. Learned meanings remain exclusively in `A` and `a0`. |
| Global indices versus candidate positions | Implementation boundary | Candidate tensors contain stable global indices into `A`, while compact score tensors are ordered by local candidate position. Measurement outcomes and feedback retain global indices; cross-entropy targets are converted explicitly with `get_positions` or `get_candidate_positions`. |
| Encoder-agnostic input integration | Paper-faithful boundary plus deliberate modernization | `TensorBrain.integrate_input` implements `q <- q + mu * g(nu)` on an already prepared state-shaped input drive. Feature extraction and normalization remain outside the core; initial DINO experiments may use the identity mapping when the feature and state dimensions agree. When they differ, an experiment-owned projection head such as `nn.Linear(input_dim, state_dim)` implements the dimensional part of `g`. The experiment also owns any parameterization of `mu`. |
| Multiple input sources | Reasonable interpretation of QTB Equation 46 | Equation 46 describes a gated sum of inputs from different brain regions. Experiments realize this as repeated `integrate_input` calls, which are algebraically equivalent because the update is additive. Each source may have its own normalization, projection head, and gate, while the explicit calls keep source integration visible within the concept-window schedule. |
| Explicit experiment schedules | Research-readability policy | Some repetition is preferred when it makes paper order, evidence integration, measurements, and evolution boundaries visible. |
| Sigmoid in the reference experiment path | Paper-faithful | `gamma = sigmoid(q)` remains invariant even though the original implementation appendix reports LeakyReLU before index scoring and ReLU modules in the evolution network. `QTBEvolution` retains sigmoid hidden evolution; `ReLUEvolution` is a named experimental extension with activation-matched initialization. |
| Evolution backend boundary | Experimental extension | xLSTM and Mamba2 are suitable future alternatives for persistent dynamic context and long-range concept-window dependencies. They must preserve the evolution contract and retain the original recurrence as the reference. |
| XOR evolution overfitting diagnostic | Research diagnostic | A four-example XOR task makes direct index scoring insufficient and tests whether each evolution backend can learn a known nonlinear transition to near-zero loss. It is an optimization/expressivity check, not a generalization benchmark. |
| Asymmetric bottom-up index adapter | Experimental extension | A learned `phi(gamma)` may enrich bottom-up scoring while selected-index feedback remains the direct embedding `A[:, k]`. The direct `A.T @ gamma` scorer remains the baseline because it preserves the paper's shared bidirectional matrix most transparently. |
| Jaxtyping at tensor boundaries | Research-code modernization | Semantic shape and dtype annotations document `q`, index sets, scores, dynamic context, and toy data contracts. Beartype enforcement is installed only by pytest, so scientific runs keep ordinary PyTorch behavior and overhead. Axis names such as `batch` and `indices` identify dimensions; Python size values remain `batch_size` and `num_indices`. |

Update this ledger whenever an approved decision changes or a new non-obvious interpretation
is introduced.

## Experimental fidelity

Experiments should reproduce or extend capabilities rather than numerical results. Examples
include:

- perception with and without index feedback and dynamic context;
- semantic completion and nonvisual enrichment;
- episodic storage, recall, recency, and similarity-based retrieval;
- compositional and embedded symbolic reasoning;
- continual or self-supervised establishment of new indices;
- prediction, anticipation, and memory-supported decisions on video;
- controlled comparisons of alternative dynamic-context mechanisms.

Each experiment should state which Tensor Brain capability it tests, which paper operation it
uses, which parts are modernized, and which controls could falsify the proposed explanation.

## Review rule

Before implementing a suspicious or conflicting detail:

1. Cite the relevant equations, algorithms, appendix text, and reference-code behavior.
2. Describe the plausible interpretations and their observable consequences.
3. Recommend one option, but ask for review before modifying the implementation.
4. Record the approved resolution in this document and cover it with an equation-level or
   behavioral test.

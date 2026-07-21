# Tensor Brain Repository Instructions

## Project goal

Build a minimal, paper-readable Tensor Brain research core. The repository should support
transferring the paper's capabilities to new data, extending its experiments, and testing
new Tensor Brain and broader deep-learning hypotheses. It is not a numerical reproduction
project.

## Required context

Before making scientific or architectural changes:

1. Read `docs/fidelity.md` and the relevant README sections.
2. Check the relevant equations, algorithms, appendices, and experimental text in both Tensor
   Brain papers.
3. Treat the original implementation as supporting evidence, not as automatically correct.

The papers are the primary source of truth but may contain ambiguities or small flaws. Never
silently reconcile conflicting sources. If a point is suspicious or materially changes model
behavior, present the evidence and ask the user before implementing it.

## Implementation principles

- Preserve the paper notation: `q` is pre-CBS, `gamma = sigmoid(q)` is CBS, and `h` is
  dynamic context.
- Keep the shared index matrix `A` visibly responsible for both scoring and feedback.
- Keep measurement, attention, and evolution distinct.
- Keep experiment order explicit. Do not add generic protocol runners, composers, or callback
  frameworks that hide concept windows or evolution boundaries.
- Abstract only trainable components that are meaningful controlled experimental variables.
- Keep perception outside the core; initially consume precomputed DINO features.
- Retain the original recurrence as a reference when adding xLSTM, Mamba, or other evolution
  variants.
- Define labels and candidate groups in dataset or experiment code rather than hard-coding the
  original VRD ontology into the model.
- Distinguish paper-faithful behavior, reasonable interpretations, deliberate modernizations,
  experimental extensions, and suspected discrepancies.
- Update `docs/fidelity.md` after an approved non-obvious decision.

## Research-code standards

- Prefer direct equations and descriptive tensor names over framework abstractions.
- Annotate public tensor boundaries with Jaxtyping. Use semantic axis names such as `batch`,
  `state`, `indices`, `frames`, and `context`; reserve `batch_size`, `state_dim`, `num_indices`,
  and similar names for Python integers that hold axis sizes. Runtime checks belong in tests,
  not ordinary training execution.
- Test equations, batched behavior, candidate restriction, feedback, gradients, and evolution
  state explicitly.
- Training experiments should include tiny-data overfitting and relevant ablations.
- Marimo notebooks must expose intermediate states and call package logic rather than contain a
  second implementation.
- Preserve unrelated user changes in a dirty worktree.

## Toy versus real experiment structure

Toy experiments may keep synthetic data, rendering, vocabulary construction, explicit model
schedules, training, and evaluation together when that makes the whole experiment readable in
order. Do not generalize that layout to real datasets. PVSG and other real experiments must keep
dataset access, cached feature contracts, experiment/model schedules, and evaluation in clearly
separate modules.

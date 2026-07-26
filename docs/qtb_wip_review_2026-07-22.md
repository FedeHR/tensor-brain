# Review of the July 22, 2026 QTB WIP

## Scope and comparison basis

This report compares:

- `papers/qtb_LATEST.pdf`, the 100-page WIP dated July 22, 2026; and
- `papers/qtb_current_arxiv.pdf`, the 57-page arXiv version 2510.13894v2 dated October 24, 2025.

It also checks implementation consequences against the repository's current core, tests,
README, and `docs/fidelity.md`. The original Tensor Brain paper remains relevant for the
recurrent dynamic-context reference, but the comparison below is primarily WIP versus arXiv
QTB.

### Parse and visual validation

The new PDF is parseable and suitable for detailed technical review.

- The PDF is unencrypted, has a usable text layer, and contains about 30,650 extracted words.
- All 100 pages were rendered and visually inspected. Equations are TeX text/vector content and
  remain legible in the rendered pages; the layout-aware extraction preserves the equations well
  enough to compare operators, arguments, normalizers, and algorithm steps.
- Algorithms 1-9 are legible. Algorithm 5 has an awkward line break in text extraction but is
  clear in the page rendering.
- The principal raster figures were inspected at full-page resolution: the TB architecture on
  page 18, interpolation surfaces on page 51, the neural gHMM on page 56, and the qualitative
  ImageNet example on page 87. Figure 2 on page 35 is vector artwork and is also legible.
- Tables and equation numbering are readable, although the WIP still contains unresolved
  references such as `Eq. ??`, `Algorithm xxx`, and `Figure XXX`.

The manuscript is therefore technically accessible, but it is unmistakably a working document:
pages 2-3 are author notes, several later sections are research scratchpads, and multiple claims
or derivations are explicitly marked as uncertain.

## Executive assessment

The WIP is much more than an edited arXiv revision. It changes the center of gravity of the
paper from a compact quantum-to-probabilistic-to-neural derivation into a broader theory of a
generative, symbolic, memory-supported brain. The mathematical spine is still recognizable,
but it is now surrounded by explicit accounts of causal postselection, sampling, order effects,
embodiment, modular brain inputs, actions, planning, chain-of-thought (CoT), and consciousness.

The most important concrete correction for this repository is the neural evolution input:

```text
gamma = sigmoid(q)
h = sigmoid(v0 + V gamma)
q_next = W h
```

The arXiv paper instead displayed `h = sigmoid(v0 + V q)`. The new derivation in Equations
(30)-(31) and WIP Algorithm 1 consistently makes `gamma`, not `q`, the input to `V`. The current
`QTBEvolution` and its equation-level test already implement the corrected form.

The other major implementation message is that the paper now exposes a family of operating
regimes rather than a single measurement rule:

```text
q_next = alpha q + beta a_k
```

- `(alpha, beta) = (0, 1)`: neural PVM;
- `(1, 1)`: neural HB-POVM / Tensor Brain;
- `(1, 0)`: generative RNN without index feedback;
- intermediate values: partially retaining state and/or outcome feedback.

The repository already exposes these as `retain_gate` and `feedback_gate`. This is a strong
match to the new paper and an excellent experimental seam.

No implementation should be changed solely from this WIP yet. Several source-level
inconsistencies materially affect semantics and should be resolved first, especially the
lifetime of `h`, the gate domain, and the competing gHMM posterior approximations.

## What changed

### Document scale and structure

The paper grew from 57 to 100 pages and from 14 numbered sections/appendices to 25. Some of the
growth is polished conceptual material, while some is a research notebook embedded in the PDF.

| WIP section | Relation to the arXiv version | Main change |
|---|---|---|
| 1. Introduction | Heavily expanded | Adds four brain principles, section-by-section claims, symbolic indices as cognitive measurements, sampling as a central mechanism, and a stronger efficiency contrast with Bayes. |
| 2. Background | Expanded | Adds measurement theories (GRW, CSL, Diosi-Penrose, TSVF, Barandes) and separates the Bayesian Brain discussion. |
| 3. Tensor Brain | Revised | Clarifies concept windows, subject/object/predicate sequencing, mutually dependent outcomes, and the three algorithms. Corrects `Vq` to `V gamma`. |
| 4. Common Themes | New | Provides a common state/operator vocabulary and a comparison table for interference, state erasure, and order effects. |
| 5. Quantum States and Operators | Reorganized from old Section 4 | Isolates quantum states, evolution, PVM, POVM, and HB-POVM; postselection and qubits are moved to later dedicated sections. |
| 6. Probabilistic Quantum | Expanded from old Section 5 | Separates probabilistic PVM as a generative Markov model and HB-POVM as a gHMM; adds a conjecture and an objective-probability discussion. |
| 7. Quantum Phenomena in Probabilistic Quantum | New synthesis | Explicitly analyzes interference, wave-like operators, generative measurement, order effects, and the informed-versus-ignorant nature distinction. |
| 8. Reducing Outcomes | New dedicated section | Distinguishes classical preselection, noncausal global postselection, and tractable causal postselection. |
| 9. Qubits and Probits | Expanded/reorganized | Separates quantum and probabilistic tensorization, measurement-operator tensorization, unistochastic gates, and the probabilistic quantum computer. |
| 10. Neural approximations | Substantially rewritten from old Section 6 | Cleans up the pro-bit derivation, reduces the evolution approximation sequence from four items to three, corrects the neural input to `gamma`, and adds end-to-end learning, postselection, gHMM, external input, and embodiment discussions. |
| 11. Towards the TB Algorithm | Much broader than old Section 7 | Gives separate accounts of the representation, index, and dynamic-context layers, then perception, episodic memory, semantic memory, and language. |
| 12. Quantum Effects and the Brain | Mostly new, with some old discussion relocated | Makes generative sampling, measurement initiation, action effects, gate regimes, order effects, and localized versus distributed indices central. |
| 13. Decisions, Actions and Planning | New | Connects TB to state-space models, LLMs, flow matching, memory-based decision support, CoT, near/long-term planning, actions, and conscious experience. |
| 14. Conclusion | Greatly expanded | Reframes TB as a fast Heisenberg-style alternative to intractable Bayes, but still includes unresolved author notes and speculative claims. |
| 15. Thesis projects | New | Lists proposed experiments on sampling, order, gates, Bayes approximations, CBS sampling, CoT, data, and autoencoding. |
| 16-25. Appendices | Expanded workbench | Adds inverse probabilistic computation, partial measurements, attention derivations, order effects, preliminary ImageNet results, extensive probabilistic-model notes, interference, and local postselection. |

### Abstract and thesis

The arXiv abstract emphasized the sequence of mathematical reductions and the tractability of
probabilistic quantum algorithms relative to Bayes. The WIP abstract is shorter on the detailed
derivation and stronger on common structure across quantum, probabilistic, and neural views:
states, generative measurements, sampling, and measurement-induced state change. It also adds
an evolutionary claim for symbolic index representations and elevates the comparison to LLMs:
memory retrieval as RAG and repeated evolution as CoT.

This makes the paper's empirical burden broader. It is no longer enough to validate the
quantum/probabilistic derivation. The paper now motivates testable claims about symbolic
feedback, memory, order, planning, creativity, active perception, and modular gating.

### The corrected evolution derivation

The old neural approximation used:

```text
h <- sigmoid(v0 + V q)
q <- W h
```

The WIP first defines the factorized probabilistic state `Bern(i; gamma)`, derives the
polynomial mapping `gamma -> gamma'`, and then approximates that map as:

```text
gamma' = sigmoid(f_evol(gamma))                 (30)
h = sigmoid(v0 + V gamma), f_evol(gamma) = W h  (31)
```

Algorithm 1 now repeats `V gamma`. This is not a cosmetic substitution. `q` is unbounded logit
space, whereas `gamma` is the vector of Bernoulli parameters on which the factorized contraction
was derived. Feeding `q` would no longer be the stated neural approximation of the probabilistic
operator.

Repository status: already aligned. `QTBEvolution.forward` computes `gamma = sigmoid(q)` and
applies `V` to `gamma`; `tests/test_evolution.py` checks the equation directly. The README also
states the corrected equation. The focused model/evolution test suite passes (22 tests).

### Measurement, gates, and operating regimes

WIP Algorithm 3 makes the generalized update part of the main algorithm rather than only a
side interpretation:

```text
k ~ softmax(a0,k + a_k^T gamma)
q <- alpha q + beta a_k
```

Section 12.2.2 then names the canonical regimes. This greatly strengthens the case for treating
`alpha` and `beta` as controlled experimental variables. It also clarifies three distinct causal
questions:

1. Does the previous state survive the measurement?
2. Does the sampled symbol feed back into the state?
3. Is the symbol sampled, selected by argmax, or supplied by a teacher?

The repository represents all three axes cleanly. Its `measure` method separates gates from
selection, and the `PVM/HB-POVM/gRNN` regimes have equation-level tests.

### Causal postselection becomes a first-class operation

The old paper treated postselection compactly. The WIP distinguishes:

- preselection/state reduction, which renormalizes a conditional likelihood and can make
  inference difficult;
- global postselection, which is rejection sampling over repeated runs and can be noncausal;
- causal postselection, which restricts the currently available outcomes and renormalizes at
  the current measurement.

The Tensor Brain uses the causal form. Repository candidate groups implement the same local
operation: score only the permitted global indices, normalize within that set, then map the
local selected position back to the global index. This is more than a dataset convenience; it is
the implementation of the paper's causal postselection boundary.

### The neural approximation is more explicit

The WIP replaces the old four-step evolution approximation with a clearer three-stage story:

1. factorize the conditional output distribution;
2. contract a factorized Bernoulli input and project the result back to a factorized form;
3. replace the exponentially large polynomial map with a neural interpolator.

It also makes the approximation risk visible. Figure 3 shows that a network can match all binary
corners yet behave incorrectly in the interior of `[0,1]^n`; the text calls for high capacity plus
smoothness-promoting regularization. This is an important experimental point that the current
XOR overfit diagnostic does not test.

For index probabilities, the WIP distinguishes the exact polynomial expectation in Equation
(35) from the Jensen approximation in Equation (36):

```text
P(k | state) ~= softmax_k(a0,k + a_k^T gamma)
```

This identifies the direct linear scorer as a specific approximation, not merely an architectural
convention. A nonlinear bottom-up scorer is therefore a principled experimental relaxation,
provided selected-index feedback remains the direct column `a_k` if the shared symbolic
embedding hypothesis is being tested.

### Embodiment, modular inputs, memory, and actions

Several ideas that were peripheral in the arXiv paper now receive explicit equations or sections:

- External input remains additive in logit space, `q <- q + g(nu)` (Equation 43).
- An inverse/top-down map `g+` is proposed so an index embedding can be decoded back toward a
  sensory or embodied representation. The cycle `nu_k -> a_k -> nu_hat_k` is described as an
  autoencoder and a possible self-supervised objective.
- Multiple brain modules contribute `sum_k mu_k g(nu_k)` (Equation 46), making the `mu_k`
  values explicit module gates.
- A perceptual concept window combines evolved context, current input, and all selected symbolic
  embeddings: `q <- W h + g(nu) + sum_{k in S} a_k` (Equation 47).
- Episodic recall is treated like perception except that an episodic index, rather than sensory
  input, initializes the state.
- Semantic recall activates an index and then related indices, enabling embodied symbolic
  completion.
- Actions are proposed as ordinary indices whose embodiment can affect the external world.

These additions fit the repository's goal very well, but they belong in explicit dataset and
experiment schedules rather than in a larger generic core abstraction.

### Order effects are now split into distinct mechanisms

The WIP usefully separates effects that are easy to conflate:

- A neural PVM (`alpha=0`) is order-sensitive because each outcome overwrites the previous
  state.
- An unpostselected neural HB-POVM (`alpha=1`) has an additive, order-invariant posterior within
  a concept window.
- Causal postselection can still produce order effects in the HB-POVM likelihood because the
  available candidate event changes the conditional normalization.
- Evolution between concepts is itself order-sensitive because it is nonlinear and the concept
  sequence matters.
- Specific-to-general symbolic chains (for example, tiger then danger) add a learned semantic
  order effect that may be useful even when the HB posterior update is commutative.

The ImageNet tables are retained and moved into the later discussion/appendix. They should be
treated as preliminary: the paper presents exact KL/JSD and reversal-rate claims on 100,000
images, but the repository currently contains neither this experiment nor a reproducible data,
checkpoint, or evaluation path.

### CoT and planning are promoted from analogy to operating mode

The arXiv version mainly compared TB memory with RAG. The WIP now says that repeated evolution,
with indices sampled between steps and no new perceptual input, implements a CoT-like rollout.
Section 13.5 further divides this into near-term, state-grounded planning and longer-term scenario
construction. Candidate trajectories can be sampled, evaluated by reward, and used for action
selection.

This is one of the strongest new directions because it uses existing TB operations in a new
schedule rather than requiring a new monolithic architecture.

## Implementation impact and required decisions

| Item | WIP evidence | Repository status | Recommendation |
|---|---|---|---|
| `V gamma`, not `V q` | Eqs. 30-31; Algorithm 1 | Correct already | Keep the equation-level test and cite the WIP correction in future fidelity notes once the manuscript stabilizes. |
| `q` versus `gamma` boundary | Algorithm 1 says input/output CBS but returns `q`; Eq. 30 makes `gamma' = sigmoid(q_next)` | Repository consistently returns pre-CBS `q_next` | Treat the implementation as the coherent reading, but correct the paper's Algorithm 1 input/output labels. |
| Shared matrix `A` | Eqs. 32, 36, 38; Algorithms 2-3 | Correct already | Keep direct `A.T @ gamma` as baseline and direct `A[:,k]` feedback as the symbolic path. |
| Gate domain | Algorithm 3 states `0 <= alpha,beta <= 1` | Core accepts arbitrary tensors/floats; one test deliberately uses `1.2` | Keep the core permissive for extensions, but require bounded parameterization in paper-faithful experiments and rename/document the out-of-range test as an extension. This needs approval because it touches the fidelity ledger. |
| Canonical gate regimes | Section 12.2.2 | Correct already | Build a standard ablation helper in experiment code, not a new measurement class hierarchy. |
| Causal postselection | Sections 8.2.2 and 10.5 | Candidate groups implement the operation | Document candidate restriction explicitly as causal postselection in experiment reports. |
| Attention | Eq. 44 is expected embedding feedback | `attend` matches it | Preserve attention and sampled measurement as distinct operations. |
| External input | Eq. 43 | Precomputed features are added to `q` in experiment code | Correct and appropriately outside the core. |
| QTB hidden state `h` | Algorithm 1 is feed-forward; Section 11.3 says `h` remains active through the next concept interval | QTB computes `h` but returns no context; recurrent backends return context | Do not silently make QTB recurrent. Consider exposing `h` as an optional trace/per-window activation for analysis, distinct from recurrent context. Resolve wording with the authors first. |
| Smooth interpolation | Section 10.2 and Figure 3 | No dedicated diagnostic | Add a corner-versus-interior interpolation benchmark and regularization sweep. |
| Embodiment decoder `g+` | Section 10.8 | Not implemented | Add only in an experiment/perception module; do not place a decoder in the core TB state/index machinery. |
| Multi-module gates `mu_k` | Eq. 46 | Not implemented | Compare fixed, learned scalar, vector, and context-dependent gates in experiment code. |
| Episodic/semantic memory | Sections 11.6-11.7 | Vocabulary exists; no memory store/retriever | Keep memory stores dataset/experiment-specific initially; use the same global index IDs and direct `A` feedback. |
| Stochasticity | Sections 12.1.2 and 14.1.2 locate inference randomness primarily in outcome sampling | `selection=sample` is stochastic; `argmax` and teacher modes are deterministic | Separate inference sampling from training noise, data noise, dropout, and randomized initialization in every experiment. |
| Distributed indices | Section 12.3.2 | Core uses localized global indices | Treat distributed indices as an explicit alternative model, not a transparent refactor of `IndexVocabulary`. |

### Manuscript inconsistencies that should be resolved before code changes

1. **Algorithm 1 types:** the input is labeled CBS `gamma` and the output "updated CBS", but
   the returned value is `q`. Equations (30)-(31) support returning pre-CBS `q`, followed by
   `gamma'=sigmoid(q)`.
2. **Gate typo in Section 10.5:** the text says `alpha = 1m beta = 0` yields the neural PVM.
   Algorithm 3 and Section 12.2.2 show the intended PVM setting is `(0,1)`.
3. **Gate range versus experimental freedom:** Algorithm 3 explicitly bounds both gates to
   `[0,1]`; the current fidelity ledger allows unconstrained learned values. These can coexist
   only if the latter is clearly labeled an experimental extension.
4. **Lifetime of `h`:** the mathematical algorithm is feed-forward, but Sections 11.3 and
   14.1.2 give `h` persistent biological meaning across a concept interval. Numerical persistence
   may be unnecessary once `q=Wh` has been formed, but the distinction matters for observability
   and for comparisons with recurrent evolution backends.
5. **gHMM posterior:** Section 10.6 displays two alternatives, Equations (41) and (42), with
   "or should this be". No implementation should claim the WIP gHMM approximation until this is
   settled.
6. **Attention typo:** Equation (44) appears to contain `a_{ell=1,k}` where `a_{ell,k}` is
   intended.
7. **Claim strength:** the manuscript repeatedly says all uncertainty comes from quantum
   measurement sampling. This should be framed as a model assumption at inference time, not as
   an established property of biological brains or of the training process.
8. **Preliminary results:** Sections 12.2.5 and 22 state strong ImageNet results without a
   reproducible experiment in this repository. Until reconstructed, they should not anchor new
   implementation decisions.
9. **Draft markers:** the WIP still contains a large number of `New:`, `??`, `xxx`, `Maybe
   Remove`, `MOVE TO`, and author-addressed prompts. Sections 15-25 should be read as a research
   backlog and derivation notebook, not all as settled paper claims.
10. **Editorial cleanup:** the WIP title currently says "probabilistc", the abstract contains
    "episoidc", and similar spelling/grammar errors occur throughout. These do not obstruct
    parsing, but a global editorial pass should happen only after the unresolved derivations and
    section moves are settled.

## Recommended experimental program

The most useful sequence is to validate the approximations first, then validate the distinctive
TB mechanisms, and only then scale to video. This prevents dataset complexity from obscuring a
wrong equation or an ill-understood inference approximation.

### Priority 0: equation and approximation experiments

#### 1. `V gamma` versus `V q`

Train otherwise identical evolution networks on both a known polynomial transition and a small
concept-window prediction task. Evaluate binary corners and interior states separately.

- Hypothesis: `V gamma` better matches the derived operator and generalizes more smoothly inside
  the probability cube.
- Controls: matched initialization, capacity, optimizer, and parameter count.
- Metrics: interior MSE/KL, corner error, Jacobian norm, calibration, and rollout stability.
- Value: validates the newly corrected equation empirically instead of treating it only as a
  typo fix.

#### 2. Exact Bayes versus TB on small states

Finish and formalize WIP Section 16.1. For small `n`, enumerate all `2^n` states and compare:

- exact Bayes;
- exact posterior projected to independent Bernoulli marginals;
- variational mean field;
- the TB logit-addition update;
- no update of the prior;
- optionally particle or importance sampling.

Sweep prior concentration, `A` scale, `a0`, number of sequential measurements, and candidate-set
size. Report both posterior quality and wall-clock/memory scaling. This is the single most
important experiment for the paper's tractability-versus-accuracy thesis.

#### 3. Gate phase diagram

Sweep `(alpha,beta)` over `[0,1]^2`, with separately learned bounded gates as a second condition.
Measure retention, feedback strength, calibration, order sensitivity, gradients, and task
accuracy. Include the three named corners and a no-state `(0,0)` sanity control.

This directly tests whether intermediate gates are useful regimes or only an interpolation
convenience. Learned gates should be parameterized through a sigmoid for the paper-faithful
condition; an unconstrained condition can be reported as an extension.

#### 4. Causal postselection

Compare full-vocabulary measurement, preselection, global rejection-based postselection, and
causal candidate restriction on an exactly enumerable problem. Vary the retained fraction and
whether candidate sets depend on earlier outcomes. Measure distributional correctness,
efficiency, and induced order effects.

#### 5. Neural interpolation regularity

Recreate Figure 3 as a train/test benchmark rather than only an illustration. Fit values at
binary corners and evaluate the unseen interior under weight decay, spectral normalization,
Jacobian penalties, monotonic constraints, and different hidden activations. This provides a
principled basis for the current sigmoid/ReLU evolution comparison and later xLSTM/Mamba
extensions.

### Priority 1: distinctive TB behavior

#### 6. Specific-to-general order effects

Rebuild the fine/coarse experiment in a fully reproducible form, then extend it beyond ImageNet
hierarchy labels.

- Compare PVM, HB-POVM, gRNN/no feedback, attention-only, and learned gates.
- Separate posterior order invariance from likelihood order effects caused by changing candidate
  sets.
- Compare a linear scorer with `A.T @ phi(gamma)` to test the WIP question of whether a nonlinear
  scorer learns useful semantic order automatically.
- Include reverse or unrelated hierarchies as falsification controls.

#### 7. CoT as an explicit Tensor Brain mode

Define CoT as a readable schedule, not a new opaque model:

```text
seed q from perception, memory, or a query
repeat:
    q <- evolve(q)                 # no new perceptual input
    k <- measure(q, candidates)    # sampled or teacher-forced thought index
    optionally retrieve memory or apply action/reward feedback
until stop index, budget, or convergence
```

Start with synthetic relational tasks whose proof or plan length is known. Compare zero-step,
fixed-depth, adaptive-stop, and multi-sample search. Evaluate exact answer, path validity,
diversity, calibration, and whether intermediate indices are causally useful by intervening on or
shuffling them.

Evolution backends should be a controlled variable: corrected QTB feed-forward evolution,
original TB recurrence, vanilla RNN/GRU, and later xLSTM or Mamba. The measurement, candidate
sets, and shared `A` path should remain unchanged across backends.

#### 8. Embodiment and index autoencoding

Add an experiment-owned decoder `g+` from state/index embeddings back to perception features or
images. Compare classification-only learning with cycle consistency, contrastive reconstruction,
and masked-prediction objectives. Test whether embodiment improves few-shot index grounding,
semantic completion, and robustness to partial visual evidence.

#### 9. Episodic memory and index recruitment

Create episodic indices online, store their `A` columns with time metadata, and retrieve them by
the standard scoring/attention path. Test recency, similarity, remote retrieval, interference,
and continual creation of new indices. A reserve pool of unused indices is a simple first model;
dynamic vocabulary growth can come later.

#### 10. Modular input gating

Use Equation (46) to combine perception, episodic retrieval, semantic retrieval, reward, and
action-state modules. Compare fixed gates, learned global scalars, per-feature gates,
context-dependent gates, and a sparse mixture-of-experts gate. The key falsification test is
whether a learned gate selects the genuinely informative module under controlled corruption,
not merely whether it improves average accuracy.

### Priority 2: a video-centered program

A useful video dataset should support more than frame classification. The selection criteria
should be driven by the TB questions:

- persistent object identities or tracks across frames;
- temporally localized subject-object-predicate annotations;
- relation changes and action transitions, not only static scene graphs;
- clips long enough to distinguish recent from remote episodic memory;
- enough label hierarchy for specific-to-general experiments;
- compositional or identity-disjoint splits;
- access to raw video and a stable license so precomputed DINO-style features can be versioned;
- episode and timestamp metadata suitable for serialized episodic indices.

PVSG is a reasonable starting direction already named in the repository fidelity document, but
the final dataset choice should be made against this contract rather than because it is marketed
as a video scene-graph dataset.

Recommended video experiments, in increasing difficulty:

1. **Explicit concept-window transfer:** scene -> subject -> object -> predicate, with evolution
   boundaries visible. Compare direct perception, expected attention, and sampled feedback.
2. **Temporal relation recognition:** test whether dynamic context improves predicates that
   cannot be inferred from a single frame. Use temporally shuffled and single-frame controls.
3. **Object permanence and occlusion:** remove visual evidence temporarily and test whether
   episodic/index feedback maintains identity and relation state.
4. **Anticipation:** predict the next relation, action, or salient entity before it appears.
   Compare no memory, recent episodic memory, remote similar episodes, and evolution backends.
5. **Event boundary and episodic storage:** learn when to create an episodic index and whether
   the resulting memory supports later recognition or decision making.
6. **Memory-supported decision:** retrieve similar past situations and predict actions/outcomes;
   include retrieval-shuffle controls to show that improvement is due to the right memory.
7. **Active perception / gorilla-style intervention:** manipulate top-down index priors and
   candidate groups, then measure missed unexpected events versus improved detection of expected
   ones. This directly tests the WIP claim that perception is an interaction between bottom-up
   evidence and top-down construction.
8. **Video CoT and future rollouts:** after observing a prefix, stop external input and unroll
   evolution/measurement steps to generate possible future concepts, relations, or actions.
   Score both best-of-N accuracy and diversity/calibration of the sampled futures.
9. **Multi-timescale evolution:** compare frame-level, event-level, and shortcut transitions to
   future episodic states. This addresses the WIP note that useful future memories may avoid long
   iterative rollouts.

For every real-video experiment, keep dataset access, cached feature contracts, explicit model
schedules, and evaluation in separate modules, as required by the repository instructions.

## Architectural extensions that fit the paper and repository

The following are especially compatible with the modular research core because each changes one
scientific boundary while preserving the others:

1. **Evolution backend:** corrected QTB, original TB recurrence, GRU/xLSTM, Mamba, or a
   multi-timescale model behind the same concept-window transition.
2. **Bottom-up scorer:** direct `A.T @ gamma` versus `A.T @ phi(gamma)` or a separate scorer,
   while keeping direct `A[:,k]` symbolic feedback.
3. **Measurement gates:** fixed canonical regimes, learned bounded scalar gates, per-state gates,
   or module-conditioned gates.
4. **Outcome rule:** sampling, argmax, teacher forcing, top-k/nucleus sampling, or structured
   candidate generation. Probability calibration must remain observable.
5. **Index representation:** localized columns as the baseline versus distributed codes as a
   clearly named alternative.
6. **Embodiment:** no decoder, feature decoder, image/video decoder, or action actuator.
7. **Memory policy:** fixed episodic indices, online recruitment, consolidation, forgetting, and
   recent/remote retrieval policies.
8. **Operating schedule:** perception, recall, attention, CoT, planning, or action mode. These
   should remain explicit schedules rather than a generic protocol engine.

## Sections most promising for future work

My strongest ranking is:

1. **Section 10.2 and Figure 3:** it converts the quantum-inspired narrative into a precise
   approximation question that can be falsified with small, exact experiments.
2. **Sections 10.3-10.6 and Appendix 16:** the gap between exact Bayes, factorized Bayes, and the
   TB logit update is the paper's central computational claim and currently lacks adequate
   empirical support.
3. **Sections 12.2 and 15.4:** gate-controlled order effects provide a clean experimental axis
   linking PVM, HB-POVM, and gRNN behavior without changing the rest of the model.
4. **Sections 13.4-13.5:** CoT/planning is a natural new mode of operation and a strong test of
   whether sampled indices are useful internal interventions rather than decorative labels.
5. **Sections 10.8 and 11.4-11.7:** embodiment plus modular memory inputs could turn the shared
   index matrix into a genuinely testable grounding and retrieval mechanism.
6. **Section 8:** causal postselection explains candidate restriction in a way that can be tested
   for both correctness and efficiency.
7. **Video-facing parts of Sections 11 and 13:** video provides the temporal structure needed to
   separate perception, context, episodic memory, anticipation, and planning - capabilities that
   static image benchmarks cannot disentangle.

The distributed-index and consciousness sections are conceptually interesting, but I would not
prioritize them experimentally until the exact-Bayes comparison, gates, memory, and CoT schedule
have established that the core mechanisms work.

## Suggested execution order for this repository

1. Preserve the current corrected `V gamma` implementation and tests.
2. Add a small exact-inference benchmark for Section 16.1.
3. Add the gate/order/postselection experiment suite.
4. Add the interpolation regularity diagnostic.
5. Specify CoT as one or more explicit experiment schedules on synthetic tasks.
6. Select a video dataset using the temporal/memory contract above and build the feature/data
   adapter separately from the experiment schedule.
7. Transfer perception first, then add episodic memory and anticipation, and only then video CoT
   or planning.
8. Revisit `docs/fidelity.md` only after the authors decide the gate-domain and `h`-lifetime
   ambiguities.

This sequence gives the paper a credible empirical foundation while keeping the codebase small,
readable, and open to architectural changes.

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
| Index score offsets | Paper-faithful conditions plus controlled extensions | The original TB perception equations use direct `A.T @ gamma` scores. The current QTB arXiv explicitly defines `a0,k = -sum_l softplus(a_l,k)` in Equation (31). The July 22 latest draft reorganizes the derivation: Equation (25) gives the generic factorized-Bernoulli log-normalizer, Equations (32)-(33) set `q = a_k` for an outcome column, and Equation (34) uses the resulting `a0,k` in the score. Thus `a0` remains derived from `A`, although the latest draft does not restate the columnwise formula beside Equation (34). Experiments name four conditions: `direct`, `softplus-bias`, the neutral-offset control `centered`, and the repository's earlier ordinary learned `learned-bias`. No condition changes the direct `A` feedback path. |
| `IndexVocabulary` owns no parameters | Implementation boundary | It maps symbolic names and candidate groups to stable global IDs. Learned meanings remain in `A` and, only for the `learned-bias` condition, its ordinary bias. |
| Global indices versus candidate positions | Implementation boundary | Candidate tensors contain stable global indices into `A`, while compact score tensors are ordered by local candidate position. Measurement outcomes and feedback retain global indices; cross-entropy targets are converted explicitly with `get_positions` or `get_candidate_positions`. |
| PVSG ontology metadata versus model indices | Approved minimal implementation boundary | Human-reviewed hierarchy and property definitions remain versioned experiment data rather than a second runtime object model. An experiment constructs the existing `IndexVocabulary` directly from deterministic named groups; one label receives one global column of `A` even when it belongs to several groups. The exact vocabulary is saved beside each checkpoint. Video identities and episodes join the same global vocabulary as experiment-specific indices, while boundary source and whether a fact is semantic or episodic remain record metadata. |
| PVSG initial vocabulary layout | Implementation boundary | Each protocol supplies exactly the identities supervised by its training or enrollment records. The initial builder includes predicates supported by complete pair evidence in the frozen model-selection training subset and only category labels supported by those identities; semantic-property, semantic-relation, and episodic indices remain absent. Deterministically ordered static predicate and selected category groups precede the protocol-specific identity group so later fixed few-shot enrollment can extend identity columns without changing the meaning of the static prefix of `A`. |
| PVSG four-level object hierarchy | Experimental extension and dataset interpretation | The pinned source inventory maps included observations through strict fine-to-basic-to-coarse kind-of paths and then into one broad semantic domain. The four level vocabularies are pairwise disjoint and strictly decrease in size. `human`, `animal`, and `plant_life` are basic branches of `living_being`; `animal` is never a parent of `human`; source `thing`/`stuff` are not semantic parents. All retained `bat` tracks receive identity-level corrections to baseball bat, golf club, tennis racket, or badminton racket. `board`, `gift`, `others`, `paper`, `powder`, `rack`, `ring`, and `stand` remain outside semantic supervision rather than receiving forced paths. The mapping yields 121 fine, 78 basic, 22 coarse, and 5 domain labels, hence 226 distinct static columns. The five domains contain 17–29 fine labels and at least two coarse children each, permitting within-domain held-out-coarse generalization; uneven track frequency still requires class-aware training and macro reporting. Human-reviewed semantics define the hierarchy; frozen DINO diagnostics test visual support without silently changing it. The validated hierarchy is copied into each frozen PVSG manifest snapshot so a run never depends on mutable repository metadata. |
| PVSG unary category readouts | Paper-aligned operation plus dataset-transfer extension | Original Algorithm 1 lines 18–22 identify the subject, feed its index embedding into `q`, and then decode a unary label; Table 5 evaluates entity and B/P/G class plus attribute labels. Integral PVSG models therefore score category groups from the post-identity-feedback state, while P-Direct scores them independently from the corresponding object feature. Applying the same unary readout to the object participant is the symmetric tracked-entity extension. Reviewed fine/basic/coarse labels are the closest B/P/G analogues and domain is an additional abstraction. Category predictions are not fed back initially. The model exposes the readouts without forcing their loss: identity-only, official-source-category, and reviewed-hierarchy checkpoints are separately named conditions because auxiliary semantic supervision changes shared parameters. Missing reviewed paths mask only the unavailable hierarchy losses. |
| PVSG semantic property and relation inventory | Experimental extension grounded in existing norms | The initial candidate inventory separates taxonomic labels from semantic features following CSLB/McRae, normalizes visual property families from VAW, and adopts four THINGSplus affordance dimensions. It contains 49 unary values and 9 semantic-relation predicates. Omission always means unknown. Transient visual state is episodic; category-typical affordance and positive physical-harm risk are semantic. Ownership and kinship are identity relations with explicit inverses or symmetry, not unary attributes. The inventory defines indices only; no fact is asserted without a separate versioned assignment and provenance artifact. |
| Encoder-agnostic input integration | Paper-faithful boundary plus deliberate modernization | `TensorBrain.integrate_input` implements `q <- q + mu * g(nu)` on an already prepared state-shaped input drive. Feature extraction, normalization, scaling, and projection remain outside the core. An identity mapping is appropriate only when both dimension and pre-CBS scale agree; dimensional agreement alone is insufficient. When dimensions differ, an experiment-owned projection head such as `nn.Linear(input_dim, state_dim)` implements the dimensional part of `g`. The experiment also owns any parameterization of `mu`. |
| Multiple input sources | Reasonable interpretation of QTB Equation 46 | Equation 46 describes a gated sum of inputs from different brain regions. Experiments realize this as repeated `integrate_input` calls, which are algebraically equivalent because the update is additive. Each source may have its own normalization, projection head, and gate, while the explicit calls keep source integration visible within the concept-window schedule. |
| Explicit experiment schedules | Research-readability policy | Some repetition is preferred when it makes paper order, evidence integration, measurements, and evolution boundaries visible. |
| Sigmoid in the reference experiment path | Paper-faithful | `gamma = sigmoid(q)` remains invariant even though the original implementation appendix reports LeakyReLU before index scoring and ReLU modules in the evolution network. `QTBEvolution` retains sigmoid hidden evolution; `ReLUEvolution` is a named experimental extension with activation-matched initialization. |
| Evolution backend boundary | Experimental extension | xLSTM and Mamba2 are suitable future alternatives for persistent dynamic context and long-range concept-window dependencies. They must preserve the evolution contract and retain the original recurrence as the reference. |
| XOR evolution overfitting diagnostic | Research diagnostic | A four-example XOR task makes direct index scoring insufficient and tests whether each evolution backend can learn a known nonlinear transition to near-zero loss. It is an optimization/expressivity check, not a generalization benchmark. |
| Asymmetric bottom-up index adapter | Experimental extension | A learned `phi(gamma)` may enrich bottom-up scoring while selected-index feedback remains the direct embedding `A[:, k]`. The direct `A.T @ gamma` scorer remains the baseline because it preserves the paper's shared bidirectional matrix most transparently. |
| Jaxtyping at tensor boundaries | Research-code modernization | Semantic shape and dtype annotations document `q`, index sets, scores, dynamic context, and toy data contracts. Beartype enforcement is installed only by pytest, so scientific runs keep ordinary PyTorch behavior and overhead. Axis names such as `batch` and `indices` identify dimensions; Python size values remain `batch_size` and `num_indices`. |
| PVSG DINOv3 perception boundary | Deliberate modernization | The pinned ViT-B/16 DINOv3 encoder consumes the complete frame at an approximately aspect-preserving, patch-aligned resolution without cropping or a square warp. Its normalized CLS token supplies scene evidence; spatial tokens are pooled by exact mask/patch overlap for objects and enclosing union boxes for pairs. CUDA extraction uses fixed FP16 autocast on both Turing and newer GPUs so one feature snapshot does not mix hardware-selected FP16 and BF16 inference. The encoder stays outside `src/tb`, and cached artifacts retain the exact model revision, preprocessing contract, and inference dtype. |
| PVSG object and pair regions | Experimental interpretation | Oracle panoptic masks define object evidence without object bounding-box context. A pair's enclosing union rectangle deliberately includes both participants and their intervening space. All simultaneously visible unordered pairs are cached once; task code later supplies subject/object direction and labels. |
| PVSG relation-span convention | Reasonable interpretation resolving a suspected discrepancy | Relation endpoints are inclusive. Boundary-frame inspection showed the annotated relations continuing through the stated endpoint, and 1,787 source spans end at the final valid frame. OpenPVSG is inconsistent: relation evaluation uses `range(start, end + 1)` while one training-preparation path uses `range(start, end)`. Task construction intersects every inclusive span with `[0, num_frames - 1]`, records every clipped or empty span, and never silently discards an entire partly valid interval. |
| PVSG source exclusions | Dataset-quality decision | The initial snapshot excludes six versioned videos listed in `experiments/pvsg/exclusions.json`: two encoded videos have fewer decodable frames than `pvsg.json` and four masks contain object IDs absent from their video annotations. The materializer requires every other artifact and records the complete allowlist and reasons in provenance. |
| PVSG observed predicate ontology | Dataset interpretation | The retained annotations contain seven semantically meaningful predicates absent from the declared 57-label list: `climbing`, `enclosing`, `getting down on`, `going down`, `pouring`, `squatting on`, and `squeezing`. They are retained without automatic synonym or inverse-relation mappings, producing 64 active labels. The source-only `moving` annotation belongs to an excluded video and is documented with zero retained support. Metrics distinguish all labels, train-supported labels, and train-unseen labels. |
| PVSG initial feature normalization and access | Deliberate modernization plus implementation boundary | Cached DINO tensors remain raw. For a feature dimension `D`, the initial experiment converts each addressed scene, object, and union vector to float32 and applies `sqrt(D) * L2-normalize(x)` with epsilon `1e-12`. Thus every nonzero input has component RMS one: it preserves DINO direction and removes source-dependent norm differences without pinning `sigmoid(q)` near `0.5`. This scale was not used in the original TB, whose jointly trained VGG mapping could choose its output scale, and must be reported as a modernization rather than paper-faithful preprocessing. Only addressed rows are normalized. Training shuffles rows within each video, divides them into bounded 1,024-observation chunks, and shuffles chunks globally. This retains useful raw-artifact locality without allowing an entire video to determine a long contiguous optimization phase. Raw input and learned input mappings remain named ablations; future mappings never overwrite the cached features. |
| PVSG scale and initialization trace | Required research diagnostic | Every overfit and full run evaluates a fixed addressed batch at initialization, fixed early steps, and the final checkpoint. `scale_trace.jsonl` records raw cached feature norms; per-window input and `q` scale; sigmoid quantiles and saturation; the `q`-space and CBS-space changes caused separately by input, feedback, and evolution; effective neutral score offsets versus data-dependent scores; index norms by candidate group; P-SA and winner feedback ratios; attention entropy; and parameter-group gradient norms. The run saves the exact normalization, score mode, initialization, gates, diagnostic addresses, cadence, and aggregation definitions. Future normalization or initialization changes receive new condition names and ledger rows and reuse the same raw features and diagnostic batch, allowing these traces to become a standalone empirical analysis rather than transient debugging output. |
| PVSG first full object grid | Controlled screening experiment | The first full-data run crosses only original recurrent versus QTB evolution, centered versus QTB softplus-normalized scoring, and Adam learning rates `1e-4`, `3e-4`, and `1e-3` at seed 0. All 12 conditions use the frozen RMS-normalized DINO inputs, activation-matched initialization, batch size 128, 10,000 P-SA updates, no weight decay, and the four hierarchy losses. The initial run used whole-video blocks and exposed synchronized validation oscillations across conditions. Its exact grid is repeated under a new output name with shuffled 1,024-observation within-video chunks; the original outputs remain an optimization-order diagnostic rather than being overwritten. Checkpoints are selected by summed hierarchy cross-entropy on one deterministic 20,000-observation subset of held-out development videos. The best checkpoint is then evaluated with P-SA and P-Samp on all development observations for semantics and on the training-role blocked interval for known-identity recognition. Official evaluation videos remain untouched. This is a mechanism-relevant hyperparameter screen, not a 12-way claim of final test performance. |
| PVSG object evaluation aggregation | Required reporting boundary | Observation-micro accuracy, macro recall over supported target classes, identity-macro accuracy, and video-macro accuracy are reported separately for every semantic level. Their means across hierarchy levels remain explicitly named level means rather than `category_macro`. Known-identity recognition additionally reports observation-, identity-, and video-aggregated accuracy. Adjacent frames are not treated as independent replicates for paper claims. |
| PVSG fixed-pair overfit gate | Research diagnostic | Before full training, one deterministic 200-row prefix of the immutable video-major training-pair manifest is loaded into memory and optimized as a single batch. This selection minimizes feature I/O and is never treated as an evaluation sample. Identity targets are explicit candidate positions. Predicate labels define a uniform categorical target over the simultaneously active labels; its soft-target cross-entropy reduces to ordinary cross-entropy for the 97% single-label majority. The logged KL term removes only the target entropy and lets the overfit criterion retain a zero optimum. Each enabled unary category group contributes an unweighted cross-entropy for both entities; unavailable reviewed paths mask only that semantic term. Success requires every available target to be correct and cross-entropy excess at most `0.01`. Integral training uses differentiable P-SA, while P-Samp is an evaluation-only readout of the same checkpoint. Multi-label Bernoulli BCE remains a future named extension rather than the initial objective. The exact rows, vocabulary, objective components, predictions, optimizer state, and scale trace are saved. |
| PVSG object observation schedule | Paper-aligned truncation and video extension | The object schedule is the original perception program's scene window followed by one entity window: scene input, evolution, object-mask input, identity readout, identity feedback, and unary category readouts. It stops before the second entity and predicate windows, so it tests identity indexing, semantics, blocked recognition, and few-shot re-identification without claiming to test relation transport. Applying the paper's subject operation to any tracked entity is the dataset-transfer extension. |
| PVSG object-first frozen views | Approved experimental scope | Object observations are first-class records and never require an annotated relation, a visible partner, or union evidence. The canonical frame view includes every retained frame, including empty-visible-object frames, and stores ascending visible object IDs with their feature rows. This is the primary substrate for identity, category, sequential attention, disappearance/reappearance, episodic-memory, and re-identification experiments. An absent object has no fabricated feature; memory experiments must specify how the state evolves while it is absent. |
| PVSG initial complete-pair view | Approved relation-task scope | The initial four-input relation transfer consumes only positive-pair records with scene, subject-mask, object-mask, and distinct-pair union rows. The feature audit proves that every distinct pair of simultaneously mask-visible objects is cached, so an incomplete join represents annotation/mask disagreement or a source self-relation rather than an extractor omission. Canonical pair records retain nullable feature addresses and explicit subject/object/union evidence flags. The ontology stores per-predicate support for both all positive frames and the complete subset, making evidence selection visible without another audit artifact. Later missing-input conditions may consume incomplete records only with an explicit evidence mask and state-update rule; they must not silently pad, interpolate, or teacher-force absent evidence. This restriction does not filter object or frame views. |
| PVSG model-selection split | Dataset protocol decision | Within each source, a salted deterministic hash reserves 15% of retained official-training videos for development and leaves the remainder for training. The official validation videos remain untouched for final evaluation. Membership and the salt are frozen in `splits.json`; splitting by video prevents adjacent-frame leakage. |
| PVSG nested few-shot support | Dataset protocol decision | The retained data are asserted to run at 5 FPS. Enrollment uses the earliest ten mask-visible observations separated by at least five frames, then requires a 25-frame query embargo after the tenth support. Evaluating only the first `k` ranked supports for `k in {1, 3, 5, 10}` gives an identical eligible-identity pool and identical query set across shot counts. Parallel development-video manifests support model selection; official-validation identities remain final evaluation. The manifest records support frames and temporal span. The stricter ten-support requirement reduces the eligible pool, so both enrollment counts must be inspected before snapshot promotion. |
| PVSG train-supported predicate view | Approved closed-set task definition | The active 64-label ontology remains immutable, but the first model candidate group contains only predicates with at least one complete positive-pair frame in the frozen model-selection training subset. The exact supported and unseen labels are derived during materialization and saved in `ontology.json`. Evaluation removes unsupported assignments from mixed targets and omits records whose complete target set is unsupported. This avoids presenting independently parameterized, never-positive predicate indices as learnable zero-shot classes without adding automatic synonym mappings. |

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

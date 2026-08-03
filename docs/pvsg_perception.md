# PVSG perception and memory experiments

## Purpose

This document defines the experiments before it defines their implementation. The goal is
not to reproduce the original Tensor Brain Section 6 line by line. The goal is to test the
same central ideas on real video:

1. perception binds visual evidence to persistent identities and semantic concepts;
2. the dynamic context transports information between perceptual acts;
3. index feedback can make later perception depend on what was recognized earlier;
4. a model can use past perception to recognize entities and relationships later.

PVSG provides oracle panoptic masks and stable per-video object IDs. The experiments must
therefore be described as perception given ground-truth segmentation and tracking, not as
open-world object detection.

## Perceptual evidence

Feature extraction is independent of relation labels, experiment splits, and Tensor Brain
indices. For each video it records three flat observation tables.

### Scene observations

One whole-frame DINO feature for every decoded PVSG frame. The initial extractor uses the
final normalized DINOv3 CLS token, the encoder's explicit whole-image representation, rather
than averaging the spatial tokens:

```text
frame_index[S]
feature[S, D]
```

### Object observations

One feature for every visible tracked object. Object features are pooled over the exact
panoptic mask rather than over a bounding box. If `z[t,p]` is the DINO vector of patch `p`
and `w[o,t,p]` is the fractional mask coverage of that patch, then

\[
f_o(t)=\frac{\sum_p w_{o,t,p}z_{t,p}}{\sum_p w_{o,t,p}}.
\]

The stored table is

```text
frame_index[O]
object_id[O]
mask_area[O]
feature[O, D]
```

Bounding boxes are not the object representation. They can be reconstructed from the source
masks if a later geometric analysis needs them.

### Pair observations

For every unordered pair of objects simultaneously visible in a frame, the extractor pools a
feature over the smallest rectangle enclosing both masks. This union region deliberately
includes the participants and the space between them:

\[
B_{ij}(t)=\operatorname{bbox}(M_i(t)\cup M_j(t)).
\]

```text
frame_index[P]
object_a_id[P]          # object_a_id < object_b_id
object_b_id[P]
union_bbox_xyxy[P, 4]   # half-open original-frame coordinates
feature[P, D]
```

The union geometry is symmetric, so it is stored once. Subject/object direction comes from a
task target. The exact union box is retained because it defines the pooled evidence and is
useful for qualitative inspection.

All tables store explicit frame indices. The manifest records the DINO model and weights,
feature dimension, patch size, preprocessing, dtype, frame rate, coordinate convention, and
source-data identity. Raw features are stored without experiment-specific normalization.

Before full extraction, a small-cohort audit must report object and pair counts, estimated
storage, objects lost at patch resolution, and the fraction of annotated positive relations
whose two participants have valid features.

The initial feature snapshot pins `facebook/dinov3-vitb16-pretrain-lvd1689m` and its exact Hub
revision. Frames are resized without cropping or a square warp to an approximately 448-pixel
long edge; both dimensions are rounded to multiples of the 16-pixel patch size, introducing
only the small aspect-ratio approximation required by the patch grid.
Object weights are exact overlaps between source-mask pixels and these patch cells, including
for non-divisible source dimensions. Features are stored in float16 after float32 pooling.

The initial experiment loads each addressed cached vector as float32 and applies
`sqrt(D) * L2-normalize(x)` with epsilon `1e-12`, where `D` is the feature dimension. Every
nonzero input therefore has component RMS one. This preserves DINO direction and removes the
systematic norm difference between CLS and pooled-region evidence while placing the input at a
meaningful pre-CBS scale. Raw caches remain unchanged. Mask area and union-box coordinates are
analysis/provenance fields, not numerical inputs to the initial model. Six source-defective
videos are explicitly excluded by `experiments/pvsg/exclusions.json`; task materialization
accepts no other missing or invalid artifact.

The complete data path is:

```text
PVSG videos + panoptic masks ──DINOv3 extraction──> per-video float16 feature tables
PVSG annotations + feature rows ──materialization──> audited protocol JSONL manifests
manifest row + addressed feature rows ──float32 RMS normalization──> model evidence
                                                                   scene / subject / object / union
```

Training visits videos in shuffled blocks and shuffles rows within each video. The loader caches
the small set of current raw video tables and normalizes only the addressed rows, avoiding a
whole-video float32 conversion for every randomly selected example.

## Three tasks

### 1. Object perception and later recognition

An object observation supplies scene and mask-pooled object evidence. Training targets the
actual identity `(video_id, object_id)`. Official PVSG category and semantic-hierarchy targets
are added as named experimental conditions.

This stream defines what it means to have observed an object. It is not restricted to frames
where that object participates in an annotated relation.

Object observations are the primary evidence stream for identity, category, semantic-memory,
episodic-memory, and re-identification experiments. Pair completeness never filters this stream.
The frozen snapshot also contains one record for every retained video frame, including frames
with no mask-visible object, with the visible object IDs and feature rows ordered by object ID.
This supports explicit sequential-object schedules and makes disappearance and reappearance
observable without fabricating an object feature during an occlusion.

### 2. Positive-pair predicate recognition

For a directed pair `(s,o)` at frame `t`, let `Y_t(s,o)` be the complete set of active
annotated predicates. The initial Section 6 relation task contains records for which

\[
Y_t(s,o)\ne\varnothing.
\]

Its evidence is

\[
x_t^{s,o}=\left(f_{scene}(t), f_s(t), f_o(t), f_{\{s,o\}}(t)\right),
\]

and its target is one multi-hot vector over the active predicate vocabulary:

\[
y_{t,p}^{s,o}=\mathbf 1[p\in Y_t(s,o)].
\]

PVSG declares 57 predicates but uses seven additional predicates in the retained annotations.
The initial active vocabulary therefore contains 64 labels: the declared order followed by the
additional labels in deterministic alphabetical order. No string similarity or automatic
semantic collapsing is applied. Simultaneous predicates for the same directed pair and frame
remain one multi-hot target. Records lacking either visible participant or their union evidence
are retained with evidence flags but excluded from the initial complete-perception views.

This complete-evidence restriction belongs only to the initial four-input relation task. It is
not a dataset-wide filter. Later missing-evidence experiments may select canonical incomplete
records, but must define an explicit evidence mask and state-update rule; absent tensors are not
silently padded, interpolated, or replaced by prior observations.

The initial closed-set model view is narrower than the immutable ontology. It contains only
predicates with at least one complete positive-pair frame in the frozen model-selection training
subset. The exact supported and unseen labels are stored in `ontology.json`; they are derived
only after the development videos are reserved. Evaluation removes unsupported assignments from
mixed targets and omits records whose targets are entirely unsupported; each run stores the exact
counts. This is not a claim that those labels are semantically invalid. Their independent index
embeddings simply have no positive training evidence, so evaluating them as ordinary zero-shot
classes would not test the initial Tensor Brain model.

This is predicate recognition conditioned on an oracle positive pair and frame. It does not
test whether a relationship exists.

### 3. Relationship prediction and anticipation

The all-visible pair table also permits later tasks that include pairs with no active annotated
predicate. Current relationship prediction uses all ordered pairs visible at `t`. Relationship
anticipation uses only candidates available at the observation time and predicts a future
target:

\[
\mathcal C_t\longrightarrow Y_{t+\Delta}(s,o),
\qquad (s,o)\in\mathcal C_t.
\]

Candidate construction must never use visibility or relation labels from `t+Delta`. Results
must distinguish relation onset, persistence, and cessation; otherwise persistence can dominate
the headline number.

PVSG's published relation pipeline pairs feature tubes using ground-truth relations. It is not
yet established that every unannotated visible pair is a verified negative. All-pair current
prediction and anticipation may therefore be positive-unlabeled problems. We will audit this
before assigning ordinary negative labels. This uncertainty does not affect positive-pair
predicate recognition and does not block the first experiment.

## Section 6 perceptual program

The original paper processes scene, subject, object, and predicate evidence in sequence. It
compares independent bottom-up decoding with dynamic-context transport and index feedback.
PVSG lets us reproduce that capability while replacing VGG boxes with scene, mask, and union
DINO evidence.

### Tensor Brain schedule

The perception-only baseline has no episodic index. Scene evidence initializes the
representation; the Tensor Brain then moves through subject, object, and predicate windows:

```text
scene input
  -> evolve
subject mask input -> identity readout -> identity feedback -> semantic readouts
  -> evolve
object mask input  -> identity readout -> identity feedback -> semantic readouts
  -> evolve
union-box input    -> multi-label predicate readout
```

This baseline separates the contribution of memory from perception and does not justify
omitting scene evidence. Controlled memory conditions add episodic indices as described below.

### Episode formation and episodic-index creation

The first episodic-memory conditions use a predetermined `relation_state_episode`: a maximal,
nonempty video interval during which the complete active set of `(subject identity, predicate,
object identity)` facts is unchanged. Overlapping predicate spans are therefore partitioned
into non-overlapping relational states rather than treated as overlapping episodes. The episode
kind describes the resulting interval; separate provenance records whether its boundaries came
from annotations, a fixed-window control, or a learned policy. Only information from the
observation prefix may create an episode used by a causal evaluation protocol.

Predetermined episodes deliberately isolate whether the Tensor Brain can store, retrieve, and
use an episodic index from the harder question of when a new episode should exist. A later
experimental extension makes boundary detection and memory creation learnable. Its policy may
use prediction error, changes in `q` or dynamic context `h`, and the expected utility of later
retrieval to decide when to start or end an episode, allocate a new index, update an existing
one, or decline to write. It should be compared with annotation-derived relation-state episodes
and fixed-duration windows while controlling for the number and duration of stored memories.
Success is measured by downstream recall, recognition, anticipation, or decision quality, not
only agreement with human or annotation boundaries.

### First comparison

Only two models need to be trained for the first paper-parallel result:

1. **P-Direct** independently decodes identity and semantics from each object feature and
   predicates from the union feature. It has no dynamic context and no index feedback.
2. **Integral TB** trains with differentiable expected identity feedback. The same checkpoint
   is evaluated as:
   - **P-SA**, retaining expected identity feedback;
   - **P-Samp**, replacing each expected feedback vector with the embedding of the
     highest-scoring identity.

For identity candidates `I`,

\[
z_k=a_k^\top\sigma(q),\qquad
\pi_k=\operatorname{softmax}(z)_k.
\]

P-SA uses

\[
q'=q+\sum_{k\in I}\pi_k a_k,
\]

whereas P-Samp uses

\[
k^*=\arg\max_{k\in I}z_k,\qquad q'=q+a_{k^*}.
\]

P-Samp is an inference condition, not a separately trained model.

P-Direct is faithful to the broad comparison in the paper, but it is not a matched-information
causal control: its predicate decision sees the union, whereas the TB transports scene,
subject, and object evidence into that decision. Before claiming that a gain is specifically
caused by TB recurrence or feedback, add:

- a flat fusion model that receives the same four feature sources without TB operations;
- a separately trained sequential model with dynamic context but no index feedback.

These controls follow the first end-to-end validation; they do not complicate the first runner.

### Training targets

The identity-only condition uses actual `(video_id, object_id)` indices and predicate indices.
For a positive-pair batch, the initial probabilistic objective is

\[
\mathcal L_{pair}
=\frac1B\sum_b\sum_p\operatorname{BCE}(z_{b,p},y_{b,p})
+\operatorname{CE}(z_S,s)
+\operatorname{CE}(z_O,o)
+\sum_{g\in\mathcal G}\left[
  \operatorname{CE}(z_{S,g},c_{S,g})
  +\operatorname{CE}(z_{O,g},c_{O,g})
\right].
\]

Summing the predicate Bernoulli terms, rather than silently averaging them over the vocabulary,
makes this the negative log-likelihood of the multi-label target. The runner logs both the
summed predicate loss and its per-label mean and both identity losses; the scale trace records
parameter gradients at the same checkpoints. `G` is empty for the identity-only condition,
contains the official source category for that condition, and contains the four reviewed
hierarchy levels for the hierarchy condition.
An unavailable reviewed path uses the ordinary cross-entropy ignore index for that term only.
No configurable loss weights are introduced until these diagnostics show a concrete problem.

The overfit gate uses only pair batches so that it exercises the complete sequential schedule.
The subsequent full experiment combines those pair rows with object-observation batches, which
train identity and semantic readouts from every eligible visible-object exposure. It reports the
exact number of examples contributed by both streams; neither is silently resampled.

### Semantic conditions

Unary semantic decoding is part of the original Tensor Brain perception experiment, not an
unrelated downstream probe. In Algorithm 1, the subject identity is read, fed back into `q`, and
then decoded into a unary label before evolution continues. Table 5 reports entity, B-Class,
P-Class, G-Class, age, color, and activity readouts. The initial PVSG models therefore always
support named category candidate groups. P-Direct scores each group from the corresponding object
feature alone; Integral TB scores it from the post-identity-feedback subject or object state and
does not feed the resulting category prediction back.

The availability of a readout is distinct from applying its loss. Semantic supervision changes
the shared `A` and can change later states through identity learning, so checkpoints use explicit
conditions rather than silently enabling every target:

1. actual identity only;
2. identity plus the official PVSG category;
3. identity plus the manually reviewed, versioned four-level
   [PVSG object hierarchy](pvsg_object_hierarchy.md).

The reviewed fine, basic, and coarse levels are the closest PVSG analogues of the paper's
B-Class, P-Class, and G-Class; the domain level is an additional abstraction. Applying the same
post-feedback unary readout to both directed-pair participants is the symmetric extension needed
to evaluate both tracked entities. Records whose reviewed path is intentionally unresolved remain
eligible for identity, source-category, and predicate losses; only their unavailable hierarchy
losses are masked. The hierarchy condition is the paper-motivated primary semantic condition,
while identity-only and official-category checkpoints are necessary ablations. Frozen probes on
the identity-only model can later test whether semantic structure emerged without direct
supervision.

### Tiny-data overfit gate

Before any comparison or full-data training, `experiments/pvsg/overfit.py` loads the first 200
rows of the immutable, video-major `heldout_video/train_pairs.jsonl` manifest into one fixed
batch. A manifest prefix is an I/O-efficient deterministic diagnostic selection, not an estimate
of population performance. Its identity candidate set contains exactly the entities occurring in
that batch; predicate candidates remain the complete train-supported closed set. The default
condition is Integral TB with the original recurrent dynamic context and reviewed hierarchy
supervision. Identity-only and official-source-category conditions are explicit alternatives.

The optimizer is ordinary Adam over the complete model. The gate succeeds only when every
enabled identity, predicate, and available category target is correct for every example and the
unweighted total objective is at most `0.01`. Initialization, fixed early checkpoints, and the
final state are diagnosed on this same batch. This runner intentionally contains no baseline
composer, callback system, full-dataset sampler, or validation protocol; those belong to the
subsequent experiment after the computational graph has passed the overfit gate.

Each run directory is immutable and contains:

- `config.json`, `vocabulary.json`, and `batch.jsonl` for exact reconstruction;
- `training_trace.jsonl` and `scale_trace.jsonl` for optimization and scale analysis;
- `checkpoint.pt`, `predictions.pt`, and `result.json` for the final P-SA checkpoint and its
  evaluation-only P-Samp readout.

### Input-scale and initialization trace

The initial DINO mapping is a deliberate modernization rather than a paper reproduction. Every
tiny overfit and full training run must therefore evaluate it on a fixed, versioned diagnostic
batch and write `scale_trace.jsonl`. Each row identifies the run, checkpoint step, evidence
window, candidate group, and feedback mode. It records:

- input-drive and pre-CBS `q` norm, component RMS, mean, and standard deviation after integration,
  index feedback, and evolution;
- `gamma = sigmoid(q)` quantiles and the fractions below `0.01` and above `0.99`;
- `A` column-norm and `a0` summaries for identity, category, and predicate groups;
- the effective neutral-state score offset
  `a0[k] + 0.5 * sum_i A[i, k]`, the centered data-dependent score
  `A[:, k]^T (gamma - 0.5)`, and their dispersion ratio;
- P-SA expected-feedback and winner-feedback norms relative to both the current input drive and
  pre-feedback `q`, together with attention entropy and maximum probability;
- gradient norms for `A`, `a0`, each evolution parameter, and any input or feedback gate present
  in that named condition.

The diagnostic batch stores its native video/frame/object addresses and is reused across steps,
models, seeds, and normalization ablations. Required capture points are initialization before any
optimizer step, fixed early steps, and the final checkpoint; the exact cadence belongs in
`config.json`. Raw per-example tensors may be retained in `predictions.pt`, while
`scale_trace.jsonl` contains aggregation-ready scalar summaries. Any future input mapping,
centering, gate, or initialization change receives a new condition name and ledger entry and is
compared on the same raw cached DINO features rather than replacing the feature snapshot.

VLM-derived semantics from PVSG descriptions and captions form a later, separately named
modality. Because language can reveal relations and future events, every derived feature must
have explicit observation-time or training-only provenance.

The versioned [semantic property and relation inventory](pvsg_semantic_inventory.md) defines
the additional candidate space for controlled memory experiments. It does not fabricate facts:
visual properties require explicit instance labels, category-typical affordances and risk
require reviewed semantic facts, and ownership or kinship requires identity-to-identity triples.
Transient state belongs to episodic conditions; stable category knowledge belongs to semantic
memory conditions.

## Evaluation protocols

Every split is materialized as a saved manifest of native video, frame, and object IDs.
Within each PVSG source, a deterministic hash reserves 15% of retained official-training videos
for development; the remaining 85% are model-selection training data. This is a video split, not
a row split. The exact membership and hash salt are frozen in `splits.json`. The official PVSG
validation videos are untouched and retained for final held-out-video evaluation.

### `heldout_video`

Train on model-selection training videos, select hyperparameters on development videos, and
evaluate once on official-validation videos. All evaluation identities are novel. Identity
classification is therefore not a closed-set metric and is reported as unavailable; predicate
and semantic results measure generalization, while identity attention is analysed as retrieval
of analogous training identities.

### `blocked`

Each selected training video is divided into the first 45% as an observation prefix, the middle
10% as an embargo, and the final 45% as an evaluation suffix. Object observations and
positive-pair records from the prefix train the model. Evaluation uses later records only when
both participants were observed in the prefix. Distance from each evaluation frame to the final
training exposure is stored and stratified.

This protocol is causal: evaluation never includes frames earlier than the training encounter.

### `fewshot`

Few-shot evaluation is an explicit re-identification experiment, not a row split. A base model
is trained on the model-selection training videos. Development-video identities provide the
model-selection counterpart and official-validation identities remain the final evaluation. Each
eligible identity contributes its earliest ten mask-visible observations separated by at least
five frames (one second at PVSG's asserted 5 FPS), using identity supervision only. The same
identity pool and queries are evaluated at `k in {1, 3, 5, 10}` by exposing only the first `k`
ranked supports. The snapshot records the complete support sequence and temporal span. Adaptation
changes the new identity embedding only by default. Queries begin at least 25 frames (five
seconds) after the tenth support observation. A pair query begins only after both participants
have been enrolled and passed this embargo.

This definition prevents later predicate records from giving an identity more than `k`
exposures. It is intentionally implemented after the held-out and blocked pipelines are
validated.

Random frame splitting is only a leakage diagnostic and is never a headline result.

## Metrics and saved evidence

For positive-pair predicate recognition, report:

- predicate negative log-likelihood and per-label BCE;
- micro, macro, and per-example Recall@K;
- Precision@K and mean/per-predicate average precision;
- per-predicate support;
- seen versus unseen category-predicate-category triples;
- known-identity Hits@1/5/10 and MRR where defined;
- temporal-distance strata for blocked and few-shot queries;
- P-SA/P-Samp entropy, hard/soft agreement, and paired per-example differences;
- predicate-frequency and category-pair priors.

Each run writes:

```text
config.json
provenance.json
split.json
checkpoint.pt
results.json
predictions.pt
scale_trace.jsonl
tensorboard/
```

`predictions.pt` retains frame and object addresses, union boxes, complete predicate targets
and logits, compact identity retrievals, and condition labels. Quantitative tables and
qualitative overlays are generated later from these artifacts; plotting is not part of the
training loop.

## Implementation order

1. Implement the task-neutral extractor and one-video audit.
2. Extract a small cohort and inspect coverage, pair count, and storage.
3. Implement transparent annotation expansion and target construction.
4. Verify dataset and split statistics before adding a model.
5. Overfit a tiny object dataset, then a tiny complete-positive-pair dataset.
6. Run object identity/category and input-ablation controls on `heldout_video`.
7. Run the causal `blocked` object protocol and explicit few-shot re-identification.
8. Add episodic recall and sequential-object schedules over the canonical frame stream.
9. Run P-Direct and one integral TB checkpoint as P-SA/P-Samp on complete relation evidence.
10. Add category and reviewed semantic-memory conditions.
11. Add matched-information and no-feedback controls.
12. Audit PVSG negatives before implementing all-pair relationship prediction.
13. Implement pair-conditioned and then scene-wide relationship anticipation.

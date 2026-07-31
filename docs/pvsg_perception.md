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

The initial experiment loads every cached vector as float32 and independently applies L2
normalization with epsilon `1e-12`. Raw caches remain unchanged. Mask area and union-box
coordinates are analysis/provenance fields, not numerical inputs to the initial model. Six
source-defective videos are explicitly excluded by `experiments/pvsg/exclusions.json`; task
materialization accepts no other missing or invalid artifact.

## Three tasks

### 1. Object perception and later recognition

An object observation supplies scene and mask-pooled object evidence. Training targets the
actual identity `(video_id, object_id)`. Official PVSG category and semantic-hierarchy targets
are added as named experimental conditions.

This stream defines what it means to have observed an object. It is not restricted to frames
where that object participates in an annotated relation.

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

The initial closed-set model view is narrower than the immutable ontology. It contains only
predicates with at least one complete positive-pair frame in the official training split. For
`section6-v1`, this excludes `grabbing`, `riding`, `going down`, and `squeezing`. Evaluation
removes those assignments from mixed targets and omits records whose targets are entirely
excluded; the exact record and assignment counts are saved in each run's `split.json`. This is
not a claim that the labels are semantically invalid. Their independent index embeddings simply
have no positive training evidence, so evaluating them as ordinary zero-shot classes would not
test the initial Tensor Brain model.

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
+\operatorname{CE}(z_O,o).
\]

Summing the predicate Bernoulli terms, rather than silently averaging them over the vocabulary,
makes this the negative log-likelihood of the multi-label target. The runner logs both the
summed predicate loss and its per-label mean, both identity losses, and their gradient norms in
the pilot. No configurable loss weights are introduced until these diagnostics show a concrete
problem.

Object-observation batches train identity and later semantic readouts from every eligible
visible-object exposure. Pair batches train the full sequential schedule. The exact number of
examples contributed by both streams is reported; neither is silently resampled in the first
baseline.

### Semantic conditions

Semantic supervision is a central experiment, introduced in controlled stages:

1. actual identity only;
2. identity plus the official PVSG category;
3. identity plus the manually reviewed, versioned four-level
   [PVSG object hierarchy](pvsg_object_hierarchy.md).

Semantic targets are read from the post-feedback object state, as in the paper's unary
readouts. They do not replace identity feedback. Frozen probes on the identity-only model can
test whether semantic structure emerged without direct supervision.

VLM-derived semantics from PVSG descriptions and captions form a later, separately named
modality. Because language can reveal relations and future events, every derived feature must
have explicit observation-time or training-only provenance.

## Evaluation protocols

Every split is materialized as a saved manifest of native video, frame, and object IDs.
Hyperparameters are selected on a development subset of official training videos. The official
PVSG validation videos are retained for final held-out-video evaluation.

### `heldout_video`

Train on training videos and evaluate on held-out videos. All evaluation identities are novel.
Identity classification is therefore not a closed-set metric and is reported as unavailable;
predicate and semantic results measure generalization, while identity attention is analysed as
retrieval of analogous training identities.

### `blocked`

Each selected training video is divided into the first 45% as an observation prefix, the middle
10% as an embargo, and the final 45% as an evaluation suffix. Object observations and
positive-pair records from the prefix train the model. Evaluation uses later records only when
both participants were observed in the prefix. Distance from each evaluation frame to the final
training exposure is stored and stratified.

This protocol is causal: evaluation never includes frames earlier than the training encounter.

### `fewshot`

Few-shot evaluation is an explicit re-identification experiment, not a row split. A base model
is trained on official-training videos. For each official-validation video, a new identity index
is enrolled from its first five mask-visible observations, using identity supervision only.
Adaptation changes the new identity embedding only by default. Queries begin at least 25 frames
(five seconds at PVSG's 5 FPS) after the fifth support observation. A pair query begins only
after both participants have been enrolled and passed this embargo.

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
5. Overfit tiny object and positive-pair datasets.
6. Run P-Direct and one integral TB checkpoint as P-SA/P-Samp on `heldout_video`.
7. Add the causal `blocked` protocol, then explicit few-shot enrollment.
8. Add category and reviewed WordNet conditions.
9. Add matched-information and no-feedback controls.
10. Audit PVSG negatives before implementing all-pair relationship prediction.
11. Implement pair-conditioned and then scene-wide relationship anticipation.

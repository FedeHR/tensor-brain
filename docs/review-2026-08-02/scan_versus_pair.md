# Multi-object scan versus subject → object → predicate

## The crux: optional versus necessary dependency

This is the argument that decides it, and it is not the one I used earlier.

- **Multi-object scan.** Recognizing `chair₃` does not *require* having recognized `person₁`. Context
  is merely helpful — a scene prior. The dependency is **optional**, so the effect size is whatever
  scene-level priors happen to be worth, and a null result is uninformative.
- **Subject → object → predicate.** The predicate decision **cannot** be made without information
  from both entity windows. And by the one-brain hypothesis the representation layer cannot hold two
  embeddings concatenated, so subject identity *must* be transported through `q`/`h` to be available
  at the predicate window. The dependency is **necessary**, and its failure is observable.

A necessary dependency is a much sharper test of transport than an optional one. This is the
strongest single argument, and it favours the pair schedule.

## Where my earlier "information-matched" argument does not apply

I justified the scan partly on the grounds that the pair schedule confounds information with
mechanism. That is true of the **P-Direct baseline** (the predicate readout sees the union region,
which the entity windows did not) — it was the paper's own Table 4 confound.

It is **not** true of the feedback contrast. M2 (evolution, no feedback) and M3 (evolution +
feedback) both see identical evidence at every window in either schedule. So for the question you
actually care about — does index feedback do anything — the pair schedule is equally matched. That
argument does not distinguish them.

## Comparison

| | Multi-object scan | Subject → object → predicate |
|---|---|---|
| Dependency tested | optional (scene prior) | **necessary (binding)** |
| Contrast shape | **10-point accumulation curve** | 2 entity windows, one predicate readout |
| Paper comparability | none — a schedule the paper never ran | **direct parallel to Tables 4–5** |
| Target richness | single-label category / identity | **multi-label predicates, richer headroom** |
| Role asymmetry testable | no | **yes — and it is the sharpest test available** |
| Shuffled control | clean (objects from other frames) | weaker (destroys the predicate target) |
| Data volume | 1.5 M, complete, 2.3 GB, fits in RAM | 515 K complete joins, 12.8 GB |
| Open data questions | none | 26.7 % incomplete joins; `on` = 36 % of assignments |
| **Code needed today** | new schedule, new dataset over `frames.jsonl`, new collate | **already built and tested** |

That last row is decisive on timing. `PVSGPairDataset`, `IntegralTB.forward`, `build_pair_targets`,
`pair_losses`, `pair_metrics` and `overfit.py` all exist and are covered by tests — the ~700 lines I
earlier called dormant. The pair schedule costs *less* new code than the scan.

## Recommendation

**Run the pair schedule as the centerpiece. Keep the scan as a supporting curve if time allows.**

The scan's one genuine advantage is the graded accumulation curve, and a curve is worth a lot. But
it measures an optional dependency with an unknown and probably small effect, it is not the paper's
experiment, and it needs building. The pair schedule tests a necessary dependency, is directly
comparable to the published tables, and is ready now.

The mask-area and recency stratifications apply unchanged to either.

## Two experiments the pair schedule enables and the scan cannot

### 1. Role swap — the sharpest available test of the one-brain hypothesis

The original paper states that the nonlinearity of `g(·)` breaks subject/object symmetry: "It is
clear which entity is the subject and which one is the object." That has never been measured.

Take a directed positive pair `(s, o)` with predicate set `Y`. Evaluate the same checkpoint on
`(o, s)` — identical features, identical union region, only the window order changed. Then:

- if the prediction is **invariant** to the swap, the dynamic context is not encoding role, and the
  claim is in trouble;
- if it shifts toward the **inverse relation** where one exists in the ontology, that is strong
  positive evidence for role-encoding transport;
- if it merely **degrades**, report the magnitude — that is still evidence of role sensitivity.

Cheap (evaluation-only, one extra pass), falsifiable in both directions, and no analogue exists in
the scan.

### 2. Identity-substitution intervention, where it should bite hardest

Replace the subject's fed-back identity with a wrong one, stratified by distance in the reviewed
hierarchy (wrong individual same fine class → wrong domain), and measure predicate degradation. This
is experiment **E-D** run in the setting where the symbolic bottleneck is load-bearing rather than
decorative. In the scan there is no downstream decision that *requires* the substituted symbol, so
the intervention has much less to act on.

## One confound to fix if you do run the scan

Scan order is currently ascending `object_id`, and PVSG object IDs are plausibly assigned in
annotation order, which tends to correlate with salience and first appearance. If so, "position *n*"
is contaminated by a systematic difference in what kind of object appears at that position, and the
accumulation curve would partly measure that instead of transport. **Randomize the scan order per
epoch** and report the fixed-order result separately. Cheap, and it removes the objection entirely.

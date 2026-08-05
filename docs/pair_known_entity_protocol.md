# The known-entity pair protocol: a VRD-EX analogue on real video

This is the experiment `docs/index_feedback_evidence.md` §7 lists as the first thing
that would settle the identity-feedback question. That analysis established two facts:

1. The corrected pair runs found identity feedback null, but they ran in the regime
   where **the paper itself reports it as near-null** — novel entities, VRD-E, where
   P-SA beats P-Direct by +0.12 points on unary labels.
2. The regime where **the paper reports feedback as decisive** — known entities,
   VRD-EX, where P-Samp gains +6.37 points — had not been tested here at all.

On held-out video the null is not just expected, it is structural: PVSG identities are
video-scoped (`identity_name(source, video_id, object_id)`), so reserving whole videos
makes every one of the 2,474 identity candidates wrong by construction. The identity
bank cannot carry information about the entity in front of it, and the validation trace
correspondingly reports no identity accuracy at all.

The blocked protocol removes exactly that obstacle and changes nothing else.

## What runs

One `--protocol blocked` run trains on the first 45% of the frames of each training
video and reports **the same checkpoint** on two evaluation sets:

| set | manifest | entities at evaluation | VRD analogue |
|---|---|---|---|
| `development` | `heldout_video/development_pairs.jsonl` | novel — different videos | **VRD-E** |
| `blocked` | `blocked/evaluation_pairs.jsonl` | **same instances**, later frames | **VRD-EX** |

Both numbers come from one training run, so the novel-versus-known contrast is a
within-checkpoint comparison rather than a comparison across two training runs. This
mirrors what `object_experiment.py` already does for unary labels; the pair experiment
simply had `heldout_video/` hardcoded.

Checkpoint selection stays on the **novel-entity** set. The known-entity result is
therefore reported, never selected for.

### Blocked is a stronger VRD-EX than VRD-EX

The paper built VRD-EX by distorting copies of the training images (tb_original p.23),
so "known entity" partly meant "nearly the same pixels". Blocked evaluates genuinely
later frames of the same video, separated from the observation window by a 10% temporal
embargo (`protocols.blocked_boundary`: 45% observation / 10% embargo / 45% evaluation).
Recognizing the entity requires re-identification across appearance, pose, and
occlusion change rather than image memorization. A positive result here is a stronger
claim than the paper's; a null result here is not automatically a refutation of the
paper's, because the task is harder.

## The conditions, and what each one isolates

`cluster/pvsg/pair_known_entities.sbatch`, array of five:

| condition | index layer | dynamic context | identity feedback | isolates |
|---|:-:|:-:|:-:|---|
| `priors` | — | — | — | the floor: predicate frequency and directed category-pair counts |
| `union-only` | — | — | — | visual evidence with no index layer at all |
| `p-direct` | yes | **no** | **no** | the paper's P-Direct |
| `integral-none` | yes | yes | **no** | the one-factor control the paper never ran |
| `integral-p-sa` | yes | yes | yes | feedback; read back as P-SA *and* P-Samp |

Two differences from the paper's Table 5 row are deliberate:

- **`priors` is not optional here.** Blocked evaluates the same videos it trained on, so
  the predicate marginal at evaluation is much closer to the training marginal than it
  is on held-out video. Without the prior floor measured on *both* sets, any
  blocked-versus-development gap is uninterpretable — it could be entirely a
  distribution shift rather than anything the model does.
- **`integral-none` separates the two things P-Direct removes.** Per tb_original §6.3,
  P-Direct removes top-down index feedback *and* the dynamic context layer
  simultaneously. The paper attributes its large binary-label gap to the dynamic
  context layer three separate times and never to index feedback. So
  `p-direct → integral-none` is the dynamic-context effect and
  `integral-none → integral-p-sa` is the feedback effect, measured separately.

## What gets reported that could not be reported before

Where the protocol enrolls the evaluated entities, `evaluate_pairs` additionally reports:

- **Identity accuracy** for both participants (observation-, identity-, and
  video-macro), plus `accuracy/identity_pair_exact`, the rate at which both are
  recognized. On `development` these keys are absent, which is the correct answer
  rather than a missing measurement.
- **Predicate metrics partitioned by recognition**:
  `stratum/identity_pair_correct/...` versus `stratum/identity_pair_incorrect/...`.
  Does retrieving the right entity index coincide with a better predicate?
- **Predicate and identity metrics partitioned by re-identification delay**:
  `stratum/delay/{0-2s,2-5s,5-10s,10s+}/...`, from the delay the manifest records for
  each evaluation pair. This axis has no counterpart in the original experiments,
  which had no temporal separation between enrollment and test at all.

Both partitions are conditional associations, not interventions. The recognition
partition conditions on a model output, so easy pairs plausibly drive both terms; the
delay partition conditions on a dataset property that also correlates with video length
and motion. They locate a mechanism, they do not prove one.

## Entity enrollment, and why it comes from the object manifest

A blocked evaluation pair only requires both participants to have been *mask-visible*
before the observation boundary — not to have appeared in an annotated pair. Enrolling
identity columns from training pair participants alone would leave some evaluation
entities without a column, which would quietly turn "known entities" into "a filtered
subset of known entities".

So under `--protocol blocked` the identity group is enrolled from
`blocked/train_objects.jsonl`, the observation-window object manifest. This is also the
faithful reading of the index layer: it holds a column per entity the system has
encountered, not per entity it has reasoned about. The consequences are recorded rather
than hidden:

- `config.json` reports total identity columns and how many are supervised by a
  training pair. Columns enrolled but never pair-supervised sit near initialization and
  act as distractors in the identity readout — a real cost, reported as a number.
- Every training-pair participant must be enrolled, or the run refuses to start.
- Every entity in a known-entity evaluation set must own a column, or the run refuses
  to start. The known-entity claim holds for the whole set or the run does not happen.

`--protocol heldout_video` keeps the original rule (enroll exactly the training-pair
participants) so it stays comparable with the four completed corrected pair runs.

## Pre-registered interpretation

Read `integral-p-sa` on `blocked` against `integral-none` on `blocked`, with the
`development` column of the same runs as the novel-entity control.

| observation | reading |
|---|---|
| identity accuracy on `blocked` is near chance | the bank does not re-identify across the embargo. Feedback has nothing to transport, and no feedback result is interpretable until this is fixed. **Check this first.** |
| identity accuracy is high, and `integral-p-sa` > `integral-none` on `blocked` but not on `development` | the paper's VRD-EX result transfers. Feedback matters exactly when the retrieved index can be correct, which is the mechanism claim. |
| identity accuracy is high but feedback is still null on `blocked` | a genuine negative result about the pathway, no longer explainable by "the identity bank cannot be right". This is the outcome that would justify redirecting effort to the decoder / growable index layer work. |
| `priors` gains as much from `development → blocked` as the models do | the blocked gain is distribution shift, not memory. Report the prior-relative delta, not the absolute. |
| `p-direct` ≈ `integral-none` on both sets | the dynamic context layer is not doing the work the paper attributes it. Worth stating: the old pre-correction run gave +1.70 pp against the paper's +15.16 pp. |
| feedback helps only in `stratum/delay/0-2s` | the memory is real but short-lived — a decay result the original experiments could not have produced. |

The first row is a genuine possibility and it is the reason identity accuracy is now
reported. A near-null feedback effect on top of a broken re-identifier would look
exactly like the existing null while meaning something completely different.

## Interaction with the category-feedback run

`pair_category_feedback.sbatch` (the other branch of §7) is orthogonal to this one and
composes with it: `integral-cat-sa` and `integral-id-cat-sa` accept `--protocol blocked`
unchanged. The interesting cell, once both have landed, is whether category feedback
and known-entity identity feedback add or overlap — the complementarity diagnostic says
categories carry the predicate signal, so identity feedback earning anything *on top of*
category feedback would be the strongest available evidence for the index layer.

## Running it

```bash
sbatch cluster/pvsg/pair_known_entities.sbatch                          # blocked, 5 conditions
PAIR_PROTOCOL=heldout_video sbatch cluster/pvsg/pair_known_entities.sbatch  # novel-entity row
```

The second form is what fills the corrected `p-direct` cell that both existing pair
arrays are missing (`index_feedback_evidence.md` §6a), giving the complete 2×3 table of
{P-Direct, no-feedback, P-SA/P-Samp} × {novel, known} entities from one runner.

Manifests already exist — `blocked/train_pairs.jsonl`,
`blocked/evaluation_pairs.jsonl`, and `blocked/train_objects.jsonl` are all in the
frozen snapshot (`materialize.py` `JSONL_PATHS`). No re-materialization is required.

# Review of the `materialize.py` / `protocols.py` changes — snapshot v2 readiness

Scope: the data/manifest side only. Model and training issues (S1–S3, S11) are untouched and out
of scope here.

## Verdict

**Not yet — three cheap must-fixes first, then go.** Two of them would cost you a full cluster
rerun, and one would silently corrupt a downstream contract. Everything else is optional.

## What the changes get right

All four things I flagged on the data side are properly addressed, and two of them are better than
what I proposed.

- **S6 (dev split).** `development_video_ids` is deterministic, salted **per source**
  (`_experiment_splits` composes `salt/{source}`), stratified so each source contributes, and
  guarded against consuming a whole source. `experiment_split` is carried on every record and
  `splits.json` writes the full assignment. Correct.
- **S7 (few-shot support).** The greedy `minimum_support_gap_frames` rule is the right shape.
  Measured on the legacy track data as a proxy (4,197 tracks): enrollment falls only
  **85.8 % → 80.1 %** while the median support span rises from ~1 s to **4.0 s**. That is a very
  good trade — the old "5 shots" really were one second of near-duplicates.
- **S8 (occlusion stratum).** `has_subject_evidence` / `has_object_evidence` / `has_union_evidence`
  on every canonical pair record, plus `positive_predicate_support` alongside
  `complete_predicate_support` per role. The occlusion condition in E-B is now constructible
  without regenerating anything.
- **E-A substrate.** `canonical/frames.jsonl` with `scene_row`, ascending `visible_object_ids` and
  `visible_object_rows` is exactly what the object-scan experiment needs.
- **Bonus, and a good call:** copying `object_hierarchy.json` into the snapshot, and adding
  `official_split` / `experiment_split` to every ontology identity record. Both make the snapshot
  self-contained.

## Must fix before scheduling

### M1. Hoist all cheap validation above the expensive loop

`_identity_records` (line 626) and `load_object_hierarchy` (line 671) run **after** the full
394-video pass. Both depend only on `annotation`, `source_by_video`, `official_split`,
`experiment_split` and `excluded_video_ids` — all available by line 320. A hierarchy validation
failure currently surfaces after you have loaded every feature artifact.

Add in the same pre-flight: an existence check over all non-excluded
`feature_root/videos/{source}/{video_id}.pt`. Today the first missing artifact raises at line 359,
possibly at video 380 of 400.

*Cost: minutes. Saves: one full cluster run.*

### M2. Assert `fps == 5`, or derive the seconds fields from it

`provenance.json` now hardcodes `minimum_support_gap_seconds: 1.0` and `embargo_seconds: 5.0`
(lines 766, 768), while every `*_seconds` record field divides by `manifest_row["fps"]`. If any
retained video has a different `meta.fps`, the provenance block and the record fields disagree and
nothing detects it. `prepare.py` states the videos are 5 FPS, so this is almost certainly a no-op —
which is exactly why it should be a one-line assertion rather than an assumption.

### M3. Bump `ONTOLOGY_SCHEMA_VERSION` to 2 and update the consumer

`ontology.json` changed shape — identity records gained `official_split` / `experiment_split`, and
`predicate_support` keys went from `train_complete_pair_frames` to
`{role}_positive_pair_frames` / `{role}_complete_pair_frames`. `ONTOLOGY_SCHEMA_VERSION` stayed at
1, and `experiments/pvsg/vocabulary.py:61` still asserts `schema_version == 1`. A v1 consumer will
therefore load a v2 document and silently read nothing where it expects the old keys. Bump both.

### Related, not a code fix: the closed set will change

`train_supported_predicates` is now derived from `complete_predicate_support["train"]`, which
excludes the ~15 % development videos. This is the correct definition, but it means the ledger's
frozen claim — that `grabbing`, `riding`, `going down` and `squeezing` are the excluded four — is
about `section6-v1` and must be **re-derived** from the v2 snapshot, not copied
(`docs/fidelity.md:114`).

## Should fix — needed by the experiment plan, both cheap

### S1'. There is no development counterpart for the `fewshot` protocol

The few-shot branch is gated on `else:` at line 421, i.e. official-validation videos only. So
enrollment hyperparameters — the one-shot write normalization and the feedback gate in **E-E** —
can only be tuned on the final test set.

The fix is nearly free and it is *already correct by construction*: `fewshot/base_training.json`
now points at `heldout_video/train_*`, which excludes development videos, so **development-video
identities are genuinely novel to the base model**. Emit `fewshot/development_{enrollment,
support_objects, query_objects, query_pairs}.jsonl` from `experiment_role == "development"` videos
and you have a legitimate model-selection set for enrollment at no scientific cost.

### S2'. Raise `SUPPORT_COUNT` to 10 so E-E gets a curve instead of a point

E-E asks how one-shot enrollment compares to gradient training *as a function of exposures*. With a
fixed 5, you can only subset downward to k ≤ 5, and the query set shifts with k.

Emitting 10 support frames with `support_rank` (already recorded) and defining queries relative to
the **10th** support frame lets you evaluate k ∈ {1, 3, 5, 10} on an **identical identity pool and
an identical query set** — the clean comparison. Measured yield on the proxy data:

| `support_count` | identities enrolled | median queries / identity |
|---:|---:|---:|
| 1 | 3 710 (88.4 %) | 112 |
| 3 | 3 525 (84.0 %) | 110 |
| 5 | 3 361 (80.1 %) | 110 |
| **10** | **3 017 (71.9 %)** | **100** |
| 25 | 1 813 (43.2 %) | 89 |

10 costs 8 points of enrollment and buys a four-point curve. 25 halves the pool; not worth it.

### S3'. Decide, and document, whether development videos belong in `blocked/train_*`

The blocked branches are gated on `official_split == "train"` (lines 407, 486, 568), so
development-role videos are inside `blocked/train_objects.jsonl` and `blocked/train_pairs.jsonl`.
This is not a leak — `experiment_split` is on every record, so downstream code *can* filter — but it
is undocumented and easy to forget.

Two defensible options:
1. **Leave it, and select blocked hyperparameters on `heldout_video/development`.** Hyperparameters
   transfer; blocked keeps 100 % of its training data. Cheapest, and my recommendation. Write it
   into `provenance.json["protocols"]["blocked"]` so the choice is explicit.
2. Restrict blocked to `experiment_role == "train"` so every video has exactly one role in every
   protocol. Cleaner invariant, costs 15 % of blocked training data.

Either is fine. Silence is not.

## Optional polish

- **`visible_object_mask_areas` in `canonical/frames.jsonl`.** E-B stratifies the object scan by
  mask area; without it every scan row needs a join back to the object records. One extra list per
  frame makes the scan manifest self-sufficient.
- **Count frames with zero visible objects.** `_frame_object_rows` emits them; downstream must skip
  them. Put the count in `counts` so it is visible rather than discovered.
- **Count few-shot identities rejected by the support rule**, so the 80 % figure appears in the
  snapshot rather than having to be recomputed for the write-up.
- **Record the exact per-source salt.** `splits.json` stores `DEVELOPMENT_SPLIT_SALT` but
  `_experiment_splits` actually uses `f"{salt}/{source}"`. Store the composition rule or the three
  resolved salts; this repo is otherwise precise about exactly this kind of thing.

## Confirmed working

`88 tests pass` (up from 84), so the protocol changes are covered. `MANIFEST_SCHEMA_VERSION = 2`
and the new `JSONL_PATHS` are consistent with the writers. `_frame_object_rows` is safe against
out-of-range frame indices given the extractor's own `num_frames` assertion. The
`fewshot_support_and_queries` query filter is correct at `embargo_frames = 0` as well as the
default, and interleaved non-support frames are correctly excluded from both support and query.

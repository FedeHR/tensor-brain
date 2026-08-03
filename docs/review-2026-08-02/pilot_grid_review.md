# What the pilot grid teaches, and the one-hour changes that make it an experiment

## The grid as it stands

`cluster/pvsg/object_grid.sbatch`: `evolution ∈ {original, qtb}` × `score ∈ {centered,
softplus-bias}` × `lr ∈ {1e-4, 3e-4, 1e-3}` = **12 runs**. Every cell trains `IntegralTB` with
`feedback_mode="p-sa"` hardcoded at `object_experiment.py:114`.

**What it answers well.** Two things, and both are worth having:

1. **Engineering sanity.** Does the object-first TB train on real PVSG at all — loss down, identity
   and category accuracy above chance, no non-finite losses, sane step rate.
2. **The S1 scale question, empirically and in detail.** `scale_trace.jsonl` at seven checkpoints
   gives `‖feedback‖/‖q‖`, `‖a_k‖` by group, attention entropy, and the neutral-versus-data score
   decomposition. This is the main open question in the repository and the instrumentation for it is
   genuinely good.

**What it cannot answer: anything scientific.** There is no control condition anywhere in the grid.
All 12 cells have index feedback, scene evidence, and evolution. So:

- The P-SA / P-Samp contrast at evaluation is uninterpretable, because there is no "no feedback"
  reference point to place either of them against.
- `original` vs `qtb` is a comparison of two evolution operators with nothing to say whether
  evolution matters at all.
- Blocked (known identities) vs development (novel identities) is the VRD-EX / VRD-E analogue, but
  without a control it cannot show that memory is what produces the difference.

**In one sentence: as configured, this is a hyperparameter sweep with excellent instrumentation. One
axis swap turns it into an experiment at identical cost.**

---

## Three findings that are already instrumented and that you may not be expecting

These come out of `scale_trace.jsonl` for free. Look at them first — each could change what you run
next.

1. **Does the softplus normalizer survive training?** `readout_rows` records
   `neutral_over_data_std`. At initialization it is **0.03** (versus 2.39 for `direct`). But that
   holds because `softplus(x) ≈ log 2 + x/2` is a good approximation while `‖a_k‖ ≈ 1`. As `a_k`
   grows the linearization breaks and the cancellation degrades. If the ratio drifts back above 1,
   that is a real and reportable result about the paper's normalizer, and nobody has measured it.

2. **Is P-SA attention flat?** `feedback_rows` records `normalized_entropy_mean` and
   `maximum_probability_mean` over the identity candidates. If entropy stays near 1, P-SA feedback is
   effectively the mean embedding `ā` regardless of the input, and the P-SA/P-Samp comparison is not
   well posed. This tells you *before* you interpret anything whether that comparison has content.

3. **The 41× magnitude confound, quantified.** `expected_feedback` and `winner_feedback` norms are
   recorded separately at every window. If they differ by an order of magnitude at convergence, then
   any P-SA/P-Samp difference is a scale effect and needs the gate on `attend` before it means
   anything.

---

## The changes worth making before you send

Ordered by return. Items 1–3 are the ones I would insist on; 4 is cheap insurance.

### 1. Swap the `score_mode` axis for a `feedback_mode` axis — **the whole recommendation in one line**

You are dropping to `softplus-bias` only, which frees exactly half the grid. Spend it on the control
condition rather than shrinking the array:

```bash
FEEDBACKS=(p-sa none)          # replaces SCORES
SCORES=(softplus-bias)
```

`evolution {2} × feedback {2} × lr {3}` = **12 runs. Same array size, same wall clock.** The grid
goes from "which hyperparameters" to "does the paper's central mechanism do anything on real video",
which is the M2-versus-M3 rung of the E-A ladder and the single contrast the thesis most needs.

Code: `models.py` already supports `feedback_mode="none"` (`_identity_feedback` returns `q`
unchanged with the ordinary probabilities), and it is legal during training. Required edits:

- `ObjectExperimentConfig`: add `feedback_mode: Literal["p-sa", "none"] = "p-sa"`
- `_forward` (`object_experiment.py:114`): take the mode instead of hardcoding `"p-sa"`
- `_parse_args`: `parser.add_argument("--feedback-mode", choices=("p-sa", "none"))`

This is the correct control, not `PDirect`: the `none` arm keeps scene evidence, evolution, and
identical identity supervision, and removes only the feedback injection. `PDirect` would vary
information and capacity at the same time — the exact confound flagged as **S2**.

### 2. Evaluate every checkpoint at `none` as well — one word, zero compute

`object_experiment.py:306`:

```python
for mode in ("none", "p-sa", "p-samp")
```

plus widening the `Literal` in `evaluate` and in `evaluate_objects`. `IntegralTB.forward_object`
already accepts `"none"`.

This asks a different and complementary question from item 1: *given a model trained with feedback,
does it still need feedback at inference?* If `none ≈ p-sa` on the same checkpoint, the feedback
pathway is decorative in the trained model. Free, and it makes the P-SA/P-Samp pair interpretable
by giving it a floor.

### 3. Separate individuation from video recognition — ~10 lines, closes **S4**

Identity labels are `identity:{source}/{video_id}/{object_id}`, so the candidate→video map is
parseable from the vocabulary with no manifest change:

```python
candidate_video = [label.split("/")[1] for label in vocabulary.group_labels("identity")]
```

Map to integer ids once, then per batch (`batch["video_id"]` is already in every object record):

```python
same_video = candidate_video_ids[None, :] == example_video_ids[:, None]   # [B, K]
video_hit  = candidate_video_ids[logits.argmax(-1)] == example_video_ids
within     = logits.masked_fill(~same_video, -inf).argmax(-1) == target
```

Report `accuracy/identity` (global), `accuracy/identity_video` (predicted identity is in the right
video), and `accuracy/identity_within_video`. **The third is the scientifically meaningful one.**
Without it, a high global identity number mostly measures the scene token recognizing the video, and
the individuation claim — the premise of the whole thesis — is not actually tested.

### 4. Keep one `direct` run, as an entry rather than an axis

Dropping `score_mode` entirely also drops the contrast that makes the softplus result publishable.
`direct` is the *original* TB score (no bias) and the uncorrected condition; `softplus-bias` is the
latest QTB and removes 83 % of the state-independent offset at init. Add one or two array entries at
the best-guess learning rate rather than restoring a full crossed axis. Cost: +1–2 runs.

---

## Deliberately not now

- **`PDirect` / M0–M1 rungs.** Different `forward_object` signature, and confounded as a control
  anyway. The `none` arm is the better contrast and is nearly free.
- **A gate on `attend`.** Needed before a P-SA/P-Samp *difference* can be attributed to
  expected-versus-sampled, but the diagnostics already *measure* the confound, which is enough for a
  pilot. Add it once you see the numbers.
- **The frozen-DINO nearest-centroid floor**, **prototype-versus-classifier-row cosines**, and
  **chance/prior baselines.** All three are **post-hoc on the saved checkpoint, vocabulary and
  manifests** — they need no rerun and must not delay the grid. Do them while it runs.
- **Anything episodic.** Agreed.

## One practical caution

Twelve concurrent jobs each stream per-video `.pt` artifacts from NFS through an
`lru_cache(maxsize=4)`. The compute here is trivial — roughly 27 TFLOP for a whole 10 000-step run —
so these jobs will be **I/O bound, not GPU bound**. Launch one array element first, check the step
rate for a minute, and only then release the remaining eleven. If it is slow, raise `--num-workers`
before raising anything else.

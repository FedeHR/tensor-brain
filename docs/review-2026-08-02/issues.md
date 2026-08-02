# Issue register — 2026-08-02

Ordered by how much each endangers a future experimental result, not by how hard it is to fix.
Severity: **S** = would silently corrupt or null a headline result; **H** = would confound a
result or block an experiment; **M** = would weaken a claim or waste time; **L** = correctness
hygiene and provenance.

All line references are against the working tree at commit `c546232` plus the uncommitted
`experiments/pvsg/{data,models,vocabulary}.py`.

---

## S1 — Index feedback is numerically negligible; the first comparison would produce a null result

**Where.** `src/tb/model.py:40` (init `std = state_dim**-0.5`), `experiments/pvsg/data.py:24-38`
(`normalize_dino` → `sqrt(D) · L2`), `src/tb/model.py:70` (`integrate_input`, default `μ = 1`),
`experiments/pvsg/models.py:92-98` (feedback via `attend`).

**Measured.** Run against the repository's own classes at `D = 768`:

| identity candidates `K` | `‖q‖` | `‖γ‖` | `γ` sd | `‖a_k‖` | logit sd | `‖attend feedback‖` | feedback / `‖q‖` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 27.71 | 15.05 | 0.209 | 1.000 | 0.50 | 0.1430 | 0.0052 |
| 121 | 27.71 | 15.04 | 0.208 | 1.000 | 0.47 | 0.1003 | 0.0036 |
| 1 000 | 27.71 | 14.93 | 0.208 | 1.000 | 0.54 | 0.0395 | 0.0014 |
| **7 143** (actual identity group) | 27.71 | 15.00 | 0.209 | 0.999 | 0.54 | **0.0245** | **0.00088** |
| P-Samp (single column) | 27.71 | — | — | 1.000 | — | 1.000 | **0.036** |

**Consequences, all of them bad.**

- P-SA changes `q` by **0.09 %**. `IntegralTB` in P-SA mode is, to numerical precision, the same
  model as one with no index feedback. A comparison against a no-feedback control is guaranteed to
  come out flat, and you would have no way to tell that from a genuine negative result.
- P-SA and P-Samp differ in feedback **magnitude by 41×**. Any difference between them is
  attributable to scale before it is attributable to expected-versus-sampled. The paper's central
  VRD-E/VRD-EX contrast is therefore not testable in this configuration.
- The `attend` feedback shrinks as `1/√K_eff` because it averages near-orthogonal unit columns, so
  the effect gets *worse* exactly as the identity vocabulary gets more interesting.

**Root cause, and why the fix is the faithful direction rather than a patch.** In the paper the
input map is a *learned* network — `q̃S ← u f(BBsub) + W sig(h)` with `f` the trainable DCNN
(`tb_original.txt` Algorithm 1 lines 6/17/24/29, and the DCNN reference at line 774). Its output
scale co-adapts with `‖a_k‖` during training. This repository replaced `f` with a fixed identity
map at a pinned RMS of 1, which removes exactly that adaptivity. `docs/fidelity.md:99` frames a
learned projection as the *fallback* for a dimension mismatch; in fact it is the more faithful
choice, and the identity map is the deviation.

Note also that the recent `sqrt(D)` normalization fix was correct and necessary — before it, `γ`
was pinned at `0.5 ± 0.009` and the representation layer was effectively constant. It fixed the
input side and widened the feedback side. Both need to be in range, not one.

**Recommended fix (do all three).**

1. Make `g(·)` a learned `nn.Linear(feature_dim, state_dim)` per input source, outside `src/tb`, as
   the paper's `f` is learned. This decouples `state_dim` from 768 and lets the scale be found.
2. Add `retain_gate` / `feedback_gate` to `attend`, mirroring `measure` (see **S11**). Run `β` fixed
   at several values as a controlled condition, then learned through a sigmoid parameterization —
   at which point *how strongly the model wants index feedback* becomes a reportable number rather
   than an accident of initialization.
3. Log `‖a_k‖` per index group, `‖feedback‖ / ‖q‖` per window, and the `γ` component histogram every
   epoch, for every run, permanently. This is the diagnostic that would have caught the problem.

Do **not** fix it by re-initializing `A` to `std = 1`: that gives `‖a_k‖ ≈ 27.7` against
`‖γ‖ ≈ 15` and pushes logits toward ±400. The shared bidirectional `A` is simultaneously a
logit-scale readout and a state-scale write, and the two roles want incompatible norms. That
tension is a real structural property of the paper's model and deserves a paragraph in the thesis,
not a silent patch.

---

## S2 — `PDirect` is neither information-matched nor capacity-matched

**Where.** `experiments/pvsg/models.py:14-48`.

`PDirect` receives no scene evidence (documented in its docstring), has no evolution operator, and
therefore owns only `A` and `a0`. `IntegralTB` additionally owns `V`, `W` and, for the recurrent
backend, `B`. So the comparison varies **three** things at once: information, mechanism, and
parameter count.

The paper's own P-Direct is weaker than the repo assumes — Table 5's caption says "In P-Direct,
there are no links from `n` to `q`, and `q` and `h` are independent", i.e. it retains the pipeline
and removes the feedback links. `docs/pvsg_perception.md:266-274` already identifies this and
defers the matched controls to "after the first end-to-end validation". Given that the headline
claim *is* the comparison, the matched controls are the experiment, not a follow-up.

**Fix.** Replace the two-model comparison with the ladder in `experiment_plan.md` E-A, where each
rung adds exactly one mechanism and every rung sees identical evidence per concept window.

---

## S3 — The implemented schedule omits the readout that the paper's memory claim is made of

**Where.** `experiments/pvsg/models.py:92-98` and `:100-107`.

Original Algorithm 1:

```
18  nS(s) ← a_s^T sig(q̃S)          # entity scores
19  s*    ~ softmax(nS)
20  qS    ← q̃S + a_{s*}             # entity feedback
21  nC(c) ← a_c^T sig(qS)          # <-- unary/semantic labels read AFTER feedback
22  c*    ~ softmax(nC)
```

Line 21 is Table 5. It is the mechanism behind every "memory enriches perception" statement in the
paper, including the one result that only works for known entities (the `Y/O` column: 76.54 →
94.58 from P-Direct to P-Samp on VRD-EX).

`IntegralTB` performs lines 18–20 and then discards `qS` into `evolve`. There is no `nC` readout
anywhere in the codebase. As implemented, the model can only report predicate accuracy — the one
metric for which the paper's memory story is *weakest*.

`docs/pvsg_perception.md:203` specifies the correct schedule
(`identity readout → identity feedback → semantic readouts`). The code is simply behind the doc.

**Fix.** Add post-feedback readouts over the reviewed hierarchy groups at the subject and object
windows. This is a handful of lines and it converts the model from "predicate classifier" to
"testbed for the paper's actual claim".

---

## S4 — Per-video identity labels let the scene token leak the answer

**Where.** `experiments/pvsg/records.py:131-134` — `identity:{source}/{video_id}/{object_id}`;
`experiments/pvsg/materialize.py:132-156` — one identity per (video, object), ≈7 143 columns over
394 videos, ≈18 per video.

The scene CLS token nearly determines the video. Identity therefore decomposes into "which video"
(easy, and available from the scene feature that `IntegralTB` integrates first) times "which of ~18
slots" (the actual scientific question). A global identity Hits@1 number will look impressive for
a reason that has nothing to do with individuation.

**Fix — metric, not data.**

- Report **within-video identity accuracy** (candidates restricted to that video's identities) as
  the primary number. This is the question "can it tell `person_3` from `person_5` in the same
  kitchen", and it is the one VRD could not ask.
- Report global identity accuracy as a secondary number.
- Add a **scene-only baseline** that predicts identity from the scene feature alone. The gap
  between it and the full model is the honest measure of what object evidence contributes.

---

## S5 — On `heldout_video`, identity feedback injects embeddings of the wrong individuals

**Where.** `experiments/pvsg/vocabulary.py:47-59` ("`identity_names` is the set supervised by the
protocol's training or enrollment records"), `experiments/pvsg/models.py:94-97`.

Official-validation identities are novel and therefore absent from `A`. At evaluation, `attend`
mixes *training* identity embeddings and `measure(selection="argmax")` injects one specific wrong
individual. `docs/pvsg_perception.md:331` acknowledges this and reframes it as "retrieval of
analogous training identities", which is a legitimate reading — but it means the `heldout_video`
and `blocked` protocols are running *different feedback experiments* and their P-SA/P-Samp numbers
are not comparable.

**Fix.** On `heldout_video`, make the feedback group the **category/hierarchy** group, not the
identity group — that is the honest VRD-E analogue (novel entities, semantic attention). Keep
identity feedback for `blocked` and `fewshot`, which are the VRD-EX analogue. State the split
explicitly; it is the paper's own distinction, and running it deliberately is a strength.

---

## S6 — There is no development split, so every hyperparameter choice leaks into the headline number

**Where.** `experiments/pvsg/materialize.py:35-50` (`JSONL_PATHS` contains no dev manifest);
`experiments/pvsg/materialize.py:73-81` (`_official_splits` yields only `train`/`val`).

`docs/pvsg_perception.md:327` requires "a development subset of official training videos". It does
not exist. Without it, the official validation videos are simultaneously the tuning set and the
final held-out set.

**Fix.** Deterministic partition of the official-train video IDs (hash of video ID, ~15 %) into a
dev manifest. Manifest-level regeneration only; minutes, no re-extraction.

---

## S7 — The `fewshot` protocol's "5 shots" are 5 consecutive frames, i.e. ~1 shot

**Where.** `experiments/pvsg/protocols.py:38-54` — `support = visible_frames[:support_count]`.

At PVSG's 5 FPS (`experiments/pvsg/prepare.py:5`, "feature jobs decode the original 5 FPS videos"),
the first five *visible* frames of a track are typically five consecutive frames spanning one
second. Five near-duplicate crops is not five exposures. This conflates exposure count with
appearance diversity, will understate few-shot performance, and makes any "accuracy versus number
of exposures" curve meaningless — which is precisely the curve the one-shot-write experiment needs.

**Fix.** Spread the support set across the track (evenly spaced over the track's first *X* %, or
the five most feature-diverse of the first *N* visible frames), and report the mean pairwise cosine
distance of each support set so the diversity is visible in the results. Manifest-level change.

---

## S8 — The complete-evidence filter removes exactly the frames where memory matters most

**Where.** `experiments/pvsg/materialize.py:193-196` (`has_complete_evidence`),
`experiments/pvsg/data.py:124-125` (hard raise on incomplete rows); counts from
`pvsg-audit/report.json`.

Of 702 732 positive-pair frame records, 515 402 (73.3 %) have complete evidence. The missing
components are `missing_subject` 135 440, `missing_object` 100 133, `missing_union` 187 330. A
missing participant means the relation is annotated while that participant's mask is absent —
occlusion, or annotator persistence through occlusion.

This is not missing-at-random. It is a systematic filter that discards the occlusion frames, which
are the frames where "the dynamic context transports information" and "memory supplements
perception" would actually be doing work.

**Fix.** Keep the complete-evidence view as the headline (it is the right first move), but treat
"relation annotated while a participant is unobservable" as a **first-class occlusion condition**.
It is free: the flag is already retained on every canonical record. It is one of the two genuine
information-deficit axes available at zero annotation cost (see **E-B** in the plan).

---

## S9 — Provenance claim contradicts the artifacts: 157 of 394 videos were extracted on Pascal

**Where.** `docs/fidelity.md:107` states FP16 autocast "on both Turing and newer GPUs".
`pvsg-audit/report.json` provenance groups give: GTX 1080 Ti (compute capability 6.1) **157
videos**, RTX 2080 Ti 191, RTX 2080 46.

The 1080 Ti is Pascal, not Turing. FP16 autocast runs there, but without tensor cores the kernel
and accumulation paths differ from Turing. The ledger sentence is factually wrong about its own
snapshot, and the snapshot mixes two hardware generations.

**Fix.** Do not re-extract. Re-extract ~5 videos on a second GPU family and report the cosine
agreement (expect > 0.999); then correct the ledger sentence to describe what actually happened.
Five minutes of compute buys an honest provenance paragraph.

---

## S10 — 94 of 126 source-category labels silently share a global column with hierarchy `fine` labels

**Where.** `experiments/pvsg/hierarchy.py:152-158` enforces disjointness *among* fine/basic/coarse/
domain but not against the source group built at `experiments/pvsg/vocabulary.py:132-136`.
Measured on `object_hierarchy.json`: `|source ∩ fine| = 94`, `|source ∩ basic| = |source ∩ coarse|
= |source ∩ domain| = 0`.

Because `IndexVocabulary.from_groups` gives one label one column, selecting both `source` and
`fine` means 94 columns are supervised by two different softmaxes over two different candidate sets
(126 vs 121). The intended ladder "identity → + official category → + reviewed hierarchy"
(`docs/pvsg_perception.md:301-310`) is then not a clean nesting.

**Fix.** Decide explicitly. Recommended: drop the `source` level entirely once the reviewed
hierarchy exists — the reviewed `fine` level is a strict improvement over the source categories and
keeping both buys nothing. Otherwise namespace it (`source_category:`) and accept the duplication.

---

## S11 — `attend` exposes no gates while `measure` does, so the trained path is the uncorrectable one

**Where.** `src/tb/model.py:86-104` versus `src/tb/model.py:106-153`.

`IntegralTB` trains through `attend`. `docs/fidelity.md:92` already establishes that gates are the
QTB generalization and that the core does not constrain their values. Adding
`retain_gate`/`feedback_gate` to `attend` is an API completion, not a model change, and it is the
prerequisite for the controlled `β` sweep that **S1** requires.

---

## S12 — There is no training, evaluation, or metrics code at all

**Where.** `experiments/pvsg/` contains `data.py`, `models.py`, `vocabulary.py`, and the
materialization/extraction/audit stack. A repository-wide search for `optimizer`, `backward`,
`cross_entropy` finds hits only in `experiments/evolution_overfit.py` (the XOR diagnostic) and in
tests. There is no runner, no loss, no metric, no checkpointing, no `results.json` writer.

Steps 5–11 of `docs/pvsg_perception.md:392-405` are entirely unstarted. Stale `__pycache__` entries
(`run_section6`, `section6_evaluation`, `section6_model`, `section6_overfit`) show an earlier
attempt that was removed at commit `4fe713d` and never replaced.

**This, not the data, is the schedule risk.** Three working weeks remain and the experimental
apparatus is at zero. The plan document treats building one runner, one metrics module and one
diagnostics module as the first three days for exactly this reason.

---

## S13 — Batches are drawn from a single video, which is badly non-iid against a 7 143-way softmax

**Where.** `experiments/pvsg/data.py:59-83` (`VideoBlockSampler` yields all of one video's rows
contiguously).

With batch size *B*, every batch comes from one video, so every batch contains ~18 distinct
identity labels out of 7 143 and a nearly constant scene. Gradient noise is strongly correlated and
the softmax sees a near-constant label subset per step.

The sampler exists to avoid repeatedly loading and normalizing whole video tables. **For the object
stream that problem does not exist**: 1 495 227 × 768 × fp16 = 2.3 GB, plus 0.23 GB of scene
features. The entire object stream fits in RAM, and on most GPUs. Load it once, shuffle globally.

Keep the block sampler for the pair stream (8 330 261 × 768 × fp16 = 12.8 GB), where it is the
right design.

---

## M14 — Episodic attention is omitted, so the repo's P-Direct/P-SA/P-Samp are not the paper's conditions

`tb_original.txt` §5.3: "EA is the default in all experiments on perception." Algorithm 1 lines
8–14 sample an episodic index `t*` from the scene and apply `qT ← q̃T + a_{t*}` *before* the first
evolution. `experiments/pvsg/models.py:86-89` goes straight from scene input to `evolve`.

This is a defensible simplification (there are no episodic indices yet) and
`docs/pvsg_perception.md:197` states it. But it must be said plainly in the thesis that the
reproduced conditions differ from the paper's in this respect. It is also an argument for adding an
episodic index group early: it requires no core change, only another named group in the
experiment-side vocabulary builder.

## M15 — `a0` is inside the attention softmax; the original paper's SA equation omits it

`tb_original.txt` §5.3 gives `qS ← q̃S + A softmax_β(A^T sig(q̃S))` with no bias. QTB Algorithm 2
line 821 includes `a_{0,k}`. `src/tb/model.py:84` includes it. The code follows QTB; say so once.

## M16 — `blocked` evaluation requires both participants observed in the prefix, biasing toward long stable tracks

`experiments/pvsg/materialize.py:477-480`. Report the retained fraction and the track-length
distribution of retained versus discarded pairs, or the temporal-distance curve will be measured on
an easy subpopulation.

## M17 — The sigmoid CBS gives every logit a large index-specific DC term

`γ = sig(q) ∈ (0,1)^768` with mean ≈ 0.5, so `a_k·γ ≈ 0.5·Σ_i a_ik + a_k·δ` where `‖δ‖ ≈ 5.8`. At
initialization the DC term has sd 0.50 and the discriminative term sd 0.21 — the measured total
logit sd of 0.54 is exactly `√(0.50² + 0.21²)`. `a0` must learn to cancel a per-index constant that
itself moves as `a_k` trains. Not a bug, but it is a real property of the shared-`A` design, it
explains why `a0` will carry large values, and it is worth a paragraph and a plot.

## L18 — Superseded 1.8 GB DINOv2 snapshot in `data/`

`data/pvsg/features/dinov2_vitb14/legacy_v1/` holds 289 videos of a previous encoder. It cannot be
loaded by mistake — `experiments/pvsg/data.py:50-55` rejects `schema_version != 2` and the path
layout differs — so this is disk hygiene, not a correctness risk. Delete or move it out of `data/`.

## L19 — Sampling is unseeded at the module level

`src/tb/model.py:142` uses `torch.distributions.Categorical(...).sample()` with the global RNG.
Fine, but P-Samp evaluations must set a global seed and record it in `config.json`, or the
winner-take-all/sampling comparison is not reproducible. (Note `selection="argmax"` — the condition
the paper actually uses for P-Samp — is deterministic, so this only affects `selection="sample"`.)

## L20 — Verify `meta.fps == 5` for all 394 retained videos

Every `seconds_since_*` field (`materialize.py:402-404`, `:425-426`, `:516-518`) and the
"25 frames = 5 seconds" claim (`materialize.py:616`, `docs/pvsg_perception.md:352`) assume 5 FPS.
`prepare.py` states the videos are 5 FPS, and the materializer already asserts manifest/annotation
agreement on `fps` (`materialize.py:230-243`), so this is almost certainly fine — but it is a
one-line assertion that converts "almost certainly" into "checked".

---

## Things that are right and should not be touched

Worth stating, because a long issue list can read as a verdict on the whole repository, and it is
not one.

- **The core is genuinely faithful.** I checked `index_scores`, `attend`, `measure` and both
  evolution operators line by line against Algorithm 1 of the original paper and Algorithms 1–3 of
  QTB. `attend` matches `qS ← q̃S + A softmax(A^T sig(q̃S))`. `OriginalTBDynamicContext` matches
  `h ← B sig[sig(h) + V sig(q)]`, `q̃ ← f(input) + W sig(h)`, including the additive input and the
  `h = 0` initialization. `QTBEvolution` matches `h ← sig(v0 + Vγ)`, `q ← Wh`. The predicate window
  correctly applies *no* feedback (`qP ← q̃P`, Algorithm 1 line 30). Group-wise softmax over
  candidate sets matches the paper's appendix. This is a good implementation.
- **The global-index / candidate-position separation** (`src/tb/vocabulary.py:12-57`) is a real
  correctness hazard in this model, and it is handled explicitly and tested.
- **The extraction numerics are careful.** Exact fractional mask/patch overlap including
  non-divisible dimensions, the float32 boundary-rounding guard at `features.py:177-184`, atomic
  writes, self-describing artifacts, and a contract check on reuse.
- **The provenance discipline is better than most published work.** Pinned Hub revisions, SHA-256
  of the annotation file, per-video provenance groups, an exclusion allowlist with written reasons,
  and a materializer that refuses to overwrite a snapshot.
- **84 tests pass** and they test equations, batching, candidate restriction, gradients and
  evolution state — not just plumbing.

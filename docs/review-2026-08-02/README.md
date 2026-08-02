# Repository and direction review — 2026-08-02

Three documents, written after reading `src/tb`, `experiments/pvsg`, the two papers, the
existing design docs, and the audit artifacts, and after running the test suite (84 pass) and
measuring the initialization scales directly.

| Document | Question it answers |
|---|---|
| [`issues.md`](issues.md) | What is wrong or risky, ordered by how much it endangers a result |
| [`assessment.md`](assessment.md) | Is the repo good, is the direction right, and where did the time go |
| [`experiment_plan.md`](experiment_plan.md) | What to run in the ~3 working weeks that remain |

## Relationship to the existing planning docs

There is an earlier worktree, `.claude/worktrees/pvsg-experiment-program`, containing
`pvsg_experiment_program.md`, `pvsg_core_capability_plan.md`, `tb_missing_components.md` and
`tb_feature_decoder_plan.md`. I read them and this review builds on them rather than repeating
them. Where I agree: the object-first pivot, the scale problem, the information-matched ladder, the
decoder, and the intervention harness. Where I differ:

- **Those plans are sized for 8–10 weeks.** You have three. The plan here is the ruthless cut.
- They treat a growable index vocabulary as a keystone component to build. It is not — a **reserve
  column suffix** gets you the same thing in one line (see `experiment_plan.md` §1.5).
- They defer the decoder `g⁺`. Fixing **S1** correctly hands it to you for free as the pseudo-inverse
  of the learned input map, so it should not be deferred.
- This review adds the concrete code-level audit those documents do not contain: they are
  conceptual, `issues.md` is line-referenced and measured.

## The five things that matter most

1. **Index feedback is currently a 0.09 % perturbation of the state it is supposed to inform.**
   Measured, not inferred. `‖input drive‖ = 27.71` against `‖a_k‖ ≈ 1.0`. The P-SA condition is
   numerically indistinguishable from no feedback, and P-SA vs P-Samp differ by **41× in feedback
   magnitude** before they differ in anything scientific. If you run the planned first comparison
   today, you get a null result caused by initialization. This is issue **S1** and it is the single
   highest-priority fix in the repository.

2. **The implemented `IntegralTB` never performs the readout the paper's memory claim is about.**
   Original Algorithm 1 reads the *unary/semantic* labels **after** entity feedback (lines 20–22).
   That is Table 5, and it is the whole "memory enriches perception" result. `models.py` reads only
   identity and then throws the post-feedback state into evolution. Your own design doc specifies
   the semantic readouts; the code lags the doc. Issue **S3**.

3. **The baseline is not matched.** `PDirect` has no scene input, no dynamic context, and fewer
   parameters than `IntegralTB`. The paper's own P-Direct kept the pipeline. Any gap you measure is
   attributable to information and capacity, not to feedback. Issue **S2**.

4. **You are not behind on data. You are behind on apparatus.** There is no training loop, no loss,
   no metrics, no checkpointing anywhere in `experiments/pvsg`. Steps 5–11 of your own
   implementation order are unstarted. The data is *done* — 394 videos, 1.5 M object observations,
   audited and provenanced. Issue **S12**, and it is the real schedule risk.

5. **The object stream fits in memory.** 1,495,227 × 768 × fp16 = **2.3 GB** (with scene, 2.5 GB).
   The whole object-first program is an in-RAM job with epochs measured in minutes. The pair stream
   is 12.8 GB and is not. This is a decisive practical argument for the object-first pivot that the
   existing plans do not make.

## On your worry about having over-invested in data

Partly justified, mostly not. The curation is genuinely finished and genuinely good — it is the
part of this project a reviewer cannot attack. What went wrong is **sequencing**: the semantic
inventory (49 unary values, 9 relations) and the four-level hierarchy were built before a single
model was trained, so nothing has yet told you which of them the experiments actually need. The
hierarchy will earn its place immediately. The semantic inventory may not be used at all in this
thesis, and that is fine — it is a defensible appendix.

The only further data work I recommend is **three manifest-level regenerations that reuse the
cached features and run in minutes**: a dev split, temporally spread few-shot support frames, and
frame-grouped object-scan records. No re-extraction. No new annotation. Details in
[`experiment_plan.md`](experiment_plan.md) §1.

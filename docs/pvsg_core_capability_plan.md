# Core-capability experiments: verification, plan review, and an execution order

## Purpose

This document does two things:

1. verifies the input-scale fix in the working tree against the scale problem raised in
   [the feature decoder plan](tb_feature_decoder_plan.md), and reports what it does and does not
   close;
2. reviews the core-capability experiment plan, corrects two traps, and turns it into an execution
   order with an explicit readiness assessment, since the goal is to start running experiments now
   rather than to prepare more data.

The strategic decision — test the core claims before building extensions — is the right one and is
adopted here without argument. Embodiment and index recruitment are chapters that only become
interesting if perception, identity binding, dynamic context and measurement work. This plan
covers the core; [missing components](tb_missing_components.md) stays queued behind it, with one
exception noted in Section 5.

---

## 1. Verification of the scale fix

### 1.1 What changed, and it is correct

The working tree replaces per-vector L2 normalization with `sqrt(D) * L2-normalize(x)`, so every
nonzero input has component RMS one. The fidelity ledger states the reason precisely: it removes
the source-dependent norm difference "without pinning `sigmoid(q)` near `0.5`."

Measured at `D = 768`, that is exactly what it achieves:

| | `‖drive‖` | `σ(q)` sd | `‖a_k‖` | `‖a_k‖ / ‖drive‖` |
|---|---:|---:|---:|---:|
| Previous (plain L2) | 1.00 | **0.0090** | 1.03 | **1.03** |
| Current (RMS) | 27.71 | **0.2080** | 1.03 | **0.037** |

The CBS dynamic range improves by a factor of 23. Under the previous normalization `σ(q)` was
pinned to `0.5 ± 0.009` and the representation layer was, for practical purposes, constant. **This
was a real bug and the fix is right.**

### 1.2 It is a different problem from the one raised, and it widens that one

The decoder plan's concern was the *index-embedding* side: `A` initializes at
`std = state_dim^{-1/2}`, so `‖a_k‖ ≈ 1` and `σ(a_k)` has sd `0.0093`. That initialization is
unchanged in the working tree, and the fix does not touch it.

Note the third column of the table. Before the change, drive and index column were *matched* at
norm ≈ 1 — both too small to move the CBS, but symmetric. After it, the drive is 27.7 and the index
column is still 1. So the input-side fix, which was necessary, has left an asymmetry between the
two things that write into `q`:

**index feedback is now a 3.7% perturbation of the state it is supposed to inform.**

### 1.3 The sharper consequence: P-SA and P-Samp differ in feedback magnitude by up to 70×

Expected feedback `Σ_k π_k a_k` averages near-orthogonal unit columns, so its norm shrinks toward
`1/√K_eff`. Sampled feedback injects one full column and does not shrink:

| Identity candidates `K` (near-uniform) | `‖feedback‖` | relative to `‖q‖ ≈ 27.7` |
|---:|---:|---:|
| 10 | 0.311 | `1.1 × 10⁻²` |
| 100 | 0.096 | `3.5 × 10⁻³` |
| 1,000 | 0.030 | `1.1 × 10⁻³` |
| 4,000 | 0.015 | `5.5 × 10⁻⁴` |
| **P-Samp (single column)** | **1.03** | **`3.7 × 10⁻²`** |

Early in training, before the identity distribution sharpens, P-SA's write is roughly two orders of
magnitude weaker than P-Samp's.

**Why this matters for exactly the experiments planned.** P-SA versus P-Samp is meant to test
expected attention against sampled commitment — the discrete-bottleneck question. As currently
implemented it also varies feedback magnitude by up to 70×. A difference between the two conditions
cannot presently be attributed to the approximation rather than to scale.

### 1.4 An API asymmetry that makes this harder to control

`TensorBrain.measure` exposes `retain_gate` and `feedback_gate`. `TensorBrain.attend` exposes
neither — it returns `q + feedback` unconditionally. Since `IntegralTB` trains through `attend`
(P-SA) and evaluates through `measure(selection="argmax")` (P-Samp), **the weaker of the two paths
is the one that cannot be scaled through the public API.**

The fidelity ledger already establishes that gates are the paper's own QTB generalization and that
"the core does not constrain their values: experiments may supply tensors or learned parameters."
Adding the same gate pair to `attend` is therefore an API completion consistent with the existing
approved decision, not a model change — and it is the minimal way to bring the two conditions onto
comparable footing.

### 1.5 Recommendation

**First, measure — one hour, no model change.** Log during training:

- `‖a_k‖` by index group (predicate, category, identity), as a distribution over training;
- `‖feedback‖ / ‖q‖` at each concept window, separately for P-SA and P-Samp;
- component histograms of `σ(q)` and `σ(a_k)` at convergence.

If `‖a_k‖` grows to within a factor of ~3 of `‖q‖` on its own, the issue resolves itself and
nothing needs changing. This is a genuine possibility and it should be checked before acting.

**If it does not, prefer the gate over re-initializing `A`.** Do not simply set `A`'s init to
`std = 1`. The shared matrix serves two roles with incompatible scale requirements: as a readout,
`a_k^T σ(q)` with `‖σ(q)‖ ≈ 15` needs `‖a_k‖` of order 1 to produce sane logits; as a write, it
needs `‖a_k‖` of order `‖q‖ ≈ 27.7` to move the state. Raising the init to satisfy the write role
would push logits toward ±400.

That tension is not a bug to be patched away — **it is a structural property of the shared
bidirectional `A`, and it deserves a paragraph in the thesis.** The paper's single matrix is
simultaneously a logit-scale readout and a state-scale write, and the two roles pull in opposite
directions. The clean resolution that preserves fidelity is a feedback gate `β`:

- add `retain_gate` / `feedback_gate` to `attend`, mirroring `measure`;
- run `β` fixed at several values as a controlled condition;
- then run `β` **learned** through a sigmoid parameterization, at which point `β` becomes a
  measurement in its own right: *how strongly does the model want index feedback?* That is a
  reportable result, and it is the kind of question the gate formalism exists to ask.

---

## 2. Review of the experiment plan

The plan is strong, and several items in it are sharper than what I proposed earlier. Rather than
restate agreement, here is what stands out and what I would change.

### 2.1 The strongest items

- **Within-category individuation, stratified by same-category competitor count.** This is the best
  idea in the plan. It is the tightest possible operationalization of "encode individuals, not just
  classes," and VRD structurally could not pose it — one entity per box, mostly distinct
  categories. Three people in one kitchen is the actual test. Make this the headline experiment.
- **Chain-length scaling of order sensitivity.** Predicting that PVM order-sensitivity *grows with
  chain length* while HB-POVM stays flat is stronger than a point comparison at one chain length,
  because it is a scaling claim that a confound is unlikely to reproduce. This is better than the
  version in the experiment program document.
- **Identity as level 0.** The hypothesis that measuring identity first collapses semantic
  uncertainty and suppresses order effects *even for PVM* is genuinely novel, and it connects the
  index layer to the measurement formalism rather than treating them as separate stories. If it
  holds, it is a thesis result on its own.
- **Tree consistency rather than reversal rate.** Correct and important: reversal rate cannot
  distinguish "the answer changed" from "the answer became incoherent." The disjoint, catch-all-free
  hierarchy is what makes consistency well defined.
- **Scrambled-hierarchy falsification control.** Right, and it should be run for the order-effect
  experiment specifically, since that is where structure could be manufactured by the measurement
  schedule rather than found in the data.
- **Persistence baseline for anticipation, and the oracle-mask framing.** Both correct. The framing
  point in particular — the claim is about binding and memory *given* grouping, not about detection
  — is more defensible and more interesting, and it should appear early in the thesis rather than
  as a limitations note.

### 2.2 Correction: the span-convention trap is already closed

The plan flags the half-open/inclusive discrepancy as something to resolve first. It is already
resolved. The fidelity ledger records the decision (inclusive endpoints), the evidence
(boundary-frame inspection; 1,787 source spans end at the final valid frame; OpenPVSG is internally
inconsistent), and the handling (intersect with `[0, num_frames - 1]`, record every clipped or
empty span). `materialize.py` writes the convention into provenance.

**This is not a blocker, which is good news for starting immediately.**

### 2.3 The real trap: right-censored relation spans

There is a censoring problem in the same area, it is larger, and it is not in the plan.

Of 6,035 relation spans, **1,787 — 29.6% — end at the final valid frame of their video.** Those are
not observed cessations. The relation did not stop; the recording did. A further 35 end at
`num_frames` and 59 fall outside inclusive bounds.

If cessation strata are built naively, **roughly three in ten "cessations" are recording artifacts**,
and experiments 3, 4 and 6 all depend on cessation strata. The consequence is systematic: censored
endpoints have no visual state change, so a model that correctly keeps asserting the relation is
scored as wrong, and the "context helps at cessation" prediction gets diluted precisely where it
should be tested.

**Fix, and it is cheap:** mark every span endpoint as observed or censored during materialization,
and exclude censored endpoints from onset/cessation strata. Apply the same reasoning to onsets —
spans beginning at frame 0 are left-censored, since the relation may have started before the video.
The audit does not currently report that count; add it in the same pass.

This is the single highest-value correction in this document after Section 1, and it takes an
afternoon.

### 2.4 Additions

- **The feedback-magnitude diagnostic of Section 1 must precede experiments 1–3.** Those experiments
  test exactly the claim it could invalidate.
- **Calibration is missing.** The discrete-bottleneck framing needs ECE, NLL and Brier, not
  accuracy alone: the interesting claim is what commitment costs in *probability quality* as chain
  length grows, and accuracy cannot show it. Add calibration to every P-SA/P-Samp table.
- **Run the flat-fusion information-matched control everywhere,** not only in experiment 3. It is
  the baseline that determines whether any result is about mechanism or about evidence, and it is
  cheap once written.
- **Split the `P-noI` ablation in two.** Remove identity indices entirely (the paper's version), and
  separately keep identity but remove hierarchy levels. Only the pair isolates whether the benefit
  is identity-specific rather than "more supervision helps."
- **Reduce the order-effect design.** With four semantic levels plus identity, the full factorial is
  120 orders; even the four-level version is 24. Interpretation degrades badly under that many
  conditions. Use a fractional design: forward, reverse, identity-first, identity-last, and a random
  sample of about eight, with the full enumeration reserved for a three-level subset where 6 orders
  is exhaustive and legible.

---

## 3. Readiness: what can run today

This is the answer to "start experimenting ASAP." Six of the seven experiments need **no new
feature extraction and no new core component.**

| # | Experiment | Data ready? | What is still needed |
|---|---|---|---|
| 1 | Within-category individuation | **Yes** | competitor counts, derivable from existing manifests + hierarchy |
| 2 | Identity persistence vs Δt | **Yes** | `blocked/boundaries.jsonl` and `blocked_last_observation` are already materialized; appearance drift is one pass over cached features |
| 3 | Dynamic context, factorial | **Yes** | flat-fusion model class (small); onset/cessation strata **after the censoring fix** |
| 4 | Occlusion / permanence | **Yes** | gap detection from object observation tables; explicit absent-evidence condition |
| 5 | Order effects + hierarchy | **Yes** | sequential readout schedules; `vocabulary.py` already wires `("source", *HIERARCHY_LEVELS)` |
| 6 | Anticipation | **Yes** | Δ-shifted targets; persistence baseline; **censoring fix is a prerequisite** |
| 7 | Continual index recruitment | **No** | needs the growable vocabulary / reserve pool (component 3.1) — the one genuine blocker |

The materialized protocol set already includes `heldout_video`, `blocked` (with boundaries) and
`fewshot` (enrollment, support, queries). `IndexVocabulary`, `PDirect` and `IntegralTB` exist. The
practical implication is that the pipeline is not the bottleneck — writing runners and metrics is.

**Experiment 7 is the exception, and it is worth noting that the blocking component is small.** The
reserve-pool allocator is the cheapest item in the missing-components catalogue. If continual
recruitment matters for the thesis, that one component can be pulled forward without opening the
extensions program.

---

## 4. Execution order

Largely the plan's own ordering, with the diagnostics inserted and the pairs that share
infrastructure merged.

**Days 1–2 — unblock and de-risk.**
Feedback-magnitude diagnostic (Section 1.5). Span censoring fix and audit extension (Section 2.3).
Neither is an experiment; both invalidate experiments if skipped.

**Week 1 — pipeline validation.**
P-Direct, flat fusion, TB-no-feedback, P-SA, P-Samp on `heldout_video`, positive-pair predicate
recognition only. Macro-AP with per-predicate support and a prior-only baseline; ≥5 seeds with
paired tests. Exit criterion: the information-versus-mechanism decomposition is known.

**Weeks 2–3 — identity, the core claim.**
Experiments 1 and 2 together — they share the identity readout, the `blocked` protocol and the
retrieval metrics. Individuation stratified by same-category competitor count; persistence
stratified by Δt, occlusion gap length and appearance drift. Both `P-noI` variants. Frozen-DINO
nearest-centroid as the mandatory baseline throughout.

**Weeks 4–5 — measurement and order.**
Experiment 5 with the reduced design, tree consistency, the scrambled-hierarchy control, the
chain-length scaling curve, and the identity-as-level-0 condition. Calibration reported alongside
accuracy.

**Weeks 6–7 — temporal.**
Experiments 4 and 6. Occlusion decay curves; anticipation against persistence at Δ = 1s/2s/4s, with
onset, persistence and cessation reported separately on censoring-corrected strata.

**Week 8+ — recruitment and modern baselines.**
Pull the reserve-pool allocator forward, run experiment 7, and add the contemporary reference set
(frozen-DINO probe, matched-budget transformer over the same four tokens, VLM zero-shot on the same
frames for characterization rather than competition).

---

## 5. One dependency worth flagging early

The plan's modern framings are well chosen, and three of the four are reachable with the core
program. The exception is **causal use of sampled indices** — intervening by injecting a wrong
identity and measuring downstream predicate change.

That is a core-capability test, not an extension: it is the falsification test for whether index
feedback is load-bearing or decorative, which is the central claim of the whole framework. It needs
only the small intervention harness (component 3.11), and it should be run alongside Weeks 2–3
rather than deferred with the extensions. If injecting a wrong identity index barely changes the
predicate distribution, that finding reframes every other result in the thesis, and it is better to
know it in week 3 than in month six.

Note also the direct link to Section 1: if feedback is a `10⁻³`-relative perturbation, the
intervention will show nothing *for scale reasons rather than architectural ones*. Run the
diagnostic first, then the intervention, and report them together.

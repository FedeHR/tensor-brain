# Handover: the pair-predicate chapter (VRD-E / VRD-EX transfer)

Written for an agent with fresh context who will produce the thesis section, its
figures, and its tables. Everything needed to do that without re-deriving it is here or
in the analyses this document points at.

**Read the six analyses first.** They contain the numbers, the mechanism arguments, and
the caveats. They live under `runs/` and are **not in git** (`.gitignore:232`), so they
exist only on this machine — back them up before relying on them.

| analysis | what it establishes |
|---|---|
| `runs/pair-corrected-original-seed0/ANALYSIS.md` | original-evolution + `direct` scoring; superseded, keep for the scoring-rule argument |
| `runs/pair-corrected-qtb-seed0/ANALYSIS.md` | QTB + `softplus-bias` wins by ~5 pp; why (log-normalizer keeps the readout data-driven) |
| `runs/pair-blocked-seed0/ANALYSIS.md` | known-entity protocol; the dynamic-context decomposition; identity recognition works |
| `runs/pair-category-feedback-seed0/ANALYSIS.md` | category feedback at β=1; the information-theoretic argument |
| `runs/pair-feedback-gate-seed0/ANALYSIS.md` | the β ladder 1→32 (superseded headline, valid mechanism section) |
| `runs/pair-learned-gate-seed0/ANALYSIS.md` | learned β: 26.65 category, 0.002 identity, on held-out video |
| `runs/pair-learned-gate-blocked-seed0/ANALYSIS.md` | learned β on known entities: identity recovers to 2.33 — **the flagship result** |

Supporting: `docs/index_feedback_evidence.md` (what the papers claim vs what we tested),
`docs/pair_known_entity_protocol.md` (protocol design), `docs/fidelity.md` (every
interpretation decision, including the ones that must be disclosed).

---

## 1. What the chapter claims

Six claims. Each names the runs that support it. Do not make a claim the seed plan in
§2 does not cover.

**C1 — Dynamic context transfers to real video; index feedback at the paper's gate does
not.** P-Direct → Integral is +10.83 pp R@1 on novel entities and +9.02 pp on known
ones; identity feedback at β=1 is +0.08 and +0.27 pp. *This decomposition is the
chapter's central contribution*, because the source papers cannot make it: original §6.3
removes both mechanisms in P-Direct simultaneously ("there are no links from n to q, **and
q and h are independent**"), yet attributes the whole +15.16 pp binary-label gap to the
dynamic context layer three separate times (§6.5, Table 4 caption, Table 6 caption). We
separated them and the attribution holds.

**C2 — The feedback null at β=1 is an artifact of injection scale, and the scale was
never chosen.** β=1 puts the injected embedding at 6.5% of the state norm. Sweeping
β ∈ {1,2,4,8,16,32} improves KL monotonically to a minimum at β=16, where the injection
is 1.07× the state norm. See §4 for the causal explanation to write out.

**C3 — Identity feedback is conditional on the index being retrievable.** A learned β
settles at **0.002** on novel entities and **2.33** on known ones — a thousand-fold
difference from changing only which entities the evaluation contains. The trajectory
shows the gate collapsing for ~4,000 steps then recovering as identity accuracy becomes
usable. This is the paper's VRD-E/VRD-EX contrast (+0.12 vs +6.37 points) reduced to one
learned scalar, and it is the strongest single result in the chapter.

**C4 — Category feedback dominates identity feedback, and buys generalization rather
than memorization.** Learned category β ≈ 25–27 in both protocols; on known entities it
gives +8.75 pp R@1 and +10.08 pp macro against no feedback, roughly 5× the identity
effect, and it *improves* unseen-triple recall. Amplified identity feedback instead
gains +1.90 pp R@1 while **losing 6.81 pp on unseen triples** — memorization traded for
composition, exactly what original Table 4's caption asserts ("entity indices permit
some memorization").

**C5 — Where the Tensor Brain stands against baselines.** It beats every visual-only
model, loses to the directed category-pair prior on seen triples, and wins decisively on
unseen ones. The prior must be presented as an *oracle-metadata* diagnostic: it consumes
ground-truth subject and object categories at evaluation time.

**C6 — Mechanism diagnostics** (single seed is sufficient; these are measurements, not
contrasts): injection-to-state ratios, attention entropy and concentration, feedback
direction cosine, CBS occupancy, per-group gradient norms.

---

## 2. The seed plan

**Three seeds (0, 1, 2) of the seeded set below, at one commit, `PAIR_MAX_STEPS=15000`.**
Rerun seed 0 too — the existing seed-0 runs span four commits, and a thesis table should
not need a footnote about which code produced which row. Do not mix step budgets.

Why 15,000 and not 10,000: every run so far selected its checkpoint at step 8,000–9,500
of 10,000 with mAP and category accuracy still climbing. 15,000 puts the selection well
inside the budget. **The learned identity gate may still not converge** — it was rising
at 3.51 when its 10,000-step run ended — so report that gate as a lower bound and say so.
The qualitative contrast that carries C3 (0.002 versus >2) is already decisive.

### Seeded set — 9 jobs per seed, 27 total

| protocol | conditions | command |
|---|---|---|
| held-out | P-Direct, Integral-none, Integral-P-SA β=1 | `PAIR_PROTOCOL=heldout_video PAIR_SEED=$S PAIR_MAX_STEPS=15000 sbatch --array=2-4 cluster/pvsg/pair_known_entities.sbatch` |
| held-out | learned β, category + identity | `PAIR_SEED=$S PAIR_MAX_STEPS=15000 sbatch cluster/pvsg/pair_learned_gate.sbatch` |
| blocked | Integral-none, Integral-P-SA β=1 | `PAIR_PROTOCOL=blocked PAIR_SEED=$S PAIR_MAX_STEPS=15000 sbatch --array=3-4 cluster/pvsg/pair_known_entities.sbatch` |
| blocked | learned β, category + identity | `PAIR_PROTOCOL=blocked PAIR_SEED=$S PAIR_MAX_STEPS=15000 sbatch cluster/pvsg/pair_learned_gate.sbatch` |

Run seeds sequentially so a queue problem costs one seed, not three.

### Single-seed set — 8 jobs

| purpose | command | why one seed suffices |
|---|---|---|
| priors, both protocols | `--array=0` of `pair_known_entities.sbatch` with each `PAIR_PROTOCOL` | count-based, **deterministic** — no training, no seed dependence |
| union-only, both protocols | `--array=1` of the same, each protocol | a floor, reported without error bars |
| β ladder | `PAIR_GATE_VALUES="1 2 4 8 16 32" PAIR_MAX_STEPS=15000 sbatch --array=0-5 cluster/pvsg/pair_feedback_gate.sbatch` | a curve; its endpoints are seeded by the learned-gate cells |

**Total 35 jobs at 15k steps**, against 51 at 20k in the first draft — roughly 45% of
the original compute.

### What was cut, and what it costs

| cut | cost |
|---|---|
| `p-direct` on blocked | C1 rests on held-out video alone. Acceptable: the effect is +10.83 pp there. |
| dedicated `cat-sa` β=1 cells (3 seeds) | Covered by the β=1 point of the ladder and by the learned-gate trajectories, which all start at β=1. Loses error bars on a **null**, which is the cheapest place to lose them. |
| `priors` and `union-only` at 3 seeds | Priors are deterministic, so nothing is lost. Union-only loses error bars on a floor. |
| step budget 20k → 15k | The learned identity gate may not converge; report it as a lower bound. |

### If you need to cut further

Drop the **β ladder** (6 jobs, → 29 total). C2 then rests on the learned gate alone,
which finds the optimum and gives the trajectory but loses the monotone curve of F2 —
the most visually convincing evidence that the scale effect is real rather than a lucky
setting. Cut this before cutting anything else.

**Never drop:** P-Direct and Integral-none on held-out (C1), Integral-P-SA β=1 on both
protocols (C1, C3), or either learned-gate run on either protocol (C3, C4). Those five
cells are the chapter.

### Parallelism

Array tasks run concurrently up to the `%N` throttle in each script: `%3` for
`pair_known_entities`, `%2` for the others. Separate `sbatch` submissions are
independent jobs with no cross-cap, so submitting one seed's four commands puts up to
nine tasks in flight. Override with `sbatch --array=2-4%3` to raise or lower it.

## 3. Figures and tables to produce

Load `docs/dataviz` guidance before writing any plotting code. Nine artifacts:

**T1 — Main results table.** Rows: frequency prior, category-pair prior, union-only,
P-Direct, Integral no-feedback, Integral P-SA β=1, category learned β, identity learned
β. Columns: KL, exact, R@1, R@5, macro R@1, video R@1, mAP, seen@1, unseen@1. Two
blocks, one per protocol. **Mean ± s.d. over 3 seeds**; mark oracle-metadata rows.

**F1 — The decomposition (C1).** Grouped bars, R@1 with seed error bars: P-Direct →
Integral-none → Integral-P-SA, both protocols side by side. Annotate the two deltas.
This is the chapter's headline figure.

**F2 — The β ladder (C2).** Twin-axis line plot over β ∈ {1…32}, log-x: predicate KL
(left) and class-macro R@1 (right), with a second panel showing injection/‖q‖ on the
same x. Mark β=1 (the papers' value) and the KL minimum at 16.

**F3 — The learned-gate trajectories (C3).** β against training step, four curves:
{identity, category} × {held-out, blocked}, log-y. This single figure carries the
flagship claim — the identity curves separate by three orders of magnitude, and the
blocked identity curve visibly collapses then recovers. Overlay the step at which
training identity accuracy first exceeds chance.

**F4 — Memorization versus composition (C4).** Scatter or slope chart: Δ R@1 against
Δ unseen-triple R@1 for each feedback condition relative to no-feedback. Identity
feedback goes up-and-left, category feedback up-and-right. Makes the trade-off visible
in one image.

**F5 — Per-predicate redistribution.** Diverging horizontal bars, Δ R@1 by predicate at
the best category gate, ordered by support. Shows the gain is tail coverage, not
head-sharpening.

**F6 — Mechanism panel (C6).** Small multiples from `scale_trace.jsonl`: injection/‖q‖,
attention max probability, feedback direction cosine, CBS occupancy — each against β.
Demonstrates the sweep was one-factor.

**T2 — Auxiliary cost table.** Subject and object category accuracy, and identity
accuracy where defined, across gates. Shows the predicate gain is paid for by the unary
readout at high β.

**T3 — Fidelity table.** Every deviation from the papers, with its ledger category:
DINOv3 backbone, PVSG ontology, `√D` input normalization, `A` initialization,
`argmax` for P-Samp, entity-only attention candidates, β > 1, and the
attention-vs-measurement schedule discrepancy. Source: `docs/fidelity.md`.

---

## 4. The scale explanation to write out (C2)

This needs a paragraph in the chapter, not a footnote.

The injected embedding is small relative to the state because of **two independent
normalization decisions that were never considered jointly**:

1. Inputs are mapped to `√D · L2normalize(x)` with `D = 768`, so `‖g(ν)‖ ≈ 27.7`.
2. `A` is initialized so each column has expected squared norm one, so `‖a_k‖ ≈ 1.0`.

Their ratio is **27.7 : 1 at initialization and 13.9 : 1 at a trained checkpoint**. Nobody
decided top-down should be ~4–7% of bottom-up; it fell out. Both papers then fix β = 1
and bound it by 1, so the default configuration cannot correct it.

**State clearly that scaling `A` is not an equivalent fix.** `score_k = σ(q)ᵀa_k + a₀,k`,
so scaling `A` scales the injection *and* every index score, whereas β scales only the
injection. Formally β = c is equivalent to `A`-scaled-by-c **plus** an index-score
inverse temperature of 1/c — and that temperature is not implemented
(`fidelity.md`: "No inverse-temperature argument initially"). Under `softplus-bias` the
equivalence is not even exact, since `a₀ = −Σ softplus(a)` is not linear in `A`. So β is
a genuine third degree of freedom, not a symptom of bad initialization, and a corrected
initialization should not be expected to reproduce the β-sweep results.

The honest framing: the papers' equations are not wrong — β=1 reproduces them exactly —
but β=1 delivers roughly a tenth of the available benefit on this task, and the shortfall
is traceable to normalization choices made elsewhere in the pipeline.

---

## 5. Pitfalls — read before computing any number

Each of these has already caused, or nearly caused, a wrong claim.

1. **mAP is not comparable across the AP fix.** Runs before commit `9203a40` rank raw
   logits; after, categorical probabilities. Never put pre- and post-fix mAP in one
   table. Affects `runs/pair-seed0` and `runs/pair-complementarity-seed0`.
2. **`result.json` shape changed at `c55cf6f`.** Older runs are
   `evaluation[mode]`; newer are `evaluation[set][mode]`. Handle both.
3. **Blocked and held-out numbers are not cross-comparable.** Blocked trains on the
   observation window only (168,621 pairs vs 367,132). Compare *within* a run across
   its two evaluation sets, or across runs *within* a protocol — never diagonally.
4. **Seen/unseen triples are saturated on blocked.** 96.9% of its assignments involve a
   training-seen triple, so `triple_seen@1` there is near-meaningless. The unseen column
   is where blocked claims should be made.
5. **The category-pair prior is an oracle.** It consumes ground-truth categories at
   evaluation. Always label it as such; it beats every learned model on blocked (77.66
   R@1) for that reason.
6. **β ties attention and measurement.** One gate covers Algorithm 2's attention and
   Algorithm 3's injection, so no result attributes the gain to one operation. Disclose.
7. **The delay stratification is confounded** with video length (the embargo is 10% of
   each video), and its 0–2 s bin has 4 examples. Do not report it as memory persistence.
8. **`runs/` is gitignored.** The analyses and results are not backed up by version
   control.
9. **P-Samp is `argmax`, not sampling.** It is the β→∞ inverse-temperature limit, which
   is what the original paper used for its P-Samp experiments. The `sequential-sample`
   readout is the only one that samples at temperature 1.
10. **Do not describe this as a reproduction.** Backbone, dataset, dimensionality and
    ontology all changed; the numbers could never have matched. It is a controlled
    decomposition of which mechanisms transfer.

---

## 6. Analysis commands

Re-evaluate finished checkpoints under other readouts or gates without retraining:

```bash
uv run python -m experiments.pvsg.pair_evaluate \
  --run-dir "$PVSG_RUN_ROOT/<run>" \
  --manifest-root "$PVSG_SECTION6_MANIFEST_ROOT" \
  --feature-root "$DINO_FEATURE_ROOT" \
  --feedback-gates 1,8,16
```

Per-run artifacts: `result.json` (final metrics), `validation_trace.jsonl` (per-250-step
curve), `training_trace.jsonl` (per-100-step losses, plus `feedback_gate` when learned),
`scale_trace.jsonl` (mechanism diagnostics at steps 0/1/10/100/1k/5k/10k and `best`),
`config.json` (full configuration incl. `protocol_layout`), `vocabulary.json`.

---

## 7. What is deliberately out of scope

Not run, and the chapter should say so rather than imply otherwise: episodic indices and
episodic attention (**note: original §6.5 says EA was enabled in *all* of the paper's
perception experiments — this is a fidelity gap that must be disclosed**), category
feedback into `q` during the object schedule, the `g⁺` decoder and reconstruction
objective, cross-frame evolution, per-group gates, and the concept-set attention variant
(Algorithm 1 lines 18/25 score over the full concept set `C`, while §5.3 restricts to
entity columns — an unresolved discrepancy recorded in the ledger).

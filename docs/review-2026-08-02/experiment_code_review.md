# Review of the experiment code on `main` (through `8471744`)

## Recommendation: keep it, spend half a day, then send

**Do not roll back to the worktree checkpoint.** That would discard 112 passing tests, correct
module boundaries, and `diagnostics.py`, which is precisely the instrument the S1 scale question
needs. Rolling back costs days and gains nothing.

**Do not send exactly as-is either** — not because the code is bad, but because the pilot's two
score-mode arms (`centered` and `softplus-bias`) are *both* already corrected for the offset problem
of issue M17, so the run cannot show what the correction buys. Adding the uncorrected original-TB
arm (`direct`) costs one run and turns the pilot into a real contrast. See **C1b**.

**The honest diagnosis is that the code is not chaotic — it is early.** The structure is right. The
volume comes from three locatable places, and only one of them is worth fixing before the pilot.

---

## Why "chaotic" is the wrong word

The module boundaries match `AGENTS.md` and they are the boundaries I would have chosen:

```
data.py        cached-feature access and normalization
indices.py     vocabulary construction
supervision.py targets, losses, metrics
models.py      explicit forward schedules
evaluation.py  eval loops
diagnostics.py scalar instrumentation
runtime.py     CLI, device, model construction, checkpointing
object_experiment.py / overfit.py   one experiment each, readable top-to-bottom
```

`runtime.py` is the file most likely to have become the "generic protocol runner" the repo policy
forbids. It did not — it is 189 lines of argparse, device seeding, `build_model`, and
`save_checkpoint`, and the experiment schedule stays in the experiment. That was the main structural
risk and you avoided it.

`diagnostics.py` (346 lines) is *good* bloat: a leaf module with no coupling that you write once and
never read again. It measures exactly what S1 asked for — `‖feedback‖/‖q‖`, `‖a_k‖` by group,
neutral-versus-data-dependent score decomposition, per-group gradient norms. Leave it alone.

## Where the volume actually comes from

**1. The pair path is fully built and currently unused (~700–800 lines).**
`IntegralTB.forward` / `PDirect.forward`, `PairTargets` / `PairLosses` / `pair_losses` /
`pair_metrics` / `build_pair_targets`, `PVSGPairDataset`, `overfit.py`, `readout_rows`,
`scale_trace_rows`, and the pair arms of `baselines.py`. `object_experiment.py` imports none of it.
Under the object-first plan this is a later chapter. It is not wrong; it doubles the surface a
reader has to hold in their head today.

**2. Trace instrumentation is interleaved with the model schedule.**
`IntegralTB.forward` is 113 lines, of which roughly 55 are trace bookkeeping. The paper schedule —
about eight lines — is buried inside it. This is the single largest contributor to "I can't read my
own model", and it works directly against the repo's stated policy that the operation order is part
of the scientific model.

**3. Six forward signatures for two conceptual schedules.**
`forward` × `forward_object` × {`PDirect`, `IntegralTB`}, plus `LinearProbe` mirroring both.

---

## Correctness and fidelity issues

### C1 — `softplus-bias` is the paper-faithful mode; only its equation number is wrong

*(Corrected after checking `papers/qtb_LATEST.pdf` directly. My first pass claimed the softplus form
was a derived interpretation rather than a paper equation. It is a paper equation.)*

The chain in the latest QTB, verified on the PDF:

- **Eq. (25), p. 49** — `q₀ = −Σ_ℓ log(1 + exp(q_ℓ))`, giving `Bern(i; γ) = exp(q₀ + qᵀi)`.
  `log(1 + exp(·))` is softplus.
- **Eq. (32)–(33), p. 51** — activating index `k` sets `q ← a_k`, inducing
  `p_i ← Bern(i; sig(a_k)) ≡ b_{i,k}`.
- **Eq. (34), p. 52** — "Using the identity in Equation (25), we obtain
  `P(k | i) = softmax_k(a_{0,k} + a_kᵀ i)`."

Substituting `q → a_k` into (25) gives exactly

```
a_{0,k} = −Σ_ℓ log(1 + exp(a_{ℓ,k})) = −Σ_ℓ softplus(a_{ℓ,k})
```

So in the latest QTB **`a_{0,k}` is derived, not free**, and `softplus-bias` is the faithful mode.
`learned-bias` is the deviation — a free-parameter relaxation of the Bernoulli-consistent
normalizer. That is still worth running, as an ablation asking whether the probabilistic
interpretation costs accuracy.

**The only defect is the citation.** `src/tb/model.py:106-107` cites Equation (31), which is the
evolution operator `h = sig(v₀ + Vγ), f_NN^evol(γ) = Wh` (p. 50, confirmed). It should cite
**Equation (25), applied at Equation (34)**. One-token fix.

**`direct` is also a paper equation** — the *original* TB has no index bias anywhere: Algorithm 1
line 18 is `nS(s) ← a_sᵀ sig(q̃S)` and §5.3 is `qS ← q̃S + A softmax_β(Aᵀ sig(q̃S))`. So the four
modes map cleanly onto sources, and the naming is defensible as it stands:

| mode | source |
|---|---|
| `direct` | original TB (2021), no bias |
| `softplus-bias` | latest QTB, Eq. (25)/(34) — derived normalizer |
| `learned-bias` | free-parameter relaxation (deviation) |
| `centered` | `γ − 0.5` ablation, least paper-grounded |

### C1b — the real gap: the pilot's two arms are both already corrected

Measured at init, `D = 768`, `K = 7143`, decomposing each score into its state-independent offset
and its state-dependent part:

| `score_mode` | logit sd | offset sd | data sd | offset / data |
|---|---:|---:|---:|---:|
| `direct` | 0.539 | 0.498 | 0.208 | **2.39** |
| `learned-bias` | 0.539 | 0.498 | 0.208 | **2.39** |
| `centered` | 0.208 | 0.000 | 0.208 | 0.00 |
| `softplus-bias` | 0.208 | 0.006 | 0.208 | **0.03** |

This is a genuinely nice result and it should go in the thesis. **The paper's softplus normalizer
is, to first order, exactly the centering correction that issue M17 identified as necessary** — and
it achieves it with no free parameters. Expanding `softplus(x) ≈ log 2 + x/2` gives
`a_{0,k} ≈ −n log 2 − ½Σ_ℓ a_{ℓ,k}`, whose constant term cancels in the softmax and whose varying
term cancels the `½Σ_ℓ a_{ℓ,k}` DC component of `a_kᵀγ`. Measured: an **83× reduction** in the
offset, from 0.498 to 0.006.

So M17 is not a defect of the framework — it is something the latest QTB already fixes, and this
repository can demonstrate that empirically.

But it means the currently configured pilot (`centered` and `softplus-bias`) runs **two corrected
variants against each other** and cannot show what the correction buys. **Add `direct` as a third
arm.** It is the original-TB equation, it is the uncorrected condition, and it costs one run.
`learned-bias` as a fourth arm is optional but cheap, and answers whether a free bias beats the
derived one.

### C3 — should fix: `identities` means three different things in `run_object_experiment`

- line 148: `identities` = the set of identity **names** in the vocabulary
- line 238: `identities: bool` = whether to evaluate identity at all
- line 299: `identities` = the loop variable rebinding that flag per evaluation split

They are in the same function and the last one shadows the first. This is very likely part of why
the logic stopped feeling clear. Rename to `train_identity_names` / `score_identity`.

### C4 — note in the write-up, not a bug: model selection ignores identity

The checkpoint is selected on `loss/category_total` over development videos, where identity targets
do not exist by construction (novel identities). Blocked identity accuracy is then reported from a
checkpoint chosen purely for category generalization. That is defensible and probably correct — but
say so explicitly, or a reader will assume the checkpoint was selected for the number you report.

### What is right and I checked specifically

- `VideoBlockSampler` positions are interpreted correctly against `Subset` (sampler yields subset
  positions, `Subset` maps to dataset indices). Correct.
- Development videos are excluded from `blocked/train_*` via `_role_indices(..., "train")`, so the
  earlier open question about role overlap is now resolved in the clean direction.
- The `blocked` evaluation identity-subset assertion at line 155 is a good guard and will hold,
  because blocked-eval rows require prefix visibility.
- `diagnose()` brackets its backward pass with `zero_grad` on both sides, so it cannot corrupt the
  next optimizer step.
- Development evaluation uses `build_category_targets(allow_unknown=True)`, which correctly avoids a
  `KeyError` on novel identities. This is exactly the S5 fix.
- `weight_decay` defaults to 0.0, so the reserve-column decay hazard does not apply.

---

## The half-day plan

In priority order. Stop when the pilot is interpretable; the rest can wait.

| # | Change | Time | Why now |
|---|---|---|---|
| 1 | Add `direct` (and optionally `learned-bias`) to `--score-mode`; re-cite the normalizer to Eq. (25)/(34) | 20 min | **C1/C1b** — without `direct` the pilot cannot show what the normalizer buys |
| 2 | Rename the three `identities` bindings | 10 min | **C3** — cheapest readability win in the repo |
| 3 | Collapse per-window trace bookkeeping into one `_entity_trace(...)` helper called once per window | 30 min | restores the ~8-line paper schedule to visibility |
| 4 | Add a "what is live right now" block to `experiments/pvsg/__init__.py` or a short `experiments/pvsg/README.md` | 15 min | the reader's real problem is not knowing which half is dormant |
| 5 | Split the final-evaluation block out of `run_object_experiment` | 20 min | that function is 190 lines with two closures capturing eight variables |

**Explicitly do not do now:** moving the pair path to separate modules (churn without insight —
item 4 solves the perceived problem), unifying the four forward methods (they will diverge again
when the ladder lands), or touching `diagnostics.py`.

## One scope observation

`build_model` supports `p-direct`, but `run_object_experiment` always constructs `integral`. The
information-matched ladder (M0–M5 in the plan) is not wired yet. That is fine for a scale pilot —
just do not let the pilot's single-model shape harden into the experiment's shape, because the
ladder is the centerpiece result.

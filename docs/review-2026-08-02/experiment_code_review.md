# Review of the experiment code on `main` (through `8471744`)

## Recommendation: keep it, spend half a day, then send

**Do not roll back to the worktree checkpoint.** That would discard 112 passing tests, correct
module boundaries, and `diagnostics.py`, which is precisely the instrument the S1 scale question
needs. Rolling back costs days and gains nothing.

**Do not send exactly as-is either** — not because the code is bad, but because of one fidelity
problem that would make the pilot uninterpretable (**C1** below): the object experiment can only be
run with `centered` or `softplus-bias`, and neither of those is the paper's score equation. You
would be settling the scale question without the baseline in the comparison.

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

### C1 — must fix before the pilot: the paper's score equation is not runnable

`src/tb/model.py` changed `a0` from an always-present learned parameter to one that exists **only**
in `score_mode="learned-bias"`. The default is now `"direct"`, which registers `a0 = None` and
returns a zero bias. So:

- The name `direct` now denotes a **new no-bias ablation**, not the paper equation. Anyone reading
  `score_mode: ScoreMode = "direct"` will reasonably assume the opposite.
- `object_experiment.py:317` restricts `--score-mode` to `("centered", "softplus-bias")`. **The
  paper-faithful arm cannot be selected at all**, so the first full run compares two non-paper
  variants against each other.

QTB Equation (40) is `P(k) ← z_k softmax(a_{0,k} + Σ_ℓ γ_ℓ a_{ℓ,k})`, and Algorithm 2 line 821 has
the same free `a_{0,k}`. That is `learned-bias`. It must be in the comparison, and it should
probably be the default.

**Fix:** add `learned-bias` to the CLI choices, run it as the baseline arm, and either rename
`direct` to `no-bias` or make `learned-bias` the default.

### C2 — must fix: the softplus normalizer cites the wrong equation

`src/tb/model.py:106-107` says `softplus-bias` is "QTB Equation (31)'s Bernoulli normalizer".
Equation (31) is the **evolution** operator, `h = sig(v0 + Vγ), f_NN(γ) = Wh`.

The mathematics is fine and worth keeping — `log(1 − sig(a)) = −softplus(a)`, so
`a_k·γ − Σ_ℓ softplus(a_{ℓ,k})` is exactly the factorized-Bernoulli log-likelihood of `γ` under
parameters `sig(a_k)`, which is a genuinely well-motivated variant. But it is a **derived
interpretation**, not a paper equation, and citing (31) will not survive a careful reader. Re-label
it and add a ledger row.

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
| 1 | Add `learned-bias` to `--score-mode`; fix the eq-(31) citation; rename `direct` → `no-bias` | 30 min | **C1/C2** — without it the pilot has no paper baseline |
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

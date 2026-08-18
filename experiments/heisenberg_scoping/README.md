# Scoping scripts for the Heisenberg experimental chapter

Standalone scripts backing the numbers in
[`docs/heisenberg_experiment_design.md`](../../docs/heisenberg_experiment_design.md).
They are scoping evidence, not part of the experiment: each is self-contained,
runs in seconds, and prints a table that appears verbatim in that document.

The four that need the analysis code import it across the worktree boundary from
the `bayes-approximation` branch, following the pattern already used by
`thesis/section2/run_experiments.py`:

```bash
cd ../tensor-brain-bayes-approximation
PYTHONPATH=".:src" uv run --extra bayes python <script>.py
```

| script | what it establishes | needs repo B |
|---|---|---|
| `tau_check.py` | the gate exponent τ family: correction `(1−τ)·M·logZ` is exact at every τ | no |
| `tau_identify.py` | τ is recoverable from observation counts by Poisson regression, unbiased | no (needs `scipy`) |
| `gate_family.py` | the general law `correction = M·[logZ − log g(Z)]` for four gate shapes | no |
| `gate_scale.py` | a saturating gate cancels only where `Z ≪ 1` — which the QTB offset enforces | no |
| `gap_probe.py` | ontology-structured `A` leaves a measurable gap; affine fraction 0.735 | yes |
| `downstream_probe.py` | task metrics and order invariance per rule, both annotation processes | yes |
| `seed_sensitivity.py` | effect sizes vs corpus-seed spread — sizes the replication needed | yes |
| `confound_test.py` | the §10 ranking flip is confounded by prior misspecification | yes |

`tau_identify.py` needs SciPy, which is not in the `bayes` extra:

```bash
PYTHONPATH=".:src" uv run --extra bayes --with scipy python tau_identify.py
```

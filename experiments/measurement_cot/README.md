# Chain of thought as repeated measurement

Discrete chain-of-thought and continuous latent reasoning (COCONUT) are the sharp and
degenerate limits of a single Tensor Brain update. Every intermediate reasoning step here
applies

```text
q <- alpha * q + beta * sum_k w_k a_k
```

and conditions differ only in how the feedback weights `w` are formed from the candidate
index scores. Two of the conditions are already TB core operations: `w = delta_k` with `k`
sampled is `tb.measure`, and `w = p` is `tb.attend`, which QTB reads as a degenerate
measurement whose outcome is not revealed.

The second gate, `alpha`, is the one the LLM setting hides. It controls how much of the
latent state survives a step, and therefore whether the index layer is a bottleneck at all.
The headline finding is that measurement sharpness is irrelevant at `alpha = 1` and decisive
at `alpha = 0`.

## Task

A fixed layered DAG plays the role of semantic memory. A query writes a start concept and
two candidate terminal concepts into the workspace; exactly one terminal is reachable. Two
controls keep the task about traversal rather than association, and both were necessary —
without them every condition scored the same:

- **quota-balanced negatives**, so each terminal occupies the reachable and unreachable slot
  equally often and a terminal-identity rule is at chance by construction;
- **a frozen random index bank**, so a trainable `A` cannot simply store (start, terminal)
  associations and skip the chain.

Splits are over distinct reachable (start, terminal) pairs, so the composition is held out
even though every start and terminal is seen.

Intermediate steps are inherently multi-valued — several children of the frontier are equally
valid — so step supervision targets the true breadth-first **frontier**, not a single gold
path. Training uses COCONUT's staged curriculum: at stage `k` the first `k` hops use the
condition's own collapse and the rest are teacher-forced onto the true frontier. Teacher
forcing is training-only; it writes a gold node whose child is the correct terminal, so it
leaks the answer and is never an evaluation condition.

## Running

```bash
uv sync --extra measurement-cot
PYTHONPATH=. uv run python -m experiments.measurement_cot.run_experiment plane
PYTHONPATH=. uv run python -m experiments.measurement_cot.run_experiment depth --retain 0.0
PYTHONPATH=. uv run python -m experiments.measurement_cot.run_experiment schedule --retain 0.0
PYTHONPATH=. uv run python -m experiments.measurement_cot.run_experiment analysis
PYTHONPATH=. uv run python -m experiments.measurement_cot.figures
```

Results land in `output/measurement_cot/*.json` and figures in
`output/measurement_cot/figures`. The whole campaign is CPU-only and takes well under an
hour per sub-command on a laptop; nothing here needs the cluster.

## Modules

| Module | Contents |
| --- | --- |
| `graph.py` | the fixed layered DAG, reachability, and per-hop breadth-first frontiers |
| `data.py` | queries, quota-balanced negatives, pair-disjoint splits, shortcut baselines |
| `collapse.py` | the collapse dial: how candidate scores become feedback weights |
| `model.py` | the TB reasoning chain with a per-hop measurement schedule |
| `train.py` | curriculum, frontier supervision, evaluation |
| `analysis.py` | frontier mass, Monte-Carlo convergence, Jensen gap, Zeno trajectories |
| `figures.py` | the report figures |

## Report

`reports/measurement_cot/main.tex` (build with `latexmk -pdf main.tex`).

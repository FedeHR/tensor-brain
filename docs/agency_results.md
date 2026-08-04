# Agency and Action Indices: Results

Lab notebook for the research line specified in [`agency_design.md`](agency_design.md). Read that
document first: it states the five claims, the environment, the concept-window schedule, and the
conditions. This one records what was run, in the order it was run, what came out, and which of
the design's assumptions did not survive contact with the data.

## 0. Reproduce

```bash
# Stage A, the capacity gate (fully observed variant)
uv run python -m experiments.agency.run_clone_gate --output runs/agency/clone_gate.json

# Stage B, the ablation grid: every condition, six seeds
for seed in 0 1 2 3 4 5; do
  for condition in $(uv run python -c \
      "from experiments.agency.conditions import CONDITIONS; print(*CONDITIONS)"); do
    uv run python -m experiments.agency.run_grid --condition "$condition" --seed "$seed" \
      --updates 2500 --output-root runs/agency/grid
  done
done

# Score every checkpoint under identical, final metric definitions
uv run python -m experiments.agency.run_reevaluate --grid-root runs/agency/grid

# Evaluation-only readouts: sampling, argmax, one-step imagination
uv run python -m experiments.agency.run_planning --grid-root runs/agency/grid

# Every figure and table in this document
uv run python -m experiments.agency.run_figures --grid-root runs/agency/grid \
  --figure-root docs/figures/agency
```

## 1. Did any of it need a change to the Tensor Brain core?

No. `src/tb` is untouched by this entire research line. That is the first result, and it is a
point in the papers' favour: if actions really are "generated as any other indices", then adding
agency should require no new operation, and it did not.

What an agent needs on top of the core turns out to be exactly the three things the papers place
*outside* it — the perceptual mapping `g(nu)`, the reward module's gated input drive, and the
coupling that lets a measured index change the world:

```python
q, action_index, probabilities = tb.measure(q, vocabulary.indices("action"))
reward, observation = environment.step(action_index)
```

Three further capabilities fall out of operations that already existed:

| Capability | Realized as | New parameters |
|---|---|---|
| Policy | the measurement distribution over the action group | none |
| Credit assignment | `log p` read from the same `measure` call, REINFORCE | none |
| Internal reward function (§13.5.2) | score of a `reward_positive` **index** | one column of `A` |
| Near-term planning (§13.5.2) | `evolve(q + a_action)` scored by that reward index | none |
| Deliberation / CoT (§13.5.2) | extra concept windows per environment step | none |

## 2. Stage A: behavioural cloning does not work as a capacity gate — and why that matters

The design proposed cloning a privileged oracle on a fully observed variant of the task as a
capacity probe, with `no-cue` as the contrast: an agent that cannot see the instruction should
clone much worse. It does not.

| condition | oracle agreement | rollout success |
|---|---|---|
| `tb-full` | 0.770 | 0.602 |
| `no-cue` | 0.768 | 0.587 |
| `no-percept-measure` | 0.807 | 0.773 |
| `deliberate-3-attend` | 0.769 | 0.448 |
| `state-128` | 0.783 | 0.655 |

Three seeds each, 1200 cloning updates, fully observed 5×5 task.

`tb-full` and `no-cue` are indistinguishable on both metrics. The reason is a property of the
objective, not of the architecture: the greedy oracle mostly moves *towards an object*, and moving
towards the wrong object agrees with it on most steps. Imitation therefore supplies almost no
gradient towards using the cue.

> **Finding 1.** In this task the instruction only becomes learnable when there is a cost to
> ignoring it. Reward, which penalizes collecting the wrong object, provides that cost; imitation
> of a competent demonstrator does not. This is a caution for any plan to bootstrap Tensor Brain
> agents from demonstrations: a demonstrator can be near-optimal and still fail to transmit *why*
> it did what it did.

Stage A is reported as a negative methodological result and is not used to initialize Stage B.

## 3. The task has a brute-force loophole, and a control found it

The design made collecting a distractor penalized but *non*-terminal, because a terminal penalty
made "never collect" an overwhelming local optimum that every architecture fell into (Section 6).
The consequence was not anticipated: an agent that knows nothing about the instruction can still
finish most episodes by collecting objects until the reward turns positive. The `no-cue` control
did exactly that, reaching 0.97 success on one seed.

> **Finding 2.** Success rate measures "did the episode end well", not "did the agent follow the
> instruction". The reported primary metric is therefore **first-choice accuracy** — was the first
> object the agent committed to the cued one — which is 1/3 for a cue-blind agent however many
> objects it collects afterwards.

This is what the `no-cue` control is for, and it is the reason it was run.

## 4. Outcomes are bimodal per seed

Runs do not vary smoothly with the seed. A run either escapes the sparse-reward local optimum in
which `collect` is suppressed and every episode times out, or it never does:

```text
tb-full  seed0  0.05 → 0.35 → 0.72 → 0.93 → 0.98 → 0.98
tb-full  seed1  0.09 → 0.38 → 0.85 → 0.95 → 0.97 → 0.98
tb-full  seed2  0.05 → 0.00 → 0.00 → 0.00 → 0.00 → 0.00
```

Trapped runs sit at exactly zero success and a return of `-max_steps × step_penalty = -0.5`.
Averaging across that bimodality reports *how many seeds escaped*, not how the architecture
behaves. Every table below is therefore split into an **escape rate** and metrics **conditional on
having escaped** (`experiments/agency/analysis.py`).

## 5. The ablation grid

TABLE_ESCAPE

![escape rate and conditional first-choice accuracy](figures/agency/escape_and_conditional.png)

![learning curves](figures/agency/learning_curves_main.png)

RESULTS_NARRATIVE

## 6. What was tried and abandoned

Recorded because the failures constrain the design as much as the successes.

1. **Terminal distractor penalty.** The original design ended the episode on any collection, with
   `-1` for a distractor. Every condition, including the GRU control, collapsed to never pressing
   `collect`: early in learning the action is negative in expectation, it is suppressed, and the
   positive outcome is then never experienced. Made non-terminal (Section 3).
2. **A larger, conjunction-heavy observation.** A fully observed 9×9 egocentric view made
   behavioural cloning plateau near 0.73 agreement. Within one concept window the map from input
   to index scores passes exactly one nonlinearity, so binding "colour AND shape AND position"
   across 81 cells needs roughly one hidden unit per cell. Reduced to a 5×5 view, where the
   discrimination that matters happens on the cell the agent occupies.
3. **Learning rate `1e-3`.** Collapsed to the never-collect optimum in every condition tried. A
   trivial one-object control task confirmed the pipeline was correct and that `3e-3` was the
   difference. Recorded because it means the recipe was tuned on the Tensor Brain agent and then
   applied unchanged to the controls, which biases the comparison in the Tensor Brain's favour.
4. **Behavioural-cloning initialization of REINFORCE.** Improved the starting point but degraded
   under policy gradient, because the cloned policy collected distractors ~68% of the time and the
   fastest way to stop losing reward was to stop collecting.

## 7. What is not claimed

- **No claim that the Tensor Brain outperforms a conventional recurrent policy.** The recipe was
  tuned on it; the GRU and LSTM controls inherit it untuned. Within-grid ablation contrasts are
  the defensible readings.
- **No claim about sample efficiency.** REINFORCE with a value baseline is a weak algorithm and
  every condition carries that handicap equally.
- **No claim that this generalizes beyond one purpose-built gridworld** with a hand-made 14-index
  vocabulary designed by the same process that designed the agent. That circularity is precisely
  why [`agency_design.md`](agency_design.md) §3.3 commits to BabyAI/MiniGrid next: published
  compositional-generalization splits and published baselines make the C4 result checkable by
  someone else.
- **No claim of biological plausibility** for the credit-assignment rule.

## 8. Where this goes next

NEXT_STEPS

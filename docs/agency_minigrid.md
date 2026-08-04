# Action Indices on BabyAI / MiniGrid

Second stage of the agency research line. [`agency_design.md`](agency_design.md) §3.3 committed to
MiniGrid as the next environment, and [`agency_results.md`](agency_results.md) §8 gave the reason:
three of the gridworld findings were bottlenecked on a task we designed ourselves. This document
records what happened on a benchmark we did not.

## 0. Reproduce

```bash
uv sync --extra minigrid

for seed in 0 1 2; do
  for level in gotolocal doorkey; do
    for condition in $(uv run python -c \
        "from experiments.agency.minigrid.conditions import LEVEL_CONDITIONS as L; print(*L['$level'])"); do
      uv run python -m experiments.agency.minigrid.run --level "$level" \
        --condition "$condition" --seed "$seed" --output-root runs/agency/minigrid
    done
  done
done

uv run python -m experiments.agency.minigrid.run_figures \
  --grid-root runs/agency/minigrid --figure-root docs/figures/agency/minigrid
```

## 1. Was the existing code enough?

Almost. Two things were needed, and only one of them was about the Tensor Brain.

**`src/tb` was again untouched.** More than that: the schedule turned out to already support what
a stronger optimizer needs. Recurrent PPO must re-evaluate a stored rollout segment under updated
parameters, which requires reproducing the recurrent state *exactly*; because ``window_cycle`` can
teacher-force both the action index and the two perceptual index samples, a segment replays
bit-for-bit. `tests/test_agency_minigrid.py::test_replay_reproduces_the_collected_log_probabilities`
asserts that an unchanged policy returns its collected log-probabilities to `1e-5`. The one
restriction is that `deliberation_mode="measure"` samples an extra index that is not stored, so PPO
rejects it; `attend` is deterministic given the state and replays fine.

**REINFORCE was the blocker.** The gridworld results were dominated by whether a seed escaped the
sparse-reward local optimum at all. MiniGrid levels are longer and sparser, so a stronger estimator
was a prerequisite rather than a refinement. `experiments/agency/minigrid/ppo.py` implements
recurrent PPO with GAE.

**Two ordinary engineering gaps.** MiniGrid environments are per-instance Python objects, so a
batched adapter was needed; and `TensorBrainAgent` was extracted from `GridAgent` so that the two
environments share one copy of the concept-window schedule instead of two that could drift.

## 2. Why this benchmark

MiniGrid is an unusually good fit for an index layer, and not because we arranged it that way.
Its observation is *already symbolic*: each of the 7x7 egocentric cells is a triple of integer
codes for object type, colour and state. BabyAI missions are generated from a small grammar over
those same symbols. So the same column of `A` is

* the perceptual label "the attended cell contains a ball", and
* the instruction "go to the ball", parsed straight out of the mission string,

with nothing invented to connect them. The gridworld had to be designed to make that true; here it
falls out of the benchmark. The perceptual target is computed from the agent's own view, never from
simulator state, so naming is solvable from what the agent sees.

Two levels, chosen for what they test rather than for difficulty:

| level | environment | tests | random policy |
|---|---|---|---|
| `gotolocal` | `BabyAI-GoToLocal-v0` | per-episode language instruction; held-out colour x object combinations | 0.31 |
| `doorkey` | `MiniGrid-DoorKey-6x6-v0` | sparse reward, forced sub-goal order: fetch key, unlock door, cross, reach goal | 0.02 |

The compositional split is built from each level's *real* instruction distribution: missions are
sampled to discover which `(colour, object)` combinations the level generates, one combination per
object type is held out, and layouts are resampled until an allowed mission appears. Held-out
objects still populate the room as distractors, so only the instruction distribution differs.

DoorKey's mission is the same string in every episode, so its cue carries no per-episode
information and `no-cue` is not run there; the level tests sequencing and memory instead.

## 3. Results

RESULTS_SECTION

## 4. What this changes about the gridworld conclusions

TRANSFER_SECTION

## 5. Limitations

- **Frame budget.** 1.02M frames on GoToLocal and 2.56M on DoorKey, three seeds. Published BabyAI
  baselines use considerably more; these numbers are not a claim about the ceiling of any
  architecture, only about their ordering under an identical budget.
- **One hyperparameter setting, shared.** Unlike the gridworld study, the PPO settings here were
  *not* tuned on the Tensor Brain agent -- they are ordinary defaults applied identically to every
  condition, including the controls. That removes the bias the gridworld study had to declare, but
  it does not make the comparison a tuned-versus-tuned one.
- **Three seeds.** Enough to see large effects, not enough to resolve small ones.

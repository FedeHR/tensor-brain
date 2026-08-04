# Memory Maze: probing the belief instead of scoring the task

Third stage of the agency research line. MiniGrid
([`agency_minigrid.md`](agency_minigrid.md)) ended on a measurement problem rather than a result:
a shuffled-mission control showed that no policy on the stock GoTo levels used the instruction at
all, so the numbers those levels produced were not about instruction grounding. The remedy there
was a stricter level. The remedy here is a benchmark that ships **ground truth for what the agent
should be remembering**, so that "does this architecture retain a usable belief" is measured rather
than inferred from a task score.

DeepMind's Memory Maze `ExtraObs` levels expose the agent's position and heading, the positions of
all three coloured targets, the vector to the current target, and the maze layout. None of it
reaches the agent, which sees only a 64x64 RGB image and has six discrete actions.

## 0. What this study does and does not claim

It does **not** claim competitive task performance. Memory Maze's published Dreamer-V3 numbers use
orders of magnitude more experience than this study spends, and a low score will not be dressed up
as anything else. The task score is reported only as evidence that the policies learned something
worth probing.

The claim is the probing one, which is valid at a modest budget:

1. **Linear probe** from the recurrent state to ground truth, for the Tensor Brain against
   capacity-matched GRU and LSTM controls. This is the benchmark's own protocol, it applies
   unchanged to all three architectures, and it is the comparison a control can win.
2. **Native readout** — the Tensor Brain's index scores over the `percept_color` group already are
   a distribution over target colours, obtained with *no probe trained at all*. There is no
   equivalent for a GRU: not a worse number, no readout to take.
3. **Write test** — the measurement update is `q <- alpha q + beta a_k`, so a colour index can be
   written into the state by adding its column `A[:, k]`, and the agent should redirect. A GRU's
   hidden state has no addressable column; the closest analogue is an arbitrary perturbation.

Measurements 2 and 3 are structural rather than quantitative. They are the reason this benchmark
was chosen: they are differences in *what can be asked of the architecture*, not differences in a
score.

## 1. Capacity

All three policies share the **same** convolutional encoder, so every parameter of the difference
sits after perception, which is where the claim is.

| condition      | total   | encoder | post-perception |
| -------------- | ------- | ------- | --------------- |
| `tb-full`      | 319,072 | 268,128 | 50,944          |
| `gru-control`  | 368,743 | 268,128 | 100,615         |
| `lstm-control` | 401,767 | 268,128 | 133,639         |

The controls carry **more** capacity than the Tensor Brain, not less. This follows the standing
rule for this project: strong architectures at comparable scale with known-good hyperparameters,
rather than deliberately weakened controls or an equal-budget sweep.

## 2. Environment recipe

This is fiddly and does not travel between machines, so it is recorded in
`experiments/agency/memorymaze/env.py` as well as here.

- **Python 3.12.** `labmaze`, a `dm_control` dependency, ships no 3.13 wheel.
- A **MuJoCo rendering backend**, which is the one setting that differs by platform:
  `MUJOCO_GL=glfw` on macOS, where `egl` and `osmesa` are unavailable; `MUJOCO_GL=egl` on a
  headless Linux node, where `glfw` needs a display a batch job does not have. `osmesa` is the
  software fallback for a node with no usable EGL device.
### Why not gymnasium

`gym` has been unmaintained since 2022 and prints a banner recommending `gymnasium` as a drop-in
replacement. That advice does not apply to this package, and the banner is misleading here:
`memory_maze` imports `gym` by name, registers its levels in *gym's* registry, and its `GymWrapper`
subclasses `gym.Env` with the pre-0.26 four-tuple `step`. `gymnasium.make` would not find the
levels. Version 1.0.3 is the final release, so no version bump fixes it.

The adapter therefore uses neither. It talks to Memory Maze through its **native `dm_env`
interface** — `memory_maze.tasks.memory_maze_9x9(global_observables=True, seed=...)` — which is what
the `gym` layer wraps anyway. That removes `gym.make`, the passive environment checker and the
`np.bool8` shim the checker needed under NumPy 2, and it fixes a real wart: the task factories take
a `seed`, so each environment in the batch is seeded properly rather than by reseeding the global
NumPy RNG before construction.

`gym` remains an *installed* dependency, because `memory_maze/__init__.py` imports it
unconditionally and re-raises if it is absent. Nothing in this repository imports it, and a test
parses the package's ASTs to keep it that way.

## 3. Throughput, and why this belongs on a cluster

Rendering is the bottleneck and it **does not parallelise inside a process**: the adapter steps its
environments in one Python loop, so a batch of eight runs no faster in wall-clock terms than a
batch of one. Measured on an M3 Pro:

| configuration            | env-steps/s (each) | env-steps/s (total) |
| ------------------------ | ------------------ | ------------------- |
| 1 process, 4 envs        | 209                | 209                 |
| 1 process, 8 envs        | 200                | 200                 |
| 4 processes              | ~178               | ~712                |
| 9 processes              | ~150               | ~1350               |

So scale comes from *processes*, one per condition and seed — which is what the Slurm array does.

Two warnings drawn from doing this the wrong way first. Nine concurrent MuJoCo processes rendering
through GLFW contend on the macOS window server and GPU rather than only on CPU, and took a 11-core
/ 18 GB laptop down; core count is not the binding constraint. And a full run outlives the shell
that starts it, so anything launched locally must be detached, not merely backgrounded.

## 4. Reproduce

Locally, for a smoke test only:

```bash
uv sync --frozen --python 3.12 --extra memorymaze --extra dev
MUJOCO_GL=glfw uv run --frozen --no-sync python -m pytest tests/test_agency_memorymaze.py

# One short run and a probe over it, to prove the pipeline rather than to measure anything.
MUJOCO_GL=glfw uv run --frozen --no-sync python -m experiments.agency.memorymaze.run \
  --condition tb-full --seed 0 --updates 3 --output-root /tmp/smoke
MUJOCO_GL=glfw uv run --frozen --no-sync python -m experiments.agency.memorymaze.run_probe \
  --run-root /tmp/smoke/9x9 --steps 40 --warmup 8
```

On the cluster:

```bash
# On a compute node: syncs the 3.12 environment and proves the render backend works.
# Run it there, not on the login node -- a login node usually has no GPU, so EGL
# fails there even when the batch nodes are fine.
cluster/agency/setup.sh

# Nine array tasks: three conditions at three seeds.
cluster/agency/submit_memorymaze.sh --partition=<name> --time=04:00:00

# Then, over the finished checkpoints:
uv run --frozen --no-sync --python 3.12 python -m experiments.agency.memorymaze.run_probe \
  --run-root "$MEMORYMAZE_RUN_ROOT/9x9"
```

`submit_memorymaze.sh` refuses to submit when experiment code is uncommitted, staged-but-uncommitted
or untracked, and records the submitted revision in `AGENCY_CODE_REVISION` — run artifacts are only
interpretable against the code that produced them.

## 5. Probe protocol

Every probe is fit on one rollout and scored on a **second rollout from different environment
seeds**. Adjacent steps within a rollout are strongly correlated, so a random split across a single
rollout would leak and report an `R^2` that means nothing.

Every trained policy is additionally probed against an **untrained policy of the same
architecture**. That control is not optional: the shared convolutional encoder is randomly
initialised, and a random projection of pixels already carries some position information. Without
it, the probe would be reporting what the camera sees rather than what the agent retained.

`R^2` is computed against the test-set mean, so 0 is the score of predicting the mean and a
negative score is worse than that.

## 6. Status

Integration, cluster scripts and probe are implemented and tested (19 tests; 216 in the suite).
**No training run has produced results yet** — the first local attempt was killed for the reason in
§3, and the study has not been submitted. Nothing in this document is a finding.

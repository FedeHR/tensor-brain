r"""Probing a trained Memory Maze policy against the benchmark's ground truth.

This is the claim of the study, and it is valid at a budget that a competitive
task score is not. Memory Maze's ``ExtraObs`` levels expose the agent position,
the positions of all three coloured targets and the maze layout, none of which
the agent ever observes. So "does this architecture retain a usable belief"
becomes a measurement on the recurrent state rather than an assertion.

Three measurements, in increasing order of how much they distinguish a Tensor
Brain from a recurrent control:

1. **Linear probe.** Fit a ridge regression from the recurrent state to ground
   truth, and report :math:`R^2` on a held-out rollout. This is the benchmark's
   own protocol and applies unchanged to all three policies, so it is the fair
   comparison. It is also the one a control can win.

2. **Native readout.** The Tensor Brain already names what it sees: its index
   scores over the ``percept_color`` group are a distribution over target
   colours, obtained with *no probe trained at all*. Scoring those against
   ground truth measures the same quantity as (1) without fitting anything.
   There is no equivalent for a GRU -- not a worse number, no readout -- and
   that asymmetry is the structural point.

3. **Write test.** The measurement update is
   :math:`q \leftarrow \alpha q + \beta a_k`, so a colour index can be *written*
   into the state by adding its column :math:`A[:, k]`. If the belief is what
   drives behaviour, writing a colour should redirect the agent toward that
   colour's target. A GRU's hidden state has no addressable column to write:
   the closest analogue is an arbitrary perturbation, which is why this test
   exists only on one side.

Every probe is fit on one rollout and evaluated on a second rollout from
different environment seeds. Adjacent steps within a rollout are strongly
correlated, so a random split across a single rollout would leak the answer and
report an :math:`R^2` that means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from experiments.agency.agent import TensorBrainAgent
from experiments.agency.memorymaze.env import COLOR_NAMES, VectorMemoryMaze
from experiments.agency.memorymaze.linear_probe import classification_probe, regression_probe

# Ground-truth fields worth probing, and why each is included.
REGRESSION_TARGETS: dict[str, str] = {
    # The memory claim: where all three targets are, only one of which is ever
    # the current goal and none of which is visible from most of the maze.
    "targets_pos": "positions of all three targets",
    # Self-localisation. A policy can do this from recent observations alone, so
    # it is the easier quantity and acts as a floor for the probe's sensitivity.
    "agent_pos": "the agent's own position",
    # The vector to the current target: the quantity most directly useful for
    # acting, and the one a purely reactive policy is least able to hold.
    "target_vec": "displacement to the current target",
}


@dataclass(frozen=True)
class Recording:
    """Recurrent states paired with the ground truth they should encode."""

    state: Float[Tensor, "samples width"]
    ground_truth: dict[str, Float[Tensor, "samples dims"]]
    # The Tensor Brain's own colour naming; ``None`` for a control, which has no
    # index layer and therefore nothing to read.
    percept_color: Float[Tensor, "samples colors"] | None
    percept_label: Int[Tensor, " samples"]

    def __len__(self) -> int:
        return int(self.state.shape[0])


@torch.no_grad()
def record(
    environment: VectorMemoryMaze, policy, steps: int, *, warmup: int = 32
) -> Recording:
    """Roll the policy out and pair each state with the ground truth behind it.

    The state recorded at a step is the one *after* that step's observation has
    been taken in, which is the state that should encode the ground truth
    observed at that step. ``warmup`` steps are discarded so the belief is not
    measured from a zeroed initial state.
    """

    device = policy.brain.A.device
    state, context = policy.initial_state(environment.num_envs, device)
    previous_reward = torch.zeros(environment.num_envs, device=device)
    vocabulary = environment.vocabulary

    states: list[Tensor] = []
    truths: list[dict[str, Tensor]] = []
    colors: list[Tensor] = []
    labels: list[Tensor] = []

    for step in range(warmup + steps):
        observation = environment.observation().to(device)
        cue_color, cue_other = environment.cue_indices()
        truth = environment.ground_truth()
        percept_color, _ = environment.percept_targets()

        trace = policy.window_cycle(
            state, context, observation, previous_reward,
            cue_color.to(device), cue_other.to(device),
        )
        if step >= warmup:
            states.append(trace.q.cpu())
            truths.append({key: truth[key] for key in REGRESSION_TARGETS})
            labels.append(vocabulary.get_positions("percept_color", percept_color))
            if trace.percept_color_probabilities is not None:
                colors.append(trace.percept_color_probabilities.cpu())

        result = environment.step(trace.action_position)
        reward = result.reward.to(device)
        done = result.done.to(device)
        state, context = policy.reset_finished(trace.q, trace.context, done)
        previous_reward = reward * (~done).float()

    return Recording(
        state=torch.cat(states),
        ground_truth={
            key: torch.cat([truth[key] for truth in truths]) for key in REGRESSION_TARGETS
        },
        percept_color=torch.cat(colors) if colors else None,
        percept_label=torch.cat(labels),
    )


def probe_regression(
    train: Recording, test: Recording, key: str, *, penalty: float = 1.0
) -> dict[str, float]:
    r"""Fit state -> ground truth and report held-out :math:`R^2`."""

    return regression_probe(
        train.state,
        train.ground_truth[key],
        test.state,
        test.ground_truth[key],
        penalty=penalty,
    )


def probe_colors(
    train: Recording, test: Recording, *, penalty: float = 1.0
) -> dict[str, float]:
    """Linear probe for *which colour is nearest*, the quantity (2) reads natively."""

    return classification_probe(
        train.state,
        train.percept_label,
        test.state,
        test.percept_label,
        classes=len(COLOR_NAMES) + 1,  # the three colours plus "nothing visible"
        penalty=penalty,
    )


def native_readout(recording: Recording) -> dict[str, float] | None:
    """Score the Tensor Brain's own colour naming, with no probe fitted.

    Returns ``None`` for a policy with no index layer, which is the honest
    answer for a GRU: not a lower number, but no such readout to take.
    """

    if recording.percept_color is None:
        return None
    classes = recording.percept_color.shape[1]
    predicted = recording.percept_color.argmax(dim=1)
    correct = (predicted == recording.percept_label).double().mean()
    majority = recording.percept_label.bincount(minlength=classes).max() / len(recording)
    return {
        "accuracy": float(correct),
        "majority_baseline": float(majority),
        "chance": 1.0 / classes,
        "probe_parameters": 0.0,
    }


@torch.no_grad()
def write_test(
    environment: VectorMemoryMaze,
    policy,
    *,
    color: str,
    horizon: int = 64,
    warmup: int = 64,
) -> dict[str, float] | None:
    r"""Write a colour index into the state and measure whether behaviour follows.

    Two rollouts share a seed, an environment and a sampling stream, so they stay
    identical through ``warmup``. Then one of them has :math:`A[:, k]` for the
    named colour added to its state -- the same operation the measurement update
    performs -- and the two are run on for ``horizon`` steps. The reported
    quantity is how much closer the written run ends up to the written colour's
    target than the untouched run does.

    ``None`` for a control policy: there is no column to write.
    """

    # Deliberately an isinstance check rather than a duck-typed one. A control
    # also has `.vocabulary` and a `.A`, but its `A` is the action head -- a
    # `6 x state` output layer, not the `state x indices` embedding whose column
    # `k` is the symbol `k`. Writing into it would be a shape error at best and
    # a meaningless perturbation at worst.
    if not isinstance(policy, TensorBrainAgent):
        return None
    index = policy.vocabulary.index(color)
    slot = COLOR_NAMES.index(color)

    def rollout(inject: bool) -> float:
        torch.manual_seed(0)
        environment.reset()
        device = policy.brain.A.device
        state, context = policy.initial_state(environment.num_envs, device)
        previous_reward = torch.zeros(environment.num_envs, device=device)
        for step in range(warmup + horizon):
            if inject and step == warmup:
                # The measurement update's feedback term, applied by hand.
                state = state + policy.brain.A[:, index]
            observation = environment.observation().to(device)
            cue_color, cue_other = environment.cue_indices()
            trace = policy.window_cycle(
                state, context, observation, previous_reward,
                cue_color.to(device), cue_other.to(device),
            )
            result = environment.step(trace.action_position)
            reward = result.reward.to(device)
            done = result.done.to(device)
            state, context = policy.reset_finished(trace.q, trace.context, done)
            previous_reward = reward * (~done).float()
        truth = environment.ground_truth()
        agent = truth["agent_pos"]
        target = truth["targets_pos"].reshape(len(agent), len(COLOR_NAMES), 2)[:, slot]
        return float((target - agent).norm(dim=-1).mean())

    baseline = rollout(inject=False)
    written = rollout(inject=True)
    return {
        "color": float(slot),
        "distance_baseline": baseline,
        "distance_written": written,
        # Positive means writing the colour moved the agent toward that target.
        "approach": baseline - written,
    }

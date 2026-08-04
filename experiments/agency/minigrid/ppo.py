"""Recurrent PPO for the Tensor Brain agent on MiniGrid.

The gridworld study used REINFORCE, and its results were dominated by whether a
seed escaped the sparse-reward local optimum at all. MiniGrid levels are longer
and sparser, so a stronger estimator is a prerequisite rather than a refinement.

The Tensor Brain needs no new machinery for PPO. The policy is still the
generative measurement over the action candidate group, and re-evaluating a
stored segment under updated parameters requires only that the *same* trajectory
be reproduced -- which ``window_cycle`` already supports, because it can
teacher-force both the action index and the two perceptual index samples. The
recurrent state is therefore replayed exactly, not approximated.

``deliberation_mode="measure"`` is rejected here: it samples an extra index over
the whole layer that is not stored, so a segment could not be replayed exactly.
Use ``attend``, which is deterministic given the state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor

# Deliberately duck-typed rather than imported: this optimizer is environment
# agnostic. It needs only `window_cycle`, `initial_state`, `reset_finished`,
# `brain.A` and `config` from a policy, and `observation`, `cue_indices` and
# `step` from an environment, so it also trains the Memory Maze agents without
# dragging in a MiniGrid dependency.
Policy = Any
VectorMiniGrid = Any


@dataclass(frozen=True)
class PPOConfig:
    """Every controlled variable of the optimizer, in one place."""

    updates: int = 400
    segment_steps: int = 64
    epochs: int = 4
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_weight: float = 0.01
    value_weight: float = 0.5
    percept_weight: float = 0.0
    grad_clip: float = 0.5
    evaluate_every: int = 20


@dataclass
class Segment:
    """One rollout segment, with everything needed to replay it exactly."""

    observation: Float[Tensor, "steps envs observation"]
    cue_color: Tensor
    cue_object: Tensor
    previous_reward: Tensor
    action_index: Tensor
    action_position: Tensor
    percept_color: Tensor
    percept_object: Tensor
    log_probability: Tensor
    value: Tensor
    reward: Tensor
    done: Tensor
    initial_state: Tensor


@dataclass
class TrainingLog:
    update: list[int] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    train_metrics: list[dict[str, float]] = field(default_factory=list)
    eval_metrics: list[dict[str, float]] = field(default_factory=list)
    holdout_metrics: list[dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class EpisodeTracker:
    """Running success rate, return and length over completed episodes."""

    def __init__(self, num_envs: int, window: int = 256) -> None:
        self.returns = torch.zeros(num_envs)
        self.lengths = torch.zeros(num_envs)
        self.finished_returns: list[float] = []
        self.finished_lengths: list[float] = []
        self.finished_successes: list[float] = []
        self.window = window

    def update(self, reward: Tensor, done: Tensor, success: Tensor) -> None:
        self.returns += reward
        self.lengths += 1.0
        for slot in done.nonzero().flatten().tolist():
            self.finished_returns.append(float(self.returns[slot]))
            self.finished_lengths.append(float(self.lengths[slot]))
            self.finished_successes.append(float(success[slot]))
            self.returns[slot] = 0.0
            self.lengths[slot] = 0.0
        for series in (self.finished_returns, self.finished_lengths, self.finished_successes):
            del series[: max(0, len(series) - self.window)]

    def metrics(self) -> dict[str, float]:
        def mean(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else float("nan")

        return {
            "success_rate": mean(self.finished_successes),
            "mean_return": mean(self.finished_returns),
            "mean_length": mean(self.finished_lengths),
            "episodes": float(len(self.finished_successes)),
        }


def _log_probability(probabilities: Tensor, position: Tensor) -> Tensor:
    return probabilities.clamp_min(1e-12).log().gather(-1, position[..., None]).squeeze(-1)


def _entropy(probabilities: Tensor) -> Tensor:
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


@torch.no_grad()
def collect(
    environment: VectorMiniGrid,
    policy: Policy,
    state: Tensor,
    context: Tensor | None,
    previous_reward: Tensor,
    tracker: EpisodeTracker,
    steps: int,
) -> tuple[Segment, Tensor, Tensor | None, Tensor]:
    """Roll the policy forward for ``steps`` and record a replayable segment."""

    device = state.device
    initial_state = state.clone()
    buffers: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "observation", "cue_color", "cue_object", "previous_reward",
            "action_index", "action_position", "percept_color", "percept_object",
            "log_probability", "value", "reward", "done",
        )
    }
    for _ in range(steps):
        observation = environment.observation().to(device)
        cue_color, cue_object = environment.cue_indices()
        cue_color, cue_object = cue_color.to(device), cue_object.to(device)
        trace = policy.window_cycle(
            state, context, observation, previous_reward, cue_color, cue_object
        )
        result = environment.step(trace.action_position)
        reward = result.reward.to(device)
        done = result.done.to(device)

        buffers["observation"].append(observation)
        buffers["cue_color"].append(cue_color)
        buffers["cue_object"].append(cue_object)
        buffers["previous_reward"].append(previous_reward)
        buffers["action_index"].append(trace.action_index)
        buffers["action_position"].append(trace.action_position)
        # Stored so that PPO can teacher-force the *same* perceptual samples and
        # reproduce the recurrent state exactly.
        zeros = torch.zeros_like(trace.action_index)
        buffers["percept_color"].append(
            trace.percept_color_index if trace.percept_color_index is not None else zeros
        )
        buffers["percept_object"].append(
            trace.percept_shape_index if trace.percept_shape_index is not None else zeros
        )
        buffers["log_probability"].append(
            _log_probability(trace.action_probabilities, trace.action_position)
        )
        buffers["value"].append(trace.value)
        buffers["reward"].append(reward)
        buffers["done"].append(done)

        tracker.update(reward.cpu(), done.cpu(), result.success)
        state, context = policy.reset_finished(trace.q, trace.context, done)
        previous_reward = reward * (~done).float()

    segment = Segment(
        **{key: torch.stack(values) for key, values in buffers.items()},
        initial_state=initial_state,
    )
    return segment, state, context, previous_reward


def generalized_advantage(
    segment: Segment, final_value: Tensor, *, discount: float, gae_lambda: float
) -> tuple[Tensor, Tensor]:
    """GAE(lambda) advantages and value targets over one segment."""

    advantages = torch.zeros_like(segment.reward)
    running = torch.zeros_like(final_value)
    next_value = final_value
    for step in reversed(range(segment.reward.shape[0])):
        alive = (~segment.done[step]).float()
        delta = segment.reward[step] + discount * next_value * alive - segment.value[step]
        running = delta + discount * gae_lambda * alive * running
        advantages[step] = running
        next_value = segment.value[step]
    return advantages, advantages + segment.value


def replay(policy: Policy, segment: Segment) -> tuple[Tensor, Tensor, Tensor]:
    """Re-run a stored segment under current parameters, with gradients.

    Actions and perceptual measurements are teacher-forced to their recorded
    outcomes, so the state trajectory is identical to the one that produced the
    stored advantages. Only the probabilities and values change.
    """

    state, context = segment.initial_state, None
    measures_percepts = getattr(policy.config, "measure_percepts", False)
    log_probabilities, entropies, values = [], [], []
    for step in range(segment.reward.shape[0]):
        trace = policy.window_cycle(
            state,
            context,
            segment.observation[step],
            segment.previous_reward[step],
            segment.cue_color[step],
            segment.cue_object[step],
            action_teacher=segment.action_index[step],
            percept_teacher=(
                (segment.percept_color[step], segment.percept_object[step])
                if measures_percepts
                else None
            ),
        )
        log_probabilities.append(
            _log_probability(trace.action_probabilities, segment.action_position[step])
        )
        entropies.append(_entropy(trace.action_probabilities))
        values.append(trace.value)
        state, context = policy.reset_finished(trace.q, trace.context, segment.done[step])
    return torch.stack(log_probabilities), torch.stack(entropies), torch.stack(values)


def train(
    environment: VectorMiniGrid,
    policy: Policy,
    config: PPOConfig,
    *,
    evaluation: Callable[[], dict[str, dict[str, float]]] | None = None,
) -> TrainingLog:
    """Optimize the action-index measurement with recurrent PPO."""

    if getattr(policy.config, "deliberation_mode", "attend") == "measure":
        raise ValueError(
            "deliberation_mode='measure' cannot be replayed exactly; use 'attend'"
        )
    device = policy.brain.A.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
    state, context = policy.initial_state(environment.num_envs, device)
    previous_reward = torch.zeros(environment.num_envs, device=device)
    tracker = EpisodeTracker(environment.num_envs)
    log = TrainingLog()

    for update in range(config.updates):
        segment, state, context, previous_reward = collect(
            environment, policy, state, context, previous_reward, tracker, config.segment_steps
        )
        with torch.no_grad():
            observation = environment.observation().to(device)
            cue_color, cue_object = environment.cue_indices()
            final = policy.window_cycle(
                state, context, observation, previous_reward,
                cue_color.to(device), cue_object.to(device),
            )
            advantages, targets = generalized_advantage(
                segment, final.value, discount=config.discount, gae_lambda=config.gae_lambda
            )
            normalized = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

        total = 0.0
        for _ in range(config.epochs):
            optimizer.zero_grad()
            log_probabilities, entropies, values = replay(policy, segment)
            ratio = (log_probabilities - segment.log_probability).exp()
            clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range)
            policy_loss = -torch.min(ratio * normalized, clipped * normalized).mean()
            value_loss = ((values - targets) ** 2).mean()
            loss = (
                policy_loss
                - config.entropy_weight * entropies.mean()
                + config.value_weight * value_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
            optimizer.step()
            total += float(loss.detach())

        if update % config.evaluate_every == 0 or update == config.updates - 1:
            log.update.append(update)
            log.frames.append((update + 1) * config.segment_steps * environment.num_envs)
            log.loss.append(total / config.epochs)
            log.train_metrics.append(tracker.metrics())
            splits = evaluation() if evaluation is not None else {}
            log.eval_metrics.append(splits.get("eval", {}))
            log.holdout_metrics.append(splits.get("holdout", {}))
    return log


@torch.no_grad()
def evaluate(
    environment: VectorMiniGrid, policy: Policy, *, episodes: int = 64
) -> dict[str, float]:
    """Run the policy until ``episodes`` episodes have finished."""

    device = policy.brain.A.device
    environment.reset()
    state, context = policy.initial_state(environment.num_envs, device)
    previous_reward = torch.zeros(environment.num_envs, device=device)
    tracker = EpisodeTracker(environment.num_envs, window=10_000)
    while len(tracker.finished_successes) < episodes:
        observation = environment.observation().to(device)
        cue_color, cue_object = environment.cue_indices()
        trace = policy.window_cycle(
            state, context, observation, previous_reward,
            cue_color.to(device), cue_object.to(device),
        )
        result = environment.step(trace.action_position)
        tracker.update(result.reward, result.done, result.success)
        state, context = policy.reset_finished(trace.q, trace.context, result.done.to(device))
        previous_reward = result.reward.to(device) * (~result.done).float().to(device)
    return tracker.metrics()

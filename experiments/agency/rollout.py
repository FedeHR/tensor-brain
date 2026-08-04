"""Episode collection: the loop that couples action indices to the world.

One environment step is one concept-window cycle. Whole episodes are collected
in a batch because the gridworld horizon is short, which keeps the return
computation exact and removes the need for bootstrapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from experiments.agency.agent import GridAgent
from experiments.agency.gridworld import SymbolicForaging


@dataclass
class EpisodeBatch:
    """Per-step tensors for one batch of complete episodes.

    Every ``[steps, envs]`` tensor is only meaningful where ``alive`` is true:
    an environment that has already terminated keeps being stepped so that the
    batch stays rectangular, and every consumer masks with ``alive``.
    """

    alive: Bool[Tensor, "steps envs"]
    reward: Float[Tensor, "steps envs"]
    value: Float[Tensor, "steps envs"]
    action_position: Int[Tensor, "steps envs"]
    action_log_probability: Float[Tensor, "steps envs"]
    action_greedy_match: Float[Tensor, "steps envs"]
    action_entropy: Float[Tensor, "steps envs"]
    percept_log_probability: Float[Tensor, "steps envs"]
    percept_correct: Float[Tensor, "steps envs"]
    collected_target: Bool[Tensor, " envs"]
    collected_distractor: Bool[Tensor, " envs"]
    first_choice_correct: Bool[Tensor, " envs"]
    first_choice_made: Bool[Tensor, " envs"]
    episode_length: Int[Tensor, " envs"]
    cue_color: Int[Tensor, " envs"]
    cue_shape: Int[Tensor, " envs"]

    @property
    def num_steps(self) -> int:
        return int(self.alive.shape[0])

    def returns_to_go(self, discount: float) -> Float[Tensor, "steps envs"]:
        """Discounted return from each step to the end of its episode."""

        returns = torch.zeros_like(self.reward)
        running = torch.zeros_like(self.reward[0])
        for step in reversed(range(self.num_steps)):
            # The future contributes only if the *next* step still belongs to
            # this episode; rewards of already-finished environments are masked.
            continues = (
                self.alive[step + 1].float()
                if step + 1 < self.num_steps
                else torch.zeros_like(running)
            )
            running = self.reward[step] * self.alive[step].float() + discount * running * continues
            returns[step] = running
        return returns


def _log_probability(
    probabilities: Float[Tensor, "envs candidates"], position: Int[Tensor, " envs"]
) -> Float[Tensor, " envs"]:
    return probabilities.clamp_min(1e-12).log().gather(1, position[:, None]).squeeze(1)


def _entropy(probabilities: Float[Tensor, "envs candidates"]) -> Float[Tensor, " envs"]:
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


def run_episodes(
    environment: SymbolicForaging,
    agent: GridAgent,
    *,
    teacher_forcing: bool = False,
    supervise_percepts: bool = False,
) -> EpisodeBatch:
    """Run one batch of complete episodes and return everything the losses need.

    ``teacher_forcing`` replaces the sampled action index with the privileged
    oracle action, which is how behavioural cloning uses the same schedule as
    reinforcement learning. ``supervise_percepts`` additionally teacher-forces
    the perceptual measurements to the true labels of the attended object, which
    is the grounded-symbol condition.
    """

    device = agent.brain.A.device
    config = environment.config
    num_envs = environment.num_envs
    environment.reset()
    q, context = agent.initial_state(num_envs, device)
    previous_reward = torch.zeros(num_envs, device=device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)

    steps: list[dict[str, Tensor]] = []
    collected_target = torch.zeros(num_envs, dtype=torch.bool, device=device)
    collected_distractor = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_choice_correct = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_choice_made = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
    cue_color = agent.color_indices[environment.state.cue_color()]
    cue_shape = agent.shape_indices[environment.state.cue_shape()]

    for step_index in range(config.max_steps):
        observation = environment.observation()
        visible_slot = environment.visible_object_slot()
        true_color, true_shape = agent.percept_targets(
            visible_slot, environment.state.object_color, environment.state.object_shape
        )
        is_first_step = torch.full(
            (num_envs,), step_index == 0, dtype=torch.bool, device=device
        )
        action_teacher = (
            agent.action_indices[environment.oracle_action()] if teacher_forcing else None
        )
        trace = agent.window_cycle(
            q,
            context,
            observation,
            previous_reward,
            cue_color,
            cue_shape,
            is_first_step=is_first_step,
            action_teacher=action_teacher,
            percept_teacher=(true_color, true_shape) if supervise_percepts else None,
        )
        q, context = trace.q, trace.context

        result = environment.step(trace.action_position)
        alive_now = alive.clone()
        percept_log_probability = torch.zeros(num_envs, device=device)
        percept_correct = torch.zeros(num_envs, device=device)
        if trace.percept_color_probabilities is not None:
            color_position = agent.vocabulary.get_positions("percept_color", true_color)
            shape_position = agent.vocabulary.get_positions("percept_shape", true_shape)
            percept_log_probability = _log_probability(
                trace.percept_color_probabilities, color_position
            ) + _log_probability(trace.percept_shape_probabilities, shape_position)
            percept_correct = 0.5 * (
                (trace.percept_color_index == true_color).float()
                + (trace.percept_shape_index == true_shape).float()
            )
        steps.append(
            {
                "alive": alive_now,
                "reward": result.reward,
                "value": trace.value,
                "action_position": trace.action_position,
                "action_log_probability": _log_probability(
                    trace.action_probabilities, trace.action_position
                ),
                "action_greedy_match": (
                    trace.action_probabilities.argmax(dim=1) == trace.action_position
                ).float(),
                "action_entropy": _entropy(trace.action_probabilities),
                "percept_log_probability": percept_log_probability,
                "percept_correct": percept_correct,
            }
        )

        collected_target |= result.collected_target & alive_now
        collected_distractor |= result.collected_distractor & alive_now
        # The first object an agent commits to is the cleanest read of whether it
        # followed the instruction: a cue-blind agent is at 1/3 here however many
        # objects it collects afterwards.
        newly_chosen = (
            (result.collected_target | result.collected_distractor)
            & alive_now
            & ~first_choice_made
        )
        first_choice_correct |= newly_chosen & result.collected_target
        first_choice_made |= newly_chosen
        episode_length += alive_now.long()
        alive = alive & ~result.done
        previous_reward = result.reward * alive_now.float()
        if not bool(alive.any()):
            break

    stacked = {
        key: torch.stack([step[key] for step in steps]) for key in steps[0]
    }
    return EpisodeBatch(
        **stacked,
        collected_target=collected_target,
        collected_distractor=collected_distractor,
        first_choice_correct=first_choice_correct,
        first_choice_made=first_choice_made,
        episode_length=episode_length,
        cue_color=cue_color,
        cue_shape=cue_shape,
    )


@dataclass(frozen=True)
class EpisodeMetrics:
    """Aggregate outcome of one batch of episodes."""

    success_rate: float
    distractor_rate: float
    timeout_rate: float
    mean_length: float
    mean_return: float
    percept_accuracy: float
    action_entropy: float
    action_agreement: float
    first_choice_accuracy: float
    first_choice_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "distractor_rate": self.distractor_rate,
            "timeout_rate": self.timeout_rate,
            "mean_length": self.mean_length,
            "mean_return": self.mean_return,
            "percept_accuracy": self.percept_accuracy,
            "action_entropy": self.action_entropy,
            "action_agreement": self.action_agreement,
            "first_choice_accuracy": self.first_choice_accuracy,
            "first_choice_rate": self.first_choice_rate,
        }


def summarize(batch: EpisodeBatch) -> EpisodeMetrics:
    """Reduce one episode batch to the reported scalars."""

    alive = batch.alive.float()
    steps = alive.sum().clamp_min(1.0)
    success = batch.collected_target.float()
    distractor = batch.collected_distractor.float()
    return EpisodeMetrics(
        success_rate=float(success.mean()),
        distractor_rate=float(distractor.mean()),
        timeout_rate=float((1.0 - success - distractor).mean()),
        mean_length=float(batch.episode_length.float().mean()),
        mean_return=float((batch.reward * alive).sum(dim=0).mean()),
        percept_accuracy=float((batch.percept_correct * alive).sum() / steps),
        action_entropy=float((batch.action_entropy * alive).sum() / steps),
        action_agreement=float((batch.action_greedy_match * alive).sum() / steps),
        first_choice_accuracy=float(
            batch.first_choice_correct.sum() / batch.first_choice_made.sum().clamp_min(1)
        ),
        first_choice_rate=float(batch.first_choice_made.float().mean()),
    )


@torch.no_grad()
def evaluate(
    environment: SymbolicForaging, agent: GridAgent, *, repeats: int = 1
) -> EpisodeMetrics:
    """Evaluate the current policy over ``repeats`` batches of fresh episodes."""

    agent.eval()
    totals: list[EpisodeMetrics] = [
        summarize(run_episodes(environment, agent)) for _ in range(repeats)
    ]
    agent.train()
    fields = totals[0].as_dict()
    return EpisodeMetrics(
        **{
            key: float(sum(item.as_dict()[key] for item in totals) / len(totals))
            for key in fields
        }
    )

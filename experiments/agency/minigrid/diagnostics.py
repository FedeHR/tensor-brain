"""Qualitative probes of a trained MiniGrid agent.

MiniGrid renders its worlds, so unlike the gridworld study these figures can show
the actual environment the agent moved through, with the agent's own sampled
symbols written alongside. The trajectory overlay is the most informative: a
whole episode's path drawn on the map, coloured by time, with the sub-goal events
(picking the key up, toggling the door open) marked where they happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from experiments.agency.minigrid.agent import MiniGridAgent, RecurrentControl
from experiments.agency.minigrid.env import VectorMiniGrid

TILE_PIXELS = 32


@dataclass
class NarratedEpisode:
    """One episode with the agent's own symbolic narration attached."""

    env_id: str
    mission: str
    position: list[tuple[int, int]] = field(default_factory=list)
    direction: list[int] = field(default_factory=list)
    action_name: list[str] = field(default_factory=list)
    action_probabilities: list[list[float]] = field(default_factory=list)
    named_color: list[str] = field(default_factory=list)
    named_object: list[str] = field(default_factory=list)
    true_color: list[str] = field(default_factory=list)
    true_object: list[str] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    carrying: list[str | None] = field(default_factory=list)
    frames: dict[int, np.ndarray] = field(default_factory=dict)
    success: bool = False

    @property
    def length(self) -> int:
        return len(self.position)

    def events(self) -> dict[str, int]:
        """Step indices at which the sub-goals were first achieved."""

        found: dict[str, int] = {}
        for step, item in enumerate(self.carrying):
            if item is not None and "picked up" not in found:
                found["picked up"] = step
                break
        for step, name in enumerate(self.action_name):
            if name == "toggle" and "opened door" not in found:
                found["opened door"] = step
        if self.success:
            found["reached goal"] = self.length - 1
        return found


@torch.no_grad()
def narrate_episode(
    environment: VectorMiniGrid,
    policy: MiniGridAgent | RecurrentControl,
    *,
    frame_every: int = 1,
) -> NarratedEpisode:
    """Run one episode on a single-environment batch and record everything."""

    device = policy.brain.A.device
    environment.reset()
    inner = environment.envs[0].unwrapped
    episode = NarratedEpisode(environment.env_id, environment.mission(0))
    state, context = policy.initial_state(1, device)
    previous_reward = torch.zeros(1, device=device)
    measures = getattr(policy.config, "measure_percepts", False)

    for step in range(environment.max_steps):
        episode.position.append(tuple(int(value) for value in inner.agent_pos))
        episode.direction.append(int(inner.agent_dir))
        episode.carrying.append(None if inner.carrying is None else inner.carrying.type)
        if step % frame_every == 0:
            episode.frames[step] = environment.render(0)
        true_color, true_object = environment.percept_targets()
        trace = policy.window_cycle(
            state,
            context,
            environment.observation().to(device),
            previous_reward,
            *[tensor.to(device) for tensor in environment.cue_indices()],
        )
        state, context = trace.q, trace.context
        vocabulary = policy.vocabulary
        episode.true_color.append(vocabulary.label(int(true_color[0])))
        episode.true_object.append(vocabulary.label(int(true_object[0])))
        if measures and trace.percept_color_index is not None:
            episode.named_color.append(vocabulary.label(int(trace.percept_color_index[0])))
            episode.named_object.append(vocabulary.label(int(trace.percept_shape_index[0])))
        episode.action_name.append(vocabulary.label(int(trace.action_index[0])))
        episode.action_probabilities.append(trace.action_probabilities[0].tolist())
        episode.value.append(float(trace.value[0]))

        result = environment.step(trace.action_position)
        episode.reward.append(float(result.reward[0]))
        if bool(result.done[0]):
            episode.success = bool(result.success[0])
            break
    return episode


@torch.no_grad()
def best_episode(
    environment: VectorMiniGrid,
    policy: MiniGridAgent | RecurrentControl,
    *,
    attempts: int = 40,
) -> NarratedEpisode:
    """Return the shortest successful episode from a sample, or the longest try.

    A successful episode is what the figure is meant to explain; among successes
    the shortest one keeps the filmstrip readable.
    """

    episodes = [narrate_episode(environment, policy) for _ in range(attempts)]
    successful = [item for item in episodes if item.success]
    if successful:
        return min(successful, key=lambda item: item.length)
    return max(episodes, key=lambda item: item.length)

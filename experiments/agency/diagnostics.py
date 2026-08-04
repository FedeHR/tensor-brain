"""Qualitative probes of a trained agent.

These are the figures that make a symbolic architecture worth having: what the
agent *named*, what it *chose*, and what its single reward column believes about
each cell of the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor

from experiments.agency.agent import GridAgent
from experiments.agency.gridworld import GridConfig, SymbolicForaging
from experiments.agency.vocabulary import COLOR_NAMES, SHAPE_NAMES


@dataclass
class NarratedEpisode:
    """One episode with the agent's own symbolic narration attached."""

    grid: GridConfig
    agent_row: list[int] = field(default_factory=list)
    agent_col: list[int] = field(default_factory=list)
    object_row: list[int] = field(default_factory=list)
    object_col: list[int] = field(default_factory=list)
    object_color: list[int] = field(default_factory=list)
    object_shape: list[int] = field(default_factory=list)
    target_slot: int = 0
    cue: tuple[str, str] = ("", "")
    named_color: list[str] = field(default_factory=list)
    named_shape: list[str] = field(default_factory=list)
    true_color: list[str] = field(default_factory=list)
    true_shape: list[str] = field(default_factory=list)
    action_name: list[str] = field(default_factory=list)
    action_probabilities: list[list[float]] = field(default_factory=list)
    percept_color_probabilities: list[list[float]] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    success: bool = False


@torch.no_grad()
def narrate_episode(
    environment: SymbolicForaging, agent: GridAgent, *, env_slot: int = 0
) -> NarratedEpisode:
    """Run one episode and record every symbol the agent activated.

    ``environment`` must have been constructed with the batch size you want;
    only ``env_slot`` is transcribed, which keeps the batched code path
    identical to training.
    """

    device = agent.brain.A.device
    config = environment.config
    environment.reset()
    q, context = agent.initial_state(environment.num_envs, device)
    previous_reward = torch.zeros(environment.num_envs, device=device)
    state = environment.state
    episode = NarratedEpisode(
        grid=config,
        object_row=state.object_row[env_slot].tolist(),
        object_col=state.object_col[env_slot].tolist(),
        object_color=state.object_color[env_slot].tolist(),
        object_shape=state.object_shape[env_slot].tolist(),
        target_slot=int(state.target_slot[env_slot]),
        cue=(
            COLOR_NAMES[int(state.cue_color()[env_slot])],
            SHAPE_NAMES[int(state.cue_shape()[env_slot])],
        ),
    )
    cue_color = agent.color_indices[state.cue_color()]
    cue_shape = agent.shape_indices[state.cue_shape()]

    for step_index in range(config.max_steps):
        episode.agent_row.append(int(environment.state.agent_row[env_slot]))
        episode.agent_col.append(int(environment.state.agent_col[env_slot]))
        visible_slot = environment.visible_object_slot()
        true_color, true_shape = agent.percept_targets(
            visible_slot, environment.state.object_color, environment.state.object_shape
        )
        trace = agent.window_cycle(
            q,
            context,
            environment.observation(),
            previous_reward,
            cue_color,
            cue_shape,
            is_first_step=torch.full(
                (environment.num_envs,), step_index == 0, dtype=torch.bool, device=device
            ),
        )
        q, context = trace.q, trace.context
        episode.true_color.append(agent.vocabulary.label(int(true_color[env_slot])))
        episode.true_shape.append(agent.vocabulary.label(int(true_shape[env_slot])))
        if trace.percept_color_index is not None:
            episode.named_color.append(
                agent.vocabulary.label(int(trace.percept_color_index[env_slot]))
            )
            episode.named_shape.append(
                agent.vocabulary.label(int(trace.percept_shape_index[env_slot]))
            )
            episode.percept_color_probabilities.append(
                trace.percept_color_probabilities[env_slot].tolist()
            )
        episode.action_name.append(
            agent.vocabulary.label(int(trace.action_index[env_slot]))
        )
        episode.action_probabilities.append(trace.action_probabilities[env_slot].tolist())
        episode.value.append(float(trace.value[env_slot]))

        result = environment.step(trace.action_position)
        episode.reward.append(float(result.reward[env_slot]))
        previous_reward = result.reward
        if bool(result.done[env_slot]):
            episode.success = bool(result.collected_target[env_slot])
            break
    return episode


@torch.no_grad()
def value_landscape(
    environment: SymbolicForaging, agent: GridAgent, *, env_slot: int = 0
) -> Float[Tensor, "rows columns"]:
    """Reward-index score with the agent teleported to every cell of one layout.

    This asks whether a *single extra column* of ``A`` learned a spatial value
    function. The layout is frozen; only the agent's position varies, and the
    cognitive state is reinitialized at each cell so the map reflects the
    one-step readout rather than a trajectory.
    """

    config = environment.config
    device = agent.brain.A.device
    state = environment.state
    landscape = torch.zeros(config.size, config.size)
    original_row = state.agent_row.clone()
    original_col = state.agent_col.clone()
    cue_color = agent.color_indices[state.cue_color()]
    cue_shape = agent.shape_indices[state.cue_shape()]
    for row in range(config.size):
        for column in range(config.size):
            state.agent_row = torch.full_like(original_row, row)
            state.agent_col = torch.full_like(original_col, column)
            q, context = agent.initial_state(environment.num_envs, device)
            trace = agent.window_cycle(
                q,
                context,
                environment.observation(),
                torch.zeros(environment.num_envs, device=device),
                cue_color,
                cue_shape,
                is_first_step=torch.ones(
                    environment.num_envs, dtype=torch.bool, device=device
                ),
            )
            landscape[row, column] = float(trace.value[env_slot])
    state.agent_row = original_row
    state.agent_col = original_col
    return landscape


@torch.no_grad()
def index_similarity(agent: GridAgent) -> tuple[Float[Tensor, "indices indices"], list[str]]:
    """Cosine similarity between all index embedding columns of ``A``."""

    columns = torch.nn.functional.normalize(agent.brain.A, dim=0)
    return columns.T @ columns, list(agent.vocabulary.labels)


@torch.no_grad()
def action_alignment(
    agent: GridAgent,
) -> tuple[Float[Tensor, "cues actions"], list[str], list[str]]:
    """How much each cue embedding excites each action index.

    ``a_action^T sigma(a_cue)`` is the score the action layer would give if the
    representation layer contained nothing but the cue embedding. It is the most
    direct readable statement of "this symbol pushes towards that action".
    """

    cue_indices = torch.cat([agent.color_indices, agent.shape_indices])
    cue_states = torch.sigmoid(agent.brain.A[:, cue_indices].T)
    scores = agent.brain.index_scores(cue_states, agent.action_indices)
    cue_labels = [agent.vocabulary.label(int(index)) for index in cue_indices]
    action_labels = [agent.vocabulary.label(int(index)) for index in agent.action_indices]
    return scores, cue_labels, action_labels

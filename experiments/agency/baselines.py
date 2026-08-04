"""Non-Tensor-Brain control policies for the agency gridworld.

These exist to keep the Tensor Brain results honest. ``GRUPolicy`` has the same
inputs, the same optimizer, the same rollout code and the same policy-gradient
loss, but no index layer: the instruction arrives as a *factored one-hot* vector
rather than as index embeddings, and the action distribution comes from an
ordinary linear head rather than from a generative measurement over ``A``.

The factored cue is deliberately the fair control for zero-shot recombination:
a one-hot over the nine cue pairs could not generalize by construction, so it
would make the Tensor Brain look good for the wrong reason.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from experiments.agency.agent import WindowTrace
from experiments.agency.gridworld import NUM_ACTIONS, GridConfig
from experiments.agency.vocabulary import build_vocabulary


class GRUPolicy(nn.Module):
    """A conventional recurrent policy with a value head.

    It exposes the same ``window_cycle`` contract as :class:`GridAgent` so that
    ``run_episodes`` is literally the same code path for both.
    """

    def __init__(self, grid: GridConfig, *, state_dim: int = 64) -> None:
        super().__init__()
        self.grid = grid
        self.state_dim = state_dim
        # Kept so that shared rollout code can build percept targets and so that
        # the control reports the same metrics; it holds no parameters.
        self.vocabulary = build_vocabulary(grid)
        self.cue_dim = grid.num_colors + grid.num_shapes
        self.encoder = nn.Linear(grid.observation_dim + self.cue_dim + 1, state_dim)
        self.cell = nn.GRUCell(state_dim, state_dim)
        self.action_head = nn.Linear(state_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(state_dim, 1)
        self.register_buffer(
            "action_indices", self.vocabulary.indices("action"), persistent=False
        )
        self.register_buffer(
            "color_indices", self.vocabulary.indices("color"), persistent=False
        )
        self.register_buffer(
            "shape_indices", self.vocabulary.indices("shape"), persistent=False
        )
        self.nothing_index = self.vocabulary.index("nothing_visible")

        class _Config:
            measure_percepts = False
            action_selection = "sample"

        self.config = _Config()

    # The control has no Tensor Brain, but the rollout needs these two hooks.
    @property
    def brain(self) -> GRUPolicy:
        return self

    @property
    def A(self) -> Tensor:  # noqa: N802 - mirrors TensorBrain.A for device lookup
        return self.encoder.weight

    def initial_state(
        self, num_envs: int, device: torch.device
    ) -> tuple[Float[Tensor, "envs state"], None]:
        return torch.zeros(num_envs, self.state_dim, device=device), None

    def percept_targets(
        self,
        visible_slot: Int[Tensor, " envs"],
        object_color: Int[Tensor, "envs objects"],
        object_shape: Int[Tensor, "envs objects"],
    ) -> tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]]:
        visible = visible_slot >= 0
        safe_slot = visible_slot.clamp_min(0)[:, None]
        color = self.color_indices[object_color.gather(1, safe_slot).squeeze(1)]
        shape = self.shape_indices[object_shape.gather(1, safe_slot).squeeze(1)]
        nothing = torch.full_like(color, self.nothing_index)
        return torch.where(visible, color, nothing), torch.where(visible, shape, nothing)

    def window_cycle(
        self,
        q: Float[Tensor, "envs state"],
        context: Float[Tensor, "envs context"] | None,
        observation: Float[Tensor, "envs observation"],
        previous_reward: Float[Tensor, " envs"],
        cue_color: Int[Tensor, " envs"],
        cue_shape: Int[Tensor, " envs"],
        *,
        is_first_step: Bool[Tensor, " envs"] | None = None,
        action_teacher: Int[Tensor, " envs"] | None = None,
        percept_teacher: tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]] | None = None,
    ) -> WindowTrace:
        """One recurrent step, matching the Tensor Brain agent's interface."""

        del is_first_step, percept_teacher
        colors = torch.zeros(q.shape[0], self.grid.num_colors, device=q.device)
        shapes = torch.zeros(q.shape[0], self.grid.num_shapes, device=q.device)
        # Global index -> factor position: the colour group occupies the first
        # `num_colors` global indices and the shape group follows it.
        colors.scatter_(1, (cue_color - int(self.color_indices[0]))[:, None], 1.0)
        shapes.scatter_(1, (cue_shape - int(self.shape_indices[0]))[:, None], 1.0)
        features = torch.cat([observation, colors, shapes, previous_reward[:, None]], dim=1)
        state = self.cell(torch.relu(self.encoder(features)), q)
        probabilities = torch.softmax(self.action_head(state), dim=-1)
        if action_teacher is not None:
            position = action_teacher - int(self.action_indices[0])
        else:
            position = torch.distributions.Categorical(probabilities).sample()
        return WindowTrace(
            q=state,
            context=None,
            action_index=self.action_indices[position],
            action_position=position,
            action_probabilities=probabilities,
            value=self.value_head(state).squeeze(-1),
            percept_color_probabilities=None,
            percept_shape_probabilities=None,
            percept_color_index=None,
            percept_shape_index=None,
        )

    def reset_finished(
        self,
        q: Float[Tensor, "envs state"],
        context: Float[Tensor, "envs context"] | None,
        done: Bool[Tensor, " envs"],
    ) -> tuple[Float[Tensor, "envs state"], None]:
        return q * (~done)[:, None].float(), None


class LSTMPolicy(GRUPolicy):
    """LSTM control: asks whether the specific recurrence matters or just memory.

    The GRU control shares this class's inputs and heads; only the recurrent
    cell differs. The cell state is carried inside the same ``q`` tensor as a
    concatenated pair so that the shared rollout contract is unchanged.
    """

    def __init__(self, grid: GridConfig, *, state_dim: int = 64) -> None:
        super().__init__(grid, state_dim=state_dim)
        del self.cell
        self.lstm = nn.LSTMCell(state_dim, state_dim)

    def initial_state(self, num_envs, device):
        return torch.zeros(num_envs, 2 * self.state_dim, device=device), None

    def window_cycle(self, q, context, observation, previous_reward, cue_color, cue_shape,
                     *, is_first_step=None, action_teacher=None, percept_teacher=None):
        del is_first_step, percept_teacher
        colors = torch.zeros(q.shape[0], self.grid.num_colors, device=q.device)
        shapes = torch.zeros(q.shape[0], self.grid.num_shapes, device=q.device)
        colors.scatter_(1, (cue_color - int(self.color_indices[0]))[:, None], 1.0)
        shapes.scatter_(1, (cue_shape - int(self.shape_indices[0]))[:, None], 1.0)
        features = torch.cat([observation, colors, shapes, previous_reward[:, None]], dim=1)
        hidden, cell = q.split(self.state_dim, dim=-1)
        hidden, cell = self.lstm(
            torch.relu(self.encoder(features)),
            (hidden.contiguous(), cell.contiguous()),
        )
        probabilities = torch.softmax(self.action_head(hidden), dim=-1)
        if action_teacher is not None:
            position = action_teacher - int(self.action_indices[0])
        else:
            position = torch.distributions.Categorical(probabilities).sample()
        return WindowTrace(
            q=torch.cat([hidden, cell], dim=-1),
            context=None,
            action_index=self.action_indices[position],
            action_position=position,
            action_probabilities=probabilities,
            value=self.value_head(hidden).squeeze(-1),
            percept_color_probabilities=None,
            percept_shape_probabilities=None,
            percept_color_index=None,
            percept_shape_index=None,
        )

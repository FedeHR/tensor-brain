"""Tensor Brain and capacity-matched recurrent controls for Memory Maze.

All three policies share the *same* convolutional encoder, so the comparison is
between what happens after perception, not between perception front-ends. The
Tensor Brain requires no special treatment: a channel only has to be mapped into
pre-CBS coordinates, and a small CNN does that exactly as a one-hot projection or
a frozen DINO head would.

Parameter counts are reported by the runner and kept within the same order of
magnitude by construction, since the encoder dominates and is shared.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from experiments.agency.agent import AgentConfig, IndexLayout, TensorBrainAgent, WindowTrace
from experiments.agency.memorymaze.env import ACTION_NAMES, IMAGE_SIDE, build_vocabulary


class PixelEncoder(nn.Module):
    r"""``g(nu)`` for pixels: the standard small Atari-style convolutional stack.

    Four strided convolutions over 64x64x3, then a linear map into ``state_dim``.
    This is the encoder used by both the Tensor Brain and the controls.
    """

    def __init__(self, state_dim: int, *, channels: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 4, stride=2), nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 4, stride=2), nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, 4, stride=2), nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            width = self.body(torch.zeros(1, 3, IMAGE_SIDE, IMAGE_SIDE)).shape[-1]
        self.project = nn.Linear(width, state_dim)

    def forward(
        self, observation: Float[Tensor, "*batch observation"]
    ) -> Float[Tensor, "*batch state"]:
        images = observation.reshape(-1, IMAGE_SIDE, IMAGE_SIDE, 3).permute(0, 3, 1, 2)
        return self.project(self.body(images))


class MemoryMazeAgent(TensorBrainAgent):
    """The Tensor Brain agent for Memory Maze."""

    def __init__(self, config: AgentConfig, *, channels: int = 32) -> None:
        super().__init__(
            build_vocabulary(),
            PixelEncoder(config.state_dim, channels=channels),
            IndexLayout(percept_factors=("percept_color", "percept_distance")),
            config,
        )


class RecurrentControl(nn.Module):
    """GRU or LSTM policy with the same encoder and no index layer.

    The instruction arrives as a one-hot over target colours, which is the
    conventional way to goal-condition a recurrent policy, and the action head is
    an ordinary linear layer rather than a generative measurement over ``A``.
    """

    def __init__(
        self, config: AgentConfig, *, cell: str = "gru", channels: int = 32
    ) -> None:
        super().__init__()
        self.config = config
        self.cell_type = cell
        self.state_dim = config.state_dim
        self.vocabulary = build_vocabulary()
        self.encoder = PixelEncoder(config.state_dim, channels=channels)
        self.num_colors = len(self.vocabulary.group_labels("color"))
        self.cue_projection = nn.Linear(self.num_colors + 1, config.state_dim)
        cell_type = nn.GRUCell if cell == "gru" else nn.LSTMCell
        self.recurrent = cell_type(config.state_dim, config.state_dim)
        self.action_head = nn.Linear(config.state_dim, len(ACTION_NAMES))
        self.value_head = nn.Linear(config.state_dim, 1)
        self.register_buffer(
            "action_bank", self.vocabulary.indices("action"), persistent=False
        )
        self._color_base = int(self.vocabulary.indices("color")[0])

    @property
    def action_indices(self) -> Int[Tensor, " indices"]:
        return self.action_bank

    @property
    def brain(self) -> RecurrentControl:
        return self

    @property
    def A(self) -> Tensor:  # noqa: N802 - mirrors TensorBrain.A for device lookup
        return self.action_head.weight

    @property
    def state_width(self) -> int:
        return self.state_dim * (2 if self.cell_type == "lstm" else 1)

    def initial_state(
        self, num_envs: int, device: torch.device
    ) -> tuple[Float[Tensor, "envs state"], None]:
        return torch.zeros(num_envs, self.state_width, device=device), None

    def window_cycle(
        self,
        q: Float[Tensor, "envs state"],
        context: Float[Tensor, "envs context"] | None,
        observation: Float[Tensor, "envs observation"],
        previous_reward: Float[Tensor, " envs"],
        cue_color: Int[Tensor, " envs"],
        cue_other: Int[Tensor, " envs"],
        *,
        is_first_step: Bool[Tensor, " envs"] | None = None,
        action_teacher: Int[Tensor, " envs"] | None = None,
        percept_teacher: tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]] | None = None,
    ) -> WindowTrace:
        del cue_other, is_first_step, percept_teacher
        colors = torch.zeros(q.shape[0], self.num_colors, device=q.device)
        colors.scatter_(1, (cue_color - self._color_base)[:, None], 1.0)
        drive = self.encoder(observation) + self.cue_projection(
            torch.cat([colors, previous_reward[:, None]], dim=1)
        )
        if self.cell_type == "lstm":
            hidden, memory = q.split(self.state_dim, dim=-1)
            hidden, memory = self.recurrent(
                torch.relu(drive), (hidden.contiguous(), memory.contiguous())
            )
            state, packed = hidden, torch.cat([hidden, memory], dim=-1)
        else:
            state = self.recurrent(torch.relu(drive), q)
            packed = state
        probabilities = torch.softmax(self.action_head(state), dim=-1)
        if action_teacher is not None:
            position = action_teacher - int(self.action_bank[0])
        else:
            position = torch.distributions.Categorical(probabilities).sample()
        return WindowTrace(
            q=packed,
            context=None,
            action_index=self.action_bank[position],
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
        del context
        return q * (~done)[:, None].float(), None

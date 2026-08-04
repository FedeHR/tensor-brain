"""The Tensor Brain agent and its controls for MiniGrid.

Only two things change relative to the gridworld study: the vocabulary comes
from MiniGrid's own symbol tables, and ``g(nu)`` embeds the 7x7x3 symbolic view
instead of projecting a one-hot vector. The concept-window schedule, the gates,
the deliberation windows, the reward index and the planning readout are the same
code in ``experiments/agency/agent.py``.
"""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from torch import Tensor, nn

from experiments.agency.agent import AgentConfig, IndexLayout, TensorBrainAgent, WindowTrace
from experiments.agency.minigrid.env import VIEW_SIZE
from experiments.agency.minigrid.vocabulary import build_vocabulary

NUM_DIRECTIONS = 4


class SymbolicViewEncoder(nn.Module):
    r"""``g(nu)`` for MiniGrid: embed the symbolic view, then project to ``q``.

    The observation arrives packed as ``[7*7*3 codes, direction]``. Each of the
    three code channels gets its own embedding table, the three are summed per
    cell -- so a cell's representation is the sum of "what", "what colour" and
    "what state" -- and the flattened grid is concatenated with a facing-direction
    embedding before a single linear map into pre-CBS coordinates.

    This stays firmly outside the Tensor Brain: it is feature extraction, and the
    core only ever sees the resulting ``state_dim`` drive.
    """

    def __init__(self, state_dim: int, *, embed_dim: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.object_embedding = nn.Embedding(len(OBJECT_TO_IDX), embed_dim)
        self.color_embedding = nn.Embedding(len(COLOR_TO_IDX), embed_dim)
        self.state_embedding = nn.Embedding(len(STATE_TO_IDX), embed_dim)
        self.direction_embedding = nn.Embedding(NUM_DIRECTIONS, embed_dim)
        self.project = nn.Linear(VIEW_SIZE * VIEW_SIZE * embed_dim + embed_dim, state_dim)

    def forward(
        self, observation: Float[Tensor, "*batch observation"]
    ) -> Float[Tensor, "*batch state"]:
        codes = observation.long()
        cells = codes[..., :-1].reshape(*codes.shape[:-1], VIEW_SIZE * VIEW_SIZE, 3)
        embedded = (
            self.object_embedding(cells[..., 0])
            + self.color_embedding(cells[..., 1])
            + self.state_embedding(cells[..., 2])
        ).flatten(start_dim=-2)
        direction = self.direction_embedding(codes[..., -1])
        return self.project(torch.cat([embedded, direction], dim=-1))


class MiniGridAgent(TensorBrainAgent):
    """Tensor Brain agent for BabyAI / MiniGrid levels."""

    def __init__(self, config: AgentConfig, *, embed_dim: int = 12) -> None:
        super().__init__(
            build_vocabulary(),
            SymbolicViewEncoder(config.state_dim, embed_dim=embed_dim),
            IndexLayout(percept_factors=("percept_color", "percept_object")),
            config,
        )


class RecurrentControl(nn.Module):
    """GRU or LSTM control with the same inputs and no index layer.

    The instruction is supplied as a *factored* one-hot over colour and object
    type, exactly as in the gridworld study, so the control can in principle
    perform the same zero-shot recombination and the Tensor Brain does not win by
    construction. Actions come from an ordinary linear head rather than from a
    generative measurement over ``A``.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        cell: str = "gru",
        embed_dim: int = 12,
    ) -> None:
        super().__init__()
        self.config = config
        self.cell_type = cell
        self.state_dim = config.state_dim
        self.vocabulary = build_vocabulary()
        self.encoder = SymbolicViewEncoder(config.state_dim, embed_dim=embed_dim)
        self.num_colors = len(self.vocabulary.group_labels("color"))
        self.num_objects = len(self.vocabulary.group_labels("object"))
        num_actions = len(self.vocabulary.group_labels("action"))
        self.cue_projection = nn.Linear(self.num_colors + self.num_objects + 1, config.state_dim)
        if cell == "gru":
            self.recurrent = nn.GRUCell(config.state_dim, config.state_dim)
        else:
            self.recurrent = nn.LSTMCell(config.state_dim, config.state_dim)
        self.action_head = nn.Linear(config.state_dim, num_actions)
        self.value_head = nn.Linear(config.state_dim, 1)
        self.register_buffer(
            "action_bank", self.vocabulary.indices("action"), persistent=False
        )
        self._color_base = int(self.vocabulary.indices("color")[0])
        self._object_base = int(self.vocabulary.indices("object")[0])

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
        cue_object: Int[Tensor, " envs"],
        *,
        is_first_step: Bool[Tensor, " envs"] | None = None,
        action_teacher: Int[Tensor, " envs"] | None = None,
        percept_teacher: tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]] | None = None,
    ) -> WindowTrace:
        del is_first_step, percept_teacher
        colors = torch.zeros(q.shape[0], self.num_colors, device=q.device)
        objects = torch.zeros(q.shape[0], self.num_objects, device=q.device)
        colors.scatter_(1, (cue_color - self._color_base)[:, None], 1.0)
        objects.scatter_(1, (cue_object - self._object_base)[:, None], 1.0)
        drive = self.encoder(observation) + self.cue_projection(
            torch.cat([colors, objects, previous_reward[:, None]], dim=1)
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
        """Clear the recurrent state of environments whose episode just ended."""

        del context
        return q * (~done)[:, None].float(), None

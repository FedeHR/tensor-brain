"""A batched symbolic-foraging gridworld for Tensor Brain agency experiments.

The environment is deliberately small and dependency free so that a complete
ablation grid runs on a laptop CPU. Its purpose is to make the Tensor Brain's
distinctive machinery load bearing rather than decorative:

* the agent only sees an egocentric window, so the task is a POMDP and the
  dynamic context has something to remember;
* the goal is named by a ``(color, shape)`` cue, and every distractor shares at
  most one of the two factors, so success requires the *conjunction* of two
  symbolic cues rather than a single attribute;
* ``collect`` is an explicit action, so the action index group is not merely a
  four-way compass.

Everything is a tensor with a leading ``envs`` axis. There is no per-environment
Python object and no gym dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

MOVE_NORTH: Final = 0
MOVE_SOUTH: Final = 1
MOVE_WEST: Final = 2
MOVE_EAST: Final = 3
COLLECT: Final = 4
NUM_ACTIONS: Final = 5

ACTION_NAMES: Final = ("move_north", "move_south", "move_west", "move_east", "collect")

# (row, column) deltas indexed by action. `collect` does not move the agent.
_ACTION_DELTAS: Final = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


@dataclass(frozen=True)
class GridConfig:
    """Static description of one gridworld task instance."""

    size: int = 6
    num_objects: int = 3
    num_colors: int = 3
    num_shapes: int = 3
    view_radius: int = 2
    max_steps: int = 40
    step_penalty: float = 0.02
    empty_collect_penalty: float = 0.05
    target_reward: float = 1.0
    distractor_reward: float = -0.25
    # Collecting a distractor is penalised but does *not* end the episode. A
    # terminal distractor penalty makes "never collect" a strong local optimum:
    # early in learning the collect action is negative in expectation, it is
    # suppressed, and the agent then never experiences the positive outcome.
    # Keeping the episode alive preserves the discrimination requirement -- the
    # conjunction of both cue factors is still necessary -- while leaving the
    # positive outcome reachable.
    distractor_terminates: bool = False

    def __post_init__(self) -> None:
        if self.num_objects > self.size * self.size - 1:
            raise ValueError("the grid must have a free cell for the agent")
        if self.num_objects > self.num_colors * self.num_shapes:
            raise ValueError("attribute pairs are drawn without replacement")
        if self.view_radius < 0:
            raise ValueError("view_radius must be non-negative")

    @property
    def view_side(self) -> int:
        return 2 * self.view_radius + 1

    @property
    def num_channels(self) -> int:
        # colors, shapes, "an object is here", "this cell is outside the grid".
        return self.num_colors + self.num_shapes + 2

    @property
    def observation_dim(self) -> int:
        return self.view_side * self.view_side * self.num_channels


@dataclass
class GridState:
    """Mutable batched state of ``envs`` independent episodes."""

    agent_row: Int[Tensor, " envs"]
    agent_col: Int[Tensor, " envs"]
    object_row: Int[Tensor, "envs objects"]
    object_col: Int[Tensor, "envs objects"]
    object_color: Int[Tensor, "envs objects"]
    object_shape: Int[Tensor, "envs objects"]
    target_slot: Int[Tensor, " envs"]
    step_count: Int[Tensor, " envs"]

    @property
    def num_envs(self) -> int:
        return int(self.agent_row.shape[0])

    def cue_color(self) -> Int[Tensor, " envs"]:
        return self.object_color.gather(1, self.target_slot[:, None]).squeeze(1)

    def cue_shape(self) -> Int[Tensor, " envs"]:
        return self.object_shape.gather(1, self.target_slot[:, None]).squeeze(1)


@dataclass(frozen=True)
class StepResult:
    """Outcome of one batched environment transition."""

    reward: Float[Tensor, " envs"]
    terminated: Bool[Tensor, " envs"]
    truncated: Bool[Tensor, " envs"]
    collected_target: Bool[Tensor, " envs"]
    collected_distractor: Bool[Tensor, " envs"]

    @property
    def done(self) -> Bool[Tensor, " envs"]:
        return self.terminated | self.truncated


def _random_scores(
    shape: tuple[int, ...], generator: torch.Generator, device: torch.device
) -> Float[Tensor, ...]:
    return torch.rand(shape, generator=generator, device=device)


class SymbolicForaging:
    """Batched cue-conditioned foraging in a partially observed grid.

    ``allowed_cues`` optionally restricts which ``(color, shape)`` pairs may be
    the *target*. Held-out pairs still occur as distractors, so the visual
    statistics of the training and evaluation distributions are identical and
    only the instruction distribution differs.
    """

    def __init__(
        self,
        config: GridConfig,
        num_envs: int,
        *,
        seed: int = 0,
        allowed_cues: frozenset[tuple[int, int]] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.cue_pairs = torch.tensor(
            sorted(
                allowed_cues
                if allowed_cues is not None
                else {
                    (color, shape)
                    for color in range(config.num_colors)
                    for shape in range(config.num_shapes)
                }
            ),
            dtype=torch.long,
            device=self.device,
        )
        self.state = self._sample_episodes(
            torch.ones(num_envs, dtype=torch.bool, device=self.device)
        )

    # ---------------------------------------------------------------- sampling

    def _sample_episodes(self, mask: Bool[Tensor, " envs"]) -> GridState:
        """Sample fresh layouts for the environments selected by ``mask``.

        The cue pair is drawn first, uniformly over the allowed split, and the
        distractor pairs are then drawn without replacement from the remaining
        product set. Sampling in this order keeps the *instruction* distribution
        exactly uniform over the allowed cues while guaranteeing that no
        distractor shares both factors with the target.
        """

        config = self.config
        num_envs = self.num_envs
        cells = _random_scores(
            (num_envs, config.size * config.size), self.generator, self.device
        ).argsort(dim=1)[:, : config.num_objects + 1]

        num_pairs = config.num_colors * config.num_shapes
        cue_choice = (
            _random_scores((num_envs, len(self.cue_pairs)), self.generator, self.device)
            .argmax(dim=1)
        )
        cue = self.cue_pairs[cue_choice]
        cue_flat = cue[:, 0] * config.num_shapes + cue[:, 1]

        # Distractor pairs: rank all product-set pairs, push the cue pair to the
        # end so it cannot be redrawn, and take the leading slots.
        pair_scores = _random_scores((num_envs, num_pairs), self.generator, self.device)
        pair_scores.scatter_(1, cue_flat[:, None], 2.0)
        distractors = pair_scores.argsort(dim=1)[:, : config.num_objects - 1]
        pairs_flat = torch.cat([cue_flat[:, None], distractors], dim=1)

        # Place the target, currently in slot 0, at a uniformly random slot.
        permutation = _random_scores(
            (num_envs, config.num_objects), self.generator, self.device
        ).argsort(dim=1)
        pairs_flat = pairs_flat.gather(1, permutation)
        target_slot = (permutation == 0).long().argmax(dim=1)

        fresh = GridState(
            agent_row=cells[:, 0] // config.size,
            agent_col=cells[:, 0] % config.size,
            object_row=cells[:, 1:] // config.size,
            object_col=cells[:, 1:] % config.size,
            object_color=pairs_flat // config.num_shapes,
            object_shape=pairs_flat % config.num_shapes,
            target_slot=target_slot,
            step_count=torch.zeros(num_envs, dtype=torch.long, device=self.device),
        )
        if bool(mask.all()):
            return fresh
        return self._merge(self.state, fresh, mask.to(self.device))

    @staticmethod
    def _merge(old: GridState, new: GridState, mask: Bool[Tensor, " envs"]) -> GridState:
        def pick(old_value: Tensor, new_value: Tensor) -> Tensor:
            selector = mask if old_value.ndim == 1 else mask[:, None]
            return torch.where(selector, new_value, old_value)

        return GridState(
            *(
                pick(getattr(old, field), getattr(new, field))
                for field in (
                    "agent_row",
                    "agent_col",
                    "object_row",
                    "object_col",
                    "object_color",
                    "object_shape",
                    "target_slot",
                    "step_count",
                )
            )
        )

    def reset(self) -> None:
        """Resample every episode."""

        self.state = self._sample_episodes(
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        )

    def reset_done(self, done: Bool[Tensor, " envs"]) -> None:
        """Resample only the finished episodes, leaving the others untouched."""

        if bool(done.any()):
            self.state = self._sample_episodes(done)

    # ------------------------------------------------------------- observation

    def observation(self) -> Float[Tensor, "envs observation"]:
        """Egocentric one-hot view, flattened.

        Channel layout per cell: ``num_colors`` colour channels, ``num_shapes``
        shape channels, one object-present channel, one out-of-bounds channel.
        The agent's absolute position is never encoded.
        """

        config = self.config
        state = self.state
        side = config.view_side
        offsets = torch.arange(-config.view_radius, config.view_radius + 1, device=self.device)
        rows = state.agent_row[:, None] + offsets  # [envs, side]
        cols = state.agent_col[:, None] + offsets

        # [envs, side, side] absolute coordinates of every viewed cell.
        view_rows = rows[:, :, None].expand(-1, -1, side)
        view_cols = cols[:, None, :].expand(-1, side, -1)
        inside = (
            (view_rows >= 0) & (view_rows < config.size)
            & (view_cols >= 0) & (view_cols < config.size)
        )

        view = torch.zeros(
            self.num_envs, side, side, config.num_channels, device=self.device
        )
        view[..., -1] = (~inside).float()

        # Object hit test: [envs, objects, side, side].
        matches = (
            (state.object_row[:, :, None, None] == view_rows[:, None])
            & (state.object_col[:, :, None, None] == view_cols[:, None])
        )
        env_ids, slot_ids, row_ids, col_ids = matches.nonzero(as_tuple=True)
        reported_color, reported_shape = self.observed_attributes()
        colors = reported_color[env_ids, slot_ids]
        shapes = reported_shape[env_ids, slot_ids]
        view[env_ids, row_ids, col_ids, colors] = 1.0
        view[env_ids, row_ids, col_ids, config.num_colors + shapes] = 1.0
        view[env_ids, row_ids, col_ids, config.num_colors + config.num_shapes] = 1.0
        return view.reshape(self.num_envs, -1)

    def observed_attributes(
        self,
    ) -> tuple[Int[Tensor, "envs objects"], Int[Tensor, "envs objects"]]:
        """Attributes as *reported* by the sensor; here, the true ones.

        Subclasses corrupt this to make perception unreliable without
        duplicating the view-rendering logic.
        """

        return self.state.object_color, self.state.object_shape

    def visible_object_slot(self) -> Int[Tensor, " envs"]:
        """Slot of the nearest visible object, or ``-1`` when none is in view.

        This is the gridworld's stand-in for the paper's serial region-of-interest
        selection: exactly one object is attended per concept-window cycle.
        Ties are broken by slot order, which is already a random permutation.
        """

        state = self.state
        row_gap = (state.object_row - state.agent_row[:, None]).abs()
        col_gap = (state.object_col - state.agent_col[:, None]).abs()
        chebyshev = torch.maximum(row_gap, col_gap)
        visible = chebyshev <= self.config.view_radius
        distance = chebyshev.masked_fill(~visible, self.config.size * 2)
        nearest = distance.argmin(dim=1)
        return torch.where(visible.any(dim=1), nearest, torch.full_like(nearest, -1))

    # -------------------------------------------------------------- transition

    def step(self, action: Int[Tensor, " envs"]) -> StepResult:
        """Apply one action per environment and return the transition outcome."""

        config = self.config
        state = self.state
        action = action.to(self.device)
        deltas = torch.tensor(_ACTION_DELTAS, device=self.device)[action]
        state.agent_row = (state.agent_row + deltas[:, 0]).clamp(0, config.size - 1)
        state.agent_col = (state.agent_col + deltas[:, 1]).clamp(0, config.size - 1)
        state.step_count = state.step_count + 1

        on_object = (
            (state.object_row == state.agent_row[:, None])
            & (state.object_col == state.agent_col[:, None])
        )
        is_collect = action == COLLECT
        collected_any = is_collect & on_object.any(dim=1)
        on_target = on_object.gather(1, state.target_slot[:, None]).squeeze(1)
        collected_target = collected_any & on_target
        collected_distractor = collected_any & ~on_target

        reward = torch.full(
            (self.num_envs,), -config.step_penalty, device=self.device
        )
        reward = reward + collected_target.float() * config.target_reward
        reward = reward + collected_distractor.float() * config.distractor_reward
        reward = reward - (is_collect & ~collected_any).float() * config.empty_collect_penalty

        terminated = collected_target | (collected_distractor & config.distractor_terminates)
        truncated = (state.step_count >= config.max_steps) & ~terminated
        return StepResult(reward, terminated, truncated, collected_target, collected_distractor)

    # ------------------------------------------------------------------ oracle

    def oracle_action(self) -> Int[Tensor, " envs"]:
        """Privileged greedy action towards the cued object.

        The oracle sees the true target position, which the agent does not. It
        exists only to provide behavioural-cloning targets and an upper
        reference for episode length; it is never used as a policy baseline.
        """

        state = self.state
        target_row = state.object_row.gather(1, state.target_slot[:, None]).squeeze(1)
        target_col = state.object_col.gather(1, state.target_slot[:, None]).squeeze(1)
        row_gap = target_row - state.agent_row
        col_gap = target_col - state.agent_col

        action = torch.full_like(row_gap, COLLECT)
        # Close the larger coordinate gap first; ties resolve to the row move.
        move_row = row_gap.abs() >= col_gap.abs()
        row_action = torch.where(
            row_gap < 0, torch.full_like(row_gap, MOVE_NORTH), torch.full_like(row_gap, MOVE_SOUTH)
        )
        col_action = torch.where(
            col_gap < 0, torch.full_like(col_gap, MOVE_WEST), torch.full_like(col_gap, MOVE_EAST)
        )
        moving = (row_gap != 0) | (col_gap != 0)
        return torch.where(moving, torch.where(move_row, row_action, col_action), action)


def latin_square_holdout(num_colors: int, num_shapes: int) -> frozenset[tuple[int, int]]:
    """Return the diagonal ``(color, shape)`` pairs held out from training cues.

    A diagonal keeps every colour and every shape present in the training cue
    set, so a held-out pair is unseen only as a *combination*. That is exactly
    the zero-shot recombination question for the shared index layer.
    """

    side = min(num_colors, num_shapes)
    return frozenset((index, index) for index in range(side))


def train_cues(num_colors: int, num_shapes: int) -> frozenset[tuple[int, int]]:
    """All cue pairs except the held-out diagonal."""

    holdout = latin_square_holdout(num_colors, num_shapes)
    return frozenset(
        (color, shape)
        for color in range(num_colors)
        for shape in range(num_shapes)
        if (color, shape) not in holdout
    )

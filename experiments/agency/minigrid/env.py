"""A batched adapter from Gymnasium MiniGrid environments to the agent contract.

MiniGrid environments are per-instance Python objects, while the Tensor Brain
agent consumes batched tensors. This module keeps a list of environments, steps
them together, auto-resets the finished ones, and exposes exactly the tensors
``window_cycle`` needs: a packed symbolic observation, the two cue indices
parsed from the mission, and the perceptual labels of the attended cell.

Nothing here is privileged. The perceptual target is read out of the agent's own
7x7 egocentric view, not from the simulator state, so the naming task is
genuinely solvable from what the agent sees.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from minigrid.core.constants import OBJECT_TO_IDX
from torch import Tensor

from experiments.agency.minigrid.vocabulary import (
    NOTHING,
    build_vocabulary,
    object_label,
    parse_mission,
)
from tb import IndexVocabulary

VIEW_SIZE = 7
# The agent occupies the bottom-centre cell of its own egocentric view and looks
# towards decreasing y.
AGENT_VIEW_CELL = (VIEW_SIZE // 2, VIEW_SIZE - 1)

# Codes that count as "an object worth naming" when choosing the attended cell.
_ATTENDABLE = tuple(
    OBJECT_TO_IDX[name] for name in ("key", "ball", "box", "door", "goal", "lava")
)


@dataclass(frozen=True)
class MiniGridStep:
    """Outcome of one batched transition."""

    reward: Float[Tensor, " envs"]
    terminated: Bool[Tensor, " envs"]
    truncated: Bool[Tensor, " envs"]
    success: Bool[Tensor, " envs"]

    @property
    def done(self) -> Bool[Tensor, " envs"]:
        return self.terminated | self.truncated


class VectorMiniGrid:
    """A fixed-size batch of MiniGrid environments with automatic reset.

    ``allowed_cues`` optionally restricts which ``(colour, object)`` missions may
    appear, by resampling the layout until an allowed mission is generated. The
    *world* distribution is unchanged -- excluded objects still populate the room
    as distractors -- so only the instruction distribution differs, which is the
    same compositional-split construction used in the gridworld study.
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        *,
        seed: int = 0,
        allowed_cues: frozenset[tuple[str, str]] | None = None,
        render: bool = False,
        max_steps: int | None = None,
    ) -> None:
        self.env_id = env_id
        self.num_envs = num_envs
        self.allowed_cues = allowed_cues
        self.vocabulary: IndexVocabulary = build_vocabulary()
        options = {"render_mode": "rgb_array"} if render else {}
        if max_steps is not None:
            options["max_steps"] = max_steps
        self.envs = [gym.make(env_id, **options) for _ in range(num_envs)]
        self.num_actions = int(self.envs[0].action_space.n)
        self._rng = np.random.default_rng(seed)
        self._images = np.zeros((num_envs, VIEW_SIZE, VIEW_SIZE, 3), dtype=np.int64)
        self._directions = np.zeros(num_envs, dtype=np.int64)
        self._cue_color = np.zeros(num_envs, dtype=np.int64)
        self._cue_object = np.zeros(num_envs, dtype=np.int64)
        self._steps = np.zeros(num_envs, dtype=np.int64)

        labels = self.vocabulary
        self._color_bank = labels.indices("color").numpy()
        self._object_bank = labels.indices("object").numpy()
        self._nothing = labels.index(NOTHING)
        self._percept_color_of = {
            name: labels.index(name) for name in labels.group_labels("percept_color")
        }
        self._percept_object_of = {
            name: labels.index(name) for name in labels.group_labels("percept_object")
        }
        self._cue_color_of = {
            name: labels.index(name) for name in labels.group_labels("color")
        }
        self._cue_object_of = {
            name: labels.index(name) for name in labels.group_labels("object")
        }
        self.reset()
        # BabyAI levels compute their step budget during `reset`, so this must be
        # read afterwards; before the first reset it is still zero.
        self.max_steps = int(self.envs[0].unwrapped.max_steps)

    # ---------------------------------------------------------------- resets

    def _reset_one(self, slot: int) -> None:
        """Reset one environment, resampling until its mission is allowed."""

        for _ in range(64):
            # BabyAI's own rejection sampling prints to stdout when it discards a
            # layout; that is normal operation, not a warning worth surfacing.
            with contextlib.redirect_stdout(io.StringIO()):
                observation, _ = self.envs[slot].reset(
                    seed=int(self._rng.integers(2**31))
                )
            cue = parse_mission(str(observation["mission"]))
            if self.allowed_cues is None or (cue.color, cue.object_type) in self.allowed_cues:
                break
        self._images[slot] = observation["image"]
        self._directions[slot] = int(observation["direction"])
        self._cue_color[slot] = self._cue_color_of[cue.color]
        self._cue_object[slot] = self._cue_object_of[cue.object_type]
        self._steps[slot] = 0

    def reset(self) -> None:
        for slot in range(self.num_envs):
            self._reset_one(slot)

    # ----------------------------------------------------------- observation

    def observation(self) -> Float[Tensor, "envs observation"]:
        """Packed symbolic view: 7x7x3 integer codes plus the facing direction.

        The codes are returned as floats only because the agent boundary is
        float-typed; they are small integers and the encoder casts them back
        before embedding, so nothing is quantized away.
        """

        flat = self._images.reshape(self.num_envs, -1)
        return torch.from_numpy(
            np.concatenate([flat, self._directions[:, None]], axis=1)
        ).float()

    @property
    def observation_dim(self) -> int:
        return VIEW_SIZE * VIEW_SIZE * 3 + 1

    def cue_indices(self) -> tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]]:
        """Global index of the mission's colour and object type."""

        return (
            torch.from_numpy(self._cue_color.copy()),
            torch.from_numpy(self._cue_object.copy()),
        )

    def percept_targets(self) -> tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]]:
        """Labels of the nearest attendable cell in the agent's own view.

        This is the MiniGrid analogue of the paper's serial region-of-interest
        selection: one cell is attended per concept-window cycle and named. When
        no object is in view both labels are ``nothing_visible``.
        """

        colors = np.full(self.num_envs, self._nothing, dtype=np.int64)
        objects = np.full(self.num_envs, self._nothing, dtype=np.int64)
        columns, rows = np.meshgrid(
            np.arange(VIEW_SIZE), np.arange(VIEW_SIZE), indexing="ij"
        )
        distance = np.abs(columns - AGENT_VIEW_CELL[0]) + np.abs(rows - AGENT_VIEW_CELL[1])
        for slot in range(self.num_envs):
            codes = self._images[slot, :, :, 0]
            visible = np.isin(codes, _ATTENDABLE)
            if not visible.any():
                continue
            masked = np.where(visible, distance, VIEW_SIZE * 4)
            column, row = np.unravel_index(masked.argmin(), masked.shape)
            name = object_label(int(codes[column, row]))
            if name is None:
                continue
            objects[slot] = self._percept_object_of[name]
            colour_code = int(self._images[slot, column, row, 1])
            colour_name = self.vocabulary.group_labels("color")[colour_code]
            colors[slot] = self._percept_color_of[colour_name]
        return torch.from_numpy(colors), torch.from_numpy(objects)

    # ------------------------------------------------------------ transition

    def step(self, actions: Int[Tensor, " envs"]) -> MiniGridStep:
        """Apply one action per environment; finished episodes reset immediately."""

        chosen = actions.detach().cpu().numpy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for slot in range(self.num_envs):
            observation, reward, term, trunc, _ = self.envs[slot].step(int(chosen[slot]))
            rewards[slot] = float(reward)
            terminated[slot] = bool(term)
            truncated[slot] = bool(trunc)
            self._steps[slot] += 1
            if term or trunc:
                self._reset_one(slot)
            else:
                self._images[slot] = observation["image"]
                self._directions[slot] = int(observation["direction"])
        # MiniGrid pays a positive, step-discounted reward only on success.
        success = terminated & (rewards > 0.0)
        return MiniGridStep(
            torch.from_numpy(rewards),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
            torch.from_numpy(success),
        )

    def render(self, slot: int = 0) -> np.ndarray:
        """RGB frame of one environment, for trajectory figures."""

        return self.envs[slot].render()

    def mission(self, slot: int = 0) -> str:
        return str(self.envs[slot].unwrapped.mission)

    def close(self) -> None:
        for environment in self.envs:
            environment.close()


def cue_combinations(env_id: str, *, samples: int = 400, seed: int = 0) -> set[tuple[str, str]]:
    """Discover which ``(colour, object)`` missions a level actually generates.

    The compositional split has to be built from the level's real instruction
    distribution rather than from the full product set, because BabyAI levels
    differ in which object types they place.
    """

    environment = gym.make(env_id)
    rng = np.random.default_rng(seed)
    found: set[tuple[str, str]] = set()
    for _ in range(samples):
        with contextlib.redirect_stdout(io.StringIO()):
            observation, _ = environment.reset(seed=int(rng.integers(2**31)))
        cue = parse_mission(str(observation["mission"]))
        found.add((cue.color, cue.object_type))
    environment.close()
    return found


def diagonal_holdout(combinations: set[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """Hold out one deterministic combination per object type.

    Every colour and every object type remains present in the training cue set,
    so a held-out mission is unseen only as a *combination* -- the zero-shot
    recombination test.
    """

    by_object: dict[str, list[str]] = {}
    for color, object_type in sorted(combinations):
        by_object.setdefault(object_type, []).append(color)
    holdout = set()
    for position, (object_type, colors) in enumerate(sorted(by_object.items())):
        if len(colors) > 1:
            holdout.add((colors[position % len(colors)], object_type))
    return frozenset(holdout)

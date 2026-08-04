"""Batched adapter for DeepMind's Memory Maze.

Memory Maze is the benchmark this study needs because it ships ground truth for
*what the agent should remember*. The ``ExtraObs`` variants expose the agent's
position and heading, the positions of all three coloured targets, and the maze
layout, none of which the agent ever sees. That turns "does the architecture
retain a usable belief" from an assertion into a measurement.

The adapter exposes exactly the contract ``experiments/agency/minigrid/ppo.py``
already consumes, so the same recurrent PPO trains the Tensor Brain and the
controls here without modification.

Environment recipe, which is fiddly enough to be worth recording:

* Python **3.12** -- ``labmaze`` (a ``dm_control`` dependency) has no 3.13 wheel.
* the legacy ``gym`` package, not ``gymnasium``; Memory Maze registers there.
* ``MUJOCO_GL=glfw`` -- offscreen rendering works headless on macOS this way,
  while ``egl`` and ``osmesa`` are unavailable.
* ``np.bool8`` must be shimmed and ``disable_env_checker=True`` passed, because
  gym 0.26's passive checker predates NumPy 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from tb import IndexVocabulary

os.environ.setdefault("MUJOCO_GL", "glfw")
if not hasattr(np, "bool8"):  # pragma: no cover - NumPy 2 compatibility shim
    np.bool8 = np.bool_

IMAGE_SIDE = 64
NUM_TARGETS = 3
COLOR_NAMES = ("red", "green", "blue")
DISTANCE_NAMES = ("near", "far")
NOTHING = "nothing_visible"
REWARD_POSITIVE = "reward_positive"
ACTION_NAMES = ("noop", "forward", "back", "left", "right", "turn")
NEAR_RADIUS = 3.0


def build_vocabulary() -> IndexVocabulary:
    """Index layout: three target colours, six actions, one reward index.

    The colour columns are shared between the *instruction* -- which target the
    maze is currently asking for -- and the perceptual naming of which target is
    nearest, which is the property under test.
    """

    return IndexVocabulary.from_groups(
        {
            "color": COLOR_NAMES,
            "percept_color": (*COLOR_NAMES, NOTHING),
            "percept_distance": (*DISTANCE_NAMES, NOTHING),
            "action": ACTION_NAMES,
            "reward": (REWARD_POSITIVE,),
        }
    )


@dataclass(frozen=True)
class MazeStep:
    reward: Float[Tensor, " envs"]
    terminated: Bool[Tensor, " envs"]
    truncated: Bool[Tensor, " envs"]
    success: Bool[Tensor, " envs"]

    @property
    def done(self) -> Bool[Tensor, " envs"]:
        return self.terminated | self.truncated


class VectorMemoryMaze:
    """A fixed-size batch of Memory Maze environments with automatic reset."""

    def __init__(
        self,
        num_envs: int,
        *,
        level: str = "memory_maze:MemoryMaze-9x9-ExtraObs-v0",
        seed: int = 0,
        max_steps: int = 1000,
    ) -> None:
        import gym  # imported lazily so the rest of the package stays importable

        self.level = level
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.vocabulary = build_vocabulary()
        # Memory Maze's wrapper predates the seeded-reset API, so per-environment
        # variation comes from seeding the global RNG before each construction.
        self.envs = []
        for index in range(num_envs):
            np.random.seed(seed + index)
            self.envs.append(gym.make(level, disable_env_checker=True))
        self.num_actions = int(self.envs[0].action_space.n)
        self._rng = np.random.default_rng(seed)
        self._observations: list[dict] = [{} for _ in range(num_envs)]
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self._color_bank = self.vocabulary.indices("color").numpy()
        self._percept_color = {
            name: self.vocabulary.index(name) for name in (*COLOR_NAMES, NOTHING)
        }
        self._percept_distance = {
            name: self.vocabulary.index(name) for name in (*DISTANCE_NAMES, NOTHING)
        }
        self.reset()

    # ---------------------------------------------------------------- resets

    def _store(self, slot: int, observation) -> None:
        self._observations[slot] = dict(observation)

    def _reset_one(self, slot: int) -> None:
        observation = self.envs[slot].reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        self._store(slot, observation)
        self._steps[slot] = 0

    def reset(self) -> None:
        for slot in range(self.num_envs):
            self._reset_one(slot)

    # ----------------------------------------------------------- observation

    def observation(self) -> Float[Tensor, "envs observation"]:
        """Flattened 64x64 RGB in [0, 1]; the encoder reshapes it back."""

        images = np.stack(
            [item["image"] for item in self._observations]
        ).astype(np.float32) / 255.0
        return torch.from_numpy(images.reshape(self.num_envs, -1))

    @property
    def observation_dim(self) -> int:
        return IMAGE_SIDE * IMAGE_SIDE * 3

    def _target_slots(self) -> np.ndarray:
        """Which of the three targets is currently being asked for.

        ``target_color`` is an RGB triple for one of three saturated colours, so
        its argmax identifies the colour without needing a lookup table. The
        assertion below fails loudly if a level ever violates that.
        """

        colors = np.stack([item["target_color"] for item in self._observations])
        assert colors.shape[1] == NUM_TARGETS
        return colors.argmax(axis=1)

    def cue_indices(self) -> tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]]:
        """The instruction: the global index of the requested target colour.

        Both returned tensors are the colour index; Memory Maze's instruction has
        a single factor, so the schedule's second cue slot repeats it rather than
        inventing a spurious second factor.
        """

        colour = torch.from_numpy(self._color_bank[self._target_slots()])
        return colour, colour.clone()

    # ------------------------------------------------------------ ground truth

    def ground_truth(self) -> dict[str, Tensor]:
        """Everything the agent never observes, for probing."""

        def stack(key: str) -> Tensor:
            return torch.from_numpy(
                np.stack([item[key] for item in self._observations]).astype(np.float32)
            )

        return {
            "agent_pos": stack("agent_pos"),
            "agent_dir": stack("agent_dir"),
            "target_pos": stack("target_pos"),
            "targets_pos": stack("targets_pos").reshape(self.num_envs, -1),
            "target_vec": stack("target_vec"),
            "target_slot": torch.from_numpy(self._target_slots()).long(),
        }

    def percept_targets(self) -> tuple[Int[Tensor, " envs"], Int[Tensor, " envs"]]:
        """Labels for the nearest target and whether it is close.

        These are *probe labels* derived from ground truth. They are never fed to
        the agent; the agent must recover them from pixels if it is to name them.
        """

        agent = np.stack([item["agent_pos"] for item in self._observations])
        targets = np.stack([item["targets_pos"] for item in self._observations])
        distances = np.linalg.norm(targets - agent[:, None, :], axis=-1)
        nearest = distances.argmin(axis=1)
        closest = distances.min(axis=1)
        colours = np.array(
            [self._percept_color[COLOR_NAMES[slot]] for slot in nearest], dtype=np.int64
        )
        bands = np.where(
            closest <= NEAR_RADIUS,
            self._percept_distance["near"],
            self._percept_distance["far"],
        )
        return torch.from_numpy(colours), torch.from_numpy(bands.astype(np.int64))

    # ------------------------------------------------------------ transition

    def step(self, actions: Int[Tensor, " envs"]) -> MazeStep:
        chosen = actions.detach().cpu().numpy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        done = np.zeros(self.num_envs, dtype=bool)
        for slot in range(self.num_envs):
            outcome = self.envs[slot].step(int(chosen[slot]))
            observation, reward = outcome[0], float(outcome[1])
            finished = bool(outcome[2])
            rewards[slot] = reward
            self._steps[slot] += 1
            truncated = self._steps[slot] >= self.max_steps
            if finished or truncated:
                done[slot] = True
                self._reset_one(slot)
            else:
                self._store(slot, observation)
        # Memory Maze pays +1 each time the requested target is reached; there is
        # no terminal success, so "success" is the per-step pickup event.
        success = rewards > 0.0
        return MazeStep(
            torch.from_numpy(rewards),
            torch.from_numpy(done),
            torch.zeros(self.num_envs, dtype=torch.bool),
            torch.from_numpy(success),
        )

    def render(self, slot: int = 0) -> np.ndarray:
        return self._observations[slot]["image"]

    def close(self) -> None:
        for environment in self.envs:
            environment.close()

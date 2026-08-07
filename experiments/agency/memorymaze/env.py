"""Batched adapter for DeepMind's Memory Maze.

Memory Maze is the benchmark this study needs because it ships ground truth for
*what the agent should remember*. The ``ExtraObs`` variants expose the agent's
position and heading, the positions of all three coloured targets, and the maze
layout, none of which the agent ever sees. That turns "does the architecture
retain a usable belief" from an assertion into a measurement.

The adapter exposes exactly the contract ``experiments/agency/minigrid/ppo.py``
already consumes, so the same recurrent PPO trains the Tensor Brain and the
controls here without modification.

This adapter talks to Memory Maze through its **native ``dm_env`` interface**
rather than through the ``gym`` environments the package also registers. That is
worth a note, because the obvious move -- migrating to ``gymnasium``, which is
the maintained replacement for ``gym`` -- does not work here:

* ``memory_maze`` imports ``gym`` specifically, registers its levels in *gym's*
  registry, and its ``GymWrapper`` subclasses ``gym.Env`` with the pre-0.26
  four-tuple ``step``. ``gymnasium.make`` would not find the levels, and the
  wrapper would not satisfy gymnasium's API. Version 1.0.3 is the latest
  release, so this is not something a version bump fixes.
* The ``gym`` layer is a thin convenience over a ``dm_env`` that this adapter
  re-wraps anyway, so bypassing it costs nothing and removes the dependency from
  our code path entirely -- along with ``gym.make``, the passive environment
  checker, and the ``np.bool8`` shim that checker needed under NumPy 2.
* It also buys correctness: ``dm_env`` task factories take a ``seed`` argument,
  so each environment in the batch is seeded properly instead of by reseeding
  the global NumPy RNG before construction.

``gym`` nonetheless remains an *installed* dependency, because
``memory_maze/__init__.py`` imports it unconditionally and re-raises if it is
missing. Nothing in this repository imports it.

The remaining environment recipe:

* Python **3.12** -- ``labmaze`` (a ``dm_control`` dependency) has no 3.13 wheel.
* a MuJoCo rendering backend, which differs by platform and is the one setting
  that does not travel: ``glfw`` on macOS, where ``egl`` and ``osmesa`` are
  unavailable, and ``egl`` on a headless Linux node, where ``glfw`` needs a
  display that a batch job does not have. ``MUJOCO_GL`` is only defaulted here,
  so a caller or an ``sbatch`` script can override it; ``osmesa`` is the
  software fallback for a node without a usable GPU.

Rendering is also the throughput limit, and it does not parallelise inside a
process: the environments are stepped in one Python loop, so a batch of eight
runs no faster in wall-clock terms than a batch of one (measured: ~205
environment steps per second either way, on an M3 Pro). Scale comes from running
separate processes -- one Slurm array task per condition and seed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from tb import IndexVocabulary

# `setdefault`, so that a batch script can select `osmesa` on a node whose GPU
# has no EGL device without editing the module. This must happen before
# `memory_maze` is imported, which is why the import below is deferred.
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

IMAGE_SIDE = 64
NUM_TARGETS = 3
COLOR_NAMES = ("red", "green", "blue")
DISTANCE_NAMES = ("near", "far")
NOTHING = "nothing_visible"
REWARD_POSITIVE = "reward_positive"
# The six discrete actions, in the order `memory_maze.tasks` builds them:
# `DiscreteActionSetWrapper(env, [[0,0], [-1,0], [0,-1], [0,+1], [-1,-1], [-1,+1]])`.
# The walker is a rolling ball driven by torques, not a grid mover, so these are
# accelerations and the ball carries momentum between steps.
ACTION_NAMES = ("noop", "forward", "left", "right", "forward_left", "forward_right")
NEAR_RADIUS = 3.0

# The maze sizes Memory Maze ships, named by the `memory_maze.tasks` factory that
# builds each one. Every level in this study is built with
# `global_observables=True`, which is what the registered `ExtraObs` ids select
# and what exposes the ground truth the probe consumes.
LEVEL_TASKS = {
    "9x9": "memory_maze_9x9",
    "11x11": "memory_maze_11x11",
    "13x13": "memory_maze_13x13",
    "15x15": "memory_maze_15x15",
}


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
        level: str = "9x9",
        seed: int = 0,
        max_steps: int = 1000,
    ) -> None:
        # Deferred so that `MUJOCO_GL` above is set before MuJoCo initialises,
        # and so the rest of the package stays importable without a maze install.
        from memory_maze import tasks

        if level not in LEVEL_TASKS:
            raise KeyError(f"unknown level {level!r}; expected one of {sorted(LEVEL_TASKS)}")
        self.level = level
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.vocabulary = build_vocabulary()
        # Each environment is seeded through the task factory, so the batch is
        # reproducible without touching the global NumPy RNG.
        build = getattr(tasks, LEVEL_TASKS[level])
        self.envs = [
            build(global_observables=True, seed=seed + index) for index in range(num_envs)
        ]
        self.num_actions = int(self.envs[0].action_spec().num_values)
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
        # `dm_env.Environment.reset` returns a TimeStep, whose `observation` is
        # the dict of image plus global observables.
        self._store(slot, self.envs[slot].reset().observation)
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

    def raw_observations(self) -> list[dict]:
        """The per-environment observation dicts exactly as Memory Maze emits them.

        The corpus recorder needs fields this adapter does not otherwise surface
        (``targets_vec``, ``maze_layout``), and needs them without the float cast
        ``ground_truth`` applies, so it reads the dicts directly.
        """

        return self._observations

    def target_slots(self) -> np.ndarray:
        """Which of the three targets is currently requested, as a slot index."""

        return self._target_slots()

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
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        for slot in range(self.num_envs):
            timestep = self.envs[slot].step(int(chosen[slot]))
            # `reward` is None on a TimeStep that follows a reset; everywhere
            # else it is a float.
            rewards[slot] = 0.0 if timestep.reward is None else float(timestep.reward)
            self._steps[slot] += 1
            # The level's own time limit ends the episode; `max_steps` is this
            # adapter's separate cap, so the two are reported apart.
            terminated[slot] = bool(timestep.last())
            truncated[slot] = self._steps[slot] >= self.max_steps
            if terminated[slot] or truncated[slot]:
                self._reset_one(slot)
            else:
                self._store(slot, timestep.observation)
        # Memory Maze pays +1 each time the requested target is reached; there is
        # no terminal success, so "success" is the per-step pickup event.
        success = rewards > 0.0
        return MazeStep(
            torch.from_numpy(rewards),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
            torch.from_numpy(success),
        )

    def render(self, slot: int = 0) -> np.ndarray:
        return self._observations[slot]["image"]

    def close(self) -> None:
        for environment in self.envs:
            environment.close()

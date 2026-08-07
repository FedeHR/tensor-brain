"""The scripted behaviour policy that generates the offline corpus.

This is not an agent and is never evaluated. Its only job is to produce a
trajectory distribution that covers the maze, so that a filter trained on the
recording has something worth remembering. It is thrown away after recording.

Why a scripted explorer rather than a uniform random policy: the walker is a
rolling ball driven by torques, so a policy that redraws an action every step
averages its own inputs to nearly zero and jitters in place. Coverage matters
here more than anywhere, because the memory-horizon curve is measured in "steps
since a target was last visible" -- a policy that never leaves its starting room
puts every sample in the "never seen" bucket and the curve has no domain.

The remedy is momentum: draw an action, hold it for a geometrically distributed
dwell, then redraw. Held actions integrate into sustained motion, and the dwell
length sets the trade-off between distance covered and turning often enough to
enter new corridors.

The action weights favour the forward-ish actions over pure turns, and give
``noop`` a small share so that the corpus contains stationary steps -- without
them a filter could learn "the view always changes" as a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from experiments.agency.memorymaze.env import ACTION_NAMES

# Indexed by `ACTION_NAMES`: noop, forward, left, right, forward_left, forward_right.
DEFAULT_WEIGHTS: tuple[float, ...] = (0.05, 0.35, 0.12, 0.12, 0.18, 0.18)


@dataclass(frozen=True)
class ExplorerConfig:
    """Every controlled variable of the behaviour policy."""

    # Mean number of steps an action is held. At 4 Hz control, 6 steps is about
    # 1.5 seconds of sustained torque, which is roughly one corridor segment.
    dwell_mean: float = 6.0
    weights: tuple[float, ...] = DEFAULT_WEIGHTS

    def __post_init__(self) -> None:
        if len(self.weights) != len(ACTION_NAMES):
            raise ValueError(
                f"weights must cover {len(ACTION_NAMES)} actions, got {len(self.weights)}"
            )
        if self.dwell_mean < 1.0:
            raise ValueError("dwell_mean must be at least 1 step")


class ScriptedExplorer:
    """A batched momentum random walk over the discrete action set."""

    def __init__(
        self, num_envs: int, config: ExplorerConfig | None = None, *, seed: int = 0
    ) -> None:
        config = config or ExplorerConfig()
        self.num_envs = num_envs
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        self.weights = torch.tensor(config.weights, dtype=torch.float32)
        self.action = torch.zeros(num_envs, dtype=torch.long)
        # `remaining <= 0` forces a redraw, so the first `act` draws for everyone.
        self.remaining = torch.zeros(num_envs, dtype=torch.long)

    def _draw(self, mask: Bool[Tensor, " envs"]) -> None:
        """Redraw the held action and its dwell for the masked environments."""

        count = int(mask.sum())
        if not count:
            return
        drawn = torch.multinomial(
            self.weights.expand(count, -1), num_samples=1, generator=self.generator
        ).squeeze(1)
        # A geometric dwell with the configured mean, floored at one step. The
        # memoryless dwell is what makes the walk a proper renewal process, so
        # the corpus has no periodicity a filter could exploit.
        probability = 1.0 / self.config.dwell_mean
        uniform = torch.rand(count, generator=self.generator).clamp_min(1e-9)
        dwell = (uniform.log() / torch.log1p(torch.tensor(-probability))).floor().long() + 1
        self.action[mask] = drawn
        self.remaining[mask] = dwell

    def reset(self, done: Bool[Tensor, " envs"] | None = None) -> None:
        """Force a redraw, for all environments or only the finished ones."""

        mask = (
            torch.ones(self.num_envs, dtype=torch.bool) if done is None else done.cpu().bool()
        )
        self.remaining[mask] = 0

    def act(self) -> Int[Tensor, " envs"]:
        """The action to take this step, redrawing wherever the dwell expired."""

        self._draw(self.remaining <= 0)
        self.remaining -= 1
        return self.action.clone()

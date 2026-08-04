"""Named conditions and the training budget for the Memory Maze study.

The condition set is deliberately three wide. Memory Maze is expensive -- MuJoCo
rendering runs at roughly 150-205 environment steps per second and does *not*
speed up with more parallel environments, because the adapter steps them in one
Python loop -- so the budget buys seeds rather than ablations. The ablation grid
already exists in the gridworld and MiniGrid studies; what this benchmark adds is
ground truth, and ground truth is what the probe consumes.

Capacity is comparable by construction at ``state_dim=128``:

============  =======  =========  ================
condition       total    encoder    post-perception
============  =======  =========  ================
tb-full       319,072    268,128            50,944
gru-control   368,743    268,128           100,615
lstm-control  401,767    268,128           133,639
============  =======  =========  ================

The encoder is identical in all three, so every parameter of the difference sits
after perception -- which is exactly where the claim is. Note the controls carry
*more* capacity than the Tensor Brain, not less: this follows the agreed rule of
strong baselines at comparable scale with known-good hyperparameters, rather
than deliberately weakened controls or an equal-budget sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.agency.agent import AgentConfig

# The reference agent. `state_dim=128` is what makes the three policies
# capacity-comparable; at 64 the controls fall to within 10% of the Tensor Brain
# and the comparison loses its "the controls are not handicapped" guarantee.
REFERENCE = AgentConfig(
    state_dim=128,
    hidden_dim=128,
    evolution="original",
    score_mode="direct",
    deliberation_windows=1,
)

# `None` selects a non-Tensor-Brain control; the recurrent cell is read off the name.
CONDITIONS: dict[str, AgentConfig | None] = {
    "tb-full": REFERENCE,
    "gru-control": None,
    "lstm-control": None,
}


@dataclass(frozen=True)
class MazeLevel:
    """One Memory Maze level and the budget spent on it."""

    # The maze size, naming a `memory_maze.tasks` factory rather than a gym id:
    # the adapter builds the `dm_env` directly. See `env.LEVEL_TASKS`.
    env_id: str
    tests: str
    num_envs: int
    updates: int
    segment_steps: int = 64
    max_steps: int = 1000

    @property
    def frames(self) -> int:
        return self.updates * self.segment_steps * self.num_envs


LEVELS: dict[str, MazeLevel] = {
    # The 9x9 level is the smallest Memory Maze ships. The published Dreamer-V3
    # numbers for it use orders of magnitude more experience than this study
    # spends, so task score here is *not* a competitive claim and is reported
    # only as evidence that the policies learned something to probe.
    "9x9": MazeLevel(
        env_id="9x9",
        tests="retention of target locations across an episode",
        num_envs=8,
        updates=1000,  # 512k frames
    ),
}

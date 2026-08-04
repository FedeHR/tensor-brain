"""Named conditions for the MiniGrid study.

A deliberately small subset of the gridworld grid: the reference, the effect that
was largest there, the controls that decided the negative results, and the one
score mode that failed completely. The question is which of those findings
survive a benchmark that we did not design.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from experiments.agency.agent import AgentConfig

REFERENCE = AgentConfig(
    state_dim=64,
    hidden_dim=64,
    evolution="original",
    score_mode="direct",
    deliberation_windows=1,
)

# `None` selects a non-Tensor-Brain control; the cell is chosen by name.
CONDITIONS: dict[str, AgentConfig | None] = {
    "tb-full": REFERENCE,
    "gru-control": None,
    "lstm-control": None,
    # The largest effect in the gridworld study.
    "deliberate-3-attend": replace(REFERENCE, deliberation_windows=3),
    # Does the instruction matter, or is the level solvable without it?
    "no-cue": replace(REFERENCE, cue_mode="none"),
    # Is the serial symbolic bottleneck still a net cost on a real benchmark?
    "no-percept-measure": replace(REFERENCE, measure_percepts=False),
    # Does the shared bidirectional matrix earn its place here?
    "decoupled-feedback": replace(REFERENCE, decouple_feedback=True),
    # The order-effect regime that beat the HB-POVM default in the gridworld.
    "pvm-action": replace(REFERENCE, action_retain_gate=0.0),
    # The QTB normalizer that never escaped under REINFORCE. Does a stronger
    # estimator rescue it, or is the failure a property of the score mode?
    "score-softplus-bias": replace(REFERENCE, score_mode="softplus-bias"),
    # --- input balance (QTB Equation 46's gates) --------------------------
    # Measured at initialization on this level: the encoder drive has component
    # RMS 1.04 while the two summed cue columns have RMS 0.17, and after
    # training the ratio grows to 28x. In the gridworld the same two quantities
    # were 0.16 and 0.18 -- balanced by accident of the one-hot view. These
    # three conditions ask whether the instruction is simply being drowned.
    "cue-gain-8": replace(REFERENCE, cue_gate=8.0),
    "cue-gate-learned": replace(REFERENCE, learn_cue_gate=True),
    "normalized-drive": replace(REFERENCE, normalize_drive=True),
}


@dataclass(frozen=True)
class Level:
    """One benchmark level and what it is included to test."""

    env_id: str
    tests: str
    compositional: bool
    num_envs: int
    updates: int
    segment_steps: int = 64

    @property
    def frames(self) -> int:
        return self.updates * self.segment_steps * self.num_envs


LEVELS: dict[str, Level] = {
    # Per-episode language instruction over colour x object type, so the
    # compositional split and the cue-blind floor are both meaningful.
    "gotolocal": Level(
        env_id="BabyAI-GoToLocal-v0",
        tests="instruction following and zero-shot cue recombination",
        compositional=True,
        num_envs=16,
        updates=1000,   # 1.02M frames; a random policy scores 0.31 here
    ),
    # A fixed mission, sparse reward, and a required sub-goal order: fetch the
    # key, unlock the door, cross, reach the goal. Tests sequencing and memory
    # rather than instruction following.
    "doorkey": Level(
        env_id="MiniGrid-DoorKey-6x6-v0",
        tests="sparse-reward sub-goal sequencing",
        compositional=False,
        num_envs=16,
        updates=2500,   # 2.56M frames; a random policy scores 0.02 here
    ),
}

# `no-cue` is only meaningful where the mission varies per episode.
LEVEL_CONDITIONS: dict[str, tuple[str, ...]] = {
    "gotolocal": tuple(CONDITIONS),
    "doorkey": tuple(name for name in CONDITIONS if name != "no-cue"),
}

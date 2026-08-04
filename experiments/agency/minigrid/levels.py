"""A strict BabyAI pickup level, registered so it can be used like any other.

The shuffled-mission control showed that on `BabyAI-GoToLocal-v0` *no* policy --
Tensor Brain or recurrent control -- uses the instruction: permuting the missions
across the batch changes success by at most 0.04. The reason is structural rather
than architectural. `PickupInstr` takes a ``strict`` flag that fails the episode
when the wrong object is picked up, but it defaults to ``False`` and none of the
registered BabyAI levels set it, so touching a distractor costs nothing. In a
single room with a 64-step budget, sweeping every object is a winning policy and
the instruction is decoration.

That makes the stock GoTo/Pickup levels unable to test whether an architecture
grounds an instruction. This level restores the requirement with the library's
own mechanism: identical generation to ``PickupLoc``, but ``strict=True``, so
picking up a distractor ends the episode with zero reward. It is the BabyAI
analogue of the gridworld's distractor penalty, and the minimum change that makes
the instruction load-bearing.
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry
from minigrid.envs.babyai.core.verifier import PickupInstr
from minigrid.envs.babyai.pickup import PickupLoc

STRICT_PICKUP_ID = "TB-PickupLocStrict-v0"


class StrictPickupLoc(PickupLoc):
    """``PickupLoc`` in which picking up the wrong object fails the episode.

    Only the verifier's ``strict`` flag differs from the parent level. Room
    layout, object sampling, mission grammar and step budget are inherited
    unchanged, so the world distribution is exactly BabyAI's.
    """

    def gen_mission(self) -> None:
        super().gen_mission()
        if isinstance(self.instrs, PickupInstr):
            self.instrs.strict = True


def register_levels() -> None:
    """Register the strict level with Gymnasium, idempotently."""

    if STRICT_PICKUP_ID in registry:
        return
    register(
        id=STRICT_PICKUP_ID,
        entry_point="experiments.agency.minigrid.levels:StrictPickupLoc",
    )


register_levels()

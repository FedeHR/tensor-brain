"""The Tensor Brain index layout for BabyAI / MiniGrid.

MiniGrid is an unusually good fit for an index layer: its observation is already
symbolic. Each of the 7x7 egocentric cells is a triple of integer codes for
object type, colour and state, and BabyAI missions are generated from a small
grammar over exactly those same symbols. So the *same* column of ``A`` can be

* the perceptual label "the attended cell contains a ball",
* the instruction "go to the ball", parsed out of the mission string, and
* nothing else has to be invented to connect them.

That is the claim from the gridworld study restated on a published benchmark
rather than on a vocabulary we designed ourselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from tb import IndexVocabulary

# The action names of `minigrid.core.actions.Actions`, in index order.
ACTION_NAMES = ("left", "right", "forward", "pickup", "drop", "toggle", "done")

# Object types a BabyAI mission can refer to. `door` is included because the
# GoTo/Open level families use it; `goal` and `lava` are perceivable but never
# named by a mission, and are therefore percept-only labels.
REFERABLE_OBJECTS = ("key", "ball", "box", "door")
PERCEPT_ONLY_OBJECTS = ("wall", "goal", "lava", "floor")
COLOR_NAMES = tuple(sorted(COLOR_TO_IDX, key=COLOR_TO_IDX.get))

NOTHING = "nothing_visible"
REWARD_POSITIVE = "reward_positive"
ANY_COLOR = "any_color"

# "go to the red ball", "go to a green key", "open the yellow door", ...
_MISSION = re.compile(
    r"(?:go to|pick up|open|put)\s+(?:the|a|an)\s+"
    r"(?:(?P<color>" + "|".join(COLOR_NAMES) + r")\s+)?"
    r"(?P<object>" + "|".join(REFERABLE_OBJECTS) + r")"
)


@dataclass(frozen=True)
class Cue:
    """A parsed mission: which colour and which object type it names."""

    color: str
    object_type: str


def build_vocabulary() -> IndexVocabulary:
    """Construct the MiniGrid index layout with deterministic group order.

    ``any_color`` is a real index, not a placeholder: BabyAI missions such as
    "go to a ball" name an object type without a colour, and the agent still has
    to inject *something* top-down. Giving the underspecified case its own
    learned column keeps the cue injection uniform.
    """

    return IndexVocabulary.from_groups(
        {
            "color": (*COLOR_NAMES, ANY_COLOR),
            "object": REFERABLE_OBJECTS,
            "percept_color": (*COLOR_NAMES, NOTHING),
            "percept_object": (*REFERABLE_OBJECTS, *PERCEPT_ONLY_OBJECTS, NOTHING),
            "action": ACTION_NAMES,
            "reward": (REWARD_POSITIVE,),
        }
    )


def parse_mission(mission: str) -> Cue:
    """Extract the cue factors from a BabyAI mission string.

    Compound missions are matched at their first referable phrase, so DoorKey's
    "use the key to open the door and then get to the goal" yields
    ``(any_color, door)``. That is a real sub-goal of the level, but it is
    *identical in every episode*, so on such levels the cue carries no
    per-episode information and the agent must rely on perception alone.
    Missions matching nothing at all fall back to the first referable object.
    """

    match = _MISSION.search(mission)
    if match is None:
        return Cue(ANY_COLOR, REFERABLE_OBJECTS[0])
    return Cue(match.group("color") or ANY_COLOR, match.group("object"))


def object_label(object_index: int) -> str | None:
    """Map a MiniGrid object code to a percept label, or ``None`` if unnamed."""

    for name, code in OBJECT_TO_IDX.items():
        if code == object_index and name in (*REFERABLE_OBJECTS, *PERCEPT_ONLY_OBJECTS):
            return name
    return None

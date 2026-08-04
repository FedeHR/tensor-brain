"""The global index layout for the agency gridworld.

The scientific point of this module is the *sharing*: the column ``a_red`` is
simultaneously

* the perceptual label "the attended object is red" (bottom-up score), and
* the instruction "find something red" (top-down feedback into ``q``).

The action indices live in the same matrix ``A`` and are read out by the same
operation. A reward index is included so that the internal value function is one
more column of ``A`` rather than a separate module.
"""

from __future__ import annotations

from experiments.agency.gridworld import ACTION_NAMES, GridConfig
from tb import IndexVocabulary

COLOR_NAMES = ("red", "green", "blue", "yellow", "purple", "cyan")
SHAPE_NAMES = ("key", "ball", "box", "cup", "star", "ring")
NOTHING = "nothing_visible"
REWARD_POSITIVE = "reward_positive"


def build_vocabulary(config: GridConfig) -> IndexVocabulary:
    """Construct the agency vocabulary with deterministic group order.

    Groups:

    ``color`` / ``shape``
        The bare attribute labels. These are the columns injected as cue
        feedback and the columns a downstream language task would reuse.
    ``percept_color`` / ``percept_shape``
        The attribute labels plus ``nothing_visible``. Perceptual measurement
        must be able to report an empty region of interest, but the cue is never
        "find nothing", so the candidate groups differ by exactly that one index.
    ``action``
        The five action indices. Measuring over this group *is* the policy.
    ``reward``
        The single ``reward_positive`` index whose score is the value readout.
    """

    if config.num_colors > len(COLOR_NAMES) or config.num_shapes > len(SHAPE_NAMES):
        raise ValueError("extend COLOR_NAMES/SHAPE_NAMES for this configuration")
    colors = COLOR_NAMES[: config.num_colors]
    shapes = SHAPE_NAMES[: config.num_shapes]
    return IndexVocabulary.from_groups(
        {
            "color": colors,
            "shape": shapes,
            "percept_color": (*colors, NOTHING),
            "percept_shape": (*shapes, NOTHING),
            "action": ACTION_NAMES,
            "reward": (REWARD_POSITIVE,),
        }
    )

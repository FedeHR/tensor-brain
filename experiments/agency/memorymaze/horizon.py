"""When was a target last visible, and what does the belief still know?

The memory-horizon curve is the figure this study exists for: probe error for a
target's position as a function of how many steps have passed since that target
was last in view. A single averaged probe score cannot distinguish a filter that
reads off the current frame from one that remembers, because most samples have
seen something recently. Conditioning on the gap separates them.

Visibility is *derived*, not recorded. It depends on a field of view and an
occlusion rule that are both approximations, and baking an approximation into a
10 GB corpus would mean re-rendering to revise it. The corpus stores the raw
geometry instead and this module turns it into a boolean.

The geometry conventions below were verified against a recorded corpus rather
than read off the documentation, which does not state them:

* ``maze_layout[row, col]`` is indexed by ``row = floor(y)``, ``col = floor(x)``
  for a position ``(x, y)``, and **1 means free, 0 means wall**. Checked by
  confirming that the agent stands on a free cell at every recorded step.
* ``targets_vec[..., 1]`` is the **forward** component in the agent's frame and
  ``targets_vec[..., 0]`` the lateral one. Checked by matching
  ``dot(agent_dir, targets_pos - agent_pos)`` against each component: the
  forward axis agrees to zero, the lateral one does not.

The field of view is the one genuinely soft parameter. It is a threshold on an
angle, and the curve is a comparison *between conditions* measured under the
same threshold, so the ordering it produces is insensitive to the exact value
even though the absolute step counts are not.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Half-angle of the visibility cone, in degrees. An approximation of the
# walker's egocentric camera rather than a value read from its spec.
DEFAULT_HALF_FOV = 45.0
# Beyond this many cells a target occupies too few pixels to be identifiable.
DEFAULT_MAX_DISTANCE = 6.0
# The value used for "this target has not been visible yet". Larger than any
# episode, so it sorts into the final bucket rather than silently becoming a
# small gap.
NEVER = 10**6


def line_of_sight(
    layout: np.ndarray, start: np.ndarray, end: np.ndarray, *, resolution: float = 0.1
) -> bool:
    """Whether the straight segment from ``start`` to ``end`` stays in free cells.

    Sampled rather than traced with a supercover algorithm: at 0.1-cell steps the
    sampling misses only grazing corner cuts, which are exactly the cases where
    "visible" is ambiguous anyway.
    """

    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return True
    samples = max(2, int(distance / resolution))
    for step in range(1, samples):
        point = start + delta * (step / samples)
        row, col = int(math.floor(point[1])), int(math.floor(point[0]))
        if not (0 <= row < layout.shape[0] and 0 <= col < layout.shape[1]):
            return False
        if layout[row, col] == 0:
            return False
    return True


def visibility(
    agent_pos: Float[Tensor, "steps 2"],
    targets_pos: Float[Tensor, "steps targets 2"],
    targets_vec: Float[Tensor, "steps targets 2"],
    layout: Int[Tensor, "rows cols"],
    *,
    half_fov: float = DEFAULT_HALF_FOV,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> Bool[Tensor, "steps targets"]:
    """Whether each target is in view at each step of one episode."""

    agent = agent_pos.numpy()
    targets = targets_pos.numpy()
    vectors = targets_vec.numpy()
    grid = layout.numpy()

    forward = vectors[..., 1]
    lateral = np.abs(vectors[..., 0])
    distance = np.linalg.norm(vectors, axis=-1)
    # In the cone, in front, and close enough to resolve.
    in_cone = (forward > 0.0) & (lateral <= forward * math.tan(math.radians(half_fov)))
    near = distance <= max_distance
    candidate = in_cone & near

    visible = np.zeros_like(candidate, dtype=bool)
    for step in range(candidate.shape[0]):
        for target in range(candidate.shape[1]):
            if candidate[step, target]:
                visible[step, target] = line_of_sight(
                    grid, agent[step], targets[step, target]
                )
    return torch.from_numpy(visible)


def steps_since_visible(visible: Bool[Tensor, "steps targets"]) -> Int[Tensor, "steps targets"]:
    """How many steps since each target was last in view, ``NEVER`` if never.

    Zero at a step where the target *is* visible. Computed causally: only the
    past is consulted, which is what makes the resulting curve a statement about
    memory rather than about the episode as a whole.
    """

    steps, targets = visible.shape
    gaps = torch.full((steps, targets), NEVER, dtype=torch.long)
    last = torch.full((targets,), -1, dtype=torch.long)
    for step in range(steps):
        seen = visible[step]
        last = torch.where(seen, torch.full_like(last, step), last)
        gaps[step] = torch.where(last >= 0, step - last, torch.full_like(last, NEVER))
    return gaps


# Bucket edges in steps. Fine near zero, where the belief is still being
# refreshed, and coarse out past a hundred steps, where samples are scarce.
DEFAULT_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 5), (5, 15), (15, 40), (40, 100), (100, 300), (300, NEVER), (NEVER, NEVER + 1),
)


def bucket_label(low: int, high: int) -> str:
    if low >= NEVER:
        return "never"
    if high >= NEVER:
        return f"{low}+"
    if high - low == 1:
        return str(low)
    return f"{low}-{high - 1}"


def horizon_curve(
    error: Float[Tensor, "samples targets"],
    gaps: Int[Tensor, "samples targets"],
    *,
    buckets: tuple[tuple[int, int], ...] = DEFAULT_BUCKETS,
) -> list[dict[str, float | str]]:
    """Mean per-target probe error within each staleness bucket."""

    curve: list[dict[str, float | str]] = []
    for low, high in buckets:
        selected = (gaps >= low) & (gaps < high)
        count = int(selected.sum())
        curve.append(
            {
                "bucket": bucket_label(low, high),
                "low": low,
                "samples": count,
                "rmse": float(error[selected].mean().sqrt()) if count else float("nan"),
            }
        )
    return curve

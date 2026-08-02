"""Explicit temporal boundaries for the three initial PVSG protocols."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

DEVELOPMENT_FRACTION = 0.15
DEVELOPMENT_SPLIT_SALT = "tensor-brain-pvsg-development-v1"


@dataclass(frozen=True)
class BlockedBoundary:
    """Half-open frame intervals for observation, embargo, and evaluation."""

    observation_end: int
    evaluation_start: int
    num_frames: int

    def role(self, frame_index: int) -> str:
        if not 0 <= frame_index < self.num_frames:
            raise ValueError("frame_index is outside the video")
        if frame_index < self.observation_end:
            return "observation"
        if frame_index < self.evaluation_start:
            return "embargo"
        return "evaluation"


def blocked_boundary(num_frames: int) -> BlockedBoundary:
    """Divide a video conservatively into 45% / 10% / 45%."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    observation_end = 45 * num_frames // 100
    evaluation_start = (55 * num_frames + 99) // 100
    if not 0 < observation_end < evaluation_start < num_frames:
        raise ValueError("video is too short for a non-empty 45/10/45 split")
    return BlockedBoundary(observation_end, evaluation_start, num_frames)


def development_video_ids(
    video_ids: Sequence[str],
    *,
    fraction: float = DEVELOPMENT_FRACTION,
    salt: str = DEVELOPMENT_SPLIT_SALT,
) -> set[str]:
    """Select a deterministic development subset of one source's training videos."""

    if len(video_ids) < 2 or len(video_ids) != len(set(video_ids)):
        raise ValueError("video_ids must contain at least two unique videos")
    if not 0.0 < fraction < 1.0 or not salt:
        raise ValueError("fraction must be between zero and one and salt must be nonempty")
    count = min(len(video_ids) - 1, max(1, int(len(video_ids) * fraction + 0.5)))
    ranked = sorted(
        video_ids,
        key=lambda video_id: hashlib.sha256(f"{salt}\0{video_id}".encode()).digest(),
    )
    return set(ranked[:count])


def fewshot_support_and_queries(
    visible_frames: list[int],
    *,
    support_count: int = 5,
    minimum_support_gap_frames: int = 5,
    embargo_frames: int = 25,
) -> tuple[list[int], list[int]]:
    """Return early, temporally distinct support exposures and later queries."""

    if support_count <= 0 or minimum_support_gap_frames <= 0 or embargo_frames < 0:
        raise ValueError(
            "support_count and minimum_support_gap_frames must be positive and "
            "embargo_frames non-negative"
        )
    if visible_frames != sorted(set(visible_frames)):
        raise ValueError("visible_frames must be unique and increasing")
    support = []
    for frame in visible_frames:
        if not support or frame >= support[-1] + minimum_support_gap_frames:
            support.append(frame)
            if len(support) == support_count:
                break
    if len(support) < support_count:
        return [], []
    first_query_frame = support[-1] + embargo_frames
    queries = [
        frame
        for frame in visible_frames
        if frame > support[-1] and frame >= first_query_frame
    ]
    if not queries:
        return [], []
    return support, queries

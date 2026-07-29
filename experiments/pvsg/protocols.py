"""Explicit temporal boundaries for the three initial PVSG protocols."""

from __future__ import annotations

from dataclasses import dataclass


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


def fewshot_support_and_queries(
    visible_frames: list[int], *, support_count: int = 5, embargo_frames: int = 25
) -> tuple[list[int], list[int]]:
    """Return first exposures and sufficiently later re-identification queries."""

    if support_count <= 0 or embargo_frames < 0:
        raise ValueError("support_count must be positive and embargo_frames non-negative")
    if visible_frames != sorted(set(visible_frames)):
        raise ValueError("visible_frames must be unique and increasing")
    if len(visible_frames) < support_count:
        return [], []
    support = visible_frames[:support_count]
    first_query_frame = support[-1] + embargo_frames
    queries = [frame for frame in visible_frames[support_count:] if frame >= first_query_frame]
    if not queries:
        return [], []
    return support, queries

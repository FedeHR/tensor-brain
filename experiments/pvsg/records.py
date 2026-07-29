"""PVSG annotation semantics for address-only experiment records."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXCLUSIONS_PATH = Path(__file__).with_name("exclusions.json")


@dataclass(frozen=True)
class SpanIssue:
    """A source span changed or rejected when intersected with the video."""

    video_id: str
    subject_id: int
    object_id: int
    predicate: str
    source_span: tuple[int, int]
    retained_span: tuple[int, int] | None


def load_exclusions(path: Path = EXCLUSIONS_PATH) -> dict[str, dict[str, Any]]:
    """Load and validate the reviewed source-video exclusion allowlist."""

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1 or not isinstance(document.get("videos"), list):
        raise ValueError(f"invalid PVSG exclusion document: {path}")
    exclusions = {}
    indices = set()
    for row in document["videos"]:
        required = {"source", "video_id", "extraction_array_index", "reason"}
        if set(row) != required or not row["reason"]:
            raise ValueError(f"invalid PVSG exclusion row: {row}")
        video_id = row["video_id"]
        if video_id in exclusions or row["extraction_array_index"] in indices:
            raise ValueError("duplicate PVSG exclusion video or array index")
        exclusions[video_id] = row
        indices.add(row["extraction_array_index"])
    return exclusions


def inclusive_clipped_frames(
    start: int, end: int, num_frames: int
) -> tuple[range, tuple[int, int] | None]:
    """Return the valid-frame intersection of an inclusive source span.

    The second return value is the retained inclusive span when clipping was
    needed, or ``None`` when the source span was already in bounds. An empty
    range denotes a span with no valid-frame intersection.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    retained_start = max(0, start)
    retained_end = min(end, num_frames - 1)
    if end < start or retained_start > retained_end:
        return range(0), None
    clipped = (retained_start, retained_end)
    return (
        range(retained_start, retained_end + 1),
        clipped if clipped != (start, end) else None,
    )


def active_predicates(
    annotation: dict[str, Any], excluded_video_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Return declared predicates plus retained annotation-only predicates."""

    declared = list(annotation["relations"])
    if len(declared) != len(set(declared)):
        raise ValueError("declared PVSG predicates are not unique")
    observed = {
        predicate
        for video in annotation["data"]
        if video["video_id"] not in excluded_video_ids
        for _, _, predicate, _ in video["relations"]
    }
    additional = sorted(observed - set(declared))
    return declared + additional, additional


def relation_targets(
    video: dict[str, Any],
    *,
    predicate_vocabulary: set[str],
) -> tuple[dict[tuple[int, int, int], tuple[str, ...]], list[SpanIssue]]:
    """Group inclusive relation labels by directed pair and frame."""

    targets: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    issues = []
    num_frames = video["meta"]["num_frames"]
    for subject_id, object_id, predicate, spans in video["relations"]:
        if predicate not in predicate_vocabulary:
            raise ValueError(f"predicate is absent from the active vocabulary: {predicate}")
        for start, end in spans:
            frames, clipped = inclusive_clipped_frames(start, end, num_frames)
            if not frames:
                issues.append(
                    SpanIssue(
                        video_id=video["video_id"],
                        subject_id=subject_id,
                        object_id=object_id,
                        predicate=predicate,
                        source_span=(start, end),
                        retained_span=None,
                    )
                )
                continue
            if clipped is not None:
                issues.append(
                    SpanIssue(
                        video_id=video["video_id"],
                        subject_id=subject_id,
                        object_id=object_id,
                        predicate=predicate,
                        source_span=(start, end),
                        retained_span=clipped,
                    )
                )
            for frame_index in frames:
                targets[(frame_index, subject_id, object_id)].add(predicate)
    return {key: tuple(sorted(values)) for key, values in targets.items()}, issues


def identity_name(source: str, video_id: str, object_id: int) -> str:
    """Return the stable symbolic name of an actual tracked PVSG identity."""

    return f"identity:{source}/{video_id}/{object_id}"

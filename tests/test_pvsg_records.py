from experiments.pvsg.records import (
    active_predicates,
    inclusive_clipped_frames,
    load_exclusions,
    relation_targets,
)


def test_reviewed_exclusions_are_exact_and_versioned() -> None:
    exclusions = load_exclusions()

    assert len(exclusions) == 6
    assert exclusions["ec2e69c1-fd07-48ec-adff-0b2cf3ab25b6"][
        "extraction_array_index"
    ] == 52
    assert exclusions["1019_3004044251"]["extraction_array_index"] == 257


def test_inclusive_span_is_clipped_without_discarding_valid_frames() -> None:
    frames, clipped = inclusive_clipped_frames(1, 3, num_frames=3)
    empty, empty_clip = inclusive_clipped_frames(5, 8, num_frames=3)

    assert list(frames) == [1, 2]
    assert clipped == (1, 2)
    assert not empty
    assert empty_clip is None


def test_relation_targets_group_simultaneous_predicates_and_report_clipping() -> None:
    video = {
        "video_id": "video-1",
        "meta": {"num_frames": 3},
        "relations": [
            [1, 2, "holding", [[1, 3]]],
            [1, 2, "looking at", [[2, 2]]],
            [1, 2, "holding", [[8, 9]]],
        ],
    }

    targets, issues = relation_targets(
        video, predicate_vocabulary={"holding", "looking at"}
    )

    assert targets == {
        (1, 1, 2): ("holding",),
        (2, 1, 2): ("holding", "looking at"),
    }
    assert issues[0].source_span == (1, 3)
    assert issues[0].retained_span == (1, 2)
    assert issues[1].source_span == (8, 9)
    assert issues[1].retained_span is None


def test_active_ontology_retains_annotation_labels_but_not_excluded_only_labels() -> None:
    annotation = {
        "relations": ["on", "holding"],
        "data": [
            {"video_id": "kept", "relations": [[1, 2, "pouring", [[0, 1]]]]},
            {"video_id": "excluded", "relations": [[1, 2, "moving", [[0, 1]]]]},
        ],
    }

    predicates, additional = active_predicates(annotation, {"excluded"})

    assert predicates == ["on", "holding", "pouring"]
    assert additional == ["pouring"]

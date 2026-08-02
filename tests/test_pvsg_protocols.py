import pytest

from experiments.pvsg.protocols import (
    blocked_boundary,
    development_video_ids,
    fewshot_support_and_queries,
)


def test_blocked_boundary_is_conservative_45_10_45() -> None:
    exact = blocked_boundary(100)
    rounded = blocked_boundary(75)

    assert (exact.observation_end, exact.evaluation_start) == (45, 55)
    assert exact.role(44) == "observation"
    assert exact.role(45) == "embargo"
    assert exact.role(54) == "embargo"
    assert exact.role(55) == "evaluation"
    assert (rounded.observation_end, rounded.evaluation_start) == (33, 42)


def test_development_split_is_deterministic_and_nonempty() -> None:
    videos = [f"video-{index}" for index in range(20)]

    first = development_video_ids(videos)
    second = development_video_ids(list(reversed(videos)))

    assert first == second
    assert len(first) == 3


def test_fewshot_spaces_five_exposures_then_applies_query_embargo() -> None:
    support, queries = fewshot_support_and_queries(
        [0, 1, 5, 6, 10, 11, 15, 16, 20, 30, 44, 45, 60],
        support_count=5,
        minimum_support_gap_frames=5,
        embargo_frames=25,
    )

    assert support == [0, 5, 10, 15, 20]
    assert queries == [45, 60]


def test_fewshot_can_fix_queries_after_ten_nested_supports() -> None:
    support, queries = fewshot_support_and_queries(
        [*range(0, 50, 5), 69, 70, 85],
        support_count=10,
        minimum_support_gap_frames=5,
        embargo_frames=25,
    )

    assert support == list(range(0, 50, 5))
    assert queries == [70, 85]


def test_fewshot_omits_identity_without_a_later_query() -> None:
    assert fewshot_support_and_queries([0, 5, 10, 15, 20, 30]) == ([], [])


def test_protocol_helpers_reject_unsorted_or_out_of_range_inputs() -> None:
    with pytest.raises(ValueError, match="unique and increasing"):
        fewshot_support_and_queries([0, 2, 1, 3, 4, 30])
    with pytest.raises(ValueError, match="outside"):
        blocked_boundary(100).role(100)
    with pytest.raises(ValueError, match="at least two unique"):
        development_video_ids(["only-one"])

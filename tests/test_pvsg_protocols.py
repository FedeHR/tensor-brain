import pytest

from experiments.pvsg.protocols import blocked_boundary, fewshot_support_and_queries


def test_blocked_boundary_is_conservative_45_10_45() -> None:
    exact = blocked_boundary(100)
    rounded = blocked_boundary(75)

    assert (exact.observation_end, exact.evaluation_start) == (45, 55)
    assert exact.role(44) == "observation"
    assert exact.role(45) == "embargo"
    assert exact.role(54) == "embargo"
    assert exact.role(55) == "evaluation"
    assert (rounded.observation_end, rounded.evaluation_start) == (33, 42)


def test_fewshot_uses_five_exposures_then_a_25_frame_embargo() -> None:
    support, queries = fewshot_support_and_queries(
        [0, 1, 2, 3, 4, 20, 28, 29, 40], support_count=5, embargo_frames=25
    )

    assert support == [0, 1, 2, 3, 4]
    assert queries == [29, 40]


def test_fewshot_omits_identity_without_a_later_query() -> None:
    assert fewshot_support_and_queries([0, 1, 2, 3, 4, 10]) == ([], [])


def test_protocol_helpers_reject_unsorted_or_out_of_range_inputs() -> None:
    with pytest.raises(ValueError, match="unique and increasing"):
        fewshot_support_and_queries([0, 2, 1, 3, 4, 30])
    with pytest.raises(ValueError, match="outside"):
        blocked_boundary(100).role(100)

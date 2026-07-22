import pytest
import torch

from tb import IndexVocabulary, get_candidate_positions


def test_vocabulary_maps_names_groups_and_global_indices() -> None:
    vocabulary = IndexVocabulary.from_groups(
        {
            "entities": ["entity:sparky", "entity:ball"],
            "concepts": ["entity:sparky", "class:dog"],
            "predicates": ["predicate:chases"],
        }
    )
    assert vocabulary.labels == (
        "entity:sparky",
        "entity:ball",
        "class:dog",
        "predicate:chases",
    )
    assert vocabulary.index("class:dog") == 2
    assert vocabulary.label(3) == "predicate:chases"
    torch.testing.assert_close(vocabulary.indices("concepts"), torch.tensor([0, 2]))
    assert vocabulary.group_labels("entities") == ("entity:sparky", "entity:ball")


def test_vocabulary_serialization_preserves_column_order() -> None:
    vocabulary = IndexVocabulary.from_groups(
        {"entities": ["entity:sparky"], "classes": ["class:dog"]}
    )
    restored = IndexVocabulary.from_dict(vocabulary.to_dict())
    assert restored.labels == vocabulary.labels
    assert restored.groups == vocabulary.groups
    assert restored.to_dict() == vocabulary.to_dict()


def test_get_positions_maps_global_indices_to_noncontiguous_group_order() -> None:
    vocabulary = IndexVocabulary(
        labels=[f"symbol:{index}" for index in range(8)],
        groups={"candidates": [6, 1, 4]},
    )

    positions = vocabulary.get_positions("candidates", torch.tensor([4, 6, 1]))

    # Compact scores follow candidate order [6, 1, 4], so their CE targets are
    # positions [2, 0, 1], not the stable global indices [4, 6, 1].
    torch.testing.assert_close(positions, torch.tensor([2, 0, 1]))


def test_get_candidate_positions_supports_an_ad_hoc_candidate_subset() -> None:
    positions = get_candidate_positions([17, 42, 91], torch.tensor([91, 17, 42]))

    torch.testing.assert_close(positions, torch.tensor([2, 0, 1]))


def test_get_candidate_positions_rejects_targets_outside_the_candidate_set() -> None:
    with pytest.raises(ValueError, match="outside the candidate set"):
        get_candidate_positions([17, 42, 91], torch.tensor([17, 18]))

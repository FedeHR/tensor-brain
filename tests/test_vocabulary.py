import torch

from tb import IndexVocabulary


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

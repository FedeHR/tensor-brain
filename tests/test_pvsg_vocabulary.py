import pytest

from experiments.pvsg.vocabulary import build_section6_vocabulary


def _ontology():
    return {
        "schema_version": 1,
        "predicates": ["holding", "riding", "looking at"],
        "train_supported_predicates": ["holding", "looking at"],
        "train_unseen_predicates": ["riding"],
        "object_categories": {"thing": ["dog", "ball", "gift"], "stuff": []},
        "identities": [
            {"name": "identity:train/dog", "category": "dog"},
            {"name": "identity:evaluation/dog", "category": "dog"},
            {"name": "identity:train/ball", "category": "ball"},
            {"name": "identity:train/gift", "category": "gift"},
        ],
    }


def _hierarchy():
    return {
        "paths": {
            "dog": ["category:dog", "category:animal", "category:living_being"],
            "ball": [
                "category:ball",
                "category:play_equipment",
                "category:sports_equipment",
            ],
            "gift": None,
        },
        "identity_paths": {},
        "domains": {
            "category:living_being": "domain:natural_and_living_world",
            "category:sports_equipment": "domain:personal_recreation_and_mobility",
        },
    }


def test_vocabulary_uses_only_supported_predicates_and_protocol_identities() -> None:
    vocabulary = build_section6_vocabulary(
        _ontology(),
        identity_names=["identity:train/ball", "identity:train/dog"],
        category_levels=("source",),
    )

    assert vocabulary.groups == ("predicate", "object_category/source", "identity")
    assert vocabulary.labels == (
        "predicate:holding",
        "predicate:looking at",
        "category:dog",
        "category:ball",
        "identity:train/dog",
        "identity:train/ball",
    )
    assert "predicate:riding" not in vocabulary.labels
    assert "identity:evaluation/dog" not in vocabulary.labels


def test_reviewed_levels_skip_excluded_semantics_but_keep_the_identity() -> None:
    vocabulary = build_section6_vocabulary(
        _ontology(),
        identity_names=["identity:train/gift", "identity:train/dog"],
        category_levels=("fine", "basic", "coarse", "domain"),
        hierarchy=_hierarchy(),
    )

    assert vocabulary.group_labels("object_category/fine") == ("category:dog",)
    assert vocabulary.group_labels("object_category/basic") == ("category:animal",)
    assert vocabulary.group_labels("object_category/coarse") == ("category:living_being",)
    assert vocabulary.group_labels("object_category/domain") == ("domain:natural_and_living_world",)
    assert vocabulary.group_labels("identity") == (
        "identity:train/dog",
        "identity:train/gift",
    )
    assert "category:gift" not in vocabulary.labels


def test_vocabulary_rejects_unknown_identities_and_invalid_predicate_partition() -> None:
    with pytest.raises(ValueError, match="unknown protocol identities"):
        build_section6_vocabulary(_ontology(), identity_names=["identity:missing"])

    ontology = _ontology()
    ontology["train_unseen_predicates"] = []
    with pytest.raises(ValueError, match="must partition predicates"):
        build_section6_vocabulary(ontology, identity_names=["identity:train/dog"])


def test_vocabulary_requires_hierarchy_for_reviewed_levels() -> None:
    with pytest.raises(ValueError, match="hierarchy is required"):
        build_section6_vocabulary(
            _ontology(),
            identity_names=["identity:train/dog"],
            category_levels=("fine",),
        )

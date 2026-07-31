import json

import pytest

from experiments.pvsg.semantic_inventory import (
    SEMANTIC_INVENTORY_PATH,
    load_semantic_inventory,
    semantic_property_groups,
    semantic_relation_labels,
)
from tb import IndexVocabulary


def _document():
    return json.loads(SEMANTIC_INVENTORY_PATH.read_text(encoding="utf-8"))


def test_semantic_inventory_maps_directly_to_disjoint_vocabulary_groups() -> None:
    inventory = load_semantic_inventory()
    groups = semantic_property_groups(inventory)
    groups["semantic_relation"] = semantic_relation_labels(inventory)
    vocabulary = IndexVocabulary.from_groups(groups)

    assert {name: len(labels) for name, labels in groups.items()} == {
        "semantic_property/color": 11,
        "semantic_property/material": 12,
        "semantic_property/shape": 10,
        "semantic_property/closure_state": 2,
        "semantic_property/content_state": 2,
        "semantic_property/surface_condition": 5,
        "semantic_property/power_state": 2,
        "semantic_property/affordance": 4,
        "semantic_property/risk": 1,
        "semantic_relation": 9,
    }
    assert len(vocabulary) == 58
    assert "property:risk/can_cause_physical_harm" in vocabulary.labels
    assert "relation:owned_by" in vocabulary.labels
    assert not any(
        label.endswith(("/other", "/unknown"))
        for labels in groups.values()
        for label in labels
    )


def test_semantic_inventory_rejects_duplicate_values(tmp_path) -> None:
    document = _document()
    document["families"]["color"]["values"].append(
        document["families"]["color"]["values"][0]
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid property values"):
        load_semantic_inventory(duplicate_path)


def test_semantic_inventory_rejects_inconsistent_inverse(tmp_path) -> None:
    document = _document()
    document["relations"]["relation:owns"]["inverse"] = None
    inconsistent_path = tmp_path / "inconsistent.json"
    inconsistent_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent inverse"):
        load_semantic_inventory(inconsistent_path)

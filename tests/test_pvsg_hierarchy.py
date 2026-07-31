import json

import pytest

from experiments.pvsg.hierarchy import (
    OBJECT_HIERARCHY_PATH,
    load_object_hierarchy,
    object_hierarchy_groups,
    object_hierarchy_path,
)
from tb import IndexVocabulary


def _document():
    return json.loads(OBJECT_HIERARCHY_PATH.read_text(encoding="utf-8"))


def _object_categories(document):
    categories = list(document["paths"])
    return {"thing": categories[:115], "stuff": categories[115:]}


def _identities(document):
    return [
        {"name": name, "category": "bat"} for name in document["identity_paths"]
    ]


def test_reviewed_hierarchy_maps_directly_to_disjoint_vocabulary_groups() -> None:
    document = _document()
    hierarchy = load_object_hierarchy(
        _object_categories(document),
        _identities(document),
    )
    groups = object_hierarchy_groups(hierarchy)
    vocabulary = IndexVocabulary.from_groups(groups)

    assert {name: len(labels) for name, labels in groups.items()} == {
        "object_category/fine": 121,
        "object_category/basic": 78,
        "object_category/coarse": 22,
        "object_category/domain": 5,
    }
    assert len(vocabulary) == 226
    level_groups = tuple(groups.values())
    assert all(
        not set(left_group) & set(right_group)
        for position, left_group in enumerate(level_groups)
        for right_group in level_groups[position + 1 :]
    )
    assert object_hierarchy_path(
        hierarchy,
        identity_name="identity:vidor/0054_5402337043/7",
        source_category="bat",
    ) == (
        "category:baseball_bat",
        "category:striking_sports_equipment",
        "category:sports_equipment",
        "domain:personal_recreation_and_mobility",
    )
    assert object_hierarchy_path(
        hierarchy,
        identity_name="identity:any/video/1",
        source_category="beverage",
    ) == (
        "category:beverage",
        "category:liquid_food",
        "category:food",
        "domain:food_and_kitchen",
    )
    assert object_hierarchy_path(
        hierarchy,
        identity_name="identity:any/video/2",
        source_category="pillow",
    ) == (
        "category:pillow",
        "category:soft_bedding",
        "category:household_textile",
        "domain:built_and_domestic_environment",
    )
    assert object_hierarchy_path(
        hierarchy,
        identity_name="identity:any/video/3",
        source_category="plant",
    ) == (
        "category:potted_plant",
        "category:plant_life",
        "category:living_being",
        "domain:natural_and_living_world",
    )


def test_hierarchy_requires_the_exact_source_category_order() -> None:
    document = _document()
    object_categories = _object_categories(document)
    object_categories["thing"][0], object_categories["thing"][1] = (
        object_categories["thing"][1],
        object_categories["thing"][0],
    )

    with pytest.raises(ValueError, match="source category order"):
        load_object_hierarchy(object_categories, _identities(document))


def test_hierarchy_requires_every_identity_of_a_refined_category(tmp_path) -> None:
    document = _document()
    object_categories = _object_categories(document)
    identities = _identities(document)
    document["identity_paths"].pop(identities[0]["name"])
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="do not cover source category"):
        load_object_hierarchy(object_categories, identities, incomplete_path)


def test_hierarchy_rejects_catch_all_concepts_and_inconsistent_parents(tmp_path) -> None:
    document = _document()
    object_categories = _object_categories(document)
    identities = _identities(document)

    document["paths"]["adult"][1] = "category:other"
    catch_all_path = tmp_path / "catch_all.json"
    catch_all_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="catch-all concept"):
        load_object_hierarchy(object_categories, identities, catch_all_path)

    document = _document()
    document["paths"]["child"][2] = "category:mammal"
    inconsistent_path = tmp_path / "inconsistent.json"
    inconsistent_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent coarse parents"):
        load_object_hierarchy(object_categories, identities, inconsistent_path)

    document = _document()
    document["domains"].pop("category:food")
    incomplete_domains_path = tmp_path / "incomplete_domains.json"
    incomplete_domains_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="every coarse label"):
        load_object_hierarchy(object_categories, identities, incomplete_domains_path)

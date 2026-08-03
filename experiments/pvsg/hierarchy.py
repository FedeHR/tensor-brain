"""Load and resolve the reviewed PVSG object-category hierarchy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.pvsg.io import read_json
from experiments.pvsg.prepare import PVSG_JSON_SHA256

OBJECT_HIERARCHY_PATH = Path(__file__).with_name("object_hierarchy.json")
CATEGORY_LEVELS = ("fine", "basic", "coarse")
HIERARCHY_LEVELS = (*CATEGORY_LEVELS, "domain")

CategoryPath = tuple[str, str, str]
HierarchyPath = tuple[str, str, str, str]

_CATCH_ALL_CONCEPTS = {
    "category:entity",
    "category:miscellaneous",
    "category:object",
    "category:other",
    "category:others",
    "category:stuff",
    "category:thing",
}
_DOCUMENT_FIELDS = {
    "schema_version",
    "pvsg_json_sha256",
    "levels",
    "paths",
    "domains",
    "identity_paths",
    "refinement_reasons",
    "exclusion_reasons",
}


def _source_categories(object_categories: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    if set(object_categories) != {"thing", "stuff"}:
        raise ValueError("object categories must contain exactly 'thing' and 'stuff'")
    categories = tuple(object_categories["thing"]) + tuple(object_categories["stuff"])
    if len(categories) != 126 or len(set(categories)) != len(categories):
        raise ValueError("object categories must contain the 126 unique PVSG labels")
    return categories


def _validated_path(value: Any, owner: str) -> CategoryPath:
    if (
        not isinstance(value, list)
        or len(value) != len(CATEGORY_LEVELS)
        or any(not isinstance(label, str) or not label for label in value)
    ):
        raise ValueError(f"invalid complete hierarchy path for {owner!r}")
    fine, basic, coarse = value
    if len({fine, basic, coarse}) != len(CATEGORY_LEVELS):
        raise ValueError(f"hierarchy levels must be distinct for {owner!r}")
    if {fine, basic, coarse} & _CATCH_ALL_CONCEPTS:
        raise ValueError(f"catch-all concept in hierarchy path for {owner!r}")
    return fine, basic, coarse


def load_object_hierarchy(
    object_categories: Mapping[str, Sequence[str]],
    identities: Sequence[Mapping[str, Any]],
    path: Path = OBJECT_HIERARCHY_PATH,
) -> dict[str, Any]:
    """Validate hierarchy paths against a materialized PVSG ontology."""

    document: Any = read_json(path)
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        raise ValueError(f"invalid PVSG hierarchy document: {path}")
    if document["schema_version"] != 3:
        raise ValueError(f"unsupported PVSG hierarchy schema: {document['schema_version']!r}")
    if document["pvsg_json_sha256"] != PVSG_JSON_SHA256:
        raise ValueError("PVSG hierarchy refers to a different annotation snapshot")
    if document["levels"] != list(HIERARCHY_LEVELS):
        raise ValueError(f"PVSG hierarchy levels must be {HIERARCHY_LEVELS}")

    source_paths = document["paths"]
    domains = document["domains"]
    identity_paths = document["identity_paths"]
    refinement_reasons = document["refinement_reasons"]
    exclusion_reasons = document["exclusion_reasons"]
    values = (source_paths, domains, identity_paths, refinement_reasons, exclusion_reasons)
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("PVSG hierarchy mappings must be JSON objects")
    if tuple(source_paths) != _source_categories(object_categories):
        raise ValueError("PVSG hierarchy does not exactly match the source category order")

    unresolved = {category for category, value in source_paths.items() if value is None}
    refined = set(refinement_reasons)
    excluded = set(exclusion_reasons)
    if refined & excluded or unresolved != refined | excluded:
        raise ValueError("every null source path must be either refined or excluded")
    for reasons in (refinement_reasons, exclusion_reasons):
        if any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values()):
            raise ValueError("every hierarchy refinement and exclusion needs a reason")

    identities_by_name = {row.get("name"): row for row in identities}
    if None in identities_by_name or len(identities_by_name) != len(identities):
        raise ValueError("PVSG identities must have unique names")
    for name in identity_paths:
        if name not in identities_by_name:
            raise ValueError(f"hierarchy override refers to an unknown identity: {name}")
        if identities_by_name[name].get("category") not in refined:
            raise ValueError(f"hierarchy override does not belong to a refined category: {name}")
    for category in refined:
        expected = {
            name
            for name, row in identities_by_name.items()
            if row.get("category") == category
        }
        actual = {
            name
            for name in identity_paths
            if identities_by_name[name].get("category") == category
        }
        if actual != expected:
            raise ValueError(f"identity refinements do not cover source category {category!r}")

    paths = [
        _validated_path(value, category)
        for category, value in source_paths.items()
        if value is not None
    ]
    paths.extend(_validated_path(value, name) for name, value in identity_paths.items())
    fine_to_parents: dict[str, tuple[str, str]] = {}
    basic_to_coarse: dict[str, str] = {}
    for fine, basic, coarse in paths:
        if fine in fine_to_parents and fine_to_parents[fine] != (basic, coarse):
            raise ValueError(f"fine label {fine!r} has inconsistent parents")
        if basic in basic_to_coarse and basic_to_coarse[basic] != coarse:
            raise ValueError(f"basic label {basic!r} has inconsistent coarse parents")
        fine_to_parents[fine] = (basic, coarse)
        basic_to_coarse[basic] = coarse

    fine_labels = set(fine_to_parents)
    basic_labels = set(basic_to_coarse)
    coarse_labels = set(basic_to_coarse.values())
    if set(domains) != coarse_labels:
        raise ValueError("every coarse label must have exactly one domain")
    if any(
        not isinstance(domain, str)
        or not domain
        or domain in _CATCH_ALL_CONCEPTS
        for domain in domains.values()
    ):
        raise ValueError("invalid domain label")
    domain_labels = set(domains.values())
    level_labels = (fine_labels, basic_labels, coarse_labels, domain_labels)
    if any(
        left & right
        for position, left in enumerate(level_labels)
        for right in level_labels[position + 1 :]
    ):
        raise ValueError("fine, basic, coarse, and domain concept sets must be disjoint")
    if not len(domain_labels) < len(coarse_labels) < len(basic_labels) < len(fine_labels):
        raise ValueError("hierarchy label counts must decrease from fine to domain")
    return document


def object_hierarchy_path(
    hierarchy: Mapping[str, Any],
    *,
    identity_name: str,
    source_category: str,
) -> HierarchyPath:
    """Resolve one tracked identity to its complete semantic path."""

    value = hierarchy["identity_paths"].get(identity_name, hierarchy["paths"][source_category])
    if value is None:
        raise KeyError(f"source category has no semantic path: {source_category}")
    fine, basic, coarse = value
    return fine, basic, coarse, hierarchy["domains"][coarse]


def object_hierarchy_groups(
    hierarchy: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return ordered level groups suitable for ``IndexVocabulary.from_groups``."""

    category_paths = [value for value in hierarchy["paths"].values() if value is not None]
    category_paths.extend(hierarchy["identity_paths"].values())
    paths = [(*path, hierarchy["domains"][path[2]]) for path in category_paths]
    return {
        f"object_category/{level}": tuple(dict.fromkeys(path[position] for path in paths))
        for position, level in enumerate(HIERARCHY_LEVELS)
    }

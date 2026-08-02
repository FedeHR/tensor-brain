"""Build fixed index vocabularies for the initial PVSG experiments."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from experiments.pvsg.hierarchy import (
    HIERARCHY_LEVELS,
    object_hierarchy_groups,
    object_hierarchy_path,
)
from tb import IndexVocabulary

CATEGORY_LEVELS = ("source", *HIERARCHY_LEVELS)


def predicate_label(predicate: str) -> str:
    """Namespace one manifest predicate without changing its source spelling."""

    if not predicate:
        raise ValueError("predicate must not be empty")
    return f"predicate:{predicate}"


def source_category_label(category: str) -> str:
    """Map one official PVSG category into the shared category namespace."""

    if not category:
        raise ValueError("category must not be empty")
    return f"category:{category}"


def _unique_strings(value: Any, owner: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{owner} must be a sequence of nonempty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{owner} must not contain duplicates")
    return result


def build_section6_vocabulary(
    ontology: Mapping[str, Any],
    *,
    identity_names: Collection[str],
    category_levels: Sequence[str] = (),
    hierarchy: Mapping[str, Any] | None = None,
) -> IndexVocabulary:
    """Build one deterministic vocabulary from a materialized Section 6 ontology.

    ``identity_names`` is the set supervised by the protocol's training or enrollment
    records. Category groups contain only labels supported by those identities. Static
    predicate and category labels precede identities in the global column order.
    """

    if ontology.get("schema_version") != 1:
        raise ValueError("unsupported PVSG ontology schema")

    predicates = _unique_strings(ontology.get("predicates"), "predicates")
    train_supported = _unique_strings(
        ontology.get("train_supported_predicates"), "train-supported predicates"
    )
    train_unseen = _unique_strings(
        ontology.get("train_unseen_predicates"), "train-unseen predicates"
    )
    supported_set = set(train_supported)
    unseen_set = set(train_unseen)
    if supported_set & unseen_set or supported_set | unseen_set != set(predicates):
        raise ValueError("train-supported and train-unseen predicates must partition predicates")
    if train_supported != tuple(name for name in predicates if name in supported_set):
        raise ValueError("train-supported predicates must preserve ontology order")
    if train_unseen != tuple(name for name in predicates if name in unseen_set):
        raise ValueError("train-unseen predicates must preserve ontology order")

    object_categories = ontology.get("object_categories")
    if not isinstance(object_categories, Mapping) or set(object_categories) != {
        "thing",
        "stuff",
    }:
        raise ValueError("object_categories must contain thing and stuff")
    source_categories = _unique_strings(
        (
            *_unique_strings(object_categories["thing"], "thing categories"),
            *_unique_strings(object_categories["stuff"], "stuff categories"),
        ),
        "object categories",
    )

    identities = ontology.get("identities")
    if not isinstance(identities, Sequence) or isinstance(identities, (str, bytes)):
        raise ValueError("identities must be a sequence")
    if isinstance(identity_names, (str, bytes)):
        raise ValueError("identity_names must be a collection of names")
    requested_identities = set(identity_names)
    if not requested_identities:
        raise ValueError("identity_names must not be empty")

    records: list[Mapping[str, Any]] = []
    known_names: set[str] = set()
    for record in identities:
        if not isinstance(record, Mapping):
            raise ValueError("every identity must be an object")
        name = record.get("name")
        category = record.get("category")
        if not isinstance(name, str) or not name or name in known_names:
            raise ValueError("identity names must be unique nonempty strings")
        if category not in source_categories:
            raise ValueError(f"identity {name!r} has an unknown source category")
        known_names.add(name)
        if name in requested_identities:
            records.append(record)
    unknown_names = requested_identities - known_names
    if unknown_names:
        raise ValueError(f"unknown protocol identities: {sorted(unknown_names)!r}")

    selected_levels = _unique_strings(category_levels, "category_levels") if category_levels else ()
    if any(level not in CATEGORY_LEVELS for level in selected_levels):
        raise ValueError(f"category levels must come from {CATEGORY_LEVELS!r}")
    if any(level != "source" for level in selected_levels) and hierarchy is None:
        raise ValueError("hierarchy is required for reviewed category levels")

    groups: dict[str, tuple[str, ...]] = {
        "predicate": tuple(predicate_label(name) for name in train_supported)
    }
    active_categories = {str(record["category"]) for record in records}
    if "source" in selected_levels:
        groups["object_category/source"] = tuple(
            source_category_label(category)
            for category in source_categories
            if category in active_categories
        )

    hierarchy_support = {level: set() for level in HIERARCHY_LEVELS}
    if hierarchy is not None and any(level in selected_levels for level in HIERARCHY_LEVELS):
        paths = hierarchy.get("paths")
        identity_paths = hierarchy.get("identity_paths")
        if not isinstance(paths, Mapping) or not isinstance(identity_paths, Mapping):
            raise ValueError("invalid reviewed hierarchy")
        for record in records:
            name = str(record["name"])
            category = str(record["category"])
            if category not in paths:
                raise ValueError(f"hierarchy is missing source category {category!r}")
            if paths[category] is None and name not in identity_paths:
                continue
            path = object_hierarchy_path(hierarchy, identity_name=name, source_category=category)
            for level, label in zip(HIERARCHY_LEVELS, path, strict=True):
                hierarchy_support[level].add(label)

        reviewed_groups = object_hierarchy_groups(hierarchy)
        for level in HIERARCHY_LEVELS:
            if level not in selected_levels:
                continue
            group = f"object_category/{level}"
            labels = tuple(
                label for label in reviewed_groups[group] if label in hierarchy_support[level]
            )
            if not labels:
                raise ValueError(f"no active identity supports category level {level!r}")
            groups[group] = labels

    groups["identity"] = tuple(str(record["name"]) for record in records)
    return IndexVocabulary.from_groups(groups)

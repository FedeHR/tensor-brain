"""Load the reviewed PVSG semantic-property and relation vocabulary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.pvsg.io import read_json

SEMANTIC_INVENTORY_PATH = Path(__file__).with_name("semantic_inventory.json")

_DOCUMENT_FIELDS = {"schema_version", "unknown_policy", "families", "relations"}
_FAMILY_FIELDS = {
    "kind",
    "stability",
    "cardinality",
    "source_conventions",
    "values",
}
_RELATION_FIELDS = {
    "kind",
    "stability",
    "inverse",
    "symmetric",
    "subject_scope",
    "object_scope",
    "source_conventions",
}
_FAMILY_KINDS = {"perceptual", "semantic"}
_STABILITIES = {"observation_transient", "identity_stable", "category_typical"}
_CARDINALITIES = {"single_label", "multi_label"}
_RELATION_KINDS = {"conceptual", "identity"}


def _validate_sources(sources: Any, owner: str) -> None:
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(source, str) or not source for source in sources)
    ):
        raise ValueError(f"invalid source conventions for {owner!r}")


def load_semantic_inventory(
    path: Path = SEMANTIC_INVENTORY_PATH,
) -> dict[str, Any]:
    """Load and structurally validate the reviewed semantic index inventory."""

    document: Any = read_json(path)
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        raise ValueError(f"invalid PVSG semantic inventory: {path}")
    if document["schema_version"] != 1:
        raise ValueError(
            f"unsupported PVSG semantic inventory schema: {document['schema_version']!r}"
        )
    if not isinstance(document["unknown_policy"], str) or not document[
        "unknown_policy"
    ].strip():
        raise ValueError("semantic inventory needs an unknown-label policy")

    families = document["families"]
    relations = document["relations"]
    if not isinstance(families, dict) or not families:
        raise ValueError("semantic property families must be a nonempty mapping")
    if not isinstance(relations, dict) or not relations:
        raise ValueError("semantic relations must be a nonempty mapping")

    all_values: set[str] = set()
    for family, specification in families.items():
        if not isinstance(family, str) or not family:
            raise ValueError("semantic property family names must be nonempty strings")
        if not isinstance(specification, dict) or set(specification) != _FAMILY_FIELDS:
            raise ValueError(f"invalid semantic property family: {family!r}")
        if specification["kind"] not in _FAMILY_KINDS:
            raise ValueError(f"invalid property kind for {family!r}")
        if specification["stability"] not in _STABILITIES:
            raise ValueError(f"invalid property stability for {family!r}")
        if specification["cardinality"] not in _CARDINALITIES:
            raise ValueError(f"invalid property cardinality for {family!r}")
        _validate_sources(specification["source_conventions"], family)

        values = specification["values"]
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str)
                or not value.startswith(f"property:{family}/")
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"invalid property values for {family!r}")
        if all_values & set(values):
            raise ValueError("semantic property values must be globally unique")
        if any(value.endswith(("/other", "/unknown")) for value in values):
            raise ValueError("unknown and other are not semantic property targets")
        all_values.update(values)

    for relation, specification in relations.items():
        if not isinstance(relation, str) or not relation.startswith("relation:"):
            raise ValueError("semantic relation labels must use the relation namespace")
        if not isinstance(specification, dict) or set(specification) != _RELATION_FIELDS:
            raise ValueError(f"invalid semantic relation: {relation!r}")
        if specification["kind"] not in _RELATION_KINDS:
            raise ValueError(f"invalid relation kind for {relation!r}")
        if specification["stability"] not in _STABILITIES | {"time_varying"}:
            raise ValueError(f"invalid relation stability for {relation!r}")
        if not isinstance(specification["symmetric"], bool):
            raise ValueError(f"invalid relation symmetry for {relation!r}")
        if any(
            not isinstance(specification[field], str) or not specification[field]
            for field in ("subject_scope", "object_scope")
        ):
            raise ValueError(f"invalid relation scope for {relation!r}")
        _validate_sources(specification["source_conventions"], relation)

        inverse = specification["inverse"]
        if inverse is not None and inverse not in relations:
            raise ValueError(f"unknown inverse relation for {relation!r}")
        if specification["symmetric"] != (inverse == relation):
            raise ValueError(f"inconsistent symmetry for {relation!r}")
        if inverse is not None and relations[inverse].get("inverse") != relation:
            raise ValueError(f"inconsistent inverse relation for {relation!r}")
    return document


def semantic_property_groups(
    inventory: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return deterministic property groups for ``IndexVocabulary.from_groups``."""

    return {
        f"semantic_property/{family}": tuple(specification["values"])
        for family, specification in inventory["families"].items()
    }


def semantic_relation_labels(inventory: Mapping[str, Any]) -> tuple[str, ...]:
    """Return semantic-relation labels in reviewed inventory order."""

    return tuple(inventory["relations"])

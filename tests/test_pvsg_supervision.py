import pytest
import torch

from experiments.pvsg.indices import build_section6_vocabulary
from experiments.pvsg.supervision import (
    IGNORE_INDEX,
    build_category_targets,
    build_object_targets,
    build_pair_targets,
    object_losses,
    object_metrics,
    pair_losses,
    pair_metrics,
)


def _ontology():
    return {
        "schema_version": 1,
        "predicates": ["holding", "looking at"],
        "train_supported_predicates": ["holding", "looking at"],
        "train_unseen_predicates": [],
        "object_categories": {"thing": ["dog", "ball", "gift"], "stuff": []},
        "identities": [
            {"name": "identity:dog", "category": "dog"},
            {"name": "identity:ball", "category": "ball"},
            {"name": "identity:gift", "category": "gift"},
        ],
    }


def _hierarchy():
    return {
        "paths": {
            "dog": ["category:dog", "category:animal", "category:living"],
            "ball": ["category:ball", "category:toy", "category:equipment"],
            "gift": None,
        },
        "identity_paths": {},
        "domains": {
            "category:living": "domain:natural",
            "category:equipment": "domain:artifact",
        },
    }


def _batch():
    return {
        "subject_identity": ("identity:dog", "identity:gift"),
        "object_identity": ("identity:ball", "identity:dog"),
        "subject_category": ("dog", "gift"),
        "object_category": ("ball", "dog"),
        "predicates": [("holding",), ("holding", "looking at")],
    }


def _vocabulary():
    return build_section6_vocabulary(
        _ontology(),
        identity_names=("identity:dog", "identity:ball", "identity:gift"),
        category_levels=("fine", "basic", "coarse", "domain"),
        hierarchy=_hierarchy(),
    )


def test_pair_targets_use_candidate_positions_multihot_predicates_and_masks() -> None:
    targets = build_pair_targets(_batch(), _vocabulary(), hierarchy=_hierarchy())

    assert targets.subject_identity.tolist() == [0, 2]
    assert targets.object_identity.tolist() == [1, 0]
    torch.testing.assert_close(
        targets.predicates,
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
    )
    for target in targets.subject_categories.values():
        assert target.tolist() == [0, IGNORE_INDEX]
    for target in targets.object_categories.values():
        assert target.tolist() == [1, 0]


def test_pair_targets_support_the_official_source_category_condition() -> None:
    vocabulary = build_section6_vocabulary(
        _ontology(),
        identity_names=("identity:dog", "identity:ball", "identity:gift"),
        category_levels=("source",),
    )

    targets = build_pair_targets(_batch(), vocabulary)

    assert targets.subject_categories["object_category/source"].tolist() == [0, 2]
    assert targets.object_categories["object_category/source"].tolist() == [1, 0]


def test_pair_objective_is_finite_and_strict_metrics_recognize_exact_predictions() -> None:
    targets = build_pair_targets(_batch(), _vocabulary(), hierarchy=_hierarchy())
    category_groups = tuple(targets.subject_categories)
    outputs = {
        "subject_identity_logits": torch.tensor(
            [[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]], requires_grad=True
        ),
        "object_identity_logits": torch.tensor(
            [[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]], requires_grad=True
        ),
        "predicate_logits": torch.tensor(
            [[5.0, -5.0], [5.0, 5.0]], requires_grad=True
        ),
        "subject_category_logits": {
            group: torch.tensor([[5.0, 0.0], [0.0, 0.0]], requires_grad=True)
            for group in category_groups
        },
        "object_category_logits": {
            group: torch.tensor([[0.0, 5.0], [5.0, 0.0]], requires_grad=True)
            for group in category_groups
        },
    }

    losses = pair_losses(outputs, targets)
    metrics = pair_metrics(outputs, targets)
    losses.total.backward()

    assert torch.isfinite(losses.total)
    torch.testing.assert_close(
        losses.predicate_cross_entropy,
        losses.predicate_kl + losses.predicate_target_entropy,
    )
    active_category_losses = [
        loss
        for owner_targets, owner_losses in (
            (targets.subject_categories, losses.subject_categories),
            (targets.object_categories, losses.object_categories),
        )
        for group, loss in owner_losses.items()
        if bool((owner_targets[group] != IGNORE_INDEX).any())
    ]
    torch.testing.assert_close(
        losses.category_block,
        torch.stack(active_category_losses).mean(),
    )
    assert metrics["accuracy/all_exact"] == 1.0
    assert outputs["predicate_logits"].grad is not None


def test_object_targets_losses_and_metrics_reuse_entity_supervision() -> None:
    vocabulary = _vocabulary()
    targets = build_object_targets(
        {"identity": ("identity:dog", "identity:gift"), "category": ("dog", "gift")},
        vocabulary,
        hierarchy=_hierarchy(),
    )
    outputs = {
        "identity_logits": torch.tensor(
            [[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]], requires_grad=True
        ),
        "category_logits": {
            group: torch.tensor([[5.0, 0.0], [0.0, 0.0]], requires_grad=True)
            for group in targets.categories
        },
    }

    losses = object_losses(outputs, targets)
    metrics = object_metrics(outputs, targets)
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert metrics["accuracy/all_exact"] == 1.0
    assert outputs["identity_logits"].grad is not None


def test_novel_identities_can_be_evaluated_against_supported_categories() -> None:
    vocabulary = build_section6_vocabulary(
        _ontology(),
        identity_names=("identity:dog", "identity:ball"),
        category_levels=("source",),
    )
    batch = {
        "identity": ("identity:novel-dog", "identity:novel-gift"),
        "category": ("dog", "gift"),
    }

    with pytest.raises(ValueError, match="outside the candidates"):
        build_category_targets(batch, vocabulary)
    targets = build_category_targets(batch, vocabulary, allow_unknown=True)

    assert targets["object_category/source"].tolist() == [0, IGNORE_INDEX]


def test_pair_targets_reject_predicates_outside_the_frozen_candidates() -> None:
    batch = _batch()
    batch["predicates"] = [("riding",), ("holding",)]

    with pytest.raises(ValueError, match="outside the candidates"):
        build_pair_targets(batch, _vocabulary(), hierarchy=_hierarchy())

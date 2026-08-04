import pytest
import torch
from torch import nn

from experiments.pvsg.evaluation import evaluate_objects
from tb import IndexVocabulary


class _FixedObjectModel(nn.Module):
    def forward_object(
        self,
        scene_features,
        object_features,
        identity_candidates,
        *,
        category_candidates,
        feedback_mode,
    ):
        del scene_features, identity_candidates, feedback_mode
        return {
            "identity_logits": object_features,
            "category_logits": {
                group: object_features for group in category_candidates
            },
        }


def _forward(model):
    return lambda batch, candidates: model.forward_object(
        batch["scene_features"],
        batch["object_features"],
        candidates["identity"],
        category_candidates={
            group: indices
            for group, indices in candidates.items()
            if group.startswith("object_category/")
        },
        feedback_mode="p-sa",
    )


def _vocabulary():
    return IndexVocabulary.from_groups(
        {
            "object_category/source": ("category:dog", "category:ball"),
            "identity": ("identity:dog", "identity:ball"),
        }
    )


def test_object_evaluation_scores_known_identities() -> None:
    batch = {
        "scene_features": torch.zeros(2, 2),
        "object_features": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        "identity": ("identity:dog", "identity:ball"),
        "category": ("dog", "ball"),
        "source": ("vidor", "vidor"),
        "video_id": ("a", "b"),
    }

    result = evaluate_objects(
        _forward(_FixedObjectModel()),
        [batch],
        _vocabulary(),
        device=torch.device("cpu"),
        hierarchy=None,
        identities=True,
    )

    assert result["accuracy/identity"] == 1.0
    assert result["accuracy/identity_macro"] == 1.0
    assert result["accuracy/identity_video_macro"] == 1.0
    assert result["accuracy/category/object_category/source"] == 1.0
    assert result[
        "accuracy/category_class_macro/object_category/source"
    ] == 1.0
    assert result["support/category/object_category/source"] == 2
    assert result[
        "support/category_given_identity_correct/object_category/source"
    ] == 2
    assert result[
        "support/category_given_identity_incorrect/object_category/source"
    ] == 0


def test_object_evaluation_ignores_unsupported_novel_categories() -> None:
    batch = {
        "scene_features": torch.zeros(2, 2),
        "object_features": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        "identity": ("identity:novel-dog", "identity:novel-gift"),
        "category": ("dog", "gift"),
        "source": ("vidor", "vidor"),
        "video_id": ("a", "b"),
    }

    result = evaluate_objects(
        _forward(_FixedObjectModel()),
        [batch],
        _vocabulary(),
        device=torch.device("cpu"),
        hierarchy=None,
        identities=False,
    )

    assert result["support/category/object_category/source"] == 1
    assert result["ignored/category/object_category/source"] == 1
    assert result["accuracy/category/object_category/source"] == 1.0


def test_object_evaluation_distinguishes_micro_and_macro_accuracies() -> None:
    batch = {
        "scene_features": torch.zeros(4, 2),
        "object_features": torch.tensor(
            [[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]
        ),
        "identity": (
            "identity:dog",
            "identity:dog",
            "identity:dog",
            "identity:ball",
        ),
        "category": ("dog", "dog", "dog", "ball"),
        "source": ("vidor",) * 4,
        "video_id": ("a", "a", "a", "b"),
    }

    result = evaluate_objects(
        _forward(_FixedObjectModel()),
        [batch],
        _vocabulary(),
        device=torch.device("cpu"),
        hierarchy=None,
        identities=True,
    )

    assert result["accuracy/identity"] == 0.5
    assert result["accuracy/identity_macro"] == pytest.approx(1 / 3)
    assert result["accuracy/identity_video_macro"] == pytest.approx(1 / 3)
    assert result[
        "accuracy/category_class_macro/object_category/source"
    ] == pytest.approx(1 / 3)
    assert result[
        "accuracy/category_identity_macro/object_category/source"
    ] == pytest.approx(1 / 3)
    assert result[
        "accuracy/category_video_macro/object_category/source"
    ] == pytest.approx(1 / 3)
    assert result["accuracy/category_level_mean_observation_micro"] == 0.5
    assert result["accuracy/category_level_mean_class_macro"] == pytest.approx(1 / 3)
    assert result[
        "support/category_given_identity_correct/object_category/source"
    ] == 2
    assert result[
        "accuracy/category_given_identity_correct/object_category/source"
    ] == 1.0
    assert result[
        "support/category_given_identity_incorrect/object_category/source"
    ] == 2
    assert result[
        "accuracy/category_given_identity_incorrect/object_category/source"
    ] == 0.0
    assert result[
        "accuracy/category_level_mean_observation_micro_given_identity_correct"
    ] == 1.0
    assert result[
        "accuracy/category_level_mean_observation_micro_given_identity_incorrect"
    ] == 0.0

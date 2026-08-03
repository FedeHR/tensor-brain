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
    }

    result = evaluate_objects(
        _FixedObjectModel(),
        [batch],
        _vocabulary(),
        device=torch.device("cpu"),
        hierarchy=None,
        feedback_mode="p-sa",
        identities=True,
    )

    assert result["accuracy/identity"] == 1.0
    assert result["accuracy/category/object_category/source"] == 1.0
    assert result["support/category/object_category/source"] == 2


def test_object_evaluation_ignores_unsupported_novel_categories() -> None:
    batch = {
        "scene_features": torch.zeros(2, 2),
        "object_features": torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        "identity": ("identity:novel-dog", "identity:novel-gift"),
        "category": ("dog", "gift"),
    }

    result = evaluate_objects(
        _FixedObjectModel(),
        [batch],
        _vocabulary(),
        device=torch.device("cpu"),
        hierarchy=None,
        feedback_mode="p-samp",
        identities=False,
    )

    assert result["support/category/object_category/source"] == 1
    assert result["ignored/category/object_category/source"] == 1
    assert result["accuracy/category/object_category/source"] == 1.0

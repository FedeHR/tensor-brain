import torch

from experiments.pvsg.audit import relation_records_for_video, validate_feature_artifact
from experiments.pvsg.extract import (
    DINO_MODEL_ID,
    DINO_MODEL_REVISION,
    FEATURE_SCHEMA_VERSION,
)
from experiments.pvsg.prepare import PVSG_JSON_SHA256


def _artifact() -> dict:
    return {
        "scene_frame_index": torch.tensor([0, 1]),
        "scene_features": torch.ones(2, 3, dtype=torch.float16),
        "object_frame_index": torch.tensor([0, 0, 1, 1]),
        "object_ids": torch.tensor([1, 2, 1, 2]),
        "object_mask_areas": torch.tensor([4, 5, 4, 5]),
        "object_features": torch.ones(4, 3, dtype=torch.float16),
        "pair_frame_index": torch.tensor([0, 1]),
        "pair_ids": torch.tensor([[1, 2], [1, 2]]),
        "union_boxes_xyxy": torch.tensor([[0, 0, 4, 3], [0, 0, 4, 3]]),
        "union_features": torch.ones(2, 3, dtype=torch.float16),
        "metadata": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "source": "ego4d",
            "video_id": "video-1",
            "num_frames": 2,
            "original_size_hw": (3, 4),
            "feature_dim": 3,
            "dino_model_id": DINO_MODEL_ID,
            "dino_model_revision": DINO_MODEL_REVISION,
            "pvsg_json_sha256": PVSG_JSON_SHA256,
            "feature_storage_dtype": "float16",
            "inference_autocast_dtype": "float16",
        },
    }


def test_feature_audit_validates_complete_visible_pairs() -> None:
    result = validate_feature_artifact(
        _artifact(),
        {
            "source": "ego4d",
            "video_id": "video-1",
            "num_frames": 2,
            "height": 3,
            "width": 4,
            "num_objects": 2,
        },
    )

    assert result["counts"] == {
        "frames": 2,
        "object_observations": 4,
        "pair_observations": 2,
    }
    assert result["pair_keys"] == {(0, 1, 2), (1, 1, 2)}


def test_relation_expansion_keeps_all_labels_and_requires_explicit_endpoint_policy() -> None:
    video = {
        "meta": {"num_frames": 3},
        "relations": [
            [1, 2, "beside", [[0, 1]]],
            [1, 2, "looking_at", [[1, 2]]],
            [1, 2, "beside", [[1, 2]]],
        ],
    }

    half_open, half_open_issues = relation_records_for_video(
        video,
        predicate_vocabulary={"beside", "looking_at"},
        convention="half_open",
    )
    inclusive, inclusive_issues = relation_records_for_video(
        video,
        predicate_vocabulary={"beside", "looking_at"},
        convention="inclusive",
    )

    assert half_open == {
        (0, 1, 2): {"beside"},
        (1, 1, 2): {"beside", "looking_at"},
    }
    assert inclusive == {
        (0, 1, 2): {"beside"},
        (1, 1, 2): {"beside", "looking_at"},
        (2, 1, 2): {"beside", "looking_at"},
    }
    assert not half_open_issues["invalid_spans"]
    assert not inclusive_issues["invalid_spans"]

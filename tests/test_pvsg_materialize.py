import torch

from experiments.pvsg.materialize import (
    EXPECTED_FPS,
    FEWSHOT_K_VALUES,
    SUPPORT_COUNT,
    _experiment_splits,
    _frame_object_rows,
    _pair_record,
)


def test_experiment_splits_reserve_development_videos_by_source() -> None:
    annotation = {
        "split": {
            "vidor": {
                "train": [f"vidor-train-{index}" for index in range(20)],
                "val": ["vidor-evaluation"],
            },
            "ego4d": {
                "train": [f"ego4d-train-{index}" for index in range(10)],
                "val": ["ego4d-evaluation"],
            },
        }
    }

    roles = _experiment_splits(annotation, {"vidor-train-0"})

    assert "vidor-train-0" not in roles
    assert sum(role == "development" for role in roles.values()) == 5
    assert roles["vidor-evaluation"] == "evaluation"
    assert roles["ego4d-evaluation"] == "evaluation"


def test_fewshot_snapshot_uses_nested_support_counts_at_five_fps() -> None:
    assert EXPECTED_FPS == 5
    assert FEWSHOT_K_VALUES == (1, 3, 5, 10)
    assert SUPPORT_COUNT == 10


def test_frame_object_rows_include_empty_frames_and_sort_visible_objects() -> None:
    artifact = {
        "object_frame_index": torch.tensor([0, 0, 2, 2, 2]),
        "object_ids": torch.tensor([3, 1, 4, 2, 1]),
    }

    frames = _frame_object_rows(artifact, num_frames=4)

    assert frames == [([1, 3], [1, 0]), ([], []), ([1, 2, 4], [4, 3, 2]), ([], [])]


def test_pair_record_preserves_each_missing_evidence_source() -> None:
    record = _pair_record(
        source="vidor",
        video_id="video",
        official_split="train",
        experiment_split="development",
        frame_index=7,
        subject_id=1,
        object_id=2,
        predicates=("holding",),
        predicate_rank={"holding": 0},
        categories={1: {"category": "adult"}, 2: {"category": "cup"}},
        object_rows={(7, 2): 9},
        pair_rows={},
    )

    assert record["subject_row"] is None
    assert record["object_row"] == 9
    assert record["union_row"] is None
    assert not record["has_subject_evidence"]
    assert record["has_object_evidence"]
    assert not record["has_union_evidence"]
    assert not record["has_complete_evidence"]

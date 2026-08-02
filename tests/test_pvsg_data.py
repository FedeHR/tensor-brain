import json
import math

import pytest
import torch
from torch.utils.data import DataLoader

from experiments.pvsg import data as pvsg_data
from experiments.pvsg.data import (
    PVSGObjectDataset,
    PVSGPairDataset,
    VideoBlockSampler,
    collate_pair_batch,
    normalize_dino,
)


def _write_artifact(root, *, source="vidor", video_id="video"):
    path = root / "videos" / source / f"{video_id}.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "metadata": {
                "schema_version": 2,
                "source": source,
                "video_id": video_id,
            },
            "scene_features": torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float16),
            "object_features": torch.tensor(
                [[0.0, 5.0], [8.0, 6.0], [1.0, 0.0]], dtype=torch.float16
            ),
            "union_features": torch.tensor([[-3.0, 0.0], [0.0, -7.0]], dtype=torch.float16),
        },
        path,
    )


def _write_jsonl(path, rows):
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def test_normalize_dino_uses_float32_per_vector_rms() -> None:
    features = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float16)

    normalized = normalize_dino(features)

    assert normalized.dtype == torch.float32
    torch.testing.assert_close(
        normalized,
        math.sqrt(2) * torch.tensor([[0.6, 0.8], [0.0, 0.0]]),
    )
    torch.testing.assert_close(normalized[0].square().mean(), torch.tensor(1.0))


def test_object_dataset_looks_up_normalized_feature_rows(tmp_path) -> None:
    feature_root = tmp_path / "features"
    _write_artifact(feature_root)
    manifest = tmp_path / "objects.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "source": "vidor",
                "video_id": "video",
                "frame_index": 1,
                "object_id": 2,
                "identity": "identity:vidor/video/2",
                "category": "dog",
                "scene_row": 1,
                "object_row": 1,
                "mask_area": 20,
            }
        ],
    )

    batch = next(iter(DataLoader(PVSGObjectDataset(manifest, feature_root), batch_size=1)))

    scale = math.sqrt(2)
    torch.testing.assert_close(batch["scene_features"], scale * torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(
        batch["object_features"], scale * torch.tensor([[0.8, 0.6]])
    )
    assert batch["identity"] == ["identity:vidor/video/2"]
    assert batch["frame_index"].tolist() == [1]


def test_pair_dataset_and_collator_preserve_multilabel_targets(tmp_path) -> None:
    feature_root = tmp_path / "features"
    _write_artifact(feature_root)
    base = {
        "source": "vidor",
        "video_id": "video",
        "frame_index": 0,
        "subject_id": 1,
        "object_id": 2,
        "subject_identity": "identity:vidor/video/1",
        "object_identity": "identity:vidor/video/2",
        "subject_category": "adult",
        "object_category": "dog",
        "scene_row": 0,
        "subject_row": 0,
        "object_row": 1,
        "union_row": 0,
        "has_complete_evidence": True,
    }
    manifest = tmp_path / "pairs.jsonl"
    _write_jsonl(
        manifest,
        [
            {**base, "predicates": ["holding"]},
            {**base, "predicates": ["holding", "looking at"]},
        ],
    )

    dataset = PVSGPairDataset(manifest, feature_root)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_pair_batch)))

    scale = math.sqrt(2)
    torch.testing.assert_close(batch["scene_features"], scale * torch.tensor([[0.6, 0.8]] * 2))
    torch.testing.assert_close(
        batch["subject_features"], scale * torch.tensor([[0.0, 1.0]] * 2)
    )
    torch.testing.assert_close(
        batch["object_features"], scale * torch.tensor([[0.8, 0.6]] * 2)
    )
    torch.testing.assert_close(
        batch["union_features"], scale * torch.tensor([[-1.0, 0.0]] * 2)
    )
    assert batch["predicates"] == [("holding",), ("holding", "looking at")]


def test_dataset_prepares_only_addressed_rows_and_reuses_video_tables(
    tmp_path, monkeypatch
) -> None:
    feature_root = tmp_path / "features"
    _write_artifact(feature_root)
    manifest = tmp_path / "objects.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "source": "vidor",
                "video_id": "video",
                "frame_index": frame_index,
                "object_id": frame_index + 1,
                "identity": f"identity:vidor/video/{frame_index + 1}",
                "category": "dog",
                "scene_row": frame_index,
                "object_row": frame_index,
                "mask_area": 20,
            }
            for frame_index in range(2)
        ],
    )
    load_calls = []
    prepared_shapes = []
    original_load = pvsg_data.load_feature_artifact
    original_normalize = pvsg_data.normalize_dino

    def counted_load(path):
        load_calls.append(path)
        return original_load(path)

    def recorded_normalize(features):
        prepared_shapes.append(tuple(features.shape))
        return original_normalize(features)

    pvsg_data._video_feature_tables.cache_clear()
    monkeypatch.setattr(pvsg_data, "load_feature_artifact", counted_load)
    monkeypatch.setattr(pvsg_data, "normalize_dino", recorded_normalize)
    dataset = PVSGObjectDataset(manifest, feature_root)

    dataset[0]
    dataset[1]

    assert len(load_calls) == 1
    assert prepared_shapes == [(2,), (2,), (2,), (2,)]


def test_video_block_sampler_keeps_each_video_contiguous() -> None:
    records = [
        {"source": "vidor", "video_id": "a"},
        {"source": "vidor", "video_id": "b"},
        {"source": "vidor", "video_id": "a"},
        {"source": "ego4d", "video_id": "c"},
        {"source": "vidor", "video_id": "b"},
        {"source": "vidor", "video_id": "a"},
    ]
    sampler = VideoBlockSampler(records, generator=torch.Generator().manual_seed(7))

    order = list(sampler)

    assert sorted(order) == list(range(len(records)))
    for key in {(record["source"], record["video_id"]) for record in records}:
        positions = [
            position
            for position, index in enumerate(order)
            if (records[index]["source"], records[index]["video_id"]) == key
        ]
        assert positions == list(range(min(positions), max(positions) + 1))


def test_pair_dataset_rejects_incomplete_canonical_rows(tmp_path) -> None:
    feature_root = tmp_path / "features"
    _write_artifact(feature_root)
    manifest = tmp_path / "incomplete.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "source": "vidor",
                "video_id": "video",
                "has_complete_evidence": False,
            }
        ],
    )

    dataset = PVSGPairDataset(manifest, feature_root)
    with pytest.raises(ValueError, match="requires complete DINO evidence"):
        dataset[0]

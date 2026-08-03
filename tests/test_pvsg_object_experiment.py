import json

import torch

from experiments.pvsg.object_experiment import (
    ObjectExperimentConfig,
    run_object_experiment,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def _artifact(feature_root, video_id, rows):
    path = feature_root / "videos" / "vidor" / f"{video_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    features = torch.tensor(rows, dtype=torch.float16)
    torch.save(
        {
            "metadata": {
                "schema_version": 2,
                "source": "vidor",
                "video_id": video_id,
            },
            "scene_features": features,
            "object_features": features.flip(dims=(1,)),
            "union_features": torch.empty(len(rows), 2, dtype=torch.float16),
        },
        path,
    )


def _record(video_id, row, identity, category, role):
    return {
        "source": "vidor",
        "video_id": video_id,
        "official_split": "train",
        "experiment_split": role,
        "frame_index": row,
        "object_id": row,
        "identity": identity,
        "category": category,
        "scene_row": row,
        "object_row": row,
        "mask_area": 10,
    }


def test_full_object_runner_writes_selected_and_final_evaluations(tmp_path) -> None:
    manifest_root = tmp_path / "snapshot"
    feature_root = tmp_path / "features"
    train_rows = [
        _record("train", 0, "identity:dog", "dog", "train"),
        _record("train", 1, "identity:ball", "ball", "train"),
    ]
    blocked_rows = [
        _record("train", 2, "identity:dog", "dog", "train"),
        _record("train", 3, "identity:ball", "ball", "train"),
    ]
    development_rows = [
        _record("development", 0, "identity:novel-dog", "dog", "development"),
        _record("development", 1, "identity:novel-gift", "gift", "development"),
    ]
    _write_jsonl(manifest_root / "blocked" / "train_objects.jsonl", train_rows)
    _write_jsonl(
        manifest_root / "blocked" / "evaluation_objects.jsonl", blocked_rows
    )
    _write_jsonl(
        manifest_root / "heldout_video" / "development_objects.jsonl",
        development_rows,
    )
    _write_json(
        manifest_root / "ontology.json",
        {
            "schema_version": 1,
            "predicates": ["holding"],
            "train_supported_predicates": ["holding"],
            "train_unseen_predicates": [],
            "object_categories": {"thing": ["dog", "ball", "gift"], "stuff": []},
            "identities": [
                {"name": "identity:dog", "category": "dog"},
                {"name": "identity:ball", "category": "ball"},
            ],
        },
    )
    _write_json(manifest_root / "provenance.json", {"schema_version": 1})
    _artifact(feature_root, "train", [[1, 2], [2, 1], [1, 3], [3, 1]])
    _artifact(feature_root, "development", [[1, 1], [2, 3]])

    result = run_object_experiment(
        manifest_root,
        feature_root,
        tmp_path / "runs",
        ObjectExperimentConfig(
            run_name="smoke",
            evolution="qtb",
            score_mode="centered",
            learning_rate=1e-3,
            semantic_condition="source",
            batch_size=2,
            max_steps=2,
            validation_every=1,
            validation_examples=2,
            log_every=1,
            num_workers=0,
            device="cpu",
        ),
    )

    run_dir = tmp_path / "runs" / "smoke"
    assert set(result["evaluation"]) == {"development", "blocked"}
    assert set(result["evaluation"]["blocked"]) == {"p-sa", "p-samp"}
    assert result["evaluation"]["blocked"]["p-sa"]["examples"] == 2
    assert result["evaluation"]["development"]["p-sa"][
        "ignored/category/object_category/source"
    ] == 1
    assert {
        "checkpoint.pt",
        "config.json",
        "result.json",
        "scale_trace.jsonl",
        "training_trace.jsonl",
        "validation_trace.jsonl",
        "vocabulary.json",
    } == {path.name for path in run_dir.iterdir()}

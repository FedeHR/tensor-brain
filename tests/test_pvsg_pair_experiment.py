import json
import sys

import pytest
import torch

from experiments.pvsg.pair_experiment import (
    PairExperimentConfig,
    _parse_args,
    run_pair_experiment,
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
            "metadata": {"schema_version": 2, "source": "vidor", "video_id": video_id},
            "scene_features": features,
            "object_features": features.flip(dims=(1,)),
            "union_features": features.roll(1, dims=1),
        },
        path,
    )


def _record(video_id, row, role, subject, object_, predicate):
    return {
        "source": "vidor",
        "video_id": video_id,
        "official_split": "train",
        "experiment_split": role,
        "frame_index": row,
        "subject_id": 1,
        "object_id": 2,
        "subject_identity": subject,
        "object_identity": object_,
        "subject_category": "dog",
        "object_category": "ball",
        "predicates": predicate,
        "scene_row": row,
        "subject_row": row,
        "object_row": row,
        "union_row": row,
        "has_complete_evidence": True,
    }


def test_pair_cli_accepts_the_cluster_arguments(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pair_experiment",
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--feature-root",
            str(tmp_path / "features"),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-name",
            "pair",
            "--condition",
            "integral-p-sa",
            "--evolution",
            "qtb",
            "--score-mode",
            "softplus-bias",
            "--learning-rate",
            "0.001",
            "--chunk-size",
            "1024",
            "--seed",
            "0",
        ],
    )

    _manifest_root, _feature_root, _output_root, config = _parse_args()

    assert config.condition == "integral-p-sa"
    assert config.chunk_size == 1024


@pytest.mark.parametrize(
    ("condition", "evaluation_modes", "expected_files"),
    (
        (
            "priors",
            {"frequency", "category-pair"},
            {"config.json", "result.json", "vocabulary.json"},
        ),
        (
            "category-only",
            {"default"},
            {"config.json", "result.json", "vocabulary.json"},
        ),
        (
            "union-only",
            {"default"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "union-category-oracle",
            {"default"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "union-category-predicted",
            {"default", "oracle-category-intervention"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "linear-probe",
            {"default"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "fused-linear",
            {"default"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "flat-fusion",
            {"default"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "p-direct",
            {"default"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "scale_trace.jsonl",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "integral-none",
            {"none"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "scale_trace.jsonl",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
        (
            "integral-p-sa",
            {"p-sa", "p-samp"},
            {
                "checkpoint.pt",
                "config.json",
                "result.json",
                "scale_trace.jsonl",
                "training_trace.jsonl",
                "validation_trace.jsonl",
                "vocabulary.json",
            },
        ),
    ),
)
def test_pair_runner_supports_every_comparison_condition(
    tmp_path, condition, evaluation_modes, expected_files
) -> None:
    manifest_root = tmp_path / "snapshot"
    feature_root = tmp_path / "features"
    train_rows = [
        _record("train", 0, "train", "identity:dog", "identity:ball", ["holding"]),
        _record("train", 1, "train", "identity:ball", "identity:dog", ["looking at"]),
    ]
    development_rows = [
        _record(
            "development",
            0,
            "development",
            "identity:novel-dog",
            "identity:novel-ball",
            ["holding", "riding"],
        ),
        _record(
            "development",
            1,
            "development",
            "identity:novel-ball",
            "identity:novel-dog",
            ["looking at"],
        ),
    ]
    development_rows[1]["subject_category"] = "ball"
    development_rows[1]["object_category"] = "dog"
    _write_jsonl(manifest_root / "heldout_video" / "train_pairs.jsonl", train_rows)
    _write_jsonl(
        manifest_root / "heldout_video" / "development_pairs.jsonl",
        development_rows,
    )
    _write_json(
        manifest_root / "ontology.json",
        {
            "schema_version": 1,
            "predicates": ["holding", "looking at", "riding"],
            "train_supported_predicates": ["holding", "looking at"],
            "train_unseen_predicates": ["riding"],
            "object_categories": {"thing": ["dog", "ball"], "stuff": []},
            "identities": [
                {"name": "identity:dog", "category": "dog"},
                {"name": "identity:ball", "category": "ball"},
                {"name": "identity:novel-dog", "category": "dog"},
                {"name": "identity:novel-ball", "category": "ball"},
            ],
        },
    )
    _artifact(feature_root, "train", [[1, 2], [2, 1]])
    _artifact(feature_root, "development", [[1, 1], [2, 3]])

    result = run_pair_experiment(
        manifest_root,
        feature_root,
        tmp_path / "runs",
        PairExperimentConfig(
            run_name=condition,
            condition=condition,
            batch_size=2,
            max_steps=2,
            validation_every=1,
            validation_examples=2,
            log_every=1,
            num_workers=0,
            device="cpu",
        ),
    )

    assert set(result["evaluation"]) == evaluation_modes
    assert expected_files == {path.name for path in (tmp_path / "runs" / condition).iterdir()}
    for metrics in result["evaluation"].values():
        assert metrics["examples"] == 2
        assert metrics["predicate_assignments"] == 2
        assert metrics["support/triple_unseen"] == 1

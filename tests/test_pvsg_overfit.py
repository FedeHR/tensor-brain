import json

import pytest
import torch

from experiments.pvsg.overfit import (
    OverfitConfig,
    PreparedOverfit,
    run_overfit,
)
from tb import IndexVocabulary


def _vocabulary():
    return IndexVocabulary.from_groups(
        {
            "predicate": ("predicate:holding", "predicate:looking at"),
            "object_category/source": ("category:dog", "category:ball"),
            "identity": ("identity:dog", "identity:ball"),
        }
    )


def _batch():
    return {
        "scene_features": torch.tensor([[0.5, -0.5], [-0.3, 0.8]]),
        "subject_features": torch.tensor([[0.2, 1.0], [0.5, -0.4]]),
        "object_features": torch.tensor([[1.0, -0.1], [-0.2, 0.7]]),
        "union_features": torch.tensor([[0.4, 0.9], [0.8, -0.6]]),
        "subject_identity": ("identity:dog", "identity:ball"),
        "object_identity": ("identity:ball", "identity:dog"),
        "subject_category": ("dog", "ball"),
        "object_category": ("ball", "dog"),
        "predicates": [("holding",), ("looking at",)],
    }


@pytest.mark.parametrize(
    "evolution", ("original", "qtb", "relu")
)
def test_integral_runner_supports_each_evolution_and_scale_trace(
    tmp_path, evolution
) -> None:
    vocabulary = _vocabulary()
    batch = _batch()
    prepared = PreparedOverfit(
        batch=batch,
        records=(
            {"video_id": "v", "frame_index": 0},
            {"video_id": "v", "frame_index": 1},
        ),
        vocabulary=vocabulary,
        hierarchy=None,
    )
    config = OverfitConfig(
        run_name=evolution,
        evolution=evolution,
        semantic_condition="source",
        num_examples=2,
        max_steps=1,
        success_loss=1e-12,
        capture_steps=(0, 1),
        device="cpu",
    )

    result = run_overfit(prepared, tmp_path / evolution, config)
    rows = [
        json.loads(line)
        for line in (tmp_path / evolution / "scale_trace.jsonl").read_text().splitlines()
    ]

    kinds = {row["kind"] for row in rows}
    assert {"state", "cbs", "attention", "feedback", "readout", "gradient"} <= kinds
    applied = next(
        row
        for row in rows
        if row["kind"] == "feedback"
        and row["window"] == "subject"
        and row["tensor"] == "applied_feedback"
    )
    assert applied["l2_over_input_drive"] >= 0.0
    assert all(
        torch.isfinite(torch.tensor(value))
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    )
    assert set(result["evaluation"]) == {"p-sa", "p-samp"}
    stored = json.loads((tmp_path / evolution / "config.json").read_text())
    assert stored["resolved_evolution"] == evolution


def test_overfit_runner_can_fit_and_save_a_fixed_batch(tmp_path) -> None:
    vocabulary = _vocabulary()
    batch = _batch()
    prepared = PreparedOverfit(
        batch=batch,
        records=(
            {"video_id": "v", "frame_index": 0},
            {"video_id": "v", "frame_index": 1},
        ),
        vocabulary=vocabulary,
        hierarchy=None,
    )
    config = OverfitConfig(
        run_name="smoke",
        model="p-direct",
        semantic_condition="source",
        num_examples=2,
        learning_rate=1.0,
        max_steps=500,
        capture_steps=(0, 1, 10, 100),
        device="cpu",
    )

    result = run_overfit(prepared, tmp_path / "smoke", config)

    assert result["success"]
    assert result["completed_steps"] <= 500
    expected = {
        "batch.jsonl",
        "checkpoint.pt",
        "config.json",
        "predictions.pt",
        "result.json",
        "scale_trace.jsonl",
        "training_trace.jsonl",
        "vocabulary.json",
    }
    assert {path.name for path in (tmp_path / "smoke").iterdir()} == expected
    stored = json.loads((tmp_path / "smoke" / "result.json").read_text())
    assert stored["success"] is True
    assert json.loads((tmp_path / "smoke" / "config.json").read_text())[
        "resolved_evolution"
    ] is None

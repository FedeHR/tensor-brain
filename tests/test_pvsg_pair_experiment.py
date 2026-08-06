import json
import sys

import pytest
import torch

from experiments.pvsg.pair_evaluate import reevaluate_pair_run
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


def _hierarchy():
    return {
        "paths": {
            "dog": ["category:dog", "category:animal", "category:living"],
            "ball": ["category:ball", "category:toy", "category:artifact"],
        },
        "identity_paths": {},
        "domains": {
            "category:living": "domain:natural",
            "category:artifact": "domain:manufactured",
        },
    }


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
            "0.0001",
            "--input-mapping-learning-rate",
            "0.00001",
            "--chunk-size",
            "1024",
            "--seed",
            "0",
        ],
    )

    _manifest_root, _feature_root, _output_root, config = _parse_args()

    assert config.condition == "integral-p-sa"
    assert config.chunk_size == 1024
    assert config.input_mapping_learning_rate == 1e-5


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
            {"p-sa", "p-samp", "p-sa-sequential", "p-sa-sequential-sample"},
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
            "integral-cat-sa",
            {"cat-sa", "cat-samp", "cat-sa-sequential", "cat-sa-sequential-sample"},
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
            "integral-id-cat-sa",
            {"id-cat-sa", "id-cat-samp", "id-cat-sa-sequential", "id-cat-sa-sequential-sample"},
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
    tmp_path, monkeypatch, condition, evaluation_modes, expected_files
) -> None:
    monkeypatch.setattr(
        "experiments.pvsg.pair_experiment.load_object_hierarchy",
        lambda *_args, **_kwargs: _hierarchy(),
    )
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
            evolution="original" if condition == "integral-none" else "qtb",
            score_mode="direct" if condition == "integral-none" else "softplus-bias",
            zero_initialize_union=condition == "union-category-oracle",
            batch_size=2,
            max_steps=2,
            validation_every=1,
            validation_examples=2,
            log_every=1,
            num_workers=0,
            device="cpu",
        ),
    )

    assert set(result["evaluation"]) == {"development"}
    assert set(result["evaluation"]["development"]) == evaluation_modes
    assert expected_files == {path.name for path in (tmp_path / "runs" / condition).iterdir()}
    for metrics in result["evaluation"]["development"].values():
        assert metrics["examples"] == 2
        assert metrics["predicate_assignments"] == 2
        assert metrics["support/triple_unseen"] == 1
        # Held-out-video entities are novel, so identity metrics stay undefined.
        assert "accuracy/subject_identity" not in metrics
    if condition in {
        "union-only",
        "union-category-oracle",
        "union-category-predicted",
    }:
        first_validation = json.loads(
            (tmp_path / "runs" / condition / "validation_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert first_validation["step"] == 0
    if condition in {"integral-cat-sa", "integral-id-cat-sa"}:
        config = json.loads(
            (tmp_path / "runs" / condition / "config.json").read_text(encoding="utf-8")
        )
        assert config["feedback"]["category"]["mode"] == "p-sa"
        assert config["feedback"]["category"]["candidates"] == "object_category/source"
        assert config["feedback"]["identity"]["mode"] == (
            "p-sa" if condition == "integral-id-cat-sa" else "none"
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "runs" / condition / "scale_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert {
            (row["kind"], row["group"])
            for row in rows
            if row["kind"] in {"attention", "feedback"}
        } >= {("attention", "category"), ("feedback", "category")}
        assert {"category_feedback"} <= {
            row["operation"] for row in rows if row["kind"] == "operation_delta"
        }
        assert all(
            "direction_cosine_mean" in row for row in rows if row["kind"] == "feedback"
        )
    if condition == "integral-p-sa":
        config = json.loads(
            (tmp_path / "runs" / condition / "config.json").read_text(encoding="utf-8")
        )
        assert config["feedback"]["category"]["mode"] == "none"
        assert config["input_mapping"]["name"] == "shared_linear_g"
        assert config["optimizer"]["input_mapping_learning_rate"] == 1e-5
        vocabulary = json.loads(
            (tmp_path / "runs" / condition / "vocabulary.json").read_text(
                encoding="utf-8"
            )
        )
        assert "object_category/source" in vocabulary["groups"]
        assert "object_category/domain" in vocabulary["groups"]
        assert "accuracy/subject_category/object_category/source" in metrics
        assert "accuracy/object_category/object_category/domain" in metrics
    if condition == "integral-none":
        config = json.loads(
            (tmp_path / "runs" / condition / "config.json").read_text(encoding="utf-8")
        )
        assert config["evolution"] == "original"
        assert config["score_mode"] == "direct"


def _object_record(video_id, row, role, identity, category):
    return {
        "source": "vidor",
        "video_id": video_id,
        "official_split": "train",
        "experiment_split": role,
        "frame_index": row,
        "object_id": 1 if identity.endswith("dog") else 2,
        "identity": identity,
        "category": category,
        "is_thing": True,
        "scene_row": row,
        "object_row": row,
        "mask_area": 100,
    }


def _blocked_record(row, subject, object_, predicate, *, subject_delay, object_delay):
    """A blocked evaluation pair carrying the re-identification delay it was built with."""

    record = _record("train", row, "train", subject, object_, predicate)
    record.update(
        {
            "subject_last_observation_frame": 1,
            "object_last_observation_frame": 1,
            "frames_since_subject_observation": int(subject_delay * 5),
            "frames_since_object_observation": int(object_delay * 5),
            "seconds_since_subject_observation": subject_delay,
            "seconds_since_object_observation": object_delay,
        }
    )
    return record


def _blocked_manifests(manifest_root, feature_root):
    """Materialize the blocked protocol: same video, later frames, same entities."""

    _write_jsonl(
        manifest_root / "blocked" / "train_pairs.jsonl",
        [
            _record("train", 0, "train", "identity:dog", "identity:ball", ["holding"]),
            _record("train", 1, "train", "identity:ball", "identity:dog", ["looking at"]),
        ],
    )
    _write_jsonl(
        manifest_root / "blocked" / "train_objects.jsonl",
        [
            _object_record("train", 0, "train", "identity:dog", "dog"),
            _object_record("train", 1, "train", "identity:ball", "ball"),
            # Observed before the boundary but never in an annotated pair. Enrolling
            # per observed entity is what makes the known-entity claim cover the whole
            # evaluation set, and it is why some columns stay unsupervised.
            _object_record("train", 1, "train", "identity:bystander", "ball"),
        ],
    )
    _write_jsonl(
        manifest_root / "blocked" / "evaluation_pairs.jsonl",
        [
            _blocked_record(
                2,
                "identity:dog",
                "identity:ball",
                ["holding"],
                subject_delay=1.0,
                object_delay=0.4,
            ),
            _blocked_record(
                3,
                "identity:ball",
                "identity:dog",
                ["looking at"],
                subject_delay=12.0,
                object_delay=11.0,
            ),
        ],
    )
    _write_jsonl(
        manifest_root / "heldout_video" / "development_pairs.jsonl",
        [
            _record(
                "development",
                0,
                "development",
                "identity:novel-dog",
                "identity:novel-ball",
                ["holding"],
            ),
            _record(
                "development",
                1,
                "development",
                "identity:novel-ball",
                "identity:novel-dog",
                ["looking at"],
            ),
        ],
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
                {"name": "identity:bystander", "category": "ball"},
                {"name": "identity:novel-dog", "category": "dog"},
                {"name": "identity:novel-ball", "category": "ball"},
            ],
        },
    )
    _artifact(feature_root, "train", [[1, 2], [2, 1], [1, 3], [3, 1]])
    _artifact(feature_root, "development", [[1, 1], [2, 3]])


def test_blocked_protocol_reports_known_and_novel_entities(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.pvsg.pair_experiment.load_object_hierarchy",
        lambda *_args, **_kwargs: _hierarchy(),
    )
    manifest_root = tmp_path / "snapshot"
    feature_root = tmp_path / "features"
    _blocked_manifests(manifest_root, feature_root)

    result = run_pair_experiment(
        manifest_root,
        feature_root,
        tmp_path / "runs",
        PairExperimentConfig(
            run_name="blocked",
            condition="integral-p-sa",
            protocol="blocked",
            evolution="qtb",
            score_mode="softplus-bias",
            batch_size=2,
            max_steps=2,
            validation_every=1,
            validation_examples=2,
            log_every=1,
            num_workers=0,
            device="cpu",
        ),
    )

    # One checkpoint, both regimes: novel entities (VRD-E) and known entities (VRD-EX).
    assert set(result["evaluation"]) == {"development", "blocked"}
    for regime in result["evaluation"].values():
        assert set(regime) == {"p-sa", "p-samp", "p-sa-sequential", "p-sa-sequential-sample"}

    known = result["evaluation"]["blocked"]["p-sa"]
    novel = result["evaluation"]["development"]["p-sa"]
    # Known entities own index columns, so the identity readout is scoreable.
    assert known["support/subject_identity"] == 2
    assert known["support/object_identity"] == 2
    assert known["ignored/subject_identity"] == 0
    assert known["support/identity_pair_enrolled"] == 2
    assert 0.0 <= known["accuracy/identity_pair_exact"] <= 1.0
    assert "accuracy/subject_identity_macro" in known
    # Novel entities cannot be scored against a training identity bank.
    assert "accuracy/subject_identity" not in novel
    assert "support/identity_pair_enrolled" not in novel

    # Predicate metrics are partitioned by recognition and by re-identification delay.
    assert (
        known["stratum/identity_pair_correct/examples"]
        + known["stratum/identity_pair_incorrect/examples"]
        == 2
    )
    assert known["stratum/delay/0-2s/examples"] == 1
    assert known["stratum/delay/10s+/examples"] == 1
    assert known["stratum/delay/2-5s/examples"] == 0
    assert "stratum/delay/0-2s/recall/predicate_example_macro@1" in known
    assert "stratum/delay/0-2s/accuracy/identity_pair_exact" in known
    # Strata suppress the per-predicate breakdown that the full set reports.
    assert "recall/predicate@1/predicate:holding" in known
    assert "stratum/delay/0-2s/recall/predicate@1/predicate:holding" not in known
    assert known["delay_seconds_max"] == 12.0

    layout = json.loads(
        (tmp_path / "runs" / "blocked" / "config.json").read_text(encoding="utf-8")
    )["protocol_layout"]
    assert layout["name"] == "blocked"
    assert layout["selection_set"] == "development"
    assert layout["identity_enrollment"]["source"] == "blocked/train_objects.jsonl"
    assert layout["evaluation_sets"]["blocked"]["vrd_analogue"] == "VRD-EX"
    assert layout["evaluation_sets"]["development"]["vrd_analogue"] == "VRD-E"
    assert layout["evaluation_sets"]["blocked"]["known_entities"] is True

    # Enrolling per observed entity leaves columns no training pair supervised. They
    # compete in the identity softmax, so their attention mass is reported.
    enrollment = layout["identity_enrollment"]
    unsupervised = enrollment["columns"] - enrollment["columns_supervised_by_training_pairs"]
    assert unsupervised > 0
    attention = [
        row
        for row in (
            json.loads(line)
            for line in (tmp_path / "runs" / "blocked" / "scale_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        if row["kind"] == "attention" and row["group"] == "identity"
    ]
    assert attention
    for row in attention:
        assert row["unsupervised_candidate_count"] == unsupervised
        assert 0.0 <= row["unsupervised_attention_mass_mean"] <= 1.0


@pytest.mark.parametrize(
    ("condition", "evaluation_labels"),
    (
        ("priors", {"frequency", "category-pair"}),
        ("union-only", {"default"}),
        ("p-direct", {"default"}),
        ("integral-none", {"none"}),
        ("integral-p-sa", {"p-sa", "p-samp", "p-sa-sequential", "p-sa-sequential-sample"}),
    ),
)
def test_blocked_protocol_runs_every_array_condition(
    tmp_path, monkeypatch, condition, evaluation_labels
) -> None:
    """Every condition in cluster/pvsg/pair_known_entities.sbatch reports both regimes."""

    monkeypatch.setattr(
        "experiments.pvsg.pair_experiment.load_object_hierarchy",
        lambda *_args, **_kwargs: _hierarchy(),
    )
    manifest_root = tmp_path / "snapshot"
    feature_root = tmp_path / "features"
    _blocked_manifests(manifest_root, feature_root)

    result = run_pair_experiment(
        manifest_root,
        feature_root,
        tmp_path / "runs",
        PairExperimentConfig(
            run_name=condition,
            condition=condition,
            protocol="blocked",
            evolution="qtb",
            score_mode="softplus-bias",
            batch_size=2,
            max_steps=2,
            validation_every=1,
            validation_examples=2,
            log_every=1,
            num_workers=0,
            device="cpu",
        ),
    )

    assert set(result["evaluation"]) == {"development", "blocked"}
    for regime in result["evaluation"].values():
        assert set(regime) == evaluation_labels
        for metrics in regime.values():
            assert metrics["examples"] == 2
    known = result["evaluation"]["blocked"]
    # Only the Tensor Brain schedules own an identity readout to score.
    if condition in {"p-direct", "integral-none", "integral-p-sa"}:
        for metrics in known.values():
            assert metrics["support/identity_pair_enrolled"] == 2
    for metrics in known.values():
        assert metrics["stratum/delay/0-2s/examples"] == 1
        assert metrics["stratum/delay/10s+/examples"] == 1


def test_blocked_protocol_rejects_an_unenrolled_known_entity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "experiments.pvsg.pair_experiment.load_object_hierarchy",
        lambda *_args, **_kwargs: _hierarchy(),
    )
    manifest_root = tmp_path / "snapshot"
    feature_root = tmp_path / "features"
    _blocked_manifests(manifest_root, feature_root)
    # Drop one entity from the observation window while it still appears at evaluation.
    _write_jsonl(
        manifest_root / "blocked" / "train_objects.jsonl",
        [_object_record("train", 0, "train", "identity:dog", "dog")],
    )

    with pytest.raises(ValueError, match="missing from the enrollment manifest"):
        run_pair_experiment(
            manifest_root,
            feature_root,
            tmp_path / "runs",
            PairExperimentConfig(
                run_name="blocked-unenrolled",
                condition="integral-p-sa",
                protocol="blocked",
                batch_size=2,
                max_steps=1,
                validation_every=1,
                validation_examples=2,
                log_every=1,
                num_workers=0,
                device="cpu",
            ),
        )


def test_reevaluation_reports_extra_readouts_from_a_finished_checkpoint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "experiments.pvsg.pair_experiment.load_object_hierarchy",
        lambda *_args, **_kwargs: _hierarchy(),
    )
    monkeypatch.setattr(
        "experiments.pvsg.pair_evaluate.load_object_hierarchy",
        lambda *_args, **_kwargs: _hierarchy(),
    )
    manifest_root = tmp_path / "snapshot"
    feature_root = tmp_path / "features"
    _blocked_manifests(manifest_root, feature_root)
    run_pair_experiment(
        manifest_root,
        feature_root,
        tmp_path / "runs",
        PairExperimentConfig(
            run_name="blocked",
            condition="integral-p-sa",
            protocol="blocked",
            batch_size=2,
            max_steps=2,
            validation_every=1,
            validation_examples=2,
            log_every=1,
            num_workers=0,
            device="cpu",
        ),
    )

    result = reevaluate_pair_run(
        tmp_path / "runs" / "blocked",
        manifest_root,
        feature_root,
        feedback_gates=(1.0, 4.0),
        device_name="cpu",
    )

    assert result["gate_applies_to"] == "attention_and_measurement_injections"
    assert set(result["evaluation"]) == {"development", "blocked"}
    for readouts in result["evaluation"].values():
        # One gate covers both operations, so every readout is reported at each beta.
        assert set(readouts) == {
            "expected@beta=1",
            "expected@beta=4",
            "winner@beta=1",
            "winner@beta=4",
            "sequential-argmax@beta=1",
            "sequential-argmax@beta=4",
            "sequential-sample@beta=1",
            "sequential-sample@beta=4",
        }
    known = result["evaluation"]["blocked"]
    assert known["expected@beta=1"]["examples"] == 2
    # The attention-only readout responds to the gate too.
    assert (
        known["expected@beta=4"]["loss/predicate_kl"]
        != known["expected@beta=1"]["loss/predicate_kl"]
    )
    # A larger measurement gate must actually change the measurement readout.
    assert (
        known["winner@beta=4"]["loss/predicate_kl"]
        != known["winner@beta=1"]["loss/predicate_kl"]
    )
    assert (tmp_path / "runs" / "blocked" / "reevaluation.json").exists()

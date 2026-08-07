import json

import pytest
import torch

from experiments.pvsg.ambiguity_audit import (
    audit_observations,
    enroll_video,
    summarize,
)

# Two tracks sit close together in feature space and share a category, and one
# sits far away. Query row 5 is unambiguously track 1; query row 6 is much
# nearer track 3, so appearance alone gets it wrong. That is the population the
# audit exists to count.
OBJECT_FEATURES = [
    [1.0, 0.0],     # 0: track 1 observation
    [1.0, 0.1],     # 1: track 1 observation
    [0.0, 1.0],     # 2: track 2 observation
    [0.3, 1.0],     # 3: track 2 observation
    [1.0, 1.0],     # 4: track 3 observation
    [1.0, 0.05],    # 5: track 1 evaluation, easy
    [1.0, 0.95],    # 6: track 1 evaluation, lost to track 3
]
CATEGORIES = {1: "dog", 2: "cat", 3: "dog"}


def _write_artifact(root, *, video_id, features, source="vidor"):
    path = root / "videos" / source / f"{video_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {"schema_version": 2, "source": source, "video_id": video_id},
            "scene_features": torch.zeros(1, 2, dtype=torch.float16),
            "object_features": torch.tensor(features, dtype=torch.float16),
            "union_features": torch.zeros(1, 2, dtype=torch.float16),
        },
        path,
    )


def _observation(video_id, object_id, object_row, frame_index, **extra):
    return {
        "source": "vidor",
        "video_id": video_id,
        "frame_index": frame_index,
        "object_id": object_id,
        "identity": f"identity:vidor/{video_id}/{object_id}",
        "category": CATEGORIES[object_id],
        "scene_row": frame_index,
        "object_row": object_row,
        "mask_area": 100 * object_row + 10,
        **extra,
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


@pytest.fixture
def audit_root(tmp_path):
    feature_root = tmp_path / "features"
    _write_artifact(feature_root, video_id="v1", features=OBJECT_FEATURES)
    # A second video whose only track would be the global nearest neighbour of
    # v1's easy query. Retrieval must never reach it.
    _write_artifact(feature_root, video_id="v2", features=[[1.0, 0.05], [1.0, 0.05]])

    manifests = tmp_path / "manifests"
    _write_jsonl(
        manifests / "blocked" / "train_objects.jsonl",
        [
            _observation("v1", 1, 0, 0),
            _observation("v1", 1, 1, 1),
            _observation("v1", 2, 2, 0),
            _observation("v1", 2, 3, 1),
            _observation("v1", 3, 4, 0),
            _observation("v2", 1, 0, 0),
        ],
    )
    _write_jsonl(
        manifests / "blocked" / "evaluation_objects.jsonl",
        [
            _observation(
                "v1", 1, 5, 60,
                last_observation_frame=1,
                frames_since_last_observation=59,
                seconds_since_last_observation=11.8,
            ),
            _observation(
                "v1", 1, 6, 20,
                last_observation_frame=1,
                frames_since_last_observation=19,
                seconds_since_last_observation=3.8,
            ),
        ],
    )
    return manifests, feature_root


def test_enrollment_orders_tracks_and_keeps_every_view(audit_root):
    manifests, feature_root = audit_root
    records = [
        json.loads(line)
        for line in (manifests / "blocked" / "train_objects.jsonl").read_text().splitlines()
    ]
    positions = [index for index, row in enumerate(records) if row["video_id"] == "v1"]

    bank = enroll_video(records, positions, feature_root)

    assert bank.identities == (
        "identity:vidor/v1/1",
        "identity:vidor/v1/2",
        "identity:vidor/v1/3",
    )
    assert bank.categories == ("dog", "cat", "dog")
    assert [view.shape[0] for view in bank.views] == [2, 2, 1]
    torch.testing.assert_close(
        bank.centroids.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
    )


def test_retrieval_stays_inside_the_video(audit_root):
    manifests, feature_root = audit_root

    rows = audit_observations(
        manifests / "blocked" / "train_objects.jsonl",
        manifests / "blocked" / "evaluation_objects.jsonl",
        feature_root,
    )

    assert len(rows) == 2
    assert {row["num_candidates"] for row in rows} == {3}


def test_an_unambiguous_observation_is_correct_with_a_positive_margin(audit_root):
    manifests, feature_root = audit_root

    easy = audit_observations(
        manifests / "blocked" / "train_objects.jsonl",
        manifests / "blocked" / "evaluation_objects.jsonl",
        feature_root,
    )[0]

    for rule in ("centroid", "nearest_view"):
        assert easy[f"{rule}/correct"] is True
        assert easy[f"{rule}/margin"] > 0.2


def test_an_ambiguous_observation_is_wrong_with_a_negative_margin(audit_root):
    manifests, feature_root = audit_root

    hard = audit_observations(
        manifests / "blocked" / "train_objects.jsonl",
        manifests / "blocked" / "evaluation_objects.jsonl",
        feature_root,
    )[1]

    assert hard["centroid/correct"] is False
    assert hard["centroid/margin"] < 0
    # The winner is the other dog, which is what makes this the interesting kind
    # of confusion rather than a category error.
    assert hard["centroid/rival_category"] == "dog"
    assert hard["centroid/rival_same_category"] is True


def test_margin_sign_agrees_with_correctness(audit_root):
    manifests, feature_root = audit_root

    rows = audit_observations(
        manifests / "blocked" / "train_objects.jsonl",
        manifests / "blocked" / "evaluation_objects.jsonl",
        feature_root,
    )

    for row in rows:
        for rule in ("centroid", "nearest_view"):
            assert row[f"{rule}/correct"] == (row[f"{rule}/margin"] > 0)


def test_an_unenrolled_evaluation_video_is_rejected(audit_root):
    manifests, feature_root = audit_root
    enrollment = manifests / "blocked" / "train_objects.jsonl"
    kept = [
        line
        for line in enrollment.read_text().splitlines()
        if json.loads(line)["video_id"] != "v1"
    ]
    enrollment.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    with pytest.raises(ValueError, match="never enrolled"):
        audit_observations(
            enrollment, manifests / "blocked" / "evaluation_objects.jsonl", feature_root
        )


def test_the_report_counts_the_ambiguous_population(audit_root):
    manifests, feature_root = audit_root
    rows = audit_observations(
        manifests / "blocked" / "train_objects.jsonl",
        manifests / "blocked" / "evaluation_objects.jsonl",
        feature_root,
    )

    report = summarize(rows)

    assert report["observations"] == 2
    assert report["videos"] == 1
    assert report["identities"] == 1
    centroid = report["rules"]["centroid"]
    assert centroid["top1"] == 0.5
    # The zero threshold is exactly the error rate, by construction.
    assert centroid["ambiguous_at"]["0.00"] == pytest.approx(1.0 - centroid["top1"])
    assert centroid["ambiguous_at"]["0.10"] >= centroid["ambiguous_at"]["0.00"]
    # The two observations were given delays in different bins.
    assert set(report["by_delay"]) == {"2-5s", "10-20s"}
    assert set(report["by_mask_area"]) == {"q1-smallest", "q4-largest"}

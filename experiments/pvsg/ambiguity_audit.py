"""Measure how far PVSG entities are separable from appearance alone.

This decides whether the streaming and episodic experiments have a target
population before either is built, and it needs no training.

The Tensor Brain's memory mechanisms can only pay where the evidence in front
of the model fails to determine the answer. The pair experiments never varied
that: a subject crop identifies its own track, so the fed-back column carried
nothing the input did not, and the measured +0.09 pp was the expected result
rather than a surprising one. Under the ``blocked`` protocol the same question
becomes answerable directly. Every track is enrolled from the observation
window, every evaluation observation is separated from its last sighting by a
recorded delay, and re-identification from appearance alone is a nearest-track
retrieval that can simply be computed.

Two retrieval rules run over the same enrollment, because they answer different
halves of the question:

``centroid``
    the mean of a track's observation-window views. This is the accumulated
    prototype an index column is supposed to become, so its accuracy is the
    standard a learned ``a_k`` would have to beat.
``nearest_view``
    the single best-matching observation-window view. Comparing the two says
    whether averaging across occurrences buys anything at all on this data,
    which is the premise the whole index-feedback argument rests on.

What the numbers mean. ``top1`` is how often appearance alone already answers
correctly; ``1 - top1`` bounds what any memory mechanism could add. ``margin``
is the correct track's similarity minus the best competitor's, so observations
near zero are the ones where temporal context could decide. If the small-margin
population is a few percent, no memory mechanism will move a headline number on
PVSG and the streaming experiment should not be built. If it is a fifth of the
data, it is a target.

One caveat is reported rather than assumed. An ambiguous observation is not
automatically one memory can fix -- a tiny or heavily occluded mask may be
irreducibly unidentifiable. The mask-area strata exist so that an ambiguous
population concentrated in the smallest masks can be recognized as such.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

# The cached per-video loader is module-private but package-internal, and it is
# the only place that validates an artifact against its manifest row. Reaching
# for it keeps this audit reading exactly the tensors training reads, and its
# small LRU means the two lookups per video cost one load.
from experiments.pvsg.data import _video_feature_tables
from experiments.pvsg.io import read_jsonl, write_json, write_jsonl

# The pair runs' bins put 87% of blocked examples into a single "10s+" bucket
# against a mean delay of 46 s, which hides exactly the range this audit is
# about. These continue into the decade where the evidence actually lives.
DELAY_BIN_EDGES = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
DELAY_BIN_LABELS = ("0-2s", "2-5s", "5-10s", "10-20s", "20-40s", "40-80s", "80s+")
# Margins at which to report the ambiguous population. Zero is "already wrong";
# the rest ask how much of the correct population is decided only narrowly.
MARGIN_THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
RETRIEVAL_RULES = ("centroid", "nearest_view")


@dataclass(frozen=True)
class VideoBank:
    """One video's enrolled tracks, in a fixed column order."""

    identities: tuple[str, ...]
    categories: tuple[str, ...]
    centroids: Float[Tensor, "tracks feature"]
    views: tuple[Float[Tensor, "views feature"], ...]

    def __len__(self) -> int:
        return len(self.identities)


def _unit(features: Float[Tensor, "*rows feature"]) -> Float[Tensor, "*rows feature"]:
    """Project onto the unit sphere, where cosine similarity is a dot product.

    The pre-CBS RMS scaling the training pipeline applies is a positive scalar
    and cannot change a cosine, so it is deliberately not reproduced here.
    """

    return F.normalize(features.float(), p=2, dim=-1, eps=1e-12)


def _group_by_video(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, record in enumerate(records):
        grouped[(record["source"], record["video_id"])].append(position)
    return dict(grouped)


def enroll_video(
    records: Sequence[dict[str, Any]],
    positions: Sequence[int],
    feature_root: Path,
) -> VideoBank:
    """Build one video's track bank from its observation-window observations."""

    if not positions:
        raise ValueError("a video bank needs at least one observation")
    first = records[positions[0]]
    table = _video_feature_tables(feature_root, first["source"], first["video_id"])[
        "object_features"
    ]
    rows_by_identity: dict[str, list[int]] = defaultdict(list)
    category_by_identity: dict[str, str] = {}
    for position in positions:
        record = records[position]
        rows_by_identity[record["identity"]].append(record["object_row"])
        category_by_identity[record["identity"]] = record["category"]

    identities = tuple(sorted(rows_by_identity))
    views = tuple(_unit(table[rows_by_identity[identity]]) for identity in identities)
    # The centroid of unit views, renormalized so both rules are read on the
    # same scale. A track whose views cancel exactly would leave a zero vector;
    # normalize's epsilon keeps that finite instead of producing a NaN.
    centroids = _unit(torch.stack([view.mean(dim=0) for view in views]))
    return VideoBank(
        identities=identities,
        categories=tuple(category_by_identity[identity] for identity in identities),
        centroids=centroids,
        views=views,
    )


def score_video(
    bank: VideoBank,
    queries: Float[Tensor, "queries feature"],
    *,
    chunk_size: int = 512,
) -> dict[str, Float[Tensor, "queries tracks"]]:
    """Return per-rule similarity of every query against every enrolled track.

    Queries are chunked because the ``nearest_view`` rule compares against every
    enrolled view, and the longest PVSG videos hold tens of thousands of them.
    """

    centroid_scores = queries @ bank.centroids.T
    nearest_scores = queries.new_empty(queries.shape[0], len(bank))
    for start in range(0, queries.shape[0], chunk_size):
        chunk = queries[start : start + chunk_size]
        for track, view in enumerate(bank.views):
            nearest_scores[start : start + chunk.shape[0], track] = (
                (chunk @ view.T).max(dim=-1).values
            )
    return {"centroid": centroid_scores, "nearest_view": nearest_scores}


def _observation_rows(
    bank: VideoBank,
    records: Sequence[dict[str, Any]],
    positions: Sequence[int],
    scores: dict[str, Float[Tensor, "queries tracks"]],
) -> list[dict[str, Any]]:
    """Turn one video's similarities into per-observation audit rows."""

    index_of = {identity: column for column, identity in enumerate(bank.identities)}
    rows = []
    for query, position in enumerate(positions):
        record = records[position]
        truth = index_of[record["identity"]]
        row: dict[str, Any] = {
            "source": record["source"],
            "video_id": record["video_id"],
            "frame_index": record["frame_index"],
            "identity": record["identity"],
            "category": record["category"],
            "mask_area": record["mask_area"],
            "seconds_since_last_observation": record["seconds_since_last_observation"],
            "num_candidates": len(bank),
        }
        for rule in RETRIEVAL_RULES:
            similarity = scores[rule][query]
            competitors = similarity.clone()
            competitors[truth] = float("-inf")
            rival = int(competitors.argmax())
            row[f"{rule}/correct"] = bool(int(similarity.argmax()) == truth)
            row[f"{rule}/margin"] = float(similarity[truth] - competitors[rival])
            row[f"{rule}/rival_category"] = bank.categories[rival]
            row[f"{rule}/rival_same_category"] = bank.categories[rival] == record["category"]
        rows.append(row)
    return rows


def audit_observations(
    enrollment_manifest: Path,
    evaluation_manifest: Path,
    feature_root: Path,
    *,
    chunk_size: int = 512,
) -> list[dict[str, Any]]:
    """Score every evaluation observation against its own video's track bank.

    Retrieval is restricted to the tracks of the same video, which is both the
    honest setting -- PVSG identities are video-scoped -- and the hard one,
    since same-video tracks are the confusable ones.
    """

    enrollment = read_jsonl(enrollment_manifest)
    evaluation = read_jsonl(evaluation_manifest)
    enrollment_videos = _group_by_video(enrollment)
    rows: list[dict[str, Any]] = []
    for video, positions in sorted(_group_by_video(evaluation).items()):
        if video not in enrollment_videos:
            raise ValueError(f"evaluation video was never enrolled: {video}")
        bank = enroll_video(enrollment, enrollment_videos[video], feature_root)
        table = _video_feature_tables(feature_root, *video)["object_features"]
        queries = _unit(table[[evaluation[position]["object_row"] for position in positions]])
        rows.extend(
            _observation_rows(bank, evaluation, positions, score_video(
                bank, queries, chunk_size=chunk_size
            ))
        )
    return rows


def _quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    probabilities = torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95], dtype=torch.float64)
    quantiles = torch.quantile(tensor, probabilities).tolist()
    return {
        "mean": float(tensor.mean()),
        **{
            f"p{int(probability * 100):02d}": value
            for probability, value in zip(probabilities.tolist(), quantiles, strict=True)
        },
    }


def _rule_summary(rows: Sequence[dict[str, Any]], rule: str) -> dict[str, Any]:
    if not rows:
        return {"observations": 0}
    correct = [row[f"{rule}/correct"] for row in rows]
    margins = [row[f"{rule}/margin"] for row in rows]
    return {
        "observations": len(rows),
        "top1": sum(correct) / len(correct),
        "margin": _quantiles(margins),
        # The share of observations decided by less than each margin. The 0.0
        # entry is the error rate; the rest are the population a memory
        # mechanism could plausibly reach.
        "ambiguous_at": {
            f"{threshold:.2f}": sum(margin < threshold for margin in margins) / len(margins)
            for threshold in MARGIN_THRESHOLDS
        },
        "rival_same_category": sum(row[f"{rule}/rival_same_category"] for row in rows)
        / len(rows),
    }


def _delay_label(seconds: float) -> str:
    position = int(torch.bucketize(
        torch.tensor(seconds, dtype=torch.float64),
        torch.tensor(DELAY_BIN_EDGES, dtype=torch.float64),
        right=True,
    ))
    return DELAY_BIN_LABELS[position]


def _strata(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group observations by delay and by mask-area quartile."""

    by_delay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_delay[_delay_label(row["seconds_since_last_observation"])].append(row)

    areas = torch.tensor([float(row["mask_area"]) for row in rows], dtype=torch.float64)
    edges = torch.quantile(areas, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    labels = ("q1-smallest", "q2", "q3", "q4-largest")
    by_area: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, area in zip(rows, areas.tolist(), strict=True):
        position = int(torch.bucketize(
            torch.tensor(area, dtype=torch.float64), edges, right=True
        ))
        by_area[labels[position]].append(row)
    return {"by_delay": dict(by_delay), "by_mask_area": dict(by_area)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-observation rows to the report that decides the question."""

    if not rows:
        raise ValueError("the audit produced no observations")
    strata = _strata(rows)
    candidates = torch.tensor(
        [float(row["num_candidates"]) for row in rows], dtype=torch.float64
    )
    return {
        "observations": len(rows),
        "videos": len({(row["source"], row["video_id"]) for row in rows}),
        "identities": len({row["identity"] for row in rows}),
        "candidates_per_observation": _quantiles(candidates.tolist()),
        "rules": {rule: _rule_summary(rows, rule) for rule in RETRIEVAL_RULES},
        "by_delay": {
            label: {
                rule: _rule_summary(strata["by_delay"][label], rule)
                for rule in RETRIEVAL_RULES
            }
            for label in DELAY_BIN_LABELS
            if label in strata["by_delay"]
        },
        "by_mask_area": {
            label: {rule: _rule_summary(group, rule) for rule in RETRIEVAL_RULES}
            for label, group in sorted(strata["by_mask_area"].items())
        },
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--enrollment-manifest", default="blocked/train_objects.jsonl",
        help="observation-window observations that build each track bank",
    )
    parser.add_argument(
        "--evaluation-manifest", default="blocked/evaluation_objects.jsonl",
        help="post-embargo observations to re-identify",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--write-observations", action="store_true",
        help="also write the per-observation rows, for later stratification",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    rows = audit_observations(
        arguments.manifest_root / arguments.enrollment_manifest,
        arguments.manifest_root / arguments.evaluation_manifest,
        arguments.feature_root,
        chunk_size=arguments.chunk_size,
    )
    report = {
        "enrollment_manifest": arguments.enrollment_manifest,
        "evaluation_manifest": arguments.evaluation_manifest,
        "feature_root": str(arguments.feature_root),
        **summarize(rows),
    }
    write_json(arguments.output_root / "ambiguity_report.json", report, sort_keys=True)
    if arguments.write_observations:
        write_jsonl(arguments.output_root / "observations.jsonl", rows)

    centroid = report["rules"]["centroid"]
    print(f"observations {report['observations']} across {report['videos']} videos")
    print(f"centroid top1 {centroid['top1']:.4f}")
    for threshold, share in centroid["ambiguous_at"].items():
        print(f"  margin < {threshold}: {share:.4f}")
    print(f"nearest_view top1 {report['rules']['nearest_view']['top1']:.4f}")


if __name__ == "__main__":
    main()

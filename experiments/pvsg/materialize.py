"""Materialize readable PVSG records and the three initial protocol manifests."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from experiments.pvsg.audit import validate_feature_artifact
from experiments.pvsg.extract import FEATURE_SCHEMA_VERSION, load_feature_artifact
from experiments.pvsg.prepare import PVSG_HUB_REVISION, PVSG_JSON_SHA256
from experiments.pvsg.protocols import blocked_boundary, fewshot_support_and_queries
from experiments.pvsg.records import (
    EXCLUSIONS_PATH,
    active_predicates,
    identity_name,
    load_exclusions,
    relation_targets,
)
from experiments.pvsg.snapshot_io import read_json, read_jsonl, sha256_file

MANIFEST_SCHEMA_VERSION = 1
SUPPORT_COUNT = 5
FEWSHOT_EMBARGO_FRAMES = 25

JSONL_PATHS = (
    "canonical/positive_pairs.jsonl",
    "heldout_video/train_objects.jsonl",
    "heldout_video/evaluation_objects.jsonl",
    "heldout_video/train_pairs.jsonl",
    "heldout_video/evaluation_pairs.jsonl",
    "blocked/boundaries.jsonl",
    "blocked/train_objects.jsonl",
    "blocked/evaluation_objects.jsonl",
    "blocked/train_pairs.jsonl",
    "blocked/evaluation_pairs.jsonl",
    "fewshot/enrollment.jsonl",
    "fewshot/support_objects.jsonl",
    "fewshot/query_objects.jsonl",
    "fewshot/query_pairs.jsonl",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_row(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, separators=(",", ":"), sort_keys=False) + "\n")


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _official_splits(annotation: dict[str, Any]) -> dict[str, str]:
    result = {}
    for _source, splits in annotation["split"].items():
        for split in ("train", "val"):
            for video_id in splits[split]:
                if video_id in result:
                    raise ValueError(f"video occurs more than once in official splits: {video_id}")
                result[video_id] = split
    return result


def _annotation_sources(annotation: dict[str, Any]) -> dict[str, str]:
    result = {}
    for source, splits in annotation["split"].items():
        for split in ("train", "val"):
            for video_id in splits[split]:
                result[video_id] = source
    return result


def _ordered_targets(predicates: tuple[str, ...], rank: dict[str, int]) -> list[str]:
    return sorted(predicates, key=rank.__getitem__)


def _feature_addresses(
    artifact: dict[str, Any],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int, int], int]]:
    objects = {
        (frame, object_id): row
        for row, (frame, object_id) in enumerate(
            zip(
                artifact["object_frame_index"].tolist(),
                artifact["object_ids"].tolist(),
                strict=True,
            )
        )
    }
    pairs = {
        (frame, pair[0], pair[1]): row
        for row, (frame, pair) in enumerate(
            zip(
                artifact["pair_frame_index"].tolist(),
                artifact["pair_ids"].tolist(),
                strict=True,
            )
        )
    }
    return objects, pairs


def _object_frames(artifact: dict[str, Any]) -> dict[int, list[int]]:
    frames: dict[int, list[int]] = defaultdict(list)
    for frame, object_id in zip(
        artifact["object_frame_index"].tolist(), artifact["object_ids"].tolist(), strict=True
    ):
        frames[object_id].append(frame)
    return dict(frames)


def _identity_records(
    annotation: dict[str, Any],
    *,
    source_by_video: dict[str, str],
    excluded_video_ids: set[str],
) -> list[dict[str, Any]]:
    records = []
    for video in annotation["data"]:
        video_id = video["video_id"]
        if video_id in excluded_video_ids:
            continue
        source = source_by_video[video_id]
        for obj in video["objects"]:
            records.append(
                {
                    "name": identity_name(source, video_id, obj["object_id"]),
                    "source": source,
                    "video_id": video_id,
                    "object_id": obj["object_id"],
                    "category": obj["category"],
                    "is_thing": obj["is_thing"],
                }
            )
    records.sort(key=lambda row: (row["source"], row["video_id"], row["object_id"]))
    return records


def _pair_record(
    *,
    source: str,
    video_id: str,
    official_split: str,
    frame_index: int,
    subject_id: int,
    object_id: int,
    predicates: tuple[str, ...],
    predicate_rank: dict[str, int],
    categories: dict[int, dict[str, Any]],
    object_rows: dict[tuple[int, int], int],
    pair_rows: dict[tuple[int, int, int], int],
) -> dict[str, Any]:
    canonical_pair = tuple(sorted((subject_id, object_id)))
    subject_row = object_rows.get((frame_index, subject_id))
    object_row = object_rows.get((frame_index, object_id))
    union_row = pair_rows.get((frame_index, *canonical_pair))
    return {
        "source": source,
        "video_id": video_id,
        "official_split": official_split,
        "frame_index": frame_index,
        "subject_id": subject_id,
        "object_id": object_id,
        "subject_identity": identity_name(source, video_id, subject_id),
        "object_identity": identity_name(source, video_id, object_id),
        "subject_category": categories[subject_id]["category"],
        "object_category": categories[object_id]["category"],
        "predicates": _ordered_targets(predicates, predicate_rank),
        "scene_row": frame_index,
        "subject_row": subject_row,
        "object_row": object_row,
        "union_row": union_row,
        "has_complete_evidence": all(
            row is not None for row in (subject_row, object_row, union_row)
        ),
    }


def materialize_manifests(
    *,
    dataset_root: Path,
    extraction_manifest: Path,
    feature_root: Path,
    output_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate the retained snapshot and atomically materialize all manifests."""

    if output_root.exists():
        raise FileExistsError(f"refusing to replace an existing manifest snapshot: {output_root}")
    manifest = read_jsonl(extraction_manifest)
    if len(manifest) != 400:
        raise ValueError(f"expected 400 extraction rows, found {len(manifest)}")
    manifest_by_video = {row["video_id"]: row for row in manifest}
    if len(manifest_by_video) != len(manifest):
        raise ValueError("extraction manifest video IDs are not unique")

    annotation_path = dataset_root / "pvsg.json"
    if sha256_file(annotation_path) != PVSG_JSON_SHA256:
        raise ValueError("dataset annotation is not the pinned pvsg.json")
    annotation = read_json(annotation_path)
    videos = {video["video_id"]: video for video in annotation["data"]}
    if set(videos) != set(manifest_by_video):
        raise ValueError("annotation and extraction manifest video IDs differ")
    official_split = _official_splits(annotation)
    source_by_video = _annotation_sources(annotation)
    for video_id, video in videos.items():
        row = manifest_by_video[video_id]
        meta = video["meta"]
        expected = {
            "source": source_by_video[video_id],
            "num_frames": meta["num_frames"],
            "height": meta["height"],
            "width": meta["width"],
            "fps": meta["fps"],
            "num_objects": len(video["objects"]),
        }
        actual = {name: row.get(name) for name in expected}
        if actual != expected:
            raise ValueError(
                f"extraction manifest disagrees with pvsg.json for {video_id}: "
                f"expected {expected}, found {actual}"
            )

    exclusions = load_exclusions()
    for video_id, exclusion in exclusions.items():
        row = manifest_by_video.get(video_id)
        index = exclusion["extraction_array_index"]
        if row is None or row["source"] != exclusion["source"] or manifest[index] != row:
            raise ValueError(f"exclusion does not match extraction row {index}: {video_id}")
    excluded_video_ids = set(exclusions)
    predicates, additional_predicates = active_predicates(annotation, excluded_video_ids)
    predicate_rank = {name: index for index, name in enumerate(predicates)}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    counts: Counter[str] = Counter(
        {
            **{path: 0 for path in JSONL_PATHS},
            "retained_videos": 0,
            "excluded_videos": 0,
            "complete_pair_records": 0,
            "incomplete_pair_records": 0,
            "multi_label_complete_records": 0,
        }
    )
    train_predicate_support: Counter[str] = Counter()
    validation_predicate_support: Counter[str] = Counter()
    span_issues = []
    provenance_groups: Counter[str] = Counter()
    try:
        with contextlib.ExitStack() as stack:
            writers = {}
            for relative in JSONL_PATHS:
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                writers[relative] = stack.enter_context(path.open("w", encoding="utf-8"))

            for manifest_index, manifest_row in enumerate(manifest):
                source = manifest_row["source"]
                video_id = manifest_row["video_id"]
                if video_id in excluded_video_ids:
                    counts["excluded_videos"] += 1
                    continue
                artifact_path = feature_root / "videos" / source / f"{video_id}.pt"
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"non-excluded feature artifact is missing: {artifact_path}"
                    )
                artifact = load_feature_artifact(artifact_path)
                validated = validate_feature_artifact(artifact, manifest_row)
                provenance_groups[
                    json.dumps(validated["provenance"], sort_keys=True)
                ] += 1
                counts["retained_videos"] += 1

                video = videos[video_id]
                split = official_split[video_id]
                categories = {obj["object_id"]: obj for obj in video["objects"]}
                object_rows, pair_rows = _feature_addresses(artifact)
                frames_by_identity = _object_frames(artifact)
                boundary = blocked_boundary(manifest_row["num_frames"])
                blocked_last_observation = {
                    object_id: max(frame for frame in frames if frame < boundary.observation_end)
                    for object_id, frames in frames_by_identity.items()
                    if any(frame < boundary.observation_end for frame in frames)
                }
                fewshot = {
                    object_id: fewshot_support_and_queries(
                        frames,
                        support_count=SUPPORT_COUNT,
                        embargo_frames=FEWSHOT_EMBARGO_FRAMES,
                    )
                    for object_id, frames in frames_by_identity.items()
                }
                fewshot = {
                    object_id: value
                    for object_id, value in fewshot.items()
                    if value[0] and value[1]
                }
                support_rank = {
                    (object_id, frame): rank
                    for object_id, (support, _queries) in fewshot.items()
                    for rank, frame in enumerate(support, start=1)
                }
                query_frames = {
                    object_id: set(queries)
                    for object_id, (_support, queries) in fewshot.items()
                }

                if split == "train":
                    _write_row(
                        writers["blocked/boundaries.jsonl"],
                        {
                            "source": source,
                            "video_id": video_id,
                            "num_frames": manifest_row["num_frames"],
                            "observation": [0, boundary.observation_end],
                            "embargo": [boundary.observation_end, boundary.evaluation_start],
                            "evaluation": [boundary.evaluation_start, manifest_row["num_frames"]],
                        },
                    )
                    counts["blocked/boundaries.jsonl"] += 1
                else:
                    for object_id, (support, queries) in fewshot.items():
                        _write_row(
                            writers["fewshot/enrollment.jsonl"],
                            {
                                "source": source,
                                "video_id": video_id,
                                "object_id": object_id,
                                "identity": identity_name(source, video_id, object_id),
                                "category": categories[object_id]["category"],
                                "support_frames": support,
                                "last_support_frame": support[-1],
                                "first_query_frame": queries[0],
                                "num_query_observations": len(queries),
                            },
                        )
                        counts["fewshot/enrollment.jsonl"] += 1

                for object_row, (frame_index, object_id, mask_area) in enumerate(
                    zip(
                        artifact["object_frame_index"].tolist(),
                        artifact["object_ids"].tolist(),
                        artifact["object_mask_areas"].tolist(),
                        strict=True,
                    )
                ):
                    record = {
                        "source": source,
                        "video_id": video_id,
                        "official_split": split,
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "identity": identity_name(source, video_id, object_id),
                        "category": categories[object_id]["category"],
                        "is_thing": categories[object_id]["is_thing"],
                        "scene_row": frame_index,
                        "object_row": object_row,
                        "mask_area": mask_area,
                    }
                    heldout_role = "train" if split == "train" else "evaluation"
                    heldout_path = f"heldout_video/{heldout_role}_objects.jsonl"
                    _write_row(writers[heldout_path], record)
                    counts[heldout_path] += 1
                    if split == "train":
                        role = boundary.role(frame_index)
                        if role == "observation":
                            path = "blocked/train_objects.jsonl"
                            _write_row(writers[path], record)
                            counts[path] += 1
                        elif role == "evaluation" and object_id in blocked_last_observation:
                            path = "blocked/evaluation_objects.jsonl"
                            last_frame = blocked_last_observation[object_id]
                            _write_row(
                                writers[path],
                                {
                                    **record,
                                    "last_observation_frame": last_frame,
                                    "frames_since_last_observation": frame_index - last_frame,
                                    "seconds_since_last_observation": (
                                        frame_index - last_frame
                                    ) / manifest_row["fps"],
                                },
                            )
                            counts[path] += 1
                    elif (object_id, frame_index) in support_rank:
                        path = "fewshot/support_objects.jsonl"
                        _write_row(
                            writers[path],
                            {**record, "support_rank": support_rank[(object_id, frame_index)]},
                        )
                        counts[path] += 1
                    elif object_id in query_frames and frame_index in query_frames[object_id]:
                        path = "fewshot/query_objects.jsonl"
                        last_support = fewshot[object_id][0][-1]
                        _write_row(
                            writers[path],
                            {
                                **record,
                                "last_support_frame": last_support,
                                "frames_since_support": frame_index - last_support,
                                "seconds_since_support": (
                                    frame_index - last_support
                                ) / manifest_row["fps"],
                            },
                        )
                        counts[path] += 1

                targets, video_span_issues = relation_targets(
                    video, predicate_vocabulary=set(predicates)
                )
                span_issues.extend(asdict(issue) for issue in video_span_issues)
                for (frame_index, subject_id, object_id), target_predicates in sorted(
                    targets.items()
                ):
                    record = _pair_record(
                        source=source,
                        video_id=video_id,
                        official_split=split,
                        frame_index=frame_index,
                        subject_id=subject_id,
                        object_id=object_id,
                        predicates=target_predicates,
                        predicate_rank=predicate_rank,
                        categories=categories,
                        object_rows=object_rows,
                        pair_rows=pair_rows,
                    )
                    canonical_path = "canonical/positive_pairs.jsonl"
                    _write_row(writers[canonical_path], record)
                    counts[canonical_path] += 1
                    completeness_key = (
                        "complete_pair_records"
                        if record["has_complete_evidence"]
                        else "incomplete_pair_records"
                    )
                    counts[completeness_key] += 1
                    if not record["has_complete_evidence"]:
                        continue
                    heldout_role = "train" if split == "train" else "evaluation"
                    heldout_path = f"heldout_video/{heldout_role}_pairs.jsonl"
                    _write_row(writers[heldout_path], record)
                    counts[heldout_path] += 1
                    support_counter = train_predicate_support
                    if split != "train":
                        support_counter = validation_predicate_support
                    support_counter.update(record["predicates"])
                    counts["multi_label_complete_records"] += len(record["predicates"]) > 1
                    if split == "train":
                        role = boundary.role(frame_index)
                        if role == "observation":
                            path = "blocked/train_pairs.jsonl"
                            _write_row(writers[path], record)
                            counts[path] += 1
                        elif role == "evaluation" and {
                            subject_id,
                            object_id,
                        } <= blocked_last_observation.keys():
                            path = "blocked/evaluation_pairs.jsonl"
                            subject_last = blocked_last_observation[subject_id]
                            object_last = blocked_last_observation[object_id]
                            _write_row(
                                writers[path],
                                {
                                    **record,
                                    "subject_last_observation_frame": subject_last,
                                    "object_last_observation_frame": object_last,
                                    "frames_since_subject_observation": frame_index
                                    - subject_last,
                                    "frames_since_object_observation": frame_index - object_last,
                                    "seconds_since_subject_observation": (
                                        frame_index - subject_last
                                    )
                                    / manifest_row["fps"],
                                    "seconds_since_object_observation": (
                                        frame_index - object_last
                                    )
                                    / manifest_row["fps"],
                                },
                            )
                            counts[path] += 1
                    elif subject_id in fewshot and object_id in fewshot:
                        later_support = max(
                            fewshot[subject_id][0][-1], fewshot[object_id][0][-1]
                        )
                        if frame_index >= later_support + FEWSHOT_EMBARGO_FRAMES:
                            path = "fewshot/query_pairs.jsonl"
                            _write_row(
                                writers[path],
                                {
                                    **record,
                                    "last_support_frame": later_support,
                                    "frames_since_support": frame_index - later_support,
                                    "seconds_since_support": (
                                        frame_index - later_support
                                    ) / manifest_row["fps"],
                                },
                            )
                            counts[path] += 1

                if manifest_index % 25 == 0 or manifest_index == len(manifest) - 1:
                    print(
                        f"materialized through video {manifest_index + 1}/{len(manifest)}",
                        flush=True,
                    )

        identities = _identity_records(
            annotation,
            source_by_video=source_by_video,
            excluded_video_ids=excluded_video_ids,
        )
        train_supported = [name for name in predicates if train_predicate_support[name] > 0]
        train_unseen = [name for name in predicates if train_predicate_support[name] == 0]
        source_observed = sorted(
            {
                predicate
                for video in annotation["data"]
                for _, _, predicate, _spans in video["relations"]
            }
        )
        _write_json(
            staging / "ontology.json",
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "predicates": predicates,
                "declared_predicates": annotation["relations"],
                "additional_retained_predicates": additional_predicates,
                "source_observed_but_not_retained": sorted(set(source_observed) - set(predicates)),
                "train_supported_predicates": train_supported,
                "train_unseen_predicates": train_unseen,
                "predicate_support": {
                    name: {
                        "train_complete_pair_frames": train_predicate_support[name],
                        "validation_complete_pair_frames": validation_predicate_support[name],
                    }
                    for name in predicates
                },
                "object_categories": annotation["objects"],
                "identities": identities,
            },
        )
        _write_json(staging / "span_issues.json", span_issues)
        _write_json(
            staging / "fewshot/base_training.json",
            {
                "objects": "../heldout_video/train_objects.jsonl",
                "pairs": "../heldout_video/train_pairs.jsonl",
                "note": (
                    "The base model uses official-training videos; support contains identity "
                    "labels only."
                ),
            },
        )
        file_manifest = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                relative = path.relative_to(staging).as_posix()
                file_manifest[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": counts.get(relative),
                }
        provenance = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_revision": _git_revision(repository_root),
            "pvsg_hub_revision": PVSG_HUB_REVISION,
            "pvsg_json_sha256": PVSG_JSON_SHA256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_root": str(feature_root),
            "exclusions_path": str(EXCLUSIONS_PATH.relative_to(repository_root)),
            "exclusions_sha256": sha256_file(EXCLUSIONS_PATH),
            "excluded_videos": list(exclusions.values()),
            "relation_spans": "inclusive endpoints intersected with [0, num_frames - 1]",
            "complete_pair_evidence": "scene, subject, object, and union rows all present",
            "feature_normalization_for_initial_experiment": {
                "name": "per-vector L2",
                "compute_dtype": "float32",
                "epsilon": 1e-12,
                "applied_when": "experiment loading, never cache materialization",
            },
            "protocols": {
                "heldout_video": "official train versus official validation videos",
                "blocked": {
                    "observation_fraction": 0.45,
                    "embargo_fraction": 0.10,
                    "evaluation_fraction": 0.45,
                },
                "fewshot": {
                    "support_observations_per_identity": SUPPORT_COUNT,
                    "embargo_frames": FEWSHOT_EMBARGO_FRAMES,
                    "embargo_seconds": 5.0,
                    "support_supervision": "identity only",
                },
            },
            "counts": dict(sorted(counts.items())),
            "feature_provenance_groups": [
                {"videos": count, "metadata": json.loads(value)}
                for value, count in sorted(provenance_groups.items())
            ],
            "files": file_manifest,
        }
        _write_json(staging / "provenance.json", provenance)
        os.replace(staging, output_root)
        return provenance
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    provenance = materialize_manifests(
        dataset_root=arguments.dataset_root.resolve(),
        extraction_manifest=arguments.extraction_manifest.resolve(),
        feature_root=arguments.feature_root.resolve(),
        output_root=arguments.output_root.resolve(),
        repository_root=repository_root,
    )
    counts = provenance["counts"]
    print(f"wrote {arguments.output_root}")
    print(
        f"retained {counts['retained_videos']} videos; "
        f"{counts['complete_pair_records']} complete and "
        f"{counts['incomplete_pair_records']} incomplete positive-pair records"
    )


if __name__ == "__main__":
    main()

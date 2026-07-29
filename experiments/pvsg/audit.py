"""Audit a complete PVSG DINO snapshot and render relation-boundary examples."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import torch

from experiments.pvsg.extract import (
    DINO_MODEL_ID,
    DINO_MODEL_REVISION,
    FEATURE_SCHEMA_VERSION,
    load_feature_artifact,
)
from experiments.pvsg.prepare import PVSG_JSON_SHA256
from experiments.pvsg.records import active_predicates, inclusive_clipped_frames, load_exclusions

SpanConvention = Literal["half_open", "inclusive", "inclusive_clipped"]
FEATURE_TABLES = ("scene_features", "object_features", "union_features")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_annotation(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PVSG_JSON_SHA256:
        raise ValueError(f"annotation is not the pinned pvsg.json: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tensor_contract(
    artifact: dict[str, Any], feature_dim: int
) -> dict[str, tuple[int, torch.dtype]]:
    return {
        "scene_frame_index": (1, torch.long),
        "scene_features": (2, torch.float16),
        "object_frame_index": (1, torch.long),
        "object_ids": (1, torch.long),
        "object_mask_areas": (1, torch.long),
        "object_features": (2, torch.float16),
        "pair_frame_index": (1, torch.long),
        "pair_ids": (2, torch.long),
        "union_boxes_xyxy": (2, torch.long),
        "union_features": (2, torch.float16),
    }


def validate_feature_artifact(
    artifact: dict[str, Any], manifest_row: dict[str, Any]
) -> dict[str, Any]:
    """Validate one artifact and return compact rows used by the relation audit."""

    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata is missing")
    expected_metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source": manifest_row["source"],
        "video_id": manifest_row["video_id"],
        "num_frames": manifest_row["num_frames"],
        "fps": manifest_row["fps"],
        "original_size_hw": (manifest_row["height"], manifest_row["width"]),
        "dino_model_id": DINO_MODEL_ID,
        "dino_model_revision": DINO_MODEL_REVISION,
        "pvsg_json_sha256": PVSG_JSON_SHA256,
        "feature_storage_dtype": "float16",
        "inference_autocast_dtype": "float16",
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if key == "original_size_hw":
            actual = tuple(actual or ())
        if actual != expected:
            raise ValueError(f"metadata {key!r}: expected {expected!r}, found {actual!r}")

    feature_dim = metadata.get("feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("metadata feature_dim must be a positive integer")
    required = set(_tensor_contract(artifact, feature_dim)) | {"metadata"}
    if set(artifact) != required:
        raise ValueError(f"artifact fields differ: {sorted(set(artifact) ^ required)}")
    for name, (ndim, dtype) in _tensor_contract(artifact, feature_dim).items():
        value = artifact[name]
        if not isinstance(value, torch.Tensor) or value.ndim != ndim or value.dtype != dtype:
            raise ValueError(f"{name} has the wrong tensor rank or dtype")
    if artifact["scene_features"].shape[1] != feature_dim:
        raise ValueError("scene feature dimension disagrees with metadata")
    if artifact["object_features"].shape[1] != feature_dim:
        raise ValueError("object feature dimension disagrees with metadata")
    if artifact["union_features"].shape[1] != feature_dim:
        raise ValueError("union feature dimension disagrees with metadata")
    if artifact["pair_ids"].shape[1] != 2 or artifact["union_boxes_xyxy"].shape[1] != 4:
        raise ValueError("pair IDs or union boxes have the wrong width")

    num_frames = manifest_row["num_frames"]
    scene_frames = artifact["scene_frame_index"]
    if not torch.equal(scene_frames, torch.arange(num_frames)):
        raise ValueError("scene rows are not exactly one ordered row per frame")
    if len(scene_frames) != len(artifact["scene_features"]):
        raise ValueError("scene frame and feature row counts differ")

    object_rows = len(artifact["object_ids"])
    if not (
        len(artifact["object_frame_index"])
        == len(artifact["object_mask_areas"])
        == len(artifact["object_features"])
        == object_rows
    ):
        raise ValueError("object table row counts differ")
    pair_rows = len(artifact["pair_ids"])
    if not (
        len(artifact["pair_frame_index"])
        == len(artifact["union_boxes_xyxy"])
        == len(artifact["union_features"])
        == pair_rows
    ):
        raise ValueError("pair table row counts differ")

    object_frames = artifact["object_frame_index"]
    object_ids = artifact["object_ids"]
    if object_rows:
        if int(object_frames.min()) < 0 or int(object_frames.max()) >= num_frames:
            raise ValueError("object frame index is outside the video")
        if int(object_ids.min()) < 1 or int(object_ids.max()) > manifest_row["num_objects"]:
            raise ValueError("object ID is outside the annotation vocabulary")
        if bool((artifact["object_mask_areas"] <= 0).any()):
            raise ValueError("object mask areas must be positive")
    object_keys = list(zip(object_frames.tolist(), object_ids.tolist(), strict=True))
    if object_keys != sorted(set(object_keys)):
        raise ValueError("object rows are not unique and lexicographically ordered")

    pair_frames = artifact["pair_frame_index"]
    pair_ids = artifact["pair_ids"]
    if pair_rows:
        if int(pair_frames.min()) < 0 or int(pair_frames.max()) >= num_frames:
            raise ValueError("pair frame index is outside the video")
        if bool((pair_ids[:, 0] >= pair_ids[:, 1]).any()):
            raise ValueError("pair IDs are not canonical ascending pairs")
        boxes = artifact["union_boxes_xyxy"]
        height, width = manifest_row["height"], manifest_row["width"]
        if bool(
            (boxes[:, 0] < 0).any()
            or (boxes[:, 1] < 0).any()
            or (boxes[:, 2] > width).any()
            or (boxes[:, 3] > height).any()
            or (boxes[:, 2] <= boxes[:, 0]).any()
            or (boxes[:, 3] <= boxes[:, 1]).any()
        ):
            raise ValueError("union boxes are invalid or outside the original frame")
    pair_keys = [
        (frame, pair[0], pair[1])
        for frame, pair in zip(pair_frames.tolist(), pair_ids.tolist(), strict=True)
    ]
    if pair_keys != sorted(set(pair_keys)):
        raise ValueError("pair rows are not unique and lexicographically ordered")

    visible_by_frame: dict[int, list[int]] = defaultdict(list)
    for frame, object_id in object_keys:
        visible_by_frame[frame].append(object_id)
    expected_pairs = {
        (frame, first, second)
        for frame, visible_ids in visible_by_frame.items()
        for first, second in combinations(visible_ids, 2)
    }
    if set(pair_keys) != expected_pairs:
        missing = len(expected_pairs - set(pair_keys))
        extra = len(set(pair_keys) - expected_pairs)
        raise ValueError(f"visible-pair table is incomplete: {missing} missing, {extra} extra")
    for name in FEATURE_TABLES:
        if not bool(torch.isfinite(artifact[name]).all()):
            raise ValueError(f"{name} contains NaN or infinity")

    box_by_pair = {
        key: box
        for key, box in zip(pair_keys, artifact["union_boxes_xyxy"].tolist(), strict=True)
    }
    return {
        "object_keys": set(object_keys),
        "pair_keys": set(pair_keys),
        "box_by_pair": box_by_pair,
        "counts": {
            "frames": num_frames,
            "object_observations": object_rows,
            "pair_observations": pair_rows,
        },
        "feature_norms": {name: _norm_summary(artifact[name]) for name in FEATURE_TABLES},
        "provenance": {
            key: metadata.get(key)
            for key in (
                "torch_version",
                "torch_cuda_version",
                "cuda_device_name",
                "cuda_device_capability",
                "processed_size_hw",
            )
        },
    }


def _norm_summary(features: torch.Tensor) -> dict[str, float | int | None]:
    if not len(features):
        return {"count": 0, "sum": 0.0, "sum_squares": 0.0, "min": None, "max": None}
    norms = torch.linalg.vector_norm(features.float(), dim=1)
    return {
        "count": len(norms),
        "sum": float(norms.sum()),
        "sum_squares": float(norms.square().sum()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }


def _merge_norms(total: dict[str, Any], current: dict[str, Any]) -> None:
    total["count"] += current["count"]
    total["sum"] += current["sum"]
    total["sum_squares"] += current["sum_squares"]
    values = [value for value in (total["min"], current["min"]) if value is not None]
    total["min"] = min(values) if values else None
    values = [value for value in (total["max"], current["max"]) if value is not None]
    total["max"] = max(values) if values else None


def _finish_norms(summary: dict[str, Any]) -> dict[str, float | int | None]:
    count = summary["count"]
    if not count:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = summary["sum"] / count
    variance = max(0.0, summary["sum_squares"] / count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": summary["min"],
        "max": summary["max"],
    }


def _valid_frames(
    start: int, end: int, num_frames: int, convention: SpanConvention
) -> range | None:
    if convention == "half_open" and 0 <= start < end <= num_frames:
        return range(start, end)
    if convention == "inclusive" and 0 <= start <= end < num_frames:
        return range(start, end + 1)
    if convention == "inclusive_clipped":
        frames, _clipped = inclusive_clipped_frames(start, end, num_frames)
        return frames or None
    return None


def relation_records_for_video(
    video: dict[str, Any],
    *,
    predicate_vocabulary: set[str],
    convention: SpanConvention,
) -> tuple[dict[tuple[int, int, int], set[str]], dict[str, Any]]:
    """Expand spans and group all predicates for one directed pair/frame."""

    targets: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    invalid_spans = []
    unknown_predicates: Counter[str] = Counter()
    num_frames = video["meta"]["num_frames"]
    for subject_id, object_id, predicate, spans in video["relations"]:
        if predicate not in predicate_vocabulary:
            unknown_predicates[predicate] += 1
            continue
        for start, end in spans:
            frames = _valid_frames(start, end, num_frames, convention)
            if frames is None:
                invalid_spans.append(
                    {"predicate": predicate, "subject_id": subject_id, "object_id": object_id,
                     "span": [start, end]}
                )
                continue
            for frame_index in frames:
                targets[(frame_index, subject_id, object_id)].add(predicate)
    return dict(targets), {
        "invalid_spans": invalid_spans,
        "unknown_predicate_records": dict(unknown_predicates),
    }


def _new_relation_totals() -> dict[str, Any]:
    return {
        "positive_pair_frame_records": 0,
        "predicate_label_assignments": 0,
        "multi_predicate_records": 0,
        "complete_feature_joins": 0,
        "missing_scene": 0,
        "missing_subject": 0,
        "missing_object": 0,
        "missing_union": 0,
        "invalid_spans": [],
        "unknown_predicate_records": Counter(),
        "predicate_support": Counter(),
        "records_by_official_split": Counter(),
        "train_category_triples": set(),
        "validation_category_triples": [],
    }


def _update_relation_totals(
    totals: dict[str, Any],
    *,
    video: dict[str, Any],
    video_id: str,
    official_split: str,
    targets: dict[tuple[int, int, int], set[str]],
    issues: dict[str, Any],
    feature_rows: dict[str, Any],
) -> None:
    categories = {record["object_id"]: record["category"] for record in video["objects"]}
    totals["invalid_spans"].extend(
        {"video_id": video_id, **record} for record in issues["invalid_spans"]
    )
    totals["unknown_predicate_records"].update(issues["unknown_predicate_records"])
    object_keys = feature_rows["object_keys"]
    pair_keys = feature_rows["pair_keys"]
    num_frames = video["meta"]["num_frames"]
    for (frame_index, subject_id, object_id), predicates in targets.items():
        totals["positive_pair_frame_records"] += 1
        totals["predicate_label_assignments"] += len(predicates)
        totals["multi_predicate_records"] += len(predicates) > 1
        totals["predicate_support"].update(predicates)
        totals["records_by_official_split"][official_split] += 1
        missing = False
        if not 0 <= frame_index < num_frames:
            totals["missing_scene"] += 1
            missing = True
        if (frame_index, subject_id) not in object_keys:
            totals["missing_subject"] += 1
            missing = True
        if (frame_index, object_id) not in object_keys:
            totals["missing_object"] += 1
            missing = True
        canonical_pair = tuple(sorted((subject_id, object_id)))
        if (frame_index, *canonical_pair) not in pair_keys:
            totals["missing_union"] += 1
            missing = True
        if not missing:
            totals["complete_feature_joins"] += 1
        for predicate in predicates:
            triple = (categories[subject_id], predicate, categories[object_id])
            if official_split == "train":
                totals["train_category_triples"].add(triple)
            else:
                totals["validation_category_triples"].append(triple)


def _finish_relation_totals(totals: dict[str, Any]) -> dict[str, Any]:
    records = totals["positive_pair_frame_records"]
    validation_triples = totals.pop("validation_category_triples")
    train_triples = totals.pop("train_category_triples")
    unseen = sum(triple not in train_triples for triple in validation_triples)
    totals["multi_predicate_fraction"] = (
        totals["multi_predicate_records"] / records if records else None
    )
    totals["feature_join_fraction"] = (
        totals["complete_feature_joins"] / records if records else None
    )
    totals["validation_label_assignments_on_unseen_category_triples"] = unseen
    totals["validation_label_assignments"] = len(validation_triples)
    totals["num_train_category_triples"] = len(train_triples)
    totals["invalid_span_count"] = len(totals["invalid_spans"])
    totals["unknown_predicate_records"] = dict(
        sorted(totals["unknown_predicate_records"].items())
    )
    totals["predicate_support"] = dict(sorted(totals["predicate_support"].items()))
    totals["records_by_official_split"] = dict(totals["records_by_official_split"])
    return totals


def _split_by_video(annotation: dict[str, Any]) -> dict[str, str]:
    result = {}
    for _source, split in annotation["split"].items():
        for name in ("train", "val"):
            for video_id in split[name]:
                if video_id in result:
                    raise ValueError(f"video occurs twice in official split: {video_id}")
                result[video_id] = name
    return result


def _span_overview(annotation: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for video in annotation["data"]:
        num_frames = video["meta"]["num_frames"]
        for _, _, _, spans in video["relations"]:
            for start, end in spans:
                counts["total"] += 1
                counts["end_equals_num_frames"] += end == num_frames
                counts["end_equals_last_frame"] += end == num_frames - 1
                counts["start_equals_end"] += start == end
                counts["outside_half_open_bounds"] += not (0 <= start < end <= num_frames)
                counts["outside_inclusive_bounds"] += not (0 <= start <= end < num_frames)
    return dict(counts)


def _gallery_candidates(
    video: dict[str, Any],
    *,
    source: str,
    predicate_vocabulary: set[str],
    feature_rows: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = []
    num_frames = video["meta"]["num_frames"]
    object_keys = feature_rows["object_keys"]
    pair_keys = feature_rows["pair_keys"]
    for subject_id, object_id, predicate, spans in video["relations"]:
        if predicate not in predicate_vocabulary:
            continue
        pair = tuple(sorted((subject_id, object_id)))
        for start, end in spans:
            if not (0 <= start < end < num_frames):
                continue
            if not all(
                (frame, subject_id) in object_keys
                and (frame, object_id) in object_keys
                and (frame, *pair) in pair_keys
                for frame in (end - 1, end)
            ):
                continue
            candidates.append(
                {
                    "source": source,
                    "video_id": video["video_id"],
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "predicate": predicate,
                    "span": [start, end],
                    "ends_on_last_frame": end == num_frames - 1,
                }
            )
    return candidates


def _render_gallery(
    candidates: list[dict[str, Any]],
    *,
    manifest_by_video: dict[str, dict[str, Any]],
    dataset_root: Path,
    feature_root: Path,
    output_directory: Path,
    sample_count: int,
) -> int:
    if sample_count <= 0 or not candidates:
        return 0
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    candidates.sort(
        key=lambda row: (
            not row["ends_on_last_frame"],
            row["video_id"],
            row["span"],
            row["predicate"],
        )
    )
    selected = []
    used_videos = set()
    for row in candidates:
        if row["video_id"] not in used_videos:
            selected.append(row)
            used_videos.add(row["video_id"])
        if len(selected) == sample_count:
            break
    if len(selected) < sample_count:
        selected.extend(row for row in candidates if row not in selected)
        selected = selected[:sample_count]

    image_directory = output_directory / "gallery"
    image_directory.mkdir(parents=True, exist_ok=True)
    cards = []
    for candidate_index, candidate in enumerate(selected):
        manifest = manifest_by_video[candidate["video_id"]]
        requested = {candidate["span"][1] - 1, candidate["span"][1]}
        video_path = dataset_root / manifest["video_path"]
        decoded = {}
        with av.open(str(video_path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame_index in requested:
                    decoded[frame_index] = frame.to_image().convert("RGB")
                if len(decoded) == len(requested):
                    break
        if set(decoded) != requested:
            raise ValueError(f"could not decode requested gallery frames from {video_path}")

        artifact_path = (
            feature_root / "videos" / candidate["source"] / f"{candidate['video_id']}.pt"
        )
        artifact = load_feature_artifact(artifact_path)
        keys = [
            (frame, pair[0], pair[1])
            for frame, pair in zip(
                artifact["pair_frame_index"].tolist(),
                artifact["pair_ids"].tolist(),
                strict=True,
            )
        ]
        boxes = dict(zip(keys, artifact["union_boxes_xyxy"].tolist(), strict=True))
        pair = tuple(sorted((candidate["subject_id"], candidate["object_id"])))
        for frame_index in sorted(requested):
            mask_path = dataset_root / manifest["mask_directory"] / f"{frame_index:04d}.png"
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image, copy=True)
            subject_mask = mask == candidate["subject_id"]
            object_mask = mask == candidate["object_id"]
            ys, xs = np.nonzero(subject_mask | object_mask)
            expected_box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            stored_box = boxes[(frame_index, *pair)]
            if stored_box != expected_box:
                raise ValueError(
                    f"stored union box disagrees with masks: {candidate['video_id']}/{frame_index}"
                )

            image = decoded[frame_index].convert("RGBA")
            overlay_array = np.zeros((image.height, image.width, 4), dtype=np.uint8)
            overlay_array[subject_mask] = (0, 210, 255, 100)
            overlay_array[object_mask] = (255, 40, 180, 100)
            image = Image.alpha_composite(image, Image.fromarray(overlay_array, mode="RGBA"))
            image.thumbnail((960, 960), Image.Resampling.LANCZOS)
            scale_x = image.width / manifest["width"]
            scale_y = image.height / manifest["height"]
            draw = ImageDraw.Draw(image)
            x0, y0, x1, y1 = stored_box
            draw.rectangle(
                (x0 * scale_x, y0 * scale_y, (x1 - 1) * scale_x, (y1 - 1) * scale_y),
                outline=(255, 225, 0, 255),
                width=3,
            )
            role = (
                "last half-open frame"
                if frame_index == candidate["span"][1] - 1
                else "inclusive-only endpoint"
            )
            label = (
                f"{candidate['video_id']} frame {frame_index} | {candidate['subject_id']} "
                f"-{candidate['predicate']}-> {candidate['object_id']} | {role}"
            )
            text_box = draw.textbbox((0, 0), label)
            draw.rectangle(
                (0, 0, min(image.width, text_box[2] + 8), text_box[3] + 8),
                fill=(0, 0, 0, 210),
            )
            draw.text((4, 4), label, fill=(255, 255, 255, 255))
            filename = f"{candidate_index:02d}-{candidate['video_id']}-{frame_index:06d}.png"
            image.convert("RGB").save(image_directory / filename, optimize=True)
            cards.append(
                f"<figure><img src='gallery/{html.escape(filename)}'><figcaption>"
                f"{html.escape(label)}<br>cyan = subject mask, magenta = object mask, "
                "yellow = stored union box</figcaption></figure>"
            )
    gallery_html = """<!doctype html><meta charset='utf-8'><title>PVSG audit gallery</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:2rem auto}img{max-width:100%}
figure{border:1px solid #bbb;padding:1rem;margin:0 0 2rem}figcaption{margin-top:.6rem}</style>
<h1>PVSG relation-boundary audit</h1>
<p>Each pair compares the last frame under half-open spans with the extra endpoint included by
inclusive spans. These images are evidence for review; the audit does not choose a convention.</p>
""" + "\n".join(cards)
    (output_directory / "gallery.html").write_text(gallery_html, encoding="utf-8")
    return len(cards)


def audit_snapshot(
    *,
    dataset_root: Path,
    extraction_manifest: Path,
    feature_root: Path,
    output_directory: Path,
    gallery_samples: int,
) -> tuple[dict[str, Any], bool]:
    """Audit all videos, write a report and gallery, and return report plus success."""

    manifest = _read_jsonl(extraction_manifest)
    manifest_by_video = {row["video_id"]: row for row in manifest}
    if len(manifest_by_video) != len(manifest):
        raise ValueError("manifest video IDs are not globally unique")
    annotation = _load_annotation(dataset_root / "pvsg.json")
    videos = {video["video_id"]: video for video in annotation["data"]}
    if set(videos) != set(manifest_by_video):
        raise ValueError("annotation and extraction manifest video IDs differ")
    official_split = _split_by_video(annotation)
    exclusions = load_exclusions()
    excluded_video_ids = set(exclusions)
    predicates = set(active_predicates(annotation, excluded_video_ids)[0])

    missing_artifacts = []
    invalid_artifacts = []
    completed = 0
    counts: Counter[str] = Counter()
    norm_totals = {
        name: {"count": 0, "sum": 0.0, "sum_squares": 0.0, "min": None, "max": None}
        for name in FEATURE_TABLES
    }
    provenance: Counter[str] = Counter()
    relation_totals = {"inclusive_clipped": _new_relation_totals()}
    gallery_candidates = []
    for index, manifest_row in enumerate(manifest, start=1):
        key = f"{manifest_row['source']}/{manifest_row['video_id']}"
        if manifest_row["video_id"] in excluded_video_ids:
            continue
        artifact_path = feature_root / "videos" / f"{key}.pt"
        if not artifact_path.is_file():
            missing_artifacts.append(key)
            continue
        try:
            artifact = load_feature_artifact(artifact_path)
            feature_rows = validate_feature_artifact(artifact, manifest_row)
        except Exception as error:
            invalid_artifacts.append({"video": key, "error": str(error)})
            continue
        completed += 1
        counts.update(feature_rows["counts"])
        provenance[json.dumps(feature_rows["provenance"], sort_keys=True)] += 1
        for name in FEATURE_TABLES:
            _merge_norms(norm_totals[name], feature_rows["feature_norms"][name])
        video = videos[manifest_row["video_id"]]
        for convention in ("inclusive_clipped",):
            targets, issues = relation_records_for_video(
                video,
                predicate_vocabulary=predicates,
                convention=convention,
            )
            _update_relation_totals(
                relation_totals[convention],
                video=video,
                video_id=manifest_row["video_id"],
                official_split=official_split[manifest_row["video_id"]],
                targets=targets,
                issues=issues,
                feature_rows=feature_rows,
            )
        gallery_candidates.extend(
            _gallery_candidates(
                video,
                source=manifest_row["source"],
                predicate_vocabulary=predicates,
                feature_rows=feature_rows,
            )
        )
        if index % 25 == 0 or index == len(manifest):
            print(f"audited {index}/{len(manifest)} manifest videos", flush=True)

    output_directory.mkdir(parents=True, exist_ok=True)
    rendered_images = _render_gallery(
        gallery_candidates,
        manifest_by_video=manifest_by_video,
        dataset_root=dataset_root,
        feature_root=feature_root,
        output_directory=output_directory,
        sample_count=gallery_samples,
    )
    report = {
        "schema_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "paths": {
            "dataset_root": str(dataset_root),
            "extraction_manifest": str(extraction_manifest),
            "feature_root": str(feature_root),
        },
        "artifacts": {
            "source_videos": len(manifest),
            "expected_retained": len(manifest) - len(exclusions),
            "valid": completed,
            "excluded": list(exclusions.values()),
            "missing": missing_artifacts,
            "invalid": invalid_artifacts,
            **dict(counts),
            "feature_norms": {
                name: _finish_norms(summary) for name, summary in norm_totals.items()
            },
            "provenance_groups": [
                {"videos": count, "metadata": json.loads(value)}
                for value, count in sorted(provenance.items())
            ],
        },
        "relation_spans": _span_overview(annotation),
        "relation_records": {
            convention: _finish_relation_totals(totals)
            for convention, totals in relation_totals.items()
        },
        "gallery": {
            "boundary_candidates": len(gallery_candidates),
            "rendered_images": rendered_images,
            "html": str(output_directory / "gallery.html") if rendered_images else None,
        },
        "decision": (
            "Inclusive endpoints clipped to valid frames; six reviewed source-defective videos "
            "excluded; 64 retained predicates; complete evidence required by initial protocols."
        ),
    }
    temporary = output_directory / f".report.json.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_directory / "report.json")
    success = completed == len(manifest) - len(exclusions) and not invalid_artifacts
    return report, success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--gallery-samples",
        type=int,
        default=6,
        help="Number of relation boundaries to render (two images each).",
    )
    arguments = parser.parse_args()
    report, success = audit_snapshot(
        dataset_root=arguments.dataset_root,
        extraction_manifest=arguments.extraction_manifest,
        feature_root=arguments.feature_root,
        output_directory=arguments.output_directory,
        gallery_samples=arguments.gallery_samples,
    )
    artifacts = report["artifacts"]
    print(
        f"valid retained artifacts: {artifacts['valid']}/{artifacts['expected_retained']}; "
        f"missing: {len(artifacts['missing'])}; invalid: {len(artifacts['invalid'])}"
    )
    for convention, statistics in report["relation_records"].items():
        print(
            f"{convention}: {statistics['positive_pair_frame_records']} records; "
            f"multi-label fraction {statistics['multi_predicate_fraction']}; "
            f"feature join fraction {statistics['feature_join_fraction']}"
        )
    print(f"wrote {arguments.output_directory / 'report.json'}")
    if report["gallery"]["html"]:
        print(f"wrote {report['gallery']['html']}")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

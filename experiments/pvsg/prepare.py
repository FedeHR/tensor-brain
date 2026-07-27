"""Prepare the pinned PVSG snapshot without preserving its dirty archive paths.

This is a one-time cluster-side data operation. It validates the annotation and
all ZIP members before writing a canonical dataset tree. It does not extract an
additional RGB-frame copy; feature jobs decode the original 5 FPS videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

PVSG_HUB_REPOSITORY = "Jingkang/PVSG"
PVSG_HUB_REVISION = "7e5f1ec9fd8f323182e84d990819854bb72da478"
PVSG_JSON_SHA256 = "2cbc23b060386ccf090475f90cb0282a3e96cefb29c69772f9f8fd916995ba08"
SOURCES = ("ego4d", "epic_kitchen", "vidor")


@dataclass(frozen=True)
class ArchiveSpec:
    relative_path: str
    source: str
    kind: Literal["videos", "masks"]
    md5: str


ARCHIVES = (
    ArchiveSpec("Ego4D/ego4d_videos.zip", "ego4d", "videos", "9334c74f5c831c80774862afa9d3f7f0"),
    ArchiveSpec("Ego4D/ego4d_masks.zip", "ego4d", "masks", "218ce689e1e8284e25b50280a5d29612"),
    ArchiveSpec(
        "EpicKitchen/epic_kitchen_videos.zip",
        "epic_kitchen",
        "videos",
        "b791a71ef24b14721a7b5041190ba4a3",
    ),
    ArchiveSpec(
        "EpicKitchen/epic_kitchen_masks.zip",
        "epic_kitchen",
        "masks",
        "03757120075de23328a11e56660175f4",
    ),
    ArchiveSpec("VidOR/vidor_videos.zip", "vidor", "videos", "fcc1a6f54ef60aa16fab335a6270c960"),
    ArchiveSpec("VidOR/vidor_masks.zip", "vidor", "masks", "17bfa5ec13235d86273bc9d067776862"),
)


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def _digest(path: Path, algorithm: str, *, progress_label: str | None = None) -> str:
    digest = hashlib.new(algorithm)
    file_size = path.stat().st_size
    bytes_read = 0
    next_report = max(file_size // 10, 512 * 1024 * 1024)
    started_at = time.monotonic()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            bytes_read += len(block)
            if progress_label is not None and bytes_read >= next_report:
                elapsed = time.monotonic() - started_at
                print(
                    f"{progress_label}: checked {_format_bytes(bytes_read)} / "
                    f"{_format_bytes(file_size)} in {elapsed:.0f}s",
                    flush=True,
                )
                next_report += max(file_size // 10, 512 * 1024 * 1024)
    return digest.hexdigest()


def _load_annotation(path: Path) -> dict[str, Any]:
    if _digest(path, "sha256") != PVSG_JSON_SHA256:
        raise ValueError(f"{path} is not the pinned PVSG annotation")
    with path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    if set(annotation) != {"data", "objects", "relations", "split"}:
        raise ValueError("pvsg.json has unexpected top-level fields")
    return annotation


def _annotation_index(
    annotation: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]:
    thing_categories = set(annotation["objects"]["thing"])
    stuff_categories = set(annotation["objects"]["stuff"])
    if len(thing_categories) != 115 or len(stuff_categories) != 11:
        raise ValueError("unexpected PVSG object vocabulary")
    if thing_categories & stuff_categories:
        raise ValueError("PVSG thing and stuff vocabularies overlap")

    source_by_video: dict[str, str] = {}
    for source in SOURCES:
        split = annotation["split"][source]
        if set(split) != {"train", "val"}:
            raise ValueError(f"unexpected split fields for {source}")
        for video_id in (*split["train"], *split["val"]):
            if video_id in source_by_video:
                raise ValueError(f"video ID occurs in multiple splits or sources: {video_id}")
            source_by_video[video_id] = source

    videos_by_id: dict[str, dict[str, Any]] = {}
    unknown_predicates: Counter[str] = Counter()
    invalid_relation_spans: list[dict[str, Any]] = []
    predicate_vocabulary = set(annotation["relations"])
    for video in annotation["data"]:
        video_id = video["video_id"]
        if video_id in videos_by_id:
            raise ValueError(f"duplicate annotation record: {video_id}")
        if video_id not in source_by_video:
            raise ValueError(f"video is absent from the official split: {video_id}")

        meta = video["meta"]
        if meta["fps"] != 5 or min(meta["height"], meta["width"], meta["num_frames"]) <= 0:
            raise ValueError(f"invalid video metadata: {video_id}")
        if meta["duration"] * meta["fps"] != meta["num_frames"]:
            raise ValueError(f"duration/frame-count mismatch: {video_id}")

        object_ids = [obj["object_id"] for obj in video["objects"]]
        if object_ids != list(range(1, len(object_ids) + 1)):
            raise ValueError(f"object IDs are not consecutive and list-aligned: {video_id}")
        for object_record in video["objects"]:
            category = object_record["category"]
            expected_is_thing = category in thing_categories
            if category not in thing_categories | stuff_categories:
                raise ValueError(f"unknown object category in {video_id}: {category}")
            if object_record["is_thing"] is not expected_is_thing:
                raise ValueError(f"is_thing disagrees with the vocabulary: {video_id}/{category}")
        valid_object_ids = set(object_ids)
        for subject_id, object_id, predicate, spans in video["relations"]:
            if subject_id not in valid_object_ids or object_id not in valid_object_ids:
                raise ValueError(f"relation references an unknown object: {video_id}")
            if predicate not in predicate_vocabulary:
                unknown_predicates[predicate] += 1
            for start, end in spans:
                if not (0 <= start < end <= meta["num_frames"]):
                    invalid_relation_spans.append(
                        {"video_id": video_id, "predicate": predicate, "span": [start, end]}
                    )
        videos_by_id[video_id] = video

    if set(videos_by_id) != set(source_by_video) or len(videos_by_id) != 400:
        raise ValueError("pvsg.json data records do not exactly match its train/val splits")

    audit = {
        "unknown_predicate_records": dict(sorted(unknown_predicates.items())),
        "invalid_relation_spans": invalid_relation_spans,
        "note": (
            "Reported without filtering or clipping; task construction needs an explicit policy."
        ),
    }
    return source_by_video, videos_by_id, audit


def archive_member_destination(
    member_name: str,
    *,
    source: str,
    kind: Literal["videos", "masks"],
    expected_video_ids: set[str],
) -> Path:
    """Map one dirty PVSG archive member to a canonical relative path.

    Arbitrary leading author-cluster components are discarded, but the expected
    ``<source>/<kind>/...`` suffix and every video ID are checked exactly.
    """

    if "\\" in member_name:
        raise ValueError(f"archive member uses backslashes: {member_name}")
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {member_name}")
    matches = [
        index
        for index in range(len(path.parts) - 1)
        if path.parts[index].lower() == source and path.parts[index + 1].lower() == kind
    ]
    if len(matches) != 1:
        raise ValueError(f"member does not contain one {source}/{kind} boundary: {member_name}")
    suffix = path.parts[matches[0] + 2 :]

    if kind == "videos":
        if len(suffix) != 1:
            raise ValueError(f"unexpected video member layout: {member_name}")
        filename = PurePosixPath(suffix[0])
        if filename.suffix.lower() != ".mp4" or filename.stem not in expected_video_ids:
            raise ValueError(f"unexpected PVSG video member: {member_name}")
        return Path(f"{filename.stem}.mp4")

    if len(suffix) != 2 or suffix[0] not in expected_video_ids:
        raise ValueError(f"unexpected mask member layout: {member_name}")
    frame = PurePosixPath(suffix[1])
    if frame.suffix.lower() != ".png" or not frame.stem.isdecimal():
        raise ValueError(f"unexpected PVSG mask member: {member_name}")
    return Path(suffix[0]) / f"{int(frame.stem):04d}.png"


def _expected_destinations(
    spec: ArchiveSpec,
    video_ids: set[str],
    videos_by_id: dict[str, dict[str, Any]],
) -> set[Path]:
    if spec.kind == "videos":
        return {Path(f"{video_id}.mp4") for video_id in video_ids}
    return {
        Path(video_id) / f"{frame_index:04d}.png"
        for video_id in video_ids
        for frame_index in range(videos_by_id[video_id]["meta"]["num_frames"])
    }


def _extract_archive(
    archive_path: Path,
    spec: ArchiveSpec,
    *,
    dataset_root: Path,
    staging_root: Path,
    video_ids: set[str],
    videos_by_id: dict[str, dict[str, Any]],
) -> None:
    label = f"{spec.source}/{spec.kind}"
    final_directory = dataset_root / spec.source / spec.kind
    if final_directory.exists():
        print(f"{label}: already published; skipping", flush=True)
        return
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    print(f"{label}: checking archive MD5", flush=True)
    if _digest(archive_path, "md5", progress_label=label) != spec.md5:
        raise ValueError(f"MD5 mismatch: {archive_path}")
    staging_directory = staging_root / f"{spec.source}_{spec.kind}"
    if staging_directory.exists():
        raise FileExistsError(
            f"incomplete staging directory exists: {staging_directory}; inspect and remove it first"
        )

    expected = _expected_destinations(spec, video_ids, videos_by_id)
    members: list[tuple[zipfile.ZipInfo, Path]] = []
    destinations: set[Path] = set()
    with zipfile.ZipFile(archive_path) as archive:
        print(f"{label}: validating archive members", flush=True)
        for member in archive.infolist():
            if member.is_dir() or "__MACOSX" in PurePosixPath(member.filename).parts:
                continue
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode) or member.flag_bits & 0x1:
                raise ValueError(f"links and encrypted members are forbidden: {member.filename}")
            destination = archive_member_destination(
                member.filename,
                source=spec.source,
                kind=spec.kind,
                expected_video_ids=video_ids,
            )
            if destination in destinations:
                raise ValueError(f"duplicate normalized archive destination: {destination}")
            destinations.add(destination)
            members.append((member, destination))

        if destinations != expected:
            missing = sorted(str(path) for path in expected - destinations)[:5]
            extra = sorted(str(path) for path in destinations - expected)[:5]
            raise ValueError(f"archive content mismatch; missing={missing}, extra={extra}")

        staging_directory.mkdir(parents=True)
        total_files = len(members)
        total_bytes = sum(member.file_size for member, _ in members)
        print(
            f"{label}: extracting {total_files:,} files "
            f"({_format_bytes(total_bytes)})",
            flush=True,
        )
        started_at = time.monotonic()
        bytes_written = 0
        next_percent = 10
        for index, (member, relative_destination) in enumerate(members, start=1):
            destination = staging_directory / relative_destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, destination.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
            bytes_written += member.file_size
            percent = index * 100 // total_files
            if percent >= next_percent or index == total_files:
                elapsed = time.monotonic() - started_at
                rate = bytes_written / max(elapsed, 1e-9)
                print(
                    f"{label}: {percent}% ({index:,} / {total_files:,} files, "
                    f"{_format_bytes(bytes_written)} / {_format_bytes(total_bytes)}, "
                    f"{elapsed:.0f}s, {_format_bytes(int(rate))}/s)",
                    flush=True,
                )
                while percent >= next_percent:
                    next_percent += 10

    final_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory.replace(final_directory)
    print(f"{label}: published {final_directory}", flush=True)


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_pvsg(pvsg_root: Path) -> None:
    archives_root = pvsg_root / "archives"
    dataset_root = pvsg_root / "dataset"
    staging_root = pvsg_root / "staging"
    manifests_root = pvsg_root / "manifests"
    annotation_source = archives_root / "pvsg.json"
    annotation = _load_annotation(annotation_source)
    source_by_video, videos_by_id, annotation_audit = _annotation_index(annotation)

    for spec in ARCHIVES:
        video_ids = {
            video_id for video_id, source in source_by_video.items() if source == spec.source
        }
        _extract_archive(
            archives_root / spec.relative_path,
            spec,
            dataset_root=dataset_root,
            staging_root=staging_root,
            video_ids=video_ids,
            videos_by_id=videos_by_id,
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    annotation_target = dataset_root / "pvsg.json"
    if annotation_target.exists():
        if _digest(annotation_target, "sha256") != PVSG_JSON_SHA256:
            raise ValueError(f"conflicting annotation already exists: {annotation_target}")
    else:
        shutil.copy2(annotation_source, annotation_target)

    rows = []
    for source in SOURCES:
        source_video_ids = {
            video_id for video_id, video_source in source_by_video.items() if video_source == source
        }
        video_directory = dataset_root / source / "videos"
        mask_root = dataset_root / source / "masks"
        actual_videos = {
            path.name for path in video_directory.iterdir() if path.is_file()
        }
        expected_videos = {f"{video_id}.mp4" for video_id in source_video_ids}
        actual_mask_directories = {
            path.name for path in mask_root.iterdir() if path.is_dir()
        }
        if actual_videos != expected_videos or actual_mask_directories != source_video_ids:
            raise ValueError(f"canonical {source} video/mask IDs do not match pvsg.json")

        for video_id in sorted(source_video_ids):
            meta = videos_by_id[video_id]["meta"]
            video_path = Path(source) / "videos" / f"{video_id}.mp4"
            mask_directory = Path(source) / "masks" / video_id
            if not (dataset_root / video_path).is_file():
                raise FileNotFoundError(dataset_root / video_path)
            mask_paths = list((dataset_root / mask_directory).iterdir())
            if not all(path.is_file() and path.suffix == ".png" for path in mask_paths):
                raise ValueError(f"unexpected entry in mask directory: {source}/{video_id}")
            actual_masks = {path.name for path in mask_paths}
            expected_masks = {f"{index:04d}.png" for index in range(meta["num_frames"])}
            if actual_masks != expected_masks:
                raise ValueError(f"mask frames do not match pvsg.json: {source}/{video_id}")
            rows.append(
                {
                    "source": source,
                    "video_id": video_id,
                    "video_path": video_path.as_posix(),
                    "mask_directory": mask_directory.as_posix(),
                    "num_frames": meta["num_frames"],
                    "height": meta["height"],
                    "width": meta["width"],
                    "fps": meta["fps"],
                    "num_objects": len(videos_by_id[video_id]["objects"]),
                }
            )

    manifests_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl_atomic(manifests_root / "videos.jsonl", rows)
    _write_json_atomic(
        manifests_root / "source.json",
        {
            "schema_version": 1,
            "hub_repository": PVSG_HUB_REPOSITORY,
            "hub_revision": PVSG_HUB_REVISION,
            "pvsg_json_sha256": PVSG_JSON_SHA256,
            "num_videos": len(rows),
            "num_masks": sum(row["num_frames"] for row in rows),
            "source_counts": dict(Counter(row["source"] for row in rows)),
            "archives": [spec.__dict__ for spec in ARCHIVES],
            "annotation_audit": annotation_audit,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pvsg-root",
        type=Path,
        default=Path("/nfs/data8/harjes/MASTER/data/pvsg"),
        help="Root containing archives/, staging/, dataset/, and manifests/.",
    )
    arguments = parser.parse_args()
    prepare_pvsg(arguments.pvsg_root.resolve())


if __name__ == "__main__":
    main()

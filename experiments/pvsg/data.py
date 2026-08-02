"""Load materialized PVSG rows and their cached DINO evidence."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset, Sampler, default_collate

from experiments.pvsg.extract import FEATURE_SCHEMA_VERSION, load_feature_artifact
from experiments.pvsg.snapshot_io import read_jsonl

NORMALIZATION_EPSILON = 1e-12
_FEATURE_TABLES = ("scene_features", "object_features", "union_features")


def normalize_dino(
    features: Float[Tensor, "*rows feature"],
) -> Float[Tensor, "*rows feature"]:
    """Convert DINO evidence to float32 with unit per-component RMS.

    Scaling each nonzero L2-normalized vector by ``sqrt(feature_dim)`` preserves
    its direction while putting its components at the natural pre-CBS scale.
    """

    feature_dim = features.shape[-1]
    if feature_dim <= 0:
        raise ValueError("DINO features must have a non-empty feature axis")
    return math.sqrt(feature_dim) * F.normalize(
        features.float(), p=2, dim=-1, eps=NORMALIZATION_EPSILON
    )


@lru_cache(maxsize=4)
def _video_feature_tables(
    feature_root: Path,
    source: str,
    video_id: str,
) -> dict[str, Tensor]:
    path = feature_root / "videos" / source / f"{video_id}.pt"
    artifact = load_feature_artifact(path)
    metadata = artifact.get("metadata", {})
    if (
        metadata.get("schema_version") != FEATURE_SCHEMA_VERSION
        or metadata.get("source") != source
        or metadata.get("video_id") != video_id
    ):
        raise ValueError(f"feature artifact does not match the manifest row: {path}")
    return {name: artifact[name] for name in _FEATURE_TABLES}


class VideoBlockSampler(Sampler[int]):
    """Shuffle videos and their rows while keeping each video's I/O contiguous."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        blocks: dict[tuple[str, str], list[int]] = {}
        for index, record in enumerate(records):
            key = (record["source"], record["video_id"])
            blocks.setdefault(key, []).append(index)
        self.blocks = tuple(tuple(indices) for indices in blocks.values())
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        block_order = torch.randperm(len(self.blocks), generator=self.generator).tolist()
        for block_index in block_order:
            block = self.blocks[block_index]
            row_order = torch.randperm(len(block), generator=self.generator).tolist()
            yield from (block[position] for position in row_order)

    def __len__(self) -> int:
        return sum(len(block) for block in self.blocks)


class PVSGObjectDataset(Dataset[dict[str, Any]]):
    """Object observations with RMS-normalized scene and mask-pooled DINO evidence."""

    def __init__(self, manifest_path: Path, feature_root: Path) -> None:
        self.records = read_jsonl(manifest_path)
        self.feature_root = feature_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tables = _video_feature_tables(self.feature_root, record["source"], record["video_id"])
        return {
            "scene_features": normalize_dino(tables["scene_features"][record["scene_row"]]),
            "object_features": normalize_dino(tables["object_features"][record["object_row"]]),
            "identity": record["identity"],
            "category": record["category"],
            "source": record["source"],
            "video_id": record["video_id"],
            "frame_index": record["frame_index"],
            "object_id": record["object_id"],
            "mask_area": record["mask_area"],
        }


class PVSGPairDataset(Dataset[dict[str, Any]]):
    """Positive-pair rows with the four RMS-normalized Section 6 evidence vectors."""

    def __init__(self, manifest_path: Path, feature_root: Path) -> None:
        self.records = read_jsonl(manifest_path)
        self.feature_root = feature_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if not record["has_complete_evidence"]:
            raise ValueError("the initial pair dataset requires complete DINO evidence")
        tables = _video_feature_tables(self.feature_root, record["source"], record["video_id"])
        return {
            "scene_features": normalize_dino(tables["scene_features"][record["scene_row"]]),
            "subject_features": normalize_dino(
                tables["object_features"][record["subject_row"]]
            ),
            "object_features": normalize_dino(tables["object_features"][record["object_row"]]),
            "union_features": normalize_dino(tables["union_features"][record["union_row"]]),
            "subject_identity": record["subject_identity"],
            "object_identity": record["object_identity"],
            "subject_category": record["subject_category"],
            "object_category": record["object_category"],
            "predicates": tuple(record["predicates"]),
            "source": record["source"],
            "video_id": record["video_id"],
            "frame_index": record["frame_index"],
            "subject_id": record["subject_id"],
            "object_id": record["object_id"],
        }


def collate_pair_batch(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack pair examples while retaining each variable-length predicate set."""

    predicates = [example["predicates"] for example in examples]
    collated = default_collate(
        [
            {key: value for key, value in example.items() if key != "predicates"}
            for example in examples
        ]
    )
    collated["predicates"] = predicates
    return collated

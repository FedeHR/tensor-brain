"""Extract task-neutral PVSG scene, object-mask, and pair-union DINOv3 features."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from experiments.pvsg.features import FrameRegionFeatures, extract_frame_regions
from experiments.pvsg.prepare import PVSG_HUB_REVISION, PVSG_JSON_SHA256

DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
FEATURE_SCHEMA_VERSION = 2


def patch_aligned_size(
    image_size: tuple[int, int], *, long_edge: int, patch_size: int
) -> tuple[int, int]:
    """Resize a full frame approximately isotropically onto the DINO patch grid."""

    height, width = image_size
    if min(height, width, long_edge, patch_size) <= 0:
        raise ValueError("image, long-edge, and patch dimensions must be positive")
    scale = long_edge / max(height, width)

    def aligned(value: int) -> int:
        return max(patch_size, round(value * scale / patch_size) * patch_size)

    return aligned(height), aligned(width)


def split_dinov3_tokens(
    sequence: torch.Tensor,
    *,
    processed_size: tuple[int, int],
    patch_size: int,
    num_register_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split normalized CLS and spatial patch tokens from a DINOv3 sequence."""

    if sequence.ndim != 3:
        raise ValueError("DINOv3 sequence must have shape [batch, tokens, feature]")
    height, width = processed_size
    if height % patch_size or width % patch_size:
        raise ValueError("processed dimensions must be divisible by the patch size")
    grid_height, grid_width = height // patch_size, width // patch_size
    expected_tokens = 1 + num_register_tokens + grid_height * grid_width
    if sequence.shape[1] != expected_tokens:
        raise ValueError(
            f"DINOv3 returned {sequence.shape[1]} tokens, expected {expected_tokens}"
        )
    scene_features = sequence[:, 0]
    patch_features = sequence[:, 1 + num_register_tokens :].reshape(
        sequence.shape[0], grid_height, grid_width, sequence.shape[2]
    )
    return scene_features, patch_features


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [(row["source"], row["video_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("the extraction manifest contains duplicate source/video IDs")
    return rows


def _contained_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"manifest path must be relative: {relative}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes the dataset root: {relative}") from error
    return path


def _append_regions(
    tables: dict[str, list[torch.Tensor]],
    regions: FrameRegionFeatures,
    *,
    frame_index: int,
) -> None:
    tables["scene_frame_index"].append(torch.tensor([frame_index], dtype=torch.long))
    tables["scene_features"].append(regions.scene_feature[None].to(torch.float16))

    object_count = len(regions.object_ids)
    tables["object_frame_index"].append(
        torch.full((object_count,), frame_index, dtype=torch.long)
    )
    tables["object_ids"].append(regions.object_ids.cpu())
    tables["object_mask_areas"].append(regions.object_mask_areas.cpu())
    tables["object_features"].append(regions.object_features.cpu().to(torch.float16))

    pair_count = len(regions.pair_ids)
    tables["pair_frame_index"].append(torch.full((pair_count,), frame_index, dtype=torch.long))
    tables["pair_ids"].append(regions.pair_ids.cpu())
    tables["union_boxes_xyxy"].append(regions.union_boxes_xyxy.cpu())
    tables["union_features"].append(regions.union_features.cpu().to(torch.float16))


def _empty_tables() -> dict[str, list[torch.Tensor]]:
    return {
        "scene_frame_index": [],
        "scene_features": [],
        "object_frame_index": [],
        "object_ids": [],
        "object_mask_areas": [],
        "object_features": [],
        "pair_frame_index": [],
        "pair_ids": [],
        "union_boxes_xyxy": [],
        "union_features": [],
    }


def _finalize_tables(tables: dict[str, list[torch.Tensor]], feature_dim: int) -> dict[str, Any]:
    shapes = {
        "scene_frame_index": (0,),
        "scene_features": (0, feature_dim),
        "object_frame_index": (0,),
        "object_ids": (0,),
        "object_mask_areas": (0,),
        "object_features": (0, feature_dim),
        "pair_frame_index": (0,),
        "pair_ids": (0, 2),
        "union_boxes_xyxy": (0, 4),
        "union_features": (0, feature_dim),
    }
    dtypes = {
        "scene_frame_index": torch.long,
        "scene_features": torch.float16,
        "object_frame_index": torch.long,
        "object_ids": torch.long,
        "object_mask_areas": torch.long,
        "object_features": torch.float16,
        "pair_frame_index": torch.long,
        "pair_ids": torch.long,
        "union_boxes_xyxy": torch.long,
        "union_features": torch.float16,
    }
    return {
        key: torch.cat(chunks) if chunks else torch.empty(shapes[key], dtype=dtypes[key])
        for key, chunks in tables.items()
    }


def _load_mask(path: Path, expected_size: tuple[int, int], num_objects: int) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        array = np.array(image, copy=True)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"mask is not a single-channel integer image: {path}")
    if array.shape != expected_size:
        raise ValueError(f"mask dimensions disagree with pvsg.json: {path}")
    mask = torch.from_numpy(array.astype(np.int64, copy=False))
    if int(mask.min()) < 0 or int(mask.max()) > num_objects:
        raise ValueError(f"mask contains an unknown object ID: {path}")
    return mask


def _existing_artifact_is_valid(
    path: Path,
    record: dict[str, Any],
    *,
    long_edge: int,
    inference_autocast_dtype: str,
) -> bool:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    metadata = artifact.get("metadata", {})
    return (
        metadata.get("schema_version") == FEATURE_SCHEMA_VERSION
        and metadata.get("source") == record["source"]
        and metadata.get("video_id") == record["video_id"]
        and metadata.get("num_frames") == record["num_frames"]
        and tuple(metadata.get("original_size_hw", ()))
        == (record["height"], record["width"])
        and metadata.get("long_edge") == long_edge
        and metadata.get("inference_autocast_dtype") == inference_autocast_dtype
        and metadata.get("dino_model_id") == DINO_MODEL_ID
        and metadata.get("dino_model_revision") == DINO_MODEL_REVISION
        and metadata.get("pvsg_hub_revision") == PVSG_HUB_REVISION
        and metadata.get("pvsg_json_sha256") == PVSG_JSON_SHA256
    )


def extract_video(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    long_edge: int,
) -> Path:
    """Extract one manifest video and atomically write one self-describing artifact."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction was requested, but CUDA is unavailable")

    video_path = _contained_path(dataset_root, record["video_path"])
    mask_directory = _contained_path(dataset_root, record["mask_directory"])
    output_path = output_root / "videos" / record["source"] / f"{record['video_id']}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inference_autocast_dtype = "float16" if device.type == "cuda" else "float32"
    if output_path.exists():
        if _existing_artifact_is_valid(
            output_path,
            record,
            long_edge=long_edge,
            inference_autocast_dtype=inference_autocast_dtype,
        ):
            key = f"{record['source']}/{record['video_id']}"
            print(f"valid artifact already exists; skipping {key}")
            return output_path
        raise FileExistsError(
            f"existing artifact has a different or invalid contract: {output_path}"
        )

    try:
        import av
        import transformers
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        message = "install the pvsg optional dependency group before extraction"
        raise RuntimeError(message) from error

    processor = AutoImageProcessor.from_pretrained(
        DINO_MODEL_ID,
        revision=DINO_MODEL_REVISION,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        DINO_MODEL_ID,
        revision=DINO_MODEL_REVISION,
        local_files_only=True,
    ).to(device)
    model.eval()
    patch_size = int(model.config.patch_size)
    feature_dim = int(model.config.hidden_size)
    num_register_tokens = int(model.config.num_register_tokens)
    original_size = (int(record["height"]), int(record["width"]))
    processed_size = patch_aligned_size(
        original_size, long_edge=long_edge, patch_size=patch_size
    )

    tables = _empty_tables()
    seen_object_ids: set[int] = set()
    pending_images: list[Any] = []
    pending_masks: list[torch.Tensor] = []
    pending_indices: list[int] = []

    def process_batch() -> None:
        if not pending_images:
            return
        inputs = processor(
            images=pending_images,
            size={"height": processed_size[0], "width": processed_size[1]},
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"]
        if tuple(pixel_values.shape[-2:]) != processed_size:
            raise ValueError("DINOv3 processor did not produce the requested full-frame size")
        pixel_values = pixel_values.to(device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            outputs = model(pixel_values=pixel_values)
        scenes, patches = split_dinov3_tokens(
            outputs.last_hidden_state,
            processed_size=processed_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
        )
        for local_index, (frame_index, mask) in enumerate(
            zip(pending_indices, pending_masks, strict=True)
        ):
            regions = extract_frame_regions(
                scenes[local_index].float().cpu(),
                patches[local_index].float().cpu(),
                mask,
            )
            seen_object_ids.update(regions.object_ids.tolist())
            _append_regions(tables, regions, frame_index=frame_index)
        pending_images.clear()
        pending_masks.clear()
        pending_indices.clear()

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        decoded_frames = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index >= record["num_frames"]:
                raise ValueError(f"decoded more frames than pvsg.json declares: {video_path}")
            image = frame.to_image().convert("RGB")
            if (image.height, image.width) != original_size:
                raise ValueError(f"decoded frame dimensions disagree with pvsg.json: {video_path}")
            mask_path = mask_directory / f"{frame_index:04d}.png"
            pending_images.append(image)
            pending_masks.append(_load_mask(mask_path, original_size, record["num_objects"]))
            pending_indices.append(frame_index)
            decoded_frames += 1
            if len(pending_images) == batch_size:
                process_batch()
        process_batch()

    if decoded_frames != record["num_frames"]:
        raise ValueError(
            f"decoded {decoded_frames} frames, but pvsg.json declares {record['num_frames']}"
        )
    expected_object_ids = set(range(1, record["num_objects"] + 1))
    if seen_object_ids != expected_object_ids:
        raise ValueError(
            f"objects absent from all masks: {sorted(expected_object_ids - seen_object_ids)}"
        )

    artifact = _finalize_tables(tables, feature_dim)
    artifact["metadata"] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source": record["source"],
        "video_id": record["video_id"],
        "num_frames": record["num_frames"],
        "fps": record["fps"],
        "original_size_hw": original_size,
        "processed_size_hw": processed_size,
        "long_edge": long_edge,
        "resize_policy": f"full frame, long edge {long_edge}, nearest patch multiple",
        "scene_feature": "final normalized CLS token",
        "object_feature": "exact source-mask/patch-cell overlap pooling",
        "pair_feature": "enclosing union-box/patch-cell overlap pooling",
        "union_box_coordinates": "half-open xyxy in original frame coordinates",
        "feature_storage_dtype": "float16",
        "inference_autocast_dtype": inference_autocast_dtype,
        "dino_model_id": DINO_MODEL_ID,
        "dino_model_revision": DINO_MODEL_REVISION,
        "feature_dim": feature_dim,
        "patch_size": patch_size,
        "num_register_tokens": num_register_tokens,
        "processor": processor.to_dict(),
        "pvsg_hub_revision": PVSG_HUB_REVISION,
        "pvsg_json_sha256": PVSG_JSON_SHA256,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "cuda_device_capability": (
            torch.cuda.get_device_capability(device) if device.type == "cuda" else None
        ),
        "transformers_version": transformers.__version__,
        "av_version": av.__version__,
    }
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(artifact, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(
        f"wrote {output_path}: frames={len(artifact['scene_frame_index'])}, "
        f"objects={len(artifact['object_frame_index'])}, "
        f"pairs={len(artifact['pair_frame_index'])}, size_mib={size_mib:.1f}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--long-edge", type=int, default=448)
    arguments = parser.parse_args()

    rows = _read_manifest(arguments.manifest.resolve())
    if not 0 <= arguments.index < len(rows):
        raise IndexError(f"manifest index {arguments.index} is outside [0, {len(rows)})")
    extract_video(
        rows[arguments.index],
        dataset_root=arguments.dataset_root.resolve(),
        output_root=arguments.output_root.resolve(),
        device=torch.device(arguments.device),
        batch_size=arguments.batch_size,
        long_edge=arguments.long_edge,
    )


if __name__ == "__main__":
    main()

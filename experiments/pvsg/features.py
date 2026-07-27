"""Region pooling for task-neutral PVSG frame features.

This module knows about DINO patch grids and PVSG panoptic masks. It does not
know about relation labels, experiment splits, ontologies, or Tensor Brain
indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
from jaxtyping import Float, Int
from torch import Tensor


@dataclass(frozen=True)
class FrameRegionFeatures:
    """Scene, visible-object, and visible-pair evidence for one video frame."""

    scene_feature: Float[Tensor, " feature"]
    object_ids: Int[Tensor, " objects"]
    object_mask_areas: Int[Tensor, " objects"]
    object_features: Float[Tensor, "objects feature"]
    pair_ids: Int[Tensor, "pairs two"]
    union_boxes_xyxy: Int[Tensor, "pairs four"]
    union_features: Float[Tensor, "pairs feature"]


def _validate_inputs(scene_feature: Tensor, patch_features: Tensor, panoptic_mask: Tensor) -> None:
    if patch_features.ndim != 3 or not torch.is_floating_point(patch_features):
        raise ValueError("patch_features must be a floating tensor [patch_y, patch_x, feature]")
    if (
        scene_feature.ndim != 1
        or not torch.is_floating_point(scene_feature)
        or scene_feature.shape[0] != patch_features.shape[-1]
    ):
        raise ValueError("scene_feature must be a floating tensor [feature]")
    if 0 in patch_features.shape:
        raise ValueError("patch_features must have no empty dimensions")
    if panoptic_mask.ndim != 2 or torch.is_floating_point(panoptic_mask):
        raise ValueError("panoptic_mask must be an integer tensor [height, width]")
    if 0 in panoptic_mask.shape:
        raise ValueError("panoptic_mask must have no empty dimensions")


def _visible_object_ids(mask: Tensor, *, void_id: int) -> Tensor:
    object_ids = torch.unique(mask)
    return object_ids[object_ids != void_id].sort().values.to(torch.long)


def _axis_cell_overlaps(
    source_size: int,
    target_cells: int,
    *,
    device: torch.device,
) -> Tensor:
    """Return source-pixel overlap with equal target-grid cells."""

    source_edges = torch.arange(source_size, device=device, dtype=torch.float32)
    target_edges = (
        torch.arange(target_cells + 1, device=device, dtype=torch.float32)
        * source_size
        / target_cells
    )
    return (
        torch.minimum(target_edges[1:, None], source_edges[None] + 1)
        - torch.maximum(target_edges[:-1, None], source_edges[None])
    ).clamp(min=0)


def _object_patch_weights(mask: Tensor, object_ids: Tensor, grid_size: tuple[int, int]) -> Tensor:
    """Return fractional mask coverage for every DINO patch.

    Each source pixel is treated as a unit square. Its exact overlap with every
    patch cell is accumulated, including when image dimensions are not divisible
    by the DINO patch-grid dimensions.
    """

    if object_ids.numel() == 0:
        return mask.new_empty((0, *grid_size), dtype=torch.float32)
    grid_height, grid_width = grid_size
    height_overlaps = _axis_cell_overlaps(mask.shape[0], grid_height, device=mask.device)
    width_overlaps = _axis_cell_overlaps(mask.shape[1], grid_width, device=mask.device)
    binary_masks = (mask.unsqueeze(0) == object_ids[:, None, None]).to(torch.float32)
    weights = torch.einsum(
        "yh,ohw,xw->oyx",
        height_overlaps,
        binary_masks,
        width_overlaps,
    )
    return weights


def _pool_weights(patch_features: Tensor, weights: Tensor) -> Tensor:
    """Mean-pool a patch grid using one non-negative weight map per region."""

    if weights.ndim != 3 or weights.shape[1:] != patch_features.shape[:2]:
        raise ValueError("weights must have shape [regions, patch_y, patch_x]")
    flat_weights = weights.to(patch_features.dtype).flatten(1)
    denominators = flat_weights.sum(dim=1, keepdim=True)
    if bool((denominators <= 0).any()):
        raise ValueError("every pooled region must overlap at least one patch")
    flat_features = patch_features.flatten(0, 1)
    return flat_weights @ flat_features / denominators


def _object_boxes_xyxy(mask: Tensor, object_ids: Tensor) -> Tensor:
    """Return half-open original-mask boxes used only to construct pair unions."""

    boxes = []
    for object_id in object_ids.tolist():
        ys, xs = torch.nonzero(mask == object_id, as_tuple=True)
        boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not boxes:
        return torch.empty((0, 4), dtype=torch.long, device=mask.device)
    return torch.tensor(boxes, dtype=torch.long, device=mask.device)


def _union_rows(object_ids: Tensor, object_boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Construct canonical pair IDs and their enclosing boxes."""

    pair_positions = list(combinations(range(len(object_ids)), 2))
    if not pair_positions:
        return (
            torch.empty((0, 2), dtype=torch.long, device=object_ids.device),
            torch.empty((0, 4), dtype=torch.long, device=object_ids.device),
        )

    positions = torch.tensor(pair_positions, dtype=torch.long, device=object_ids.device)
    first = object_boxes[positions[:, 0]]
    second = object_boxes[positions[:, 1]]
    boxes = torch.stack(
        (
            torch.minimum(first[:, 0], second[:, 0]),
            torch.minimum(first[:, 1], second[:, 1]),
            torch.maximum(first[:, 2], second[:, 2]),
            torch.maximum(first[:, 3], second[:, 3]),
        ),
        dim=1,
    )
    return object_ids[positions], boxes


def rectangle_patch_weights(
    boxes_xyxy: Int[Tensor, "boxes four"],
    *,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    dtype: torch.dtype,
) -> Float[Tensor, "boxes patch_y patch_x"]:
    """Measure the fractional overlap of original-image boxes with a patch grid."""

    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape [boxes, 4]")
    image_height, image_width = image_size
    grid_height, grid_width = grid_size
    if min(image_height, image_width, grid_height, grid_width) <= 0:
        raise ValueError("image and grid dimensions must be positive")
    if boxes_xyxy.numel() == 0:
        return torch.empty(
            (0, grid_height, grid_width), dtype=dtype, device=boxes_xyxy.device
        )

    boxes = boxes_xyxy.to(dtype)
    x0 = boxes[:, 0] * (grid_width / image_width)
    y0 = boxes[:, 1] * (grid_height / image_height)
    x1 = boxes[:, 2] * (grid_width / image_width)
    y1 = boxes[:, 3] * (grid_height / image_height)
    if bool((x0 < 0).any() or (y0 < 0).any()):
        raise ValueError("boxes must lie within the image")
    if bool((x1 > grid_width).any() or (y1 > grid_height).any()):
        raise ValueError("boxes must lie within the image")
    if bool((x1 <= x0).any() or (y1 <= y0).any()):
        raise ValueError("boxes must have positive half-open area")

    columns = torch.arange(grid_width, device=boxes.device, dtype=dtype)
    rows = torch.arange(grid_height, device=boxes.device, dtype=dtype)
    x_overlap = (
        torch.minimum(x1[:, None], columns[None] + 1)
        - torch.maximum(x0[:, None], columns[None])
    ).clamp(min=0, max=1)
    y_overlap = (
        torch.minimum(y1[:, None], rows[None] + 1)
        - torch.maximum(y0[:, None], rows[None])
    ).clamp(min=0, max=1)
    return y_overlap[:, :, None] * x_overlap[:, None, :]


def extract_frame_regions(
    scene_feature: Float[Tensor, " feature"],
    patch_features: Float[Tensor, "patch_y patch_x feature"],
    panoptic_mask: Int[Tensor, "height width"],
    *,
    void_id: int = 0,
) -> FrameRegionFeatures:
    """Pool one DINO patch grid into task-neutral PVSG region features.

    The scene is DINOv3's normalized CLS token. Objects use exact fractional
    mask coverage. Pairs use the enclosing union box and are stored canonically
    by ascending object ID.
    """

    _validate_inputs(scene_feature, patch_features, panoptic_mask)
    scene_feature = scene_feature.to(device=patch_features.device)
    mask = panoptic_mask.to(device=patch_features.device)
    object_ids = _visible_object_ids(mask, void_id=void_id)
    grid_size = (patch_features.shape[0], patch_features.shape[1])

    object_weights = _object_patch_weights(mask, object_ids, grid_size)
    object_features = _pool_weights(patch_features, object_weights)
    if object_ids.numel():
        object_areas = torch.stack([(mask == object_id).sum() for object_id in object_ids]).long()
    else:
        object_areas = torch.empty(0, dtype=torch.long, device=mask.device)

    object_boxes = _object_boxes_xyxy(mask, object_ids)
    pair_ids, union_boxes = _union_rows(object_ids, object_boxes)
    union_weights = rectangle_patch_weights(
        union_boxes,
        image_size=(mask.shape[0], mask.shape[1]),
        grid_size=grid_size,
        dtype=patch_features.dtype,
    )
    union_features = _pool_weights(patch_features, union_weights)

    return FrameRegionFeatures(
        scene_feature=scene_feature,
        object_ids=object_ids,
        object_mask_areas=object_areas,
        object_features=object_features,
        pair_ids=pair_ids,
        union_boxes_xyxy=union_boxes,
        union_features=union_features,
    )

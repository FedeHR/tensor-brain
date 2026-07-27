import pytest
import torch

from experiments.pvsg.features import extract_frame_regions, rectangle_patch_weights


def test_objects_are_pooled_from_masks_and_not_boxes() -> None:
    patches = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ]
    )

    regions = extract_frame_regions(torch.tensor([9.0]), patches, mask)

    torch.testing.assert_close(regions.scene_feature, torch.tensor([9.0]))
    torch.testing.assert_close(regions.object_ids, torch.tensor([1, 2]))
    torch.testing.assert_close(regions.object_mask_areas, torch.tensor([8, 8]))
    torch.testing.assert_close(regions.object_features, torch.tensor([[2.0], [3.0]]))


def test_irregular_object_mask_does_not_include_the_rest_of_its_box() -> None:
    patches = torch.tensor(
        [
            [[1.0], [100.0]],
            [[100.0], [3.0]],
        ]
    )
    mask = torch.tensor(
        [
            [7, 7, 0, 0],
            [7, 7, 0, 0],
            [0, 0, 7, 7],
            [0, 0, 7, 7],
        ]
    )

    regions = extract_frame_regions(torch.tensor([0.0]), patches, mask)

    torch.testing.assert_close(regions.object_features, torch.tensor([[2.0]]))
    assert regions.pair_ids.shape == (0, 2)
    assert regions.union_features.shape == (0, 1)


def test_pair_feature_includes_the_space_between_objects() -> None:
    patches = torch.tensor([[[1.0], [10.0], [20.0], [4.0]]])
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 9, 9]])

    regions = extract_frame_regions(torch.tensor([0.0]), patches, mask)

    torch.testing.assert_close(regions.pair_ids, torch.tensor([[1, 9]]))
    torch.testing.assert_close(regions.union_boxes_xyxy, torch.tensor([[0, 0, 8, 1]]))
    torch.testing.assert_close(regions.union_features, torch.tensor([[8.75]]))


def test_union_rows_are_canonical_and_cover_every_visible_pair() -> None:
    patches = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3, 1)
    mask = torch.tensor(
        [
            [8, 0, 2],
            [0, 5, 0],
            [0, 0, 0],
        ]
    )

    regions = extract_frame_regions(torch.tensor([0.0]), patches, mask)

    torch.testing.assert_close(
        regions.pair_ids,
        torch.tensor([[2, 5], [2, 8], [5, 8]]),
    )
    assert bool((regions.pair_ids[:, 0] < regions.pair_ids[:, 1]).all())


def test_rectangle_pooling_accounts_for_partial_patch_overlap() -> None:
    boxes = torch.tensor([[1, 0, 3, 2]])

    weights = rectangle_patch_weights(
        boxes,
        image_size=(2, 4),
        grid_size=(1, 2),
        dtype=torch.float32,
    )

    torch.testing.assert_close(weights, torch.tensor([[[0.5, 0.5]]]))


def test_mask_pooling_is_exact_for_non_divisible_image_size() -> None:
    patches = torch.tensor([[[0.0], [10.0]]])
    mask = torch.tensor([[1, 1, 0]])

    regions = extract_frame_regions(torch.tensor([0.0]), patches, mask)

    # A 3-pixel row split over two patches has cells [0, 1.5] and [1.5, 3].
    # Object 1 therefore covers 1.5 and 0.5 source-pixel units respectively.
    torch.testing.assert_close(regions.object_features, torch.tensor([[2.5]]))


def test_invalid_region_inputs_fail_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="patch_features"):
        extract_frame_regions(
            torch.ones(3), torch.ones(2, 2), torch.ones(2, 2, dtype=torch.long)
        )
    with pytest.raises(ValueError, match="panoptic_mask"):
        extract_frame_regions(torch.ones(3), torch.ones(2, 2, 3), torch.ones(2, 2))
    with pytest.raises(ValueError, match="scene_feature"):
        extract_frame_regions(
            torch.ones(2), torch.ones(2, 2, 3), torch.ones(2, 2, dtype=torch.long)
        )


def test_void_only_frame_has_scene_evidence_and_empty_region_tables() -> None:
    regions = extract_frame_regions(
        torch.full((4,), 3.0),
        torch.ones(2, 3, 4),
        torch.zeros(8, 12, dtype=torch.long),
    )

    torch.testing.assert_close(regions.scene_feature, torch.full((4,), 3.0))
    assert regions.object_ids.shape == (0,)
    assert regions.object_mask_areas.shape == (0,)
    assert regions.object_features.shape == (0, 4)
    assert regions.pair_ids.shape == (0, 2)
    assert regions.union_boxes_xyxy.shape == (0, 4)
    assert regions.union_features.shape == (0, 4)

import pytest
import torch

from experiments.pvsg.extract import patch_aligned_size, split_dinov3_tokens


def test_patch_aligned_size_preserves_full_frame_aspect_approximately() -> None:
    size = patch_aligned_size((360, 480), long_edge=448, patch_size=16)

    assert size == (336, 448)
    assert size[0] % 16 == size[1] % 16 == 0
    assert size[0] / size[1] == pytest.approx(360 / 480)


def test_dinov3_token_split_removes_cls_and_registers() -> None:
    sequence = torch.arange(2 * 11 * 3, dtype=torch.float32).reshape(2, 11, 3)

    scenes, patches = split_dinov3_tokens(
        sequence,
        processed_size=(32, 48),
        patch_size=16,
        num_register_tokens=4,
    )

    torch.testing.assert_close(scenes, sequence[:, 0])
    torch.testing.assert_close(patches.reshape(2, 6, 3), sequence[:, 5:])


def test_dinov3_token_split_rejects_stale_token_assumptions() -> None:
    with pytest.raises(ValueError, match="returned 10 tokens, expected 11"):
        split_dinov3_tokens(
            torch.ones(1, 10, 3),
            processed_size=(32, 48),
            patch_size=16,
            num_register_tokens=4,
        )

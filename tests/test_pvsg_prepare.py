from pathlib import Path

import pytest

from experiments.pvsg.prepare import archive_member_destination


def test_dirty_pvsg_mask_path_keeps_video_directory() -> None:
    destination = archive_member_destination(
        "mnt/lustre/jkyang/CVPR23/openpvsg/data/ego4d/masks/video-1/0007.png",
        source="ego4d",
        kind="masks",
        expected_video_ids={"video-1"},
    )

    assert destination == Path("video-1/0007.png")


def test_epic_video_extension_is_normalized() -> None:
    destination = archive_member_destination(
        "mnt/lustre/jkyang/CVPR23/openpvsg/data/epic_kitchen/videos/P01_03.MP4",
        source="epic_kitchen",
        kind="videos",
        expected_video_ids={"P01_03"},
    )

    assert destination == Path("P01_03.mp4")


@pytest.mark.parametrize(
    "member",
    [
        "/ego4d/masks/video-1/0000.png",
        "ego4d/masks/../video-1/0000.png",
        "ego4d/masks/video-1/0000.jpg",
        "ego4d/masks/unknown/0000.png",
        "ego4d/masks/video-1/nested/0000.png",
    ],
)
def test_unsafe_or_unexpected_archive_members_are_rejected(member: str) -> None:
    with pytest.raises(ValueError):
        archive_member_destination(
            member,
            source="ego4d",
            kind="masks",
            expected_video_ids={"video-1"},
        )

from pathlib import Path

import pytest

from gym_privacy import VideoInput

BASE_DIR = Path(__file__).parent
SAMPLE_VIDEO = BASE_DIR / "data" / "test_video_1s_1080p.mp4"


@pytest.mark.skipif(
    not SAMPLE_VIDEO.is_file(),
    reason="Sample video file not found"
)
def test_video_input_opens_valid_file() -> None:
    video = VideoInput(SAMPLE_VIDEO)

    assert video.path == SAMPLE_VIDEO
    assert video.fps >= 0  # some cameras/videos report 0
    assert video.width > 0
    assert video.height > 0

    ret, frame = video.read_frame()
    assert ret is True
    assert frame is not None
    assert frame.shape[0] == video.height
    assert frame.shape[1] == video.width

    video.release()

def test_video_input_invalid_path_raises() -> None:
    bad_path = Path("data/this_does_not_exist.mp4")
    with pytest.raises(FileNotFoundError):
        VideoInput(bad_path)

def test_video_input_directory_raises() -> None:
    directory = Path(".")
    with pytest.raises(ValueError):
        VideoInput(directory)
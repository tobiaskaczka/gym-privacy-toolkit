from pathlib import Path

import numpy as np
import pytest

from gym_privacy import VideoInput, YuNetFaceDetector

BASE_DIR = Path(__file__).parent
SAMPLE_VIDEO = BASE_DIR / "data" / "test_video_1s_1080p.mp4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


@pytest.mark.skipif(
    not DEFAULT_MODEL.is_file(), 
    reason="YuNet model file not found"
)
def test_yunet_detector_loads_default_model() -> None:
    detector = YuNetFaceDetector()
    assert detector.model_path == DEFAULT_MODEL


@pytest.mark.skipif(
    not SAMPLE_VIDEO.is_file() or not DEFAULT_MODEL.is_file(),
    reason="Sample video or YuNet model file not found",
)
def test_detect_returns_bbox_list_on_real_frame() -> None:
    video = VideoInput(SAMPLE_VIDEO)
    try:
        ret, frame = video.read_frame()
        assert ret is True
        assert frame is not None

        detector = YuNetFaceDetector()
        boxes = detector.detect(frame)

        assert isinstance(boxes, list)
        for box in boxes:
            assert isinstance(box, tuple)
            assert len(box) == 4
            assert all(isinstance(v, int) for v in box)
    finally:
        video.release()


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(), reason="YuNet model file not found")
def test_detect_invalid_frame_raises() -> None:
    detector = YuNetFaceDetector()

    with pytest.raises(ValueError):
        detector.detect(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        detector.detect(np.array([1, 2, 3], dtype=np.uint8))

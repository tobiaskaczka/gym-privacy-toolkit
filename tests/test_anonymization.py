from pathlib import Path

import numpy as np
import pytest

from gym_privacy import BlurFaceAnonymizer, VideoInput, YuNetFaceDetector

BASE_DIR = Path(__file__).parent
SAMPLE_VIDEO = BASE_DIR / "data" / "test_video_1s_1080p.mp4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def test_anonymize_empty_boxes_returns_unchanged_equivalent_frame() -> None:
    anonymizer = BlurFaceAnonymizer()
    frame = np.arange(120 * 160 * 3, dtype=np.uint8).reshape((120, 160, 3))

    output = anonymizer.anonymize(frame, [])

    assert output is not frame
    assert np.array_equal(output, frame)


def test_anonymize_valid_bbox_modifies_roi_and_preserves_shape_dtype() -> None:
    anonymizer = BlurFaceAnonymizer(kernel_size=21)
    rng = np.random.default_rng(123)
    frame = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    box = (20, 30, 60, 50)

    output = anonymizer.anonymize(frame, [box])

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype

    x, y, w, h = box
    original_roi = frame[y : y + h, x : x + w]
    output_roi = output[y : y + h, x : x + w]
    assert not np.array_equal(output_roi, original_roi)


def test_anonymize_multiple_boxes_processes_all_regions() -> None:
    anonymizer = BlurFaceAnonymizer(kernel_size=19)
    rng = np.random.default_rng(99)
    frame = rng.integers(0, 256, size=(100, 140, 3), dtype=np.uint8)
    boxes = [(10, 10, 30, 30), (70, 40, 40, 40)]

    output = anonymizer.anonymize(frame, boxes)

    for x, y, w, h in boxes:
        original_roi = frame[y : y + h, x : x + w]
        output_roi = output[y : y + h, x : x + w]
        assert not np.array_equal(output_roi, original_roi)

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype


def test_anonymize_does_not_modify_pixels_outside_boxes() -> None:
    anonymizer = BlurFaceAnonymizer(kernel_size=21)
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(90, 120, 3), dtype=np.uint8)
    box = (30, 20, 40, 30)

    output = anonymizer.anonymize(frame, [box])

    x, y, w, h = box
    # Outside the bbox: top strip
    assert np.array_equal(output[:y, :, :], frame[:y, :, :])
    # Outside the bbox: bottom strip
    assert np.array_equal(output[y + h :, :, :], frame[y + h :, :, :])
    # Outside the bbox: left strip (same y-range as bbox)
    assert np.array_equal(output[y : y + h, :x, :], frame[y : y + h, :x, :])
    # Outside the bbox: right strip (same y-range as bbox)
    assert np.array_equal(output[y : y + h, x + w :, :], frame[y : y + h, x + w :, :])


def test_anonymize_invalid_frame_raises() -> None:
    anonymizer = BlurFaceAnonymizer()

    with pytest.raises(ValueError):
        anonymizer.anonymize(None, [])  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        anonymizer.anonymize(np.array([1, 2, 3], dtype=np.uint8), [])


@pytest.mark.skipif(
    not SAMPLE_VIDEO.is_file() or not DEFAULT_MODEL.is_file(),
    reason="Sample video or YuNet model file not found",
)
def test_blur_anonymizer_smoke_with_yunet_detections() -> None:
    video = VideoInput(SAMPLE_VIDEO)
    try:
        ret, frame = video.read_frame()
        assert ret is True
        assert frame is not None

        detector = YuNetFaceDetector()
        boxes = detector.detect(frame)

        anonymizer = BlurFaceAnonymizer()
        output = anonymizer.anonymize(frame, boxes)

        assert output.shape == frame.shape
        assert output.dtype == frame.dtype
    finally:
        video.release()

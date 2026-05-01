from pathlib import Path

import numpy as np
import pytest

from gym_privacy import BBox, BlurFaceAnonymizer, VideoInput, YuNetFaceDetector

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
    bbox = BBox.from_xywh(20, 30, 60, 50)

    output = anonymizer.anonymize(frame, [bbox])

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype

    original_roi = frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    output_roi = output[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    assert not np.array_equal(output_roi, original_roi)


def test_anonymize_multiple_boxes_processes_all_regions() -> None:
    anonymizer = BlurFaceAnonymizer(kernel_size=19)
    rng = np.random.default_rng(99)
    frame = rng.integers(0, 256, size=(100, 140, 3), dtype=np.uint8)
    bboxes = [BBox.from_xywh(10, 10, 30, 30), BBox.from_xywh(70, 40, 40, 40)]

    output = anonymizer.anonymize(frame, bboxes)

    for bbox in bboxes:
        original_roi = frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        output_roi = output[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        assert not np.array_equal(output_roi, original_roi)

    assert output.shape == frame.shape
    assert output.dtype == frame.dtype


def test_anonymize_does_not_modify_pixels_outside_boxes() -> None:
    anonymizer = BlurFaceAnonymizer(kernel_size=21)
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(90, 120, 3), dtype=np.uint8)
    bbox = BBox.from_xywh(30, 20, 40, 30)

    output = anonymizer.anonymize(frame, [bbox])

    # Outside the bbox: top strip
    assert np.array_equal(output[: bbox.y1, :, :], frame[: bbox.y1, :, :])
    # Outside the bbox: bottom strip
    assert np.array_equal(output[bbox.y2 :, :, :], frame[bbox.y2 :, :, :])
    # Outside the bbox: left strip (same y-range as bbox)
    assert np.array_equal(output[bbox.y1 : bbox.y2, : bbox.x1, :], frame[bbox.y1 : bbox.y2, : bbox.x1, :])
    # Outside the bbox: right strip (same y-range as bbox)
    assert np.array_equal(output[bbox.y1 : bbox.y2, bbox.x2 :, :], frame[bbox.y1 : bbox.y2, bbox.x2 :, :])


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
        bboxes = detector.detect(frame)

        anonymizer = BlurFaceAnonymizer()
        output = anonymizer.anonymize(frame, bboxes)

        assert output.shape == frame.shape
        assert output.dtype == frame.dtype
    finally:
        video.release()

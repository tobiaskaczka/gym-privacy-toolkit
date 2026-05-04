from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from gym_privacy import BBox, DetectedFace, FaceDetector, Point, VideoInput, YuNetFaceDetector

BASE_DIR = Path(__file__).parent
SAMPLE_VIDEO = BASE_DIR / "data" / "test_video_1s_1080p.mp4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


class BBoxOnlyDetector(FaceDetector):
    def __init__(self) -> None:
        super().__init__("box-only-model")

    def detect_faces(self, frame: np.ndarray) -> list[DetectedFace]:
        return [DetectedFace(bbox=BBox.from_xywh(1, 2, 3, 4))]


@pytest.mark.skipif(
    not DEFAULT_MODEL.is_file(), 
    reason="YuNet model file not found"
)
def test_yunet_detector_loads_default_model() -> None:
    detector = YuNetFaceDetector()
    assert detector.model_path == DEFAULT_MODEL


def test_base_detector_detect_returns_boxes_from_detected_faces() -> None:
    detector = BBoxOnlyDetector()

    bboxes = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert bboxes == [BBox(top_left=Point(1, 2), bottom_right=Point(4, 6))]


def test_bbox_from_xywh_converts_to_corner_coordinates() -> None:
    bbox = BBox.from_xywh(10, 20, 30, 40)

    assert bbox.top_left == Point(10, 20)
    assert bbox.bottom_right == Point(40, 60)
    assert bbox.x1 == 10
    assert bbox.y1 == 20
    assert bbox.x2 == 40
    assert bbox.y2 == 60
    assert bbox.width == 30
    assert bbox.height == 40


def test_bbox_is_immutable() -> None:
    bbox = BBox.from_xywh(10, 20, 30, 40)

    with pytest.raises(FrozenInstanceError):
        bbox.top_left = Point(0, 0)  # type: ignore[misc]


def test_bbox_rejects_negative_dimensions() -> None:
    with pytest.raises(ValueError):
        BBox.from_xywh(10, 20, -1, 40)

    with pytest.raises(ValueError):
        BBox.from_xywh(10, 20, 30, -1)


def test_detected_face_supports_optional_metadata() -> None:
    bbox = BBox.from_xywh(1, 2, 3, 4)
    face = DetectedFace(bbox=bbox)

    assert face.bbox == bbox
    assert face.landmarks is None
    assert face.score is None


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
        bboxes = detector.detect(frame)

        assert isinstance(bboxes, list)
        for bbox in bboxes:
            assert isinstance(bbox, BBox)
            assert isinstance(bbox.top_left, Point)
            assert isinstance(bbox.bottom_right, Point)
            assert all(isinstance(v, int) for v in [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            assert bbox.width >= 0
            assert bbox.height >= 0
    finally:
        video.release()


@pytest.mark.skipif(
    not SAMPLE_VIDEO.is_file() or not DEFAULT_MODEL.is_file(),
    reason="Sample video or YuNet model file not found",
)
def test_detect_faces_returns_landmarks_and_scores_on_real_frame() -> None:
    video = VideoInput(SAMPLE_VIDEO)
    try:
        ret, frame = video.read_frame()
        assert ret is True
        assert frame is not None

        detector = YuNetFaceDetector()
        faces = detector.detect_faces(frame)

        assert isinstance(faces, list)
        for face in faces:
            assert isinstance(face, DetectedFace)
            assert isinstance(face.bbox, BBox)
            assert all(isinstance(v, int) for v in [face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2])
            assert face.landmarks is not None
            assert len(face.landmarks) == 5
            for point in face.landmarks:
                assert isinstance(point, Point)
                assert isinstance(point.x, int)
                assert isinstance(point.y, int)
            assert face.score is not None
            assert isinstance(face.score, float)
    finally:
        video.release()


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(), reason="YuNet model file not found")
def test_detect_invalid_frame_raises() -> None:
    detector = YuNetFaceDetector()

    with pytest.raises(ValueError):
        detector.detect(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        detector.detect(np.array([1, 2, 3], dtype=np.uint8))


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(), reason="YuNet model file not found")
def test_detect_faces_invalid_frame_raises() -> None:
    detector = YuNetFaceDetector()

    with pytest.raises(ValueError):
        detector.detect_faces(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        detector.detect_faces(np.array([1, 2, 3], dtype=np.uint8))

from pathlib import Path

import numpy as np
import pytest

from gym_privacy import DetectedFace, FaceDetector, VideoInput, YuNetFaceDetector

BASE_DIR = Path(__file__).parent
SAMPLE_VIDEO = BASE_DIR / "data" / "test_video_1s_1080p.mp4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"


class BoxOnlyDetector(FaceDetector):
    def __init__(self) -> None:
        super().__init__("box-only-model")

    def detect_faces(self, frame: np.ndarray) -> list[DetectedFace]:
        return [DetectedFace(box=(1, 2, 3, 4))]


@pytest.mark.skipif(
    not DEFAULT_MODEL.is_file(), 
    reason="YuNet model file not found"
)
def test_yunet_detector_loads_default_model() -> None:
    detector = YuNetFaceDetector()
    assert detector.model_path == DEFAULT_MODEL


def test_base_detector_detect_returns_boxes_from_detected_faces() -> None:
    detector = BoxOnlyDetector()

    boxes = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

    assert boxes == [(1, 2, 3, 4)]


def test_detected_face_supports_optional_metadata() -> None:
    face = DetectedFace(box=(1, 2, 3, 4))

    assert face.box == (1, 2, 3, 4)
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
        boxes = detector.detect(frame)

        assert isinstance(boxes, list)
        for box in boxes:
            assert isinstance(box, tuple)
            assert len(box) == 4
            assert all(isinstance(v, int) for v in box)
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
            assert len(face.box) == 4
            assert all(isinstance(v, int) for v in face.box)
            assert face.landmarks is not None
            assert len(face.landmarks) == 5
            for point in face.landmarks:
                assert len(point) == 2
                assert all(isinstance(v, int) for v in point)
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

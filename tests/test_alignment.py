import numpy as np
import pytest

from gym_privacy import ArcFaceAligner, BBox, DetectedFace, Point


def test_arcface_align_returns_112_face_by_default() -> None:
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    face = DetectedFace(
        bbox=BBox.from_xywh(20, 20, 90, 90),
        landmarks=(
            Point(38, 52),
            Point(74, 52),
            Point(56, 72),
            Point(42, 92),
            Point(71, 92),
        ),
    )

    aligned = ArcFaceAligner().align(frame, face)

    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == frame.dtype


def test_arcface_align_uses_configured_image_size() -> None:
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    face = DetectedFace(
        bbox=BBox.from_xywh(20, 20, 90, 90),
        landmarks=(
            Point(38, 52),
            Point(74, 52),
            Point(56, 72),
            Point(42, 92),
            Point(71, 92),
        ),
    )

    aligned = ArcFaceAligner(image_size=224).align(frame, face)

    assert aligned.shape == (224, 224, 3)


def test_arcface_align_rejects_unsupported_image_size() -> None:
    with pytest.raises(ValueError, match="image_size"):
        ArcFaceAligner(image_size=100)

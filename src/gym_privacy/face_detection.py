from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BBox:
    """Pixel-space bbox using top-left and bottom-right edge coordinates."""

    top_left: Point
    bottom_right: Point

    def __post_init__(self) -> None:
        if self.bottom_right.x < self.top_left.x:
            raise ValueError("bbox bottom_right.x must be greater than or equal to top_left.x")
        if self.bottom_right.y < self.top_left.y:
            raise ValueError("bbox bottom_right.y must be greater than or equal to top_left.y")

    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int) -> "BBox":
        return cls(
            top_left=Point(x, y),
            bottom_right=Point(x + width, y + height),
        )

    @property
    def x1(self) -> int:
        return self.top_left.x

    @property
    def y1(self) -> int:
        return self.top_left.y

    @property
    def x2(self) -> int:
        return self.bottom_right.x

    @property
    def y2(self) -> int:
        return self.bottom_right.y

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class DetectedFace:
    bbox: BBox
    landmarks: tuple[Point, ...] | None = None
    score: float | None = None


class FaceDetector(ABC):

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """Return detected face bounding boxes for a frame."""
        return [face.bbox for face in self.detect_faces(frame)]

    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> list[DetectedFace]:
        """Return detected faces with any backend-specific metadata."""
        raise NotImplementedError


class YuNetFaceDetector(FaceDetector):

    def __init__(self, model_path: str | Path | None = None, score_threshold: float = 0.6) -> None:
        if model_path is None:
            project_root = Path(__file__).resolve().parents[2]
            model_path = project_root / "models" / "face_detection_yunet_2023mar.onnx"
        super().__init__(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"YuNet model file not found: {self.model_path}")

        # Placeholder input size, updated for each frame.
        self.det_res = (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            self.det_res,
            score_threshold,  # confidence threshold
            0.5,  # nms threshold
            5000,  # top_k
        )
        self.detector.setInputSize(self.det_res)

    def detect_faces(self, frame: np.ndarray) -> list[DetectedFace]:
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("frame must be a valid numpy image array")

        if frame.ndim < 2:
            raise ValueError("frame must have at least 2 dimensions (H, W[, C])")

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("frame height and width must be positive")

        # Keep detector input size with current frame.
        if self.det_res != (w, h):
            self.det_res = (w, h)
            self.detector.setInputSize(self.det_res)

        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        detected_faces: list[DetectedFace] = []
        for face in faces:
            x, y, bw, bh = face[:4]
            landmarks = (
                Point(int(face[4]), int(face[5])),
                Point(int(face[6]), int(face[7])),
                Point(int(face[8]), int(face[9])),
                Point(int(face[10]), int(face[11])),
                Point(int(face[12]), int(face[13])),
            )
            detected_faces.append(
                DetectedFace(
                    bbox=BBox.from_xywh(int(x), int(y), int(bw), int(bh)),
                    landmarks=landmarks,
                    score=float(face[14]),
                )
            )
        return detected_faces

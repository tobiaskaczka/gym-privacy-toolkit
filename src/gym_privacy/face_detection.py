from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Bounding box (x, y, w, h)
BBox = tuple[int, int, int, int]
Point = tuple[int, int]


@dataclass(frozen=True)
class DetectedFace:
    box: BBox
    landmarks: tuple[Point, ...] | None = None
    score: float | None = None


class FaceDetector(ABC):

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def detect(self, frame: np.ndarray) -> list[BBox]:
        """Return detected face bounding boxes for a frame."""
        return [face.box for face in self.detect_faces(frame)]

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
                (int(face[4]), int(face[5])),
                (int(face[6]), int(face[7])),
                (int(face[8]), int(face[9])),
                (int(face[10]), int(face[11])),
                (int(face[12]), int(face[13])),
            )
            detected_faces.append(
                DetectedFace(
                    box=(int(x), int(y), int(bw), int(bh)),
                    landmarks=landmarks,
                    score=float(face[14]),
                )
            )
        return detected_faces

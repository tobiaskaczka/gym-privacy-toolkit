from abc import ABC, abstractmethod

import cv2
import numpy as np

from .face_detection import BBox


class FaceAnonymizer(ABC):
    @abstractmethod
    def anonymize(self, frame: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        """Return a frame with anonymization applied to all given face boxes."""
        raise NotImplementedError


class BlurFaceAnonymizer(FaceAnonymizer):
    def __init__(self, kernel_size: int = 51) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = kernel_size

    def anonymize(self, frame: np.ndarray, boxes: list[BBox]) -> np.ndarray:
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("frame must be a valid numpy image array")

        if frame.ndim < 2:
            raise ValueError("frame must have at least 2 dimensions (H, W[, C])")

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("frame height and width must be positive")

        output = frame.copy()
        for x, y, bw, bh in boxes:
            x2 = x + bw
            y2 = y + bh
            roi = output[y:y2, x:x2]
            if roi.size == 0:
                continue

            kx, ky = self._kernel_for_roi(roi.shape[1], roi.shape[0])
            blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
            output[y:y2, x:x2] = blurred

        return output

    def _kernel_for_roi(self, roi_w: int, roi_h: int) -> tuple[int, int]:
        # Gaussian kernel dimensions must be positive odd values.
        max_kx = roi_w if roi_w % 2 == 1 else roi_w - 1
        max_ky = roi_h if roi_h % 2 == 1 else roi_h - 1
        kx = max(1, min(self.kernel_size, max_kx))
        ky = max(1, min(self.kernel_size, max_ky))
        return kx, ky
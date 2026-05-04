from abc import ABC, abstractmethod

import cv2
import numpy as np

from .detection import DetectedFace, Point


class FaceAligner(ABC):
    @abstractmethod
    def align(self, frame: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return an aligned face crop for recognition."""
        raise NotImplementedError


class ArcFaceAligner(FaceAligner):
    _REF_KPS = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self, image_size: int = 112) -> None:
        if image_size % 112 != 0 and image_size % 128 != 0:
            raise ValueError("image_size must be divisible by 112 or 128")
        self.image_size = image_size

    def align(self, frame: np.ndarray, face: DetectedFace) -> np.ndarray:

        # Ensure the input landmarks have exactly 5 points (as expected for face alignment)
        if len(face.landmarks) != 5:
            raise ValueError("Expected exactly 5 landmarks for alignment")

        # Validate that image_size is divisible by either 112 or 128 (common image sizes for face recognition models)
        if self.image_size % 112 != 0 and self.image_size % 128 != 0:
            raise ValueError("image_size must be divisible by 112 or 128")

        # Adjust the scaling factor (ratio) based on the desired image size (112 or 128)
        if self.image_size % 112 == 0:
            ratio = float(self.image_size) / 112.0
            diff_x = 0.0  # No horizontal shift for 112 scaling
        else:
            ratio = float(self.image_size) / 128.0
            diff_x = 8.0 * ratio  # Horizontal shift for 128 scaling

        # Apply the scaling and shifting to the reference keypoints
        dst = self._REF_KPS * ratio
        dst[:, 0] += diff_x  # Apply the horizontal shift

        # Estimate the similarity transformation matrix to align the landmarks with the reference keypoints
        src = np.array([point.as_tuple() for point in face.landmarks], dtype=np.float32)
        transform, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=1000)
        if transform is None or inliers is None or not np.all(inliers):
            raise ValueError("could not estimate face alignment transform")

        # Apply the affine transformation to the input image to align the face
        return cv2.warpAffine(
            frame,
            transform,
            (self.image_size, self.image_size),
            borderValue=0.0,
        )

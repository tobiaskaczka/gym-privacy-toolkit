import cv2
from pathlib import Path
from numpy import ndarray

class VideoInput:

    def __init__(self, source: str | Path) -> None:

        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        if not path.is_file():
            raise ValueError(f"Source must be a file, not a directory: {path}")

        self.path = path
        self.cap = cv2.VideoCapture(str(path))

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def read_frame(self) -> tuple[bool, ndarray]:
        """Reads a single frame from video"""
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        """Releases the video cap"""
        self.cap.release()

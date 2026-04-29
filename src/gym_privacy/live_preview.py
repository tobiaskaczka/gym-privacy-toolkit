import argparse

import cv2
import numpy as np

from gym_privacy import BlurFaceAnonymizer, DetectedFace, YuNetFaceDetector
from gym_privacy.face_detection import BBox

WINDOW_NAME = "Gym Privacy Live Preview"
MODES = ["raw", "detect", "landmarks", "blur"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live camera preview for face pipeline stages.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index to read from.")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="detect",
        help="Pipeline stage to preview.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.6, help="YuNet confidence threshold.")
    parser.add_argument("--kernel-size", type=int, default=51, help="Gaussian blur kernel size (odd positive integer).")
    return parser.parse_args()


def draw_boxes(frame: np.ndarray, boxes: list[BBox]) -> np.ndarray:
    preview = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return preview


def draw_faces_with_landmarks(frame: np.ndarray, faces: list[DetectedFace]) -> np.ndarray:
    preview = frame.copy()
    for face in faces:
        x, y, w, h = face.box
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if face.score is not None:
            cv2.putText(
                preview,
                f"{face.score:.2f}",
                (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if face.landmarks is None:
            continue

        for point in face.landmarks:
            cv2.circle(preview, point, 3, (0, 0, 255), -1)
    return preview


def render_preview(
    frame: np.ndarray,
    mode: str,
    detector: YuNetFaceDetector,
    anonymizer: BlurFaceAnonymizer,
) -> np.ndarray:
    if mode == "raw":
        return frame

    if mode == "detect":
        boxes = detector.detect(frame)
        return draw_boxes(frame, boxes)

    if mode == "landmarks":
        faces = detector.detect_faces(frame)
        return draw_faces_with_landmarks(frame, faces)

    if mode == "blur":
        boxes = detector.detect(frame)
        return anonymizer.anonymize(frame, boxes)

    raise ValueError(f"Unknown preview mode: {mode}")


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index: {args.camera_index}")

    detector = YuNetFaceDetector(score_threshold=args.score_threshold)
    anonymizer = BlurFaceAnonymizer(kernel_size=args.kernel_size)

    print("Starting live preview")
    print(f"Camera index: {args.camera_index}")
    print(f"Mode: {args.mode}")
    print("Keys: 1 raw | 2 detect | 3 landmarks | 4 blur | q quit")

    mode = args.mode
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            preview = render_preview(frame, mode, detector, anonymizer)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
                mode = MODES[key - ord("1")]
                print(f"Mode: {mode}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

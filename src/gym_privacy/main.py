import argparse
from pathlib import Path

import cv2

from gym_privacy import BlurFaceAnonymizer, VideoInput, YuNetFaceDetector


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    default_input = project_root / "tests" / "data" / "test_video_1s_1080p.mp4"
    default_output = project_root / "output" / "processed_blurred.mp4"

    parser = argparse.ArgumentParser(description="Run gym privacy blur pipeline.")
    parser.add_argument("--input", type=Path, default=default_input, help="Input video path.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output video path.")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional YuNet ONNX model path.")
    parser.add_argument("--score-threshold", type=float, default=0.6, help="YuNet confidence threshold.")
    parser.add_argument("--kernel-size", type=int, default=51, help="Gaussian blur kernel size (odd positive integer).")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames to process.")
    parser.add_argument(
        "--debug-previews",
        action="store_true",
        help="Save first-frame debug side-by-side image (original | blurred).",
    )
    return parser.parse_args()


def save_debug_previews(output_path: Path, original_frame, blurred_frame) -> None:
    output_dir = output_path.parent
    stem = output_path.stem

    side_by_side_path = output_dir / f"{stem}_frame0_side_by_side.jpg"

    side_by_side = cv2.hconcat([original_frame, blurred_frame])
    cv2.imwrite(str(side_by_side_path), side_by_side)

    print("Saved debug preview:")
    print(f"  - {side_by_side_path}")


def main() -> None:
    args = parse_args()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video = VideoInput(args.input)
    detector = YuNetFaceDetector(model_path=args.model_path, score_threshold=args.score_threshold)
    anonymizer = BlurFaceAnonymizer(kernel_size=args.kernel_size)

    fps = video.fps if video.fps and video.fps > 0 else 30.0
    width = int(video.width)
    height = int(video.height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Could not open video writer for output path: {output_path}")

    print("Starting pipeline run")
    print(f"Input:  {video.path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps:.2f} | Frame size: {width}x{height}")

    frame_count = 0
    total_faces = 0

    try:
        while True:
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            ret, frame = video.read_frame()
            if not ret:
                break

            bboxes = detector.detect(frame)
            blurred_frame = anonymizer.anonymize(frame, bboxes)
            writer.write(blurred_frame)

            if args.debug_previews and frame_count == 0:
                save_debug_previews(output_path, frame, blurred_frame)

            total_faces += len(bboxes)
            frame_count += 1
    finally:
        writer.release()
        video.release()

    print("Run complete")
    print(f"Frames processed: {frame_count}")
    print(f"Total faces detected: {total_faces}")
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()


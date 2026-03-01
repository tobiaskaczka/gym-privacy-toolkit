from pathlib import Path
from gym_privacy import VideoInput

def main() -> None:

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    video_path = PROJECT_ROOT / "tests" / "data" / "test_video_1s_1080p.mp4"
    video = VideoInput(video_path)

    print(f"Path:   {video.path}")
    print(f"FPS:    {video.fps}")
    print(f"Width:  {video.width}")
    print(f"Height: {video.height}")

    # reading a few frames
    for i in range(5):
        ret, frame = video.read_frame()
        if not ret:
            print(f"Stopped at frame {i}: no more frames or read failed.")
            break
        print(f"Frame {i}: shape={frame.shape}")

    video.release()


if __name__ == "__main__":
    main()

from pathlib import Path

from video_input import VideoInput  


def main() -> None:
    video_path = Path("data/sample.mp4")  # point this at a real file on disk

    video = VideoInput(video_path)

    print(f"Path:   {video.path}")
    print(f"FPS:    {video.fps}")
    print(f"Width:  {video.width}")
    print(f"Height: {video.height}")

    # Try reading a few frames
    for i in range(5):
        ret, frame = video.read_frame()
        if not ret:
            print(f"Stopped at frame {i}: no more frames or read failed.")
            break
        print(f"Frame {i}: shape={frame.shape}")

    video.release()


if __name__ == "__main__":
    main()

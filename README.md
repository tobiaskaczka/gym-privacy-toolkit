# Gym Privacy Toolkit

Python toolkit for privacy-preserving gym footage processing.

Current focus:
- read video files,
- detect faces with YuNet locally,
- anonymize detected faces with blur,
- keep the pipeline modular and backend swappable.

## Current status
- `VideoInput` is implemented and tested.
- `YuNetFaceDetector` is implemented and tested.
- `BlurFaceAnonymizer` is implemented and tested.
- `main.py` runs an end-to-end pass and writes a processed output video.

## Quick start
```bash
poetry install
poetry run python -m gym_privacy.main
```
Default output:
- `output/processed_blurred.mp4`

## Main CLI examples
Run full sample video:
```bash
poetry run python -m gym_privacy.main
```

Run with custom output name:
```bash
poetry run python -m gym_privacy.main --output output/session1_blurred.mp4
```

Save first-frame side-by-side debug preview (`original | blurred`):
```bash
poetry run python -m gym_privacy.main --debug-previews
```

Run only first 30 frames:
```bash
poetry run python -m gym_privacy.main --max-frames 30
```

## Run tests
```bash
poetry run pytest -q
```

## Project direction
Target pipeline:
1. Input video
2. Detect faces
3. Anonymize faces
4. Write processed output video

This is an in-progress learning project.

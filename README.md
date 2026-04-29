# Gym Privacy Toolkit

Python toolkit for privacy-preserving gym footage processing.

Current focus:
- read video files,
- detect faces with YuNet locally,
- anonymize detected faces with blur,
- keep the pipeline modular and backend swappable.
- future pipeline: detecting faces, aligning them for consistency, extracting unique facial features (embeddings), comparing those features against a database of known individuals, then deciding which faces to blur.

## Current status
- `VideoInput` is implemented and tested.
- `YuNetFaceDetector` is implemented and tested.
- `BlurFaceAnonymizer` is implemented and tested.
- `main.py` runs an end-to-end pass and writes a processed output video.

## New device setup
This project uses Python 3.13+ and Poetry.

Install Poetry with `pipx`:
```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```

## Quick start
```bash
poetry install
poetry run python -m gym_privacy.main
```

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

## Live preview
Use a webcam or phone webcam app such as DroidCam to test the pipeline live:
```bash
poetry run python -m gym_privacy.live_preview
```

Preview modes:
- `raw`: camera feed only
- `detect`: draw face boxes
- `landmarks`: draw face boxes, landmark dots, and detection score
- `blur`: blur detected faces

Switch modes while the preview is running:
- `1`: raw
- `2`: detect
- `3`: landmarks
- `4`: blur
- `q`: quit

Useful options:
```bash
poetry run python -m gym_privacy.live_preview --camera-index 1
poetry run python -m gym_privacy.live_preview --mode landmarks --score-threshold 0.85
poetry run python -m gym_privacy.live_preview --mode blur --kernel-size 71
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

Future pipeline direction:
1. Detect faces in each frame.
2. Align detected faces for consistent recognition input.
3. Extract unique facial features as embeddings.
4. Compare embeddings against a database of known individuals.
5. Blur faces based on recognition results.

This is an in-progress learning project.

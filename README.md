# Face Recognition

Face recognition with a jarvis-style dot overlay. Uses OpenCV only — **YuNet** for
face detection, **SFace** for deep-learned face embeddings. No dlib, no cmake.

## Setup

```
pip install -r requirements.txt
python setup_models.py      # downloads YuNet + SFace ONNX models once
```

`opencv-contrib-python` includes everything the code needs; the two model files
are downloaded automatically from the OpenCV model zoo.

## Register a person

```
python register_face.py John john.jpg
```

Use a clear front-facing photo. Register the same person multiple times with
different photos (angles/lighting) — each call re-embeds all of that person's
photos and stores the mean embedding, improving accuracy.

## Run

```
python main.py                                   # webcam
python main.py --source photo.jpg --output result.jpg
python main.py --source video.mp4 --output result.mp4
```

Known face = name + cyan box + jarvis dots + match %. Unknown = red box.
Press `q` to quit (webcam/video mode).

## Register from an Indonesian ID card (KTP)

```
python ktp_register.py 3273011501900001 ktp_scan.jpg
```

Pipeline: detects the card corners, warps it flat (perspective correction),
extracts the card holder's face, and registers it under the holder's **NIK**
(validated as exactly 16 digits). The card's face area is preferred; if the
cascade detects nothing there it falls back to the largest face on the card.

## Verify live against a registered NIK (1:1)

```
python verify.py 3273011501900001                 # webcam
python verify.py 3273011501900001 --source cam.mp4
python verify.py 3273011501900001 --source photo.jpg   # static image, liveness will fail
```

The identity must keep matching the registered NIK for ~2s **while some facial
movement is detected** (nose-tracking), so a printed photo of a KTP cannot pass.
Tune strictness with `--threshold` (default 0.45). Options: `--no-show` for
headless runs, press `q` to quit.

## Reuse for a new client

1. Copy this whole folder.
2. Delete everything inside `data/`.
3. Register the new client's people.
4. Done — same code, new data.

## Privacy (PDP / GDPR)

NIK and face templates are **personal data**. This project keeps everything
local: embeddings are stored only under `data/` (which is `.gitignore`d) and
nothing ever leaves your machine. When publishing, audits, or sharing screenshots,
mask NIKs (the repo masks them in its own display). Delete `data/` before handing
the repo out to a new client.

## Notes

- `COSINE_THRESHOLD` in main.py (0.5) = strictness. Similarity is a cosine
  score in 0..1 — higher threshold = stricter match. Pass `--threshold 0.7`
  on the CLI (main.py and verify.py) instead of editing code.
- Accuracy improves a lot with 3-5 photos per person vs just 1.
- YuNet + SFace handle odd angles and low light far better than a Haar/LBPH
  pipeline, and need only a few MB of ONNX models downloaded once.
- Multi-sample registrations store the L2-normalized mean embedding, so adding
  more photos never biases the similarity score.
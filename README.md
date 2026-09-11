# Face Recognition

Face recognition with a jarvis-style dot overlay. Uses OpenCV only — **YuNet** for
face detection and **AdaFace** (ONNX, MIT license) for 512-d face embeddings.
No dlib, no cmake.

## Setup

```
pip install -r requirements.txt
python setup_models.py      # downloads YuNet + AdaFace ONNX models once
```

`opencv-contrib-python` handles detection; `onnxruntime` runs the **AdaFace IR-101**
embedding model (MIT-licensed, free to use commercially). The model files are
downloaded automatically: YuNet from the OpenCV model zoo and AdaFace from the
[adaface-onnx release](https://github.com/yakhyo/adaface-onnx).

## Upgrading from the old SFace embeddings

Data registered with the previous SFace-based embedding is **not compatible**.
`verify.py` refuses mismatched registrations with a "re-register" hint — delete
`data/` and register everyone again once.

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

### Auto-read NIK / name with OCR (optional)

If Tesseract is installed, `--ocr` reads the NIK (and best-effort name) straight
from the KTP photo so you don't have to type it:

```
python ktp_register.py --ocr ktp_scan.jpg
```

`python ktp_ocr.py ktp_scan.jpg` prints the raw extracted fields for inspection.
Tesseract is detected automatically on Windows/Linux; override with
`--tesseract-cmd` / `--tessdata-dir`. Indonesian (`ind`) is used when the
traineddata file is available, otherwise English. NIK digits are re-OCR'd with a
digits-only whitelist for reliability — the 16-digit NIK is the critical field.

## Verify live against a registered NIK (1:1)

```
python verify.py 3273011501900001                 # webcam
python verify.py 3273011501900001 --source cam.mp4
python verify.py 3273011501900001 --source photo.jpg   # static image, liveness will fail
```

The identity must keep matching the registered NIK for ~3 s **while a blink is
detected** (eye-area brightness impulse while the face is steady), so a printed
photo of a KTP cannot pass. For cameras with poor eyelid detail, `--motion-only`
falls back to nose-tracking movement instead. Natural blinks don't break the
streak — brief interruptions only decay it, while a clearly different person
hard-resets it. Tune strictness with `--threshold` (default 0.45). Options:
`--no-show` for headless runs, press `q` to quit.

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
  score (~1 for the same person, below 0 for different people) — higher
  threshold = stricter match. Pass `--threshold 0.7`
  on the CLI (main.py and verify.py) instead of editing code.
- Accuracy improves a lot with 3-5 photos per person vs just 1.
- YuNet + AdaFace handle odd angles and low light far better than a Haar/LBPH
  pipeline, and need only a few hundred MB of ONNX models downloaded once
  (AdaFace IR-101 is ~260 MB; the faster IR-18 variant is a drop-in swap).
- Multi-sample registrations store the L2-normalized mean embedding, so adding
  more photos never biases the similarity score.
- Blink liveness is a heuristic (eye-area brightness impulse while the face is
  steady); on hardware where YuNet can't track the eyes it may need
  `--motion-only`. Liveness checks defend against still photos, not 3D masks.

## Tests

```
pip install -r requirements.txt pytest pytesseract
# apt/dnf/winget install tesseract-ocr   # optional — OCR tests skip if missing
python -m pytest
```

The suite covers detector/embedder math, KTP dewarp + face extraction, register-
and-verify flows (blink liveness, motion-only, impostor rejection), and OCR NIK
parsing. CI runs it on Ubuntu via `.github/workflows/ci.yml`.
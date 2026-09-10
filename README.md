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

## Alur end-to-end

```mermaid
flowchart TD
    subgraph SETUP["Setup (sekali)"]
        M[python setup_models.py] --> MR[(models/ YuNet + SFace .onnx)]
    end

    subgraph REG["Registrasi dari KTP (ktp_register.py)"]
        A[Foto / scan KTP] --> D[Deteksi sudut kartu + dewarp perspektif]
        D --> E[Extract daerah wajah pemegang kartu]
        A -.opsional.- O[ktp_ocr.py: OCR NIK 16 digit + nama]
        O --> N{valid? 16 digit}
        N -- ya --> E
        N -- tidak --> X1[Tolak registrasi]
        E --> F[YuNet: deteksi wajah]
        F --> G[alignCrop + embed SFace]
        G --> I[mean_embedding semua sampel NIK]
        I --> ST[(data/faces/&lt;NIK&gt;/*.jpg<br/>data/embeddings.json)]
    end

    subgraph VER["Verifikasi 1:1 (verify.py)"]
        V[Kamera / video / foto] --> VF[YuNet detect per frame]
        VF --> VA[align + embed SFace]
        VA --> VC{cosine sim >= threshold?}
        VC -- ya --> VS[streak match bertambah]
        VC -- tidak --> VD[streak decay<br/>hard reset bila sim jauh / gap panjang]
        VS --> L{Liveness}
        L -- default --> LB{ada blink<br/>(impuls kecerahan area mata)}
        L -- "--motion-only" --> LM{pergerakan hidung cukup?}
        LB -- ya --> S{streak >= ~3 detik?}
        LM -- ya --> S
        S -- ya --> PASS([VERIFIED OK])
        S -- tidak --> VF
        LB -- tidak --> VF
        ST -. target NIK .-> VC
        MR -. model .-> VF
    end
```

Versi ASCII ringkas untuk dibaca cepat:

```
  Setup:   setup_models.py ──► models/*.onnx (YuNet + SFace)
                    │
                    ▼
  Registrasi: KTP foto ──► dewarp ──► ambil wajah ──► YuNet ──► SFace embed
                    │                                        │
                    ├─ opsional ktp_ocr.py (--ocr)           │
                    │        └─► NIK 16 digit + nama         │
                    ▼                                        ▼
              data/embeddings.json ◄── mean_embedding ◄── data/faces/<NIK>/
                    │
                    ▼
  Verifikasi: kamera/video/foto ──► YuNet ──► SFace ──► cosine vs NIK target
                    │                                    │
                    │                         cocok ─────► streak bertambah
                    │                         tidak ─────► streak decay / reset
                    │                                    │
                    ▼                                    ▼
              liveness: blink (default) ──► butuh blink >= 1
                        --motion-only ────► butuh gerak cukup
                    │
                    ▼
              streak >= ~3 detik ──► VERIFIED OK
```

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
  score in 0..1 — higher threshold = stricter match. Pass `--threshold 0.7`
  on the CLI (main.py and verify.py) instead of editing code.
- Accuracy improves a lot with 3-5 photos per person vs just 1.
- YuNet + SFace handle odd angles and low light far better than a Haar/LBPH
  pipeline, and need only a few MB of ONNX models downloaded once.
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
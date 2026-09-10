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

## Reuse for a new client

1. Copy this whole folder.
2. Delete everything inside `data/`.
3. Register the new client's people.
4. Done — same code, new data.

## Notes

- `COSINE_THRESHOLD` in main.py (0.5) = strictness. Similarity is a cosine
  score in 0..1 — higher threshold = stricter match.
- Accuracy improves a lot with 3-5 photos per person vs just 1.
- YuNet + SFace handle odd angles and low light far better than a Haar/LBPH
  pipeline, and need only a few MB of ONNX models downloaded once.
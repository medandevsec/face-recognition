import math
import os

import cv2
import numpy as np
import pytest

from core.detector import detect_faces
from core.embedder import align_face, load_embeddings, mean_embedding, save_embeddings
from core.ui import draw_box
from verify import BlinkDetector, Verifier

from conftest import models_available

pytestmark = pytest.mark.skipif(not models_available(), reason="ONNX models not downloaded (run python setup_models.py)")


def register_lena(embedder_paths, face_images, nik="1212121212121212"):
    from conftest import face_of

    img, face = face_of(face_images["lena"])
    aligned = align_face(img, face)
    person_dir = os.path.join(embedder_paths.FACES_DIR, nik)
    os.makedirs(person_dir, exist_ok=True)
    cv2.imwrite(os.path.join(person_dir, "0.jpg"), aligned)
    embedding = mean_embedding(person_dir)
    embeddings = load_embeddings()
    embeddings[nik] = {"embedding": embedding, "samples": 1}
    save_embeddings(embeddings)
    return nik


def gen_frames(face_images, mode="static", n=120, fw=200):
    img = cv2.imread(face_images["lena"])
    fh = int(img.shape[0] * fw / img.shape[1])
    H, W = 480, 640
    cy = (H - fh) // 2
    rng = np.random.default_rng(0)
    frame_face = cv2.resize(img, (fw, fh))
    frames = []
    for i in range(n):
        if mode == "static":
            cx = (W - fw) // 2
        elif mode == "pan":
            cx = (W - fw) // 2 + int(24 * math.sin(i / 3.0))
        else:
            cx = (W - fw) // 2 + int(rng.integers(-2, 3))
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[cy:cy + fh, cx:cx + fw] = frame_face
        if mode == "blink" and i % 12 < 2:
            canvas = (canvas * 0.45).astype(np.uint8)
        frames.append(canvas)
    return frames


@pytest.fixture
def nik(embedder_paths, face_images):
    return register_lena(embedder_paths, face_images)


def run_verify(frames, meta, motion_only=False):
    v = Verifier("", meta, 0.45, motion_only=motion_only, verify_frames=25)
    for f in frames:
        v.step(f)
    return v


def load_target(embedder_paths, nik):
    return load_embeddings()[nik]


def feed_blink_detector(frames):
    bd = BlinkDetector()
    prev_nose = None
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(f)
        if faces is None:
            continue
        face = faces[0]
        w = float(face[2])
        nx, ny = float(face[8]), float(face[9])
        stable = prev_nose is None or math.hypot(nx - prev_nose[0], ny - prev_nose[1]) < w * 0.03
        bd.update(gray, face, stable)
        prev_nose = (nx, ny)
    return bd


def test_blink_detector_ignores_pan(face_images):
    assert feed_blink_detector(gen_frames(face_images, "pan")).blinks == 0


def test_blink_detector_counts_blinks(face_images):
    assert feed_blink_detector(gen_frames(face_images, "blink")).blinks >= 3


def test_static_photo_rejected(embedder_paths, face_images, nik):
    v = run_verify(gen_frames(face_images, "static"), load_target(embedder_paths, nik))
    assert not v.verified


def test_blink_liveness_verified(embedder_paths, face_images, nik):
    v = run_verify(gen_frames(face_images, "blink"), load_target(embedder_paths, nik))
    assert v.verified
    assert v.blink.blinks >= 1


def test_motion_only_verified(embedder_paths, face_images, nik):
    v = run_verify(gen_frames(face_images, "pan"), load_target(embedder_paths, nik),
                   motion_only=True)
    assert v.verified


def test_motion_alone_is_not_blink_liveness(embedder_paths, face_images, nik):
    v = run_verify(gen_frames(face_images, "pan"), load_target(embedder_paths, nik))
    assert not v.verified


def test_verifier_on_unknown_person(embedder_paths, face_images, nik):
    meta = load_target(embedder_paths, nik)
    img = cv2.imread(face_images["messi"])
    faces = detect_faces(img)
    assert faces is not None
    fh = int(img.shape[0] * 200 / img.shape[1])
    canvas = np.zeros((480, 640, 3), np.uint8)
    canvas[(480 - fh) // 2:(480 - fh) // 2 + fh, (640 - 200) // 2:(640 - 200) // 2 + 200] = \
        cv2.resize(img, (200, fh))
    v = Verifier("", meta, 0.45, verify_frames=25)
    for f in [canvas.copy()] * 30:
        v.step(f)
    assert not v.verified


def test_draw_unknown_uses_red_box():
    frame = np.zeros((480, 640, 3), np.uint8)
    draw_box(frame, 20, 30, 200, 200, "Unknown", 0.0)
    assert np.count_nonzero(frame[:, :, 2]) > 0  # red channel present for Unknown
